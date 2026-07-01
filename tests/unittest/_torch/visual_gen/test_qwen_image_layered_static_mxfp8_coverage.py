# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from scripts.visualgen_eval import qwen_image_layered_static_mxfp8_coverage as coverage
from tensorrt_llm.quantization.mode import QuantAlgo


class _FakeQuantConfig:
    def __init__(self, quant_algo=None, exclude_modules=None) -> None:
        self.quant_algo = quant_algo
        self.exclude_modules = exclude_modules


class FP8BlockScalesLinearMethod:
    pass


class UnquantizedLinearMethod:
    pass


class _FakeLinear(nn.Module):
    def __init__(
        self,
        *,
        weight_dtype: torch.dtype,
        quant_algo=None,
        quant_method=None,
        with_scales: bool = False,
    ) -> None:
        super().__init__()
        self.weight = torch.empty((1, 1), dtype=weight_dtype)
        self.bias = torch.empty((1,), dtype=torch.bfloat16)
        self.quant_config = _FakeQuantConfig(quant_algo)
        self.quant_method = quant_method or UnquantizedLinearMethod()
        if with_scales:
            self.weight_scale = torch.ones((1, 1), dtype=torch.float32)
            self.input_scale = torch.tensor(1.0, dtype=torch.float32)
            self.inv_input_scale = torch.tensor(1.0, dtype=torch.float32)


class _FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = _static_mxfp8_linear()
        self.to_k = _static_mxfp8_linear()
        self.to_v = _static_mxfp8_linear()
        self.to_out = nn.Sequential(_static_mxfp8_linear())
        self.add_q_proj = _static_mxfp8_linear()
        self.add_k_proj = _static_mxfp8_linear()
        self.add_v_proj = _static_mxfp8_linear()
        self.to_add_out = _static_mxfp8_linear()


class _FakeMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = _static_mxfp8_linear()
        self.down_proj = _static_mxfp8_linear()


class _FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.img_mod = nn.Sequential(nn.Identity(), _static_mxfp8_linear())
        self.txt_mod = nn.Sequential(nn.Identity(), _static_mxfp8_linear())
        self.attn = _FakeAttention()
        self.img_mlp = _FakeMlp()
        self.txt_mlp = _FakeMlp()


class _FakeNormOut(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = _bf16_linear()


class _FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.img_in = _bf16_linear()
        self.txt_in = _bf16_linear()
        self.transformer_blocks = nn.ModuleList(
            [_FakeBlock() for _ in range(coverage.QWEN_LAYER_COUNT)]
        )
        self.norm_out = _FakeNormOut()
        self.proj_out = _bf16_linear()


class _FakePipelineConfig:
    def __init__(
        self,
        *,
        quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
        dynamic_weight_quant: bool = False,
        force_dynamic_quantization: bool = False,
        exclude_modules=None,
    ) -> None:
        self.quant_config = _FakeQuantConfig(
            quant_algo,
            list(exclude_modules or coverage.QWEN_STATIC_BF16_EXCLUSIONS),
        )
        self.dynamic_weight_quant = dynamic_weight_quant
        self.force_dynamic_quantization = force_dynamic_quantization


class _FakePipeline:
    def __init__(self, transformer: nn.Module, pipeline_config: _FakePipelineConfig) -> None:
        self.transformer = transformer
        self.pipeline_config = pipeline_config


def _static_mxfp8_linear() -> _FakeLinear:
    return _FakeLinear(
        weight_dtype=torch.float8_e4m3fn,
        quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
        quant_method=FP8BlockScalesLinearMethod(),
        with_scales=True,
    )


def _bf16_linear() -> _FakeLinear:
    return _FakeLinear(weight_dtype=torch.bfloat16)


def _base_config(tmp_path, **overrides) -> coverage.QwenImageLayeredLoaderCoverageConfig:
    data = {
        "model": str(tmp_path / "static_checkpoint"),
        "output_dir": str(tmp_path / "coverage"),
    }
    data.update(overrides)
    return coverage.QwenImageLayeredLoaderCoverageConfig(**data)


def _get_module(root: nn.Module, name: str) -> nn.Module:
    modules = dict(root.named_modules())
    return modules[name]


def test_build_qwen_block_targets_matches_840_contract():
    targets = coverage.build_qwen_block_linear_targets()

    assert len(targets) == 840
    assert targets[0] == "transformer_blocks.0.img_mod.1"
    assert targets[-1] == "transformer_blocks.59.txt_mlp.down_proj"
    assert "img_in" not in targets
    assert "proj_out" not in targets


def test_config_rejects_844_target_count(tmp_path):
    with pytest.raises(ValidationError, match="expected_target_count=840"):
        _base_config(tmp_path, expected_target_count=844)


def test_loaded_transformer_coverage_accepts_840_targets_and_four_exclusions(tmp_path):
    pipeline = _FakePipeline(_FakeTransformer(), _FakePipelineConfig())

    report = coverage.analyze_loaded_transformer_coverage(
        pipeline,
        _base_config(tmp_path),
        linear_cls=_FakeLinear,
    )

    assert report["status"] == "passed"
    assert report["target_count"] == 840
    assert report["static_mxfp8_target_count"] == 840
    assert report["bf16_exclusion_count"] == 4
    assert report["total_linear_count"] == 844
    assert "844/844" not in json.dumps(report)


def test_loaded_transformer_coverage_rejects_dynamic_weight_quant(tmp_path):
    pipeline = _FakePipeline(
        _FakeTransformer(),
        _FakePipelineConfig(dynamic_weight_quant=True),
    )

    with pytest.raises(ValueError, match="dynamic_weight_quant=True"):
        coverage.analyze_loaded_transformer_coverage(
            pipeline,
            _base_config(tmp_path),
            linear_cls=_FakeLinear,
        )


def test_loaded_transformer_coverage_rejects_missing_target_scale(tmp_path):
    transformer = _FakeTransformer()
    target = _get_module(transformer, "transformer_blocks.0.attn.to_q")
    delattr(target, "weight_scale")
    pipeline = _FakePipeline(transformer, _FakePipelineConfig())

    with pytest.raises(ValueError, match="target_not_static_mxfp8"):
        coverage.analyze_loaded_transformer_coverage(
            pipeline,
            _base_config(tmp_path),
            linear_cls=_FakeLinear,
        )


def test_loaded_transformer_coverage_rejects_bf16_target_fallback(tmp_path):
    transformer = _FakeTransformer()
    target = _get_module(transformer, "transformer_blocks.0.attn.to_v")
    target.weight = torch.empty((1, 1), dtype=torch.bfloat16)
    pipeline = _FakePipeline(transformer, _FakePipelineConfig())

    with pytest.raises(ValueError, match="target_not_static_mxfp8"):
        coverage.analyze_loaded_transformer_coverage(
            pipeline,
            _base_config(tmp_path),
            linear_cls=_FakeLinear,
        )


def test_loaded_transformer_coverage_rejects_quantized_non_target(tmp_path):
    transformer = _FakeTransformer()
    transformer.proj_out = _static_mxfp8_linear()
    pipeline = _FakePipeline(transformer, _FakePipelineConfig())

    with pytest.raises(ValueError, match="unexpected_quantized_non_targets"):
        coverage.analyze_loaded_transformer_coverage(
            pipeline,
            _base_config(tmp_path),
            linear_cls=_FakeLinear,
        )


def test_loaded_transformer_coverage_rejects_missing_bf16_exclusion(tmp_path):
    transformer = _FakeTransformer()
    del transformer._modules["txt_in"]
    pipeline = _FakePipeline(transformer, _FakePipelineConfig())

    with pytest.raises(ValueError, match="missing_bf16_exclusions"):
        coverage.analyze_loaded_transformer_coverage(
            pipeline,
            _base_config(tmp_path),
            linear_cls=_FakeLinear,
        )
