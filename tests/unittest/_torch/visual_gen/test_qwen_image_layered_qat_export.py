# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json

import pytest
import torch
from pydantic import ValidationError

from scripts.visualgen_eval import qwen_image_layered_qat_export as qat_export
from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm.quantization.mode import QuantAlgo


def _base_config(tmp_path, **overrides) -> dict:
    data = {
        "model": str(tmp_path / "source"),
        "qat_checkpoint": str(tmp_path / "checkpoint.pt"),
        "output_dir": str(tmp_path / "export"),
        "target_policy": "explicit",
        "target_layers": ["transformer_blocks.0.attn.to_q"],
        "expected_target_count": 1,
    }
    data.update(overrides)
    return data


def test_build_qwen_block_policy_records_target_count_and_exclusions(tmp_path):
    config = qat_export.QwenImageLayeredQatExportConfig(
        **_base_config(
            tmp_path,
            target_policy="qwen_block_linears_840",
            target_layers=None,
            expected_target_count=840,
        )
    )

    policy = qat_export.build_target_policy(config)

    assert policy.name == "qwen_block_linears_840"
    assert len(policy.target_layers) == 840
    assert policy.target_layers[0] == "transformer_blocks.0.img_mod.1"
    assert policy.target_layers[-1] == "transformer_blocks.59.txt_mlp.down_proj"
    assert policy.bf16_exclusions == (
        "img_in",
        "txt_in",
        "norm_out.linear",
        "proj_out",
    )


def test_config_rejects_ambiguous_or_unsupported_target_policy(tmp_path):
    with pytest.raises(ValidationError, match="explicit target_policy requires target_layers"):
        qat_export.QwenImageLayeredQatExportConfig(**_base_config(tmp_path, target_layers=None))

    with pytest.raises(ValidationError, match="target_layers are only valid"):
        qat_export.QwenImageLayeredQatExportConfig(
            **_base_config(
                tmp_path,
                target_policy="qwen_block_linears_840",
                expected_target_count=840,
            )
        )

    with pytest.raises(ValidationError, match="Input should be"):
        qat_export.QwenImageLayeredQatExportConfig(
            **_base_config(tmp_path, target_policy="unknown")
        )


def test_merge_qat_trainable_state_maps_fake_linear_names():
    source_weight = torch.zeros(128, 128, dtype=torch.bfloat16)
    source_bias = torch.zeros(128, dtype=torch.bfloat16)
    trained_weight = torch.ones(128, 128, dtype=torch.float32)
    trained_bias = torch.ones(128, dtype=torch.float32)
    source_state = {
        "transformer_blocks.0.attn.to_q.weight": source_weight,
        "transformer_blocks.0.attn.to_q.bias": source_bias,
    }
    trainable_state = {
        "transformer_blocks.0.attn.to_q.linear.weight": trained_weight,
        "transformer_blocks.0.attn.to_q.linear.bias": trained_bias,
    }

    merged, merged_names = qat_export.merge_qat_trainable_state(source_state, trainable_state)

    assert merged_names == (
        "transformer_blocks.0.attn.to_q.bias",
        "transformer_blocks.0.attn.to_q.weight",
    )
    assert merged["transformer_blocks.0.attn.to_q.weight"].dtype == torch.bfloat16
    assert torch.allclose(
        merged["transformer_blocks.0.attn.to_q.weight"].float(),
        torch.ones_like(source_weight).float(),
    )
    assert torch.allclose(
        merged["transformer_blocks.0.attn.to_q.bias"].float(),
        torch.ones_like(source_bias).float(),
    )


def test_merge_qat_trainable_state_requires_lora_config():
    with pytest.raises(ValueError, match="requires checkpoint config"):
        qat_export.merge_qat_trainable_state(
            {"transformer_blocks.0.attn.to_q.weight": torch.zeros(128, 128)},
            {
                "transformer_blocks.0.attn.to_q.lora_down.weight": torch.zeros(4, 128),
                "transformer_blocks.0.attn.to_q.lora_up.weight": torch.zeros(128, 4),
            },
        )


def test_merge_qat_trainable_state_applies_lora_delta():
    target = "transformer_blocks.0.attn.to_q"
    source_weight = torch.zeros(3, 4, dtype=torch.bfloat16)
    lora_down = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.5, 1.0, 1.5, 2.0],
        ],
        dtype=torch.float32,
    )
    lora_up = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    merged, merged_names = qat_export.merge_qat_trainable_state(
        {f"{target}.weight": source_weight},
        {
            f"{target}.lora_down.weight": lora_down,
            f"{target}.lora_up.weight": lora_up,
        },
        qat_config={"lora_rank": 2, "lora_alpha": 4.0},
        injections=(
            {
                "module_name": target,
                "trainable_parameter_names": ["lora_down.weight", "lora_up.weight"],
            },
        ),
    )

    expected_delta = torch.matmul(lora_up, lora_down) * 2.0
    assert merged_names == (f"{target}.weight",)
    assert merged[f"{target}.weight"].dtype == torch.bfloat16
    assert torch.allclose(merged[f"{target}.weight"].float(), expected_delta)


def test_export_static_state_dict_writes_fp8_auxiliary_tensors():
    target = "transformer_blocks.0.attn.to_q"
    state = {
        f"{target}.weight": torch.randn(130, 129, dtype=torch.bfloat16),
        f"{target}.bias": torch.randn(130, dtype=torch.bfloat16),
        "proj_out.weight": torch.randn(128, 128, dtype=torch.bfloat16),
    }

    exported, quantized_layers = qat_export.export_static_mxfp8_state_dict(
        state,
        target_layers=(target,),
    )

    assert quantized_layers == (target,)
    assert exported[f"{target}.weight"].dtype == torch.float8_e4m3fn
    assert tuple(exported[f"{target}.weight"].shape) == (130, 129)
    assert tuple(exported[f"{target}.weight_scale"].shape) == (2, 2)
    assert exported[f"{target}.weight_scale"].dtype == torch.float32
    assert exported[f"{target}.input_scale"].item() == pytest.approx(1.0)
    assert exported[f"{target}.inv_input_scale"].item() == pytest.approx(1.0)
    assert exported[f"{target}.bias"].dtype == torch.bfloat16
    assert exported["proj_out.weight"].dtype == torch.bfloat16


def test_static_quantization_config_parses_as_static_fp8():
    policy = qat_export.TargetPolicy(
        name="explicit",
        target_layers=("transformer_blocks.0.attn.to_q",),
        bf16_exclusions=("proj_out",),
        expected_target_count=1,
    )

    quant_config, layer_quant_config, dynamic_weight_quant, dynamic_activation_quant = (
        DiffusionPipelineConfig.load_diffusion_quant_config(
            qat_export.build_static_quantization_config(policy)
        )
    )

    assert quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
    assert quant_config.group_size == 128
    assert quant_config.exclude_modules == ["proj_out"]
    assert layer_quant_config is None
    assert dynamic_weight_quant is False
    assert dynamic_activation_quant is False


def test_export_qat_checkpoint_writes_static_transformer_directory(tmp_path):
    from safetensors.torch import load_file, save_file

    source_dir = tmp_path / "source"
    transformer_dir = source_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    (source_dir / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "QwenImageLayeredPipeline",
                "transformer": ["diffusers", "QwenImageTransformer2DModel"],
            }
        ),
        encoding="utf-8",
    )
    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "QwenImageTransformer2DModel",
                "num_layers": 1,
            }
        ),
        encoding="utf-8",
    )
    target = "transformer_blocks.0.img_mlp.up_proj"
    source_target = "transformer_blocks.0.img_mlp.net.0.proj"
    source_weight = torch.zeros(128, 128, dtype=torch.bfloat16)
    trained_weight = torch.full((128, 128), 0.25, dtype=torch.float32)
    save_file(
        {
            f"{source_target}.weight": source_weight,
            f"{source_target}.bias": torch.zeros(128, dtype=torch.bfloat16),
            "proj_out.weight": torch.ones(128, 128, dtype=torch.bfloat16),
        },
        str(transformer_dir / "diffusion_pytorch_model.safetensors"),
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format": qat_export.LAYERED_TRAINING_FORMAT,
            "trainable_state_dict": {
                f"{target}.linear.weight": trained_weight,
            },
            "injections": [
                {
                    "module_name": target,
                    "block_index": 0,
                    "role": "attn.to_q",
                    "recipe": "partial_unfreeze",
                    "trainable_parameter_names": ["linear.weight"],
                }
            ],
            "best_validation_loss": 0.5,
        },
        checkpoint_path,
    )

    config = qat_export.QwenImageLayeredQatExportConfig(
        **_base_config(
            tmp_path,
            model=str(source_dir),
            qat_checkpoint=str(checkpoint_path),
            target_layers=[target],
        )
    )
    result = qat_export.export_qwen_image_layered_qat(config)

    exported = load_file(str(result.weight_path))
    assert exported[f"{target}.weight"].dtype == torch.float8_e4m3fn
    assert exported[f"{target}.weight_scale"].shape == (1, 1)
    assert exported[f"{target}.input_scale"].item() == pytest.approx(1.0)
    assert exported[f"{target}.inv_input_scale"].item() == pytest.approx(1.0)
    assert exported["proj_out.weight"].dtype == torch.bfloat16

    exported_config = json.loads(result.transformer_config_path.read_text(encoding="utf-8"))
    quant_metadata = exported_config["quantization_config"]
    assert quant_metadata["quant_algo"] == "FP8_BLOCK_SCALES"
    assert quant_metadata["config_groups"]["default"]["weights"]["dynamic"] is False

    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert provenance["format"] == qat_export.EXPORT_FORMAT
    assert provenance["quantized_weight_count"] == 1
    assert provenance["merged_parameter_names"] == [f"{target}.weight"]


@pytest.mark.parametrize(
    "checkpoint_format",
    (qat_export.LAYERED_TRAINING_FORMAT, qat_export.QWEN_IMAGE_TRAINING_FORMAT),
)
def test_export_qat_lora_checkpoint_writes_static_transformer_directory(
    tmp_path, checkpoint_format
):
    from safetensors.torch import load_file, save_file

    source_dir = tmp_path / "source"
    transformer_dir = source_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    (source_dir / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "QwenImageLayeredPipeline",
                "transformer": ["diffusers", "QwenImageTransformer2DModel"],
            }
        ),
        encoding="utf-8",
    )
    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "QwenImageTransformer2DModel",
                "num_layers": 1,
            }
        ),
        encoding="utf-8",
    )
    target = "transformer_blocks.0.attn.to_q"
    save_file(
        {
            f"{target}.weight": torch.zeros(128, 128, dtype=torch.bfloat16),
            f"{target}.bias": torch.zeros(128, dtype=torch.bfloat16),
        },
        str(transformer_dir / "diffusion_pytorch_model.safetensors"),
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format": checkpoint_format,
            "config": {
                "recipe": "lora_adapter",
                "lora_rank": 2,
                "lora_alpha": 4.0,
            },
            "trainable_state_dict": {
                f"{target}.lora_down.weight": torch.full((2, 128), 0.25),
                f"{target}.lora_up.weight": torch.full((128, 2), 0.5),
            },
            "injections": [
                {
                    "module_name": target,
                    "block_index": 0,
                    "role": "attn.to_q",
                    "recipe": "lora_adapter",
                    "trainable_parameter_names": ["lora_down.weight", "lora_up.weight"],
                }
            ],
            "best_validation_loss": 0.25,
        },
        checkpoint_path,
    )

    config = qat_export.QwenImageLayeredQatExportConfig(
        **_base_config(
            tmp_path,
            model=str(source_dir),
            qat_checkpoint=str(checkpoint_path),
            target_layers=[target],
        )
    )
    result = qat_export.export_qwen_image_layered_qat(config)

    exported = load_file(str(result.weight_path))
    assert exported[f"{target}.weight"].dtype == torch.float8_e4m3fn
    assert exported[f"{target}.weight_scale"].shape == (1, 1)

    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert provenance["qat_checkpoint_format"] == checkpoint_format
    assert provenance["qat_best_validation_loss"] == pytest.approx(0.25)
    assert provenance["merged_parameter_names"] == [f"{target}.weight"]
