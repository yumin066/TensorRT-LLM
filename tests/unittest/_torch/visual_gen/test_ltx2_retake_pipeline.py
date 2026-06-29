# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
from enum import Enum
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline
from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_retake import LTX2RetakePipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY


def _minimal_retake_config():
    return DiffusionPipelineConfig(
        model_configs={"transformer": DiffusionModelConfig(pretrained_config=SimpleNamespace())},
        extra_attrs={"workflow": "retake"},
    )


def test_ltx2_workflow_retake_resolves_retake_variant():
    cfg = SimpleNamespace(extra_attrs={"workflow": "retake"}, cache_backend=None)

    assert LTX2Pipeline.resolve_variant(cfg) is LTX2RetakePipeline


def test_ltx2_workflow_rejects_unknown_value():
    cfg = SimpleNamespace(extra_attrs={"workflow": "unknown"}, cache_backend=None)

    with pytest.raises(ValueError, match="Unsupported LTX-2 workflow"):
        LTX2Pipeline.resolve_variant(cfg)


def test_ltx2_retake_declares_required_extra_params():
    pipeline = LTX2RetakePipeline(_minimal_retake_config())

    specs = pipeline.extra_param_specs

    assert specs["retake_video_path"].type == "str"
    assert specs["retake_start_time"].type == "float"
    assert specs["retake_end_time"].type == "float"
    assert specs["retake_regenerate_video"].default is True
    assert specs["retake_regenerate_audio"].default is True
    assert pipeline.default_generation_params == {"num_inference_steps": 40}


def test_ltx2_retake_requires_video_path_before_pipeline_call():
    pipeline = LTX2RetakePipeline(_minimal_retake_config())
    req = SimpleNamespace(
        prompt="replacement line",
        params=SimpleNamespace(
            extra_params={
                "retake_start_time": 1.0,
                "retake_end_time": 2.0,
                "retake_regenerate_video": True,
                "retake_regenerate_audio": True,
                "retake_enhance_prompt": False,
                "retake_max_batch_size": 1,
            },
            seed=1,
            negative_prompt="",
            num_inference_steps=8,
        ),
    )

    with pytest.raises(ValueError, match="retake_video_path"):
        pipeline.infer(req)


def test_ltx2_retake_reports_optional_dependency_load_errors(monkeypatch):
    pipeline = LTX2RetakePipeline(_minimal_retake_config())
    real_import = builtins.__import__

    def fail_ltx_import(name, *args, **kwargs):
        if name.startswith("ltx_core"):
            raise OSError("libtorchaudio.so: undefined symbol")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_ltx_import)

    with pytest.raises(ImportError, match="ltx-pipelines"):
        pipeline.load_standard_components(
            "/checkpoint",
            torch.device("cpu"),
            text_encoder_path="/gemma",
        )


def test_ltx2_retake_materializes_video_chunks_as_batched_uint8():
    chunk0 = torch.full((1, 2, 3, 3), 0.25, dtype=torch.float32)
    chunk1 = torch.full((2, 2, 3, 3), 0.5, dtype=torch.float32)

    video = LTX2RetakePipeline._materialize_video(iter([chunk0, chunk1]))

    assert video.shape == (1, 3, 2, 3, 3)
    assert video.dtype == torch.uint8
    assert video[0, 0, 0, 0, 0].item() == 63
    assert video[0, 1, 0, 0, 0].item() == 127


def test_ltx2_retake_normalizes_audio_object():
    audio = SimpleNamespace(waveform=torch.ones(2, 4), sampling_rate=48000)

    waveform, sample_rate = LTX2RetakePipeline._normalize_audio(audio)

    assert waveform.shape == (1, 2, 4)
    assert waveform.dtype == torch.float32
    assert sample_rate == 48000


def test_ltx2_retake_prefers_non_meta_language_model_device():
    model = SimpleNamespace(
        model=SimpleNamespace(language_model=torch.nn.Linear(1, 1)),
        device=torch.device("meta"),
    )

    device = LTX2RetakePipeline._resolve_non_meta_model_device(model)

    assert device.type == "cpu"


def test_ltx2_retake_resolves_offload_mode_from_extra_attrs():
    class OffloadMode(Enum):
        NONE = "none"
        CPU = "cpu"
        DISK = "disk"

    config = _minimal_retake_config()
    config.extra_attrs["retake_offload_mode"] = "cpu"
    pipeline = LTX2RetakePipeline(config)

    assert pipeline._resolve_offload_mode(OffloadMode) is OffloadMode.CPU


def test_ltx2_registry_accepts_retake_offload_mode_config():
    defaults = PIPELINE_REGISTRY["LTX2Pipeline"].defaults

    assert defaults["retake_offload_mode"] == "none"
    assert defaults["retake_prompt_cache_size"] == 16


def test_ltx2_retake_prompt_encoder_cache_reuses_matching_prompts():
    class PromptEncoder:
        def __init__(self):
            self.calls = 0
            self._trtllm_prompt_cache_size = 2

        def __call__(
            self,
            prompts,
            *,
            enhance_first_prompt=False,
            enhance_prompt_image=None,
            enhance_prompt_seed=42,
        ):
            self.calls += 1
            return [(tuple(prompts), self.calls)]

    LTX2RetakePipeline._install_prompt_encoder_cache(PromptEncoder)
    encoder = PromptEncoder()

    assert encoder(["hello"]) == [(("hello",), 1)]
    assert encoder(["hello"]) == [(("hello",), 1)]
    assert encoder.calls == 1

    assert encoder(["other"]) == [(("other",), 2)]
    assert encoder(["third"]) == [(("third",), 3)]
    assert encoder(["hello"]) == [(("hello",), 4)]
