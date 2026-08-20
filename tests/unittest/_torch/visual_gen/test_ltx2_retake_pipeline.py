# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused CPU tests for the native LTX-2 retake pipeline."""

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.patchifier import VideoLatentPatchifier
from tensorrt_llm._torch.visual_gen.models.ltx2_retake.pipeline_ltx2_retake import (
    LTX2RetakePipeline,
    _init_retake_patchified_latents,
    _retake_conditioned_latent_ranges,
    _retake_pixel_window,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY
from tensorrt_llm.visual_gen.args import VisualGenArgs


def test_retake_window_maps_to_unconditioned_latents() -> None:
    pixel_start, pixel_end = _retake_pixel_window(
        start_time=2.9667,
        end_time=3.9333,
        fps=30.0,
        num_frames=209,
    )

    assert (pixel_start, pixel_end) == (89, 118)
    assert _retake_conditioned_latent_ranges(
        pixel_start=pixel_start,
        pixel_end=pixel_end,
        num_frames=209,
        temporal_ratio=8,
    ) == [(0, 12), (16, 27)]


def test_retake_initial_noise_preserves_conditioned_tokens() -> None:
    patchifier = VideoLatentPatchifier(patch_size=1)
    source = patchifier.patchify(
        torch.arange(1 * 4 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 4, 3, 2, 2)
    )
    noise = torch.randn(source.shape, generator=torch.Generator().manual_seed(42))
    denoise_mask = torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 0.0]])

    initialized = _init_retake_patchified_latents(noise, source, denoise_mask)

    assert torch.equal(initialized[:, [0, 3, 5]], source[:, [0, 3, 5]])
    assert torch.equal(initialized[:, [1, 2, 4]], noise[:, [1, 2, 4]])


def test_retake_pipeline_registration_and_config_schema() -> None:
    entry = PIPELINE_REGISTRY["LTX2RetakePipeline"]
    assert entry.pipeline_cls is LTX2RetakePipeline
    assert set(entry.defaults) == {
        "text_encoder_path",
        "retake_lora_path",
        "retake_lora_strength",
        "retake_prompt_conditioning_path",
        "retake_fp8_linear_steps",
    }

    args = VisualGenArgs(
        model="/tmp/ltx2-retake.safetensors",
        pipeline="LTX2RetakePipeline",
        pipeline_config={"retake_lora_strength": 0.5},
    )
    resolved = PipelineLoader(args)._resolve_pipeline_config(args.model)
    assert resolved["retake_lora_strength"] == 0.5
