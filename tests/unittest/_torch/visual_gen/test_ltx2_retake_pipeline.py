# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for LTX-2 native retake window mapping."""

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.patchifier import VideoLatentPatchifier
from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_retake import (
    _init_retake_patchified_latents,
    _retake_conditioned_latent_ranges,
    _retake_pixel_window,
)


def test_delete_disfluency_uses_the_canonical_half_open_frame_window() -> None:
    pixel_window = _retake_pixel_window(
        start_time=2.9667,
        end_time=3.9333,
        fps=30.0,
        num_frames=209,
    )

    assert pixel_window == (89, 118)


def test_delete_disfluency_regenerates_only_the_touched_latents() -> None:
    conditioned_ranges = _retake_conditioned_latent_ranges(
        pixel_start=89,
        pixel_end=118,
        num_frames=209,
        temporal_ratio=8,
    )

    # The 209 source frames produce 27 causal VAE latents. Pixel frames
    # [89, 118) touch latent frames [12, 16), so only the ranges outside that
    # window are conditioned on the encoded source.
    assert conditioned_ranges == [(0, 12), (16, 27)]


def test_retake_initial_noise_uses_upstream_patchified_layout() -> None:
    patchifier = VideoLatentPatchifier(patch_size=1)
    source_5d = torch.arange(1 * 4 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 4, 3, 2, 2)
    source_patch = patchifier.patchify(source_5d)

    seed = 42
    direct_generator = torch.Generator().manual_seed(seed)
    patchified_noise = torch.randn(source_patch.shape, generator=direct_generator)

    wrong_generator = torch.Generator().manual_seed(seed)
    five_d_noise = patchifier.patchify(torch.randn(source_5d.shape, generator=wrong_generator))
    assert not torch.equal(patchified_noise, five_d_noise)

    denoise_mask = torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 0.0]])
    initialized = _init_retake_patchified_latents(
        patchified_noise,
        source_patch,
        denoise_mask,
    )

    assert torch.equal(initialized[:, [0, 3, 5]], source_patch[:, [0, 3, 5]])
    assert torch.equal(initialized[:, [1, 2, 4]], patchified_noise[:, [1, 2, 4]])
