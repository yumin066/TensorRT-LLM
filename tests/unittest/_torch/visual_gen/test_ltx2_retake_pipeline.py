# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for LTX-2 native retake window mapping."""

from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_retake import (
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
