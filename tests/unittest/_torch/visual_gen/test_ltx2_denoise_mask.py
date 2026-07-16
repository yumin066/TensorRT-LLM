# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host tests for LTX2Pipeline._build_denoise_mask (two-sided retake window).

Pure tensor logic: exercised via the unbound method with a lightweight fake
``self`` so no checkpoint/GPU is needed. Convention: 0.0 = conditioned (don't
denoise), 1.0 = unconditioned (denoise).
"""

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline


def _fake_pipeline(patch=(1, 1, 1)):
    return SimpleNamespace(
        video_patchifier=SimpleNamespace(patch_size=patch),
        device=torch.device("cpu"),
    )


def _shape(frames, height, width):
    return SimpleNamespace(frames=frames, height=height, width=width)


def _build(fake, shape, **kwargs):
    return LTX2Pipeline._build_denoise_mask(fake, shape, **kwargs)


def test_leading_frame_mask_is_backward_compatible():
    # 5 latent frames, 2x2 tokens/frame -> 20 tokens; condition the first 2 frames.
    mask = _build(_fake_pipeline(), _shape(5, 2, 2), num_cond_latent_frames=2, strength=1.0)

    assert mask.shape == (1, 20)
    assert torch.equal(mask[0, :8], torch.zeros(8))  # frames 0-1 conditioned
    assert torch.equal(mask[0, 8:], torch.ones(12))  # frames 2-4 denoised


def test_two_sided_window_conditions_both_sides_and_denoises_middle():
    # Condition frame 0 (leading context) and frame 4 (trailing context); the
    # middle window (frames 1-3) is left unconditioned (regenerated).
    mask = _build(
        _fake_pipeline(),
        _shape(5, 2, 2),
        cond_latent_frame_ranges=[(0, 1), (4, 5)],
        strength=1.0,
    )

    assert torch.equal(mask[0, 0:4], torch.zeros(4))  # frame 0 conditioned
    assert torch.equal(mask[0, 4:16], torch.ones(12))  # frames 1-3 denoised
    assert torch.equal(mask[0, 16:20], torch.zeros(4))  # frame 4 conditioned


def test_two_sided_window_clamps_out_of_range_frames():
    # Negative start and past-end stop are clamped to [0, grid_f]; result matches
    # the exact (0,1)/(4,5) ranges.
    mask = _build(
        _fake_pipeline(),
        _shape(5, 2, 2),
        cond_latent_frame_ranges=[(-3, 1), (4, 99)],
        strength=1.0,
    )

    assert torch.equal(mask[0, 0:4], torch.zeros(4))
    assert torch.equal(mask[0, 4:16], torch.ones(12))
    assert torch.equal(mask[0, 16:20], torch.zeros(4))


def test_empty_and_inverted_ranges_are_skipped():
    mask = _build(
        _fake_pipeline(),
        _shape(3, 2, 2),
        cond_latent_frame_ranges=[(1, 1), (3, 1)],  # empty, inverted
        strength=1.0,
    )

    assert torch.equal(mask[0], torch.ones(12))  # nothing conditioned


def test_conditioning_strength_sets_partial_mask_value():
    mask = _build(
        _fake_pipeline(),
        _shape(4, 2, 2),
        cond_latent_frame_ranges=[(0, 1), (3, 4)],
        strength=0.5,
    )

    # cond_value = 1 - strength = 0.5 on conditioned frames.
    assert torch.allclose(mask[0, 0:4], torch.full((4,), 0.5))
    assert torch.equal(mask[0, 4:12], torch.ones(8))
    assert torch.allclose(mask[0, 12:16], torch.full((4,), 0.5))
