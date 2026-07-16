# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host tests for LTX2Pipeline._encode_video_window (native arbitrary-window VAE encode).

Exercised via the unbound method with a lightweight fake ``self`` (stub encoder),
so no checkpoint/GPU is needed. The real multi-frame VAE numerics are validated
separately on GPU.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline


def test_encode_video_window_forwards_multiframe_to_encoder():
    captured = {}

    def encoder(x):
        captured["x"] = x
        # A causal video VAE with temporal ratio 8: 9 frames -> 2 latent frames.
        return torch.zeros(x.shape[0], 8, 2, x.shape[3] // 32, x.shape[4] // 32)

    fake = SimpleNamespace(video_encoder=encoder)
    video = torch.zeros(1, 3, 9, 64, 64)

    out = LTX2Pipeline._encode_video_window(fake, video)

    assert captured["x"].shape == (1, 3, 9, 64, 64)
    assert out.shape == (1, 8, 2, 2, 2)


def test_encode_image_delegates_single_frame_to_window():
    captured = {}

    def encoder(x):
        captured["x"] = x
        return x

    fake = SimpleNamespace(video_encoder=encoder)
    # _encode_image delegates via self._encode_video_window; bind it on the fake.
    fake._encode_video_window = lambda v: LTX2Pipeline._encode_video_window(fake, v)

    LTX2Pipeline._encode_image(fake, torch.zeros(1, 3, 1, 16, 16))

    assert captured["x"].shape == (1, 3, 1, 16, 16)


def test_encode_video_window_requires_encoder():
    fake = SimpleNamespace(video_encoder=None)
    with pytest.raises(RuntimeError, match="VAE encoder"):
        LTX2Pipeline._encode_video_window(fake, torch.zeros(1, 3, 9, 32, 32))


def test_encode_video_window_rejects_non_5d():
    fake = SimpleNamespace(video_encoder=lambda x: x)
    with pytest.raises(ValueError, match=r"B, 3, T, H, W"):
        LTX2Pipeline._encode_video_window(fake, torch.zeros(3, 9, 32, 32))


def test_encode_video_window_rejects_wrong_channels():
    fake = SimpleNamespace(video_encoder=lambda x: x)
    with pytest.raises(ValueError, match=r"B, 3, T, H, W"):
        LTX2Pipeline._encode_video_window(fake, torch.zeros(1, 4, 9, 32, 32))
