# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.modality import Modality as NativeModality
from tensorrt_llm._torch.visual_gen.models.ltx2.retake_adapter import LTX2RetakeNativeAdapter


class _StubNativeModel(torch.nn.Module):
    """Records calls and returns a constant velocity for each modality latent."""

    def __init__(self, velocity_scale=1.0):
        super().__init__()
        self.prepare_calls = 0
        self.velocity_scale = velocity_scale
        self.last_text_cache = None
        self.last_video = None

    def prepare_text_cache(self, **kwargs):
        self.prepare_calls += 1
        return SimpleNamespace(tag="text_cache", kwargs=kwargs)

    def forward(
        self,
        video=None,
        audio=None,
        perturbations=None,
        *,
        text_cache=None,
        timestep=None,
        step_index=None,
    ):
        self.last_text_cache = text_cache
        self.last_video = video
        vv = None if video is None else torch.full_like(video.latent, self.velocity_scale)
        va = None if audio is None else torch.full_like(audio.latent, self.velocity_scale)
        return vv, va


def _up_modality(latent, timesteps, context, positions, context_mask=None, attention_mask=None):
    return SimpleNamespace(
        latent=latent,
        timesteps=timesteps,
        positions=positions,
        context=context,
        context_mask=context_mask,
        enabled=True,
        attention_mask=attention_mask,
    )


def _default_positions(latent):
    return torch.zeros(latent.shape[0], 3, latent.shape[1], 2)


def test_velocity_to_x0_uses_per_token_timesteps():
    latent = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
    velocity = torch.tensor([[[0.5, 0.5], [1.0, 1.0]]])
    timesteps = torch.tensor([[0.0, 0.5]])  # token0 pinned to t=0, token1 active at t=0.5
    mod = NativeModality(
        latent=latent,
        timesteps=timesteps,
        positions=torch.zeros(1, 3, 2, 2),
        context=torch.zeros(1, 1, 4),
    )

    x0 = LTX2RetakeNativeAdapter._velocity_to_x0(velocity, mod)

    # token0: x0 = 1 - 0.5*0 = 1 (clean); token1: x0 = 2 - 1.0*0.5 = 1.5
    assert torch.allclose(x0[0, 0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(x0[0, 1], torch.tensor([1.5, 1.5]))


def test_forward_returns_x0_and_reuses_text_cache():
    model = _StubNativeModel(velocity_scale=0.25)
    adapter = LTX2RetakeNativeAdapter(model, model_dtype=torch.float32)
    ctx = torch.zeros(1, 4, 8)
    latent = torch.ones(1, 3, 2)
    ts = torch.full((1, 3), 2.0)
    pos = _default_positions(latent)

    x0v, x0a = adapter(video=_up_modality(latent, ts, ctx, pos), audio=None)

    assert torch.allclose(x0v, torch.full_like(latent, 0.5))  # 1 - 0.25*2
    assert x0a is None
    assert model.prepare_calls == 1

    # Same context AND positions objects -> text cache reused (no rebuild).
    adapter(video=_up_modality(latent, ts, ctx, pos), audio=None)
    assert model.prepare_calls == 1

    # Different context tensor -> rebuild.
    adapter(video=_up_modality(latent, ts, torch.zeros(1, 4, 8), pos), audio=None)
    assert model.prepare_calls == 2


def test_changed_positions_rebuilds_text_cache():
    # Regression: same prompt context object but different positions (e.g. a
    # different retake window/shape) must rebuild the cache, since positions
    # drive RoPE/PE. Keying only on context would replay stale embeddings.
    model = _StubNativeModel()
    adapter = LTX2RetakeNativeAdapter(model, model_dtype=torch.float32)
    ctx = torch.zeros(1, 4, 8)
    latent = torch.ones(1, 3, 2)
    ts = torch.zeros(1, 3)

    adapter(video=_up_modality(latent, ts, ctx, _default_positions(latent)), audio=None)
    assert model.prepare_calls == 1

    # Same ctx object, brand-new positions tensor -> must rebuild.
    adapter(video=_up_modality(latent, ts, ctx, torch.ones(1, 3, 3, 2)), audio=None)
    assert model.prepare_calls == 2


def test_positions_kept_fp32_latent_cast_to_model_dtype():
    model = _StubNativeModel()
    adapter = LTX2RetakeNativeAdapter(model, model_dtype=torch.bfloat16)
    video = _up_modality(
        torch.ones(1, 3, 2, dtype=torch.float32),
        torch.zeros(1, 3),
        torch.zeros(1, 4, 8, dtype=torch.float32),
        torch.zeros(1, 3, 3, 2, dtype=torch.float32),
    )

    adapter(video=video, audio=None)

    assert model.last_video.positions.dtype == torch.float32
    assert model.last_video.latent.dtype == torch.bfloat16
    assert model.last_text_cache.kwargs["video_positions"].dtype == torch.float32


def test_non_none_attention_mask_fails_fast():
    adapter = LTX2RetakeNativeAdapter(_StubNativeModel())
    latent = torch.ones(1, 3, 2)
    video = _up_modality(
        latent,
        torch.zeros(1, 3),
        torch.zeros(1, 4, 8),
        _default_positions(latent),
        attention_mask=torch.ones(1, 3, 3),
    )
    with pytest.raises(NotImplementedError, match="attention_mask"):
        adapter(video=video, audio=None)


def test_non_none_perturbations_fails_fast():
    adapter = LTX2RetakeNativeAdapter(_StubNativeModel())
    latent = torch.ones(1, 3, 2)
    video = _up_modality(
        latent, torch.zeros(1, 3), torch.zeros(1, 4, 8), _default_positions(latent)
    )
    with pytest.raises(NotImplementedError, match="perturbations"):
        adapter(video=video, audio=None, perturbations=object())
