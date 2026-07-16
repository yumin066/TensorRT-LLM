# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 1 native-transformer adapter for the LTX-2 retake workflow.

The upstream Lightricks ``ltx_pipelines`` retake orchestration
(``DiffusionStage.run``) owns the source decode, temporal-region mask,
composite, sigma schedule, guider, and denoise loop. It drives a transformer
through a minimal contract: each step it calls
``transformer(video, audio, perturbations)`` with upstream ``Modality`` objects
and expects **x0** (denoised-sample) predictions back — it wraps the object only
in ``BatchSplitAdapter``, not in the upstream ``X0Model`` velocity->x0 shim.

The native TensorRT-LLM ``LTXModel`` instead returns **velocity** and requires a
pre-built ``TextCache``. This adapter bridges the two so the native transformer
(and its config-driven acceleration stack) can be injected into the upstream
loop without modifying ``../LTX2.3-eval/packages/``. It:

- converts each upstream ``Modality`` into a native ``Modality`` (positions kept
  in fp32 for RoPE parity; latent/context cast to the model dtype),
- builds the step-invariant native ``TextCache`` once and reuses it,
- calls the native forward, and
- converts the returned velocity to x0 with per-token timesteps:
  ``x0 = latent - velocity * timesteps[..., None]``.

Scope: matches the retake workflow (``regenerate_video=True``, distilled path,
no STG perturbations, unmasked self/cross attention). Non-``None`` upstream
perturbations or self-attention masks fail fast rather than silently diverge.
"""

from __future__ import annotations

import torch

from .ltx2_core.modality import Modality as NativeModality


class LTX2RetakeStage1Adapter(torch.nn.Module):
    """Impersonate the upstream ``X0Model`` around a native ``LTXModel``.

    Args:
        native_model: A built + weight-loaded native ``LTXModel``.
        model_dtype: dtype to cast latent/context to (default bf16).
    """

    def __init__(self, native_model: torch.nn.Module, model_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self._model = native_model
        self._dtype = model_dtype
        self._text_cache = None
        self._text_cache_key = None

    def forward(self, video=None, audio=None, perturbations=None):
        if perturbations is not None:
            raise NotImplementedError(
                "LTX-2 Stage 1 retake adapter does not translate STG perturbations; "
                "the distilled retake path uses a non-guided denoiser (perturbations=None)."
            )

        native_video = self._to_native_modality(video)
        native_audio = self._to_native_modality(audio)

        text_cache = self._get_text_cache(native_video, native_audio, video, audio)

        velocity_video, velocity_audio = self._model(
            video=native_video,
            audio=native_audio,
            perturbations=None,
            text_cache=text_cache,
            timestep=None,
            step_index=None,
        )

        x0_video = self._velocity_to_x0(velocity_video, native_video)
        x0_audio = self._velocity_to_x0(velocity_audio, native_audio)
        return x0_video, x0_audio

    def _to_native_modality(self, upstream):
        if upstream is None:
            return None
        # Retake self/cross attention is unmasked (FA4-eligible); the native
        # Modality has no attention_mask field, so a non-None one must fail fast.
        attention_mask = getattr(upstream, "attention_mask", None)
        if attention_mask is not None:
            raise NotImplementedError(
                "Upstream attention_mask is not supported by the native LTX-2 Modality; "
                "retake attention is expected to be unmasked."
            )
        return NativeModality(
            latent=upstream.latent.to(dtype=self._dtype),
            timesteps=upstream.timesteps,
            # Keep positions in fp32: casting fractional time positions to bf16
            # perturbs RoPE and breaks oracle parity (task3 D2).
            positions=upstream.positions.to(dtype=torch.float32),
            context=upstream.context.to(dtype=self._dtype),
            enabled=getattr(upstream, "enabled", True),
            context_mask=getattr(upstream, "context_mask", None),
        )

    def _get_text_cache(self, native_video, native_audio, upstream_video, upstream_audio):
        # Text context/positions are step-invariant within a denoise loop, so the
        # (expensive) KV-projection text cache is built once and reused while the
        # same upstream context tensors are passed in.
        key = (
            id(getattr(upstream_video, "context", None)),
            id(getattr(upstream_audio, "context", None)),
        )
        if self._text_cache is None or self._text_cache_key != key:
            self._text_cache = self._model.prepare_text_cache(
                video_context=native_video.context if native_video is not None else None,
                video_context_mask=native_video.context_mask if native_video is not None else None,
                video_positions=native_video.positions if native_video is not None else None,
                audio_context=native_audio.context if native_audio is not None else None,
                audio_context_mask=native_audio.context_mask if native_audio is not None else None,
                audio_positions=native_audio.positions if native_audio is not None else None,
                dtype=self._dtype,
            )
            self._text_cache_key = key
        return self._text_cache

    @staticmethod
    def _velocity_to_x0(velocity, native_modality):
        """Convert a velocity (flow) prediction to a denoised-sample (x0).

        ``x0 = latent - velocity * timesteps[..., None]`` using per-token
        timesteps so retake prefix/suffix tokens pinned to timestep 0 stay clean.
        """
        if velocity is None or native_modality is None:
            return None
        timesteps = native_modality.timesteps.to(torch.float32)
        # Broadcast per-token (B, T) timesteps over the channel dim of (B, T, D).
        while timesteps.dim() < velocity.dim():
            timesteps = timesteps.unsqueeze(-1)
        x0 = native_modality.latent.to(torch.float32) - velocity.to(torch.float32) * timesteps
        return x0.to(velocity.dtype)
