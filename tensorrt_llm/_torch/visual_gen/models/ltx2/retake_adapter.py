# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native-transformer adapter for the LTX-2 retake workflow.

The upstream Lightricks ``ltx_pipelines`` retake orchestration
(``DiffusionStage.run``) owns the source decode, temporal-region mask,
composite, sigma schedule, guider, and denoise loop. It drives a transformer
through a minimal contract: each step it calls
``transformer(video, audio, perturbations)`` with upstream ``Modality`` objects
and expects **x0** (denoised-sample) predictions back — it wraps the object only
in ``BatchSplitAdapter``, not in the upstream ``X0Model`` velocity->x0 shim.

The native TensorRT-LLM ``LTXModel`` instead returns **velocity** and requires a
pre-built ``TextCache``. This adapter bridges the two so the native transformer
(and its config-driven acceleration stack) can be driven by the upstream loop
without modifying ``../LTX2.3-eval/packages/``. It:

- converts each upstream ``Modality`` into a native ``Modality`` (positions kept
  in fp32 for RoPE parity; latent/context cast to the model dtype),
- builds the native ``TextCache`` (step-invariant within a denoise loop) and
  reuses it while the same text context and positions are passed in,
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


class LTX2RetakeNativeAdapter(torch.nn.Module):
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
                "The LTX-2 retake native adapter does not translate STG perturbations; "
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
        # The prompt AdaLN (cross_attention_adaln=True, e.g. the LTX-2.3 22b
        # distilled checkpoint) needs a per-batch scalar noise level. The upstream
        # Modality carries only per-token ``timesteps`` (denoise_mask * sigma), so
        # recover the scalar sigma as the max over tokens (the unmasked value) when
        # the upstream object does not carry it directly.
        sigma = getattr(upstream, "sigma", None)
        if sigma is None and upstream.timesteps is not None:
            _ts = upstream.timesteps
            sigma = _ts.reshape(_ts.shape[0], -1).amax(dim=1)
        return NativeModality(
            latent=upstream.latent.to(dtype=self._dtype),
            timesteps=upstream.timesteps,
            # Keep positions in fp32: casting fractional time positions to bf16
            # perturbs RoPE and breaks parity with the checkpoint-derived RoPE.
            positions=upstream.positions.to(dtype=torch.float32),
            context=upstream.context.to(dtype=self._dtype),
            enabled=getattr(upstream, "enabled", True),
            context_mask=getattr(upstream, "context_mask", None),
            sigma=sigma,
        )

    @staticmethod
    def _tensor_key(tensor):
        return None if tensor is None else (id(tensor), tuple(tensor.shape))

    @classmethod
    def _modality_cache_key(cls, upstream):
        # The text cache depends on text context, its mask, AND positions (which
        # drive RoPE/PE). Keying only on context would replay stale positional
        # embeddings when the same prompt context object is reused with different
        # positions (e.g. a different retake window/shape). Include all three.
        if upstream is None:
            return None
        return (
            cls._tensor_key(getattr(upstream, "context", None)),
            cls._tensor_key(getattr(upstream, "positions", None)),
            cls._tensor_key(getattr(upstream, "context_mask", None)),
        )

    def _get_text_cache(self, native_video, native_audio, upstream_video, upstream_audio):
        # Context/positions are step-invariant within a denoise loop, so the
        # (expensive) KV-projection text cache is built once and reused while the
        # same context/positions/masks are passed in.
        key = (
            self._modality_cache_key(upstream_video),
            self._modality_cache_key(upstream_audio),
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
