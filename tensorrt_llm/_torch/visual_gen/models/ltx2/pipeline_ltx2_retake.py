# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional LTX-2 retake workflow for VisualGen.

This pipeline keeps a resident retake model for ``trtllm-serve``. By default it
runs a fully native retake pre/post runtime path: it composes a native
``LTX2Pipeline`` (video VAE encoder/decoder, video patchifier, scheduler, Gemma
text encoder + connectors) that shares the retake pipeline's already-built
native transformer (``LTXModel``), then drives native source-encode -> native
masked denoise -> native decode and composites the regenerated pixel window back
into the source so non-window frames stay byte-identical. The masked-denoise
mechanism is the native ``LTX2Pipeline`` i2v machinery generalized to a
two-sided retake window (leading + trailing context conditioned, middle window
regenerated).

The earlier upstream-stage path — which injects ``LTX2RetakeNativeAdapter`` into
the upstream Lightricks ``DiffusionStage.run`` loop — is preserved behind the
``pipeline_config.extra_attrs['retake_use_upstream_stage']`` switch (default
``False`` / native) as a named oracle for comparison and GPU verification.

Only the deterministic source pixel/audio decode and stream metadata are read
from the upstream ``ltx_pipelines.utils.media_io`` readers; no upstream retake
orchestration is used on the native path.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, Optional

import torch

from tensorrt_llm._torch.visual_gen.output import PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline, ExtraParamSchema
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm.logger import logger

from .ltx2_core.patchifier import get_pixel_coords
from .ltx2_core.scheduler_adapter import NativeSchedulerAdapter
from .ltx2_core.types import VIDEO_SCALE_FACTORS, VideoLatentShape, VideoPixelShape
from .ltx2_core.video_vae import TilingConfig
from .pipeline_ltx2 import (
    LTX2Pipeline,
    _find_safetensors_files,
    _load_ltx2_transformer_weights,
    build_ltx2_transformer,
)
from .retake_adapter import LTX2RetakeNativeAdapter

# Distilled retake noise schedule (8 Euler steps). Domain constant mirroring the
# LTX-2 distillation sigma values; kept local so the native retake path does not
# depend on upstream orchestration for its schedule (the upstream-stage oracle
# still sources its own schedule from ``ltx_pipelines``).
_RETAKE_DISTILLED_SIGMA_VALUES = [
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
]


def _retake_pixel_window(start_time: float, end_time: float, fps: float, num_frames: int) -> tuple:
    """Half-open source pixel-frame window ``[start, end)`` for a retake window.

    Frames are indexed by ``round(time * fps)`` and clamped to ``[0, num_frames]``
    with ``start <= end`` so out-of-range or inverted times are safe (an inverted
    or degenerate window yields an empty ``[start, start)`` span).
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if num_frames < 0:
        raise ValueError(f"num_frames must be non-negative, got {num_frames}")
    start = max(0, min(int(round(start_time * fps)), num_frames))
    end = max(start, min(int(round(end_time * fps)), num_frames))
    return start, end


def _composite_retake_window(
    source: torch.Tensor, window: torch.Tensor, start_frame: int, end_frame: int
) -> torch.Tensor:
    """Splice regenerated ``window`` frames into ``source`` over ``[start, end)``.

    ``source`` and ``window`` are ``(..., T, H, W, C)`` video tensors that share
    every dimension except the frame count (dim ``-4``). Frames outside
    ``[start_frame, end_frame)`` are copied byte-for-byte from ``source``; frames
    inside are replaced by ``window``, which must have exactly
    ``end_frame - start_frame`` frames. This is the native composite-back that
    keeps non-retake frames byte-identical to the source.
    """
    if source.dim() < 4:
        raise ValueError(f"source must be (...,T,H,W,C); got {tuple(source.shape)}")
    expected = end_frame - start_frame
    if window.shape[-4] != expected:
        raise ValueError(
            f"composite window frame count {window.shape[-4]} != window span "
            f"{expected} (for [{start_frame}, {end_frame}))"
        )
    if source.shape[:-4] != window.shape[:-4] or source.shape[-3:] != window.shape[-3:]:
        raise ValueError(
            f"composite source/window shape mismatch: source {tuple(source.shape)} "
            f"window {tuple(window.shape)} (must match except the frame dim -4)"
        )
    out = source.clone()
    if expected > 0:
        out[..., start_frame:end_frame, :, :, :] = window.to(
            dtype=source.dtype, device=source.device
        )
    return out


def _pixel_frame_to_latent_index(pixel_frame: int, temporal_ratio: int) -> int:
    """Latent-frame index of a source pixel frame under the causal LTX-2 VAE.

    Pixel frame 0 maps to latent frame 0; pixel frames ``[1+(i-1)*r, 1+i*r)`` map
    to latent frame ``i`` (``r = temporal_ratio``), i.e. ``(f - 1)//r + 1``.
    """
    if pixel_frame <= 0:
        return 0
    return (pixel_frame - 1) // temporal_ratio + 1


def _latent_frame_count(num_frames: int, temporal_ratio: int) -> int:
    """Latent frame count for ``num_frames`` pixel frames: ``(T-1)//r + 1``."""
    return (num_frames - 1) // temporal_ratio + 1


def _retake_conditioned_latent_ranges(
    pixel_start: int, pixel_end: int, num_frames: int, temporal_ratio: int
) -> tuple:
    """Two-sided conditioned latent-frame ranges for a pixel retake window.

    Given a half-open source pixel window ``[pixel_start, pixel_end)`` (from
    :func:`_retake_pixel_window`), returns ``(latent_window, cond_ranges)``:

    - ``latent_window`` = ``(lat_start, lat_end)``: the latent frames the pixel
      window touches (regenerated). Any latent frame overlapped even partially by
      the pixel window is regenerated (conservative for a seamless retake).
    - ``cond_ranges``: the conditioned (context) latent-frame ranges outside the
      window — leading ``(0, lat_start)`` and trailing ``(lat_end, L)`` — ready to
      pass to ``_build_denoise_mask(cond_latent_frame_ranges=...)``.

    A full-frame window yields ``cond_ranges == []`` (everything regenerated); an
    empty/inverted window yields ``[(0, L)]`` (everything conditioned).
    """
    if temporal_ratio <= 0:
        raise ValueError(f"temporal_ratio must be positive, got {temporal_ratio}")
    if num_frames < 0:
        raise ValueError(f"num_frames must be non-negative, got {num_frames}")
    total_latent = _latent_frame_count(num_frames, temporal_ratio)
    if pixel_end <= pixel_start:
        return (0, 0), [(0, total_latent)]
    lat_start = max(0, min(_pixel_frame_to_latent_index(pixel_start, temporal_ratio), total_latent))
    lat_end = _pixel_frame_to_latent_index(pixel_end - 1, temporal_ratio) + 1
    lat_end = max(lat_start, min(lat_end, total_latent))
    cond_ranges = []
    if lat_start > 0:
        cond_ranges.append((0, lat_start))
    if lat_end < total_latent:
        cond_ranges.append((lat_end, total_latent))
    return (lat_start, lat_end), cond_ranges


def _init_retake_latents(
    noise_latents: torch.Tensor,
    source_latents: torch.Tensor,
    cond_latent_ranges: list,
) -> torch.Tensor:
    """Initialize native retake video latents for the two-sided window.

    Both tensors are ``(B, C, T_lat, H, W)`` and must share shape. The conditioned
    context latent frames (``cond_latent_ranges`` from
    :func:`_retake_conditioned_latent_ranges`) are taken from ``source_latents``
    (the ``_encode_video_window`` output); every other latent frame — the
    regenerated middle window — keeps the seeded ``noise_latents``. Returns a new
    tensor; inputs are unmodified. Ranges are clamped to ``[0, T_lat]``.
    """
    if noise_latents.dim() != 5:
        raise ValueError(f"expected (B, C, T, H, W) latents; got {tuple(noise_latents.shape)}")
    if noise_latents.shape != source_latents.shape:
        raise ValueError(
            f"noise/source latent shape mismatch: {tuple(noise_latents.shape)} vs "
            f"{tuple(source_latents.shape)}"
        )
    out = noise_latents.clone()
    total_latent = noise_latents.shape[2]
    for start_frame, end_frame in cond_latent_ranges:
        start_frame = max(0, start_frame)
        end_frame = min(total_latent, end_frame)
        if end_frame > start_frame:
            out[:, :, start_frame:end_frame] = source_latents[:, :, start_frame:end_frame]
    return out


# Comfy-format LoRA key conventions. The two exporters use different factor
# suffixes (PEFT ``lora_A``/``lora_B`` vs comfy ``lora_down``/``lora_up``) and
# different module prefixes; both are supported so the retake fusion matches
# whatever the real checkpoint uses (mirrors ``_load_lora_deltas`` in
# ``pipeline_ltx2_two_stages.py``).
_LORA_DOWN_SUFFIXES = (".lora_A.weight", ".lora_down.weight")  # A: (rank, in)
_LORA_UP_SUFFIXES = (".lora_B.weight", ".lora_up.weight")  # B: (out, rank)
# Longest-first so the most specific prefix is stripped when both would match.
_LORA_MODULE_PREFIXES = ("model.diffusion_model.", "diffusion_model.")


def _strip_lora_suffix(key: str, suffixes: tuple) -> Optional[str]:
    for suffix in suffixes:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def _strip_lora_prefix(module: str) -> str:
    for prefix in _LORA_MODULE_PREFIXES:
        if module.startswith(prefix):
            return module[len(prefix) :]
    return module


def _fuse_lora_into_transformer_weights(weights: dict, lora_path: str, strength: float) -> dict:
    """Fuse a comfy-format LoRA into base transformer weights, in place.

    ``weights`` are the base transformer state dict in the
    ``model.diffusion_model.``-stripped key space (e.g.
    ``transformer_blocks.0.attn1.to_q.weight``). The LoRA stores per-module
    low-rank factors: a down factor ``<prefix><module>.lora_A.weight`` /
    ``.lora_down.weight`` (shape ``(rank, in)``) and an up factor
    ``.lora_B.weight`` / ``.lora_up.weight`` (shape ``(out, rank)``), optionally
    with an ``<prefix><module>.alpha`` scalar. Stripping the module prefix
    (``model.diffusion_model.`` or ``diffusion_model.``) aligns ``<module>`` with
    the base keys. For each matched module this applies
    ``W += (alpha / rank) * strength * (B @ A)`` (aggregated in the base weight
    dtype). ``alpha`` defaults to ``rank`` (so the scale is just ``strength``),
    matching ``_load_lora_deltas``. Fusing happens in checkpoint-key space, before
    the native key remap (``LTXModel.load_weights`` handles ff/q_norm/QKV), so it
    needs no knowledge of native parameter names.
    """
    import safetensors.torch

    down: dict = {}
    up: dict = {}
    alphas: dict = {}
    # Accept a single .safetensors file, a directory, or split shards (like the
    # generation-side ``_load_lora_deltas``).
    for path in _find_safetensors_files(lora_path):
        with safetensors.torch.safe_open(path, framework="pt") as f:
            for key in f.keys():
                base = _strip_lora_suffix(key, _LORA_DOWN_SUFFIXES)
                if base is not None:
                    down[base] = f.get_tensor(key)
                    continue
                base = _strip_lora_suffix(key, _LORA_UP_SUFFIXES)
                if base is not None:
                    up[base] = f.get_tensor(key)
                    continue
                if key.endswith(".alpha"):
                    alphas[key[: -len(".alpha")]] = float(f.get_tensor(key).item())

    fused = 0
    missing = []
    for module in sorted(down):
        if module not in up:
            continue
        weight_key = _strip_lora_prefix(module) + ".weight"
        base = weights.get(weight_key)
        if base is None:
            missing.append(module)
            continue
        a = down[module]  # (rank, in)
        b = up[module]  # (out, rank)
        rank = a.shape[0]
        scale = strength * alphas.get(module, float(rank)) / rank
        dtype = base.dtype
        delta = torch.matmul(b.to(dtype), a.to(dtype)) * scale
        weights[weight_key] = (delta + base.to(delta.dtype)).to(dtype)
        fused += 1

    logger.info(
        f"LTX-2 retake LoRA fusion: fused {fused}/{len(down)} modules "
        f"from {lora_path} (strength={strength})"
    )
    if missing:
        logger.warning(
            f"LTX-2 retake LoRA: {len(missing)} LoRA modules had no matching base "
            f"weight and were skipped (e.g. {missing[:3]})"
        )
    if fused == 0:
        raise ValueError(
            f"LTX-2 retake LoRA at {lora_path} fused 0 modules; the key convention "
            f"does not match (expected '<prefix><module>.lora_A/lora_B.weight' or "
            f"'.lora_down/lora_up.weight', prefix 'model.diffusion_model.' or "
            f"'diffusion_model.'). Sampled base keys: {sorted(down)[:3]}"
        )
    return weights


class _NativeLTX2Companion(LTX2Pipeline):
    """Native ``LTX2Pipeline`` that shares an externally-built transformer.

    Constructed by :class:`LTX2RetakePipeline` so the native retake pre/post
    path reuses the native video VAE encoder/decoder, video patchifier,
    scheduler, Gemma text encoder, and connectors while the masked denoise runs
    on the retake pipeline's already-built native transformer. The transformer
    instance is shared (no second transformer is constructed and no transformer
    weights are loaded twice), and its CUDA-graph wrapping is owned by the retake
    pipeline, so this companion neither builds nor re-wraps it.
    """

    def __init__(self, pipeline_config, shared_transformer):
        # Stash the shared transformer directly in ``__dict__`` before
        # ``nn.Module.__init__`` runs: assigning a Module attribute pre-init
        # raises, and ``_init_transformer`` (called inside the base __init__)
        # reads it to share instead of building a fresh transformer.
        self.__dict__["_shared_transformer_box"] = (shared_transformer,)
        super().__init__(pipeline_config)

    def _init_transformer(self) -> None:
        self.transformer = self._shared_transformer_box[0]

    def _setup_cuda_graphs(self) -> None:
        # The shared transformer's CUDA-graph wrapping is owned by the retake
        # pipeline; do not wrap the same forward a second time here.
        return


class LTX2RetakePipeline(BasePipeline):
    """Persistent VisualGen adapter for LTX-2 retake requests."""

    def __init__(self, pipeline_config):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._retake_pipeline = None
        self._retake_params = None
        self._tiling_config = None
        self._get_videostream_metadata = None
        # Native pre/post companion (default path) and its source-media readers.
        self._native = None
        self._decode_video_by_frame = None
        self._decode_audio_from_file = None
        super().__init__(pipeline_config)

    @property
    def dtype(self):
        return torch.bfloat16

    @property
    def device(self):
        return self._device

    @property
    def default_generation_params(self):
        return {
            "num_inference_steps": 40,
        }

    @property
    def extra_param_specs(self):
        return {
            "retake_video_path": ExtraParamSchema(
                type="str",
                description="Path to the source video file for retake.",
            ),
            "retake_start_time": ExtraParamSchema(
                type="float",
                description="Start time in seconds for the regenerated window.",
            ),
            "retake_end_time": ExtraParamSchema(
                type="float",
                description="End time in seconds for the regenerated window.",
            ),
            "retake_regenerate_video": ExtraParamSchema(
                type="bool",
                default=True,
                description="Regenerate video inside the retake window.",
            ),
            "retake_regenerate_audio": ExtraParamSchema(
                type="bool",
                default=False,
                description=(
                    "Regenerate audio inside the retake window. The native retake path "
                    "is video-only and preserves the source audio, so this defaults to "
                    "False; True is rejected."
                ),
            ),
            "retake_enhance_prompt": ExtraParamSchema(
                type="bool",
                default=False,
                description="Enhance the retake prompt with the LTX text encoder.",
            ),
            "retake_max_batch_size": ExtraParamSchema(
                type="int",
                default=1,
                range=(1, 64),
                description="Maximum internal retake batch size.",
            ),
        }

    def _init_transformer(self) -> None:
        # Build the native LTX-2 transformer so retake runs on the native model
        # (and can inherit the config-driven acceleration stack) instead of the
        # no-op ``None`` that forced the eager upstream-wrapper path.
        self.transformer = build_ltx2_transformer(self.pipeline_config)

    def load_transformer_weights(self, checkpoint_dir: str) -> dict:
        # Read native transformer weights from the LTX-2 checkpoint (same format
        # and prefixes as the generation pipeline) instead of the previous no-op.
        return _load_ltx2_transformer_weights(
            checkpoint_dir,
            LTX2Pipeline._TRANSFORMER_PREFIX,
            exclude_prefixes=LTX2Pipeline._TRANSFORMER_EXCLUDE_PREFIXES,
        )

    def load_weights(self, weights: dict) -> None:
        if self.transformer is not None and hasattr(self.transformer, "load_weights"):
            transformer_weights = weights.get("transformer", weights)
            # Optionally fuse an identity/style LoRA into the base transformer
            # weights so the native path matches an upstream retake build that
            # used the same LoRA (e.g. the TalkVid identity LoRA). Fusing here,
            # before the native key remap, keeps it independent of native names.
            lora_path = self.pipeline_config.extra_attrs.get("retake_lora_path")
            if lora_path:
                strength = float(self.pipeline_config.extra_attrs.get("retake_lora_strength", 1.0))
                transformer_weights = _fuse_lora_into_transformer_weights(
                    transformer_weights, lora_path, strength
                )
            self.transformer.load_weights(transformer_weights)

    def post_load_weights(self) -> None:
        # Finalize the native transformer (e.g. dynamic-quant weight loading)
        # when it exposes a post-load hook.
        if self.transformer is not None and hasattr(self.transformer, "post_load_weights"):
            self.transformer.post_load_weights()
        # The native transformer is constructed on CPU (build_ltx2_transformer)
        # and its weights are loaded in-place; move it onto the pipeline device
        # so the denoise forward matches the CUDA latents/context tensors.
        if self.transformer is not None:
            self.transformer.to(self._device)

    def _setup_cuda_graphs(self) -> None:
        # Retake now builds the native LTXModel, whose ``forward`` takes
        # ``Modality`` dataclasses (not flat tensors). Delegate to the
        # generation pipeline's Modality-aware setup so cuda graph uses
        # ``_LTX2CUDAGraphRunner`` rather than the base ``CUDAGraphRunner`` and
        # is allowed to compose with torch.compile (the base implementation
        # disables cuda graph whenever torch_compile is enabled).
        LTX2Pipeline._setup_cuda_graphs(self)

    def _use_upstream_stage(self) -> bool:
        """Whether the upstream ``DiffusionStage.run`` oracle path is selected.

        Native retake pre/post is the default; setting
        ``pipeline_config.extra_attrs['retake_use_upstream_stage'] = True``
        selects the preserved upstream-stage path for comparison / verification.
        """
        return bool(self.pipeline_config.extra_attrs.get("retake_use_upstream_stage", False))

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list] = None,
        *,
        text_encoder_path: str = "",
        **kwargs,
    ) -> None:
        self._device = device
        if not text_encoder_path:
            raise ValueError(
                "LTX-2 retake workflow requires pipeline_config.text_encoder_path "
                "to point at the Gemma text encoder directory."
            )
        if self._use_upstream_stage():
            self._load_upstream_stage_components(checkpoint_dir, device, text_encoder_path)
        else:
            self._load_native_retake_components(checkpoint_dir, device, text_encoder_path)

    def _load_native_retake_components(
        self, checkpoint_dir: str, device: torch.device, text_encoder_path: str
    ) -> None:
        """Load the native pre/post companion and source-media readers.

        Builds a native ``LTX2Pipeline`` that shares this pipeline's native
        transformer (so no second transformer is constructed and no transformer
        weights are loaded twice) and loads its native video VAE encoder/decoder,
        patchifier, scheduler, Gemma text encoder, and connectors. Only the
        deterministic source pixel/audio decode and stream metadata come from the
        upstream ``media_io`` readers.
        """
        try:
            from ltx_pipelines.utils.media_io import (
                decode_audio_from_file,
                decode_video_by_frame,
                get_videostream_metadata,
            )
        except (ImportError, OSError) as exc:
            raise ImportError(
                "LTX-2 retake workflow requires the optional Lightricks "
                "`ltx-pipelines` package for deterministic source video/audio "
                "decode. Install it in the VisualGen runtime environment before "
                "starting trtllm-serve with pipeline_config.workflow=retake."
            ) from exc

        self._get_videostream_metadata = get_videostream_metadata
        self._decode_video_by_frame = decode_video_by_frame
        self._decode_audio_from_file = decode_audio_from_file

        logger.info(
            "Loading LTX-2 native retake pre/post components "
            f"(checkpoint={checkpoint_dir}, text_encoder={text_encoder_path})"
        )
        native = _NativeLTX2Companion(self.pipeline_config, self.transformer)
        native.load_standard_components(checkpoint_dir, device, text_encoder_path=text_encoder_path)
        # Derived attribute used to build the video latent shape; normally set by
        # LTX2Pipeline.post_load_weights, which we skip to avoid re-running the
        # shared transformer's post-load hook.
        native.transformer_in_channels = native.transformer._transformer_config.get(
            "in_channels", 128
        )
        self._native = native

    def _load_upstream_stage_components(
        self, checkpoint_dir: str, device: torch.device, text_encoder_path: str
    ) -> None:
        """Load the preserved upstream ``DiffusionStage.run`` oracle components."""
        try:
            from ltx_core.model.video_vae import TilingConfig
            from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
            from ltx_pipelines.retake import RetakePipeline
            from ltx_pipelines.utils.blocks import PromptEncoder
            from ltx_pipelines.utils.constants import detect_params
            from ltx_pipelines.utils.media_io import get_videostream_metadata
            from ltx_pipelines.utils.types import OffloadMode
        except (ImportError, OSError) as exc:
            raise ImportError(
                "LTX-2 retake workflow requires the optional Lightricks "
                "`ltx-pipelines` package. Install it in the VisualGen runtime "
                "environment before starting trtllm-serve with "
                "pipeline_config.workflow=retake."
            ) from exc

        self._install_gemma_meta_device_workaround(GemmaTextEncoder)
        self._install_prompt_encoder_cache(PromptEncoder)
        distilled = bool(self.pipeline_config.extra_attrs.get("retake_distilled", True))
        offload_mode = self._resolve_offload_mode(OffloadMode)
        logger.info(
            "Loading LTX-2 retake workflow via ltx_pipelines "
            f"(distilled={distilled}, offload_mode={offload_mode.value}, checkpoint={checkpoint_dir})"
        )
        self._retake_pipeline = RetakePipeline(
            checkpoint_path=checkpoint_dir,
            gemma_root=text_encoder_path,
            loras=[],
            device=device,
            distilled=distilled,
            offload_mode=offload_mode,
        )
        prompt_cache_size = int(
            self.pipeline_config.extra_attrs.get("retake_prompt_cache_size", 16)
        )
        self._retake_pipeline.prompt_encoder._trtllm_prompt_cache_size = prompt_cache_size
        self._retake_pipeline.prompt_encoder._trtllm_prompt_cache = OrderedDict()
        self._retake_params = detect_params(checkpoint_dir)
        self._tiling_config = TilingConfig.default()
        self._get_videostream_metadata = get_videostream_metadata

    def _resolve_offload_mode(self, offload_mode_cls):
        raw_mode = self.pipeline_config.extra_attrs.get("retake_offload_mode", "none")
        if isinstance(raw_mode, offload_mode_cls):
            return raw_mode
        try:
            return offload_mode_cls(str(raw_mode).lower())
        except ValueError as exc:
            valid = ", ".join(mode.value for mode in offload_mode_cls)
            raise ValueError(
                f"Unsupported LTX-2 retake offload mode {raw_mode!r}; expected one of: {valid}."
            ) from exc

    @staticmethod
    def _resolve_non_meta_model_device(model) -> torch.device:
        inner_model = getattr(model, "model", None)
        language_model = getattr(inner_model, "language_model", None)
        candidates = (language_model, inner_model, model)
        for module in candidates:
            if module is None:
                continue
            parameters = getattr(module, "parameters", None)
            if parameters is not None:
                for tensor in parameters(recurse=True):
                    if tensor.device.type != "meta":
                        return tensor.device
            buffers = getattr(module, "buffers", None)
            if buffers is not None:
                for tensor in buffers(recurse=True):
                    if tensor.device.type != "meta":
                        return tensor.device

        device = getattr(model, "device", torch.device("cpu"))
        if isinstance(device, torch.device):
            return device
        return torch.device(device)

    @classmethod
    def _install_gemma_meta_device_workaround(cls, gemma_text_encoder_cls) -> None:
        if getattr(gemma_text_encoder_cls, "_trtllm_ltx_retake_device_patch", False):
            return

        original_encode = gemma_text_encoder_cls.encode

        def encode_with_non_meta_device(self, prompts: list[str], padding_side: str = "left"):
            if not prompts:
                return []

            tokenized = [self.tokenizer.tokenize_with_weights(t)["gemma"] for t in prompts]
            device = cls._resolve_non_meta_model_device(self.model)
            input_ids = torch.tensor(
                [[tok for tok, _ in pairs] for pairs in tokenized],
                device=device,
            )
            attention_mask = torch.tensor(
                [[w for _, w in pairs] for pairs in tokenized],
                device=device,
            )
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states
            del outputs
            return [
                (tuple(h[i : i + 1] for h in hidden_states), attention_mask[i : i + 1])
                for i in range(len(prompts))
            ]

        encode_with_non_meta_device.__name__ = original_encode.__name__
        encode_with_non_meta_device.__doc__ = original_encode.__doc__
        encode_with_non_meta_device._trtllm_original_encode = original_encode
        gemma_text_encoder_cls.encode = encode_with_non_meta_device
        gemma_text_encoder_cls._trtllm_ltx_retake_device_patch = True

    @classmethod
    def _install_prompt_encoder_cache(cls, prompt_encoder_cls) -> None:
        if getattr(prompt_encoder_cls, "_trtllm_ltx_retake_prompt_cache_patch", False):
            return

        original_call = prompt_encoder_cls.__call__

        def call_with_prompt_cache(
            self,
            prompts: list[str],
            *,
            enhance_first_prompt: bool = False,
            enhance_prompt_image: str | None = None,
            enhance_prompt_seed: int = 42,
        ):
            cache_size = int(getattr(self, "_trtllm_prompt_cache_size", 0) or 0)
            if cache_size <= 0:
                return original_call(
                    self,
                    prompts,
                    enhance_first_prompt=enhance_first_prompt,
                    enhance_prompt_image=enhance_prompt_image,
                    enhance_prompt_seed=enhance_prompt_seed,
                )

            cache = getattr(self, "_trtllm_prompt_cache", None)
            if cache is None:
                cache = OrderedDict()
                self._trtllm_prompt_cache = cache
            key = (
                tuple(prompts),
                bool(enhance_first_prompt),
                enhance_prompt_image,
                int(enhance_prompt_seed),
                str(getattr(self, "_gemma_root", "")),
                str(getattr(self, "_checkpoint_path", "")),
                str(getattr(self, "_dtype", "")),
                str(getattr(self, "_device", "")),
                str(getattr(self, "_offload_mode", "")),
            )
            if key in cache:
                cache.move_to_end(key)
                logger.info("LTX-2 retake prompt encoding cache hit")
                return cache[key]

            result = original_call(
                self,
                prompts,
                enhance_first_prompt=enhance_first_prompt,
                enhance_prompt_image=enhance_prompt_image,
                enhance_prompt_seed=enhance_prompt_seed,
            )
            cache[key] = result
            cache.move_to_end(key)
            while len(cache) > cache_size:
                cache.popitem(last=False)
            logger.info("LTX-2 retake prompt encoding cache miss")
            return result

        call_with_prompt_cache.__name__ = original_call.__name__
        call_with_prompt_cache.__doc__ = original_call.__doc__
        call_with_prompt_cache._trtllm_original_call = original_call
        prompt_encoder_cls.__call__ = call_with_prompt_cache
        prompt_encoder_cls._trtllm_ltx_retake_prompt_cache_patch = True

    def warmup(self) -> None:
        logger.info("Skipping LTX-2 retake warmup; retake requires a source video request.")

    @torch.inference_mode()
    def infer(self, req):
        extra = req.params.extra_params or {}
        video_path = self._require_extra(extra, "retake_video_path")
        start_time = float(self._require_extra(extra, "retake_start_time"))
        end_time = float(self._require_extra(extra, "retake_end_time"))
        if start_time >= end_time:
            raise ValueError(
                f"retake_start_time ({start_time}) must be less than retake_end_time ({end_time})"
            )

        prompt = self._single_prompt(req.prompt)
        retake_start = time.perf_counter()
        if self._use_upstream_stage():
            if self._retake_pipeline is None:
                raise RuntimeError("LTX-2 retake upstream-stage pipeline has not been loaded.")
            video_iter, audio, output_shape = self._run_native_retake(
                req, extra, video_path, start_time, end_time, prompt
            )
            video = self._materialize_video(video_iter)
        else:
            if self._native is None:
                raise RuntimeError("LTX-2 native retake pipeline has not been loaded.")
            video, audio, output_shape = self._run_native_pre_post_retake(
                req, extra, video_path, start_time, end_time, prompt
            )
        audio_tensor, sample_rate = self._normalize_audio(audio)
        elapsed = time.perf_counter() - retake_start
        return PipelineOutput(
            video=video,
            audio=audio_tensor,
            frame_rate=float(output_shape.fps),
            audio_sample_rate=sample_rate,
            denoise=elapsed,
        )

    @staticmethod
    def _import_upstream_retake_symbols() -> SimpleNamespace:
        # Import the upstream retake pre/post building blocks lazily (same
        # optional Lightricks dependency as ``load_standard_components``), so the
        # module imports cleanly in environments without ``ltx-pipelines`` and so
        # host tests can monkeypatch this one seam to inject fakes.
        try:
            from ltx_core.components.noisers import GaussianNoiser
            from ltx_core.conditioning.types.noise_mask_cond import TemporalRegionMask
            from ltx_pipelines.utils.constants import DISTILLED_SIGMAS
            from ltx_pipelines.utils.denoisers import SimpleDenoiser
            from ltx_pipelines.utils.helpers import audio_latent_from_file, video_latent_from_file
            from ltx_pipelines.utils.types import ModalitySpec
        except (ImportError, OSError) as exc:
            raise ImportError(
                "LTX-2 retake native path requires the optional Lightricks "
                "`ltx-pipelines`/`ltx-core` packages for source I/O, the "
                "temporal-region mask, and the distilled denoise schedule. "
                "Install them in the VisualGen runtime before serving retake."
            ) from exc
        return SimpleNamespace(
            GaussianNoiser=GaussianNoiser,
            TemporalRegionMask=TemporalRegionMask,
            DISTILLED_SIGMAS=DISTILLED_SIGMAS,
            SimpleDenoiser=SimpleDenoiser,
            audio_latent_from_file=audio_latent_from_file,
            video_latent_from_file=video_latent_from_file,
            ModalitySpec=ModalitySpec,
        )

    def _run_native_retake(self, req, extra, video_path, start_time, end_time, prompt):
        """Run retake denoising on the native transformer via the upstream loop.

        Reuses the resident upstream pre/post components (image/audio
        conditioners, prompt encoder, diffusion stage, VAE decoders) but injects
        the native ``LTXModel`` into ``DiffusionStage.run`` via
        ``LTX2RetakeNativeAdapter`` instead of calling the eager
        ``RetakePipeline.__call__`` (which would build and free an upstream
        transformer). Mirrors ``RetakePipeline.__call__`` so the schedule,
        temporal-region mask, and modality specs stay byte-for-byte identical;
        only the denoiser transformer differs.

        Returns ``(decoded_video, decoded_audio, output_shape)``.
        """
        if self.transformer is None:
            raise RuntimeError(
                "LTX-2 retake native path requires the native transformer, "
                "but self.transformer is None."
            )

        retake = self._retake_pipeline
        if not getattr(retake, "distilled", False):
            raise NotImplementedError(
                "LTX-2 retake native path currently supports only the distilled "
                "(non-guided) schedule; the native adapter rejects STG "
                "perturbations required by the guided path."
            )

        regenerate_video = bool(extra["retake_regenerate_video"])
        regenerate_audio = bool(extra["retake_regenerate_audio"])
        if not regenerate_video:
            raise NotImplementedError(
                "LTX-2 retake native path regenerates the video window; set "
                "retake_regenerate_video=True."
            )
        if regenerate_audio:
            raise NotImplementedError(
                "LTX-2 retake native path is video-only and preserves the source "
                "audio; retake_regenerate_audio=True is not supported."
            )

        up = self._import_upstream_retake_symbols()
        dtype = self.dtype
        device = self._device
        seed = req.params.seed
        generator = torch.Generator(device=device).manual_seed(seed)
        noiser = up.GaussianNoiser(generator=generator)

        output_shape = self._get_videostream_metadata(video_path)
        initial_video_latent = retake.image_conditioner(
            lambda enc: up.video_latent_from_file(
                video_encoder=enc,
                file_path=video_path,
                output_shape=output_shape,
                dtype=dtype,
                device=device,
            )
        )
        initial_audio_latent = retake.audio_conditioner(
            lambda enc: up.audio_latent_from_file(
                audio_encoder=enc,
                file_path=video_path,
                output_shape=output_shape,
                dtype=dtype,
                device=device,
            )
        )

        contexts = retake.prompt_encoder(
            [prompt],
            enhance_first_prompt=bool(extra["retake_enhance_prompt"]),
            enhance_prompt_seed=seed,
        )
        v_context_p = contexts[0].video_encoding
        a_context_p = contexts[0].audio_encoding

        # Regenerate the video window; keep the source audio frozen. Mirrors the
        # ``regenerate_video=True, regenerate_audio=False`` branch of upstream
        # ``RetakePipeline.__call__``.
        video_modality_spec = up.ModalitySpec(
            context=v_context_p,
            conditionings=[
                up.TemporalRegionMask(
                    start_time=start_time, end_time=end_time, fps=output_shape.fps
                )
            ],
            initial_latent=initial_video_latent,
            frozen=False,
        )
        audio_modality_spec = up.ModalitySpec(
            context=a_context_p,
            conditionings=[],
            initial_latent=initial_audio_latent,
            frozen=initial_audio_latent is not None,
        )

        sigmas = up.DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
        denoiser = up.SimpleDenoiser(v_context=v_context_p, a_context=a_context_p)
        adapter = LTX2RetakeNativeAdapter(self.transformer, model_dtype=dtype)

        video_state, audio_state = retake.stage.run(
            adapter,
            denoiser,
            sigmas,
            noiser,
            output_shape.width,
            output_shape.height,
            output_shape.frames,
            output_shape.fps,
            video=video_modality_spec,
            audio=audio_modality_spec,
            max_batch_size=int(extra["retake_max_batch_size"]),
        )

        decoded_video = retake.video_decoder(video_state.latent, self._tiling_config, generator)
        decoded_audio = (
            retake.audio_decoder(audio_state.latent) if audio_state is not None else None
        )
        return decoded_video, decoded_audio, output_shape

    # ------------------------------------------------------------------
    # Native retake pre/post runtime path (default)
    # ------------------------------------------------------------------

    def _run_native_pre_post_retake(self, req, extra, video_path, start_time, end_time, prompt):
        """Native source-encode -> masked denoise -> decode -> composite-back.

        Reuses the native ``LTX2Pipeline`` machinery (video VAE encoder/decoder,
        video patchifier, scheduler, Gemma text encoder + connectors, and the
        masked-denoise entry :meth:`LTX2Pipeline._masked_transformer_step`) with
        retake-specific inputs: initial latents seeded from the encoded source,
        a two-sided ``denoise_mask`` conditioning the leading + trailing context
        while regenerating the middle window, and a two-sided ``clean_latent``
        from the source. The regenerated pixel window is spliced back into the
        original source frames so non-window frames are byte-identical, and the
        source audio is passed through unchanged (video-only regeneration).

        Returns ``(composited_video_uint8, source_audio, output_shape)`` where
        ``composited_video_uint8`` is ``(1, T, H, W, C)`` uint8.
        """
        native = self._native
        if native is None:
            raise RuntimeError(
                "LTX-2 native retake path requires the native companion pipeline, "
                "but self._native is None."
            )

        regenerate_video = bool(extra["retake_regenerate_video"])
        regenerate_audio = bool(extra["retake_regenerate_audio"])
        if not regenerate_video:
            raise NotImplementedError(
                "LTX-2 native retake regenerates the video window; set "
                "retake_regenerate_video=True."
            )
        if regenerate_audio:
            raise NotImplementedError(
                "LTX-2 native retake is video-only and preserves the source audio; "
                "retake_regenerate_audio=True is not supported."
            )

        device = self._device
        dtype = self.dtype
        seed = req.params.seed
        generator = torch.Generator(device=device).manual_seed(seed)

        # ---- 1. Source read + validation --------------------------------
        output_shape = self._get_videostream_metadata(video_path)
        num_frames = int(output_shape.frames)
        height = int(output_shape.height)
        width = int(output_shape.width)
        fps = float(output_shape.fps)
        self._validate_retake_source(num_frames, height, width)

        source_uint8, source_norm_5d = self._read_source_video(
            video_path, num_frames, height, width, device, dtype
        )

        # ---- 2. Retake windows ------------------------------------------
        temporal_ratio = VIDEO_SCALE_FACTORS.time
        pixel_start, pixel_end = _retake_pixel_window(start_time, end_time, fps, num_frames)
        _latent_window, conditioned_latent_ranges = _retake_conditioned_latent_ranges(
            pixel_start, pixel_end, num_frames, temporal_ratio
        )

        # ---- 3. Native VAE encode + seed initial latents ----------------
        pixel_shape = VideoPixelShape(
            batch=1, frames=num_frames, height=height, width=width, fps=fps
        )
        video_shape = VideoLatentShape.from_pixel_shape(
            pixel_shape, latent_channels=native.transformer_in_channels
        )
        source_window_latents = native._encode_video_window(source_norm_5d).float()
        expected_latent_shape = tuple(video_shape.to_torch_shape())
        if tuple(source_window_latents.shape) != expected_latent_shape:
            raise ValueError(
                "LTX-2 native retake: encoded source latent shape "
                f"{tuple(source_window_latents.shape)} != expected "
                f"{expected_latent_shape}; check source resolution/frame count."
            )

        noise_latents = torch.randn(
            video_shape.to_torch_shape(),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        initial_latents = _init_retake_latents(
            noise_latents, source_window_latents, conditioned_latent_ranges
        )
        latents = native.video_patchifier.patchify(initial_latents)

        clean_5d = torch.zeros_like(noise_latents)
        total_latent = video_shape.frames
        for start_frame, end_frame in conditioned_latent_ranges:
            start_frame = max(0, start_frame)
            end_frame = min(total_latent, end_frame)
            if end_frame > start_frame:
                clean_5d[:, :, start_frame:end_frame] = source_window_latents[
                    :, :, start_frame:end_frame
                ]
        clean_latent = native.video_patchifier.patchify(clean_5d)

        denoise_mask = native._build_denoise_mask(
            video_shape, cond_latent_frame_ranges=conditioned_latent_ranges
        )

        # ---- 4. Native prompt encode + connectors + text cache ----------
        max_sequence_length = getattr(req.params, "max_sequence_length", None) or 1024
        prompt_embeds, prompt_attention_mask = native._encode_prompt(
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        video_embeds, _audio_embeds, connector_mask = native._process_connectors(
            prompt_embeds, prompt_attention_mask
        )

        video_positions = native.video_patchifier.get_patch_grid_bounds(video_shape, device=device)
        video_positions = get_pixel_coords(
            video_positions.float(), VIDEO_SCALE_FACTORS, causal_fix=True
        )
        video_positions[:, 0, ...] = video_positions[:, 0, ...] / fps
        video_positions = video_positions.to(dtype)

        text_cache = native.transformer.prepare_text_cache(
            video_context=video_embeds,
            video_context_mask=connector_mask,
            video_positions=video_positions,
            audio_context=None,
            audio_context_mask=None,
            audio_positions=None,
            dtype=dtype,
        )

        # ---- 5. Native masked denoise (distilled, video-only, non-guided) --
        scheduler = NativeSchedulerAdapter()
        scheduler.sigmas = self._retake_distilled_sigmas(device)
        scheduler._step_index = 0
        timesteps = scheduler.timesteps
        num_steps = len(timesteps)

        def retake_forward_fn(
            video_latents,
            extra_stream_latents,
            step_index,
            timestep,
            encoder_hidden_states,
            extra_tensors,
        ):
            denoised_video, _ = native._masked_transformer_step(
                video_latents,
                None,
                step_index,
                timestep,
                video_embeds,
                None,
                connector_mask,
                video_positions=video_positions,
                audio_positions=None,
                denoise_mask=denoise_mask,
                clean_latent=clean_latent,
                num_steps=num_steps,
                text_cache=text_cache,
            )
            return denoised_video, {}

        denoised_latents = native.denoise(
            latents=latents,
            scheduler=scheduler,
            prompt_embeds=video_embeds,
            guidance_scale=1.0,
            forward_fn=retake_forward_fn,
            timesteps=timesteps,
        )

        # ---- 6. Native decode -------------------------------------------
        video_latents_5d = native.video_patchifier.unpatchify(denoised_latents, video_shape).to(
            dtype
        )
        chunks = list(
            native.video_decoder.tiled_decode(
                video_latents_5d, TilingConfig.default(), generator=generator
            )
        )
        decoded = torch.cat(chunks, dim=2)  # (B, C, T, H, W)
        decoded = postprocess_video_tensor(decoded)  # (B, T, H, W, C) uint8

        # ---- 7. Composite regenerated window back into the source -------
        regenerated_pixel_window = decoded[:, pixel_start:pixel_end]
        composited_video = _composite_retake_window(
            source_uint8, regenerated_pixel_window, pixel_start, pixel_end
        ).contiguous()

        # ---- 8. Source audio passthrough --------------------------------
        source_audio = self._read_source_audio(video_path, device)

        return composited_video, source_audio, output_shape

    @staticmethod
    def _validate_retake_source(num_frames: int, height: int, width: int) -> None:
        """Fail fast on source video shapes the native VAE cannot round-trip.

        The LTX-2 causal video VAE requires ``8k + 1`` pixel frames and spatial
        dimensions that are multiples of 32, so encode -> decode reconstructs the
        exact source frame count / resolution (required for a byte-identical
        composite of the non-window frames).
        """
        ratio = VIDEO_SCALE_FACTORS.time
        if num_frames <= 0:
            raise ValueError(f"retake source must have frames; got {num_frames}")
        if (num_frames - 1) % ratio != 0:
            snapped = ((num_frames - 1) // ratio) * ratio + 1
            raise ValueError(
                f"retake source frame count must satisfy {ratio}k+1 (e.g. 97, 193); "
                f"got {num_frames}. Use a source with {snapped} frames."
            )
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(
                f"retake source resolution must be a multiple of 32; got {height}x{width}."
            )

    def _read_source_video(self, video_path, num_frames, height, width, device, dtype):
        """Read the source video as raw uint8 frames and a normalized VAE input.

        Returns ``(source_uint8, source_norm_5d)`` where ``source_uint8`` is
        ``(1, T, H, W, C)`` uint8 (the original pixels, kept for a byte-identical
        composite) and ``source_norm_5d`` is ``(1, 3, T, H, W)`` in ``[-1, 1]``
        (the VAE encoder input). Pixel decode is deterministic (sequential frame
        index via the upstream ``media_io`` reader).
        """
        frames = list(
            self._decode_video_by_frame(path=video_path, device=device, frame_cap=num_frames)
        )
        if not frames:
            raise ValueError(f"retake source video decoded no frames: {video_path}")
        source_uint8 = torch.cat(frames, dim=0).unsqueeze(0)  # (1, T, H, W, C)
        decoded_frames = source_uint8.shape[1]
        if decoded_frames != num_frames:
            raise ValueError(
                f"retake source decoded {decoded_frames} frames but metadata "
                f"reported {num_frames}; cannot composite a byte-identical result."
            )
        if source_uint8.shape[2] != height or source_uint8.shape[3] != width:
            raise ValueError(
                f"retake source frame size {tuple(source_uint8.shape[2:4])} does not "
                f"match metadata {(height, width)}."
            )
        # uint8 [0, 255] -> [-1, 1], laid out (1, C, T, H, W) for the VAE encoder.
        normalized = source_uint8[0].to(torch.float32) / 127.5 - 1.0  # (T, H, W, C)
        source_norm_5d = normalized.permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=dtype)
        return source_uint8, source_norm_5d

    def _read_source_audio(self, video_path, device):
        """Read the source audio unchanged (video-only retake preserves audio).

        Returns the upstream ``Audio`` object (``.waveform`` / ``.sampling_rate``)
        or ``None`` when the source has no audio stream.
        """
        return self._decode_audio_from_file(video_path, device)

    @staticmethod
    def _retake_distilled_sigmas(device) -> torch.Tensor:
        return torch.tensor(_RETAKE_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=device)

    @staticmethod
    def _single_prompt(prompt: str | list[str]) -> str:
        if isinstance(prompt, str):
            return prompt
        if len(prompt) == 1:
            return prompt[0]
        raise ValueError("LTX-2 retake workflow supports one prompt per request.")

    @staticmethod
    def _require_extra(extra: dict[str, Any], key: str) -> Any:
        value = extra.get(key)
        if value is None:
            raise ValueError(f"extra_params['{key}'] is required for LTX-2 retake.")
        return value

    @staticmethod
    def _to_uint8_video_chunk(chunk: torch.Tensor) -> torch.Tensor:
        if chunk.dtype == torch.uint8:
            return chunk
        if torch.is_floating_point(chunk):
            return (chunk.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        return chunk.to(torch.uint8)

    @classmethod
    def _materialize_video(cls, video: torch.Tensor | Iterator[torch.Tensor]) -> torch.Tensor:
        chunks = [video] if isinstance(video, torch.Tensor) else list(video)
        if not chunks:
            raise ValueError("LTX-2 retake produced no video frames.")

        normalized = []
        for chunk in chunks:
            chunk = cls._to_uint8_video_chunk(chunk.detach())
            if chunk.dim() == 3:
                chunk = chunk.unsqueeze(0)
            if chunk.dim() != 4:
                raise ValueError(
                    "LTX-2 retake video chunks must have shape (T,H,W,C) "
                    f"or (H,W,C); got {tuple(chunk.shape)}"
                )
            normalized.append(chunk)
        return torch.cat(normalized, dim=0).unsqueeze(0).contiguous()

    @staticmethod
    def _normalize_audio(audio) -> tuple[torch.Tensor | None, int | None]:
        if audio is None:
            return None, None
        waveform = getattr(audio, "waveform", audio)
        sample_rate = getattr(audio, "sampling_rate", None)
        if waveform is None:
            return None, sample_rate
        waveform = waveform.detach().to(torch.float32)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() != 3:
            raise ValueError(
                "LTX-2 retake audio must have shape (samples), "
                f"(channels,samples), or (B,channels,samples); got {tuple(waveform.shape)}"
            )
        return waveform.contiguous(), sample_rate
