# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native LTX-2 retake workflow for VisualGen.

This pipeline keeps a resident retake model for ``trtllm-serve``. A native
``LTX2Pipeline`` companion shares the retake pipeline's transformer and provides
the VAE, scheduler, Gemma text encoder, and connectors. Retake generalizes the
native image-to-video mask to condition both sides of the regenerated window.

Source video/audio decode, stream metadata, audio/video VAE encode, diffusion,
and VAE decode all use TensorRT-LLM's native LTX-2 implementation. PyAV is the
only source-media I/O dependency; no upstream LTX pipeline is imported.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from tensorrt_llm._torch.visual_gen.output import PipelineOutput, RetakeStageTimer
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline, ExtraParamSchema
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm.logger import logger

from .ltx2_core.patchifier import get_pixel_coords
from .ltx2_core.scheduler_adapter import NativeSchedulerAdapter
from .ltx2_core.types import (
    VIDEO_SCALE_FACTORS,
    AudioLatentShape,
    VideoLatentShape,
    VideoPixelShape,
)
from .ltx2_core.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from .pipeline_ltx2 import (
    LTX2Pipeline,
    _find_safetensors_files,
    _load_ltx2_transformer_weights,
    build_ltx2_transformer,
)

# Distilled retake noise schedule (8 Euler steps).
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

# Retake uses the same tiling geometry for source encode and output decode. Keep
# it separate from the generation pipeline's default because tile boundaries are
# part of the retake numerical contract.
_RETAKE_TILING_CONFIG = TilingConfig(
    spatial_config=SpatialTilingConfig(tile_size_in_pixels=768, tile_overlap_in_pixels=64),
    temporal_config=TemporalTilingConfig(tile_size_in_frames=80, tile_overlap_in_frames=24),
)


def _retake_pixel_window(
    start_time: float, end_time: float, fps: float, num_frames: int
) -> tuple[int, int]:
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
) -> list[tuple[int, int]]:
    """Return conditioned latent ranges outside a pixel retake window.

    Any latent frame touched by ``[pixel_start, pixel_end)`` is regenerated.
    Leading and trailing latent ranges are returned for conditioning.

    A full-frame window returns ``[]``; an empty window returns ``[(0, L)]``.
    """
    if temporal_ratio <= 0:
        raise ValueError(f"temporal_ratio must be positive, got {temporal_ratio}")
    if num_frames < 0:
        raise ValueError(f"num_frames must be non-negative, got {num_frames}")
    total_latent = _latent_frame_count(num_frames, temporal_ratio)
    if pixel_end <= pixel_start:
        return [(0, total_latent)]
    lat_start = max(0, min(_pixel_frame_to_latent_index(pixel_start, temporal_ratio), total_latent))
    lat_end = _pixel_frame_to_latent_index(pixel_end - 1, temporal_ratio) + 1
    lat_end = max(lat_start, min(lat_end, total_latent))
    cond_ranges = []
    if lat_start > 0:
        cond_ranges.append((0, lat_start))
    if lat_end < total_latent:
        cond_ranges.append((lat_end, total_latent))
    return cond_ranges


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


def _conform_latent_length(latent: torch.Tensor, expected_frames_count: int) -> torch.Tensor:
    """Crop or zero-pad *latent* along dim 2 so it has exactly the expected frames.

    Encoders emit a frame count driven by the decoded stream length, which need
    not match the count the target shape implies; the transformer needs the exact
    count. Missing audio latent frames carry no conditioning, so they are padded
    with zeros.
    """
    actual_frames = latent.shape[2]
    if actual_frames > expected_frames_count:
        return latent[:, :, :expected_frames_count]
    if actual_frames < expected_frames_count:
        pad_shape = list(latent.shape)
        pad_shape[2] = expected_frames_count - actual_frames
        pad = torch.zeros(pad_shape, device=latent.device, dtype=latent.dtype)
        return torch.cat([latent, pad], dim=2)
    return latent


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
    """Native ``LTX2Pipeline`` that shares its parent's transformer.

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
        pass


class LTX2RetakePipeline(BasePipeline):
    """Persistent VisualGen adapter for LTX-2 retake requests."""

    def __init__(self, pipeline_config):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._get_videostream_metadata = None
        # Native pre/post companion and its source-media readers.
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
            "num_inference_steps": 8,
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
        }

    def _init_transformer(self) -> None:
        # Build the native LTX-2 transformer so retake runs on the native model
        # and can inherit the config-driven acceleration stack.
        self.transformer = build_ltx2_transformer(self.pipeline_config)

    def load_transformer_weights(self, checkpoint_dir: str) -> dict:
        # Retake and generation checkpoints use the same transformer prefixes.
        return _load_ltx2_transformer_weights(
            checkpoint_dir,
            LTX2Pipeline._TRANSFORMER_PREFIX,
            exclude_prefixes=LTX2Pipeline._TRANSFORMER_EXCLUDE_PREFIXES,
        )

    def load_weights(self, weights: dict) -> None:
        if self.transformer is not None and hasattr(self.transformer, "load_weights"):
            transformer_weights = weights.get("transformer", weights)
            # Fuse an optional identity/style LoRA before the native key remap.
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
        self._build_fp8_step_transformer()

    def _build_fp8_step_transformer(self) -> None:
        """Build a resident FP8 transformer for selected denoising steps.

        When ``retake_fp8_linear_steps`` is set, the masked denoise runs this FP8
        transformer at those diffusion steps instead of the primary (NVFP4) one to
        recover the quality NVFP4 linears lose at those sensitive steps. Both
        transformers stay resident; the caller is expected to offload the Gemma
        text encoder after prompt encoding to make room (see
        ``_run_native_pre_post_retake``).
        """
        import copy

        steps = self.pipeline_config.extra_attrs.get("retake_fp8_linear_steps")
        if not steps or self._native is None or self.transformer is None:
            return
        from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig

        fp8_qc, _lqc, fp8_dwq, _daq = DiffusionPipelineConfig.load_diffusion_quant_config(
            {"quant_algo": "FP8", "dynamic": True}
        )
        base_mc = self.pipeline_config.model_configs["transformer"]
        fp8_mc = copy.deepcopy(base_mc)
        fp8_mc.quant_config = fp8_qc
        fp8_mc.dynamic_weight_quant = fp8_dwq
        fp8_mc.force_dynamic_quantization = True
        fp8_cfg = copy.copy(self.pipeline_config)
        fp8_cfg.model_configs = dict(self.pipeline_config.model_configs)
        fp8_cfg.model_configs["transformer"] = fp8_mc

        logger.info(f"retake: building resident FP8 step transformer for steps {sorted(steps)}")
        fp8_tf = build_ltx2_transformer(fp8_cfg)
        raw = self.load_transformer_weights(self._checkpoint_dir)
        tw = raw.get("transformer", raw)
        lora_path = self.pipeline_config.extra_attrs.get("retake_lora_path")
        if lora_path:
            strength = float(self.pipeline_config.extra_attrs.get("retake_lora_strength", 1.0))
            tw = _fuse_lora_into_transformer_weights(tw, lora_path, strength)
        fp8_tf.load_weights(tw)
        if hasattr(fp8_tf, "post_load_weights"):
            fp8_tf.post_load_weights()
        fp8_tf.to(self._device)
        # The companion runs _masked_transformer_step, so attach the swap state there.
        self._native._fp8_step_transformer = fp8_tf
        self._native._fp8_step_indices = frozenset(int(s) for s in steps)

    def _setup_cuda_graphs(self) -> None:
        # Retake now builds the native LTXModel, whose ``forward`` takes
        # ``Modality`` dataclasses (not flat tensors). Delegate to the
        # generation pipeline's Modality-aware setup so cuda graph uses
        # ``_LTX2CUDAGraphRunner`` rather than the base ``CUDAGraphRunner`` and
        # is allowed to compose with torch.compile (the base implementation
        # disables cuda graph whenever torch_compile is enabled).
        LTX2Pipeline._setup_cuda_graphs(self)

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
        self._checkpoint_dir = checkpoint_dir
        if not text_encoder_path:
            raise ValueError(
                "LTX-2 retake workflow requires pipeline_config.text_encoder_path "
                "to point at the Gemma text encoder directory."
            )
        self._load_native_retake_components(checkpoint_dir, device, text_encoder_path)

    def _load_native_retake_components(
        self, checkpoint_dir: str, device: torch.device, text_encoder_path: str
    ) -> None:
        """Load the native pre/post companion and source-media readers.

        Builds a native ``LTX2Pipeline`` that shares this pipeline's native
        transformer (so no second transformer is constructed and no transformer
        weights are loaded twice) and loads its native video VAE encoder/decoder,
        patchifier, scheduler, Gemma text encoder, and connectors. Source video,
        metadata, and audio are decoded by the native PyAV readers.
        """
        from .ltx2_core.media_io import (
            decode_audio_from_file,
            decode_video_by_frame,
            get_videostream_metadata,
        )

        self._get_videostream_metadata = get_videostream_metadata
        self._decode_video_by_frame = decode_video_by_frame
        self._decode_audio_from_file = decode_audio_from_file

        logger.info(
            "Loading LTX-2 native retake pre/post components "
            f"(checkpoint={checkpoint_dir}, text_encoder={text_encoder_path})"
        )
        native = _NativeLTX2Companion(self.pipeline_config, self.transformer)
        # The native retake path is video-only: it uses the video VAE encoder/
        # decoder, the text encoder + video/audio embedding connectors, and the
        # scheduler, but never the audio VAE decoder or the vocoder. Skip loading
        # those so retake neither wastes memory on them nor depends on their
        # checkpoint-specific config parity.
        native.load_standard_components(
            checkpoint_dir,
            device,
            text_encoder_path=text_encoder_path,
            skip_components=["audio_vae", "vocoder"],
        )
        # Derived attribute used to build the video latent shape; normally set by
        # LTX2Pipeline.post_load_weights, which we skip to avoid re-running the
        # shared transformer's post-load hook.
        native.transformer_in_channels = native.transformer._transformer_config.get(
            "in_channels", 128
        )
        # Generation does not need the audio encoder, but retake uses it to
        # condition video denoising on the preserved source audio.
        self._load_retake_audio_encoder(native, checkpoint_dir, device)
        self._native = native

    def _load_retake_audio_encoder(self, native, checkpoint_dir: str, device) -> None:
        """Load the native audio VAE encoder for retake audio conditioning."""
        from .ltx2_core.audio_vae import AudioEncoderConfigurator
        from .pipeline_ltx2 import (
            AUDIO_ENCODER_WEIGHT_PREFIXES,
            _find_safetensors_files,
            _load_component_weights,
        )

        config = native._native_config
        sft_paths = _find_safetensors_files(checkpoint_dir)
        encoder = AudioEncoderConfigurator.from_config(config)
        _load_component_weights(sft_paths, encoder, AUDIO_ENCODER_WEIGHT_PREFIXES)
        native._audio_encoder = encoder.to(device=device, dtype=native.dtype)

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
        if self._native is None:
            raise RuntimeError("LTX-2 native retake pipeline has not been loaded.")
        # Fine CUDA-event stage timing: the native driver marks each retake stage
        # boundary so ``PipelineOutput`` carries both the three public phases and
        # a fine ``stage_timings`` breakdown.
        timer = RetakeStageTimer()
        video, audio, output_shape = self._run_native_pre_post_retake(
            req, extra, video_path, start_time, end_time, prompt, timer=timer
        )
        audio_tensor, sample_rate = self._normalize_audio(audio)
        return timer.fill(
            PipelineOutput(
                video=video,
                audio=audio_tensor,
                frame_rate=float(output_shape.fps),
                audio_sample_rate=sample_rate,
            )
        )

    # ------------------------------------------------------------------
    # Native retake pre/post runtime path
    # ------------------------------------------------------------------

    def _run_native_pre_post_retake(
        self, req, extra, video_path, start_time, end_time, prompt, timer=None
    ):
        """Run native source encode, masked denoise, and full-clip decode.

        Reuses the native ``LTX2Pipeline`` machinery (video VAE encoder/decoder,
        video patchifier, scheduler, Gemma text encoder + connectors, and the
        masked-denoise entry :meth:`LTX2Pipeline._masked_transformer_step`) with
        retake-specific inputs: initial latents seeded from the encoded source,
        a two-sided ``denoise_mask`` conditioning the leading + trailing context
        while regenerating the middle window, and a two-sided ``clean_latent``
        from the source. The entire decoded clip is returned, and the source
        audio is passed through unchanged (video-only regeneration).

        Returns ``(video_uint8, source_audio, output_shape)`` where
        ``video_uint8`` is ``(1, T, H, W, C)`` uint8.
        """
        native = self._native
        if native is None:
            raise RuntimeError(
                "LTX-2 native retake path requires the native companion pipeline, "
                "but self._native is None."
            )

        if req.params.num_inference_steps != 8:
            raise ValueError(
                "LTX-2 native retake uses the fixed distilled 8-step schedule; "
                f"got num_inference_steps={req.params.num_inference_steps}."
            )

        device = self._device
        dtype = self.dtype
        seed = req.params.seed
        generator = torch.Generator(device=device).manual_seed(seed)

        # ---- 1. Source read + validation --------------------------------
        # Stage-timing boundaries (no-op on CPU): source_read begins here.
        if timer is not None:
            timer.mark("source_read")
        output_shape = self._get_videostream_metadata(video_path)
        num_frames = int(output_shape.frames)
        height = int(output_shape.height)
        width = int(output_shape.width)
        fps = float(output_shape.fps)
        self._validate_retake_source(num_frames, height, width)

        source_norm_5d = self._read_source_video(
            video_path, num_frames, height, width, device, dtype
        )

        # ---- 2. Retake windows ------------------------------------------
        temporal_ratio = VIDEO_SCALE_FACTORS.time
        pixel_start, pixel_end = _retake_pixel_window(start_time, end_time, fps, num_frames)
        conditioned_latent_ranges = _retake_conditioned_latent_ranges(
            pixel_start, pixel_end, num_frames, temporal_ratio
        )

        # ---- 3. Native VAE encode + seed initial latents ----------------
        pixel_shape = VideoPixelShape(
            batch=1, frames=num_frames, height=height, width=width, fps=fps
        )
        video_shape = VideoLatentShape.from_pixel_shape(
            pixel_shape, latent_channels=native.transformer_in_channels
        )
        if timer is not None:
            timer.mark("vae_encode")
        # Keep the encoder result in the model dtype. The initial noise blend
        # below widens it to float32 without losing any BF16 values.
        source_window_latents = native._encode_video_window(source_norm_5d, _RETAKE_TILING_CONFIG)
        if timer is not None:
            timer.mark("conditioning")
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
        # Denoising carries the latent in the model dtype and requantizes it at
        # every Euler step; retaining the float32 blend would change the trajectory.
        latents = native.video_patchifier.patchify(initial_latents).to(dtype)

        # The conditioned x0 target follows the encoded source dtype, not the
        # float32 noise dtype.
        clean_5d = torch.zeros_like(source_window_latents)
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
        video_embeds, audio_embeds, connector_mask = native._process_connectors(
            prompt_embeds, prompt_attention_mask
        )

        video_positions = native.video_patchifier.get_patch_grid_bounds(video_shape, device=device)
        video_positions = get_pixel_coords(
            video_positions.float(), VIDEO_SCALE_FACTORS, causal_fix=True
        )
        video_positions[:, 0, ...] = video_positions[:, 0, ...] / fps
        # RoPE positions remain float32 because position error is amplified by
        # the distilled schedule's large final Euler steps.

        # ---- Frozen audio conditioning ----------------------------------
        # Encode the source audio as a frozen conditioning modality. Video-only
        # retake never regenerates the audio.
        audio_latent = self._encode_source_audio_latent(
            native._audio_encoder, video_path, pixel_shape, device, dtype
        )
        # The frozen audio follows a noise-to-clean trajectory on the same sigma
        # schedule as the video rather than remaining constant.
        if audio_latent is not None:
            audio_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
            audio_clean_latents = native.audio_patchifier.patchify(audio_latent.float())
            audio_positions = native.audio_patchifier.get_patch_grid_bounds(
                audio_shape, device=device
            )
        else:
            audio_clean_latents = None
            audio_embeds = None
            audio_positions = None

        text_cache = native.transformer.prepare_text_cache(
            video_context=video_embeds,
            video_context_mask=connector_mask,
            video_positions=video_positions,
            audio_context=audio_embeds,
            audio_context_mask=connector_mask if audio_embeds is not None else None,
            audio_positions=audio_positions,
            dtype=dtype,
        )

        # Prepare the resident FP8 transformer's own text
        # cache (prepare_text_cache runs the model's quantized cross-attention
        # context) while both transformers are resident, then offload the Gemma
        # text encoder to CPU. Gemma is only used for the prompt encode above; the
        # single-process retake frees its ~24GB so the NVFP4 + FP8 transformers
        # fit alongside the denoise activations.
        fp8_step_tf = getattr(native, "_fp8_step_transformer", None)
        if fp8_step_tf is not None:
            native._fp8_step_text_cache = fp8_step_tf.prepare_text_cache(
                video_context=video_embeds,
                video_context_mask=connector_mask,
                video_positions=video_positions,
                audio_context=audio_embeds,
                audio_context_mask=connector_mask if audio_embeds is not None else None,
                audio_positions=audio_positions,
                dtype=dtype,
            )
            if getattr(native, "text_encoder", None) is not None:
                native.text_encoder.to("cpu")
                torch.cuda.empty_cache()

        # ---- 5. Native masked denoise (distilled, video-only, non-guided) --
        scheduler = NativeSchedulerAdapter()
        scheduler.sigmas = self._retake_distilled_sigmas(device)
        scheduler._step_index = 0
        timesteps = scheduler.timesteps
        num_steps = len(timesteps)

        # Euler-integrate frozen audio from noise to its clean x0 target on the
        # video sigma schedule. This keeps cross-attention noise-matched without
        # running a separate transformer prediction for audio.
        audio_traj = None
        if audio_clean_latents is not None:
            _sigs = scheduler.sigmas.float()
            # Requantize the audio state to the model dtype after every Euler step.
            a_clean = audio_clean_latents.to(dtype)
            _an = torch.randn(
                a_clean.shape, generator=generator, device=device, dtype=torch.float32
            )
            a_cur = (_an * _sigs[0] + a_clean.float() * (1.0 - _sigs[0])).to(dtype)
            _traj = [a_cur]
            for _i in range(num_steps):
                # Python scalars preserve the established Euler arithmetic; a
                # zero-dimensional tensor takes a measurably different FP32 path.
                _s = float(_sigs[_i])
                _sn = float(_sigs[_i + 1])
                _vel = (a_cur.float() - a_clean.float()) / _s
                a_cur = (a_cur.float() + _vel * (_sn - _s)).to(dtype)
                _traj.append(a_cur)
            audio_traj = _traj

        def retake_forward_fn(
            video_latents,
            _extra_stream_latents,
            step_index,
            timestep,
            _encoder_hidden_states,
            _extra_tensors,
        ):
            a_lat = audio_traj[step_index] if audio_traj is not None else None
            denoised_video, _ = native._masked_transformer_step(
                video_latents,
                a_lat,
                step_index,
                timestep,
                video_embeds,
                audio_embeds,
                connector_mask,
                video_positions=video_positions,
                audio_positions=audio_positions,
                denoise_mask=denoise_mask,
                clean_latent=clean_latent,
                num_steps=num_steps,
                text_cache=text_cache,
                a_frozen=True,
            )
            return denoised_video, {}

        def retake_post_step_fn(step_latents):
            # Per-step denoise boundary: recorded AFTER the scheduler step
            # completes (``denoise`` applies ``post_step_fn`` once per timestep,
            # after ``_scheduler_step``), so each interval is exactly one
            # completed denoise-loop iteration. Passthrough (no latent change).
            if timer is not None:
                timer.mark_step()
            return step_latents

        if timer is not None:
            timer.mark("denoise")
        denoised_latents = native.denoise(
            latents=latents,
            scheduler=scheduler,
            prompt_embeds=video_embeds,
            guidance_scale=1.0,
            forward_fn=retake_forward_fn,
            timesteps=timesteps,
            post_step_fn=retake_post_step_fn,
        )

        # ---- 6. Native decode -------------------------------------------
        # Stage-timing boundary: decode (VAE-decode) begins here.
        if timer is not None:
            timer.mark("decode")
        video_latents_5d = native.video_patchifier.unpatchify(denoised_latents, video_shape).to(
            dtype
        )
        # Encode and decode must share the retake tiling geometry.
        chunks = list(
            native.video_decoder.tiled_decode(
                video_latents_5d, _RETAKE_TILING_CONFIG, generator=generator
            )
        )
        decoded = torch.cat(chunks, dim=2)  # (B, C, T, H, W)
        if timer is not None:
            timer.mark("postprocess")
        decoded = postprocess_video_tensor(decoded)  # (B, T, H, W, C) uint8
        output_video = decoded.contiguous()
        if timer is not None:
            timer.mark("end")

        # Source-audio read is host-side I/O and stays out of CUDA stage timing.
        source_audio = self._read_source_audio(video_path, device)

        return output_video, source_audio, output_shape

    @staticmethod
    def _validate_retake_source(num_frames: int, height: int, width: int) -> None:
        """Fail fast on source video shapes the native VAE cannot round-trip.

        The LTX-2 causal video VAE requires ``8k + 1`` pixel frames and spatial
        dimensions that are multiples of 32 so encode/decode preserves shape.
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
        """Read and validate the source video as a normalized VAE input.

        Returns ``(1, 3, T, H, W)`` in ``[-1, 1]``. Pixel decoding is
        deterministic and uses sequential frame indices.
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
                f"retake source decoded {decoded_frames} frames but metadata reported {num_frames}."
            )
        if source_uint8.shape[2] != height or source_uint8.shape[3] != width:
            raise ValueError(
                f"retake source frame size {tuple(source_uint8.shape[2:4])} does not "
                f"match metadata {(height, width)}."
            )
        # uint8 [0, 255] -> [-1, 1], laid out (1, C, T, H, W) for the VAE encoder.
        normalized = source_uint8[0].to(torch.float32) / 127.5 - 1.0  # (T, H, W, C)
        source_norm_5d = normalized.permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=dtype)
        return source_norm_5d

    def _read_source_audio(self, video_path, device):
        """Read the source audio unchanged (video-only retake preserves audio).

        Uses the native ``ltx2_core.media_io.decode_audio_from_file`` reader,
        which decodes via PyAV (no torchaudio dependency — torchaudio is only
        needed by the audio VAE mel front-end). Returns an
        :class:`~.ltx2_core.types.Audio` (``.waveform`` / ``.sampling_rate``) or
        ``None`` when the source has no audio stream.
        """
        return self._decode_audio_from_file(video_path, device)

    def _encode_source_audio_latent(self, audio_encoder, video_path, pixel_shape, device, dtype):
        """Encode the source audio into the frozen conditioning latent, natively.

        Reads the audio stream for exactly the video's duration, runs the native
        audio VAE encoder over its mel spectrogram, and conforms the latent length
        to what the video pixel shape implies. Returns ``None`` when the source
        carries no audio stream.

        The duration is derived from the *video* shape (``frames / fps``) rather
        than the audio stream's own length, because the audio latent has to line
        up token-for-token with the video latent the transformer cross-attends
        to.
        """
        from .ltx2_core.audio_vae import encode_audio

        max_duration = pixel_shape.frames / pixel_shape.fps
        audio_in = self._decode_audio_from_file(video_path, device, 0.0, max_duration)
        if audio_in is None:
            return None
        latents = encode_audio(audio_in, audio_encoder).to(device, dtype)
        required_latent_frames = AudioLatentShape.from_video_pixel_shape(pixel_shape).frames
        return _conform_latent_length(latents, required_latent_frames)

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
