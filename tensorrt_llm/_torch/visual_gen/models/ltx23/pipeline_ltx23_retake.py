# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native LTX-2.3 retake workflow for VisualGen.

This pipeline keeps a resident retake model for ``trtllm-serve`` and extends the
native LTX-2.3 pipeline with source-media encoders and masked denoising.
It either loads Gemma for a live prompt or consumes precomputed post-connector
conditioning. Retake generalizes image-to-video masking to condition both sides
of the regenerated window.

Source video/audio decode, stream metadata, audio/video VAE encode, diffusion,
and VAE decode all use TensorRT-LLM native components.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import safetensors.torch
import torch
import torch.distributed as dist

from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunnerConfig, SharedGraphPool
from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm._torch.visual_gen.triton_kernels import uint8_bthwc_to_normalized_ncthw
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm.inputs.multimodal_data import AudioData
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from ..ltx2.ltx2_core.patchifier import get_pixel_coords
from ..ltx2.ltx2_core.scheduler_adapter import NativeSchedulerAdapter
from ..ltx2.ltx2_core.types import (
    VIDEO_SCALE_FACTORS,
    AudioLatentShape,
    VideoLatentShape,
    VideoPixelShape,
)
from ..ltx2.ltx2_core.upsampler import LatentUpsamplerConfigurator, upsample_video
from ..ltx2.ltx2_core.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ..ltx2.parallel_vae import tile_parallel_decode
from ..ltx2.pipeline_ltx2 import _load_component_weights
from .ltx23_core.audio_vae import AudioEncoderConfigurator, encode_audio
from .ltx23_core.modality import LTX23Modality
from .ltx23_core.video_vae_ltx23 import LTX23VideoEncoderConfigurator
from .ltx23_core.video_vae_ltx23_nvfp4 import (
    LTX23Nvfp4RetakeRecipe,
    LTX23Nvfp4VideoDecoderConfigurator,
    LTX23Nvfp4VideoEncoderConfigurator,
    load_ltx23_nvfp4_retake_recipe,
)
from .media_io import (
    decode_audio_from_file,
    decode_video_by_frame,
    get_videostream_metadata,
    pad_audio_to_video_duration,
)
from .parallel_vae import tile_parallel_encode
from .pipeline_ltx23 import LTX23Pipeline, _build_ltx23_transformer, _LTX23CUDAGraphRunner
from .text_conditioning_ltx23 import LTX23TextConditioning
from .transformer_ltx23 import LTX23Model

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
    from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest

# Distilled single-stage Retake noise schedule (8 Euler steps).
_RETAKE_DISTILLED_SIGMA_VALUES = (
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
)
_RETAKE_NUM_INFERENCE_STEPS = len(_RETAKE_DISTILLED_SIGMA_VALUES) - 1


_RETAKE_NVFP4_ATTENTION_RECIPE = (
    Path(__file__).with_name("ltx23_core") / "attention_ltx23_retake_nvfp4_global_fp32_scales.json"
)


def _load_retake_nvfp4_qk_global_scale(path: str | None = None) -> float:
    recipe_path = Path(path) if path is not None else _RETAKE_NVFP4_ATTENTION_RECIPE
    payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    qk = payload["attention"]["qk"]
    observed_amax = float(qk["observed_amax"])
    global_scale = float(qk["global_fp32_scale"])
    quant_max = float(payload["quantization"]["global_quant_max"])
    if observed_amax <= 0.0 or not math.isclose(
        global_scale, quant_max / observed_amax, rel_tol=1e-6
    ):
        raise ValueError(f"Invalid NVFP4 attention calibration recipe: {recipe_path}")
    return global_scale


# Two-stage Retake first renders a half-resolution draft and then performs a
# short full-resolution refinement.
_TWO_STAGE_STAGE_1_SIGMA_VALUES = (
    1.0,
    0.965712,
    0.921875,
    0.863856,
    0.783445,
    0.664579,
    0.471003,
    0.1,
    0.0,
)
_TWO_STAGE_STAGE_1_NUM_INFERENCE_STEPS = len(_TWO_STAGE_STAGE_1_SIGMA_VALUES) - 1
_TWO_STAGE_STAGE_2_SIGMA_VALUES = (0.909375, 0.725, 0.421875, 0.0)
_TWO_STAGE_STAGE_2_NUM_INFERENCE_STEPS = len(_TWO_STAGE_STAGE_2_SIGMA_VALUES) - 1

_DEFAULT_MAX_SEQUENCE_LENGTH = 1024
_PROMPT_CONDITIONING_KEYS = ("video_embeds", "audio_embeds", "connector_mask")

# Retake uses the same tiling geometry for source encode and output decode. Keep
# it separate from the generation pipeline's default because tile boundaries are
# part of the retake numerical contract.
_RETAKE_TILING_CONFIG = TilingConfig(
    spatial_config=SpatialTilingConfig(tile_size_in_pixels=768, tile_overlap_in_pixels=64),
    temporal_config=TemporalTilingConfig(tile_size_in_frames=80, tile_overlap_in_frames=24),
)

_TWO_STAGE_RETAKE_TILING_CONFIG = TilingConfig(
    spatial_config=SpatialTilingConfig(tile_size_in_pixels=384, tile_overlap_in_pixels=64),
    temporal_config=TemporalTilingConfig(tile_size_in_frames=1_000_000, tile_overlap_in_frames=0),
)


@dataclass(frozen=True)
class _RetakeStageConditioning:
    video_embeds: torch.Tensor
    audio_embeds: torch.Tensor | None
    connector_mask: torch.Tensor
    audio_clean_latents: torch.Tensor | None
    audio_positions: torch.Tensor | None


@dataclass(frozen=True)
class _RetakeStage:
    initial_latents: torch.Tensor
    video_positions: torch.Tensor
    denoise_mask: torch.Tensor
    sigma_values: tuple[float, ...]
    companion_transformer: LTX23Model | None
    companion_steps: frozenset[int]


def _load_prompt_conditioning(path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load precomputed post-connector prompt tensors from safetensors."""
    tensors = safetensors.torch.load_file(path, device="cpu")
    keys = set(tensors)
    required = set(_PROMPT_CONDITIONING_KEYS)
    if keys != required:
        raise ValueError(
            "Prompt-conditioning tensor keys do not match the expected schema: "
            f"missing={sorted(required - keys)}, unexpected={sorted(keys - required)}"
        )

    video_embeds, audio_embeds, connector_mask = (tensors[key] for key in _PROMPT_CONDITIONING_KEYS)
    if video_embeds.ndim != 3 or audio_embeds.ndim != 3:
        raise ValueError(
            "Prompt embeddings must be rank 3 [batch, sequence, channels]; "
            f"got video={tuple(video_embeds.shape)}, audio={tuple(audio_embeds.shape)}"
        )
    if connector_mask.ndim != 4:
        raise ValueError(
            "Connector mask must be rank 4 [batch, 1, 1, sequence]; "
            f"got {tuple(connector_mask.shape)}"
        )
    if video_embeds.shape[0] != 1:
        raise ValueError(
            "LTX-2.3 retake prompt conditioning must contain one prompt; "
            f"got batch size {video_embeds.shape[0]}."
        )
    if video_embeds.shape[:2] != audio_embeds.shape[:2] or (
        connector_mask.shape[0] != video_embeds.shape[0]
        or connector_mask.shape[-1] != video_embeds.shape[1]
    ):
        raise ValueError(
            "Prompt-conditioning batch and sequence dimensions do not align: "
            f"video={tuple(video_embeds.shape)}, audio={tuple(audio_embeds.shape)}, "
            f"mask={tuple(connector_mask.shape)}"
        )
    return video_embeds, audio_embeds, connector_mask


def _normalize_fp8_step_indices(
    steps: Any,
    *,
    num_steps: int = _RETAKE_NUM_INFERENCE_STEPS,
    setting_name: str = "retake_fp8_linear_steps",
) -> frozenset[int]:
    """Validate FP8 companion-transformer steps against a fixed schedule."""
    if steps is None:
        return frozenset()
    if isinstance(steps, (str, bytes)):
        raise TypeError(f"{setting_name} must be a sequence of integer step indices.")
    try:
        values = list(steps)
    except TypeError as exc:
        raise TypeError(f"{setting_name} must be a sequence of integer step indices.") from exc
    invalid_types = [step for step in values if isinstance(step, bool) or not isinstance(step, int)]
    if invalid_types:
        raise TypeError(f"{setting_name} entries must be integers; got {invalid_types!r}.")
    invalid_range = [step for step in values if not 0 <= step < num_steps]
    if invalid_range:
        raise ValueError(
            f"{setting_name} entries must be in [0, {num_steps}); got {invalid_range!r}."
        )
    return frozenset(values)


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


def _resolve_retake_time_window(
    start_time: float | None,
    end_time: float | None,
    fps: float,
    num_frames: int,
) -> tuple[float, float]:
    """Resolve an optional retake window against the source video duration."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if num_frames < 0:
        raise ValueError(f"num_frames must be non-negative, got {num_frames}")

    resolved_start = 0.0 if start_time is None else float(start_time)
    resolved_end = num_frames / fps if end_time is None else float(end_time)
    if not math.isfinite(resolved_start) or not math.isfinite(resolved_end):
        raise ValueError("retake_start_time and retake_end_time must be finite.")
    if resolved_start >= resolved_end:
        raise ValueError(
            f"retake_start_time ({resolved_start}) must be less than "
            f"retake_end_time ({resolved_end})"
        )
    return resolved_start, resolved_end


def _pixel_frame_to_latent_index(pixel_frame: int, temporal_ratio: int) -> int:
    """Latent-frame index of a source pixel frame under the causal LTX-2.3 VAE.

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


def _retake_soft_margin_latent_ranges(
    pixel_start: int,
    pixel_end: int,
    core_pixel_start: int,
    core_pixel_end: int,
    num_frames: int,
    temporal_ratio: int,
    soft_value: float,
) -> list[tuple[int, int, float]]:
    """Return soft-denoise latent ranges inside the edit window around its core."""
    if temporal_ratio <= 0:
        raise ValueError(f"temporal_ratio must be positive, got {temporal_ratio}")
    if not math.isfinite(soft_value) or not 0.0 <= soft_value <= 1.0:
        raise ValueError(f"retake_soft_margin_noise must be in [0, 1], got {soft_value}")
    if pixel_end <= pixel_start:
        return []
    if core_pixel_start < pixel_start or core_pixel_end > pixel_end:
        raise ValueError(
            "The semantic edit core must be contained in the outer Retake window: "
            f"core=[{core_pixel_start}, {core_pixel_end}), "
            f"window=[{pixel_start}, {pixel_end})."
        )

    total_latent = _latent_frame_count(num_frames, temporal_ratio)
    window_start = max(
        0,
        min(_pixel_frame_to_latent_index(pixel_start, temporal_ratio), total_latent),
    )
    window_end = max(
        window_start,
        min(
            _pixel_frame_to_latent_index(pixel_end - 1, temporal_ratio) + 1,
            total_latent,
        ),
    )
    core_start = max(
        window_start,
        min(_pixel_frame_to_latent_index(core_pixel_start, temporal_ratio), window_end),
    )
    core_end = max(
        core_start,
        min(
            _pixel_frame_to_latent_index(core_pixel_end - 1, temporal_ratio) + 1,
            window_end,
        ),
    )

    ranges = []
    if core_start > window_start:
        ranges.append((window_start, core_start, soft_value))
    if window_end > core_end:
        ranges.append((core_end, window_end, soft_value))
    return ranges


def _resolve_soft_margin_latent_ranges(
    extra: dict[str, Any],
    *,
    pixel_start: int,
    pixel_end: int,
    fps: float,
    num_frames: int,
    temporal_ratio: int,
) -> list[tuple[int, int, float]]:
    """Resolve optional request-level core/soft-margin settings."""
    core_start_time = extra.get("retake_core_start_time")
    core_end_time = extra.get("retake_core_end_time")
    if (core_start_time is None) != (core_end_time is None):
        raise ValueError(
            "retake_core_start_time and retake_core_end_time must be provided together."
        )

    raw_soft_value = extra.get("retake_soft_margin_noise", 1.0)
    soft_value = 1.0 if raw_soft_value is None else float(raw_soft_value)
    if not math.isfinite(soft_value) or not 0.0 <= soft_value <= 1.0:
        raise ValueError(f"retake_soft_margin_noise must be in [0, 1], got {soft_value}")
    if core_start_time is None or soft_value == 1.0:
        return []

    resolved_core_start, resolved_core_end = _resolve_retake_time_window(
        float(core_start_time),
        float(core_end_time),
        fps,
        num_frames,
    )
    core_pixel_start, core_pixel_end = _retake_pixel_window(
        resolved_core_start,
        resolved_core_end,
        fps,
        num_frames,
    )
    return _retake_soft_margin_latent_ranges(
        pixel_start,
        pixel_end,
        core_pixel_start,
        core_pixel_end,
        num_frames,
        temporal_ratio,
        soft_value,
    )


def _init_retake_patchified_latents(
    noise_latents: torch.Tensor,
    source_latents: torch.Tensor,
    denoise_mask: torch.Tensor,
) -> torch.Tensor:
    """Initialize retake latents in the denoiser's patchified token layout.

    Sampling in 5D and then patchifying produces a different seeded random field
    because the memory order changes, so noise is sampled directly in token
    layout.
    """
    if noise_latents.shape != source_latents.shape:
        raise ValueError(
            f"noise/source latent shape mismatch: {tuple(noise_latents.shape)} vs "
            f"{tuple(source_latents.shape)}"
        )
    if denoise_mask.dim() == 2:
        mask = denoise_mask.unsqueeze(-1)
    elif denoise_mask.dim() == 3:
        mask = denoise_mask
    else:
        raise ValueError(f"expected denoise mask rank 2 or 3, got {tuple(denoise_mask.shape)}")
    if mask.shape[:2] != source_latents.shape[:2] or mask.shape[-1] != 1:
        raise ValueError(
            f"denoise mask shape {tuple(denoise_mask.shape)} is not compatible with "
            f"patchified latents {tuple(source_latents.shape)}"
        )
    return torch.lerp(source_latents.float(), noise_latents.float(), mask.float()).to(
        source_latents.dtype
    )


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


# Retake shares checkpoints with LTX23Pipeline and is selected by ``workflow``.
@register_pipeline("LTX23RetakePipeline", doc="Native LTX-2.3 video retake pipeline.")
class LTX23RetakePipeline(LTX23Pipeline):
    """Persistent VisualGen pipeline for native LTX-2.3 retake requests."""

    def __init__(self, pipeline_config: "DiffusionPipelineConfig") -> None:
        if pipeline_config.cache_backend is not None:
            raise NotImplementedError("Cache acceleration is not supported for LTX-2.3 retake.")
        if pipeline_config.parallel.cfg_size != 1:
            raise NotImplementedError("LTX-2.3 retake requires parallel.cfg_size=1.")
        if pipeline_config.parallel.tp_size != 1:
            raise NotImplementedError("Tensor parallelism is not supported for LTX-2.3 retake.")
        if pipeline_config.parallel.seq_parallel_size != pipeline_config.parallel.ulysses_size:
            raise NotImplementedError(
                "LTX-2.3 retake currently supports Ulysses sequence parallelism only."
            )
        if pipeline_config.parallel.async_ulysses:
            raise NotImplementedError("Async Ulysses is not supported for LTX-2.3 retake.")
        self._fp8_step_transformer: LTX23Model | None = None
        self._fp8_step_indices = _normalize_fp8_step_indices(
            pipeline_config.extra_attrs.get("retake_fp8_linear_steps")
        )
        self._checkpoint_dir: str | None = None
        self._retake_vae_nvfp4_recipe: LTX23Nvfp4RetakeRecipe | None = None
        configured_vae_nvfp4 = pipeline_config.extra_attrs.get("retake_vae_nvfp4", True)
        if not isinstance(configured_vae_nvfp4, bool):
            raise TypeError("retake_vae_nvfp4 must be a boolean in pipeline_config.")
        configured_encoder_nvfp4 = pipeline_config.extra_attrs.get("retake_vae_encoder_nvfp4")
        configured_decoder_nvfp4 = pipeline_config.extra_attrs.get("retake_vae_decoder_nvfp4")
        for name, value in (
            ("retake_vae_encoder_nvfp4", configured_encoder_nvfp4),
            ("retake_vae_decoder_nvfp4", configured_decoder_nvfp4),
        ):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean in pipeline_config.")
        self._retake_vae_encoder_nvfp4 = (
            configured_vae_nvfp4 if configured_encoder_nvfp4 is None else configured_encoder_nvfp4
        )
        self._retake_vae_decoder_nvfp4 = (
            configured_vae_nvfp4 if configured_decoder_nvfp4 is None else configured_decoder_nvfp4
        )
        excluded_encoder_sites = pipeline_config.extra_attrs.get("retake_vae_encoder_nvfp4_exclude")
        if excluded_encoder_sites is None:
            excluded_encoder_sites = ()
        if not isinstance(excluded_encoder_sites, (list, tuple)) or not all(
            isinstance(site, str) for site in excluded_encoder_sites
        ):
            raise TypeError("retake_vae_encoder_nvfp4_exclude must be a list of Conv3d site names.")
        self._retake_vae_encoder_nvfp4_exclude = tuple(excluded_encoder_sites)
        attention_config = pipeline_config.attention
        quant_attention_config = attention_config.quant_attention_config
        if (
            attention_config.backend == "CUTEDSL"
            and quant_attention_config is not None
            and quant_attention_config.qk_dtype == "nvfp4"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] == 10
        ):
            recipe_path = pipeline_config.extra_attrs.get("retake_attention_nvfp4_recipe_path")
            global_scale = _load_retake_nvfp4_qk_global_scale(recipe_path)
            quant_attention_config.qk_global_scale = global_scale
            model_attention_config = pipeline_config.model_configs["transformer"].attention
            model_quant_attention_config = model_attention_config.quant_attention_config
            if model_quant_attention_config is None:
                raise ValueError(
                    "The SM100 Retake transformer is missing its NVFP4 attention config."
                )
            model_quant_attention_config.qk_global_scale = global_scale
            logger.info("LTX-2.3 Retake: loaded offline-calibrated NVFP4 Q/K scale.")
        configured_two_stage = pipeline_config.extra_attrs.get("retake_two_stage", False)
        if not isinstance(configured_two_stage, bool):
            raise TypeError("retake_two_stage must be a boolean in pipeline_config.")
        self._two_stage = configured_two_stage
        self._spatial_upsampler: torch.nn.Module | None = None
        self._spatial_upsampler_path: str | None = None
        self._stage2_step_transformer: LTX23Model | None = None
        stage2_quant = pipeline_config.extra_attrs.get("retake_stage2_quant")
        stage2_steps = pipeline_config.extra_attrs.get("retake_stage2_fp8_linear_steps")
        if stage2_steps is not None and stage2_quant is None:
            stage2_quant = "FP8"
        self._stage2_quant = str(stage2_quant).upper() if stage2_quant else None
        if self._stage2_quant not in {None, "BF16", "FP8"}:
            raise ValueError(
                f"retake_stage2_quant must be either 'BF16' or 'FP8'; got {stage2_quant!r}."
            )
        if self._stage2_quant is not None and stage2_steps is None:
            stage2_steps = range(_TWO_STAGE_STAGE_2_NUM_INFERENCE_STEPS)
        self._stage2_step_indices = _normalize_fp8_step_indices(
            stage2_steps,
            num_steps=_TWO_STAGE_STAGE_2_NUM_INFERENCE_STEPS,
            setting_name="retake_stage2_fp8_linear_steps",
        )
        super().__init__(pipeline_config)

    def setup_parallel_vae(self) -> None:
        """Enable tile-parallel Retake VAE encode/decode on every model rank.

        Retake needs the encoded source latent on every rank before denoising and
        currently returns a decoded video from every rank. The initial supported
        configuration therefore uses the complete model world as the VAE group.
        """
        parallel_cfg = self.pipeline_config.parallel
        vgm = self.pipeline_config.visual_gen_mapping
        requested = parallel_cfg.parallel_vae_size
        self._parallel_vae_enabled = requested > 1
        if not self._parallel_vae_enabled:
            return
        if not dist.is_initialized() or vgm is None or vgm.vae_group is None:
            raise RuntimeError("LTX-2.3 Retake parallel VAE requires an initialized VAE group.")
        world_size = dist.get_world_size()
        if requested != world_size or vgm.vae_ranks != list(range(world_size)):
            raise NotImplementedError(
                "LTX-2.3 Retake initially requires parallel_vae_size == world_size; "
                f"got parallel_vae_size={requested}, world_size={world_size}, "
                f"vae_ranks={vgm.vae_ranks}."
            )
        logger.info(
            "LTX-2.3 Retake tile-parallel VAE enabled on all ranks: parallel_vae_size=%d",
            requested,
        )

    def _encode_video_latents(
        self, video: torch.Tensor, tiling_config: TilingConfig
    ) -> torch.Tensor:
        if self.video_encoder is None:
            raise RuntimeError("LTX-2.3 Retake requires video VAE encoder weights.")
        if not self._parallel_vae_enabled:
            return self.video_encoder.tiled_encode(video, tiling_config)
        vgm = self.pipeline_config.visual_gen_mapping
        return tile_parallel_encode(self.video_encoder, video, tiling_config, pg=vgm.vae_group)

    def _decode_video_latents(
        self,
        latents: torch.Tensor,
        tiling_config: TilingConfig,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if not self._parallel_vae_enabled:
            chunks = list(
                self.video_decoder.tiled_decode(latents, tiling_config, generator=generator)
            )
            return torch.cat(chunks, dim=2)
        vgm = self.pipeline_config.visual_gen_mapping
        return tile_parallel_decode(
            self.video_decoder,
            latents,
            tiling_config,
            pg=vgm.vae_group,
            generator=generator,
        )

    def _wrap_retake_cuda_graph(self, name: str, transformer: LTX23Model) -> None:
        graph_transformers = getattr(self, "_retake_graph_transformers", set())
        if id(transformer) in graph_transformers:
            return
        graph_pool = getattr(self, "_retake_graph_pool", None)
        if graph_pool is None:
            graph_pool = SharedGraphPool()
            self._retake_graph_pool = graph_pool
        runner = _LTX23CUDAGraphRunner(
            CUDAGraphRunnerConfig(use_cuda_graph=True),
            graph_pool,
        )
        transformer.register_cuda_graph_extra_key_fns(runner)
        transformer.forward = runner.wrap(transformer.forward)
        self._cuda_graph_runners[name] = runner
        graph_transformers.add(id(transformer))
        self._retake_graph_transformers = graph_transformers
        logger.info(f"CUDA graph runner: wrapping LTX-2.3 Retake {name}.forward")

    def _setup_cuda_graphs(self) -> None:
        """Create one shared-pool graph runner for each resident transformer."""
        if not self.pipeline_config.cuda_graph.enable:
            return
        self._wrap_retake_cuda_graph("transformer", self.transformer)

    def _setup_companion_cuda_graphs(self) -> None:
        if not self.pipeline_config.cuda_graph.enable:
            return
        for name, transformer in (
            ("retake_fp8_transformer", self._fp8_step_transformer),
            ("retake_stage2_transformer", self._stage2_step_transformer),
        ):
            if transformer is not None:
                self._wrap_retake_cuda_graph(name, transformer)

    def _supports_nvfp4_video_vae(self) -> bool:
        return (
            self.pipeline_config.quant_config.quant_algo == QuantAlgo.NVFP4
            and torch.cuda.get_device_capability()[0] >= 12
        )

    def _uses_nvfp4_video_encoder(self) -> bool:
        return self._retake_vae_encoder_nvfp4 and self._supports_nvfp4_video_vae()

    def _uses_nvfp4_video_decoder(self) -> bool:
        return self._retake_vae_decoder_nvfp4 and self._supports_nvfp4_video_vae()

    def _nvfp4_video_vae_recipe(self) -> LTX23Nvfp4RetakeRecipe:
        if self._retake_vae_nvfp4_recipe is None:
            path = self.pipeline_config.extra_attrs.get("retake_vae_nvfp4_recipe_path")
            self._retake_vae_nvfp4_recipe = load_ltx23_nvfp4_retake_recipe(path)
        return self._retake_vae_nvfp4_recipe

    def _create_video_decoder(self, config: dict[str, Any]):
        if not self._uses_nvfp4_video_decoder():
            return super()._create_video_decoder(config)
        recipe = self._nvfp4_video_vae_recipe()
        logger.info("Loading LTX-2.3 Retake video decoder with static NVFP4 VAE scales.")
        return LTX23Nvfp4VideoDecoderConfigurator.from_config(
            config, activation_scales=recipe.decoder
        )

    @property
    def default_generation_params(self) -> dict[str, int]:
        return {
            "num_inference_steps": _RETAKE_NUM_INFERENCE_STEPS,
            "seed": 42,
            "max_sequence_length": _DEFAULT_MAX_SEQUENCE_LENGTH,
        }

    @property
    def extra_param_specs(self) -> dict[str, ExtraParamSchema]:
        return {
            "retake_video_path": ExtraParamSchema(
                type="str",
                description="Path to the source video file for retake.",
            ),
            "retake_prompt_conditioning_path": ExtraParamSchema(
                type="str",
                default=None,
                description="Path to precomputed post-connector prompt conditioning.",
            ),
            "retake_start_time": ExtraParamSchema(
                type="float",
                default=0.0,
                description=(
                    "Start time in seconds for the regenerated window. "
                    "Defaults to the beginning of the video."
                ),
            ),
            "retake_end_time": ExtraParamSchema(
                type="float",
                default=None,
                description=(
                    "End time in seconds for the regenerated window. "
                    "Defaults to the end of the video."
                ),
            ),
            "retake_core_start_time": ExtraParamSchema(
                type="float",
                default=None,
                description=(
                    "Start time in seconds of the semantic edit core. Tokens between "
                    "the outer retake window and this core use the soft-margin value."
                ),
            ),
            "retake_core_end_time": ExtraParamSchema(
                type="float",
                default=None,
                description="End time in seconds of the semantic edit core.",
            ),
            "retake_soft_margin_noise": ExtraParamSchema(
                type="float",
                default=1.0,
                description=(
                    "Denoise-mask value for tokens between the outer window and edit "
                    "core; 1.0 fully regenerates them and 0.0 preserves them."
                ),
            ),
            "retake_two_stage": ExtraParamSchema(
                type="bool",
                default=None,
                description="Override the configured two-resolution retake mode.",
            ),
            "retake_spatial_upsampler_path": ExtraParamSchema(
                type="str",
                default=None,
                description="Path to the learned 2x latent upsampler for two-stage retake.",
            ),
        }

    def _load_native_components(
        self,
        config: dict[str, Any],
        safetensors_paths: list[str],
        device: torch.device,
        dtype: torch.dtype,
        skip_components: list | None = None,
    ) -> None:
        """Extend the LTX-2.3 components with source-media encoders."""
        skip_components = skip_components or []
        super()._load_native_components(
            config,
            safetensors_paths,
            device,
            dtype,
            skip_components,
        )

        if "video_encoder" not in skip_components:
            encoder_blocks = config.get("vae", {}).get("encoder_blocks", [])
            if not encoder_blocks:
                raise ValueError("LTX-2.3 checkpoint config has no video VAE encoder blocks.")
            if self._uses_nvfp4_video_encoder():
                recipe = self._nvfp4_video_vae_recipe()
                logger.info("Loading LTX-2.3 Retake video encoder with static NVFP4 VAE scales.")
                self.video_encoder = LTX23Nvfp4VideoEncoderConfigurator.from_config(
                    config,
                    activation_scales=recipe.encoder,
                    excluded_sites=self._retake_vae_encoder_nvfp4_exclude,
                )
            else:
                self.video_encoder = LTX23VideoEncoderConfigurator.from_config(config)
            _load_component_weights(safetensors_paths, self.video_encoder, "vae.encoder.")
            _load_component_weights(
                safetensors_paths,
                self.video_encoder.per_channel_statistics,
                "vae.per_channel_statistics.",
            )
            self.video_encoder = self.video_encoder.to(device=device, dtype=dtype)
        else:
            self.video_encoder = None

        self._audio_encoder = AudioEncoderConfigurator.from_config(config)
        _load_component_weights(safetensors_paths, self._audio_encoder, "audio_vae.encoder.")
        _load_component_weights(
            safetensors_paths,
            self._audio_encoder.per_channel_statistics,
            "audio_vae.per_channel_statistics.",
        )
        self._audio_encoder = self._audio_encoder.to(device=device, dtype=dtype)
        self.audio_patchifier = self._audio_encoder.patchifier

    def _build_denoise_mask(
        self,
        video_shape: VideoLatentShape,
        *,
        cond_latent_frame_ranges: list[tuple[int, int]],
        soft_latent_frame_ranges: list[tuple[int, int, float]] | None = None,
    ) -> torch.Tensor:
        patch_t = self.video_patchifier.patch_size[0]
        grid_frames = video_shape.frames // patch_t
        total_tokens = self.video_patchifier.get_token_count(video_shape)
        tokens_per_frame = total_tokens // grid_frames
        mask = torch.ones(
            1,
            total_tokens,
            device=self.device,
            dtype=torch.float32,
        )
        for start, stop, value in soft_latent_frame_ranges or []:
            start = max(0, start)
            stop = min(grid_frames, stop)
            if stop > start:
                mask[:, start * tokens_per_frame : stop * tokens_per_frame] = value
        for start, stop in cond_latent_frame_ranges:
            start = max(0, start)
            stop = min(grid_frames, stop)
            if stop > start:
                mask[:, start * tokens_per_frame : stop * tokens_per_frame] = 0.0
        return mask

    def _masked_transformer_step(
        self,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor | None,
        step_index: int,
        timestep: torch.Tensor,
        *,
        video_positions: torch.Tensor,
        audio_positions: torch.Tensor | None,
        denoise_mask: torch.Tensor,
        clean_latent: torch.Tensor,
        num_steps: int,
        text_cache: LTX23TextConditioning,
        step_transformer: LTX23Model | None = None,
        step_text_cache: LTX23TextConditioning | None = None,
        step_indices: frozenset[int] = frozenset(),
    ) -> torch.Tensor:
        """Run one LTX-2.3 denoise step with two-sided video conditioning."""
        video_float = video_latents.float()
        video_model = video_latents.to(self.dtype)
        audio_model = audio_latents.to(self.dtype) if audio_latents is not None else None
        video_timestep = (denoise_mask * timestep.unsqueeze(-1)).to(self.dtype)
        sigma = timestep.to(self.dtype)
        video = LTX23Modality(
            latent=video_model,
            timesteps=video_timestep,
            positions=video_positions,
            context=text_cache.video_context,
            sigma=sigma,
            cross_modality_sigma=sigma if audio_model is not None else None,
        )
        audio = None
        if audio_model is not None:
            if audio_positions is None:
                raise ValueError("Audio positions are required when audio latents are present.")
            audio = LTX23Modality(
                latent=audio_model,
                timesteps=torch.zeros_like(timestep),
                positions=audio_positions,
                context=text_cache.audio_context,
                sigma=sigma,
                cross_modality_sigma=sigma,
            )

        transformer = self.transformer
        active_text_cache = text_cache
        if step_index in step_indices:
            if step_transformer is None or step_text_cache is None:
                raise RuntimeError("The configured Retake companion transformer is not ready.")
            transformer = step_transformer
            active_text_cache = step_text_cache

        velocity_video, _ = transformer(
            video=video,
            audio=audio,
            text_cache=active_text_cache,
            timestep=timestep.new_tensor(float(step_index) / num_steps),
            step_index=step_index,
        )
        if velocity_video is None:
            raise RuntimeError("LTX-2.3 retake transformer returned no video prediction.")
        # Soft-margin tokens receive only a fraction of the global timestep.
        # Use that same per-token sigma when reconstructing x0; using the global
        # sigma here over-denoises the margin and causes visible colour drift.
        expanded_sigma = video_timestep.float().unsqueeze(-1)
        denoised_video = (video_float - velocity_video.float() * expanded_sigma).to(self.dtype)
        blend = denoise_mask.unsqueeze(-1).float()
        return (denoised_video.float() * blend + clean_latent.float() * (1.0 - blend)).to(
            self.dtype
        )

    def post_load_weights(self) -> None:
        super().post_load_weights()
        for component in (
            getattr(self, "video_encoder", None),
            getattr(self, "video_decoder", None),
        ):
            if component is not None and hasattr(component, "post_load_weights"):
                component.post_load_weights()
        self._build_step_transformers()
        self._setup_companion_cuda_graphs()

    def _build_step_transformer(
        self,
        *,
        quant: str,
        checkpoint_dir: str,
        description: str,
    ) -> LTX23Model:
        """Build a resident companion transformer in BF16 or dynamic FP8."""
        if self._checkpoint_dir is None:
            raise RuntimeError("The Retake checkpoint path is unavailable.")

        from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig

        quant_settings = (
            {"quant_algo": "FP8", "dynamic": True} if quant == "FP8" else {"quant_algo": None}
        )
        quant_config, _, dynamic_weight_quant, _ = (
            DiffusionPipelineConfig.load_diffusion_quant_config(quant_settings)
        )
        base_model_config = self.pipeline_config.model_configs["transformer"]
        # VisualGenMapping owns live distributed process groups, including the
        # optional parallel-VAE group, and therefore must remain a shared
        # runtime object rather than being deep-copied with the model config.
        model_config = copy.deepcopy(
            base_model_config,
            memo={
                id(base_model_config.visual_gen_mapping): base_model_config.visual_gen_mapping,
            },
        )
        model_config.quant_config = quant_config
        model_config.dynamic_weight_quant = dynamic_weight_quant
        model_config.force_dynamic_quantization = quant == "FP8"
        companion_config = copy.copy(self.pipeline_config)
        companion_config.model_configs = dict(self.pipeline_config.model_configs)
        companion_config.model_configs["transformer"] = model_config

        logger.info(f"LTX-2.3 Retake: building {description} ({quant}) from {checkpoint_dir}.")
        transformer = _build_ltx23_transformer(companion_config)
        transformer_weights = self.load_transformer_weights(checkpoint_dir)
        transformer.load_weights(transformer_weights.get("transformer", transformer_weights))
        transformer.post_load_weights()
        return transformer.to(self.device)

    def _build_step_transformers(self) -> None:
        """Build resident alternate-precision transformers used by Retake steps."""
        if self.transformer is None or self._checkpoint_dir is None:
            return

        if self._fp8_step_indices:
            logger.info(f"LTX-2.3 Retake: FP8 stage-1 steps {sorted(self._fp8_step_indices)}.")
            self._fp8_step_transformer = self._build_step_transformer(
                quant="FP8",
                checkpoint_dir=self._checkpoint_dir,
                description="stage-1 companion transformer",
            )

        if not self._stage2_step_indices:
            return
        if self._stage2_quant is None:
            raise RuntimeError("Stage-2 companion steps were configured without a quant mode.")
        stage2_checkpoint = (
            self.pipeline_config.extra_attrs.get("retake_stage2_checkpoint") or self._checkpoint_dir
        )
        if (
            self._stage2_quant == "FP8"
            and stage2_checkpoint == self._checkpoint_dir
            and self._fp8_step_transformer is not None
        ):
            logger.info("LTX-2.3 Retake: reusing the FP8 companion for stage 2.")
            self._stage2_step_transformer = self._fp8_step_transformer
            return
        self._stage2_step_transformer = self._build_step_transformer(
            quant=self._stage2_quant,
            checkpoint_dir=stage2_checkpoint,
            description="stage-2 companion transformer",
        )

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: list | None = None,
        *,
        text_encoder_path: str = "",
        **kwargs: Any,
    ) -> None:
        self._checkpoint_dir = checkpoint_dir
        native_skip_components = list(skip_components or [])
        native_skip_components += [PipelineComponent.SCHEDULER, "audio_decoder", "vocoder"]
        if not text_encoder_path:
            native_skip_components.extend(
                [
                    PipelineComponent.TOKENIZER,
                    PipelineComponent.TEXT_ENCODER,
                    "connectors",
                ]
            )
        super().load_standard_components(
            checkpoint_dir,
            device,
            text_encoder_path=text_encoder_path,
            skip_components=list(dict.fromkeys(native_skip_components)),
        )
        upsampler_path = self.pipeline_config.extra_attrs.get("retake_spatial_upsampler_path")
        if self._two_stage and upsampler_path:
            self._load_spatial_upsampler(upsampler_path, device, self.dtype)

    def _load_spatial_upsampler(
        self,
        path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Load the learned 2x latent upsampler used by two-stage Retake."""
        if not path:
            raise ValueError("A spatial upsampler path is required for two-stage Retake.")
        if self._spatial_upsampler is not None:
            if self._spatial_upsampler_path != path:
                raise ValueError(
                    "The spatial upsampler cannot change after pipeline initialization: "
                    f"loaded={self._spatial_upsampler_path!r}, requested={path!r}."
                )
            return

        state_dict: dict[str, torch.Tensor] = {}
        config: dict[str, Any] = {}
        with safetensors.torch.safe_open(path, framework="pt") as tensors:
            metadata = tensors.metadata() or {}
            if "config" in metadata:
                config = json.loads(metadata["config"])
            for key in tensors.keys():
                state_dict[key] = tensors.get_tensor(key)

        upsampler = LatentUpsamplerConfigurator.from_config(config)
        missing, unexpected = upsampler.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"LTX-2.3 spatial upsampler missing keys: {missing[:5]}")
        if unexpected:
            logger.warning(f"LTX-2.3 spatial upsampler unexpected keys: {unexpected[:5]}")
        self._spatial_upsampler = upsampler.to(device=device, dtype=dtype).eval()
        self._spatial_upsampler_path = path
        logger.info(f"Loaded LTX-2.3 Retake spatial upsampler from {path}.")

    def warmup(self) -> None:
        logger.info("Skipping LTX-2.3 retake warmup; retake requires a source video request.")

    @torch.inference_mode()
    def infer(self, req: "DiffusionRequest") -> PipelineOutput:
        extra = req.params.extra_params or {}
        video_path = self._require_extra(extra, "retake_video_path")
        start_time = extra.get("retake_start_time")
        end_time = extra.get("retake_end_time")
        request_two_stage = extra.get("retake_two_stage")
        if request_two_stage is not None and not isinstance(request_two_stage, bool):
            raise TypeError(
                "retake_two_stage must be a boolean when supplied as a request extra parameter."
            )
        two_stage = self._two_stage if request_two_stage is None else bool(request_two_stage)
        if two_stage:
            upsampler_path = extra.get(
                "retake_spatial_upsampler_path"
            ) or self.pipeline_config.extra_attrs.get("retake_spatial_upsampler_path")
            if not upsampler_path:
                raise ValueError("retake_spatial_upsampler_path is required for two-stage Retake.")
            self._load_spatial_upsampler(upsampler_path, self.device, self.dtype)

        prompt = self._single_prompt(req.prompt)
        timer = CudaPhaseTimer()
        timer.mark_pre_start()
        run_retake = self._run_two_stage_retake if two_stage else self._run_retake
        video, audio, output_shape = run_retake(
            req, video_path, start_time, end_time, prompt, timer=timer
        )
        audio_tensor = audio.samples if audio is not None else None
        sample_rate = audio.sample_rate if audio is not None else None
        if audio_tensor is not None and sample_rate is not None:
            audio_tensor = pad_audio_to_video_duration(
                audio_tensor,
                num_frames=int(video.shape[1]),
                frame_rate=float(output_shape.fps),
                sample_rate=sample_rate,
            )
        return timer.fill(
            PipelineOutput(
                video=video,
                audio=audio_tensor,
                frame_rate=float(output_shape.fps),
                audio_sample_rate=sample_rate,
            )
        )

    def _prepare_prompt_conditioning(
        self,
        prompt: str,
        max_sequence_length: int,
        conditioning_path: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return supplied prompt conditioning or encode the prompt with Gemma."""
        if conditioning_path is not None:
            if prompt:
                raise ValueError(
                    "LTX-2.3 retake accepts either a prompt or precomputed prompt "
                    "conditioning, not both."
                )
            video_embeds, audio_embeds, connector_mask = _load_prompt_conditioning(
                conditioning_path
            )
            logger.info(f"Using LTX-2.3 retake prompt conditioning from {conditioning_path}.")
            return (
                video_embeds.to(device=self.device, dtype=self.dtype),
                audio_embeds.to(device=self.device, dtype=self.dtype),
                connector_mask.to(device=self.device, dtype=self.dtype),
            )

        text_encoder = getattr(self, "text_encoder", None)
        if text_encoder is None:
            raise RuntimeError("Gemma is not loaded for LTX-2.3 retake prompt encoding.")
        if isinstance(text_encoder, torch.nn.Module):
            parameter = next(text_encoder.parameters(), None)
            if parameter is not None and parameter.device != self.device:
                text_encoder.to(self.device)
        prompt_embeds, prompt_attention_mask = self._encode_prompt(
            prompt,
            max_sequence_length=max_sequence_length,
        )
        return self._process_connectors(prompt_embeds, prompt_attention_mask)

    def _run_retake(
        self,
        req: "DiffusionRequest",
        video_path: str,
        start_time: float | None,
        end_time: float | None,
        prompt: str,
        timer: CudaPhaseTimer,
    ) -> tuple[torch.Tensor, AudioData | None, VideoPixelShape]:
        """Run native source encode, masked denoise, and full-clip decode.

        Uses the native video VAE, patchifier, scheduler, prompt conditioning,
        and masked-transformer step with
        retake-specific inputs: initial latents seeded from the encoded source,
        a two-sided ``denoise_mask`` conditioning the leading + trailing context
        while regenerating the middle window, and a two-sided ``clean_latent``
        from the source. The entire decoded clip is returned, and the source
        audio is passed through unchanged (video-only regeneration).

        Returns ``(video_uint8, source_audio, output_shape)`` where
        ``video_uint8`` is ``(1, T, H, W, C)`` uint8.
        """
        if req.params.num_inference_steps != _RETAKE_NUM_INFERENCE_STEPS:
            raise ValueError(
                "LTX-2.3 native retake uses the fixed distilled "
                f"{_RETAKE_NUM_INFERENCE_STEPS}-step schedule; "
                f"got num_inference_steps={req.params.num_inference_steps}."
            )

        device = self.device
        dtype = self.dtype
        seed = req.params.seed
        generator = torch.Generator(device=device).manual_seed(seed)

        # ---- 1. Source read + validation --------------------------------
        pixel_shape = get_videostream_metadata(video_path)
        num_frames = int(pixel_shape.frames)
        height = int(pixel_shape.height)
        width = int(pixel_shape.width)
        fps = float(pixel_shape.fps)
        self._validate_retake_source(num_frames, height, width)

        start_time, end_time = _resolve_retake_time_window(start_time, end_time, fps, num_frames)

        source_norm_5d = self._read_source_video(
            video_path, num_frames, height, width, device, dtype
        )

        # ---- 2. Retake windows ------------------------------------------
        temporal_ratio = VIDEO_SCALE_FACTORS.time
        pixel_start, pixel_end = _retake_pixel_window(start_time, end_time, fps, num_frames)
        if pixel_start == pixel_end:
            raise ValueError(
                "The retake time window does not contain a source frame after "
                f"conversion at {fps:g} FPS: [{start_time}, {end_time})."
            )
        conditioned_latent_ranges = _retake_conditioned_latent_ranges(
            pixel_start, pixel_end, num_frames, temporal_ratio
        )
        soft_margin_latent_ranges = _resolve_soft_margin_latent_ranges(
            req.params.extra_params or {},
            pixel_start=pixel_start,
            pixel_end=pixel_end,
            fps=fps,
            num_frames=num_frames,
            temporal_ratio=temporal_ratio,
        )

        # ---- 3. Native VAE encode + seed initial latents ----------------
        video_shape = VideoLatentShape.from_pixel_shape(
            pixel_shape, latent_channels=self.transformer_in_channels
        )
        # Source latents and sampled noise start in the model dtype, while the
        # blend/math path widens internally.
        if self.video_encoder is None:
            raise RuntimeError("LTX-2.3 retake requires native video VAE encoder weights.")
        source_window_latents = self._encode_video_latents(source_norm_5d, _RETAKE_TILING_CONFIG)
        expected_latent_shape = tuple(video_shape.to_torch_shape())
        if tuple(source_window_latents.shape) != expected_latent_shape:
            raise ValueError(
                "LTX-2.3 native retake: encoded source latent shape "
                f"{tuple(source_window_latents.shape)} != expected "
                f"{expected_latent_shape}; check source resolution/frame count."
            )

        denoise_mask = self._build_denoise_mask(
            video_shape,
            cond_latent_frame_ranges=conditioned_latent_ranges,
            soft_latent_frame_ranges=soft_margin_latent_ranges,
        )
        source_patch_latents = self.video_patchifier.patchify(source_window_latents)
        noise_latents = torch.randn(
            source_patch_latents.shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        # Denoising carries the latent in the model dtype and casts it back after
        # every Euler step; retaining the float32 blend would change the trajectory.
        latents = _init_retake_patchified_latents(
            noise_latents, source_patch_latents, denoise_mask
        ).to(dtype)

        # Keep the full encoded source in patchified layout. Tokens with
        # denoise_mask=1 ignore it; conditioned tokens use it after every
        # transformer prediction.
        clean_latent = source_patch_latents

        # ---- 4. Native prompt conditioning + text cache -----------------
        max_sequence_length = (
            getattr(req.params, "max_sequence_length", None) or _DEFAULT_MAX_SEQUENCE_LENGTH
        )
        conditioning_path = (req.params.extra_params or {}).get("retake_prompt_conditioning_path")
        video_embeds, audio_embeds, connector_mask = self._prepare_prompt_conditioning(
            prompt, max_sequence_length, conditioning_path
        )

        video_positions = self.video_patchifier.get_patch_grid_bounds(video_shape, device=device)
        video_positions = get_pixel_coords(
            video_positions.float(), VIDEO_SCALE_FACTORS, causal_fix=True
        )
        video_positions[:, 0, ...] = video_positions[:, 0, ...] / fps
        # RoPE positions remain float32 because position error is amplified by
        # the distilled schedule's large final Euler steps.

        # ---- Frozen audio conditioning ----------------------------------
        # Encode the source audio as a frozen conditioning modality. Video-only
        # retake never regenerates the audio.
        source_audio = decode_audio_from_file(video_path)
        audio_latent = self._encode_source_audio_latent(
            source_audio,
            pixel_shape,
        )
        # Keep the encoded source audio clean throughout denoising.
        if audio_latent is not None:
            audio_shape = AudioLatentShape.from_torch_shape(audio_latent.shape)
            audio_clean_latents = self.audio_patchifier.patchify(audio_latent.float())
            audio_positions = self.audio_patchifier.get_patch_grid_bounds(
                audio_shape, device=device
            )
        else:
            audio_clean_latents = None
            audio_embeds = None
            audio_positions = None

        text_cache = self.transformer.prepare_text_cache(
            video_context=video_embeds,
            video_context_mask=connector_mask,
            video_positions=video_positions,
            audio_context=audio_embeds,
            audio_context_mask=connector_mask if audio_embeds is not None else None,
            audio_positions=audio_positions,
            dtype=dtype,
        )

        fp8_step_text_cache = None
        if self._fp8_step_transformer is not None:
            fp8_step_text_cache = self._fp8_step_transformer.prepare_text_cache(
                video_context=video_embeds,
                video_context_mask=connector_mask,
                video_positions=video_positions,
                audio_context=audio_embeds,
                audio_context_mask=connector_mask if audio_embeds is not None else None,
                audio_positions=audio_positions,
                dtype=dtype,
            )
            text_encoder = getattr(self, "text_encoder", None)
            if isinstance(text_encoder, torch.nn.Module):
                text_encoder.to("cpu")
                torch.cuda.empty_cache()

        # ---- 5. Native masked video denoise (distilled, non-guided) --------
        scheduler = NativeSchedulerAdapter()
        scheduler.sigmas = torch.tensor(
            _RETAKE_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=device
        )
        timesteps = scheduler.timesteps
        num_steps = len(timesteps)

        # Preserved audio is clean conditioning at every denoising step. Adding
        # noise here weakens the audio guidance during the early steps where lip
        # motion is formed.
        extra_streams = None
        clean_audio = None
        if audio_clean_latents is not None:
            clean_audio = audio_clean_latents.to(dtype)
            audio_scheduler = copy.deepcopy(scheduler)
            extra_streams = {
                "audio": (clean_audio, audio_scheduler),
            }

        def retake_forward_fn(
            video_latents: torch.Tensor,
            extra_stream_latents: dict[str, torch.Tensor] | None,
            step_index: int,
            timestep: torch.Tensor,
            _encoder_hidden_states: torch.Tensor,
            _extra_tensors: dict[str, Any],
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            audio_latents = (
                extra_stream_latents.get("audio") if extra_stream_latents is not None else None
            )
            denoised_video = self._masked_transformer_step(
                video_latents,
                audio_latents,
                step_index,
                timestep,
                video_positions=video_positions,
                audio_positions=audio_positions,
                denoise_mask=denoise_mask,
                clean_latent=clean_latent,
                num_steps=num_steps,
                text_cache=text_cache,
                step_transformer=self._fp8_step_transformer,
                step_text_cache=fp8_step_text_cache,
                step_indices=self._fp8_step_indices,
            )
            return denoised_video, {}

        timer.mark_denoise_start()
        denoise_result = self.denoise(
            latents=latents,
            scheduler=scheduler,
            prompt_embeds=video_embeds,
            guidance_scale=1.0,
            forward_fn=retake_forward_fn,
            timesteps=timesteps,
            extra_streams=extra_streams,
        )
        denoised_latents = denoise_result[0] if extra_streams is not None else denoise_result

        # ---- 6. Native decode -------------------------------------------
        timer.mark_post_start()
        video_latents_5d = self.video_patchifier.unpatchify(denoised_latents, video_shape).to(dtype)
        # Encode and decode must share the retake tiling geometry.
        decoded = self._decode_video_latents(
            video_latents_5d,
            _RETAKE_TILING_CONFIG,
            generator,
        )
        decoded = postprocess_video_tensor(decoded)  # (B, T, H, W, C) uint8
        output_video = decoded.contiguous()
        timer.mark_end()

        return output_video, source_audio, pixel_shape

    def _retake_video_positions(
        self,
        video_shape: VideoLatentShape,
        fps: float,
    ) -> torch.Tensor:
        positions = self.video_patchifier.get_patch_grid_bounds(video_shape, device=self.device)
        positions = get_pixel_coords(positions.float(), VIDEO_SCALE_FACTORS, causal_fix=True)
        positions[:, 0, ...] = positions[:, 0, ...] / fps
        return positions

    def _prepare_retake_stage_text_caches(
        self,
        conditioning: _RetakeStageConditioning,
        video_positions: torch.Tensor,
        companion_transformer: LTX23Model | None,
    ) -> tuple[LTX23TextConditioning, LTX23TextConditioning | None]:
        def _prepare(transformer: LTX23Model) -> LTX23TextConditioning:
            return transformer.prepare_text_cache(
                video_context=conditioning.video_embeds,
                video_context_mask=conditioning.connector_mask,
                video_positions=video_positions,
                audio_context=conditioning.audio_embeds,
                audio_context_mask=(
                    conditioning.connector_mask if conditioning.audio_embeds is not None else None
                ),
                audio_positions=conditioning.audio_positions,
                dtype=self.dtype,
            )

        primary_cache = _prepare(self.transformer)
        companion_cache = (
            _prepare(companion_transformer) if companion_transformer is not None else None
        )
        return primary_cache, companion_cache

    def _run_masked_retake_stage(
        self,
        stage: _RetakeStage,
        conditioning: _RetakeStageConditioning,
        generator: torch.Generator,
    ) -> torch.Tensor:
        clean_latent = self.video_patchifier.patchify(stage.initial_latents)
        noise_latents = torch.randn(
            clean_latent.shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        initial_sigma = stage.sigma_values[0]
        working_latents = _init_retake_patchified_latents(
            noise_latents,
            clean_latent,
            stage.denoise_mask * initial_sigma,
        ).to(self.dtype)

        scheduler = NativeSchedulerAdapter()
        scheduler.sigmas = torch.tensor(stage.sigma_values, dtype=torch.float32, device=self.device)
        timesteps = scheduler.timesteps
        num_steps = len(timesteps)
        text_cache, companion_text_cache = self._prepare_retake_stage_text_caches(
            conditioning,
            stage.video_positions,
            stage.companion_transformer,
        )

        extra_streams = None
        clean_audio = None
        if conditioning.audio_clean_latents is not None:
            clean_audio = conditioning.audio_clean_latents.to(self.dtype)
            extra_streams = {
                "audio": (clean_audio, copy.deepcopy(scheduler)),
            }

        def _forward(
            video_latents: torch.Tensor,
            extra_stream_latents: dict[str, torch.Tensor] | None,
            step_index: int,
            timestep: torch.Tensor,
            _encoder_hidden_states: torch.Tensor,
            _extra_tensors: dict[str, Any],
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            audio_latents = (
                extra_stream_latents.get("audio") if extra_stream_latents is not None else None
            )
            denoised_video = self._masked_transformer_step(
                video_latents,
                audio_latents,
                step_index,
                timestep,
                video_positions=stage.video_positions,
                audio_positions=conditioning.audio_positions,
                denoise_mask=stage.denoise_mask,
                clean_latent=clean_latent,
                num_steps=num_steps,
                text_cache=text_cache,
                step_transformer=stage.companion_transformer,
                step_text_cache=companion_text_cache,
                step_indices=stage.companion_steps,
            )
            return denoised_video, {}

        result = self.denoise(
            latents=working_latents,
            scheduler=scheduler,
            prompt_embeds=conditioning.video_embeds,
            guidance_scale=1.0,
            forward_fn=_forward,
            timesteps=timesteps,
            extra_streams=extra_streams,
        )
        return result[0] if extra_streams is not None else result

    def _run_two_stage_retake(  # noqa: PLR0915
        self,
        req: "DiffusionRequest",
        video_path: str,
        start_time: float | None,
        end_time: float | None,
        prompt: str,
        timer: CudaPhaseTimer,
    ) -> tuple[torch.Tensor, AudioData | None, VideoPixelShape]:
        """Run a half-resolution draft followed by full-resolution refinement."""
        if req.params.num_inference_steps != _TWO_STAGE_STAGE_1_NUM_INFERENCE_STEPS:
            raise ValueError(
                "LTX-2.3 two-stage Retake uses the fixed distilled "
                f"{_TWO_STAGE_STAGE_1_NUM_INFERENCE_STEPS}-step first stage; "
                f"got num_inference_steps={req.params.num_inference_steps}."
            )
        if self.video_encoder is None:
            raise RuntimeError("LTX-2.3 two-stage Retake requires video VAE encoder weights.")
        if self._spatial_upsampler is None:
            raise RuntimeError("LTX-2.3 two-stage Retake spatial upsampler is not loaded.")

        device = self.device
        dtype = self.dtype
        generator = torch.Generator(device=device).manual_seed(req.params.seed)

        pixel_shape = get_videostream_metadata(video_path)
        num_frames = int(pixel_shape.frames)
        height = int(pixel_shape.height)
        width = int(pixel_shape.width)
        fps = float(pixel_shape.fps)
        self._validate_retake_source(num_frames, height, width)
        if height % 64 != 0 or width % 64 != 0:
            raise ValueError(
                "LTX-2.3 two-stage Retake requires source height and width divisible "
                f"by 64; got {height}x{width}."
            )
        start_time, end_time = _resolve_retake_time_window(start_time, end_time, fps, num_frames)
        source_norm_5d = self._read_source_video(
            video_path, num_frames, height, width, device, dtype
        )

        batch, channels, frames, _, _ = source_norm_5d.shape
        half_frames = source_norm_5d.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        half_frames = torch.nn.functional.interpolate(
            half_frames.float(),
            size=(height // 2, width // 2),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        ).to(dtype)
        source_half_5d = (
            half_frames.reshape(batch, frames, channels, height // 2, width // 2)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

        temporal_ratio = VIDEO_SCALE_FACTORS.time
        pixel_start, pixel_end = _retake_pixel_window(start_time, end_time, fps, num_frames)
        if pixel_start == pixel_end:
            raise ValueError(
                "The retake time window does not contain a source frame after "
                f"conversion at {fps:g} FPS: [{start_time}, {end_time})."
            )
        conditioned_latent_ranges = _retake_conditioned_latent_ranges(
            pixel_start, pixel_end, num_frames, temporal_ratio
        )
        soft_margin_latent_ranges = _resolve_soft_margin_latent_ranges(
            req.params.extra_params or {},
            pixel_start=pixel_start,
            pixel_end=pixel_end,
            fps=fps,
            num_frames=num_frames,
            temporal_ratio=temporal_ratio,
        )
        total_latent_frames = _latent_frame_count(num_frames, temporal_ratio)
        regenerated_start = max(
            0,
            min(
                _pixel_frame_to_latent_index(pixel_start, temporal_ratio),
                total_latent_frames,
            ),
        )
        regenerated_end = max(
            regenerated_start,
            min(
                _pixel_frame_to_latent_index(pixel_end - 1, temporal_ratio) + 1,
                total_latent_frames,
            ),
        )

        full_shape = VideoLatentShape.from_pixel_shape(
            pixel_shape, latent_channels=self.transformer_in_channels
        )
        half_pixel_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            height=height // 2,
            width=width // 2,
            fps=fps,
        )
        half_shape = VideoLatentShape.from_pixel_shape(
            half_pixel_shape, latent_channels=self.transformer_in_channels
        )

        source_half_latents = self._encode_video_latents(
            source_half_5d, _TWO_STAGE_RETAKE_TILING_CONFIG
        )
        source_full_latents = self._encode_video_latents(
            source_norm_5d, _TWO_STAGE_RETAKE_TILING_CONFIG
        )
        for label, actual, expected in (
            ("half-resolution", source_half_latents, half_shape),
            ("full-resolution", source_full_latents, full_shape),
        ):
            expected_shape = tuple(expected.to_torch_shape())
            if tuple(actual.shape) != expected_shape:
                raise ValueError(
                    f"LTX-2.3 two-stage Retake {label} encoded latent shape "
                    f"{tuple(actual.shape)} != expected {expected_shape}."
                )

        max_sequence_length = (
            getattr(req.params, "max_sequence_length", None) or _DEFAULT_MAX_SEQUENCE_LENGTH
        )
        conditioning_path = (req.params.extra_params or {}).get("retake_prompt_conditioning_path")
        video_embeds, audio_embeds, connector_mask = self._prepare_prompt_conditioning(
            prompt, max_sequence_length, conditioning_path
        )

        source_audio = decode_audio_from_file(video_path)
        audio_latent = self._encode_source_audio_latent(source_audio, pixel_shape)
        if audio_latent is not None:
            audio_shape = AudioLatentShape.from_torch_shape(audio_latent.shape)
            audio_clean_latents = self.audio_patchifier.patchify(audio_latent.float())
            audio_positions = self.audio_patchifier.get_patch_grid_bounds(
                audio_shape, device=device
            )
        else:
            audio_clean_latents = None
            audio_embeds = None
            audio_positions = None

        conditioning = _RetakeStageConditioning(
            video_embeds=video_embeds,
            audio_embeds=audio_embeds,
            connector_mask=connector_mask,
            audio_clean_latents=audio_clean_latents,
            audio_positions=audio_positions,
        )

        timer.mark_denoise_start()
        half_positions = self._retake_video_positions(half_shape, fps)
        half_mask = self._build_denoise_mask(
            half_shape,
            cond_latent_frame_ranges=conditioned_latent_ranges,
            soft_latent_frame_ranges=soft_margin_latent_ranges,
        )
        stage_1_tokens = self._run_masked_retake_stage(
            _RetakeStage(
                initial_latents=source_half_latents,
                video_positions=half_positions,
                denoise_mask=half_mask,
                sigma_values=_TWO_STAGE_STAGE_1_SIGMA_VALUES,
                companion_transformer=self._fp8_step_transformer,
                companion_steps=self._fp8_step_indices,
            ),
            conditioning,
            generator,
        )
        stage_1_latents = self.video_patchifier.unpatchify(stage_1_tokens, half_shape).to(dtype)

        statistics = self.video_encoder.per_channel_statistics
        upsampled_latents = upsample_video(
            stage_1_latents,
            statistics,
            self._spatial_upsampler,
        ).to(dtype)
        if upsampled_latents.shape != source_full_latents.shape:
            raise ValueError(
                "LTX-2.3 two-stage Retake upsampled latent shape "
                f"{tuple(upsampled_latents.shape)} != full-resolution source latent shape "
                f"{tuple(source_full_latents.shape)}."
            )
        composite_latents = source_full_latents.clone()
        composite_latents[:, :, regenerated_start:regenerated_end] = upsampled_latents[
            :, :, regenerated_start:regenerated_end
        ]

        full_positions = self._retake_video_positions(full_shape, fps)
        full_mask = self._build_denoise_mask(
            full_shape,
            cond_latent_frame_ranges=conditioned_latent_ranges,
            soft_latent_frame_ranges=soft_margin_latent_ranges,
        )
        stage_2_tokens = self._run_masked_retake_stage(
            _RetakeStage(
                initial_latents=composite_latents,
                video_positions=full_positions,
                denoise_mask=full_mask,
                sigma_values=_TWO_STAGE_STAGE_2_SIGMA_VALUES,
                companion_transformer=self._stage2_step_transformer,
                companion_steps=self._stage2_step_indices,
            ),
            conditioning,
            generator,
        )

        timer.mark_post_start()
        final_latents = self.video_patchifier.unpatchify(stage_2_tokens, full_shape).to(dtype)
        decoded = self._decode_video_latents(
            final_latents,
            _TWO_STAGE_RETAKE_TILING_CONFIG,
            generator,
        )
        output_video = postprocess_video_tensor(decoded).contiguous()
        timer.mark_end()
        return output_video, source_audio, pixel_shape

    @staticmethod
    def _validate_retake_source(num_frames: int, height: int, width: int) -> None:
        """Fail fast on source video shapes the native VAE cannot round-trip.

        The LTX-2.3 causal video VAE requires ``8k + 1`` pixel frames and spatial
        dimensions divisible by the VAE spatial scale factors so encode/decode
        preserves shape.
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
        height_scale = VIDEO_SCALE_FACTORS.height
        width_scale = VIDEO_SCALE_FACTORS.width
        if height % height_scale != 0 or width % width_scale != 0:
            raise ValueError(
                "retake source resolution must be divisible by the VAE spatial "
                f"scale factors {height_scale}x{width_scale}; got {height}x{width}."
            )

    def _read_source_video(
        self,
        video_path: str,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Read and validate the source video as a normalized VAE input.

        Returns ``(1, 3, T, H, W)`` in ``[-1, 1]``. Pixel decoding is
        deterministic and uses sequential frame indices.
        """
        frames = list(decode_video_by_frame(video_path))
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
        # Transfer compact uint8 pixels and fuse normalize + THWC-to-NCTHW + cast
        # on the GPU. This avoids a full-sized CPU FP32 temporary and halves H2D
        # traffic relative to transferring BF16 pixels.
        source_uint8_cuda = source_uint8.to(device=device)
        return uint8_bthwc_to_normalized_ncthw(source_uint8_cuda, dtype=dtype)

    def _encode_source_audio_latent(
        self,
        audio: AudioData | None,
        pixel_shape: VideoPixelShape,
    ) -> torch.Tensor | None:
        """Encode the source audio into the frozen conditioning latent, natively.

        Runs the native audio VAE encoder over the decoded mel spectrogram and
        conforms the latent length to what the video pixel shape implies. Returns
        ``None`` when the source carries no audio stream.

        The duration is derived from the *video* shape (``frames / fps``) rather
        than the audio stream's own length, because the audio latent has to line
        up token-for-token with the video latent the transformer cross-attends
        to.
        """
        if audio is None:
            return None
        max_samples = round(pixel_shape.frames * audio.sample_rate / pixel_shape.fps)
        conditioning_audio = AudioData(
            samples=audio.samples[..., :max_samples],
            sample_rate=audio.sample_rate,
        )
        latents = encode_audio(conditioning_audio, self._audio_encoder).to(self.device, self.dtype)
        required_latent_frames = AudioLatentShape.from_video_pixel_shape(
            pixel_shape,
            sample_rate=self._audio_encoder.sample_rate,
            hop_length=self._audio_encoder.mel_hop_length,
            audio_latent_downsample_factor=(
                self._audio_encoder.patchifier.audio_latent_downsample_factor
            ),
        ).frames
        return _conform_latent_length(latents, required_latent_frames)

    @staticmethod
    def _single_prompt(prompt: str | list[str]) -> str:
        if isinstance(prompt, str):
            return prompt
        if len(prompt) == 1:
            return prompt[0]
        raise ValueError("LTX-2.3 retake workflow supports one prompt per request.")

    @staticmethod
    def _require_extra(extra: dict[str, Any], key: str) -> Any:
        value = extra.get(key)
        if value is None:
            raise ValueError(f"extra_params['{key}'] is required for LTX-2.3 retake.")
        return value
