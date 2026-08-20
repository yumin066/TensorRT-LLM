# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native LTX-2 retake workflow for VisualGen.

This pipeline keeps a resident retake model for ``trtllm-serve`` and extends the
native LTX-2 component loader with the LTX-2.3 encoder, decoder, and connectors.
It either loads Gemma for a live prompt or consumes precomputed post-connector
conditioning. Retake generalizes image-to-video masking to condition both sides
of the regenerated window.

Source video/audio decode, stream metadata, audio/video VAE encode, diffusion,
and VAE decode all use TensorRT-LLM's native LTX-2 implementation. PyAV is the
only source-media I/O dependency; no upstream LTX pipeline is imported.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Sequence

import safetensors.torch
import torch

from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm.logger import logger

from ..ltx2.ltx2_core.diffusion_steps import EulerDiffusionStep
from ..ltx2.ltx2_core.patchifier import VideoLatentPatchifier, get_pixel_coords
from ..ltx2.ltx2_core.rope import LTXRopeType
from ..ltx2.ltx2_core.scheduler_adapter import NativeSchedulerAdapter
from ..ltx2.ltx2_core.types import (
    VIDEO_SCALE_FACTORS,
    AudioLatentShape,
    VideoLatentShape,
    VideoPixelShape,
)
from ..ltx2.ltx2_core.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ..ltx2.pipeline_ltx2 import LTX2Pipeline, _find_safetensors_files
from .ltx2_retake_core.attention import FlashInferSelfAttention
from .ltx2_retake_core.audio_vae import AudioEncoderConfigurator, encode_audio
from .ltx2_retake_core.connector import (
    AudioEmbeddings1DConnectorConfigurator,
    Embeddings1DConnectorConfigurator,
    GemmaFeaturesExtractorConfigurator,
)
from .ltx2_retake_core.media_io import (
    decode_audio_from_file,
    decode_video_by_frame,
    get_videostream_metadata,
    pad_audio_to_video_duration,
)
from .ltx2_retake_core.modality import Modality
from .ltx2_retake_core.video_vae import (
    RetakeVideoDecoderConfigurator,
    RetakeVideoEncoderConfigurator,
)
from .transformer_ltx2_retake import LTX2Attention, LTXModel, LTXModelType

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
    from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest
    from tensorrt_llm._torch.visual_gen.models.ltx2.text_cache import TextCache
    from tensorrt_llm.inputs.multimodal_data import AudioData

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
_RETAKE_NUM_INFERENCE_STEPS = len(_RETAKE_DISTILLED_SIGMA_VALUES) - 1

_DEFAULT_MAX_SEQUENCE_LENGTH = 1024
_PROMPT_CONDITIONING_KEYS = ("video_embeds", "audio_embeds", "connector_mask")

# Retake uses the same tiling geometry for source encode and output decode. Keep
# it separate from the generation pipeline's default because tile boundaries are
# part of the retake numerical contract.
_RETAKE_TILING_CONFIG = TilingConfig(
    spatial_config=SpatialTilingConfig(tile_size_in_pixels=768, tile_overlap_in_pixels=64),
    temporal_config=TemporalTilingConfig(tile_size_in_frames=80, tile_overlap_in_frames=24),
)

_VIDEO_ENCODER_WEIGHT_PREFIXES = (
    ("vae.encoder.", ""),
    ("vae.per_channel_statistics.", "per_channel_statistics."),
)
_VIDEO_DECODER_WEIGHT_PREFIXES = (
    ("vae.decoder.", ""),
    ("vae.per_channel_statistics.", "per_channel_statistics."),
)
_AUDIO_ENCODER_WEIGHT_PREFIXES = (
    ("audio_vae.encoder.", ""),
    ("audio_vae.per_channel_statistics.", "per_channel_statistics."),
)


def _load_retake_component_weights(
    safetensors_paths: list[str],
    module: torch.nn.Module,
    prefixes: str | Sequence[str | tuple[str, str]],
) -> None:
    """Load a retake component with optional checkpoint-key remapping."""
    mappings = (
        [(prefixes, "")]
        if isinstance(prefixes, str)
        else [(item, "") if isinstance(item, str) else item for item in prefixes]
    )
    state_dict: dict[str, torch.Tensor] = {}
    for path in safetensors_paths:
        with safetensors.torch.safe_open(path, framework="pt") as handle:
            for key in handle.keys():
                for source, target in mappings:
                    if key.startswith(source):
                        state_dict[target + key[len(source) :]] = handle.get_tensor(key)
                        break
    if not state_dict:
        logger.warning(f"No weights found with prefixes {mappings}")
        return
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Retake component is missing {len(missing)} weights: {missing[:10]}")
    if unexpected:
        logger.warning(
            f"Retake component has {len(unexpected)} unexpected weights: {unexpected[:5]}"
        )


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
            "LTX-2 retake prompt conditioning must contain one prompt; "
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


def _normalize_fp8_step_indices(steps: Any) -> frozenset[int]:
    """Validate configured fallback steps against the fixed retake schedule."""
    if steps is None:
        return frozenset()
    if isinstance(steps, (str, bytes)):
        raise TypeError("retake_fp8_linear_steps must be a sequence of integer step indices.")
    try:
        values = list(steps)
    except TypeError as exc:
        raise TypeError(
            "retake_fp8_linear_steps must be a sequence of integer step indices."
        ) from exc

    invalid_types = [step for step in values if isinstance(step, bool) or not isinstance(step, int)]
    if invalid_types:
        raise TypeError(f"retake_fp8_linear_steps entries must be integers; got {invalid_types!r}.")
    invalid_range = [step for step in values if not 0 <= step < _RETAKE_NUM_INFERENCE_STEPS]
    if invalid_range:
        raise ValueError(
            "retake_fp8_linear_steps entries must be in "
            f"[0, {_RETAKE_NUM_INFERENCE_STEPS}); got {invalid_range!r}."
        )
    return frozenset(values)


def build_ltx2_transformer(pipeline_config: "DiffusionPipelineConfig") -> LTXModel:
    """Build the checkpoint-native LTX-2.3 transformer used by retake."""
    attention = getattr(pipeline_config, "attention", None)
    if attention is not None and getattr(attention, "quant_attention_config", None) is not None:
        raise NotImplementedError("Quantized attention is not yet supported for LTX-2 retake.")

    model_config = pipeline_config.model_configs["transformer"]
    config = model_config.pretrained_config
    rope_type = LTXRopeType(getattr(config, "rope_type", "interleaved"))
    double_precision_rope = getattr(config, "frequencies_precision", False) == "float64"
    transformer = LTXModel(
        model_type=LTXModelType.AudioVideo,
        num_attention_heads=getattr(config, "num_attention_heads", 32),
        attention_head_dim=getattr(config, "attention_head_dim", 128),
        in_channels=getattr(config, "in_channels", 128),
        out_channels=getattr(config, "out_channels", 128),
        num_layers=getattr(config, "num_layers", 48),
        cross_attention_dim=getattr(config, "cross_attention_dim", 4096),
        norm_eps=float(getattr(config, "norm_eps", 1e-6)),
        caption_channels=getattr(config, "caption_channels", 3840),
        positional_embedding_theta=float(getattr(config, "positional_embedding_theta", 10000.0)),
        positional_embedding_max_pos=getattr(
            config, "positional_embedding_max_pos", [20, 2048, 2048]
        ),
        timestep_scale_multiplier=getattr(config, "timestep_scale_multiplier", 1000),
        use_middle_indices_grid=getattr(config, "use_middle_indices_grid", True),
        audio_num_attention_heads=getattr(config, "audio_num_attention_heads", 32),
        audio_attention_head_dim=getattr(config, "audio_attention_head_dim", 64),
        audio_in_channels=getattr(config, "audio_in_channels", 128),
        audio_out_channels=getattr(config, "audio_out_channels", 128),
        audio_cross_attention_dim=getattr(config, "audio_cross_attention_dim", 2048),
        audio_positional_embedding_max_pos=getattr(
            config, "audio_positional_embedding_max_pos", [20]
        ),
        av_ca_timestep_scale_multiplier=getattr(config, "av_ca_timestep_scale_multiplier", 1),
        rope_type=rope_type,
        double_precision_rope=double_precision_rope,
        apply_gated_attention=getattr(config, "apply_gated_attention", False),
        cross_attention_adaln=getattr(config, "cross_attention_adaln", False),
        model_config=model_config,
    )
    transformer._transformer_config = vars(config)

    retake_attention_backend = pipeline_config.extra_attrs.get("retake_attention_backend")
    if retake_attention_backend is not None:
        retake_attention = FlashInferSelfAttention(retake_attention_backend)
        for module in transformer.modules():
            if isinstance(module, LTX2Attention):
                module.retake_attention = retake_attention
        logger.info(f"LTX-2 retake video self-attention: {retake_attention_backend}")

    if getattr(config, "caption_proj_before_connector", False):
        for preprocessor_name in ("video_args_preprocessor", "audio_args_preprocessor"):
            preprocessor = getattr(transformer, preprocessor_name, None)
            target = getattr(preprocessor, "simple_preprocessor", preprocessor)
            if target is not None and getattr(target, "caption_projection", None) is not None:
                target.caption_projection = torch.nn.Identity()
    return transformer


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


# PEFT and ComfyUI LoRA key conventions.
_LORA_DOWN_SUFFIXES = (".lora_A.weight", ".lora_down.weight")  # A: (rank, in)
_LORA_UP_SUFFIXES = (".lora_B.weight", ".lora_up.weight")  # B: (out, rank)
_LORA_MODULE_PREFIXES = ("model.diffusion_model.", "diffusion_model.")


def _strip_lora_suffix(key: str, suffixes: tuple[str, ...]) -> str | None:
    for suffix in suffixes:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def _strip_lora_prefix(module: str) -> str:
    for prefix in _LORA_MODULE_PREFIXES:
        if module.startswith(prefix):
            return module[len(prefix) :]
    return module


def _fuse_lora_into_transformer_weights(
    weights: dict[str, torch.Tensor], lora_path: str, strength: float
) -> dict[str, torch.Tensor]:
    """Fuse PEFT- or ComfyUI-format LoRA weights into checkpoint-key space."""

    down: dict[str, torch.Tensor] = {}
    up: dict[str, torch.Tensor] = {}
    alphas: dict[str, float] = {}
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


# Retake shares checkpoints with LTX2Pipeline, so it is selected explicitly
# rather than claiming the same Hugging Face IDs for auto-detection.
@register_pipeline(
    "LTX2RetakePipeline",
    defaults={
        "text_encoder_path": None,
        "retake_lora_path": None,
        "retake_lora_strength": 1.0,
        "retake_prompt_conditioning_path": None,
        "retake_start_time": None,
        "retake_end_time": None,
        "retake_seed": 42,
        "retake_fp8_linear_steps": None,
        "retake_attention_backend": None,
    },
    doc="Native LTX-2.3 video retake pipeline.",
)
class LTX2RetakePipeline(LTX2Pipeline):
    """Persistent VisualGen pipeline for native LTX-2.3 retake requests."""

    def __init__(self, pipeline_config: "DiffusionPipelineConfig") -> None:
        if pipeline_config.cache_backend is not None:
            raise NotImplementedError("Cache acceleration is not supported for LTX-2 retake.")
        if pipeline_config.parallel.cfg_size != 1:
            raise NotImplementedError("LTX-2 retake requires parallel.cfg_size=1.")
        if pipeline_config.parallel.parallel_vae_size != 1:
            raise NotImplementedError("Parallel VAE is not supported for LTX-2 retake.")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prompt_conditioning: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._fp8_step_transformer: LTXModel | None = None
        self._fp8_step_indices = _normalize_fp8_step_indices(
            pipeline_config.extra_attrs.get("retake_fp8_linear_steps")
        )
        self._fp8_step_text_cache: "TextCache | None" = None
        super().__init__(pipeline_config)

    @property
    def default_generation_params(self) -> dict[str, int]:
        return {
            "num_inference_steps": _RETAKE_NUM_INFERENCE_STEPS,
            "seed": int(self.pipeline_config.extra_attrs.get("retake_seed", 42)),
        }

    @property
    def extra_param_specs(self) -> dict[str, ExtraParamSchema]:
        return {
            "retake_video_path": ExtraParamSchema(
                type="str",
                description="Path to the source video file for retake.",
            ),
            "retake_start_time": ExtraParamSchema(
                type="float",
                default=self.pipeline_config.extra_attrs.get("retake_start_time"),
                description="Start time in seconds for the regenerated window.",
            ),
            "retake_end_time": ExtraParamSchema(
                type="float",
                default=self.pipeline_config.extra_attrs.get("retake_end_time"),
                description="End time in seconds for the regenerated window.",
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
        """Load LTX-2.3 components without changing the generation pipeline."""
        skip_components = skip_components or []

        if PipelineComponent.VAE not in skip_components:
            self.video_decoder = RetakeVideoDecoderConfigurator.from_config(config)
            _load_retake_component_weights(
                safetensors_paths,
                self.video_decoder,
                _VIDEO_DECODER_WEIGHT_PREFIXES,
            )
            self.video_decoder = self.video_decoder.to(device=device, dtype=dtype)

        if "connectors" not in skip_components:
            self.feature_extractor = GemmaFeaturesExtractorConfigurator.from_config(config)
            _load_retake_component_weights(
                safetensors_paths,
                self.feature_extractor,
                "text_embedding_projection.",
            )
            self.feature_extractor = self.feature_extractor.to(device=device, dtype=dtype)

            self.video_connector = Embeddings1DConnectorConfigurator.from_config(config)
            _load_retake_component_weights(
                safetensors_paths,
                self.video_connector,
                "model.diffusion_model.video_embeddings_connector.",
            )
            self.video_connector = self.video_connector.to(device=device, dtype=dtype)

            self.audio_connector = AudioEmbeddings1DConnectorConfigurator.from_config(config)
            _load_retake_component_weights(
                safetensors_paths,
                self.audio_connector,
                "model.diffusion_model.audio_embeddings_connector.",
            )
            self.audio_connector = self.audio_connector.to(device=device, dtype=dtype)

        if "video_encoder" not in skip_components:
            encoder_blocks = config.get("vae", {}).get("encoder_blocks", [])
            if encoder_blocks:
                self.video_encoder = RetakeVideoEncoderConfigurator.from_config(config)
                _load_retake_component_weights(
                    safetensors_paths,
                    self.video_encoder,
                    _VIDEO_ENCODER_WEIGHT_PREFIXES,
                )
                self.video_encoder = self.video_encoder.to(device=device, dtype=dtype)
            else:
                self.video_encoder = None
        else:
            self.video_encoder = None

        transformer_config = self.transformer._transformer_config
        self.video_patchifier = VideoLatentPatchifier(
            patch_size=transformer_config.get("patch_size", 1)
        )

    def _encode_prompt(
        self,
        prompt: str | list[str],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 1024,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw stacked Gemma hidden states for the retake connectors."""
        prompts = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompts)
        text_inputs = self.tokenizer(
            [value.strip() for value in prompts],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = torch.stack(outputs.hidden_states, dim=-1).to(dtype=self.dtype)
        _, sequence_length, hidden_dim, num_layers = hidden_states.shape
        hidden_states = hidden_states.repeat(1, num_videos_per_prompt, 1, 1).view(
            batch_size * num_videos_per_prompt,
            sequence_length,
            hidden_dim,
            num_layers,
        )
        attention_mask = attention_mask.view(batch_size, -1).repeat_interleave(
            num_videos_per_prompt, dim=0
        )
        return hidden_states, attention_mask

    def _process_connectors(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        additive_mask = (1 - attention_mask.to(hidden_states.dtype)) * -1000000.0
        additive_mask = additive_mask.unsqueeze(1).unsqueeze(1)
        video_features, audio_features = self.feature_extractor(hidden_states, attention_mask)
        video_embeds, connector_mask = self.video_connector(video_features, additive_mask)
        audio_embeds, _ = self.audio_connector(audio_features, additive_mask)
        return video_embeds, audio_embeds, connector_mask

    @torch.inference_mode()
    def _encode_video_window(
        self,
        video: torch.Tensor,
        tiling_config: TilingConfig | None = None,
    ) -> torch.Tensor:
        if self.video_encoder is None:
            raise RuntimeError("LTX-2 retake requires native video VAE encoder weights.")
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Expected a (B, 3, T, H, W) video; got {tuple(video.shape)}")
        return self.video_encoder.tiled_encode(video, tiling_config)

    def _build_denoise_mask(
        self,
        video_shape: VideoLatentShape,
        *,
        cond_latent_frame_ranges: list[tuple[int, int]],
        strength: float = 1.0,
    ) -> torch.Tensor:
        patch_t, patch_h, patch_w = self.video_patchifier.patch_size
        grid_frames = video_shape.frames // patch_t
        tokens_per_frame = (video_shape.height // patch_h) * (video_shape.width // patch_w)
        mask = torch.ones(
            1,
            grid_frames * tokens_per_frame,
            device=self.device,
            dtype=torch.float32,
        )
        for start, stop in cond_latent_frame_ranges:
            start = max(0, start)
            stop = min(grid_frames, stop)
            if stop > start:
                mask[:, start * tokens_per_frame : stop * tokens_per_frame] = 1.0 - strength
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
        text_cache: "TextCache",
    ) -> torch.Tensor:
        """Run one LTX-2.3 denoise step with two-sided video conditioning."""
        video_float = video_latents.float()
        video_model = video_latents.to(self.dtype)
        audio_model = audio_latents.to(self.dtype) if audio_latents is not None else None
        video_timestep = (denoise_mask * timestep.unsqueeze(-1)).to(self.dtype)
        sigma = timestep.to(self.dtype)
        video = Modality(
            latent=video_model,
            timesteps=video_timestep,
            positions=video_positions,
            sigma=sigma,
            cross_modality_sigma=sigma if audio_model is not None else None,
        )
        audio = None
        if audio_model is not None:
            if audio_positions is None:
                raise ValueError("Audio positions are required when audio latents are present.")
            audio = Modality(
                latent=audio_model,
                timesteps=torch.zeros_like(timestep),
                positions=audio_positions,
                sigma=sigma,
                cross_modality_sigma=sigma,
            )

        transformer = self.transformer
        active_text_cache = text_cache
        if step_index in self._fp8_step_indices:
            if self._fp8_step_transformer is None or self._fp8_step_text_cache is None:
                raise RuntimeError("The configured FP8 retake transformer is not ready.")
            transformer = self._fp8_step_transformer
            active_text_cache = self._fp8_step_text_cache

        velocity_video, _ = transformer(
            video=video,
            audio=audio,
            perturbations=None,
            text_cache=active_text_cache,
            timestep=timestep.new_tensor(float(step_index) / num_steps),
            step_index=step_index,
        )
        if velocity_video is None:
            raise RuntimeError("LTX-2 retake transformer returned no video prediction.")
        expanded_sigma = sigma.float()
        while expanded_sigma.ndim < velocity_video.ndim:
            expanded_sigma = expanded_sigma.unsqueeze(-1)
        denoised_video = (video_float - velocity_video.float() * expanded_sigma).to(self.dtype)
        blend = denoise_mask.unsqueeze(-1).float()
        return (denoised_video.float() * blend + clean_latent.float() * (1.0 - blend)).to(
            self.dtype
        )

    def _init_transformer(self) -> None:
        self.transformer = build_ltx2_transformer(self.pipeline_config)

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        transformer_weights = weights.get("transformer", weights)
        lora_path = self.pipeline_config.extra_attrs.get("retake_lora_path")
        if lora_path:
            strength = float(self.pipeline_config.extra_attrs.get("retake_lora_strength", 1.0))
            transformer_weights = _fuse_lora_into_transformer_weights(
                transformer_weights, lora_path, strength
            )
        super().load_weights(transformer_weights)

    def post_load_weights(self) -> None:
        super().post_load_weights()
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

        steps = self._fp8_step_indices
        if not steps or self.transformer is None:
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

        logger.info(f"LTX-2 retake: building resident FP8 transformer for steps {sorted(steps)}")
        fp8_tf = build_ltx2_transformer(fp8_cfg)
        tw = self.load_transformer_weights(self._checkpoint_dir)
        lora_path = self.pipeline_config.extra_attrs.get("retake_lora_path")
        if lora_path:
            strength = float(self.pipeline_config.extra_attrs.get("retake_lora_strength", 1.0))
            tw = _fuse_lora_into_transformer_weights(tw, lora_path, strength)
        fp8_tf.load_weights(tw)
        if hasattr(fp8_tf, "post_load_weights"):
            fp8_tf.post_load_weights()
        fp8_tf.to(self._device)
        self._fp8_step_transformer = fp8_tf

    def _setup_cuda_graphs(self) -> None:
        if self.pipeline_config.cuda_graph.enable:
            raise NotImplementedError("LTX-2 retake currently requires cuda_graph.enable=false.")

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: list | None = None,
        *,
        text_encoder_path: str = "",
        **kwargs: Any,
    ) -> None:
        self._device = device
        self._checkpoint_dir = checkpoint_dir
        conditioning_path = self.pipeline_config.extra_attrs.get("retake_prompt_conditioning_path")
        self._prompt_conditioning = None
        if conditioning_path and text_encoder_path:
            raise ValueError(
                "LTX-2 retake accepts either retake_prompt_conditioning_path or "
                "text_encoder_path, not both."
            )
        if conditioning_path:
            video_embeds, audio_embeds, connector_mask = _load_prompt_conditioning(
                conditioning_path
            )
            self._prompt_conditioning = (
                video_embeds.to(device=device, dtype=self.dtype),
                audio_embeds.to(device=device, dtype=self.dtype),
                connector_mask.to(device=device, dtype=self.dtype),
            )
            logger.info(f"Loaded LTX-2 retake prompt conditioning from {conditioning_path}.")
        elif not text_encoder_path:
            raise ValueError(
                "LTX-2 retake requires either precomputed prompt conditioning or "
                "pipeline_config.text_encoder_path pointing at the Gemma text encoder."
            )
        native_skip_components = list(skip_components or [])
        if self._prompt_conditioning is not None:
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
        self.transformer_in_channels = self.transformer._transformer_config.get("in_channels", 128)
        self._load_retake_audio_encoder(checkpoint_dir, device)

    def _load_retake_audio_encoder(self, checkpoint_dir: str, device: torch.device) -> None:
        """Load the native audio VAE encoder for retake audio conditioning."""
        sft_paths = _find_safetensors_files(checkpoint_dir)
        encoder = AudioEncoderConfigurator.from_config(self._native_config)
        _load_retake_component_weights(sft_paths, encoder, _AUDIO_ENCODER_WEIGHT_PREFIXES)
        self._audio_encoder = encoder.to(device=device, dtype=self.dtype)
        self.audio_patchifier = self._audio_encoder.patchifier

    def warmup(self) -> None:
        logger.info("Skipping LTX-2 retake warmup; retake requires a source video request.")

    @torch.inference_mode()
    def infer(self, req: "DiffusionRequest") -> PipelineOutput:
        extra = req.params.extra_params or {}
        video_path = self._require_extra(extra, "retake_video_path")
        start_time = float(self._require_extra(extra, "retake_start_time"))
        end_time = float(self._require_extra(extra, "retake_end_time"))
        if not math.isfinite(start_time) or not math.isfinite(end_time):
            raise ValueError("retake_start_time and retake_end_time must be finite.")
        if start_time >= end_time:
            raise ValueError(
                f"retake_start_time ({start_time}) must be less than retake_end_time ({end_time})"
            )

        prompt = self._single_prompt(req.prompt)
        timer = CudaPhaseTimer()
        timer.mark_pre_start()
        video, audio, output_shape = self._run_native_pre_post_retake(
            req, video_path, start_time, end_time, prompt, timer=timer
        )
        audio_tensor, sample_rate = self._normalize_audio(audio)
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

    # ------------------------------------------------------------------
    # Native retake pre/post runtime path
    # ------------------------------------------------------------------

    def _prepare_prompt_conditioning(
        self,
        prompt: str,
        max_sequence_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return supplied prompt conditioning or encode the prompt with Gemma."""
        if self._prompt_conditioning is not None:
            if prompt:
                raise ValueError(
                    "LTX-2 retake accepts either a prompt or precomputed prompt "
                    "conditioning, not both."
                )
            logger.info("Using precomputed LTX-2 retake prompt conditioning.")
            return self._prompt_conditioning

        text_encoder = getattr(self, "text_encoder", None)
        if text_encoder is None:
            raise RuntimeError("Gemma is not loaded for LTX-2 retake prompt encoding.")
        if isinstance(text_encoder, torch.nn.Module):
            first_parameter = next(text_encoder.parameters(), None)
            if first_parameter is not None and first_parameter.device != device:
                text_encoder.to(device)
        prompt_embeds, prompt_attention_mask = self._encode_prompt(
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        return self._process_connectors(prompt_embeds, prompt_attention_mask)

    def _run_native_pre_post_retake(
        self,
        req: "DiffusionRequest",
        video_path: str,
        start_time: float,
        end_time: float,
        prompt: str,
        timer: CudaPhaseTimer | None = None,
    ) -> tuple[torch.Tensor, "AudioData | None", VideoPixelShape]:
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
                "LTX-2 native retake uses the fixed distilled "
                f"{_RETAKE_NUM_INFERENCE_STEPS}-step schedule; "
                f"got num_inference_steps={req.params.num_inference_steps}."
            )

        device = self._device
        dtype = self.dtype
        seed = req.params.seed
        generator = torch.Generator(device=device).manual_seed(seed)

        # ---- 1. Source read + validation --------------------------------
        output_shape = get_videostream_metadata(video_path)
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
        if pixel_start == pixel_end:
            raise ValueError(
                "The retake time window does not contain a source frame after "
                f"conversion at {fps:g} FPS: [{start_time}, {end_time})."
            )
        conditioned_latent_ranges = _retake_conditioned_latent_ranges(
            pixel_start, pixel_end, num_frames, temporal_ratio
        )

        # ---- 3. Native VAE encode + seed initial latents ----------------
        pixel_shape = VideoPixelShape(
            batch=1, frames=num_frames, height=height, width=width, fps=fps
        )
        video_shape = VideoLatentShape.from_pixel_shape(
            pixel_shape, latent_channels=self.transformer_in_channels
        )
        # Source latents and sampled noise start in the model dtype, while the
        # blend/math path widens internally.
        source_window_latents = self._encode_video_window(source_norm_5d, _RETAKE_TILING_CONFIG)
        expected_latent_shape = tuple(video_shape.to_torch_shape())
        if tuple(source_window_latents.shape) != expected_latent_shape:
            raise ValueError(
                "LTX-2 native retake: encoded source latent shape "
                f"{tuple(source_window_latents.shape)} != expected "
                f"{expected_latent_shape}; check source resolution/frame count."
            )

        denoise_mask = self._build_denoise_mask(
            video_shape, cond_latent_frame_ranges=conditioned_latent_ranges
        )
        source_patch_latents = self.video_patchifier.patchify(source_window_latents)
        noise_latents = torch.randn(
            source_patch_latents.shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        # Denoising carries the latent in the model dtype and requantizes it at
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
        video_embeds, audio_embeds, connector_mask = self._prepare_prompt_conditioning(
            prompt, max_sequence_length, device
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
        audio_latent = self._encode_source_audio_latent(
            self._audio_encoder, video_path, pixel_shape, device, dtype
        )
        # The frozen audio follows a noise-to-clean trajectory on the same sigma
        # schedule as the video rather than remaining constant.
        if audio_latent is not None:
            audio_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
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

        # Prepare the resident FP8 transformer's own text cache while both
        # transformers are resident. If this request used live Gemma encoding,
        # offload Gemma after the text caches are built so the NVFP4 + FP8
        # transformers fit alongside the denoise activations. Cached prompts have
        # no text encoder to offload. Keep this condition tied to the resident FP8
        # transformer; NVFP4 by itself does not trigger offload.
        fp8_step_tf = self._fp8_step_transformer
        if fp8_step_tf is not None:
            self._fp8_step_text_cache = fp8_step_tf.prepare_text_cache(
                video_context=video_embeds,
                video_context_mask=connector_mask,
                video_positions=video_positions,
                audio_context=audio_embeds,
                audio_context_mask=connector_mask if audio_embeds is not None else None,
                audio_positions=audio_positions,
                dtype=dtype,
            )
            if getattr(self, "text_encoder", None) is not None:
                self.text_encoder.to("cpu")
                torch.cuda.empty_cache()

        # ---- 5. Native masked denoise (distilled, video-only, non-guided) --
        scheduler = NativeSchedulerAdapter()
        scheduler.sigmas = torch.tensor(
            _RETAKE_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=device
        )
        timesteps = scheduler.timesteps
        num_steps = len(timesteps)

        # Euler-integrate frozen audio from noise to its clean x0 target on the
        # video sigma schedule. This keeps cross-attention noise-matched without
        # running a separate transformer prediction for audio.
        audio_trajectory = None
        if audio_clean_latents is not None:
            sigmas = scheduler.sigmas.float()
            audio_diffusion_step = EulerDiffusionStep()
            clean_audio = audio_clean_latents.to(dtype)
            audio_noise = torch.randn(
                clean_audio.shape, generator=generator, device=device, dtype=dtype
            )
            audio_state = (audio_noise * sigmas[0] + clean_audio.float() * (1.0 - sigmas[0])).to(
                dtype
            )
            audio_trajectory = [audio_state]
            for step_index in range(num_steps):
                audio_state = audio_diffusion_step.step(
                    sample=audio_state,
                    denoised_sample=clean_audio,
                    sigmas=sigmas,
                    step_index=step_index,
                )
                audio_trajectory.append(audio_state)

        def retake_forward_fn(
            video_latents: torch.Tensor,
            _extra_stream_latents: dict[str, torch.Tensor] | None,
            step_index: int,
            timestep: torch.Tensor,
            _encoder_hidden_states: torch.Tensor,
            _extra_tensors: dict[str, Any],
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            audio_latents = audio_trajectory[step_index] if audio_trajectory is not None else None
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
            )
            return denoised_video, {}

        if timer is not None:
            timer.mark_denoise_start()
        denoised_latents = self.denoise(
            latents=latents,
            scheduler=scheduler,
            prompt_embeds=video_embeds,
            guidance_scale=1.0,
            forward_fn=retake_forward_fn,
            timesteps=timesteps,
        )

        # ---- 6. Native decode -------------------------------------------
        if timer is not None:
            timer.mark_post_start()
        video_latents_5d = self.video_patchifier.unpatchify(denoised_latents, video_shape).to(dtype)
        # Encode and decode must share the retake tiling geometry.
        chunks = list(
            self.video_decoder.tiled_decode(
                video_latents_5d, _RETAKE_TILING_CONFIG, generator=generator
            )
        )
        decoded = torch.cat(chunks, dim=2)  # (B, C, T, H, W)
        decoded = postprocess_video_tensor(decoded)  # (B, T, H, W, C) uint8
        output_video = decoded.contiguous()
        if timer is not None:
            timer.mark_end()

        # Source-audio read is host-side I/O and stays out of CUDA stage timing.
        source_audio = self._read_source_audio(video_path)

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
        frames = list(decode_video_by_frame(path=video_path, device="cpu", frame_cap=num_frames))
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

    def _read_source_audio(self, video_path: str) -> "AudioData | None":
        """Read the source audio unchanged (video-only retake preserves audio).

        PyAV performs the decode; torchaudio is only needed by the audio VAE
        mel front-end. Returns :class:`AudioData` or ``None`` when the source
        has no audio stream.
        """
        return decode_audio_from_file(video_path, "cpu")

    def _encode_source_audio_latent(
        self,
        audio_encoder: torch.nn.Module,
        video_path: str,
        pixel_shape: VideoPixelShape,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
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
        max_duration = pixel_shape.frames / pixel_shape.fps
        audio_in = decode_audio_from_file(video_path, device, 0.0, max_duration)
        if audio_in is None:
            return None
        latents = encode_audio(audio_in, audio_encoder).to(device, dtype)
        required_latent_frames = AudioLatentShape.from_video_pixel_shape(pixel_shape).frames
        return _conform_latent_length(latents, required_latent_frames)

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
    def _normalize_audio(audio: Any | None) -> tuple[torch.Tensor | None, int | None]:
        if audio is None:
            return None, None
        waveform = getattr(audio, "samples", audio)
        sample_rate = getattr(audio, "sample_rate", None)
        if waveform is None:
            return None, sample_rate
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
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
