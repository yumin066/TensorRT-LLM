# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional LTX-2 retake workflow for VisualGen.

This pipeline keeps a resident retake model for ``trtllm-serve``. It is being
migrated onto the native TensorRT-LLM LTX-2 transformer (``LTXModel``): the
native transformer is now built and weight-loaded here (via
``build_ltx2_transformer`` and the native transformer loader) so retake can
inherit the config-driven acceleration stack. The upstream Lightricks
``ltx_pipelines.retake.RetakePipeline`` is still used for retake pre/post
(source decode, temporal-region mask, composite) until that logic is ported
natively.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any, Optional

import torch

from tensorrt_llm._torch.visual_gen.output import PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline, ExtraParamSchema
from tensorrt_llm.logger import logger

from .pipeline_ltx2 import LTX2Pipeline, _load_ltx2_transformer_weights, build_ltx2_transformer


class LTX2RetakePipeline(BasePipeline):
    """Persistent VisualGen adapter for LTX-2 retake requests."""

    def __init__(self, pipeline_config):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._retake_pipeline = None
        self._retake_params = None
        self._tiling_config = None
        self._get_videostream_metadata = None
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
                default=True,
                description="Regenerate audio inside the retake window when source audio exists.",
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
            self.transformer.load_weights(transformer_weights)

    def post_load_weights(self) -> None:
        # Finalize the native transformer (e.g. dynamic-quant weight loading)
        # when it exposes a post-load hook.
        if self.transformer is not None and hasattr(self.transformer, "post_load_weights"):
            self.transformer.post_load_weights()

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

        if self._retake_pipeline is None:
            raise RuntimeError("LTX-2 retake pipeline has not been loaded.")

        prompt = self._single_prompt(req.prompt)
        retake_start = time.perf_counter()
        metadata = self._get_videostream_metadata(video_path)
        video_iter, audio = self._retake_pipeline(
            video_path=video_path,
            prompt=prompt,
            start_time=start_time,
            end_time=end_time,
            seed=req.params.seed,
            negative_prompt=req.params.negative_prompt or "",
            num_inference_steps=req.params.num_inference_steps,
            video_guider_params=getattr(self._retake_params, "video_guider_params", None),
            audio_guider_params=getattr(self._retake_params, "audio_guider_params", None),
            regenerate_video=extra["retake_regenerate_video"],
            regenerate_audio=extra["retake_regenerate_audio"],
            enhance_prompt=extra["retake_enhance_prompt"],
            tiling_config=self._tiling_config,
            max_batch_size=extra["retake_max_batch_size"],
        )
        video = self._materialize_video(video_iter)
        audio_tensor, sample_rate = self._normalize_audio(audio)
        elapsed = time.perf_counter() - retake_start
        return PipelineOutput(
            video=video,
            audio=audio_tensor,
            frame_rate=float(metadata.fps),
            audio_sample_rate=sample_rate,
            denoise=elapsed,
        )

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
