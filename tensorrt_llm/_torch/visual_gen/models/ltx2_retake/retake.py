# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python entry points for the native LTX-2 retake workflow."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from tensorrt_llm.media.encoding import save_video


@dataclass
class LTX2RetakeResult:
    output_path: str
    num_frames: int
    frame_rate: float
    has_audio: bool
    audio_sample_rate: int | None
    stage_timings: dict[str, float]


class LTX2RetakeEngine:
    """Persistent native LTX-2 retake engine.

    Build this once with :meth:`from_pretrained`, then call :meth:`retake` for
    warmup and measured requests. The model, LoRA, quantization recipe, prompt
    conditioning, and native retake pipeline remain resident across requests.
    """

    def __init__(self, pipe, *, prompt: str, num_inference_steps: int = 8):
        if num_inference_steps != 8:
            raise ValueError(
                "native LTX-2 retake currently supports the distilled 8-step schedule only"
            )
        self._pipe = pipe
        self._prompt = prompt
        self._num_inference_steps = num_inference_steps

    @classmethod
    def from_pretrained(
        cls,
        *,
        checkpoint: str,
        prompt: str | None = None,
        text_encoder: str | None = None,
        prompt_conditioning_path: str | None = None,
        lora: str | None = None,
        lora_strength: float = 1.0,
        quant_algo: str | None = None,
        nvfp4_attn: bool = False,
        fp8_linear_steps: list[int] | tuple[int, ...] | None = None,
        num_inference_steps: int = 8,
        device: torch.device | str | None = None,
    ) -> "LTX2RetakeEngine":
        if num_inference_steps != 8:
            raise ValueError(
                "native LTX-2 retake currently supports the distilled 8-step schedule only"
            )

        prompt = prompt or ""
        device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        _validate_prompt_source(
            prompt=prompt,
            text_encoder=text_encoder,
            prompt_conditioning_path=prompt_conditioning_path,
        )
        recipe = _recipe_label(quant_algo, fp8_linear_steps, nvfp4_attn, lora, lora_strength)
        print(f"[retake] building persistent native retake pipeline ({recipe})...", flush=True)
        pipe = _build_retake_pipeline(
            checkpoint,
            text_encoder,
            device,
            lora,
            lora_strength=lora_strength,
            prompt_conditioning_path=prompt_conditioning_path,
            quant_algo=quant_algo,
            fp8_linear_steps=fp8_linear_steps,
            nvfp4_attn=nvfp4_attn,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("[retake] persistent pipeline ready", flush=True)
        return cls(pipe, prompt=prompt, num_inference_steps=num_inference_steps)

    def retake(
        self,
        *,
        source: str,
        output: str,
        start: float,
        end: float,
        prompt: str | None = None,
        seed: int = 42,
        num_inference_steps: int | None = None,
        dump_frames: str | None = None,
    ) -> LTX2RetakeResult:
        """Run one hot retake request and write the full retake clip to ``output``."""
        if start >= end:
            raise ValueError(f"start ({start}) must be < end ({end}).")
        steps = self._num_inference_steps if num_inference_steps is None else num_inference_steps
        if steps != 8:
            raise ValueError(
                "native LTX-2 retake currently supports the distilled 8-step schedule only"
            )

        args = SimpleNamespace(
            source=source,
            start=start,
            end=end,
            prompt=prompt if prompt is not None else self._prompt,
            seed=seed,
            num_inference_steps=steps,
        )
        print("[retake] running infer...", flush=True)
        infer_t0 = time.perf_counter()
        out = self._pipe.infer(_make_request(args))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_wall = time.perf_counter() - infer_t0

        video = out.video
        if video is None or video.dim() != 5:
            raise RuntimeError("retake pipeline did not produce a video tensor")
        pipeline_timings = {
            "pre_denoise": out.pre_denoise,
            "denoise": out.denoise,
            "post_denoise": out.post_denoise,
        }
        print(
            f"[retake] video={tuple(video.shape)} fps={out.frame_rate} "
            f"stage_timings={pipeline_timings}",
            flush=True,
        )

        if dump_frames:
            torch.save({"video": video.cpu(), "frame_rate": float(out.frame_rate)}, dump_frames)
            print(f"[retake] dumped pre-encode frames to {dump_frames}", flush=True)

        encode_t0 = time.perf_counter()
        num_frames = int(video.shape[1])
        save_video(
            video,
            output,
            audio=out.audio,
            frame_rate=float(out.frame_rate),
            format="mp4",
            audio_sample_rate=out.audio_sample_rate or 24000,
        )
        mp4_encode_mux = time.perf_counter() - encode_t0
        has_audio = out.audio is not None and out.audio_sample_rate is not None
        print(
            f"[retake] saved {output} ({num_frames} frames, "
            f"audio={'yes @ ' + str(out.audio_sample_rate) + ' Hz' if has_audio else 'none'})",
            flush=True,
        )
        stage_timings = {
            **pipeline_timings,
            "infer_wall": infer_wall,
            "mp4_encode_mux": mp4_encode_mux,
        }
        return LTX2RetakeResult(
            output_path=output,
            num_frames=num_frames,
            frame_rate=float(out.frame_rate),
            has_audio=bool(has_audio),
            audio_sample_rate=out.audio_sample_rate,
            stage_timings=stage_timings,
        )


def run_ltx2_retake(
    *,
    checkpoint: str,
    source: str,
    output: str,
    start: float,
    end: float,
    prompt: str | None = None,
    seed: int = 42,
    text_encoder: str | None = None,
    prompt_conditioning_path: str | None = None,
    lora: str | None = None,
    lora_strength: float = 1.0,
    quant_algo: str | None = None,
    nvfp4_attn: bool = False,
    fp8_linear_steps: list[int] | tuple[int, ...] | None = None,
    num_inference_steps: int = 8,
    dump_frames: str | None = None,
    device: torch.device | str | None = None,
) -> LTX2RetakeResult:
    """Run native LTX-2 retake and write the full retake clip to ``output``.

    ``source`` is the retake input clip. ``start`` and ``end`` are local
    timestamps within that clip for the regenerated window.
    """
    trtllm_engine = LTX2RetakeEngine.from_pretrained(
        checkpoint=checkpoint,
        prompt=prompt,
        text_encoder=text_encoder,
        prompt_conditioning_path=prompt_conditioning_path,
        lora=lora,
        lora_strength=lora_strength,
        quant_algo=quant_algo,
        fp8_linear_steps=list(fp8_linear_steps or []),
        nvfp4_attn=nvfp4_attn,
        num_inference_steps=num_inference_steps,
        device=device,
    )
    return trtllm_engine.retake(
        source=source,
        output=output,
        start=start,
        end=end,
        prompt=prompt,
        seed=seed,
        num_inference_steps=num_inference_steps,
        dump_frames=dump_frames,
    )


def _recipe_label(
    quant_algo: str | None,
    fp8_linear_steps: list[int] | tuple[int, ...] | None,
    nvfp4_attn: bool,
    lora: str | None,
    lora_strength: float,
) -> str:
    recipe = quant_algo or "bf16"
    if fp8_linear_steps:
        recipe += f"+fp8@{sorted(set(fp8_linear_steps))}"
    if nvfp4_attn:
        recipe += "+nvfp4attn"
    if lora:
        recipe += f"+lora@{lora_strength}"
    return recipe


def _validate_prompt_source(
    *,
    prompt: str,
    text_encoder: str | None,
    prompt_conditioning_path: str | None,
) -> None:
    """Validate the mutually exclusive text and precomputed-conditioning paths."""
    if prompt_conditioning_path:
        if prompt:
            raise ValueError(
                "prompt and prompt_conditioning_path are mutually exclusive; "
                "leave prompt empty when supplying precomputed conditioning."
            )
        if text_encoder:
            raise ValueError("text_encoder and prompt_conditioning_path are mutually exclusive.")
        conditioning_path = Path(prompt_conditioning_path)
        if not conditioning_path.is_file():
            raise FileNotFoundError(f"Prompt-conditioning file does not exist: {conditioning_path}")
        print(f"[retake] using precomputed prompt conditioning: {conditioning_path}", flush=True)
        return

    if not text_encoder:
        raise ValueError("text_encoder is required when prompt_conditioning_path is not provided.")
    text_encoder_path = Path(text_encoder)
    if not text_encoder_path.is_dir():
        raise FileNotFoundError(f"Gemma text encoder directory does not exist: {text_encoder_path}")
    print(f"[retake] using live Gemma text encoding: {text_encoder_path}", flush=True)


def _build_retake_pipeline(
    checkpoint: str,
    text_encoder: str | None,
    device: torch.device,
    lora: str | None,
    *,
    lora_strength: float = 1.0,
    prompt_conditioning_path: str | None = None,
    quant_algo: str | None = None,
    fp8_linear_steps: list[int] | tuple[int, ...] | None = None,
    nvfp4_attn: bool = False,
):
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, DiffusionPipelineConfig
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import (
        _find_safetensors_files,
        _read_safetensors_config,
    )
    from tensorrt_llm._torch.visual_gen.models.ltx2_retake.pipeline_ltx2_retake import (
        LTX2RetakePipeline,
    )

    cfg = _read_safetensors_config(_find_safetensors_files(checkpoint)[0])
    transformer_cfg = cfg.get("transformer", cfg)
    extra = {
        "retake_lora_path": lora,
        "retake_lora_strength": lora_strength,
    }
    if prompt_conditioning_path:
        extra["retake_prompt_conditioning_path"] = prompt_conditioning_path
    if fp8_linear_steps:
        extra["retake_fp8_linear_steps"] = sorted(set(int(s) for s in fp8_linear_steps))
    model_config_kwargs = {"pretrained_config": SimpleNamespace(**transformer_cfg)}
    pipeline_config_kwargs = {"extra_attrs": extra}
    if quant_algo:
        quant_config, _layer, dynamic_weight_quant, _dyn_act = (
            DiffusionPipelineConfig.load_diffusion_quant_config(
                {"quant_algo": quant_algo, "dynamic": True}
            )
        )
        model_config_kwargs["quant_config"] = quant_config
        model_config_kwargs["dynamic_weight_quant"] = dynamic_weight_quant
        model_config_kwargs["force_dynamic_quantization"] = True
        pipeline_config_kwargs["quant_config"] = quant_config
        pipeline_config_kwargs["dynamic_weight_quant"] = dynamic_weight_quant
        pipeline_config_kwargs["force_dynamic_quantization"] = True

    pipeline_config = DiffusionPipelineConfig(
        model_configs={"transformer": DiffusionModelConfig(**model_config_kwargs)},
        **pipeline_config_kwargs,
    )
    pipeline_config.cuda_graph.enable = False

    pipe = LTX2RetakePipeline(pipeline_config)
    pipe.load_standard_components(
        checkpoint,
        device,
        text_encoder_path=text_encoder or "",
    )
    pipe.load_weights(pipe.load_transformer_weights(checkpoint))
    pipe.post_load_weights()
    if nvfp4_attn:
        import flashinfer

        from tensorrt_llm._torch.visual_gen.models.ltx2_retake.transformer_ltx2_retake import (
            LTX2Attention,
        )

        LTX2Attention._nvfp4_fi = flashinfer
        LTX2Attention._nvfp4_attn = True
    return pipe


def _make_request(args) -> SimpleNamespace:
    return SimpleNamespace(
        prompt=args.prompt,
        negative_prompt=None,
        params=SimpleNamespace(
            extra_params={
                "retake_video_path": args.source,
                "retake_start_time": args.start,
                "retake_end_time": args.end,
            },
            negative_prompt=None,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
        ),
    )
