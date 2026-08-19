# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python entry points for the native LTX-2 retake workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


@dataclass
class LTX2RetakeResult:
    output_path: str
    num_frames: int
    frame_rate: float
    has_audio: bool
    audio_sample_rate: int | None
    stage_timings: dict[str, float] | None


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
    prompt_conditioning_cache: str | None = None,
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
    if start >= end:
        raise ValueError(f"start ({start}) must be < end ({end}).")
    if num_inference_steps != 8:
        raise ValueError(
            "native LTX-2 retake currently supports the distilled 8-step schedule only"
        )

    prompt = prompt or _default_retake_prompt()
    device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    args = SimpleNamespace(
        checkpoint=checkpoint,
        source=source,
        output=output,
        start=start,
        end=end,
        prompt=prompt,
        seed=seed,
        text_encoder=text_encoder,
        prompt_conditioning_cache=prompt_conditioning_cache,
        lora=lora,
        lora_strength=lora_strength,
        quant_algo=quant_algo,
        fp8_linear_step=list(fp8_linear_steps or []),
        nvfp4_attn=nvfp4_attn,
        num_inference_steps=num_inference_steps,
    )

    prompt_conditioning = _resolve_prompt_conditioning(args)
    recipe = quant_algo or "bf16"
    if args.fp8_linear_step:
        recipe += f"+fp8@{sorted(set(args.fp8_linear_step))}"
    if nvfp4_attn:
        recipe += "+nvfp4attn"
    if lora:
        recipe += f"+lora@{lora_strength}"
    print(f"[retake] building native retake pipeline ({recipe})...", flush=True)
    pipe = _build_retake_pipeline(
        checkpoint,
        text_encoder,
        device,
        lora,
        lora_strength=lora_strength,
        prompt_conditioning=prompt_conditioning,
        quant_algo=quant_algo,
        fp8_linear_steps=args.fp8_linear_step,
        nvfp4_attn=nvfp4_attn,
    )

    print("[retake] loaded; running infer...", flush=True)
    out = pipe.infer(_make_request(args))
    video = out.video
    if video is None or video.dim() != 5:
        raise RuntimeError("retake pipeline did not produce a video tensor")
    print(
        f"[retake] video={tuple(video.shape)} fps={out.frame_rate} "
        f"stage_timings={out.stage_timings}",
        flush=True,
    )

    if dump_frames:
        torch.save({"video": video.cpu(), "frame_rate": float(out.frame_rate)}, dump_frames)
        print(f"[retake] dumped pre-encode frames to {dump_frames}", flush=True)

    num_frames = _save_video(
        video,
        float(out.frame_rate),
        output,
        audio=out.audio,
        audio_sample_rate=out.audio_sample_rate,
    )
    has_audio = out.audio is not None and out.audio_sample_rate is not None
    print(
        f"[retake] saved {output} ({num_frames} frames, "
        f"audio={'yes @ ' + str(out.audio_sample_rate) + ' Hz' if has_audio else 'none'})",
        flush=True,
    )
    return LTX2RetakeResult(
        output_path=output,
        num_frames=num_frames,
        frame_rate=float(out.frame_rate),
        has_audio=bool(has_audio),
        audio_sample_rate=out.audio_sample_rate,
        stage_timings=out.stage_timings,
    )


def _default_retake_prompt() -> str:
    from tensorrt_llm._torch.visual_gen.models.ltx2.retake_prompt_conditioning import (
        DEFAULT_RETAKE_PROMPT,
    )

    return DEFAULT_RETAKE_PROMPT


def _resolve_prompt_conditioning(args):
    """Select cached default conditioning or the live Gemma fallback."""
    from safetensors import SafetensorError

    from tensorrt_llm._torch.visual_gen.models.ltx2.retake_prompt_conditioning import (
        DEFAULT_RETAKE_MAX_SEQUENCE_LENGTH,
        checkpoint_fingerprint,
        is_default_retake_prompt,
        load_retake_prompt_conditioning,
    )

    if is_default_retake_prompt(args.prompt) and args.prompt_conditioning_cache:
        cache_path = Path(args.prompt_conditioning_cache)
        if cache_path.is_file():
            try:
                print(f"[retake] validating prompt conditioning: {cache_path}", flush=True)
                conditioning = load_retake_prompt_conditioning(
                    cache_path,
                    expected_prompt=args.prompt,
                    expected_max_sequence_length=DEFAULT_RETAKE_MAX_SEQUENCE_LENGTH,
                    expected_checkpoint_fingerprint=checkpoint_fingerprint(args.checkpoint),
                )
                print(f"[retake] using precomputed prompt conditioning: {cache_path}", flush=True)
                return conditioning
            except (OSError, SafetensorError, ValueError) as error:
                if not args.text_encoder:
                    raise RuntimeError(
                        "The default-prompt conditioning cache is invalid and no "
                        "text_encoder was provided for fallback."
                    ) from error
                print(
                    "[retake] WARNING: prompt-conditioning cache is unusable; "
                    f"falling back to Gemma ({error})",
                    flush=True,
                )
        elif not args.text_encoder:
            raise FileNotFoundError(
                "The default-prompt conditioning cache is missing and no text_encoder "
                f"was provided for fallback: {cache_path}"
            )

    if not args.text_encoder:
        raise ValueError(
            "text_encoder is required when the prompt is not served by a valid "
            "prompt-conditioning cache."
        )
    text_encoder_path = Path(args.text_encoder)
    if not text_encoder_path.is_dir():
        raise FileNotFoundError(f"Gemma text encoder directory does not exist: {text_encoder_path}")
    print(f"[retake] using live Gemma text encoding: {text_encoder_path}", flush=True)
    return None


def _build_retake_pipeline(
    checkpoint: str,
    text_encoder: str | None,
    device: torch.device,
    lora: str | None,
    *,
    lora_strength: float = 1.0,
    prompt_conditioning=None,
    quant_algo: str | None = None,
    fp8_linear_steps: list[int] | tuple[int, ...] | None = None,
    nvfp4_attn: bool = False,
):
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, DiffusionPipelineConfig
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import (
        _find_safetensors_files,
        _read_safetensors_config,
    )
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_retake import LTX2RetakePipeline

    cfg = _read_safetensors_config(_find_safetensors_files(checkpoint)[0])
    transformer_cfg = cfg.get("transformer", cfg)
    extra = {
        "workflow": "retake",
        "retake_lora_path": lora,
        "retake_lora_strength": lora_strength,
    }
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
        prompt_conditioning=prompt_conditioning,
    )
    pipe.load_weights(pipe.load_transformer_weights(checkpoint))
    pipe.post_load_weights()
    if nvfp4_attn:
        import flashinfer

        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTX2Attention

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


def _to_stereo_int16(audio: torch.Tensor, max_samples: int | None = None) -> np.ndarray:
    """Convert a float waveform to interleaved stereo int16 for AAC."""
    if audio.dim() == 3:
        if audio.shape[0] != 1:
            raise ValueError(f"retake writeout supports one audio batch; got {audio.shape[0]}")
        audio = audio[0]
    if audio.dim() != 2:
        raise ValueError(f"audio must be (B, C, S) or (C, S); got {tuple(audio.shape)}")

    channels = audio.shape[0]
    if channels == 1:
        audio = audio.expand(2, -1)
    elif channels != 2:
        raise ValueError(f"retake writeout supports mono or stereo audio; got {channels} channels")

    if max_samples is not None:
        audio = audio[:, :max_samples]

    samples = audio.detach().to(torch.float32).clamp(-1.0, 1.0).cpu().numpy()
    quantized = (samples * 32767.0).round().astype(np.int16)

    interleaved = np.empty((1, quantized.shape[1] * 2), dtype=np.int16)
    interleaved[0, 0::2] = quantized[0]
    interleaved[0, 1::2] = quantized[1]
    return interleaved


def _save_video(
    video: torch.Tensor,
    frame_rate: float,
    output_path: str,
    audio: torch.Tensor | None = None,
    audio_sample_rate: int | None = None,
) -> int:
    """Encode a ``(1, T, H, W, C)`` uint8 RGB video to H.264 via PyAV."""
    from fractions import Fraction

    import av

    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.color_conversion import (
        AVCOL_RANGE_MPEG,
        AVCOL_SPC_BT709,
        rgb_to_yuv420p_bt709_limited,
    )

    rgb = video[0].cpu()
    height, width = int(rgb.shape[1]), int(rgb.shape[2])
    i420 = rgb_to_yuv420p_bt709_limited(rgb).numpy()
    rate = Fraction(frame_rate).limit_denominator(1000000)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    num_frames = int(rgb.shape[0])

    container = av.open(str(output), mode="w")
    try:
        stream = container.add_stream(
            "libx264", rate=rate, options={"crf": "19", "preset": "veryfast"}
        )
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.codec_context.colorspace = AVCOL_SPC_BT709
        stream.codec_context.color_range = AVCOL_RANGE_MPEG

        if audio is not None and not audio_sample_rate:
            raise ValueError("audio_sample_rate is required when audio is provided")
        audio_stream = None
        if audio is not None:
            audio_stream = container.add_stream("aac", rate=int(audio_sample_rate))
            audio_stream.codec_context.time_base = Fraction(1, int(audio_sample_rate))

        for frame in i420:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="yuv420p")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

        if audio_stream is not None:
            _mux_audio(container, audio_stream, audio, int(audio_sample_rate), num_frames, rate)
    finally:
        container.close()
    return num_frames


def _mux_audio(container, audio_stream, audio, sample_rate: int, num_frames: int, rate) -> None:
    """Encode ``audio`` into ``audio_stream``, trimmed to video duration."""
    from fractions import Fraction

    import av

    max_samples = int(round(num_frames / float(rate) * sample_rate))
    interleaved = _to_stereo_int16(audio, max_samples=max_samples)
    if interleaved.shape[1] == 0:
        return

    frame = av.AudioFrame.from_ndarray(interleaved, format="s16", layout="stereo")
    frame.sample_rate = sample_rate
    frame.time_base = Fraction(1, sample_rate)
    frame.pts = 0

    resampler = av.audio.resampler.AudioResampler(
        format=audio_stream.format, layout=audio_stream.layout, rate=sample_rate
    )
    for resampled in resampler.resample(frame):
        for packet in audio_stream.encode(resampled):
            container.mux(packet)
    for packet in audio_stream.encode():
        container.mux(packet)
