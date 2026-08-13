#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""End-to-end example for the native LTX-2 retake workflow.

Regenerates an internal time-window ``[start, end)`` of a source video and emits
the whole VAE-decoded retake clip. It drives the native
:class:`LTX2RetakePipeline` in one process: source read -> VAE encode ->
two-sided masked denoise -> VAE decode.

Source-video metadata, frame decode, and audio decode use the native PyAV media
readers in ``ltx2_core.media_io``; no external LTX pipeline is required.

Example::

    : "${RETAKE_START:?set RETAKE_START to the retake-window start in seconds}"
    : "${RETAKE_END:?set RETAKE_END to the retake-window end in seconds}"

    python examples/visual_gen/ltx2_retake_e2e.py \\
        --checkpoint /path/to/ltx-2.3-22b-distilled.safetensors \\
        --text-encoder /path/to/gemma-3-12b-it \\
        --source retake_input.mp4 --output retake_output.mp4 \\
        --start "$RETAKE_START" --end "$RETAKE_END" \\
        --prompt "a person talking to the camera, natural head motion, clear speech"

``--start`` and ``--end`` intentionally have no defaults: every caller must
provide the window associated with its own ``retake_input.mp4``.

Note: the full-resolution 22B retake (source + Gemma text encoder + 22B
transformer resident) needs a large-memory GPU (e.g. H200 141GB); an 80GB GPU
can OOM at full resolution.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the native LTX-2 retake workflow end-to-end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True, help="LTX-2 checkpoint (.safetensors or dir)."
    )
    parser.add_argument(
        "--text-encoder",
        required=True,
        help="Gemma3 text encoder directory (config.json + model*.safetensors + tokenizer).",
    )
    parser.add_argument("--source", required=True, help="Source video to retake (.mp4).")
    parser.add_argument("--output", required=True, help="Output decoded retake video (.mp4).")
    parser.add_argument(
        "--dump-frames",
        default=None,
        help=(
            "Optional path to save the decoded uint8 frames (1, T, H, W, C) as a "
            ".pt tensor, before H.264 encoding. Quality evaluation should score "
            "these rather than a decoded mp4, since the codec's own re-encode loss "
            "is larger than the differences being measured."
        ),
    )
    parser.add_argument(
        "--start",
        type=float,
        required=True,
        metavar="SECONDS",
        help="Required start time of the regenerated window; no default is used.",
    )
    parser.add_argument(
        "--end",
        type=float,
        required=True,
        metavar="SECONDS",
        help="Required end time of the regenerated window; no default is used.",
    )
    parser.add_argument(
        "--prompt",
        default="a person talking to the camera, natural head motion, clear speech",
        help="Text prompt for the regenerated window.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=8,
        choices=(8,),
        help="Denoise steps; native retake uses the distilled 8-step schedule.",
    )
    parser.add_argument(
        "--quant-algo",
        default=None,
        choices=("FP8", "NVFP4"),
        help=(
            "Optional runtime weight-quantization recipe for the transformer linears. "
            "Omit for bf16; FP8/NVFP4 use the native dynamic-quant path. NVFP4 requires "
            "an NVFP4-capable GPU (e.g. RTX PRO 6000 / Blackwell)."
        ),
    )
    parser.add_argument(
        "--fp8-linear-step",
        type=int,
        action="append",
        default=None,
        metavar="STEP",
        help=(
            "Run the transformer in FP8 at this diffusion step index instead of the "
            "primary quant. Repeatable; e.g. --fp8-linear-step 4 "
            "--fp8-linear-step 7. Requires --quant-algo NVFP4; a resident FP8 transformer "
            "is built and Gemma is offloaded to CPU to fit both models."
        ),
    )
    parser.add_argument(
        "--nvfp4-attn",
        action="store_true",
        help=(
            "Use the SM120 FlashInfer NVFP4 video-self attention (fastest) instead "
            "of BF16 SDPA. Lossier on its own; pair with --fp8-linear-step 4 "
            "--fp8-linear-step 7 to recover quality."
        ),
    )
    parser.add_argument(
        "--lora",
        default=None,
        help="Optional identity/style LoRA fused into the base transformer weights.",
    )
    return parser.parse_args()


def _build_retake_pipeline(
    checkpoint: str,
    text_encoder: str,
    device: torch.device,
    lora,
    quant_algo=None,
    fp8_linear_steps=None,
    nvfp4_attn=False,
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
        "retake_lora_strength": 1.0,
    }
    if fp8_linear_steps:
        extra["retake_fp8_linear_steps"] = sorted(set(int(s) for s in fp8_linear_steps))
    model_config_kwargs = {"pretrained_config": SimpleNamespace(**transformer_cfg)}
    pipeline_config_kwargs = {"extra_attrs": extra}
    if quant_algo:
        # Native runtime dynamic-quant path: the quant config is threaded onto
        # both the transformer model config (where DynamicLinearWeightLoader keys
        # on ``dynamic_weight_quant``) and the pipeline config, and
        # ``force_dynamic_quantization`` computes the activation scale live per
        # forward (the fix for the uninitialized static-scale NVFP4 collapse).
        quant_config, _layer, dynamic_weight_quant, _dyn_act = (
            DiffusionPipelineConfig.load_diffusion_quant_config(
                {"quant_algo": quant_algo, "dynamic": True}
            )
        )
        model_config_kwargs["quant_config"] = quant_config
        model_config_kwargs["dynamic_weight_quant"] = dynamic_weight_quant
        # force_dynamic_quantization must be set on the MODEL config: the
        # transformer's _make_linear reads model_config.force_dynamic_quantization
        # to build each Linear with a live activation amax. Without it NVFP4 uses
        # an uninitialized static activation scale and the AdaLN/timestep_embedder
        # linears produce NaN (the whole regenerated window decodes to black).
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
    pipe.load_standard_components(checkpoint, device, text_encoder_path=text_encoder)
    pipe.load_weights(pipe.load_transformer_weights(checkpoint))
    pipe.post_load_weights()
    if nvfp4_attn:
        # Enable the SM120 FlashInfer NVFP4 video-self attention on every
        # LTX2Attention module (both the primary and the FP8-step transformer) for
        # the fastest recipe. The attention forward reads the kernel off these
        # class flags; only the video-self-attn role actually takes the path.
        import flashinfer

        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTX2Attention

        LTX2Attention._nvfp4_fi = flashinfer
        LTX2Attention._nvfp4_attn = True
    return pipe


def _make_request(args: argparse.Namespace) -> SimpleNamespace:
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


def _to_stereo_int16(audio: torch.Tensor, max_samples: int | None = None) -> "np.ndarray":
    """Convert a float waveform to interleaved stereo int16 for AAC.

    Accepts ``(B, C, S)`` (single batch only) or ``(C, S)``. Mono is duplicated
    to both channels; more than two channels is an error rather than a silent
    channel drop, because losing a channel is invisible in the output file.

    *max_samples* trims the waveform to the encoded video's duration.
    """
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

    # Scale by 32767 and clamp: the waveform is nominally [-1, 1] but decoded
    # float samples can exceed it slightly, and wrapping would be an audible click.
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
    """Encode a ``(1, T, H, W, C)`` uint8 RGB video to H.264 via PyAV.

    When *audio* is supplied it is muxed as AAC alongside the video. The retake
    workflow is video-only: it regenerates a time window and passes the source
    audio through unchanged, so dropping it here silently shipped a mute clip.

    Colour conversion and stream metadata use BT.709 limited range explicitly.
    ``Fraction`` preserves rates such as 30000/1001, and fixed encoder settings
    make repeated writeouts comparable.
    """
    from fractions import Fraction

    import av

    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.color_conversion import (
        AVCOL_RANGE_MPEG,
        AVCOL_SPC_BT709,
        rgb_to_yuv420p_bt709_limited,
    )

    rgb = video[0].cpu()  # (T, H, W, C) uint8
    height, width = int(rgb.shape[1]), int(rgb.shape[2])
    i420 = rgb_to_yuv420p_bt709_limited(rgb).numpy()  # (T, H*3//2, W) uint8

    # limit_denominator keeps exact rates exact (30 -> 30/1) while still
    # representing 29.97 as 30000/1001 rather than as a float.
    rate = Fraction(frame_rate).limit_denominator(1000000)

    num_frames = int(rgb.shape[0])

    container = av.open(output_path, mode="w")
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
    """Encode *audio* into *audio_stream*, trimmed to the encoded video duration.

    Source audio may outlast the decoded video, so it is trimmed before muxing.
    """
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

    # AAC does not accept packed s16 directly, and it has a fixed frame size, so
    # the resampler both converts the sample format and splits the waveform into
    # encoder-sized frames with correct timestamps.
    resampler = av.audio.resampler.AudioResampler(
        format=audio_stream.format, layout=audio_stream.layout, rate=sample_rate
    )
    for resampled in resampler.resample(frame):
        for packet in audio_stream.encode(resampled):
            container.mux(packet)
    for packet in audio_stream.encode():
        container.mux(packet)


def main() -> None:
    args = _parse_args()
    if args.start >= args.end:
        raise ValueError(f"--start ({args.start}) must be < --end ({args.end}).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    recipe = args.quant_algo or "bf16"
    if args.fp8_linear_step:
        recipe += f"+fp8@{sorted(set(args.fp8_linear_step))}"
    if args.nvfp4_attn:
        recipe += "+nvfp4attn"
    print(f"[retake] building native retake pipeline ({recipe})...", flush=True)
    pipe = _build_retake_pipeline(
        args.checkpoint,
        args.text_encoder,
        device,
        args.lora,
        args.quant_algo,
        fp8_linear_steps=args.fp8_linear_step,
        nvfp4_attn=args.nvfp4_attn,
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

    if args.dump_frames:
        torch.save({"video": video.cpu(), "frame_rate": float(out.frame_rate)}, args.dump_frames)
        print(f"[retake] dumped pre-encode frames to {args.dump_frames}", flush=True)

    num_frames = _save_video(
        video,
        float(out.frame_rate),
        args.output,
        audio=out.audio,
        audio_sample_rate=out.audio_sample_rate,
    )
    has_audio = out.audio is not None and out.audio_sample_rate
    print(
        f"[retake] saved {args.output} ({num_frames} frames, "
        f"audio={'yes @ ' + str(out.audio_sample_rate) + ' Hz' if has_audio else 'none'})",
        flush=True,
    )


if __name__ == "__main__":
    main()
