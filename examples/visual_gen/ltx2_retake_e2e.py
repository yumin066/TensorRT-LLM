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

Regenerates an internal time-window ``[start, end)`` of a source video and
composites it back so the frames outside the window stay byte-identical to the
source. Unlike the external step_a/step_b/step_c pipeline, this drives the
native :class:`LTX2RetakePipeline` in a single process: source read -> VAE
encode -> two-sided masked denoise -> VAE decode -> composite.

The source-video read uses the optional Lightricks ``ltx-pipelines``
``media_io`` helpers (the pipeline imports them lazily); install that package in
the VisualGen runtime before running.

Example::

    python examples/visual_gen/ltx2_retake_e2e.py \\
        --checkpoint /path/to/ltx-2.3-22b-distilled.safetensors \\
        --text-encoder /path/to/gemma-3-12b-it \\
        --source retake_input.mp4 --output retake_output.mp4 \\
        --start 3.0 --end 3.9667 \\
        --prompt "a person talking to the camera, natural head motion, clear speech"

Note: the full-resolution 22B retake (source + Gemma text encoder + 22B
transformer resident) needs a large-memory GPU (e.g. H200 141GB); an 80GB GPU
can OOM at full resolution.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

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
    parser.add_argument("--output", required=True, help="Output composited retake video (.mp4).")
    parser.add_argument(
        "--start", type=float, required=True, help="Start time (seconds) of the regenerated window."
    )
    parser.add_argument(
        "--end", type=float, required=True, help="End time (seconds) of the regenerated window."
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
        help="Denoise steps (native retake runs the distilled 8-step schedule).",
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
            "primary quant (M034 fp8-linear-step). Repeatable; e.g. --fp8-linear-step 4 "
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
            "--fp8-linear-step 7 to recover quality (the M034 max-speed recipe)."
        ),
    )
    parser.add_argument(
        "--upstream-stage",
        action="store_true",
        help=(
            "Use the preserved upstream DiffusionStage oracle path (step_a-identical "
            "video+audio latents via ltx_pipelines helpers, native transformer via the "
            "adapter, frozen source audio conditioning the video denoise) instead of the "
            "native masked-denoise path. For bf16 parity verification against step_a/b/c."
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
    upstream_stage=False,
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
        "retake_distilled": True,
        "retake_offload_mode": "none",
        "retake_lora_path": lora,
        "retake_lora_strength": 1.0,
    }
    if fp8_linear_steps:
        extra["retake_fp8_linear_steps"] = sorted(set(int(s) for s in fp8_linear_steps))
    if upstream_stage:
        extra["retake_use_upstream_stage"] = True

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
                "retake_regenerate_video": True,
                "retake_regenerate_audio": False,
                "retake_enhance_prompt": False,
                "retake_max_batch_size": 1,
            },
            negative_prompt=None,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
        ),
    )


def _save_video(video: torch.Tensor, frame_rate: float, output_path: str) -> int:
    """Encode a ``(1, T, H, W, C)`` uint8 RGB video to H.264 via PyAV."""
    import av

    frames = video[0].cpu().to(torch.uint8).numpy()  # (T, H, W, C)
    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(round(frame_rate)))
    stream.width, stream.height, stream.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
    for frame in frames:
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return frames.shape[0]


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
        upstream_stage=args.upstream_stage,
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

    num_frames = _save_video(video, float(out.frame_rate), args.output)
    print(f"[retake] saved {args.output} ({num_frames} frames)", flush=True)


if __name__ == "__main__":
    main()
