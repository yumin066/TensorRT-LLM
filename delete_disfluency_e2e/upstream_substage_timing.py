#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fine-grained upstream RetakePipeline LTX sub-stage timing (§2-aligned).

Runs the upstream RetakePipeline on the SAME retake-input window the native path
uses, and breaks the single every-rebuild call into subprocess_start / model_load /
video_vae_encode / audio_vae_encode / text_encode / diffusion / video_vae_decode /
audio_vae_decode / mp4_encode / LTX-wall. Instrumentation is external: each pipeline
component callable is wrapped with a CUDA-synced wall timer, so the upstream
``packages/`` sources stay byte-for-byte unchanged.
"""

import argparse
import json
import time
from pathlib import Path

_PROC_START = time.time()


def _sync(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="LTX-2 .safetensors checkpoint")
    ap.add_argument("--gemma-root", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--source", required=True, help="retake_input.mp4 window")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quantization", default="fp8-cast")
    ap.add_argument("--offload-mode", default="none")
    ap.add_argument("--prompt", default="a person talking to the camera")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--warmup", type=int, default=1, help="untimed-for-load warm calls before measure"
    )
    ap.add_argument(
        "--measured", type=int, default=2, help="resident (pure-compute) measured calls"
    )
    ap.add_argument("--code-commit", default=None)
    args = ap.parse_args()

    import torch

    # inference-only: match the upstream retake CLI (@torch.inference_mode()) so the
    # VAE-decode conv3d does not double for autograd (keeps the fp8-cast peak under 96 GB)
    torch.set_grad_enabled(False)

    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.retake import RetakePipeline
    from ltx_pipelines.utils.constants import detect_params
    from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata
    from ltx_pipelines.utils.quantization_factory import QuantizationKind
    from ltx_pipelines.utils.types import OffloadMode

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta_in = get_videostream_metadata(args.source)

    quant = (
        None
        if args.quantization == "none"
        else QuantizationKind(args.quantization).to_policy(checkpoint_path=args.checkpoint)
    )
    loras = [LoraPathStrengthAndSDOps(args.lora, args.lora_strength, LTXV_LORA_COMFY_RENAMING_MAP)]

    torch.cuda.reset_peak_memory_stats()
    _sync(torch)
    t_ml = time.time()
    pipeline = RetakePipeline(
        checkpoint_path=args.checkpoint,
        gemma_root=args.gemma_root,
        loras=loras,
        quantization=quant,
        distilled=True,
        offload_mode=OffloadMode(args.offload_mode),
    )
    _sync(torch)
    model_load_s = time.time() - t_ml
    build_peak = torch.cuda.max_memory_reserved() / 2**30

    # External instrumentation: wrap each component callable with a CUDA-synced
    # wall timer. `self.<component>(...)` then dispatches through the wrapper, so
    # no upstream source is modified.
    stages = {}

    def wrap(inner, name):
        def w(*a, **k):
            _sync(torch)
            t = time.perf_counter()
            r = inner(*a, **k)
            _sync(torch)
            stages[name] = round(time.perf_counter() - t, 4)
            return r

        return w

    pipeline.image_conditioner = wrap(pipeline.image_conditioner, "video_vae_encode")
    pipeline.audio_conditioner = wrap(pipeline.audio_conditioner, "audio_vae_encode")
    pipeline.prompt_encoder = wrap(pipeline.prompt_encoder, "text_encode")
    pipeline.stage = wrap(pipeline.stage, "diffusion")
    pipeline.video_decoder = wrap(pipeline.video_decoder, "video_vae_decode")
    pipeline.audio_decoder = wrap(pipeline.audio_decoder, "audio_vae_decode")

    params = detect_params(args.checkpoint)
    tiling = TilingConfig.default()
    chunks = get_video_chunks_number(meta_in.frames, tiling)
    retake_out = str(outdir / "retake_output.mp4")

    STAGE_KEYS = [
        "video_vae_encode",
        "audio_vae_encode",
        "text_encode",
        "diffusion",
        "video_vae_decode",
        "audio_vae_decode",
        "mp4_encode",
    ]

    def one_call():
        # weights load lazily on the FIRST use of each component, so the first call
        # folds weight-load into the compute stages; a second resident call is pure
        # compute. Return this call's per-stage seconds (+ mp4_encode).
        stages.clear()
        video_iter, audio = pipeline(
            video_path=args.source,
            prompt=args.prompt,
            start_time=args.start,
            end_time=args.end,
            seed=args.seed,
            video_guider_params=params.video_guider_params,
            audio_guider_params=params.audio_guider_params,
            regenerate_video=True,
            regenerate_audio=False,
            tiling_config=tiling,
        )
        _sync(torch)
        t_enc = time.time()
        encode_video(
            video=video_iter,
            fps=int(round(meta_in.fps)),
            audio=audio,
            output_path=retake_out,
            video_chunks_number=chunks,
        )
        _sync(torch)
        run = {k: stages.get(k) for k in STAGE_KEYS}
        run["mp4_encode"] = round(time.time() - t_enc, 4)
        return run

    torch.cuda.reset_peak_memory_stats()
    runs = [one_call() for _ in range(args.warmup + args.measured)]
    infer_peak = torch.cuda.max_memory_reserved() / 2**30
    first = runs[0]  # includes lazy weight load folded into the stages
    measured = runs[args.warmup :]  # resident -> pure compute

    def p50(k):
        xs = sorted(r[k] for r in measured if isinstance(r.get(k), (int, float)))
        return round(xs[len(xs) // 2], 4) if xs else None

    pure = {k: p50(k) for k in STAGE_KEYS}
    pure_total = sum(v for v in pure.values() if v)
    first_total = sum(v for v in (first[k] for k in STAGE_KEYS) if isinstance(v, (int, float)))
    # weight-load time = extra time in the first (cold) call over the resident call,
    # split per stage (transformer->diffusion, Gemma->text_encode, VAE->vae_encode)
    per_stage_load = {
        k: round((first[k] or 0) - (pure[k] or 0), 3) for k in STAGE_KEYS if pure[k] is not None
    }
    model_load = round(first_total - pure_total, 3)

    result = {
        "engine": "upstream",
        "quantization": args.quantization,
        "source": args.source,
        "window_s": [args.start, args.end],
        "seed": args.seed,
        "num_frames": meta_in.frames,
        "width": meta_in.width,
        "height": meta_in.height,
        "fps": float(meta_in.fps),
        "measured_n": len(measured),
        "ltx_substage": {
            "subprocess_start": round(t_ml - _PROC_START, 3),
            "model_load": model_load,
            "video_vae_encode": pure["video_vae_encode"],
            "audio_vae_encode": pure["audio_vae_encode"],
            "text_encode": pure["text_encode"],
            "diffusion": pure["diffusion"],
            "video_vae_decode": pure["video_vae_decode"],
            "audio_vae_decode": pure["audio_vae_decode"],
            "mp4_encode": pure["mp4_encode"],
            "ltx_wall_warm": round(pure_total, 3),
            "ltx_wall_rebuild": round(model_load + pure_total, 3),
        },
        "construct_seconds": round(model_load_s, 3),
        "per_stage_load": per_stage_load,
        "first_call_stages": first,
        "peak_reserved_gib": {"build": round(build_peak, 2), "infer": round(infer_peak, 2)},
        "retake_output": retake_out,
        "code_commit": args.code_commit,
    }
    (outdir / "upstream_substage_timing.json").write_text(json.dumps(result, indent=2))
    print("UPSTREAM_SUBSTAGE_DONE " + json.dumps(result))


if __name__ == "__main__":
    main()
