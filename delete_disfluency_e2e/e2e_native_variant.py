# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Native LTX-2 retake deliverable + timing/memory for one weight-quant mode.

Reuses the checked-in oracle's module-level helpers (build_pipeline / _make_request
/ _run_pipeline / _video_to_thwc_uint8 / encode_mp4). Emits a full-length composited
retake clip (fed to the upstream --external-retake composite) plus single-shot and
warm-retake timing and peak GPU memory. Runs one quant per process for isolation.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

QUANT = {"bf16": None, "fp8": "FP8", "nvfp4": "NVFP4"}


def load_oracle(repo):
    p = f"{repo}/examples/visual_gen/ltx2_retake_oracle.py"
    spec = importlib.util.spec_from_file_location("ltx2_retake_oracle", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ltx2_retake_oracle"] = m
    spec.loader.exec_module(m)
    return m


def disable_lazy_init():
    """No-op the random weight initializers.

    Safe here because every parameter is overwritten by the checkpoint load;
    skipping the CPU-side kaiming/uniform fill removes a multi-minute construction
    cost with no effect on loaded weights.
    """
    import torch.nn.init as I

    def _noop(tensor, *a, **k):
        return tensor

    for name in (
        "kaiming_uniform_",
        "kaiming_normal_",
        "uniform_",
        "normal_",
        "xavier_uniform_",
        "xavier_normal_",
        "trunc_normal_",
        "constant_",
        "ones_",
        "zeros_",
    ):
        if hasattr(I, name):
            setattr(I, name, _noop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gemma", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--quant", choices=list(QUANT), required=True)
    ap.add_argument("--measured", type=int, default=4)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--code-commit", default=None)
    ap.add_argument("--fast-init", action="store_true")
    args = ap.parse_args()

    import torch

    oracle = load_oracle(args.repo)
    if args.fast_init:
        disable_lazy_init()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    quant_algo = QUANT[args.quant]

    torch.cuda.reset_peak_memory_stats()
    tb = time.time()
    pipe = oracle.build_pipeline(
        args.checkpoint, args.gemma, args.lora, device, "VANILLA", quant_algo=quant_algo
    )
    build_s = time.time() - tb
    build_peak_resv = torch.cuda.max_memory_reserved() / 2**30

    req = oracle._make_request(
        "a person talking to the camera",
        "",
        args.source,
        args.start,
        args.end,
        args.seed,
        args.steps,
    )

    torch.cuda.reset_peak_memory_stats()
    times = []
    video = fps = None
    first_infer_s = None
    for i in range(args.measured + 1):
        torch.cuda.synchronize()
        t0 = time.time()
        video, _audio, fps, _asr = oracle._run_pipeline(pipe, req)
        torch.cuda.synchronize()
        dt = time.time() - t0
        if i == 0:
            first_infer_s = dt
        else:
            times.append(dt)
    infer_peak_resv = torch.cuda.max_memory_reserved() / 2**30
    ts = sorted(times)
    p50 = ts[len(ts) // 2]

    thwc = oracle._video_to_thwc_uint8(video)
    mp4 = str(outdir / "native.mp4")
    ok = oracle.encode_mp4(thwc, fps, mp4)

    result = {
        "quant": args.quant,
        "quant_algo": quant_algo,
        "source": args.source,
        "window_s": [args.start, args.end],
        "seed": args.seed,
        "steps": args.steps,
        "fast_init": args.fast_init,
        "num_frames": int(thwc.shape[0]),
        "height": int(thwc.shape[1]),
        "width": int(thwc.shape[2]),
        "fps": float(fps),
        "build_seconds": round(build_s, 3),
        "first_infer_seconds": round(first_infer_s, 3),
        "single_shot_seconds": round(build_s + first_infer_s, 3),
        "warm_retake_p50": round(p50, 3),
        "warm_min": round(ts[0], 3),
        "measured_n": len(times),
        "peak_reserved_gib": {
            "build": round(build_peak_resv, 2),
            "infer": round(infer_peak_resv, 2),
        },
        "mp4": mp4,
        "mp4_ok": bool(ok),
        "code_commit": args.code_commit,
    }
    (outdir / "variant.json").write_text(json.dumps(result, indent=2))
    print("VARIANT_DONE " + json.dumps(result))


if __name__ == "__main__":
    main()
