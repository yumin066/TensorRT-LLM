#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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


# Only the expensive RANDOM initializers are skipped. Deterministic fills
# (zeros_/ones_/constant_) are cheap and may seed computed non-checkpoint buffers,
# so they are left intact — narrowing to random inits keeps the speedup while
# never touching a value that is not overwritten by the checkpoint.
_RANDOM_INITS = (
    "kaiming_uniform_",
    "kaiming_normal_",
    "uniform_",
    "normal_",
    "xavier_uniform_",
    "xavier_normal_",
    "trunc_normal_",
)


# Ids of the exact tensors fast-init skipped (poison mode) — the coverage scan
# only judges THESE, so torch.empty buffers (which fast-init never touches) are
# not mistaken for uncovered inits.
_POISONED_IDS: set = set()


def disable_lazy_init(poison=False):
    """Skip the random weight initializers during model construction.

    The skipped inits touch only Linear/conv weights that the checkpoint load
    fully overwrites, so no-oping them removes a multi-minute CPU build cost with
    no effect on loaded weights. When ``poison=True`` the skipped inits fill with
    NaN and record the tensor id, so a post-load NaN scan (restricted to those
    ids) proves every tensor fast-init touched was overwritten before use.
    """
    import torch
    import torch.nn.init as I

    def _repl(tensor, *a, **k):
        if poison:
            _POISONED_IDS.add(id(tensor))
            with torch.no_grad():
                tensor.fill_(float("nan"))
        return tensor

    for name in _RANDOM_INITS:
        if hasattr(I, name):
            setattr(I, name, _repl)


def _scan_for_nan(pipe):
    """Scan every live nn.Module for a poisoned param/buffer still holding NaN.

    A surviving NaN means a skipped init was not overwritten by the checkpoint.
    Uses ``gc.get_objects()`` + ``recurse=False`` so shared/companion modules
    (the retake pipeline shares its transformer with a companion LTX2Pipeline that
    also owns the VAE / text-encoder / connector) are all covered exactly once,
    rather than only the modules reachable as attributes of ``pipe``.
    """
    import gc

    import torch
    import torch.nn as nn

    seen = set()
    nan_tensors, n_params, n_buffers, n_modules = [], 0, 0, 0
    for obj in gc.get_objects():
        # gc may hand back objects mid-collection (weakref proxies); skip those.
        try:
            if not isinstance(obj, nn.Module) or id(obj) in seen:
                continue
            seen.add(id(obj))
            n_modules += 1
            for name, p in obj.named_parameters(recurse=False):
                n_params += 1
                if id(p) in _POISONED_IDS and torch.isnan(p).any():
                    nan_tensors.append(
                        {
                            "kind": "param",
                            "module": type(obj).__name__,
                            "name": name,
                            "shape": list(p.shape),
                        }
                    )
            for name, b in obj.named_buffers(recurse=False):
                n_buffers += 1
                if id(b) in _POISONED_IDS and torch.isnan(b).any():
                    nan_tensors.append(
                        {
                            "kind": "buffer",
                            "module": type(obj).__name__,
                            "name": name,
                            "shape": list(b.shape),
                        }
                    )
        except ReferenceError:
            continue
    return {
        "modules_scanned": n_modules,
        "params_scanned": n_params,
        "buffers_scanned": n_buffers,
        "fast_init_touched_tensors": len(_POISONED_IDS),
        "uncovered_nan_count": len(nan_tensors),
        "uncovered": nan_tensors[:50],
        "fast_init_safe": len(nan_tensors) == 0,
        "note": "scan judges only tensors fast-init actually skipped (nan-poisoned ids); "
        "torch.empty buffers are outside fast-init's scope and excluded",
    }


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
    ap.add_argument(
        "--verify-init-coverage",
        action="store_true",
        help="Poison the skipped inits with NaN, build+load, and report any param/buffer "
        "still holding NaN (proves fast-init is safe). Skips the retake and exits.",
    )
    args = ap.parse_args()

    import torch

    oracle = load_oracle(args.repo)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    quant_algo = QUANT[args.quant]

    if args.verify_init_coverage:
        disable_lazy_init(poison=True)
        pipe = oracle.build_pipeline(
            args.checkpoint, args.gemma, args.lora, device, "VANILLA", quant_algo=quant_algo
        )
        cov = _scan_for_nan(pipe)
        cov.update({"quant": args.quant, "code_commit": args.code_commit})
        (outdir / "init_coverage.json").write_text(json.dumps(cov, indent=2))
        print("INIT_COVERAGE_DONE " + json.dumps(cov))
        return

    if args.fast_init:
        disable_lazy_init()

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
