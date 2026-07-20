#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture an nsys profile of ONE warm native LTX-2 retake inference.

Builds the native pipeline for one weight-quant mode, runs warmup inferences to
reach steady state, then brackets a single measured inference with the CUDA
profiler API so `nsys profile --capture-range=cudaProfilerApi` records only the
kernels of that one retake (VAE encode / Gemma / diffusion / VAE decode).
"""

import argparse
import importlib.util
import sys
import time

QUANT = {"bf16": None, "fp8": "FP8", "nvfp4": "NVFP4"}
_RANDOM_INITS = (
    "kaiming_uniform_",
    "kaiming_normal_",
    "uniform_",
    "normal_",
    "xavier_uniform_",
    "xavier_normal_",
    "trunc_normal_",
)


def load_oracle(repo):
    p = f"{repo}/examples/visual_gen/ltx2_retake_oracle.py"
    spec = importlib.util.spec_from_file_location("ltx2_retake_oracle", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ltx2_retake_oracle"] = m
    spec.loader.exec_module(m)
    return m


def fast_init():
    # no-op random initializers (build-only; does not change inference kernels)
    import torch.nn.init as I

    def noop(t, *a, **k):
        return t

    for n in _RANDOM_INITS:
        if hasattr(I, n):
            setattr(I, n, noop)


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
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--fast-init", action="store_true")
    args = ap.parse_args()

    import torch

    if args.fast_init:
        fast_init()
    oracle = load_oracle(args.repo)
    device = torch.device("cuda")
    pipe = oracle.build_pipeline(
        args.checkpoint, args.gemma, args.lora, device, "VANILLA", quant_algo=QUANT[args.quant]
    )
    req = oracle._make_request(
        "a person talking to the camera",
        "",
        args.source,
        args.start,
        args.end,
        args.seed,
        args.steps,
    )

    for _ in range(args.warmup):
        oracle._run_pipeline(pipe, req)
        torch.cuda.synchronize()

    # bracket exactly one measured inference for nsys --capture-range=cudaProfilerApi
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"retake_{args.quant}")
    torch.cuda.cudart().cudaProfilerStart()
    t0 = time.time()
    out = pipe.infer(req)
    torch.cuda.synchronize()
    dt = time.time() - t0
    torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.nvtx.range_pop()
    st = getattr(out, "stage_timings", None) or {}
    print(f"NSYS_CAPTURE_DONE quant={args.quant} infer_s={dt:.3f} stages={st}", flush=True)


if __name__ == "__main__":
    main()
