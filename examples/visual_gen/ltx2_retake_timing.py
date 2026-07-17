#! /usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native LTX-2 retake staged warm-latency harness.

Loads the native ``LTXModel`` retake pipeline ONCE (load-once, serve-many;
``retake_offload_mode: none``) and runs ``N`` warmup + ``M`` measured retakes,
reporting a four-part latency timeline:

- cold ``model_build_load`` (the one-time build + weight load),
- ``first_served`` (the first measured request, reported separately),
- one-time compile / graph-capture (not applicable for bf16/VANILLA; a
  three-way compile-cost split is a separate, later concern), and
- steady-state warm (``p50`` / ``p90`` / ``min`` over the remaining measured
  requests).

Per-stage seconds come from the native pipeline's CUDA-event phase timing
(``pre_denoise`` = source read + window VAE-encode + conditioning; ``denoise`` =
the masked denoise loop; ``post_denoise`` = VAE-decode + composite-back), plus a
synchronized wall time around each ``infer()``. Quality is informational only (a
determinism/stability check of the warm outputs); the native-vs-upstream quality
oracle lives in its own artifact.

This measures the resident pipeline directly (the same resident-warm latency a
served worker exposes, without the HTTP layer). The Mode-A (upstream, every-
rebuild) reference and the full trtllm-serve HTTP path are separate tools.

The heavy build / run is reused from the sibling ``ltx2_retake_oracle`` module
(loaded by path). The pure helpers at the top (``percentile``,
``summarize_samples``, ``split_measured``, ``aggregate_stages``, ``_json_safe``,
``build_timeline``) import on a plain CPU host with no numpy / torch /
tensorrt_llm.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

_STAGE_KEYS = ("wall", "pre_denoise", "denoise", "post_denoise")


# ----------------------------------------------------------------------------
# Pure helpers (stdlib only; host-testable without numpy / torch / tensorrt_llm).
# ----------------------------------------------------------------------------


def percentile(sorted_samples: list, p: float) -> Optional[float]:
    """Linear-interpolation percentile of an ascending-sorted list (p in [0,100])."""
    if not sorted_samples:
        return None
    if len(sorted_samples) == 1:
        return float(sorted_samples[0])
    rank = (p / 100.0) * (len(sorted_samples) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_samples[lo])
    frac = rank - lo
    return float(sorted_samples[lo] * (1.0 - frac) + sorted_samples[hi] * frac)


def summarize_samples(samples: list) -> dict:
    """``p50`` / ``p90`` / ``min`` / ``count`` over a sample list (order-independent)."""
    s = sorted(samples)
    return {
        "p50": percentile(s, 50),
        "p90": percentile(s, 90),
        "min": float(s[0]) if s else None,
        "count": len(s),
    }


def split_measured(records: list) -> tuple:
    """Split measured records into ``(first_served, steady_warm_list)``.

    The first measured request is reported separately from steady state so a
    first-call cost (lazy init, allocator warmup) is never averaged into the
    resident-warm percentiles.
    """
    if not records:
        return None, []
    return records[0], records[1:]


def aggregate_stages(records: list, stage_keys=_STAGE_KEYS) -> dict:
    """Per-stage ``summarize_samples`` across records (skips missing keys)."""
    out = {}
    for key in stage_keys:
        samples = [r[key] for r in records if key in r and r[key] is not None]
        if samples:
            out[key] = summarize_samples(samples)
    return out


def _json_safe(obj):
    """Recursively replace non-finite floats with string sentinels for strict JSON."""
    if isinstance(obj, float):
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        if math.isnan(obj):
            return "nan"
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def build_timeline(model_build_load: float, records: list, stage_keys=_STAGE_KEYS) -> dict:
    """Assemble the four-part latency timeline from the measured records."""
    first, steady = split_measured(records)
    return {
        "cold_model_build_load_seconds": float(model_build_load),
        "first_served": first,
        "steady_warm": aggregate_stages(steady, stage_keys),
        "steady_warm_count": len(steady),
        "compile_graph_capture_note": (
            "not applicable for bf16/VANILLA (no torch_compile / cuda_graph); "
            "the three-way compile/autotune cost split is a separate concern"
        ),
    }


# ----------------------------------------------------------------------------
# Heavy path (imports live here so the pure helpers load on a plain host).
# ----------------------------------------------------------------------------


def _load_oracle():
    """Load the sibling ``ltx2_retake_oracle`` module by path for reuse."""
    oracle_path = Path(__file__).resolve().with_name("ltx2_retake_oracle.py")
    spec = importlib.util.spec_from_file_location("ltx2_retake_oracle", oracle_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Native LTX-2 retake staged warm-latency harness.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--gemma", required=True)
    p.add_argument("--lora", default=None)
    p.add_argument("--source", required=True, help="video-only source clip")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2, help="N warmup requests (not measured)")
    p.add_argument("--measured", type=int, default=10, help="M measured requests")
    p.add_argument("--prompt", default="a person talking to the camera")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--code-commit", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    oracle = _load_oracle()

    import time

    import torch

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    code_commit = args.code_commit or os.environ.get("LTX2_ORACLE_CODE_COMMIT")

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # ---- Cold model build + load (once) --------------------------------
    _sync()
    t0 = time.perf_counter()
    pipe = oracle.build_pipeline(args.checkpoint, args.gemma, args.lora, device, "VANILLA")
    _sync()
    model_build_load = time.perf_counter() - t0

    request = oracle._make_request(
        args.prompt, args.negative_prompt, args.source, args.start, args.end, args.seed, args.steps
    )

    # ---- Warmup (not measured) -----------------------------------------
    for _ in range(max(0, args.warmup)):
        pipe.infer(request)
        _sync()

    # ---- Measured (load-once, serve-many) ------------------------------
    records = []
    first_thwc = None
    last_thwc = None
    for i in range(args.measured):
        _sync()
        w0 = time.perf_counter()
        out = pipe.infer(request)
        _sync()
        wall = time.perf_counter() - w0
        records.append(
            {
                "index": i,
                "wall": wall,
                "pre_denoise": float(out.pre_denoise),
                "denoise": float(out.denoise),
                "post_denoise": float(out.post_denoise),
            }
        )
        thwc = oracle._video_to_thwc_uint8(out.video)
        if first_thwc is None:
            first_thwc = thwc
        last_thwc = thwc
        print(f"[timing] measured {i}: wall={wall:.3f}s denoise={out.denoise:.3f}s", flush=True)

    # ---- Informational quality: warm-output determinism/stability ------
    quality = None
    if first_thwc is not None and last_thwc is not None and args.measured >= 2:
        quality = {
            "label": "last_vs_first_measured",
            "psnr": oracle.psnr(last_thwc, first_thwc),
            "ssim": oracle.ssim(last_thwc, first_thwc),
            "note": (
                "determinism/stability of the resident warm outputs (seed-42); "
                "native-vs-upstream quality is in the oracle artifact"
            ),
        }

    timeline = build_timeline(model_build_load, records)
    native_repo = oracle._git_info(str(Path(__file__).resolve().parents[2]))
    result = {
        "mode": "resident_warm_native_retake",
        "timeline": timeline,
        "records": records,
        "warmup_iterations": max(0, args.warmup),
        "measured_iterations": args.measured,
        "config": {
            "attention_backend": "VANILLA",
            "dtype": "bf16",
            "retake_offload_mode": "none",
            "cuda_graph": False,
            "quant_algo": None,
        },
        "quality_informational": quality,
        "env": oracle._env_metadata(),
        "code_commit": code_commit,
        "native_repo": native_repo,
        "source": str(Path(args.source).resolve()),
        "source_sha256": oracle.sha256_file(args.source),
        "window": [args.start, args.end],
        "seed": args.seed,
        "steps": args.steps,
        "note": (
            "Resident warm (load-once, serve-many) native retake staged timing. "
            "first_served is reported separately from steady-warm p50/p90/min. "
            "Mode-A (upstream every-rebuild) reference and the trtllm-serve HTTP "
            "path are separate tools; denoise_per_step and the vae_encode/vae_decode "
            "fine split are follow-ons."
        ),
    }
    (output_dir / "timing.json").write_text(
        json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False)
    )
    steady = timeline["steady_warm"].get("wall")
    print(
        "TIMING_DONE",
        json.dumps({"model_build_load": model_build_load, "steady_warm_wall": steady}),
    )
    # A meaningful run needs the cold load plus at least one steady-warm sample.
    return 0 if (model_build_load > 0 and timeline["steady_warm_count"] >= 1) else 1


if __name__ == "__main__":
    sys.exit(main())
