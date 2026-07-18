# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quantization latency + peak-memory sweep for the native LTX-2 retake denoise.

On the RTX PRO 6000 Blackwell (sm_120), this measures the native retake pipeline
across weight-quantization modes — ``bf16`` (no quant), ``FP8`` (dynamic), and
``NVFP4`` (dynamic) — reporting, per mode:

- warm ``denoise`` latency (``p50`` / ``p90`` / ``min``) over M measured retakes,
- peak GPU memory (``allocated`` + ``reserved``) for the model build/load phase
  and for the measured inference phase, plus steady resident allocated bytes,
- informational PSNR / SSIM of the regenerated window vs the ``bf16`` baseline
  output (never gates — quantization changes pixels; this only quantifies it).

Each mode runs in its OWN subprocess (``--single-mode``) so a mode that fails to
build/run is recorded and the sweep continues, and a ~near-capacity 22b never
stacks two model copies on one GPU. The pure helpers (percentile/summary, memory
ratio, status classify, JSON sanitize) are stdlib-only so they load and unit-test
on a plain host with no torch / numpy / tensorrt_llm.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# (label, quant_algo passed to build_pipeline). ``bf16`` is the no-quant baseline
# and the informational-quality anchor; it must run first so the others can
# compare against its saved output.
MODES = (("bf16", None), ("fp8", "FP8"), ("nvfp4", "NVFP4"))
BASELINE_MODE = "bf16"
_BYTES_PER_GIB = 1024**3


# --------------------------------------------------------------------------- #
# Pure helpers (stdlib-only; no torch / numpy / tensorrt_llm)
# --------------------------------------------------------------------------- #


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
    if not samples:
        return {"p50": None, "p90": None, "min": None, "count": 0}
    s = sorted(float(x) for x in samples)
    return {
        "p50": percentile(s, 50.0),
        "p90": percentile(s, 90.0),
        "min": s[0],
        "count": len(s),
    }


def gib(nbytes: Optional[float]) -> Optional[float]:
    """Bytes → GiB (rounded to 4 dp) for human-readable memory columns."""
    if nbytes is None:
        return None
    return round(float(nbytes) / _BYTES_PER_GIB, 4)


def memory_delta(baseline_bytes: Optional[float], mode_bytes: Optional[float]) -> dict:
    """Memory ratio + absolute saving of a mode vs the bf16 baseline.

    ``ratio`` = mode/baseline (``<1`` means the mode uses less); ``saved_gib`` =
    baseline−mode in GiB (positive = saved).
    """
    if not baseline_bytes or not mode_bytes or baseline_bytes <= 0 or mode_bytes <= 0:
        return {"ratio_vs_bf16": None, "saved_gib": None}
    return {
        "ratio_vs_bf16": mode_bytes / baseline_bytes,
        "saved_gib": gib(baseline_bytes - mode_bytes),
    }


def latency_delta(baseline_p50: Optional[float], p50: Optional[float]) -> dict:
    """Speedup + absolute delta of a mode's denoise p50 vs the bf16 baseline."""
    if not baseline_p50 or not p50 or baseline_p50 <= 0 or p50 <= 0:
        return {"speedup_vs_bf16": None, "delta_seconds": None}
    return {"speedup_vs_bf16": baseline_p50 / p50, "delta_seconds": p50 - baseline_p50}


def classify_mode_status(ran_ok: bool, error_reason: Optional[str]) -> tuple:
    """Map a mode's build/run outcome to ``(status, reason)`` (``ok`` / ``error``)."""
    if not ran_ok:
        return "error", error_reason
    return "ok", None


def summarize_modes(records: list) -> dict:
    """Attach per-mode latency + memory deltas vs bf16 + roll up statuses."""
    by = {r["mode"]: r for r in records}
    base = by.get(BASELINE_MODE)
    base_ok = bool(base and base.get("status") == "ok")
    base_p50 = (base.get("denoise") or {}).get("p50") if base_ok else None
    base_infer_alloc = _get(base, "inference_peak", "allocated_bytes") if base_ok else None
    for r in records:
        ok = r.get("status") == "ok"
        r["latency_delta"] = (
            latency_delta(base_p50, (r.get("denoise") or {}).get("p50"))
            if ok
            else {
                "speedup_vs_bf16": None,
                "delta_seconds": None,
            }
        )
        r["memory_delta"] = (
            memory_delta(base_infer_alloc, _get(r, "inference_peak", "allocated_bytes"))
            if ok
            else {"ratio_vs_bf16": None, "saved_gib": None}
        )
    return {
        "baseline_mode": BASELINE_MODE,
        "baseline_ran_ok": base_ok,
        "modes_ok": [r["mode"] for r in records if r.get("status") == "ok"],
        "by_status": _count_status(records),
    }


def _get(rec, *keys):
    cur = rec
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _count_status(records: list) -> dict:
    out = {}
    for r in records:
        out[r.get("status")] = out.get(r.get("status"), 0) + 1
    return out


def _json_safe(obj):
    """Recursively replace non-finite floats with string sentinels (strict JSON)."""
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


# --------------------------------------------------------------------------- #
# Heavy path (imports inside functions so pure helpers load on a plain host).
# --------------------------------------------------------------------------- #


def _load_oracle():
    """Load the sibling ``ltx2_retake_oracle`` module by path for reuse."""
    oracle_path = Path(__file__).resolve().with_name("ltx2_retake_oracle.py")
    spec = importlib.util.spec_from_file_location("ltx2_retake_oracle", oracle_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mem_snapshot(torch) -> dict:
    """Peak ``allocated`` / ``reserved`` bytes since the last reset (0 on CPU)."""
    if not torch.cuda.is_available():
        return {"allocated_bytes": 0, "reserved_bytes": 0}
    return {
        "allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _run_single_mode(args) -> int:
    """Build the retake pipeline in one quant mode, measure latency + memory, print result."""
    oracle = _load_oracle()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = args.single_mode
    quant_algo = args.quant_algo or None

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    ran_ok, error_reason = False, None
    denoise_samples: list = []
    model_load_peak = inference_peak = None
    resident_bytes = None
    quality = None

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        pipe = oracle.build_pipeline(
            args.checkpoint, args.gemma, args.lora, device, "VANILLA", quant_algo=quant_algo
        )
        _sync()
        model_load_peak = _mem_snapshot(torch)
        if torch.cuda.is_available():
            resident_bytes = int(torch.cuda.memory_allocated())

        request = oracle._make_request(
            args.prompt,
            args.negative_prompt,
            args.source,
            args.start,
            args.end,
            args.seed,
            args.steps,
        )
        for _ in range(max(0, args.warmup)):
            pipe.infer(request)
            _sync()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        last_out = None
        for _ in range(max(1, args.measured)):
            _sync()
            out = pipe.infer(request)
            _sync()
            denoise_samples.append(float(out.denoise))
            last_out = out
        inference_peak = _mem_snapshot(torch)
        ran_ok = True

        # Informational quality: save bf16's window output, or compare vs it.
        if last_out is not None and last_out.video is not None:
            thwc = oracle._video_to_thwc_uint8(last_out.video)
            fps = float(last_out.frame_rate or 0.0) or 25.0
            if label == BASELINE_MODE and args.baseline_thwc:
                torch.save(thwc.cpu(), args.baseline_thwc)
            elif args.baseline_thwc and os.path.exists(args.baseline_thwc):
                from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_retake import (
                    _retake_pixel_window,
                )

                base_thwc = torch.load(args.baseline_thwc, map_location="cpu")
                pw = _retake_pixel_window(args.start, args.end, fps, int(thwc.shape[0]))
                quality = oracle.compute_similarity(thwc.cpu(), base_thwc, pw, f"{label}_vs_bf16")
    except Exception as exc:  # noqa: BLE001 - per-mode isolation: record + continue
        error_reason = f"{type(exc).__name__}: {exc}"

    status, reason = classify_mode_status(ran_ok, error_reason)
    result = {
        "mode": label,
        "quant_algo": quant_algo,
        "status": status,
        "reason": reason,
        "denoise": summarize_samples(denoise_samples) if denoise_samples else None,
        "denoise_samples": denoise_samples,
        "model_load_peak": model_load_peak,
        "inference_peak": inference_peak,
        "resident_allocated_bytes": resident_bytes,
        "resident_allocated_gib": gib(resident_bytes),
        "quality_informational": quality,
    }
    print("MODE_RESULT " + json.dumps(_json_safe(result)))
    return 0


def parse_mode_result(stdout: str) -> Optional[dict]:
    """Extract the ``MODE_RESULT`` JSON a ``--single-mode`` subprocess prints."""
    for line in stdout.splitlines():
        if line.startswith("MODE_RESULT "):
            try:
                return json.loads(line[len("MODE_RESULT ") :])
            except json.JSONDecodeError:
                return None
    return None


def _spawn_single_mode(args, label: str, quant_algo: Optional[str], baseline_thwc: str) -> dict:
    """Run one quant mode in a fresh subprocess (full GPU release between modes)."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-mode",
        label,
        "--checkpoint",
        args.checkpoint,
        "--gemma",
        args.gemma,
        "--source",
        args.source,
        "--output-dir",
        args.output_dir,
        "--start",
        str(args.start),
        "--end",
        str(args.end),
        "--seed",
        str(args.seed),
        "--steps",
        str(args.steps),
        "--warmup",
        str(args.warmup),
        "--measured",
        str(args.measured),
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
        "--baseline-thwc",
        baseline_thwc,
    ]
    if quant_algo:
        cmd += ["--quant-algo", quant_algo]
    if args.lora:
        cmd += ["--lora", args.lora]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    rec = parse_mode_result(proc.stdout)
    if rec is None:
        rec = {
            "mode": label,
            "quant_algo": quant_algo,
            "status": "error",
            "reason": f"no MODE_RESULT (exit {proc.returncode}); stderr tail: "
            + "".join(proc.stderr.splitlines()[-3:]),
            "denoise": None,
            "model_load_peak": None,
            "inference_peak": None,
            "resident_allocated_bytes": None,
            "quality_informational": None,
        }
    return rec


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Native LTX-2 retake quantization latency + memory sweep."
    )
    p.add_argument("--checkpoint")
    p.add_argument("--gemma")
    p.add_argument("--lora", default=None)
    p.add_argument("--source")
    p.add_argument("--output-dir")
    p.add_argument("--start", type=float, default=1.0)
    p.add_argument("--end", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--measured", type=int, default=8)
    p.add_argument("--prompt", default="a person talking to the camera")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--code-commit", default=None)
    p.add_argument("--single-mode", default=None, help="internal: build+measure ONE quant mode")
    p.add_argument("--quant-algo", default=None, help="internal: quant_algo for --single-mode")
    p.add_argument("--baseline-thwc", default=None, help="path to save/load the bf16 window output")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.single_mode:
        return _run_single_mode(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_commit = args.code_commit or os.environ.get("LTX2_ORACLE_CODE_COMMIT")
    baseline_thwc = str(output_dir / "baseline_bf16_thwc.pt")

    records = []
    for label, quant_algo in MODES:
        print(f"[quant-mem] {label}: build + measure ...", flush=True)
        rec = _spawn_single_mode(args, label, quant_algo, baseline_thwc)
        print(f"[quant-mem] {label}: {rec.get('status')} ({rec.get('reason')})", flush=True)
        records.append(rec)

    summary = summarize_modes(records)
    report = {
        "mode": "native_retake_quant_latency_memory",
        "device_query": _device_query(),
        "config": {
            "checkpoint": args.checkpoint,
            "gemma": args.gemma,
            "lora": args.lora,
            "source": args.source,
            "window": [args.start, args.end],
            "seed": args.seed,
            "steps": args.steps,
            "warmup": args.warmup,
            "measured": args.measured,
            "code_commit": code_commit,
        },
        "records": records,
        "summary": summary,
    }
    out_path = output_dir / "quant_mem.json"
    with open(out_path, "w") as f:
        json.dump(_json_safe(report), f, indent=2, allow_nan=False)
    print(f"QUANT_MEM_DONE {json.dumps(_json_safe(summary))}")
    print(f"wrote {out_path}")
    return 0


def _device_query() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "name": torch.cuda.get_device_name(0),
            "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
            "total_gib": gib(torch.cuda.get_device_properties(0).total_memory),
        }
    except Exception:  # noqa: BLE001 - provenance only
        return {}


if __name__ == "__main__":
    raise SystemExit(main())
