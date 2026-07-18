# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention-backend profiling for the native LTX-2 retake denoise.

On the RTX PRO 6000 Blackwell (sm_120), this measures the native retake
pipeline's attention backends selected through
``pipeline_config.attention.backend`` — ``VANILLA`` (baseline SDPA), ``FA4``
(Flash Attention 4), and ``CUTEDSL`` (CuTe DSL FMHA cubins) — reporting:

- per-backend steady-warm ``denoise`` timing (``p50`` / ``p90`` / ``min``) and
  the delta vs the ``VANILLA`` baseline,
- the profiler-confirmed **attention share** of denoise GPU time (computed from
  an ``nsys stats --report cuda_gpu_kern_sum`` CSV via attention-kernel name
  patterns) — filled in by the cluster driver, never required by the timing run,
- the ``CUTEDSL`` head-dim / dtype capability result (``CUTEDSL`` cubins require
  ``head_dim=128``); an unsupported backend is recorded with its concrete reason
  and the sweep continues,
- an explicit note that quantized attention is not implemented on this stack.

Each backend runs in its OWN subprocess (``--single-backend``) so a failing or
unsupported backend never poisons the others and an ``nsys`` wrapper can profile
one backend cleanly. The pure helpers (percentile, attention-share from a kernel
CSV, backend delta, CuTeDSL capability, JSON sanitize) are stdlib-only so they
load and unit-test on a plain host with no torch / numpy / tensorrt_llm.
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

# The three real single-tensor attention backends worth contrasting for the
# retake denoise. TRTLLM is a paged-KV LLM backend (not the diffusion path) and
# is intentionally excluded.
BACKENDS = ("VANILLA", "FA4", "CUTEDSL")
BASELINE_BACKEND = "VANILLA"

# CuTeDSL FMHA cubins are compiled for head_dim=128 only (see
# ``attention_backend/cute_dsl.py``); bf16/fp16 are the supported element types.
CUTEDSL_REQUIRED_HEAD_DIM = 128
CUTEDSL_SUPPORTED_DTYPES = ("bfloat16", "float16", "bf16", "fp16", "half")

# Default substrings (lowercased) that mark an attention / FMHA CUDA kernel in an
# nsys ``cuda_gpu_kern_sum`` report. Broad on purpose (SDPA-flash, FA4-cute, and
# CuTeDSL kernels use different names); the matched kernel names are recorded in
# the artifact so the share is auditable rather than opaque.
DEFAULT_ATTENTION_KERNEL_PATTERNS = (
    "fmha",
    "flash",
    "attention",
    "sdpa",
    "scaled_dot_product",
    "cute_dsl",
    "cutlass_mha",
    "mha_",
)


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


def cutedsl_capability(head_dim: Optional[int], dtype: Optional[str]) -> tuple:
    """Return ``(supported, reason)`` for the CUTEDSL backend on this config.

    CUTEDSL FMHA cubins require ``head_dim == 128``; the element type must be a
    16-bit float. ``reason`` is ``None`` when supported, else a concrete string.
    """
    if head_dim is None:
        return False, "unknown_head_dim"
    if int(head_dim) != CUTEDSL_REQUIRED_HEAD_DIM:
        return False, f"head_dim={head_dim}!=128"
    if dtype is not None and str(dtype).lower() not in CUTEDSL_SUPPORTED_DTYPES:
        return False, f"dtype={dtype}_unsupported"
    return True, None


def parse_nsys_kern_csv(text: str) -> list:
    """Parse an ``nsys stats --report cuda_gpu_kern_sum --format csv`` blob.

    Returns a list of ``{"name": str, "total_ns": float}`` rows. The report has
    a ``Total Time (ns)`` (or ``Total Time`` / ``Time (ns)``) column and a
    ``Name`` column; column order varies by nsys version, so the header row is
    matched by name. Non-data / malformed lines are skipped, not fatal.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    header = None
    header_idx = 0
    for i, ln in enumerate(lines):
        low = ln.lower()
        if "name" in low and ("time" in low):
            header = _split_csv_row(ln)
            header_idx = i
            break
    if header is None:
        return []
    lower_header = [h.strip().lower() for h in header]

    def _find(cands):
        for c in cands:
            for j, h in enumerate(lower_header):
                if h == c:
                    return j
        for c in cands:
            for j, h in enumerate(lower_header):
                if c in h:
                    return j
        return None

    name_j = _find(["name", "kernel name"])
    # Prefer an explicit total-time column over a percentage/instances column.
    time_j = _find(["total time (ns)", "total time", "time (ns)", "time"])
    if name_j is None or time_j is None:
        return []
    rows = []
    for ln in lines[header_idx + 1 :]:
        cells = _split_csv_row(ln)
        if len(cells) <= max(name_j, time_j):
            continue
        name = cells[name_j].strip().strip('"')
        raw = cells[time_j].strip().strip('"').replace(",", "")
        try:
            total_ns = float(raw)
        except ValueError:
            continue
        if not name:
            continue
        rows.append({"name": name, "total_ns": total_ns})
    return rows


def _split_csv_row(line: str) -> list:
    """Split one CSV line, honoring simple double-quoted fields (kernel names)."""
    out = []
    cur = []
    in_q = False
    for ch in line:
        if ch == '"':
            in_q = not in_q
        elif ch == "," and not in_q:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def _is_attention_kernel(name: str, patterns) -> bool:
    low = name.lower()
    return any(p in low for p in patterns)


def attention_share(rows: list, patterns=DEFAULT_ATTENTION_KERNEL_PATTERNS) -> dict:
    """Attention share of total kernel time from ``parse_nsys_kern_csv`` rows.

    Returns ``share`` (attention_ns / total_ns in ``[0,1]``, or ``None`` when no
    kernels), the attention/total nanoseconds, and the matched attention kernel
    names ranked by time (for auditability — the share is only as trustworthy as
    the patterns, so we make the classification inspectable).
    """
    total_ns = sum(r["total_ns"] for r in rows)
    attn_rows = [r for r in rows if _is_attention_kernel(r["name"], patterns)]
    attn_ns = sum(r["total_ns"] for r in attn_rows)
    matched = sorted(attn_rows, key=lambda r: r["total_ns"], reverse=True)
    return {
        "share": (attn_ns / total_ns) if total_ns > 0 else None,
        "attention_ns": attn_ns,
        "total_ns": total_ns,
        "matched_kernels": [{"name": r["name"], "total_ns": r["total_ns"]} for r in matched[:12]],
        "num_matched": len(attn_rows),
    }


def backend_delta(baseline_p50: Optional[float], p50: Optional[float]) -> dict:
    """Speedup + absolute delta of a backend's denoise p50 vs the baseline p50."""
    if not baseline_p50 or not p50 or baseline_p50 <= 0 or p50 <= 0:
        return {"speedup_vs_baseline": None, "delta_seconds": None}
    return {
        "speedup_vs_baseline": baseline_p50 / p50,
        "delta_seconds": p50 - baseline_p50,
    }


def reclassify_cutedsl_capability_error(
    supported: bool, precheck_reason: Optional[str], error_reason: Optional[str]
) -> tuple:
    """Fold a CUTEDSL head-dim cubin rejection into ``(supported, precheck_reason)``.

    The CUTEDSL precheck only sees the video attention head_dim (128, which
    passes), but the LTX-2 AudioVideo model also builds audio attention at
    head_dim=64, which the cubins reject during the forward. That raises at
    runtime and would otherwise be recorded as ``error`` — but it is a capability
    limit, not a crash, so map it to ``unsupported``. Returns the (possibly
    updated) ``(supported, precheck_reason)`` pair.
    """
    if error_reason and "require head_dim" in error_reason.lower():
        return False, error_reason
    return supported, precheck_reason


def classify_backend_status(
    supported: bool,
    precheck_reason: Optional[str],
    ran_ok: bool,
    error_reason: Optional[str],
) -> tuple:
    """Map a backend's precheck + run outcome to ``(status, reason)``.

    ``unsupported`` = a capability precheck ruled it out (never built);
    ``error`` = it was eligible but the build/run raised; ``ok`` = it ran.
    """
    if not supported:
        return "unsupported", precheck_reason
    if not ran_ok:
        return "error", error_reason
    return "ok", None


def summarize_profile(records: list) -> dict:
    """Attach per-backend deltas vs the VANILLA baseline + roll up statuses."""
    by_backend = {r["backend"]: r for r in records}
    base = by_backend.get(BASELINE_BACKEND)
    base_p50 = None
    if base and base.get("status") == "ok" and base.get("denoise"):
        base_p50 = base["denoise"].get("p50")
    for r in records:
        d = r.get("denoise") or {}
        r["delta"] = (
            backend_delta(base_p50, d.get("p50"))
            if r.get("status") == "ok"
            else {
                "speedup_vs_baseline": None,
                "delta_seconds": None,
            }
        )
    ok = [r["backend"] for r in records if r.get("status") == "ok"]
    return {
        "baseline_backend": BASELINE_BACKEND,
        "baseline_ran_ok": bool(base and base.get("status") == "ok"),
        "backends_ok": ok,
        "by_status": _count_status(records),
    }


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
# Heavy path (imports live inside functions so the pure helpers load on a plain
# host with no torch / numpy / tensorrt_llm).
# --------------------------------------------------------------------------- #


def _load_oracle():
    """Load the sibling ``ltx2_retake_oracle`` module by path for reuse."""
    oracle_path = Path(__file__).resolve().with_name("ltx2_retake_oracle.py")
    spec = importlib.util.spec_from_file_location("ltx2_retake_oracle", oracle_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_head_dim(pipe) -> Optional[int]:
    """Best-effort read of the retake transformer's attention head_dim."""
    try:
        cfg = pipe.pipeline_config.model_configs["transformer"].pretrained_config
    except (AttributeError, KeyError, TypeError):
        return None
    for dim_attr, heads_attr in (
        ("hidden_size", "num_attention_heads"),
        ("dim", "num_heads"),
    ):
        dim = getattr(cfg, dim_attr, None)
        heads = getattr(cfg, heads_attr, None)
        if dim and heads:
            try:
                return int(dim) // int(heads)
            except (TypeError, ZeroDivisionError):
                continue
    hd = getattr(cfg, "attention_head_dim", None) or getattr(cfg, "head_dim", None)
    return int(hd) if hd else None


def _run_single_backend(args) -> int:
    """Build the retake pipeline with one attention backend, time denoise, print result."""
    oracle = _load_oracle()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = args.single_backend.upper()

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    supported, precheck_reason = True, None
    ran_ok, error_reason = False, None
    denoise_samples: list = []
    head_dim = None

    try:
        pipe = oracle.build_pipeline(args.checkpoint, args.gemma, args.lora, device, backend)
        head_dim = _resolve_head_dim(pipe)
        if backend == "CUTEDSL":
            supported, precheck_reason = cutedsl_capability(head_dim, "bfloat16")
        if supported:
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
            # Under nsys, bracket ONLY the measured denoise with the cudaProfiler
            # API so the profiler records the retake forward — not the ~46GB
            # weight load — when launched with ``--capture-range=cudaProfilerApi``.
            if args.nsys_capture:
                _sync()
                torch.cuda.profiler.start()
            for _ in range(max(1, args.measured)):
                _sync()
                out = pipe.infer(request)
                _sync()
                denoise_samples.append(float(out.denoise))
            if args.nsys_capture:
                _sync()
                torch.cuda.profiler.stop()
            ran_ok = True
    except Exception as exc:  # noqa: BLE001 - per-backend isolation: record + continue
        error_reason = f"{type(exc).__name__}: {exc}"

    # A CUTEDSL head-dim cubin rejection is a capability limit, not a runtime
    # error — fold it into (supported, precheck_reason) so it classifies as
    # ``unsupported`` rather than ``error``.
    supported, precheck_reason = reclassify_cutedsl_capability_error(
        supported, precheck_reason, error_reason
    )
    status, reason = classify_backend_status(supported, precheck_reason, ran_ok, error_reason)
    result = {
        "backend": backend,
        "status": status,
        "reason": reason,
        "head_dim": head_dim,
        "denoise": summarize_samples(denoise_samples) if denoise_samples else None,
        "denoise_samples": denoise_samples,
    }
    print("BACKEND_RESULT " + json.dumps(_json_safe(result)))
    return 0


def parse_backend_result(stdout: str) -> Optional[dict]:
    """Extract the ``BACKEND_RESULT`` JSON a ``--single-backend`` subprocess prints."""
    for line in stdout.splitlines():
        if line.startswith("BACKEND_RESULT "):
            try:
                return json.loads(line[len("BACKEND_RESULT ") :])
            except json.JSONDecodeError:
                return None
    return None


def _spawn_single_backend(args, backend: str) -> dict:
    """Run one backend in a fresh subprocess (full GPU release between backends)."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-backend",
        backend,
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
    ]
    if args.lora:
        cmd += ["--lora", args.lora]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    rec = parse_backend_result(proc.stdout)
    if rec is None:
        rec = {
            "backend": backend,
            "status": "error",
            "reason": f"no BACKEND_RESULT (exit {proc.returncode}); stderr tail: "
            + "".join(proc.stderr.splitlines()[-3:]),
            "head_dim": None,
            "denoise": None,
            "denoise_samples": [],
        }
    return rec


def _load_nsys_shares(path: Optional[str]) -> dict:
    """Load an optional ``{backend: attention_share_dict}`` map from the driver.

    The cluster driver runs one ``nsys`` profile per backend, exports the kernel
    summary CSV, computes ``attention_share`` (the pure helper), and writes this
    map. Absent → shares are simply omitted (the timing run never needs nsys).
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Native LTX-2 retake attention-backend profiling.")
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
    p.add_argument(
        "--single-backend",
        default=None,
        help="internal: build+time ONE attention backend and print BACKEND_RESULT",
    )
    p.add_argument(
        "--nsys-capture",
        action="store_true",
        help="bracket the measured denoise with torch.cuda.profiler.start/stop "
        "(pair with nsys --capture-range=cudaProfilerApi so only the forward is traced)",
    )
    p.add_argument(
        "--nsys-shares",
        default=None,
        help="optional JSON {backend: attention_share} from the nsys driver to merge",
    )
    p.add_argument(
        "--attn-share-from-csv",
        default=None,
        help="pure mode: compute + print attention_share from an nsys kern-sum CSV",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # Pure post-processing mode: attention share from an nsys kernel CSV.
    if args.attn_share_from_csv:
        with open(args.attn_share_from_csv) as f:
            share = attention_share(parse_nsys_kern_csv(f.read()))
        print(json.dumps(_json_safe(share), indent=2))
        return 0

    if args.single_backend:
        return _run_single_backend(args)

    # Driver: one subprocess per backend, then deltas + optional nsys shares.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_commit = args.code_commit or os.environ.get("LTX2_ORACLE_CODE_COMMIT")

    records = []
    for backend in BACKENDS:
        print(f"[attn] {backend}: build + time ...", flush=True)
        rec = _spawn_single_backend(args, backend)
        print(f"[attn] {backend}: {rec.get('status')} ({rec.get('reason')})", flush=True)
        records.append(rec)

    # Only attach a profiler attention share to a backend that actually RAN.
    # A backend that errored (e.g. FA4 metadata not wired, CUTEDSL head_dim
    # ineligible) produces no valid denoise trace, so its nsys share would be a
    # meaningless leftover of the warmup/fallback rather than that backend's
    # attention time.
    shares = _load_nsys_shares(args.nsys_shares)
    for rec in records:
        if rec["backend"] in shares and rec.get("status") == "ok":
            rec["attention_share"] = shares[rec["backend"]]

    summary = summarize_profile(records)
    report = {
        "mode": "native_retake_attention_backend_profile",
        "device_query": _device_query(),
        "quant_attention": {
            "supported": False,
            "note": "Quantized attention is not implemented for the diffusion "
            "attention backends on this stack; attention runs in bf16 while the "
            "linear layers may be NVFP4/FP8-quantized.",
        },
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
            "attention_kernel_patterns": list(DEFAULT_ATTENTION_KERNEL_PATTERNS),
            "code_commit": code_commit,
        },
        "records": records,
        "summary": summary,
    }
    out_path = output_dir / "attn_profile.json"
    with open(out_path, "w") as f:
        json.dump(_json_safe(report), f, indent=2, allow_nan=False)
    print(f"ATTN_PROFILE_DONE {json.dumps(_json_safe(summary))}")
    print(f"wrote {out_path}")
    return 0


def _device_query() -> dict:
    """Best-effort device name + capability (empty on a CPU-only host)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "name": torch.cuda.get_device_name(0),
            "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
            "count": torch.cuda.device_count(),
        }
    except Exception:  # noqa: BLE001 - provenance only
        return {}


if __name__ == "__main__":
    raise SystemExit(main())
