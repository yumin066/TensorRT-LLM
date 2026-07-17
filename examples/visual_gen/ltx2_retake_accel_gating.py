#! /usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authoritative native LTX-2 retake acceleration-gating matrix.

The authoritative per-axis, capability-gated acceleration gate on the native
``LTXModel`` retake path. Unlike the cheap early-feasibility smoke sweep (which
only records WHICH configs execute end-to-end), this harness attributes clean
resident-warm latency and peak memory to each acceleration axis and then stacks
the axes in a fixed cumulative order to report the compounded speedup.

Two families of rows run against a single ``bf16/VANILLA`` no-graph/no-compile
baseline:

- INDEPENDENT single-axis rows, each toggling exactly ONE axis off the baseline
  (torch_compile, cuda_graph, an attention backend, or NVFP4 dynamic quant), so
  each axis's contribution is measured in isolation; and
- a fixed-order cumulative STACK (torch_compile, then + cuda_graph, then + the
  fastest attention backend measured above, then + NVFP4 when the capability is
  present) built dynamically AFTER the single-axis measurement so the stack picks
  the winning attention backend rather than a hard-coded one.

Every row records a status, resident-warm ``p50`` / ``p90`` / ``min``, per-stage
timing, and peak memory. PSNR / SSIM columns are informational only (each axis's
warm output vs the baseline warm output over the retake pixel window) and never
gate the matrix. Only the native pre/post path is exercised here; the
upstream-orchestration path is out of scope for acceleration gating.

The heavy pipeline build / run is reused from the sibling ``ltx2_retake_oracle``
module and the staged warm-timing helpers from ``ltx2_retake_timing`` (both
loaded by path), so there is a single source of truth for pipeline construction
and latency aggregation. The pure helpers at the top of this module
(``build_axis_matrix``, ``precheck_capability``, ``classify_status``,
``regresses_vs_baseline``, ``fastest_attention``, ``build_stack_configs``,
``incremental_deltas``, ``build_record``, ``baseline_required_ok``,
``summarize_gating``, ``_json_safe``) import on a plain CPU host with no numpy /
torch / tensorrt_llm.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

VALID_STATUSES = (
    "ok",
    "unsupported",
    "not-applicable",
    "graph-breaks",
    "recompiles",
    "regresses",
    "exceeds-VRAM",
    "error",
)

# Deterministic tie-break order when two attention backends measure the same
# steady-warm p50: the earlier backend in this tuple wins.
_ATTENTION_ORDER = ("FA4", "CUTEDSL", "VANILLA")


# ----------------------------------------------------------------------------
# Pure helpers (stdlib only; host-testable without numpy / torch / tensorrt_llm).
# ----------------------------------------------------------------------------


def build_axis_matrix() -> list:
    """The baseline + INDEPENDENT single-axis acceleration configs.

    The baseline is ``bf16/VANILLA`` with no cuda_graph and no torch_compile.
    Every other row toggles exactly ONE acceleration axis off that baseline so
    its contribution is attributable in isolation. The fixed cumulative stack is
    built later by :func:`build_stack_configs` once the fastest attention backend
    is known, so it is intentionally absent here.
    """
    return [
        {
            "label": "bf16/VANILLA",
            "dtype": "bf16",
            "attention_backend": "VANILLA",
            "cuda_graph": False,
            "torch_compile": False,
            "quant_algo": None,
            "baseline": True,
            "axis": "baseline",
            "kind": "baseline",
        },
        {
            "label": "bf16/torch_compile",
            "dtype": "bf16",
            "attention_backend": "VANILLA",
            "cuda_graph": False,
            "torch_compile": True,
            "quant_algo": None,
            "baseline": False,
            "axis": "torch_compile",
            "kind": "single-axis",
        },
        {
            "label": "bf16/cuda_graph",
            "dtype": "bf16",
            "attention_backend": "VANILLA",
            "cuda_graph": True,
            "torch_compile": False,
            "quant_algo": None,
            "baseline": False,
            "axis": "cuda_graph",
            "kind": "single-axis",
        },
        {
            "label": "bf16/FA4",
            "dtype": "bf16",
            "attention_backend": "FA4",
            "cuda_graph": False,
            "torch_compile": False,
            "quant_algo": None,
            "baseline": False,
            "axis": "attn",
            "kind": "single-axis",
        },
        {
            "label": "bf16/CUTEDSL",
            "dtype": "bf16",
            "attention_backend": "CUTEDSL",
            "cuda_graph": False,
            "torch_compile": False,
            "quant_algo": None,
            "baseline": False,
            "axis": "attn",
            "kind": "single-axis",
        },
        {
            "label": "NVFP4/VANILLA",
            "dtype": "nvfp4",
            "attention_backend": "VANILLA",
            "cuda_graph": False,
            "torch_compile": False,
            "quant_algo": "NVFP4",
            "baseline": False,
            "axis": "quant",
            "kind": "single-axis",
        },
    ]


def precheck_capability(config: dict, caps: dict) -> tuple:
    """Return ``(supported, reason)`` from statically-known capability flags.

    ``caps`` is gathered at runtime (imports) and injected so this stays pure.
    A missing flag defaults to "attempt" (``True``) -- the actual run is the
    ground truth. Only the capabilities we can cheaply pre-determine gate here:
    ``torch_compile`` and ``cuda_graph`` are always attempted (their capability
    is discovered at run rather than pre-judged).
    """
    backend = config.get("attention_backend")
    if backend == "FA4" and not caps.get("fa4", True):
        return False, "FA4 unavailable: flash_attn cute backend not importable on this build"
    if backend == "CUTEDSL" and not caps.get("cutedsl", True):
        return False, "CuTeDSL unavailable: requires precompiled cubin + head_dim=128 + fp16/bf16"
    if config.get("quant_algo") == "NVFP4" and not caps.get("nvfp4", True):
        return False, "NVFP4 dynamic quant unavailable: native DynamicLinearWeightLoader missing"
    return True, None


def classify_status(
    supported: bool,
    precheck_reason: Optional[str],
    ran_ok: bool,
    error_reason: Optional[str],
    applicable: bool = True,
    run_condition: Optional[tuple] = None,
) -> tuple:
    """Map a config's precheck + run outcome to a status + reason.

    - ``not-applicable`` = a prerequisite (the baseline) did not run, so this row
      was never built (``applicable=False``);
    - ``unsupported`` = a capability precheck ruled it out (never built);
    - ``error`` = it was attempted but the build / run raised;
    - a ``run_condition`` ``(status, reason)`` relays a run-observed degraded
      outcome (``graph-breaks`` / ``recompiles`` / ``regresses`` / ``exceeds-VRAM``);
    - otherwise ``ok``.
    """
    if not applicable:
        return "not-applicable", precheck_reason
    if not supported:
        return "unsupported", precheck_reason
    if not ran_ok:
        return "error", error_reason
    if run_condition is not None:
        status, reason = run_condition
        if status not in VALID_STATUSES:
            raise ValueError(
                f"invalid run_condition status {status!r}; expected one of {VALID_STATUSES}"
            )
        return status, reason
    return "ok", None


def regresses_vs_baseline(steady_p50, baseline_p50, tol: float = 1.0) -> bool:
    """Whether an axis's steady-warm p50 is slower than the baseline beyond *tol*.

    ``tol`` is a multiplicative margin: a row regresses when its steady p50
    exceeds ``baseline_p50 * tol`` (default ``1.0`` flags any slowdown). This is
    informational -- a regressing row is still recorded, it simply carries the
    ``regresses`` status.
    """
    if steady_p50 is None or baseline_p50 is None or baseline_p50 <= 0:
        return False
    return steady_p50 > baseline_p50 * tol


def fastest_attention(records: list) -> Optional[str]:
    """The attention backend with the lowest steady-warm p50 among ok rows.

    Considers the single-axis attention rows AND the baseline VANILLA row (all
    that ran ``ok``). Ties break deterministically by :data:`_ATTENTION_ORDER`.
    Returns ``None`` when no eligible row produced a steady-warm p50.
    """
    candidates = []
    for r in records:
        if r.get("status") != "ok":
            continue
        if not (r.get("baseline") or r.get("axis") == "attn"):
            continue
        p50 = (r.get("steady_warm") or {}).get("p50")
        if p50 is None:
            continue
        candidates.append((r.get("attention_backend"), p50))
    if not candidates:
        return None

    def _key(item):
        backend, p50 = item
        order = (
            _ATTENTION_ORDER.index(backend)
            if backend in _ATTENTION_ORDER
            else len(_ATTENTION_ORDER)
        )
        return (p50, order)

    return min(candidates, key=_key)[0]


def build_stack_configs(fastest_attn: Optional[str], caps: dict) -> list:
    """The fixed-order cumulative acceleration stack, one row per added axis.

    Order: torch_compile, then + cuda_graph, then + the fastest attention
    backend, then + NVFP4 dynamic quant. The NVFP4 row is only included when the
    capability is present (``caps['nvfp4']``); the others are always attempted.
    Each row carries ``axis='stack'``, ``kind='stack'``, and a 1-based ``step``.
    """
    attn = fastest_attn or "VANILLA"
    configs = [
        {
            "label": "stack:torch_compile",
            "dtype": "bf16",
            "attention_backend": "VANILLA",
            "cuda_graph": False,
            "torch_compile": True,
            "quant_algo": None,
            "baseline": False,
            "axis": "stack",
            "kind": "stack",
            "step": 1,
        },
        {
            "label": "stack:+cuda_graph",
            "dtype": "bf16",
            "attention_backend": "VANILLA",
            "cuda_graph": True,
            "torch_compile": True,
            "quant_algo": None,
            "baseline": False,
            "axis": "stack",
            "kind": "stack",
            "step": 2,
        },
        {
            "label": f"stack:+{attn}",
            "dtype": "bf16",
            "attention_backend": attn,
            "cuda_graph": True,
            "torch_compile": True,
            "quant_algo": None,
            "baseline": False,
            "axis": "stack",
            "kind": "stack",
            "step": 3,
        },
    ]
    if caps.get("nvfp4", True):
        configs.append(
            {
                "label": "stack:+NVFP4",
                "dtype": "nvfp4",
                "attention_backend": attn,
                "cuda_graph": True,
                "torch_compile": True,
                "quant_algo": "NVFP4",
                "baseline": False,
                "axis": "stack",
                "kind": "stack",
                "step": 4,
            }
        )
    return configs


def incremental_deltas(stack_records: list, baseline_p50) -> list:
    """Per-stack-row speedup ratios vs the previous step and vs the baseline.

    ``delta_vs_prev`` compares each cumulative step to the one before it (the
    first step compares to the baseline); ``delta_vs_baseline`` compares to the
    baseline directly. Ratios are ``prev_p50 / cur_p50`` so a value > 1 means
    faster. Missing p50s yield ``None`` and do not advance the running previous.
    """
    out = []
    prev = baseline_p50
    for r in stack_records:
        cur = (r.get("steady_warm") or {}).get("p50")
        out.append(
            {
                "label": r.get("label"),
                "step": r.get("step"),
                "steady_p50": cur,
                "delta_vs_prev": (prev / cur) if (cur and prev) else None,
                "delta_vs_baseline": (baseline_p50 / cur) if (cur and baseline_p50) else None,
            }
        )
        if cur:
            prev = cur
    return out


def build_record(
    config: dict,
    status: str,
    reason: Optional[str],
    timing: Optional[dict],
    quality: Optional[dict],
    memory: Optional[dict],
) -> dict:
    """Assemble one strict-JSON result row for the gating matrix."""
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status {status!r}; expected one of {VALID_STATUSES}")
    timing = timing or {}
    return {
        "label": config["label"],
        "dtype": config.get("dtype", "bf16"),
        "attention_backend": config.get("attention_backend"),
        "quant_algo": config.get("quant_algo"),
        "cuda_graph": bool(config.get("cuda_graph")),
        "torch_compile": bool(config.get("torch_compile")),
        "axis": config.get("axis"),
        "kind": config.get("kind"),
        "baseline": bool(config.get("baseline")),
        "step": config.get("step"),
        "status": status,
        "reason": reason,
        "raw_samples": list(timing.get("raw_samples") or []),
        "first_served": timing.get("first_served"),
        "steady_warm": timing.get("steady_warm"),
        "per_stage": timing.get("per_stage"),
        # Informational only (each axis's warm output vs the baseline warm
        # output); None for the baseline itself and for rows that did not run.
        "quality_informational": quality,
        "peak_memory": memory,
        "cold_model_build_load": timing.get("cold_model_build_load"),
    }


def baseline_required_ok(records: list) -> bool:
    """Whether the baseline row ran ``ok`` (the anchor every axis is gated on)."""
    for r in records:
        if r.get("baseline"):
            return r.get("status") == "ok"
    return False


def _baseline_p50(records: list):
    """The baseline row's steady-warm p50 (``None`` if it did not run ok)."""
    for r in records:
        if r.get("baseline") and r.get("status") == "ok":
            return (r.get("steady_warm") or {}).get("p50")
    return None


def summarize_gating(records: list) -> dict:
    """Count rows by status and report the fastest attention + stack speedup."""
    counts: dict = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    baseline_p50 = _baseline_p50(records)
    stack_rows = [r for r in records if r.get("kind") == "stack" and r.get("status") == "ok"]
    stack_speedup = None
    if stack_rows and baseline_p50:
        full = max(stack_rows, key=lambda r: r.get("step") or 0)
        p50 = (full.get("steady_warm") or {}).get("p50")
        if p50:
            stack_speedup = baseline_p50 / p50
    return {
        "total": len(records),
        "by_status": counts,
        "baseline_ok": baseline_required_ok(records),
        "fastest_attention": fastest_attention(records),
        "stack_speedup_vs_baseline": stack_speedup,
    }


def _json_safe(obj):
    """Recursively replace non-finite floats with string sentinels for strict JSON.

    An informational PSNR of ``inf`` (an axis's output is bit-identical to the
    baseline) is a real signal, but ``json.dumps`` writes it as the literal
    ``Infinity`` which strict (non-Python) JSON parsers reject. Convert
    ``inf`` / ``-inf`` / ``nan`` to string sentinels so the emitted matrix is
    strict-RFC valid while preserving the signal.
    """
    import math

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


def _load_timing():
    """Load the sibling ``ltx2_retake_timing`` module by path for its warm-timing helpers."""
    timing_path = Path(__file__).resolve().with_name("ltx2_retake_timing.py")
    spec = importlib.util.spec_from_file_location("ltx2_retake_timing", timing_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gather_caps() -> dict:
    """Probe which acceleration capabilities are importable in this environment."""
    caps = {"fa4": False, "cutedsl": False, "nvfp4": False}
    try:
        import flash_attn.cute  # noqa: F401

        caps["fa4"] = True
    except Exception as exc:  # noqa: BLE001 - capability probe records the reason
        caps["fa4_reason"] = f"{type(exc).__name__}: {exc}"
    try:
        from tensorrt_llm._torch.visual_gen.quantization.loader import (  # noqa: F401
            DynamicLinearWeightLoader,
        )

        caps["nvfp4"] = True
    except Exception as exc:  # noqa: BLE001
        caps["nvfp4_reason"] = f"{type(exc).__name__}: {exc}"
    try:
        import cutlass  # noqa: F401

        caps["cutedsl"] = True
    except Exception as exc:  # noqa: BLE001
        caps["cutedsl_reason"] = f"{type(exc).__name__}: {exc}"
    return caps


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Authoritative native LTX-2 retake acceleration-gating matrix."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--gemma", required=True)
    p.add_argument("--lora", default=None)
    p.add_argument("--source", required=True, help="video-only source clip")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument(
        "--warmup", type=int, default=2, help="N warmup requests per config (not measured)"
    )
    p.add_argument("--measured", type=int, default=10, help="M measured requests per config")
    p.add_argument("--prompt", default="a person talking to the camera")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--code-commit", default=None, help="authoritative code commit for provenance")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    oracle = _load_oracle()
    timing = _load_timing()

    import gc
    import time

    import torch

    # ``_retake_pixel_window`` is a helper in the retake pipeline module; import
    # it here for the informational-quality window-geometry conversion.
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_retake import _retake_pixel_window

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    code_commit = args.code_commit or os.environ.get("LTX2_ORACLE_CODE_COMMIT")

    source_frames = oracle.read_source_frames(args.source)
    num_frames = int(source_frames.shape[0])

    caps = _gather_caps()
    env = oracle._env_metadata()
    request = oracle._make_request(
        args.prompt, args.negative_prompt, args.source, args.start, args.end, args.seed, args.steps
    )

    state = {"baseline_thwc": None, "pixel_window": None}

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _measure(config):
        """Build the native pipeline for one config, warm it, and measure M retakes.

        Returns ``(timing_data, quality, memory)``; raises on build / run failure
        so the caller records ``error`` and continues. The pipe is always freed in
        ``finally`` (near-capacity 22b runs OOM otherwise).
        """
        label = config["label"]
        pipe = None
        try:
            _sync()
            t0 = time.perf_counter()
            pipe = oracle.build_pipeline(
                args.checkpoint,
                args.gemma,
                args.lora,
                device,
                config["attention_backend"],
                cuda_graph_enable=config["cuda_graph"],
                quant_algo=config["quant_algo"],
                torch_compile_enable=config["torch_compile"],
            )
            _sync()
            cold_model_build_load = time.perf_counter() - t0

            for _ in range(max(0, args.warmup)):
                pipe.infer(request)
                _sync()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            measured = []
            last_thwc = None
            last_fps = None
            for i in range(args.measured):
                _sync()
                w0 = time.perf_counter()
                out = pipe.infer(request)
                _sync()
                wall = time.perf_counter() - w0
                rec = {
                    "index": i,
                    "wall": wall,
                    "pre_denoise": float(out.pre_denoise),
                    "denoise": float(out.denoise),
                    "post_denoise": float(out.post_denoise),
                    "stage_timings": out.stage_timings,
                }
                measured.append(rec)
                last_thwc = oracle._video_to_thwc_uint8(out.video)
                last_fps = float(out.frame_rate)
                print(f"[accel] {label} measured {i}: wall={wall:.3f}s", flush=True)

            memory = None
            if torch.cuda.is_available():
                memory = {
                    "allocated": int(torch.cuda.max_memory_allocated()),
                    "reserved": int(torch.cuda.max_memory_reserved()),
                }

            first, steady = timing.split_measured(measured)
            steady_source = steady if steady else measured
            steady_warm = timing.summarize_samples([r["wall"] for r in steady_source])
            per_stage = timing.aggregate_fine_stages(steady_source)
            timing_data = {
                "raw_samples": [r["wall"] for r in measured],
                "first_served": first["wall"] if first else None,
                "steady_warm": steady_warm,
                "per_stage": per_stage,
                "cold_model_build_load": cold_model_build_load,
            }

            # Window geometry (once) + informational quality vs the baseline warm
            # output over the retake pixel window (never gates).
            if state["pixel_window"] is None and last_fps is not None:
                state["pixel_window"] = _retake_pixel_window(
                    args.start, args.end, last_fps, num_frames
                )
            quality = None
            if config.get("baseline"):
                state["baseline_thwc"] = last_thwc
            elif state["baseline_thwc"] is not None and last_thwc is not None:
                quality = oracle.compute_similarity(
                    last_thwc, state["baseline_thwc"], state["pixel_window"], f"{label}_vs_baseline"
                )
            return timing_data, quality, memory
        finally:
            if pipe is not None:
                del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _run_and_record(config, baseline_p50=None):
        label = config["label"]
        supported, precheck_reason = precheck_capability(config, caps)
        if not supported:
            print(f"[accel] {label}: unsupported ({precheck_reason})", flush=True)
            return build_record(config, "unsupported", precheck_reason, None, None, None)
        print(f"[accel] {label}: building + measuring...", flush=True)
        try:
            timing_data, quality, memory = _measure(config)
        except Exception as exc:  # noqa: BLE001 - per-config isolation: record and continue
            error_reason = f"{type(exc).__name__}: {exc}"
            print(f"[accel] {label}: error ({error_reason})", file=sys.stderr, flush=True)
            traceback.print_exc()
            status, reason = classify_status(supported, precheck_reason, False, error_reason)
            return build_record(config, status, reason, None, None, None)
        run_condition = None
        if not config.get("baseline") and baseline_p50 is not None:
            p50 = (timing_data.get("steady_warm") or {}).get("p50")
            if regresses_vs_baseline(p50, baseline_p50):
                run_condition = (
                    "regresses",
                    f"steady p50 {p50:.3f}s slower than baseline {baseline_p50:.3f}s",
                )
        status, reason = classify_status(
            supported, precheck_reason, True, None, run_condition=run_condition
        )
        print(f"[accel] {label}: {status}", flush=True)
        return build_record(config, status, reason, timing_data, quality, memory)

    def _emit(records, fastest_attn, deltas):
        summary = summarize_gating(records)
        native_repo = oracle._git_info(str(Path(__file__).resolve().parents[2]))
        result = {
            "records": records,
            "summary": summary,
            "fastest_attention": fastest_attn,
            "stack_incremental_deltas": deltas,
            "capabilities": caps,
            "env": env,
            "code_commit": code_commit,
            "native_repo": native_repo,
            "source": str(Path(args.source).resolve()),
            "source_sha256": oracle.sha256_file(args.source),
            "window": [args.start, args.end],
            "seed": args.seed,
            "steps": args.steps,
            "warmup": max(0, args.warmup),
            "measured": args.measured,
            "num_frames": num_frames,
            "note": (
                "Authoritative per-axis capability-gated acceleration matrix on the "
                "native LTX-2 retake path: baseline-anchored single-axis attribution "
                "plus a fixed-order cumulative acceleration stack. PSNR/SSIM columns "
                "are informational only and never gate."
            ),
        }
        (output_dir / "accel_gating.json").write_text(
            json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False)
        )
        print("ACCEL_GATING_DONE", json.dumps(_json_safe(summary), allow_nan=False))
        return summary

    matrix = build_axis_matrix()
    baseline_config = next(c for c in matrix if c.get("baseline"))
    single_axis_configs = [c for c in matrix if c.get("kind") == "single-axis"]

    records = []

    # ---- Baseline first (the anchor every axis is gated on) ------------
    baseline_record = _run_and_record(baseline_config)
    records.append(baseline_record)
    if baseline_record["status"] != "ok":
        # Baseline failed: dependent rows are not built; they are recorded as
        # not-applicable and the gate fails.
        for c in single_axis_configs:
            records.append(
                build_record(
                    c,
                    "not-applicable",
                    "baseline did not run ok; dependent acceleration axis not measured",
                    None,
                    None,
                    None,
                )
            )
        _emit(records, None, [])
        print("[accel] baseline did not run ok; acceleration axes not measured", file=sys.stderr)
        return 1

    baseline_p50 = (baseline_record.get("steady_warm") or {}).get("p50")

    # ---- INDEPENDENT single-axis rows ----------------------------------
    for c in single_axis_configs:
        records.append(_run_and_record(c, baseline_p50=baseline_p50))

    # ---- Fixed-order cumulative stack (built after measurement) --------
    fastest_attn = fastest_attention(records)
    stack_records = []
    for c in build_stack_configs(fastest_attn, caps):
        rec = _run_and_record(c, baseline_p50=baseline_p50)
        stack_records.append(rec)
        records.append(rec)
    deltas = incremental_deltas(stack_records, baseline_p50)

    _emit(records, fastest_attn, deltas)

    # A meaningful gate requires the baseline anchor plus no run-time errors on
    # any row that was expected to run.
    success = baseline_required_ok(records) and not any(r["status"] == "error" for r in records)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
