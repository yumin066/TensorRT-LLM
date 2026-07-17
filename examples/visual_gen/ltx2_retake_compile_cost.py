#! /usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native LTX-2 retake torch.compile compile/autotune cost split.

Isolates the one-time ``torch.compile`` compile + Inductor/Triton autotune cost
on the native ``LTXModel`` retake path from steady resident-warm latency, across
three on-disk-cache situations:

- an empty-cache FIRST process pays the full compile + autotune cost on its very
  first retake (measured separately from steady warm);
- a same-process LATER retake in that same process is steady warm (the compile is
  already done -- this is the steady series of the empty-cache process, never a
  separate process); and
- a warm-disk-cache NEW process, launched after the on-disk cache is populated,
  pays only the cache-load cost on its first retake (no recompile / re-autotune).

The one-time first-call compile cost is kept SEPARATE from the steady warm
``p50`` / ``p90`` / ``min`` so the compile tax is never averaged into resident
latency. Each cache situation runs in its OWN process (compile state is process-
global and the multi-GB model is released on exit); a lightweight orchestrator
spawns the two processes, parses the JSON result line each prints, and writes the
merged split.

The heavy pipeline build / run is reused from the sibling ``ltx2_retake_oracle``
module and the warm-latency summary helpers from ``ltx2_retake_timing`` (both
loaded by path). The pure helpers at the top of this module
(``set_compile_cache_env``, ``build_cache_plan``, ``parse_cache_mode_result``,
``derived_costs``, ``summarize_compile_cost``, ``_json_safe``) import on a plain
CPU host with no numpy / torch / tensorrt_llm.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional

# ----------------------------------------------------------------------------
# Pure helpers (stdlib only; host-testable without numpy / torch / tensorrt_llm).
# ----------------------------------------------------------------------------


def set_compile_cache_env(cache_dir) -> dict:
    """Point the Inductor / Triton on-disk caches at *cache_dir* subtrees.

    Sets ``TORCHINDUCTOR_CACHE_DIR`` -> ``<cache_dir>/inductor`` and
    ``TRITON_CACHE_DIR`` -> ``<cache_dir>/triton`` (both created ``mkdir -p``) and
    returns the two resolved paths. MUST be called BEFORE ``import torch`` so the
    compile backend reads and writes the intended cache location; setting it after
    torch has cached the env has no effect.
    """
    cache_root = Path(cache_dir)
    inductor = cache_root / "inductor"
    triton = cache_root / "triton"
    inductor.mkdir(parents=True, exist_ok=True)
    triton.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor)
    os.environ["TRITON_CACHE_DIR"] = str(triton)
    return {"inductor": str(inductor), "triton": str(triton)}


def build_cache_plan() -> list:
    """The three logical compile-cost measurements across the cache situations.

    Only two SUBPROCESS modes exist (``empty`` / ``warm``): the "same-process
    later request" measurement is the STEADY series of the empty-cache process,
    not a separate subprocess, so it carries ``mode='empty'``.
    """
    return [
        {
            "mode": "empty",
            "label": "empty_cache_first_call",
            "description": (
                "first retake in a fresh process with an empty on-disk cache: pays "
                "the full torch.compile compile + Inductor/Triton autotune cost"
            ),
        },
        {
            "mode": "empty",
            "label": "same_process_steady",
            "description": (
                "later retakes in that same empty-cache process: steady resident "
                "warm with no compile (the steady series of the empty-cache process)"
            ),
        },
        {
            "mode": "warm",
            "label": "warm_disk_cache_first_call",
            "description": (
                "first retake in a NEW process after the on-disk cache is populated: "
                "pays only the cache-load cost, no recompile or re-autotune"
            ),
        },
    ]


def parse_cache_mode_result(stdout: str) -> Optional[dict]:
    """Extract the LAST ``CACHE_MODE_RESULT`` JSON a subprocess prints.

    A single-mode subprocess prints one ``CACHE_MODE_RESULT <json>`` line; the
    last matching line wins (so a trailing failure line supersedes an earlier
    partial one). Returns ``None`` when no line is found or the last one is not
    valid JSON.
    """
    result = None
    prefix = "CACHE_MODE_RESULT "
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(prefix):
            try:
                result = json.loads(line[len(prefix) :])
            except json.JSONDecodeError:
                result = None
    return result


def derived_costs(empty_result, warm_result) -> dict:
    """Derive the compile / cache-load cost split from the two mode results.

    - ``compile_cost_seconds`` = empty first_call - empty steady ``p50`` (the
      one-time compile + autotune tax the first retake pays);
    - ``warm_disk_first_seconds`` = warm first_call (cache-load only);
    - ``cache_saved_seconds`` = empty first_call - warm first_call (what a warm
      on-disk cache saves a fresh process);
    - ``steady_p50`` = empty steady ``p50`` (resident warm baseline).

    Any field whose inputs are missing (a failed / absent mode result) is
    ``None`` rather than raising, so a partial run still emits a well-formed split.
    """

    def _first(res):
        return res.get("first_call") if isinstance(res, dict) else None

    def _steady_p50(res):
        if not isinstance(res, dict):
            return None
        steady = res.get("steady") or {}
        return steady.get("p50")

    empty_first = _first(empty_result)
    empty_steady_p50 = _steady_p50(empty_result)
    warm_first = _first(warm_result)

    compile_cost = (
        empty_first - empty_steady_p50
        if empty_first is not None and empty_steady_p50 is not None
        else None
    )
    cache_saved = (
        empty_first - warm_first if empty_first is not None and warm_first is not None else None
    )
    return {
        "compile_cost_seconds": compile_cost,
        "warm_disk_first_seconds": warm_first,
        "cache_saved_seconds": cache_saved,
        "steady_p50": empty_steady_p50,
    }


def summarize_compile_cost(records, derived) -> dict:
    """Roll the two mode results + the derived block into a compact summary."""
    present = [r.get("mode") for r in records if isinstance(r, dict)]
    all_ok = bool(records) and all(isinstance(r, dict) and r.get("ok") for r in records)
    return {
        "modes": present,
        "ok": all_ok,
        "fail": not all_ok,
        "derived": derived,
    }


def _json_safe(obj):
    """Recursively replace non-finite floats with string sentinels for strict JSON.

    A degenerate timing (e.g. ``inf`` / ``nan`` slipping through a divide) would
    make ``json.dumps`` emit the literal ``Infinity`` which strict (non-Python)
    JSON parsers reject. Convert ``inf`` / ``-inf`` / ``nan`` to string sentinels
    so the emitted split is strict-RFC valid.
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
        description="Native LTX-2 retake torch.compile compile/autotune cost split."
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
    p.add_argument("--warmup", type=int, default=0, help="reserved (single-mode uses no warmup)")
    p.add_argument(
        "--measured", type=int, default=5, help="M same-process steady retakes after the first"
    )
    p.add_argument("--prompt", default="a person talking to the camera")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--code-commit", default=None, help="authoritative code commit for provenance")
    p.add_argument(
        "--single-mode",
        choices=("empty", "warm"),
        default=None,
        help="internal: run ONE cache-mode measurement and print CACHE_MODE_RESULT",
    )
    p.add_argument(
        "--cache-dir", default=None, help="internal: shared on-disk compile cache dir for a mode"
    )
    return p.parse_args(argv)


def run_single_mode(args) -> int:
    """Measure one cache mode's first-call + steady latency in this process.

    ``empty`` clears the on-disk cache first so the process pays the full compile
    + autotune on its first retake; ``warm`` leaves the already-populated cache
    intact so the first retake pays only cache-load. The first retake is timed
    SEPARATELY from the ``--measured`` steady retakes that follow (which pay no
    compile). Prints exactly one ``CACHE_MODE_RESULT`` line and exits nonzero on
    failure so the orchestrator records the mode as failed.
    """
    # Point the on-disk caches at the shared cache dir BEFORE torch is imported.
    set_compile_cache_env(args.cache_dir)
    mode = args.single_mode
    if mode == "empty":
        # An empty cache forces this process to pay the full compile + autotune.
        shutil.rmtree(args.cache_dir, ignore_errors=True)
        set_compile_cache_env(args.cache_dir)

    oracle = _load_oracle()
    timing = _load_timing()

    import gc
    import time

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    code_commit = args.code_commit or os.environ.get("LTX2_ORACLE_CODE_COMMIT")

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    first_call = None
    steady = None
    raw_samples = []
    cold_build_load = None
    peak_memory = None
    ok = False
    reason = None
    pipe = None
    try:
        _sync()
        t0 = time.perf_counter()
        pipe = oracle.build_pipeline(
            args.checkpoint,
            args.gemma,
            args.lora,
            device,
            "VANILLA",
            torch_compile_enable=True,
        )
        _sync()
        cold_build_load = time.perf_counter() - t0

        request = oracle._make_request(
            args.prompt,
            args.negative_prompt,
            args.source,
            args.start,
            args.end,
            args.seed,
            args.steps,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # First retake: pays compile + autotune (empty) or cache-load (warm).
        _sync()
        f0 = time.perf_counter()
        pipe.infer(request)
        _sync()
        first_call = time.perf_counter() - f0
        print(f"[compile_cost] {mode} first_call: {first_call:.3f}s", flush=True)

        # Steady series: same-process later retakes (compile already done).
        steady_walls = []
        for i in range(max(0, args.measured)):
            _sync()
            w0 = time.perf_counter()
            pipe.infer(request)
            _sync()
            wall = time.perf_counter() - w0
            steady_walls.append(wall)
            print(f"[compile_cost] {mode} steady {i}: wall={wall:.3f}s", flush=True)
        raw_samples = steady_walls
        steady = timing.summarize_samples(steady_walls)

        if torch.cuda.is_available():
            peak_memory = {
                "allocated": int(torch.cuda.max_memory_allocated()),
                "reserved": int(torch.cuda.max_memory_reserved()),
            }
        ok = True
    except Exception as exc:  # noqa: BLE001 - per-mode isolation: record and exit nonzero
        reason = f"{type(exc).__name__}: {exc}"
        print(f"[compile_cost] {mode}: error ({reason})", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        if pipe is not None:
            del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "mode": mode,
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "first_call": first_call,
        "steady": steady,
        "raw_samples": raw_samples,
        "cold_build_load": cold_build_load,
        "peak_memory": peak_memory,
        "ok": ok,
        "reason": reason,
        "code_commit": code_commit,
        # Each subprocess carries its own env so the orchestrator (which stays
        # torch-free) can copy it without importing torch itself.
        "env": oracle._env_metadata(),
        "capabilities": _gather_caps(),
    }
    print("CACHE_MODE_RESULT " + json.dumps(_json_safe(result), allow_nan=False), flush=True)
    return 0 if ok else 1


def _spawn_mode(args, mode: str, cache_dir) -> Optional[dict]:
    """Run one cache-mode measurement in a fresh subprocess; return its result.

    A fresh process is required for a faithful measurement: compile state is
    process-global (an in-process ``empty`` then ``warm`` would already be warm),
    and the multi-GB model is only fully released on process exit.
    """
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-mode",
        mode,
        "--cache-dir",
        str(cache_dir),
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
        "--measured",
        str(args.measured),
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
    ]
    if args.lora:
        cmd += ["--lora", args.lora]
    if args.code_commit:
        cmd += ["--code-commit", args.code_commit]
    # Inherit LD_LIBRARY_PATH / PYTHONPATH (os.environ.copy) and force unbuffered
    # stdout so the CACHE_MODE_RESULT line is not lost on a crash.
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.stdout:
        print(proc.stdout, end="")
    result = parse_cache_mode_result(proc.stdout)
    if result is None:
        print(
            f"[compile_cost] {mode} subprocess produced no CACHE_MODE_RESULT "
            f"(rc={proc.returncode}); stderr tail: {proc.stderr[-800:]}",
            file=sys.stderr,
        )
    return result


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.single_mode is not None:
        return run_single_mode(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The orchestrator stays torch-free and lightweight: it only spawns the two
    # measurement subprocesses, parses their printed JSON, and writes the merged
    # split. Env metadata is taken from the subprocess result (oracle._env_metadata
    # imports torch); the oracle is loaded here only for the stdlib-only git /
    # sha256 provenance helpers, which do not import torch.
    cache_dir = output_dir / "compile_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    empty_result = _spawn_mode(args, "empty", cache_dir)
    warm_result = _spawn_mode(args, "warm", cache_dir)

    derived = derived_costs(empty_result, warm_result)
    records = [empty_result, warm_result]
    summary = summarize_compile_cost(records, derived)

    oracle = _load_oracle()  # stdlib-only helpers (_git_info / sha256_file); no torch import
    native_repo = oracle._git_info(str(Path(__file__).resolve().parents[2]))
    env = (empty_result or {}).get("env") or (warm_result or {}).get("env")
    code_commit = args.code_commit or os.environ.get("LTX2_ORACLE_CODE_COMMIT")

    result = {
        "records": records,
        "derived": derived,
        "summary": summary,
        "config": {
            "attention_backend": "VANILLA",
            "dtype": "bf16",
            "torch_compile": True,
            "cuda_graph": False,
            "quant_algo": None,
            "retake_offload_mode": "none",
        },
        "cache_plan": build_cache_plan(),
        "env": env,
        "code_commit": code_commit,
        "native_repo": native_repo,
        "source": str(Path(args.source).resolve()),
        "source_sha256": oracle.sha256_file(args.source),
        "window": [args.start, args.end],
        "seed": args.seed,
        "steps": args.steps,
        "measured": args.measured,
        "cache_dir": str(cache_dir.resolve()),
        "note": (
            "torch.compile compile/autotune cost split on the native LTX-2 retake path. "
            "The empty-cache first process pays the full compile + autotune (first_call); "
            "its later same-process retakes are steady warm; a warm-disk-cache new process "
            "pays only cache-load on its first_call. compile_cost_seconds = empty first_call "
            "minus empty steady p50; cache_saved_seconds = empty first_call minus warm "
            "first_call. The one-time first-call cost is kept separate from steady p50/p90/min."
        ),
    }
    (output_dir / "compile_cost.json").write_text(
        json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False)
    )
    print("COMPILE_COST_DONE " + json.dumps(_json_safe(summary), allow_nan=False))
    both_ok = (
        bool(empty_result)
        and bool(warm_result)
        and empty_result.get("ok")
        and warm_result.get("ok")
    )
    # A meaningful split needs both mode processes to have run ok and a derivable
    # one-time compile cost (empty first_call minus empty steady p50).
    return 0 if (both_ok and derived.get("compile_cost_seconds") is not None) else 1


if __name__ == "__main__":
    sys.exit(main())
