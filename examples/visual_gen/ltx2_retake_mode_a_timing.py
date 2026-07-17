#! /usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mode A (every-rebuild) upstream retake reference timing.

Mode A is the reference behavior: NO persistent model. For each measured call
this tool rebuilds the upstream retake pipeline from scratch (the preserved
upstream ``DiffusionStage.run`` path, ``retake_use_upstream_stage=True``), runs
one retake, and frees it — so every call pays the full ``model_build_load`` cost.
It reports per-call ``model_build_load`` / ``run_total`` / ``total`` (p50 / p90 /
min), env + provenance, an informational determinism quality column, and a
before/after ``git status`` check proving the pristine ``../LTX2.3-eval/packages/``
reference is unchanged.

It emits the same latency-summary shape as the resident-warm native harness
(``ltx2_retake_timing.py``) so Mode A (every-rebuild) and Mode B (load-once,
serve-many) are directly comparable: Mode A pays ``model_build_load`` on EVERY
call, while Mode B pays it once and then serves warm. Pass the measured Mode B
steady-warm p50 via ``--mode-b-warm-p50-seconds`` to record the speedup.

Heavy build/run is reused from the sibling ``ltx2_retake_oracle`` module and the
percentile / JSON helpers from ``ltx2_retake_timing`` (both loaded by path). The
pure helpers here import on a plain CPU host with no numpy / torch / tensorrt_llm.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

_SUMMARY_KEYS = ("model_build_load", "run_total", "total")


# ----------------------------------------------------------------------------
# Pure helpers (stdlib only; host-testable without numpy / torch / tensorrt_llm).
# ----------------------------------------------------------------------------


def build_call_record(index: int, model_build_load: float, run_total: float) -> dict:
    """One every-rebuild call's per-stage seconds."""
    return {
        "index": index,
        "model_build_load": float(model_build_load),
        "run_total": float(run_total),
        "total": float(model_build_load) + float(run_total),
    }


def summarize_mode_a(records: list, summarize_fn, keys=_SUMMARY_KEYS) -> dict:
    """Per-key ``summarize_fn`` (p50/p90/min/count) across the measured rebuilds.

    ``summarize_fn`` is injected (the timing harness's ``summarize_samples``) so
    this stays a pure, dependency-light helper.
    """
    out = {}
    for key in keys:
        samples = [r[key] for r in records if key in r and r[key] is not None]
        if samples:
            out[key] = summarize_fn(samples)
    return out


def speedup_vs_warm(
    mode_a_total_p50: Optional[float], mode_b_warm_p50: Optional[float]
) -> Optional[dict]:
    """Mode A every-rebuild per-call total vs Mode B resident-warm p50."""
    if not mode_a_total_p50 or not mode_b_warm_p50 or mode_b_warm_p50 <= 0:
        return None
    return {
        "mode_a_total_p50_seconds": float(mode_a_total_p50),
        "mode_b_warm_p50_seconds": float(mode_b_warm_p50),
        "mode_b_speedup_x": float(mode_a_total_p50) / float(mode_b_warm_p50),
        "note": (
            "Mode A rebuilds the model on every call (model_build_load per call); "
            "Mode B pays model_build_load once then serves warm. Speedup is the "
            "per-warm-call latency ratio (amortization of the one-time load is separate)."
        ),
    }


def packages_pristine(status_before: str, status_after: str) -> dict:
    """Whether the pristine reference packages are unchanged (and were clean)."""
    before = (status_before or "").strip()
    after = (status_after or "").strip()
    return {
        "unchanged": before == after,
        "clean": after == "",
        "status_before": before,
        "status_after": after,
    }


def mode_a_exit_ok(has_records: bool, pristine: dict) -> bool:
    """Whether the CLI should exit success: a measured call AND pristine packages.

    "Pristine" means both unchanged by this tool AND clean (no pre-existing
    dirt): a dirty-but-unchanged ``../LTX2.3-eval/packages/`` must FAIL, because
    the plan requires the reference tree to be pristine, not merely untouched.
    """
    return bool(has_records) and bool(pristine.get("unchanged")) and bool(pristine.get("clean"))


# ----------------------------------------------------------------------------
# Heavy path (imports live here so the pure helpers load on a plain host).
# ----------------------------------------------------------------------------


def _load_sibling(name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().with_name(f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_status_porcelain(repo_dir: str, subpath: str = "packages") -> str:
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "status", "--porcelain", "--", subpath],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return out.stdout
    except (subprocess.SubprocessError, OSError) as exc:  # noqa: BLE001 - record, don't crash
        return f"<git status failed: {type(exc).__name__}: {exc}>"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Mode A (every-rebuild) upstream retake reference timing."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--gemma", required=True)
    p.add_argument("--lora", default=None)
    p.add_argument("--source", required=True, help="video-only source clip")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--eval-repo", required=True, help="LTX2.3-eval repo (pristine packages check)")
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--warmup", type=int, default=1, help="N warmup rebuilds (not measured)")
    p.add_argument("--measured", type=int, default=3, help="M measured every-rebuild calls")
    p.add_argument("--prompt", default="a person talking to the camera")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--mode-b-warm-p50-seconds", type=float, default=None)
    p.add_argument("--code-commit", default=None)
    p.add_argument(
        "--single-call",
        action="store_true",
        help="internal: build+run+free ONE upstream rebuild and print CALL_RESULT",
    )
    return p.parse_args(argv)


def parse_call_result(stdout: str) -> Optional[dict]:
    """Extract the ``CALL_RESULT`` JSON a ``--single-call`` subprocess prints."""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("CALL_RESULT "):
            try:
                return json.loads(line[len("CALL_RESULT ") :])
            except json.JSONDecodeError:
                return None
    return None


def _run_single_call(args) -> int:
    """Build the upstream-stage pipeline once, run one retake, print CALL_RESULT.

    Each Mode A call runs in its OWN process so the ~92 GB model is fully released
    on exit — an in-process rebuild loop leaks and OOMs on the second build. A
    fresh process is also exactly the every-rebuild reference (re-running the eval
    from scratch), so this is the faithful measurement, not just a workaround.
    """
    oracle = _load_sibling("ltx2_retake_oracle")

    import hashlib
    import time

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    _sync()
    t0 = time.perf_counter()
    pipe = oracle.build_pipeline(
        args.checkpoint,
        args.gemma,
        args.lora,
        device,
        "VANILLA",
        extra_overrides={"retake_use_upstream_stage": True},
    )
    _sync()
    model_build_load = time.perf_counter() - t0

    request = oracle._make_request(
        args.prompt, args.negative_prompt, args.source, args.start, args.end, args.seed, args.steps
    )
    _sync()
    t1 = time.perf_counter()
    video, _audio, _fps, _sr = oracle._run_pipeline(pipe, request)
    _sync()
    run_total = time.perf_counter() - t1

    thwc = oracle._video_to_thwc_uint8(video)
    sha = hashlib.sha256(thwc.detach().to("cpu").contiguous().numpy().tobytes()).hexdigest()
    print(
        "CALL_RESULT "
        + json.dumps(
            {
                "model_build_load": model_build_load,
                "run_total": run_total,
                "output_sha256": sha,
            }
        ),
        flush=True,
    )
    return 0


def _spawn_single_call(args) -> dict:
    """Run one every-rebuild call in a fresh subprocess; return its CALL_RESULT."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-call",
        "--checkpoint",
        args.checkpoint,
        "--gemma",
        args.gemma,
        "--source",
        args.source,
        "--output-dir",
        args.output_dir,
        "--eval-repo",
        args.eval_repo,
        "--start",
        str(args.start),
        "--end",
        str(args.end),
        "--seed",
        str(args.seed),
        "--steps",
        str(args.steps),
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
    ]
    if args.lora:
        cmd += ["--lora", args.lora]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    res = parse_call_result(proc.stdout)
    if res is None:
        raise RuntimeError(
            f"Mode A single-call failed (rc={proc.returncode}); stderr tail: {proc.stderr[-800:]}"
        )
    return res


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.single_call:
        return _run_single_call(args)

    oracle = _load_sibling("ltx2_retake_oracle")
    timing = _load_sibling("ltx2_retake_timing")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_commit = args.code_commit or os.environ.get("LTX2_ORACLE_CODE_COMMIT")

    eval_repo = str(Path(args.eval_repo).resolve())
    status_before = _git_status_porcelain(eval_repo)

    # Warmup rebuilds (not measured) — each in its own subprocess.
    for _ in range(max(0, args.warmup)):
        _spawn_single_call(args)

    records = []
    shas = []
    for i in range(args.measured):
        res = _spawn_single_call(args)
        records.append(build_call_record(i, res["model_build_load"], res["run_total"]))
        shas.append(res.get("output_sha256"))
        print(
            f"[mode_a] measured {i}: build={res['model_build_load']:.1f}s "
            f"run={res['run_total']:.3f}s",
            flush=True,
        )

    status_after = _git_status_porcelain(eval_repo)

    quality = None
    if shas:
        quality = {
            "label": "rebuild_output_determinism",
            "all_identical": len(set(shas)) == 1,
            "output_sha256": shas[0],
            "note": "byte-identical outputs across every-rebuild calls (seed-42)",
        }

    summary = summarize_mode_a(records, timing.summarize_samples)
    total_p50 = summary.get("total", {}).get("p50")
    comparison = speedup_vs_warm(total_p50, args.mode_b_warm_p50_seconds)
    pristine = packages_pristine(status_before, status_after)

    native_repo = oracle._git_info(str(Path(__file__).resolve().parents[2]))
    result = {
        "mode": "mode_a_every_rebuild_upstream_retake",
        "summary": summary,
        "mode_b_comparison": comparison,
        "packages_pristine": pristine,
        "records": records,
        "warmup_iterations": max(0, args.warmup),
        "measured_iterations": args.measured,
        "quality_informational": quality,
        "env": oracle._env_metadata(),
        "code_commit": code_commit,
        "native_repo": native_repo,
        "eval_repo": oracle._git_info(eval_repo),
        "source": str(Path(args.source).resolve()),
        "source_sha256": oracle.sha256_file(args.source),
        "window": [args.start, args.end],
        "seed": args.seed,
        "steps": args.steps,
        "note": (
            "Mode A reference: the upstream retake pipeline is rebuilt from scratch on "
            "EVERY call (no persistent model), so model_build_load recurs per call. "
            "Compare against the resident-warm Mode B native harness (ltx2_retake_timing.py). "
            "The pristine ../LTX2.3-eval/packages/ tree is unchanged (packages_pristine)."
        ),
    }
    (output_dir / "mode_a_timing.json").write_text(
        json.dumps(timing._json_safe(result), indent=2, sort_keys=True, allow_nan=False)
    )
    print(
        "MODE_A_DONE",
        json.dumps({"total": summary.get("total"), "pristine": pristine["unchanged"]}),
    )
    # A meaningful run needs at least one measured rebuild and a pristine packages tree.
    return 0 if mode_a_exit_ok(bool(records), pristine) else 1


if __name__ == "__main__":
    sys.exit(main())
