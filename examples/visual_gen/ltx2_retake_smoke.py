#! /usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native LTX-2 retake acceleration smoke matrix.

A cheap early-feasibility sweep: build the native ``LTXModel`` retake pipeline
under each acceleration config and run one retake per config to record WHICH
configs execute end-to-end on the target checkpoint. It is deliberately NOT the
authoritative per-axis capability gate with clean per-config latency/memory
attribution and the fixed acceleration-stacking order -- that is a separate,
later concern. Nothing here is a correctness gate: PSNR / SSIM columns are
informational (each config vs the bf16/VANILLA baseline output) and never fail
the sweep.

Configs swept: bf16/VANILLA (baseline), bf16/FA4, bf16/CUDAGraph, NVFP4/VANILLA.
Each records a status (``ok`` / ``unsupported`` / ``not-applicable`` / ``error``)
with a skip / failure reason, plus env metadata. A failing config is recorded
and the sweep CONTINUES -- one broken axis must not abort the matrix.

The heavy pipeline build / run is reused from the sibling ``ltx2_retake_oracle``
module (loaded by path), so there is a single source of truth for how the native
retake pipeline is constructed. The pure helpers at the top of this module
(``build_config_matrix``, ``precheck_capability``, ``classify_status``,
``build_record``, ``summarize_matrix``) import on a plain CPU host with no numpy
/ torch / tensorrt_llm.
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

_VALID_STATUSES = ("ok", "unsupported", "not-applicable", "error")


# ----------------------------------------------------------------------------
# Pure helpers (stdlib only; host-testable without numpy / torch / tensorrt_llm).
# ----------------------------------------------------------------------------


def build_config_matrix() -> list:
    """The four native-retake acceleration smoke configs.

    Each entry is config-driven: ``attention_backend`` selects the native
    attention module, ``cuda_graph`` toggles the Modality-aware CUDA-graph
    runner, and ``quant_algo`` (e.g. ``"NVFP4"``) enables the native runtime
    dynamic-quant path. ``bf16/VANILLA`` is the informational-quality baseline.
    """
    return [
        {
            "label": "bf16/VANILLA",
            "dtype": "bf16",
            "attention_backend": "VANILLA",
            "cuda_graph": False,
            "quant_algo": None,
            "baseline": True,
        },
        {
            "label": "bf16/FA4",
            "dtype": "bf16",
            "attention_backend": "FA4",
            "cuda_graph": False,
            "quant_algo": None,
            "baseline": False,
        },
        {
            "label": "bf16/CUDAGraph",
            "dtype": "bf16",
            "attention_backend": "VANILLA",
            "cuda_graph": True,
            "quant_algo": None,
            "baseline": False,
        },
        {
            "label": "NVFP4/VANILLA",
            "dtype": "bf16",
            "attention_backend": "VANILLA",
            "cuda_graph": False,
            "quant_algo": "NVFP4",
            "baseline": False,
        },
    ]


def precheck_capability(config: dict, caps: dict) -> tuple:
    """Return ``(supported, reason)`` from statically-known capability flags.

    ``caps`` is gathered at runtime (imports) and injected so this stays pure.
    A missing flag defaults to "attempt" (``True``) -- the actual run is the
    ground truth, and a genuine failure is recorded as ``error`` rather than
    pre-judged. Only capabilities we can cheaply pre-determine gate here.
    """
    backend = config.get("attention_backend")
    if backend == "FA4" and not caps.get("fa4_available", True):
        return False, "FA4 unavailable: flash_attn cute backend not importable on this build"
    if backend == "CUTEDSL" and not caps.get("cutedsl_available", True):
        return False, "CuTeDSL unavailable: requires precompiled cubin + head_dim=128 + fp16/bf16"
    if config.get("quant_algo") == "NVFP4" and not caps.get("nvfp4_available", True):
        return False, "NVFP4 dynamic quant unavailable: native DynamicLinearWeightLoader missing"
    return True, None


def classify_status(
    supported: bool, precheck_reason: Optional[str], ran_ok: bool, error_reason: Optional[str]
) -> tuple:
    """Map a config's precheck + run outcome to a status + reason.

    ``unsupported`` = a capability precheck ruled it out (never built);
    ``ok`` = it built and ran end-to-end; ``error`` = it was attempted but the
    build/run raised (reason carries the exception summary).
    """
    if not supported:
        return "unsupported", precheck_reason
    if ran_ok:
        return "ok", None
    return "error", error_reason


def build_record(
    config: dict,
    status: str,
    reason: Optional[str],
    quality: Optional[dict],
    duration_seconds: Optional[float],
) -> dict:
    """Assemble one per-config result row for the smoke matrix."""
    if status not in _VALID_STATUSES:
        raise ValueError(f"invalid status {status!r}; expected one of {_VALID_STATUSES}")
    return {
        "label": config["label"],
        "dtype": config.get("dtype", "bf16"),
        "attention_backend": config.get("attention_backend"),
        "cuda_graph": bool(config.get("cuda_graph")),
        "quant_algo": config.get("quant_algo"),
        "baseline": bool(config.get("baseline")),
        "status": status,
        "reason": reason,
        # Informational only (each config's output vs the bf16/VANILLA baseline);
        # None for the baseline itself and for configs that did not run.
        "quality_informational": quality,
        "duration_seconds": duration_seconds,
    }


def baseline_ran_ok(records: list) -> bool:
    """Whether the baseline config ran ``ok``.

    The baseline (bf16/VANILLA) is the informational-quality anchor every other
    config is compared against, so the sweep is only meaningful when it ran. A
    later config succeeding while the baseline failed is NOT a meaningful sweep.
    """
    for r in records:
        if r.get("baseline"):
            return r.get("status") == "ok"
    return False


def summarize_matrix(records: list) -> dict:
    """Count records by status; ``baseline_ok`` drives the exit code."""
    counts: dict = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {
        "total": len(records),
        "by_status": counts,
        "ok": counts.get("ok", 0),
        "baseline_ok": baseline_ran_ok(records),
    }


def _json_safe(obj):
    """Recursively replace non-finite floats with string sentinels.

    An informational PSNR of ``inf`` (a config's output is bit-identical to the
    baseline) is a real, useful signal, but ``json.dumps`` writes it as the
    literal ``Infinity``, which strict (non-Python) JSON parsers reject. Convert
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


def _gather_caps() -> dict:
    """Probe which acceleration capabilities are importable in this environment."""
    caps = {"fa4_available": False, "cutedsl_available": False, "nvfp4_available": False}
    try:
        import flash_attn.cute  # noqa: F401

        caps["fa4_available"] = True
    except Exception as exc:  # noqa: BLE001 - capability probe records the reason
        caps["fa4_reason"] = f"{type(exc).__name__}: {exc}"
    try:
        from tensorrt_llm._torch.visual_gen.quantization.loader import (  # noqa: F401
            DynamicLinearWeightLoader,
        )

        caps["nvfp4_available"] = True
    except Exception as exc:  # noqa: BLE001
        caps["nvfp4_reason"] = f"{type(exc).__name__}: {exc}"
    try:
        import cutlass  # noqa: F401

        caps["cutedsl_available"] = True
    except Exception as exc:  # noqa: BLE001
        caps["cutedsl_reason"] = f"{type(exc).__name__}: {exc}"
    return caps


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Native LTX-2 retake acceleration smoke matrix.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--gemma", required=True)
    p.add_argument("--lora", default=None)
    p.add_argument("--source", required=True, help="video-only source clip")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--prompt", default="a person talking to the camera")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--code-commit", default=None, help="authoritative code commit for provenance")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    oracle = _load_oracle()

    import gc
    import time

    import torch

    # ``_retake_pixel_window`` is a helper in the retake pipeline module (the
    # oracle imports it locally in its own main, so it is not an oracle module
    # attribute); import it here for the window-geometry conversion.
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_retake import _retake_pixel_window

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    code_commit = args.code_commit or os.environ.get("LTX2_ORACLE_CODE_COMMIT")

    source_frames = oracle.read_source_frames(args.source)
    num_frames = int(source_frames.shape[0])

    caps = _gather_caps()
    env = oracle._env_metadata()
    matrix = build_config_matrix()

    records = []
    baseline_thwc = None
    pixel_window = None

    for config in matrix:
        label = config["label"]
        supported, precheck_reason = precheck_capability(config, caps)
        if not supported:
            print(f"[smoke] {label}: unsupported ({precheck_reason})", flush=True)
            records.append(build_record(config, "unsupported", precheck_reason, None, None))
            continue

        ran_ok = False
        error_reason = None
        quality = None
        duration = None
        pipe = None
        video = None
        print(f"[smoke] {label}: building + running...", flush=True)
        t0 = time.time()
        try:
            pipe = oracle.build_pipeline(
                args.checkpoint,
                args.gemma,
                args.lora,
                device,
                config["attention_backend"],
                cuda_graph_enable=config["cuda_graph"],
                quant_algo=config["quant_algo"],
            )
            request = oracle._make_request(
                args.prompt,
                args.negative_prompt,
                args.source,
                args.start,
                args.end,
                args.seed,
                args.steps,
            )
            video, _audio, fps, _sr = oracle._run_pipeline(pipe, request)
            thwc = oracle._video_to_thwc_uint8(video)
            if pixel_window is None:
                pixel_window = _retake_pixel_window(args.start, args.end, fps, num_frames)
            if config.get("baseline"):
                baseline_thwc = thwc
            elif baseline_thwc is not None:
                quality = oracle.compute_similarity(
                    thwc, baseline_thwc, pixel_window, f"{label}_vs_baseline"
                )
            ran_ok = True
            duration = time.time() - t0
            print(f"[smoke] {label}: ok ({duration:.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001 - per-config isolation: record and continue
            duration = time.time() - t0
            error_reason = f"{type(exc).__name__}: {exc}"
            print(f"[smoke] {label}: error ({error_reason})", file=sys.stderr, flush=True)
            traceback.print_exc()
        finally:
            del pipe, video
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        status, reason = classify_status(supported, precheck_reason, ran_ok, error_reason)
        records.append(build_record(config, status, reason, quality, duration))

    summary = summarize_matrix(records)
    native_repo = oracle._git_info(str(Path(__file__).resolve().parents[2]))
    result = {
        "summary": summary,
        "records": records,
        "capabilities": caps,
        "env": env,
        "code_commit": code_commit,
        "native_repo": native_repo,
        "source": str(Path(args.source).resolve()),
        "source_sha256": oracle.sha256_file(args.source),
        "window": [args.start, args.end],
        "seed": args.seed,
        "steps": args.steps,
        "num_frames": num_frames,
        "note": (
            "Early acceleration feasibility smoke (which native-retake configs run); "
            "NOT authoritative per-axis capability gating. PSNR/SSIM informational only."
        ),
    }
    (output_dir / "smoke_matrix.json").write_text(
        json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False)
    )
    print("SMOKE_DONE", json.dumps(summary))
    # The baseline (bf16/VANILLA) is the quality anchor; the sweep is only
    # meaningful when it ran, regardless of how many other configs happened to
    # pass.
    return 0 if summary["baseline_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
