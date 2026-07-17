#! /usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified LTX-2 retake measurement report schema.

The native retake effort emits several independent measurement artifacts, each
with its own nested shape: the resident-warm native timing timeline, the
upstream-orchestration persistent baseline, the every-rebuild upstream reference,
the trtllm-serve HTTP surface, the per-axis acceleration capability gate (single
axis + acceleration stack), and the torch_compile compile-cost split. This tool
reads whichever of those exist and emits ONE additive report that normalizes
every measured config/mode into a single row shape WITHOUT rewriting any source
raw data.

Each row carries the same keys regardless of source: where it came from, the
measured mode, the acceleration config, a canonical shape manifest (plus a
deterministic hash of it so identical geometries collide and any changed field
diverges), the cold build / first-call timeline, the warm first-served + steady
percentiles, per-run peak memory (explicit nulls where a source has none),
informational quality, provenance (the rsync-synced native tree pin is preserved
faithfully, never invented clean), and pass-through environment metadata. The
one torch_compile single-axis row additionally receives the compile-cost block.

Everything here is stdlib only (json, hashlib, pathlib, argparse, os, sys), so
the whole aggregator imports and runs in a plain ``python3`` with no numpy /
torch / tensorrt_llm. Missing source artifacts are recorded and skipped rather
than aborting the aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# (source key, round directory basename, artifact filename)
_SOURCE_SPECS = (
    ("native_timing", "round-29-timing", "timing.json"),
    ("mode_a", "round-30-mode-a", "mode_a_timing.json"),
    ("serve", "round-32-serve-timing", "serve_timing.json"),
    ("upstream_baseline", "round-36-upstream-baseline", "timing.json"),
    ("gating", "round-39-accel-gating", "accel_gating.json"),
    ("compile_cost", "round-41-compile-cost", "compile_cost.json"),
)

_SHAPE_KEYS = (
    "resolution",
    "num_frames",
    "latent_shape",
    "window",
    "pixel_window",
    "edit_type",
    "seed",
    "steps",
    "source_sha256",
    "prompt_len",
    "negative_prompt_len",
)

_DEFAULT_ARTIFACTS_DIR = (
    Path(__file__).resolve().parents[2] / ".humanize" / "rlcr" / "2026-07-15_20-47-29" / "artifacts"
)


# ----------------------------------------------------------------------------
# Pure helpers (stdlib only; host-testable without numpy / torch / tensorrt_llm).
# ----------------------------------------------------------------------------


def _json_safe(obj):
    """Recursively replace non-finite floats with string sentinels for strict JSON."""
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


def _parse_source_geometry(source: Optional[str]) -> tuple:
    """Derive ``(resolution, num_frames)`` from a source clip name, else ``(None, None)``.

    Source clips are named like ``vonly_512x320_89f.mp4``; the geometry is not
    otherwise recorded per-record, so it is recovered from the file name.
    """
    if not source:
        return None, None
    name = Path(str(source)).name
    resolution = None
    frames = None
    res_match = re.search(r"(\d+)x(\d+)", name)
    if res_match:
        resolution = f"{res_match.group(1)}x{res_match.group(2)}"
    frame_match = re.search(r"(\d+)f\b", name)
    if frame_match:
        frames = int(frame_match.group(1))
    return resolution, frames


def canonical_shape_manifest(**fields) -> dict:
    """Assemble a canonical shape manifest from the known geometry fields.

    Every manifest carries the same keys; unknown fields are explicit ``null``
    rather than dropped, so two runs of the same geometry hash identically and a
    changed field always diverges. ``edit_type`` defaults to ``"retake"``.
    """
    manifest = {key: fields.get(key) for key in _SHAPE_KEYS}
    if manifest["edit_type"] is None:
        manifest["edit_type"] = "retake"
    return manifest


def shape_hash(manifest: dict) -> str:
    """Deterministic sha256 hex of a shape manifest (stable across equal manifests)."""
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_memory(peak: Optional[dict]) -> dict:
    """Carry per-run peak memory with explicit nulls; keys are never silently dropped.

    None of the retake sources record per-stage memory attribution, so
    ``stage_attribution`` is always ``null`` and ``has_stage_attribution`` is
    always ``False``. A source with no memory at all still yields explicit
    ``null`` for both peak fields.
    """
    peak = peak or {}
    return {
        "peak_allocated": peak.get("allocated"),
        "peak_reserved": peak.get("reserved"),
        "stage_attribution": None,
        "has_stage_attribution": False,
    }


def normalize_provenance(artifact: dict) -> dict:
    """Normalize provenance, preserving the rsync-synced native tree pin faithfully.

    The native tree was rsync-synced (not committed) for these runs, so
    ``native_repo.dirty`` is truthfully ``True`` and ``code_commit`` is the
    authoritative pin. That dirty state is preserved rather than rewritten to a
    clean commit, and a ``provenance_note`` spells out why.
    """
    native_repo = artifact.get("native_repo") or {}
    provenance = {
        "code_commit": artifact.get("code_commit"),
        "native_repo": {
            "commit": native_repo.get("commit"),
            "branch": native_repo.get("branch"),
            "dirty": native_repo.get("dirty"),
        },
        "eval_repo": artifact.get("eval_repo"),
        "patch_hash": None,
    }
    if native_repo.get("dirty"):
        provenance["provenance_note"] = (
            "native tree was rsync-synced (not committed); the working tree is "
            "dirty and code_commit is the authoritative provenance pin"
        )
    return provenance


def normalize_env(env: Optional[dict]) -> dict:
    """Pass through the recorded env and add explicit keys for fields we never captured."""
    env = env or {}
    return {
        "torch_version": env.get("torch_version"),
        "cuda_version": env.get("cuda_version"),
        "device_name": env.get("device_name"),
        "platform": env.get("platform"),
        "python": env.get("python"),
        "gpu_total_memory": env.get("gpu_total_memory"),
        "flash_attn_version": env.get("flash_attn_version"),
        "cutedsl_version": env.get("cutedsl_version"),
    }


def _source_ref(artifact_file: str, round_dir: str, path: str) -> dict:
    """The ``source`` sub-object identifying which artifact a row / block came from."""
    return {"artifact": artifact_file, "round": round_dir, "path": path}


def _config_row(cfg: Optional[dict], label: Optional[str] = None) -> dict:
    """Normalize a source config block into the unified row config shape."""
    cfg = cfg or {}
    return {
        "dtype": cfg.get("dtype", "bf16"),
        "attention_backend": cfg.get("attention_backend"),
        "quant_algo": cfg.get("quant_algo"),
        "cuda_graph": bool(cfg.get("cuda_graph", False)),
        "torch_compile": bool(cfg.get("torch_compile", False)),
        "offload": cfg.get("retake_offload_mode"),
        "retake_use_upstream_stage": bool(cfg.get("retake_use_upstream_stage", False)),
        "label": label,
    }


def _shape_for_artifact(artifact: dict) -> tuple:
    """Build the shape manifest + hash from an artifact's top-level geometry fields."""
    resolution, parsed_frames = _parse_source_geometry(artifact.get("source"))
    num_frames = artifact.get("num_frames")
    if num_frames is None:
        num_frames = parsed_frames
    manifest = canonical_shape_manifest(
        resolution=resolution,
        num_frames=num_frames,
        window=artifact.get("window"),
        edit_type="retake",
        seed=artifact.get("seed"),
        steps=artifact.get("steps"),
        source_sha256=artifact.get("source_sha256"),
    )
    return manifest, shape_hash(manifest)


def _base_row(artifact: dict, mode: str, source_ref: dict) -> dict:
    """Skeleton row with the source/provenance/env/shape fields every adapter shares."""
    manifest, digest = _shape_for_artifact(artifact)
    return {
        "source": source_ref,
        "mode": mode,
        "shape_manifest": manifest,
        "shape_hash": digest,
        "provenance": normalize_provenance(artifact),
        "env": normalize_env(artifact.get("env")),
        "status": "ok",
        "failure": None,
        "compile_cost": None,
    }


# ----------------------------------------------------------------------------
# Per-source adapters (each returns a list of unified rows).
# ----------------------------------------------------------------------------


def adapt_native_timing(
    artifact: dict, source_ref: dict, mode: str = "native_resident_warm"
) -> list:
    """Resident-warm native retake timeline -> one row (also the upstream-stage baseline)."""
    row = _base_row(artifact, mode, source_ref)
    timeline = artifact.get("timeline") or {}
    first_served = timeline.get("first_served") or {}
    steady = (timeline.get("steady_warm") or {}).get("wall")
    records = artifact.get("records") or []
    row["config"] = _config_row(artifact.get("config"))
    row["cold"] = {
        "model_build_load": timeline.get("cold_model_build_load_seconds"),
        "first_call_or_compile": None,
        "note": timeline.get("compile_graph_capture_note"),
    }
    row["warm"] = {
        "first_served": first_served.get("wall")
        if isinstance(first_served, dict)
        else first_served,
        "steady": steady,
        "raw_samples": [r.get("wall") for r in records] or None,
    }
    row["memory"] = normalize_memory(None)
    row["quality_informational"] = artifact.get("quality_informational")
    return [row]


def adapt_upstream_baseline(artifact: dict, source_ref: dict) -> list:
    """Upstream-orchestration persistent baseline -> one row (same timeline shape)."""
    return adapt_native_timing(artifact, source_ref, mode="upstream_stage_persistent")


def adapt_mode_a(artifact: dict, source_ref: dict) -> list:
    """Every-rebuild upstream reference -> one row (no warm steady; every call rebuilds)."""
    row = _base_row(artifact, "mode_a_every_rebuild", source_ref)
    summary = artifact.get("summary") or {}
    build = summary.get("model_build_load") or {}
    run_total = summary.get("run_total") or {}
    total = summary.get("total") or {}
    row["config"] = _config_row(artifact.get("config"))
    row["cold"] = {
        "model_build_load": build.get("p50"),
        "first_call_or_compile": run_total.get("p50"),
        "every_rebuild_total_p50": total.get("p50"),
        "note": (
            "every call rebuilds the upstream pipeline from scratch, so there is "
            "no resident warm steady state; model_build_load recurs per call"
        ),
    }
    row["warm"] = {"first_served": None, "steady": None, "raw_samples": None}
    row["memory"] = normalize_memory(None)
    row["quality_informational"] = artifact.get("quality_informational")
    return [row]


def adapt_serve(artifact: dict, source_ref: dict) -> list:
    """trtllm-serve HTTP resident-warm surface -> one row."""
    row = _base_row(artifact, "serve_http", source_ref)
    first_served = artifact.get("first_served")
    steady_warm = artifact.get("steady_warm") or {}
    records = artifact.get("records") or []
    if "wall" in steady_warm:
        steady = steady_warm.get("wall")
    else:
        steady = steady_warm or None
    row["config"] = _config_row(artifact.get("config"))
    run_ok = artifact.get("run_ok", True)
    row["status"] = "ok" if run_ok else "error"
    row["failure"] = None if run_ok else artifact.get("run_reason")
    row["cold"] = {
        "model_build_load": None,
        "first_call_or_compile": None,
        "note": "resident trtllm-serve HTTP worker; cold build not measured in this artifact",
    }
    row["warm"] = {
        "first_served": (
            first_served.get("wall") if isinstance(first_served, dict) else first_served
        ),
        "steady": steady,
        "raw_samples": [r.get("wall") for r in records] or None,
    }
    serve_metrics = {}
    for key in ("generation", "denoise"):
        if key in steady_warm:
            serve_metrics[key] = steady_warm.get(key)
    if serve_metrics:
        row["serve_metrics"] = serve_metrics
    row["memory"] = normalize_memory(None)
    row["quality_informational"] = artifact.get("quality_informational")
    return [row]


def adapt_gating(artifact: dict, source_ref: dict) -> list:
    """Per-axis acceleration gate -> one row per record (single-axis / baseline / stack)."""
    rows = []
    for record in artifact.get("records") or []:
        kind = record.get("kind")
        mode = "accel_gating_stack" if kind == "stack" else "accel_gating_single_axis"
        row = _base_row(artifact, mode, source_ref)
        row["config"] = _config_row(record, label=record.get("label"))
        status = record.get("status", "ok")
        row["status"] = status
        row["failure"] = record.get("reason") if status != "ok" else None
        row["cold"] = {
            "model_build_load": record.get("cold_model_build_load"),
            "first_call_or_compile": None,
            "note": None,
        }
        row["warm"] = {
            "first_served": record.get("first_served"),
            "steady": record.get("steady_warm"),
            "raw_samples": record.get("raw_samples"),
        }
        row["per_stage"] = record.get("per_stage")
        row["memory"] = normalize_memory(record.get("peak_memory"))
        row["quality_informational"] = record.get("quality_informational")
        rows.append(row)
    return rows


def build_compile_cost_block(artifact: dict, source_ref: dict) -> dict:
    """Assemble the compile-cost block (attached to the torch_compile row, not a row)."""
    derived = artifact.get("derived") or {}
    empty_first = None
    for record in artifact.get("records") or []:
        if record.get("mode") == "empty":
            empty_first = record.get("first_call")
            break
    return {
        "empty_first": empty_first,
        "same_process_steady_p50": derived.get("steady_p50"),
        "warm_disk_first": derived.get("warm_disk_first_seconds"),
        "compile_cost_seconds": derived.get("compile_cost_seconds"),
        "cache_saved_seconds": derived.get("cache_saved_seconds"),
        "source": source_ref,
    }


def attach_compile_cost(rows: list, block: Optional[dict]) -> Optional[dict]:
    """Attach the compile-cost block to the bf16/torch_compile single-axis row.

    Returns the block if it was NOT attached to any row (so it is still surfaced
    under the report's top-level ``compile_cost`` key and never lost).
    """
    if block is None:
        return None
    for row in rows:
        cfg = row.get("config") or {}
        if (
            row.get("mode") == "accel_gating_single_axis"
            and cfg.get("torch_compile")
            and cfg.get("quant_algo") is None
            and cfg.get("cuda_graph") is False
        ):
            row["compile_cost"] = block
            return None
    return block


def _sha256_file(path: Path) -> Optional[str]:
    """sha256 hex of a file's bytes, or ``None`` if it cannot be read."""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


# ----------------------------------------------------------------------------
# Aggregation entry point.
# ----------------------------------------------------------------------------


_ADAPTERS = {
    "native_timing": adapt_native_timing,
    "mode_a": adapt_mode_a,
    "serve": adapt_serve,
    "upstream_baseline": adapt_upstream_baseline,
    "gating": adapt_gating,
}


def resolve_source_paths(args) -> dict:
    """Map each source key to its path, honoring explicit overrides then the artifacts dir."""
    artifacts_dir = Path(args.artifacts_dir)
    overrides = {
        "native_timing": args.native_timing,
        "mode_a": args.mode_a,
        "serve": args.serve,
        "upstream_baseline": args.upstream_baseline,
        "gating": args.gating,
        "compile_cost": args.compile_cost,
    }
    resolved = {}
    for key, round_dir, filename in _SOURCE_SPECS:
        override = overrides.get(key)
        resolved[key] = Path(override) if override else artifacts_dir / round_dir / filename
    return resolved


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Unified LTX-2 retake measurement report schema.")
    p.add_argument("--artifacts-dir", default=str(_DEFAULT_ARTIFACTS_DIR))
    p.add_argument("--native-timing", default=None)
    p.add_argument("--mode-a", default=None)
    p.add_argument("--serve", default=None)
    p.add_argument("--upstream-baseline", default=None)
    p.add_argument("--gating", default=None)
    p.add_argument("--compile-cost", default=None)
    p.add_argument("--output", default=None, help="report.json output path")
    p.add_argument("--code-commit", default=None)
    return p.parse_args(argv)


def _round_dir_for(key: str) -> str:
    for spec_key, round_dir, _filename in _SOURCE_SPECS:
        if spec_key == key:
            return round_dir
    return key


def build_report(paths: dict, code_commit: Optional[str] = None) -> dict:
    """Load whichever sources exist, run adapters, and assemble the unified report."""
    rows = []
    generated_from = []
    missing_sources = []
    compile_cost_block = None

    for key, round_dir, filename in _SOURCE_SPECS:
        path = paths[key]
        if not Path(path).is_file():
            missing_sources.append({"source": key, "round": round_dir, "path": str(path)})
            continue
        with open(path, "r") as handle:
            artifact = json.load(handle)
        source_ref = _source_ref(filename, round_dir, str(path))
        generated_from.append(
            {
                "artifact": filename,
                "round": round_dir,
                "path": str(path),
                "sha256_of_file": _sha256_file(Path(path)),
            }
        )
        if key == "compile_cost":
            compile_cost_block = build_compile_cost_block(artifact, source_ref)
        else:
            rows.extend(_ADAPTERS[key](artifact, source_ref))

    top_level_block = attach_compile_cost(rows, compile_cost_block)
    modes = sorted({row["mode"] for row in rows})
    result = {
        "schema_version": "1",
        "rows": rows,
        "compile_cost": top_level_block,
        "generated_from": generated_from,
        "missing_sources": missing_sources,
        "row_count": len(rows),
        "modes": modes,
        "code_commit": code_commit,
        "note": (
            "Additive unified view of the LTX-2 retake measurement artifacts: every "
            "measured config/mode is normalized into one row shape without rewriting "
            "the source raw data. Missing sources are recorded, not fatal."
        ),
    }
    return result


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = resolve_source_paths(args)
    result = build_report(paths, code_commit=args.code_commit)

    has_anchor = any(
        row["mode"] in ("native_resident_warm", "accel_gating_single_axis", "accel_gating_stack")
        for row in result["rows"]
    )

    try:
        text = json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError) as exc:
        print(f"REPORT_SCHEMA_FAILED serialization error: {exc}", file=sys.stderr)
        return 1

    output = args.output or os.environ.get("LTX2_REPORT_SCHEMA_OUTPUT")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text)

    print(
        "REPORT_SCHEMA_DONE",
        json.dumps(
            {
                "row_count": result["row_count"],
                "modes": result["modes"],
                "missing_sources": [m["source"] for m in result["missing_sources"]],
            }
        ),
    )
    return 0 if has_anchor else 1


if __name__ == "__main__":
    sys.exit(main())
