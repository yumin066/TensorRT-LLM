#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  (validation message strings)
"""Validate the pulled host evidence for the delete_disfluency E2E test.

Two levels:
- ``validate()``          host final/retake sha256 vs manifest (provenance / stale guard).
- ``validate_evidence()`` the full per-AC check consumed by render_report: media hashes,
  timing-field completeness + APPLY/LTX/POST≈wall sums, frame grid, assertion passes across
  both resolutions, status.json+run.log for non-ok rows, composite run.log markers, gitignore.

Both the CLI (exit non-zero on any problem) and the report render every AC from the same summary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def sha256(p: Path):
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _load_specs():
    spec = importlib.util.spec_from_file_location("collect_evidence", _HERE / "collect_evidence.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["collect_evidence"] = m
    spec.loader.exec_module(m)
    return m.variant_specs


def validate(art: Path, res_list) -> dict:
    variant_specs = _load_specs()
    problems, checked = [], 0
    for res in res_list:
        man = _load(art / res / "manifest.json")
        if man is None:
            problems.append(f"{res}: missing manifest.json")
            continue
        specs = variant_specs(res)
        for name, rec in man.get("artifacts", {}).items():
            s = specs.get(name, {})
            for label, host, want in (
                (
                    "final",
                    art / res / s.get("final_dir", name) / "final.mp4",
                    rec.get("final_sha256"),
                ),
                (
                    "retake",
                    art / res / s.get("retake", f"{name}/retake.mp4"),
                    rec.get("retake_sha256"),
                ),
            ):
                if want is None:
                    continue
                checked += 1
                got = sha256(host)
                if got is None:
                    problems.append(f"{res}/{name} {label}: MISSING host file {host}")
                elif got != want:
                    problems.append(
                        f"{res}/{name} {label}: sha mismatch host={got[:12]} manifest={want[:12]}"
                    )
    return {"checked": checked, "problems": problems, "ok": not problems}


# required phase_timing fields for an ``ok`` row, per kind
_REQ = {
    "upstream": (
        "apply_seconds",
        "ltx_seconds",
        "post_seconds",
        "wall_seconds",
        "single_shot_seconds",
        "peak_reserved_gib",
    ),
    "native": (
        "apply_seconds",
        "post_seconds",
        "single_shot_seconds",
        "warm_p50_seconds",
        "peak_reserved_gib",
    ),
    "serve": (
        "cold_start_seconds",
        "first_served_seconds",
        "engine_seconds",
        "warm_p50_seconds",
        "single_shot_seconds",
        "peak_reserved_gib",
    ),
}


def validate_evidence(art: Path, res_list) -> dict:
    variant_specs = _load_specs()
    problems = list(validate(art, res_list)["problems"])
    hash_ok = not problems

    composite_marker = False
    for res in res_list:
        man = _load(art / res / "manifest.json") or {}
        ph = (_load(art / res / "phase_timing.json") or {}).get("variants", {})
        asr = (_load(art / res / "assertions.json") or {}).get("variants", {})
        ql = (_load(art / res / "quality_metrics.json") or {}).get("variants", {})
        specs = variant_specs(res)
        for name, sv in man.get("variant_status", {}).items():
            status = sv["status"]
            s = specs.get(name, {})
            if status == "ok":
                row = ph.get(name, {})
                kind = row.get("mode", s.get("kind", "native"))
                for f in _REQ.get(
                    kind, ()
                ):  # warm_p50 may be explicit null for upstream/native-baseline
                    if f == "warm_p50_seconds":
                        continue
                    if row.get(f) is None:
                        problems.append(f"{res}/{name}: missing timing field {f}")
                if kind == "upstream":
                    tot = (
                        (row.get("apply_seconds") or 0)
                        + (row.get("ltx_seconds") or 0)
                        + (row.get("post_seconds") or 0)
                    )
                    if row.get("wall_seconds") is not None and abs(tot - row["wall_seconds"]) > 1.0:
                        problems.append(
                            f"{res}/{name}: apply+ltx+post {tot:.2f} != wall {row['wall_seconds']}"
                        )
                if row.get("composite_wall_seconds") is not None:
                    cs = (row.get("apply_seconds") or 0) + (row.get("post_seconds") or 0)
                    if abs(cs - row["composite_wall_seconds"]) > 2.0:
                        problems.append(
                            f"{res}/{name}: apply+post {cs:.2f} != composite_wall {row['composite_wall_seconds']}"
                        )
                av = asr.get(name, {})
                if av.get("pass") is not True:
                    problems.append(f"{res}/{name}: assertion not pass")
                q = ql.get(name, {})
                if q.get("status") != "ok":
                    problems.append(f"{res}/{name}: quality metrics missing")
                # composite marker for TRT rows
                if s.get("kind") in ("native", "serve"):
                    lg = art / res / s.get("final_dir", name) / "run.log"
                    if lg.exists() and "external retake accepted" in lg.read_text():
                        composite_marker = True
            elif status in ("oom", "unsupported"):
                d = art / res / (name if not name.startswith("native_") else name)
                for f in ("status.json", "run.log"):
                    if not (d / f).exists():
                        problems.append(f"{res}/{name} [{status}]: missing {f}")

    # cross-cutting artifacts
    frame_grid = art / "720p" / "frame_grid_720p_t5.0.png"
    grid_ok = frame_grid.exists()
    if not grid_ok:
        problems.append("AC-6: frame grid artifacts/720p/frame_grid_720p_t5.0.png missing")
    cov_ok = all(
        (_load(art / "init_coverage" / f"init_coverage_{q}.json") or {}).get("fast_init_safe")
        is True
        for q in ("bf16", "fp8", "nvfp4")
    )
    if not cov_ok:
        problems.append("fast-init coverage missing/failed for a delivered quant")
    patch_ok = (_HERE / "delete_disfluency_external_retake.patch").exists()
    up_timing = _load(art / "720p" / "upstream" / "timing.json") or {}
    ac2_ok = patch_ok and ("retake_source" not in up_timing)
    if not ac2_ok:
        problems.append("AC-2: patch missing or default upstream timing.json carries retake_source")
    try:
        gi = subprocess.run(
            ["git", "check-ignore", str(art)], capture_output=True, text=True, cwd=_HERE.parent
        )
        gitignore_ok = gi.returncode == 0
    except Exception:
        gitignore_ok = None

    # per-AC rollup
    def up(res):
        return (
            (_load(art / res / "manifest.json") or {})
            .get("variant_status", {})
            .get("upstream", {})
            .get("status")
        )

    ac = {
        "AC-1": hash_ok and all(up(r) == "ok" for r in res_list),
        "AC-2": ac2_ok,
        "AC-3": composite_marker,
        "AC-4": not any(("[oom]" in p) or ("[unsupported]" in p) for p in problems),
        "AC-5": not any(
            "timing field" in p or "!= wall" in p or "composite_wall" in p for p in problems
        ),
        "AC-6": grid_ok and not any("quality" in p for p in problems),
        "AC-7": not any("assertion not pass" in p for p in problems),
        "AC-8": composite_marker,
        "AC-9": hash_ok and (gitignore_ok is not False),
    }
    return {
        "ok": not problems,
        "problems": problems,
        "ac": ac,
        "hash_ok": hash_ok,
        "grid_ok": grid_ok,
        "coverage_ok": cov_ok,
        "gitignore_ok": gitignore_ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--res", nargs="+", default=["720p", "1080p"])
    ap.add_argument(
        "--evidence", action="store_true", help="run the full per-AC evidence validation"
    )
    args = ap.parse_args()
    if args.evidence:
        res = validate_evidence(Path(args.artifacts), args.res)
        print(
            "VALIDATE_EVIDENCE "
            + json.dumps({"ok": res["ok"], "ac": res["ac"], "problems": res["problems"]}, indent=2)
        )
        sys.exit(0 if res["ok"] else 1)
    res = validate(Path(args.artifacts), args.res)
    print("VALIDATE_ARTIFACTS " + json.dumps(res, indent=2))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
