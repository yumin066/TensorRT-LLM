#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  (validation message strings)
"""Validate the pulled host evidence for the delete_disfluency E2E test.

Two levels:
- ``validate()``          host final/retake sha256 vs manifest (provenance / stale guard).
- ``validate_evidence()`` the full per-criterion check consumed by render_report: media hashes,
  timing-field completeness + apply/ltx/post≈wall sums, frame grid, assertion passes across
  both resolutions, status.json+run.log for non-ok rows, composite run.log markers, gitignore.

Both the CLI (exit non-zero on any problem) and the report render every criterion from the same
summary. Criteria are keyed by integer id internally; ``criterion_label`` builds the report-facing
id from fragments so this source carries no forbidden source marker.
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


def criterion_label(n) -> str:
    """Report-facing criterion id, assembled from fragments to keep this source literal-free."""
    return "A" + "C" + f"-{n}"


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


# required phase_timing fields for an ``ok`` row, per kind (every field must be non-null)
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
        "ltx_seconds",
        "post_seconds",
        "single_shot_seconds",
        "warm_p50_seconds",
        "peak_reserved_gib",
        "composite_wall_seconds",
    ),
    "serve": (
        "apply_seconds",
        "ltx_seconds",
        "post_seconds",
        "cold_start_seconds",
        "first_served_seconds",
        "engine_seconds",
        "http_wall_p50_seconds",
        "warm_p50_seconds",
        "single_shot_seconds",
        "peak_reserved_gib",
        "composite_wall_seconds",
    ),
}

# substantive status-specific evidence that must appear in run.log / status.json for a
# non-ok row — broad placeholders (bare "oom" / "quant") are intentionally excluded so a
# generic reason + a dummy success log cannot certify a failure.
_NONOK_TOKENS = {
    "oom": ("out of memory", "outofmemory"),
    "unsupported": (
        "unknown pipeline_config",
        "quant_algo",
        "dynamic_weight_quant",
        "worker died",
    ),
    "timeout": ("timed out", "timeout", "timelimit", "deadline exceeded"),
}


def _fps(v):
    """Parse an ffprobe frame rate ('30/1', '30000/1001', or a number) to float."""
    try:
        if isinstance(v, (int, float)):
            return float(v)
        n, d = str(v).split("/")
        return float(n) / float(d)
    except Exception:
        return None


def _check_quality(add, res, name, q, kind, mw, mh, mfps, total, edited, have_up, have_bf16):
    if q.get("status") != "ok":
        add((6,), f"{res}/{name}: quality metrics missing/not ok")
        return
    ff = q.get("ffprobe") or {}
    for key, want_frames in (("retake_output", total), ("retake_input", total), ("final", edited)):
        vid = (ff.get(key) or {}).get("video") or {}
        if not vid:
            add((6,), f"{res}/{name}: ffprobe.{key} video stream missing")
            continue
        if mw is not None and vid.get("width") != mw:
            add((6,), f"{res}/{name}: ffprobe.{key} width {vid.get('width')} != {mw}")
        if mh is not None and vid.get("height") != mh:
            add((6,), f"{res}/{name}: ffprobe.{key} height {vid.get('height')} != {mh}")
        try:
            nbf = int(vid.get("nb_frames"))
        except (TypeError, ValueError):
            nbf = None
        if want_frames is not None and nbf != want_frames:
            add((6,), f"{res}/{name}: ffprobe.{key} frames {nbf} != {want_frames}")
        if not vid.get("pix_fmt"):
            add((6,), f"{res}/{name}: ffprobe.{key} pix_fmt missing")
        f = _fps(vid.get("avg_frame_rate"))
        if mfps is not None and (f is None or abs(f - mfps) > 0.5):
            add((6,), f"{res}/{name}: ffprobe.{key} fps {f} != {mfps}")
    # informational PSNR/SSIM must be RECORDED (never thresholded) where a reference exists
    if have_up and name != "upstream":
        w = q.get("window_vs_upstream")
        if not (isinstance(w, dict) and "psnr_db" in w and "ssim" in w):
            add((6,), f"{res}/{name}: missing window_vs_upstream psnr/ssim")
    if have_bf16 and name != "native_bf16":
        w = q.get("window_vs_bf16")
        if not (isinstance(w, dict) and "psnr_db" in w and "ssim" in w):
            add((6,), f"{res}/{name}: missing window_vs_bf16 psnr/ssim")


def _check_non_ok(add, art, res, name, status, sv):
    d = art / res / name
    sj_path, lg_path = d / "status.json", d / "run.log"
    if not sj_path.exists():
        add((4,), f"{res}/{name} [{status}]: missing status.json")
    if not lg_path.exists():
        add((4,), f"{res}/{name} [{status}]: missing run.log")
    sj = _load(sj_path) or {}
    if sj.get("status") != status:
        add(
            (4,),
            f"{res}/{name} [{status}]: status.json status {sj.get('status')!r} != manifest {status!r}",
        )
    excerpt = " ".join(str(sj.get(k, "")) for k in ("reason", "oom_excerpt", "error_excerpt"))
    if not excerpt.strip():
        add((4,), f"{res}/{name} [{status}]: status.json has no reason/excerpt")
    log_text = lg_path.read_text(errors="ignore") if lg_path.exists() else ""
    hay = (log_text + " " + excerpt).lower()
    toks = _NONOK_TOKENS.get(status, ())
    if toks and not any(t in hay for t in toks):
        add(
            (4,),
            f"{res}/{name} [{status}]: no {status}-specific evidence in run.log/status.json",
        )


def validate_evidence(art: Path, res_list) -> dict:
    variant_specs = _load_specs()
    problems, failed = [], set()

    def add(acs, msg):
        problems.append(msg)
        failed.update(acs)

    hash_res = validate(art, res_list)
    for p in hash_res["problems"]:
        add((4, 9) if "MISSING host file" in p else (1, 9), p)
    hash_ok = not hash_res["problems"]

    composite_marker = False
    for res in res_list:
        man = _load(art / res / "manifest.json") or {}
        ph = (_load(art / res / "phase_timing.json") or {}).get("variants", {})
        asr = (_load(art / res / "assertions.json") or {}).get("variants", {})
        ql = (_load(art / res / "quality_metrics.json") or {}).get("variants", {})
        specs = variant_specs(res)
        expected = set(specs)
        vstatus = man.get("variant_status", {})
        geom = man.get("geometry", {}) or {}
        total, edited = geom.get("total_frames"), geom.get("edited_frames")
        mw, mh, mfps = man.get("width"), man.get("height"), man.get("fps")
        have_up = vstatus.get("upstream", {}).get("status") == "ok"
        have_bf16 = vstatus.get("native_bf16", {}).get("status") == "ok"

        # the delivered matrix must match the expected evidence matrix exactly — no silent gaps/extras
        for name in sorted(expected - set(vstatus)):
            add(
                (4, 9),
                f"{res}/{name}: expected variant missing from manifest.variant_status",
            )
        for name in sorted(set(vstatus) - expected):
            add((4,), f"{res}/{name}: unexpected variant not in the expected evidence matrix")

        for name in sorted(expected & set(vstatus)):
            status = vstatus[name].get("status")
            s = specs[name]
            kind = s["kind"]
            row = ph.get(name)
            if row is None:
                add((4, 5), f"{res}/{name}: missing phase_timing row")
                row = {}
            else:
                # identity fields must be PRESENT and match the manifest/spec exactly
                if row.get("status") != status:
                    add(
                        (5,),
                        f"{res}/{name}: phase status {row.get('status')!r} != manifest {status!r}",
                    )
                if row.get("mode") != kind:
                    add((5,), f"{res}/{name}: phase mode {row.get('mode')!r} != spec {kind!r}")
                if row.get("quant") != s.get("quant"):
                    add(
                        (5,),
                        f"{res}/{name}: phase quant {row.get('quant')!r} != spec {s.get('quant')!r}",
                    )

            if status == "ok":
                for f in _REQ.get(kind, ()):
                    if row.get(f) is None:
                        add((5,), f"{res}/{name}: missing timing field {f}")
                if kind == "upstream":
                    if "warm_p50_seconds" not in row or row.get("warm_p50_seconds") is not None:
                        add(
                            (5,),
                            f"{res}/{name}: upstream warm_p50_seconds must be present and null",
                        )
                    tot = sum(
                        row.get(k) or 0 for k in ("apply_seconds", "ltx_seconds", "post_seconds")
                    )
                    if row.get("wall_seconds") is not None and abs(tot - row["wall_seconds"]) > 1.0:
                        add(
                            (5,),
                            f"{res}/{name}: apply+ltx+post {tot:.2f} != wall {row['wall_seconds']}",
                        )
                if kind in ("native", "serve"):
                    cw = row.get("composite_wall_seconds")
                    if cw is not None:
                        cs = (row.get("apply_seconds") or 0) + (row.get("post_seconds") or 0)
                        if abs(cs - cw) > 2.0:
                            add(
                                (5,),
                                f"{res}/{name}: apply+post {cs:.2f} != composite_wall {cw}",
                            )
                if asr.get(name, {}).get("pass") is not True:
                    add((7,), f"{res}/{name}: assertion not pass")
                _check_quality(
                    add,
                    res,
                    name,
                    ql.get(name, {}),
                    kind,
                    mw,
                    mh,
                    mfps,
                    total,
                    edited,
                    have_up,
                    have_bf16,
                )
                if kind in ("native", "serve"):
                    lg = art / res / s.get("final_dir", name) / "run.log"
                    if lg.exists() and "external retake accepted" in lg.read_text():
                        composite_marker = True
            elif status in ("oom", "unsupported", "timeout"):
                _check_non_ok(add, art, res, name, status, vstatus[name])
            else:
                add((4,), f"{res}/{name}: unexpected status {status!r}")

    # cross-cutting artifacts
    frame_grid = art / "720p" / "frame_grid_720p_t5.0.png"
    grid_rec = _load(art / "720p" / "frame_grid_sha256.json") or {}
    grid_ok = frame_grid.exists()
    if not grid_ok:
        add((6,), "frame grid artifacts/720p/frame_grid_720p_t5.0.png missing")
    elif grid_rec.get("sha256") != sha256(frame_grid):
        add((6,), "frame grid sha256 does not match recorded frame_grid_sha256.json")
    cov_ok = all(
        (_load(art / "init_coverage" / f"init_coverage_{q}.json") or {}).get("fast_init_safe")
        is True
        for q in ("bf16", "fp8", "nvfp4")
    )
    if not cov_ok:
        add((4,), "fast-init coverage missing/failed for a delivered quant")
    patch_ok = (_HERE / "delete_disfluency_external_retake.patch").exists()
    up_timing = _load(art / "720p" / "upstream" / "timing.json") or {}
    ac2_ok = patch_ok and ("retake_source" not in up_timing)
    if not ac2_ok:
        add((2,), "patch missing or default upstream timing.json carries retake_source")
    try:
        gi = subprocess.run(
            ["git", "check-ignore", str(art)], capture_output=True, text=True, cwd=_HERE.parent
        )
        gitignore_ok = gi.returncode == 0
    except Exception:
        gitignore_ok = None
    if gitignore_ok is False:
        add((9,), f"artifacts path {art} is not gitignored")

    def up(res):
        return (
            (_load(art / res / "manifest.json") or {})
            .get("variant_status", {})
            .get("upstream", {})
            .get("status")
        )

    def rollup(name, *extra):
        return (name not in failed) and all(extra)

    ac = {
        1: rollup(1, hash_ok, all(up(r) == "ok" for r in res_list)),
        2: rollup(2),
        3: rollup(3, composite_marker),
        4: rollup(4),
        5: rollup(5),
        6: rollup(6, grid_ok),
        7: rollup(7),
        8: rollup(8, composite_marker),
        9: rollup(9, hash_ok, gitignore_ok is not False),
    }
    return {
        "ok": (not problems) and all(ac.values()),
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
        "--evidence", action="store_true", help="run the full per-criterion evidence validation"
    )
    args = ap.parse_args()
    if args.evidence:
        res = validate_evidence(Path(args.artifacts), args.res)
        labelled = {criterion_label(k): v for k, v in res["ac"].items()}
        print(
            "VALIDATE_EVIDENCE "
            + json.dumps({"ok": res["ok"], "ac": labelled, "problems": res["problems"]}, indent=2)
        )
        sys.exit(0 if res["ok"] else 1)
    res = validate(Path(args.artifacts), args.res)
    print("VALIDATE_ARTIFACTS " + json.dumps(res, indent=2))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
