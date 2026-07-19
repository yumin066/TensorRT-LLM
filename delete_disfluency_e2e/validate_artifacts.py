#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate pulled host artifacts against the manifest (provenance / stale guard).

For every ``ok`` variant row, recompute the host final.mp4 and canonical retake
sha256 and compare to the manifest. Exits non-zero if any is missing or mismatched,
so the manifest is a trustworthy stale-output guard for the reviewed host tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def sha256(p: Path):
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _load_specs():
    p = Path(__file__).with_name("collect_evidence.py")
    spec = importlib.util.spec_from_file_location("collect_evidence", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["collect_evidence"] = m
    spec.loader.exec_module(m)
    return m.variant_specs


def validate(art: Path, res_list) -> dict:
    variant_specs = _load_specs()
    problems, checked = [], 0
    for res in res_list:
        man_p = art / res / "manifest.json"
        if not man_p.exists():
            problems.append(f"{res}: missing manifest.json")
            continue
        man = json.loads(man_p.read_text())
        specs = variant_specs(res)
        for name, rec in man.get("artifacts", {}).items():
            s = specs.get(name, {})
            final_host = art / res / s.get("final_dir", name) / "final.mp4"
            retake_host = art / res / s.get("retake", f"{name}/retake.mp4")
            for label, host, want in (
                ("final", final_host, rec.get("final_sha256")),
                ("retake", retake_host, rec.get("retake_sha256")),
            ):
                if want is None:
                    continue
                checked += 1
                got = sha256(host)
                if got is None:
                    problems.append(f"{res}/{name} {label}: MISSING host file {host}")
                elif got != want:
                    problems.append(
                        f"{res}/{name} {label}: sha mismatch host={got[:12]} manifest={want[:12]} ({host})"
                    )
    return {"checked": checked, "problems": problems, "ok": not problems}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--res", nargs="+", default=["720p", "1080p"])
    args = ap.parse_args()
    result = validate(Path(args.artifacts), args.res)
    print("VALIDATE_ARTIFACTS " + json.dumps(result, indent=2))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
