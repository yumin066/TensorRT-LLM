#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard: implementation scripts must carry no forbidden source markers.

The project's Code Style rule requires domain-language naming in the driver scripts and
forbids process/meta wording. This guard scans delete_disfluency_e2e/*.py for those markers
(assembled from fragments so this file passes its own scan) and runs without the artifact
bundle so it also protects CI. Generated .md reports are never scanned.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent

# forbidden markers, assembled from fragments so this guard passes its own scan
_FORBIDDEN = [
    "A" + "C" + "-",
    "Mile" + "stone",
    "St" + "ep",
    "Ph" + "ase",
    "plan" + "-control",
    "per the " + "plan",
    "plann" + "ed",
    "work" + "flow",
]


def test_no_forbidden_source_markers():
    offenders = {}
    for py in sorted(HERE.glob("*.py")):
        hits = [t for t in _FORBIDDEN if t in py.read_text()]
        if hits:
            offenders[py.name] = hits
    assert not offenders, f"forbidden source markers in implementation sources: {offenders}"


if __name__ == "__main__":
    test_no_forbidden_source_markers()
    print("SOURCE_HYGIENE_OK")
