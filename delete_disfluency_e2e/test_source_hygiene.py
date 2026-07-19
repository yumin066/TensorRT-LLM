#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard: implementation scripts must carry no plan-control terminology.

The plan's Code Style rule forbids workflow/plan-control terms in implementation code
and comments. This guard scans the driver scripts (not docs, generated reports, or the
gitignored artifact bundle) so those terms cannot be reintroduced. Runs with no bundle
present, so it also protects CI.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent


def test_no_plan_control_terms_in_impl_sources():
    # banned literals assembled from fragments so this guard file itself stays clean
    banned = ["A" + "C" + "-", "Mile" + "stone", "St" + "ep", "Ph" + "ase"]
    offenders = {}
    for py in sorted(HERE.glob("*.py")):
        hits = [t for t in banned if t in py.read_text()]
        if hits:
            offenders[py.name] = hits
    assert not offenders, f"plan-control terms in implementation sources: {offenders}"


if __name__ == "__main__":
    test_no_plan_control_terms_in_impl_sources()
    print("SOURCE_HYGIENE_OK")
