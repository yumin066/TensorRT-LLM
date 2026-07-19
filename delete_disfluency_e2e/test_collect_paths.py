#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: collector reference-clip paths resolve correctly under a relative out dir.

The row-building step already stores out-dir-prefixed paths; a consumer that re-joins the
out dir would double-prefix into ``out/out/...`` and silently drop the bf16/upstream quality
references. This test pins the correct single-prefix resolution and needs no artifact bundle.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _collect():
    spec = importlib.util.spec_from_file_location("collect_evidence", HERE / "collect_evidence.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["collect_evidence"] = m
    spec.loader.exec_module(m)
    return m


def test_reference_retake_relative_out_no_double_prefix():
    ce = _collect()
    out = Path("artifacts/example")
    rows = ce.build_variant_rows(out, ce.variant_specs("720p"))

    bf16 = ce.reference_retake(rows, "native_bf16")
    up = ce.reference_retake(rows, "upstream")
    assert bf16 == Path("artifacts/example/native_bf16_final/retake_output.mp4")
    assert up == Path("artifacts/example/upstream/retake_output.mp4")
    # the double-prefix regression must not reappear
    assert bf16 != Path("artifacts/example/artifacts/example/native_bf16_final/retake_output.mp4")
    assert ce.reference_retake(rows, "does_not_exist") is None


if __name__ == "__main__":
    test_reference_retake_relative_out_no_double_prefix()
    print("COLLECT_PATHS_OK")
