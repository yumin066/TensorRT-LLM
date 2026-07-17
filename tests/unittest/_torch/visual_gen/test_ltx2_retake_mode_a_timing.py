# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side unit tests for the pure helpers in ``ltx2_retake_mode_a_timing``.

Only the stdlib-only helpers (per-call record, Mode-A summary, warm comparison,
pristine check) are exercised; the every-rebuild GPU loop lives in ``main()``.
The tool lives under ``examples/`` so it is loaded by path via ``importlib``.
"""

import importlib.util
from pathlib import Path

import pytest

_MODE_A_PATH = (
    Path(__file__).resolve().parents[4] / "examples" / "visual_gen" / "ltx2_retake_mode_a_timing.py"
)


def _load_mode_a():
    spec = importlib.util.spec_from_file_location("ltx2_retake_mode_a_timing", _MODE_A_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mode_a = _load_mode_a()


def _fake_summarize(samples):
    s = sorted(samples)
    return {"p50": s[len(s) // 2], "min": s[0], "count": len(s)}


# --------------------------------------------------------------------------- #
# build_call_record
# --------------------------------------------------------------------------- #


def test_build_call_record_total_is_build_plus_run():
    rec = mode_a.build_call_record(2, 40.0, 2.5)
    assert rec["index"] == 2
    assert rec["model_build_load"] == 40.0
    assert rec["run_total"] == 2.5
    assert rec["total"] == 42.5


# --------------------------------------------------------------------------- #
# summarize_mode_a
# --------------------------------------------------------------------------- #


def test_summarize_mode_a_per_key():
    records = [
        mode_a.build_call_record(0, 40.0, 2.0),
        mode_a.build_call_record(1, 44.0, 2.4),
    ]
    summary = mode_a.summarize_mode_a(records, _fake_summarize)
    assert summary["model_build_load"]["min"] == 40.0
    assert summary["run_total"]["min"] == 2.0
    assert summary["total"]["min"] == 42.0
    assert summary["total"]["count"] == 2


# --------------------------------------------------------------------------- #
# speedup_vs_warm
# --------------------------------------------------------------------------- #


def test_speedup_vs_warm_ratio():
    cmp = mode_a.speedup_vs_warm(42.0, 1.82)
    assert cmp["mode_b_speedup_x"] == pytest.approx(42.0 / 1.82)
    assert cmp["mode_a_total_p50_seconds"] == 42.0


def test_speedup_vs_warm_none_when_missing():
    assert mode_a.speedup_vs_warm(None, 1.82) is None
    assert mode_a.speedup_vs_warm(42.0, None) is None
    assert mode_a.speedup_vs_warm(42.0, 0.0) is None


# --------------------------------------------------------------------------- #
# parse_call_result
# --------------------------------------------------------------------------- #


def test_parse_call_result_extracts_json_line():
    stdout = (
        "some build log line\n"
        'CALL_RESULT {"model_build_load": 41.0, "run_total": 2.1, "output_sha256": "ab"}\n'
        "trailing\n"
    )
    res = mode_a.parse_call_result(stdout)
    assert res["model_build_load"] == 41.0
    assert res["output_sha256"] == "ab"


def test_parse_call_result_none_when_absent_or_bad():
    assert mode_a.parse_call_result("no result here\n") is None
    assert mode_a.parse_call_result("CALL_RESULT {not json") is None


# --------------------------------------------------------------------------- #
# packages_pristine
# --------------------------------------------------------------------------- #


def test_packages_pristine_clean_and_unchanged():
    p = mode_a.packages_pristine("", "")
    assert p["unchanged"] is True
    assert p["clean"] is True


def test_packages_pristine_detects_change():
    p = mode_a.packages_pristine("", " M packages/ltx-pipelines/x.py\n")
    assert p["unchanged"] is False
    assert p["clean"] is False


def test_packages_pristine_unchanged_but_preexisting_dirty():
    # A pre-existing (not-mine) modification that is the same before and after is
    # "unchanged" (I did not touch it) even though the tree is not "clean".
    dirty = " M packages/other.py\n"
    p = mode_a.packages_pristine(dirty, dirty)
    assert p["unchanged"] is True
    assert p["clean"] is False


# --------------------------------------------------------------------------- #
# stdlib-only import (Round 30)
# --------------------------------------------------------------------------- #


def test_mode_a_pure_helpers_run_without_numpy():
    import subprocess
    import sys

    bootstrap = (
        "import sys, importlib.util\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        top = name.split('.')[0]\n"
        "        if top in ('numpy', 'torch', 'tensorrt_llm'):\n"
        "            raise ImportError(f'{top} blocked for test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        f"spec = importlib.util.spec_from_file_location('m', {str(_MODE_A_PATH)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "r = m.build_call_record(0, 40.0, 2.0)\n"
        "assert r['total'] == 42.0\n"
        "assert m.packages_pristine('', '')['unchanged'] is True\n"
        "print('MODE_A_PURE_OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", bootstrap], capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "MODE_A_PURE_OK" in proc.stdout
