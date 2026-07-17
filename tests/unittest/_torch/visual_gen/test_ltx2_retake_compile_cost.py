# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side unit tests for the pure helpers in ``ltx2_retake_compile_cost``.

Only the stdlib-only helpers (cache-env threading, cache plan, result parsing,
derived cost split, summary, JSON sanitization) are exercised; the heavy
first-call + steady measured loop lives inside ``run_single_mode`` and reuses the
oracle / timing modules. The runner lives under ``examples/`` so it is loaded by
path via ``importlib``.
"""

import importlib.util
from pathlib import Path

import pytest

_COMPILE_COST_PATH = (
    Path(__file__).resolve().parents[4] / "examples" / "visual_gen" / "ltx2_retake_compile_cost.py"
)


def _load_compile_cost():
    spec = importlib.util.spec_from_file_location("ltx2_retake_compile_cost", _COMPILE_COST_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compile_cost = _load_compile_cost()


# --------------------------------------------------------------------------- #
# set_compile_cache_env
# --------------------------------------------------------------------------- #


def test_set_compile_cache_env_points_and_creates_dirs(tmp_path):
    import os

    dirs = compile_cost.set_compile_cache_env(tmp_path / "cache")
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == dirs["inductor"]
    assert os.environ["TRITON_CACHE_DIR"] == dirs["triton"]
    assert Path(dirs["inductor"]).is_dir()
    assert Path(dirs["triton"]).is_dir()
    # The two caches live under the requested root, not the same directory.
    assert Path(dirs["inductor"]).parent == (tmp_path / "cache")
    assert dirs["inductor"] != dirs["triton"]


# --------------------------------------------------------------------------- #
# build_cache_plan
# --------------------------------------------------------------------------- #


def test_build_cache_plan_has_empty_and_warm_modes():
    plan = compile_cost.build_cache_plan()
    modes = {e["mode"] for e in plan}
    assert "empty" in modes
    assert "warm" in modes
    assert all(e["mode"] in ("empty", "warm") for e in plan)
    assert all({"mode", "label", "description"} <= set(e) for e in plan)
    # The same-process steady measurement belongs to the empty-cache process.
    labels = {e["label"] for e in plan}
    assert "same_process_steady" in labels
    steady = next(e for e in plan if e["label"] == "same_process_steady")
    assert steady["mode"] == "empty"


# --------------------------------------------------------------------------- #
# parse_cache_mode_result
# --------------------------------------------------------------------------- #


def test_parse_cache_mode_result_last_line_wins():
    stdout = (
        "building log line\n"
        'CACHE_MODE_RESULT {"mode": "empty", "first_call": 1.0}\n'
        "noise between results\n"
        'CACHE_MODE_RESULT {"mode": "warm", "first_call": 2.0}\n'
        "trailing\n"
    )
    res = compile_cost.parse_cache_mode_result(stdout)
    assert res["mode"] == "warm"
    assert res["first_call"] == 2.0


def test_parse_cache_mode_result_none_when_absent_or_bad():
    assert compile_cost.parse_cache_mode_result("no result line here\n") is None
    assert compile_cost.parse_cache_mode_result("CACHE_MODE_RESULT {not valid json") is None


# --------------------------------------------------------------------------- #
# derived_costs
# --------------------------------------------------------------------------- #


def test_derived_costs_hand_example():
    empty = {"first_call": 10.0, "steady": {"p50": 2.0}}
    warm = {"first_call": 3.0}
    d = compile_cost.derived_costs(empty, warm)
    # compile tax = empty first_call - empty steady p50 = 10 - 2 = 8.
    assert d["compile_cost_seconds"] == pytest.approx(8.0)
    # a warm on-disk cache saves = empty first_call - warm first_call = 10 - 3 = 7.
    assert d["cache_saved_seconds"] == pytest.approx(7.0)
    assert d["warm_disk_first_seconds"] == pytest.approx(3.0)
    assert d["steady_p50"] == pytest.approx(2.0)


def test_derived_costs_guards_missing_fields():
    # Both results missing -> every derived field is None.
    d = compile_cost.derived_costs(None, None)
    assert d["compile_cost_seconds"] is None
    assert d["cache_saved_seconds"] is None
    assert d["warm_disk_first_seconds"] is None
    assert d["steady_p50"] is None

    # Missing steady p50 -> no compile cost, but cache_saved still derivable.
    d2 = compile_cost.derived_costs({"first_call": 10.0}, {"first_call": 3.0})
    assert d2["compile_cost_seconds"] is None
    assert d2["cache_saved_seconds"] == pytest.approx(7.0)

    # Missing warm first_call -> compile cost derivable, cache_saved None.
    d3 = compile_cost.derived_costs({"first_call": 10.0, "steady": {"p50": 2.0}}, {})
    assert d3["compile_cost_seconds"] == pytest.approx(8.0)
    assert d3["cache_saved_seconds"] is None


# --------------------------------------------------------------------------- #
# summarize_compile_cost
# --------------------------------------------------------------------------- #


def test_summarize_compile_cost_ok_when_all_modes_ok():
    derived = {"compile_cost_seconds": 8.0}
    records = [{"mode": "empty", "ok": True}, {"mode": "warm", "ok": True}]
    s = compile_cost.summarize_compile_cost(records, derived)
    assert s["modes"] == ["empty", "warm"]
    assert s["ok"] is True
    assert s["fail"] is False
    assert s["derived"] is derived


def test_summarize_compile_cost_fails_on_bad_or_missing_record():
    derived = {"compile_cost_seconds": None}
    # A failed mode flips ok.
    s = compile_cost.summarize_compile_cost(
        [{"mode": "empty", "ok": True}, {"mode": "warm", "ok": False}], derived
    )
    assert s["ok"] is False
    assert s["fail"] is True
    # A None record (subprocess produced no result) flips ok.
    s2 = compile_cost.summarize_compile_cost([None, {"mode": "warm", "ok": True}], derived)
    assert s2["ok"] is False
    assert s2["modes"] == ["warm"]


# --------------------------------------------------------------------------- #
# _json_safe
# --------------------------------------------------------------------------- #


def test_json_safe_sanitizes_non_finite_floats():
    import json
    import math

    raw = {
        "compile_cost_seconds": math.inf,
        "neg": -math.inf,
        "nan": math.nan,
        "ok": 8.0,
        "nested": [{"w": math.inf}],
    }
    safe = compile_cost._json_safe(raw)
    assert safe["compile_cost_seconds"] == "inf"
    assert safe["neg"] == "-inf"
    assert safe["nan"] == "nan"
    assert safe["ok"] == 8.0
    assert safe["nested"][0]["w"] == "inf"
    text = json.dumps(safe, allow_nan=False)  # strict JSON must not raise
    assert json.loads(text)["ok"] == 8.0


# --------------------------------------------------------------------------- #
# stdlib-only: the pure helpers import + run without numpy/torch
# --------------------------------------------------------------------------- #


def test_compile_cost_pure_helpers_run_without_numpy():
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
        f"spec = importlib.util.spec_from_file_location('ltx2_retake_compile_cost', {str(_COMPILE_COST_PATH)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "import os, tempfile\n"
        "dirs = m.set_compile_cache_env(tempfile.mkdtemp())\n"
        "assert os.environ['TORCHINDUCTOR_CACHE_DIR'] == dirs['inductor']\n"
        "assert os.path.isdir(dirs['triton'])\n"
        "plan = m.build_cache_plan()\n"
        "assert {e['mode'] for e in plan} >= {'empty', 'warm'}\n"
        "d = m.derived_costs({'first_call': 10.0, 'steady': {'p50': 2.0}}, {'first_call': 3.0})\n"
        "assert d['compile_cost_seconds'] == 8.0\n"
        "assert d['cache_saved_seconds'] == 7.0\n"
        "s = m.summarize_compile_cost([{'mode': 'empty', 'ok': True}], d)\n"
        "assert s['ok'] is True\n"
        "assert m._json_safe({'a': float('inf')})['a'] == 'inf'\n"
        "print('COMPILE_COST_PURE_OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", bootstrap], capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "COMPILE_COST_PURE_OK" in proc.stdout
    assert "blocked for test" not in proc.stderr


# --------------------------------------------------------------------------- #
# no plan-process labels leak into the checked-in runner
# --------------------------------------------------------------------------- #


def test_compile_cost_runner_has_no_plan_process_labels():
    # Bans plan-process terminology in checked-in implementation text. The banned
    # tokens are assembled from fragments so this guard file does not itself
    # contain them literally (its own check would otherwise flag them).
    text = _COMPILE_COST_PATH.read_text()
    banned = [
        "A" + "C-",
        "Stage" + " 1",
        "Stage" + " 2",
        "Stage" + "-2",
        "Mile" + "stone",
        "Ph" + "ase ",
    ]
    for label in banned:
        assert label not in text, (
            f"plan-process label {label!r} leaked into the compile-cost runner"
        )
