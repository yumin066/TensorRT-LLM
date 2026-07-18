# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side unit tests for the pure helpers in ``ltx2_retake_quant_mem``.

Only the stdlib-only helpers (percentile/summary, GiB, memory/latency delta,
status classify, mode summary, JSON sanitize, MODE_RESULT parse) are exercised;
the GPU build/measure lives in ``_run_single_mode``. The runner lives under
``examples/`` so it is loaded by path via ``importlib``.
"""

import importlib.util
from pathlib import Path

import pytest

_QM_PATH = (
    Path(__file__).resolve().parents[4] / "examples" / "visual_gen" / "ltx2_retake_quant_mem.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("ltx2_retake_quant_mem", _QM_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qm = _load()
_GIB = 1024**3


# --------------------------------------------------------------------------- #
# gib + memory_delta
# --------------------------------------------------------------------------- #


def test_gib_converts_and_rounds():
    assert qm.gib(_GIB) == 1.0
    assert qm.gib(None) is None
    assert qm.gib(int(1.5 * _GIB)) == 1.5


def test_memory_delta_savings():
    d = qm.memory_delta(40 * _GIB, 30 * _GIB)
    assert d["ratio_vs_bf16"] == pytest.approx(0.75)
    assert d["saved_gib"] == pytest.approx(10.0)


def test_memory_delta_none_on_missing():
    assert qm.memory_delta(None, 1)["ratio_vs_bf16"] is None
    assert qm.memory_delta(1, 0)["saved_gib"] is None


# --------------------------------------------------------------------------- #
# latency_delta
# --------------------------------------------------------------------------- #


def test_latency_delta_speedup():
    d = qm.latency_delta(1.20, 1.00)
    assert d["speedup_vs_bf16"] == pytest.approx(1.2)
    assert d["delta_seconds"] == pytest.approx(-0.20)


def test_latency_delta_none_on_missing():
    assert qm.latency_delta(None, 1.0)["speedup_vs_bf16"] is None


# --------------------------------------------------------------------------- #
# classify_mode_status
# --------------------------------------------------------------------------- #


def test_classify_ok_and_error():
    assert qm.classify_mode_status(True, None) == ("ok", None)
    assert qm.classify_mode_status(False, "RuntimeError: boom") == ("error", "RuntimeError: boom")


# --------------------------------------------------------------------------- #
# summarize_modes (deltas vs bf16)
# --------------------------------------------------------------------------- #


def test_summarize_modes_deltas_vs_bf16():
    records = [
        {
            "mode": "bf16",
            "status": "ok",
            "denoise": {"p50": 1.20},
            "inference_peak": {"allocated_bytes": 40 * _GIB},
        },
        {
            "mode": "fp8",
            "status": "ok",
            "denoise": {"p50": 1.00},
            "inference_peak": {"allocated_bytes": 30 * _GIB},
        },
        {"mode": "nvfp4", "status": "error", "reason": "boom", "denoise": None},
    ]
    summary = qm.summarize_modes(records)
    assert summary["baseline_ran_ok"] is True
    assert summary["modes_ok"] == ["bf16", "fp8"]
    fp8 = next(r for r in records if r["mode"] == "fp8")
    assert fp8["latency_delta"]["speedup_vs_bf16"] == pytest.approx(1.2)
    assert fp8["memory_delta"]["ratio_vs_bf16"] == pytest.approx(0.75)
    nv = next(r for r in records if r["mode"] == "nvfp4")
    assert nv["latency_delta"]["speedup_vs_bf16"] is None
    assert nv["memory_delta"]["ratio_vs_bf16"] is None


def test_summarize_modes_baseline_failed():
    records = [
        {"mode": "bf16", "status": "error", "reason": "oom", "denoise": None},
        {
            "mode": "fp8",
            "status": "ok",
            "denoise": {"p50": 1.0},
            "inference_peak": {"allocated_bytes": _GIB},
        },
    ]
    summary = qm.summarize_modes(records)
    assert summary["baseline_ran_ok"] is False
    fp8 = next(r for r in records if r["mode"] == "fp8")
    assert fp8["latency_delta"]["speedup_vs_bf16"] is None
    assert fp8["memory_delta"]["ratio_vs_bf16"] is None


# --------------------------------------------------------------------------- #
# parse_mode_result + _json_safe
# --------------------------------------------------------------------------- #


def test_parse_mode_result_roundtrip():
    line = 'MODE_RESULT {"mode": "nvfp4", "status": "ok"}'
    assert qm.parse_mode_result("x\n" + line + "\ny")["mode"] == "nvfp4"


def test_parse_mode_result_absent():
    assert qm.parse_mode_result("nothing") is None


def test_json_safe_sanitizes_non_finite():
    out = qm._json_safe({"a": float("inf"), "b": [float("nan")], "c": float("-inf")})
    assert out["a"] == "inf" and out["b"][0] == "nan" and out["c"] == "-inf"


# --------------------------------------------------------------------------- #
# no plan-process labels in the runner
# --------------------------------------------------------------------------- #


def test_quant_mem_has_no_plan_process_labels():
    text = _QM_PATH.read_text()
    assert "AC-" not in text
    for label in ("Milestone", "Phase "):
        assert label not in text


# --------------------------------------------------------------------------- #
# stdlib-only import (helpers load with no numpy / torch / tensorrt_llm)
# --------------------------------------------------------------------------- #


def test_quant_mem_pure_helpers_run_without_numpy():
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
        f"spec = importlib.util.spec_from_file_location('q', {str(_QM_PATH)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "assert m.gib(1024**3) == 1.0\n"
        "assert m.memory_delta(40*1024**3, 30*1024**3)['ratio_vs_bf16'] == 0.75\n"
        "assert m.latency_delta(1.2, 1.0)['speedup_vs_bf16'] == 1.2\n"
        "print('QUANT_MEM_PURE_OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", bootstrap], capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "QUANT_MEM_PURE_OK" in proc.stdout
