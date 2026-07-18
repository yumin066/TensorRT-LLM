# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side unit tests for the pure helpers in ``ltx2_retake_attn_profile``.

Only the stdlib-only helpers (percentile/summary, CuTeDSL capability, nsys
kernel-CSV parse, attention share, backend delta, status classify, profile
summary, JSON sanitize) are exercised; the GPU build/run lives in
``_run_single_backend``. The runner lives under ``examples/`` so it is loaded by
path via ``importlib``.
"""

import importlib.util
from pathlib import Path

import pytest

_PROF_PATH = (
    Path(__file__).resolve().parents[4] / "examples" / "visual_gen" / "ltx2_retake_attn_profile.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("ltx2_retake_attn_profile", _PROF_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prof = _load()


# --------------------------------------------------------------------------- #
# cutedsl_capability
# --------------------------------------------------------------------------- #


def test_cutedsl_capable_head_dim_128_bf16():
    ok, reason = prof.cutedsl_capability(128, "bfloat16")
    assert ok is True and reason is None


def test_cutedsl_rejects_non_128_head_dim():
    ok, reason = prof.cutedsl_capability(64, "bfloat16")
    assert ok is False and "64" in reason


def test_cutedsl_rejects_unsupported_dtype():
    ok, reason = prof.cutedsl_capability(128, "float32")
    assert ok is False and "dtype" in reason


def test_cutedsl_unknown_head_dim():
    ok, reason = prof.cutedsl_capability(None, "bfloat16")
    assert ok is False and reason == "unknown_head_dim"


# --------------------------------------------------------------------------- #
# parse_nsys_kern_csv + attention_share
# --------------------------------------------------------------------------- #


_NSYS_CSV = (
    "Time (%),Total Time (ns),Instances,Avg (ns),Name\n"
    '60.0,600000,10,60000,"void fmha_fwd_kernel<...>"\n'
    '25.0,250000,8,31250,"cutlass_mha_gemm"\n'
    '15.0,150000,5,30000,"elementwise_add_kernel"\n'
)


def test_parse_nsys_kern_csv_extracts_name_and_total_ns():
    rows = prof.parse_nsys_kern_csv(_NSYS_CSV)
    assert len(rows) == 3
    assert rows[0]["name"].startswith("void fmha_fwd_kernel")
    assert rows[0]["total_ns"] == 600000.0


def test_parse_nsys_kern_csv_handles_empty():
    assert prof.parse_nsys_kern_csv("") == []
    assert prof.parse_nsys_kern_csv("no header here\n1,2,3\n") == []


def test_attention_share_sums_matched_kernels():
    rows = prof.parse_nsys_kern_csv(_NSYS_CSV)
    share = prof.attention_share(rows)
    # fmha (600k) + cutlass_mha (250k) = 850k of 1000k total.
    assert share["share"] == pytest.approx(0.85)
    assert share["attention_ns"] == 850000.0
    assert share["total_ns"] == 1000000.0
    assert share["num_matched"] == 2
    assert share["matched_kernels"][0]["name"].startswith("void fmha_fwd_kernel")


def test_attention_share_none_when_no_kernels():
    share = prof.attention_share([])
    assert share["share"] is None
    assert share["total_ns"] == 0


def test_attention_share_custom_patterns():
    rows = [{"name": "my_special_attn", "total_ns": 100.0}, {"name": "gemm", "total_ns": 100.0}]
    share = prof.attention_share(rows, patterns=("special_attn",))
    assert share["share"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# backend_delta
# --------------------------------------------------------------------------- #


def test_backend_delta_speedup_and_delta():
    d = prof.backend_delta(1.20, 1.00)
    assert d["speedup_vs_baseline"] == pytest.approx(1.2)
    assert d["delta_seconds"] == pytest.approx(-0.20)


def test_backend_delta_none_on_missing():
    d = prof.backend_delta(None, 1.0)
    assert d["speedup_vs_baseline"] is None and d["delta_seconds"] is None
    d = prof.backend_delta(1.0, 0.0)
    assert d["speedup_vs_baseline"] is None


# --------------------------------------------------------------------------- #
# classify_backend_status
# --------------------------------------------------------------------------- #


def test_classify_ok():
    assert prof.classify_backend_status(True, None, True, None) == ("ok", None)


def test_classify_unsupported_keeps_precheck_reason():
    assert prof.classify_backend_status(False, "head_dim=64!=128", False, None) == (
        "unsupported",
        "head_dim=64!=128",
    )


def test_classify_error_keeps_run_reason():
    assert prof.classify_backend_status(True, None, False, "RuntimeError: boom") == (
        "error",
        "RuntimeError: boom",
    )


# --------------------------------------------------------------------------- #
# reclassify_cutedsl_capability_error (head_dim cubin failure -> unsupported)
# --------------------------------------------------------------------------- #


def test_cutedsl_head_dim_error_reclassified_unsupported():
    # A runtime head_dim cubin rejection (audio attention head_dim=64) is a
    # capability limit, so it folds into supported=False -> classifies as
    # ``unsupported``, not ``error``.
    reason = "ValueError: CUTEDSL cubins require head_dim=128, got head_dim=64."
    supported, precheck = prof.reclassify_cutedsl_capability_error(True, None, reason)
    assert supported is False and precheck == reason
    status, _ = prof.classify_backend_status(supported, precheck, False, reason)
    assert status == "unsupported"


def test_reclassify_leaves_other_errors_as_error():
    # A non-capability runtime error stays supported=True -> classifies as error.
    reason = "RuntimeError: 'NoneType' object has no attribute '_trait'"
    supported, precheck = prof.reclassify_cutedsl_capability_error(True, None, reason)
    assert supported is True and precheck is None
    status, _ = prof.classify_backend_status(supported, precheck, False, reason)
    assert status == "error"


def test_reclassify_noop_when_no_error():
    assert prof.reclassify_cutedsl_capability_error(True, None, None) == (True, None)


# --------------------------------------------------------------------------- #
# summarize_profile (deltas vs VANILLA baseline)
# --------------------------------------------------------------------------- #


def test_summarize_profile_computes_deltas_vs_vanilla():
    records = [
        {"backend": "VANILLA", "status": "ok", "denoise": {"p50": 1.20}},
        {"backend": "FA4", "status": "ok", "denoise": {"p50": 1.00}},
        {"backend": "CUTEDSL", "status": "unsupported", "reason": "x", "denoise": None},
    ]
    summary = prof.summarize_profile(records)
    assert summary["baseline_ran_ok"] is True
    assert summary["backends_ok"] == ["VANILLA", "FA4"]
    fa4 = next(r for r in records if r["backend"] == "FA4")
    assert fa4["delta"]["speedup_vs_baseline"] == pytest.approx(1.2)
    cute = next(r for r in records if r["backend"] == "CUTEDSL")
    assert cute["delta"]["speedup_vs_baseline"] is None


def test_summarize_profile_baseline_failed():
    records = [
        {"backend": "VANILLA", "status": "error", "reason": "boom", "denoise": None},
        {"backend": "FA4", "status": "ok", "denoise": {"p50": 1.00}},
    ]
    summary = prof.summarize_profile(records)
    assert summary["baseline_ran_ok"] is False
    # No baseline p50 → FA4 delta is None, not a crash.
    fa4 = next(r for r in records if r["backend"] == "FA4")
    assert fa4["delta"]["speedup_vs_baseline"] is None


# --------------------------------------------------------------------------- #
# parse_backend_result + _json_safe
# --------------------------------------------------------------------------- #


def test_parse_backend_result_roundtrip():
    line = 'BACKEND_RESULT {"backend": "FA4", "status": "ok"}'
    assert prof.parse_backend_result("noise\n" + line + "\nmore")["backend"] == "FA4"


def test_parse_backend_result_absent():
    assert prof.parse_backend_result("nothing here") is None


def test_json_safe_sanitizes_non_finite():
    out = prof._json_safe({"a": float("inf"), "b": [float("nan"), 1.0], "c": float("-inf")})
    assert out["a"] == "inf" and out["b"][0] == "nan" and out["c"] == "-inf"


# --------------------------------------------------------------------------- #
# no plan-process labels in the runner
# --------------------------------------------------------------------------- #


def test_attn_profile_has_no_plan_process_labels():
    text = _PROF_PATH.read_text()
    assert "AC-" not in text
    for label in ("Milestone", "Phase "):
        assert label not in text


# --------------------------------------------------------------------------- #
# stdlib-only import (helpers load with no numpy / torch / tensorrt_llm)
# --------------------------------------------------------------------------- #


def test_attn_profile_pure_helpers_run_without_numpy():
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
        f"spec = importlib.util.spec_from_file_location('p', {str(_PROF_PATH)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "assert m.cutedsl_capability(128, 'bfloat16')[0] is True\n"
        'rows = m.parse_nsys_kern_csv(\'Time,Total Time (ns),Name\\n1,100,"fmha_x"\\n1,100,"gemm"\\n\')\n'
        "assert m.attention_share(rows)['share'] == 0.5\n"
        "print('ATTN_PURE_OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", bootstrap], capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "ATTN_PURE_OK" in proc.stdout
