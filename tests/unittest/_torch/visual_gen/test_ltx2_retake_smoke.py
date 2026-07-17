# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side unit tests for the pure helpers in ``ltx2_retake_smoke``.

Only the stdlib-only helpers (config matrix, capability precheck, status
classification, record assembly, summary) are exercised; the heavy pipeline
build/run lives inside ``main()`` and reuses the oracle module. The runner lives
under ``examples/`` so it is loaded by path via ``importlib``.
"""

import importlib.util
from pathlib import Path

import pytest

_SMOKE_PATH = (
    Path(__file__).resolve().parents[4] / "examples" / "visual_gen" / "ltx2_retake_smoke.py"
)


def _load_smoke():
    spec = importlib.util.spec_from_file_location("ltx2_retake_smoke", _SMOKE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load_smoke()


# --------------------------------------------------------------------------- #
# build_config_matrix
# --------------------------------------------------------------------------- #


def test_matrix_has_four_expected_configs():
    matrix = smoke.build_config_matrix()
    labels = [c["label"] for c in matrix]
    assert labels == ["bf16/VANILLA", "bf16/FA4", "bf16/CUDAGraph", "NVFP4/VANILLA"]


def test_matrix_exactly_one_baseline():
    matrix = smoke.build_config_matrix()
    baselines = [c for c in matrix if c.get("baseline")]
    assert len(baselines) == 1
    assert baselines[0]["label"] == "bf16/VANILLA"


def test_matrix_axis_wiring():
    by_label = {c["label"]: c for c in smoke.build_config_matrix()}
    assert by_label["bf16/FA4"]["attention_backend"] == "FA4"
    assert by_label["bf16/CUDAGraph"]["cuda_graph"] is True
    assert by_label["NVFP4/VANILLA"]["quant_algo"] == "NVFP4"
    # The baseline drives no acceleration axis.
    base = by_label["bf16/VANILLA"]
    assert base["attention_backend"] == "VANILLA"
    assert base["cuda_graph"] is False
    assert base["quant_algo"] is None


# --------------------------------------------------------------------------- #
# precheck_capability
# --------------------------------------------------------------------------- #


def test_precheck_fa4_unsupported_when_capability_absent():
    cfg = {"attention_backend": "FA4"}
    ok, reason = smoke.precheck_capability(cfg, {"fa4_available": False})
    assert ok is False and "FA4" in reason


def test_precheck_nvfp4_unsupported_when_capability_absent():
    cfg = {"attention_backend": "VANILLA", "quant_algo": "NVFP4"}
    ok, reason = smoke.precheck_capability(cfg, {"nvfp4_available": False})
    assert ok is False and "NVFP4" in reason


def test_precheck_supported_when_capabilities_present():
    cfg = {"attention_backend": "FA4", "quant_algo": "NVFP4"}
    ok, reason = smoke.precheck_capability(cfg, {"fa4_available": True, "nvfp4_available": True})
    assert ok is True and reason is None


def test_precheck_defaults_to_attempt_when_flag_missing():
    # A missing capability flag defaults to "attempt" (True), so the run is the
    # ground truth rather than pre-judging.
    cfg = {"attention_backend": "VANILLA", "quant_algo": None}
    ok, reason = smoke.precheck_capability(cfg, {})
    assert ok is True and reason is None


# --------------------------------------------------------------------------- #
# classify_status
# --------------------------------------------------------------------------- #


def test_classify_unsupported_carries_precheck_reason():
    status, reason = smoke.classify_status(False, "no FA4", False, None)
    assert status == "unsupported" and reason == "no FA4"


def test_classify_ok_when_ran():
    status, reason = smoke.classify_status(True, None, True, None)
    assert status == "ok" and reason is None


def test_classify_error_when_run_failed():
    status, reason = smoke.classify_status(True, None, False, "RuntimeError: boom")
    assert status == "error" and "boom" in reason


# --------------------------------------------------------------------------- #
# build_record / summarize_matrix
# --------------------------------------------------------------------------- #


def test_build_record_fields():
    cfg = smoke.build_config_matrix()[1]  # bf16/FA4
    rec = smoke.build_record(cfg, "ok", None, {"psnr": 30.0}, 12.3)
    assert rec["label"] == "bf16/FA4"
    assert rec["attention_backend"] == "FA4"
    assert rec["status"] == "ok"
    assert rec["quality_informational"] == {"psnr": 30.0}
    assert rec["duration_seconds"] == 12.3


def test_build_record_rejects_invalid_status():
    cfg = smoke.build_config_matrix()[0]
    with pytest.raises(ValueError):
        smoke.build_record(cfg, "totally-bogus", None, None, None)


def test_json_safe_sanitizes_non_finite_floats():
    import json
    import math

    raw = {
        "psnr": math.inf,
        "neg": -math.inf,
        "nan": math.nan,
        "ok": 30.5,
        "nested": [{"w": math.inf}],
    }
    safe = smoke._json_safe(raw)
    assert safe["psnr"] == "inf"
    assert safe["neg"] == "-inf"
    assert safe["nan"] == "nan"
    assert safe["ok"] == 30.5
    assert safe["nested"][0]["w"] == "inf"
    # Strict JSON (rejects NaN/Infinity) must now serialize cleanly.
    text = json.dumps(safe, allow_nan=False)
    assert json.loads(text)["ok"] == 30.5


def test_summarize_counts_by_status():
    records = [
        smoke.build_record(smoke.build_config_matrix()[0], "ok", None, None, 1.0),
        smoke.build_record(smoke.build_config_matrix()[1], "error", "boom", None, 1.0),
        smoke.build_record(smoke.build_config_matrix()[2], "ok", None, None, 1.0),
        smoke.build_record(smoke.build_config_matrix()[3], "unsupported", "no nvfp4", None, None),
    ]
    summary = smoke.summarize_matrix(records)
    assert summary["total"] == 4
    assert summary["ok"] == 2
    assert summary["by_status"] == {"ok": 2, "error": 1, "unsupported": 1}
    # records[0] is the bf16/VANILLA baseline and it is ok here.
    assert summary["baseline_ok"] is True


# --------------------------------------------------------------------------- #
# baseline-anchored success gate (Round 26)
# --------------------------------------------------------------------------- #


def test_baseline_ran_ok_true_when_baseline_ok():
    matrix = smoke.build_config_matrix()
    records = [smoke.build_record(matrix[0], "ok", None, None, 1.0)]
    assert smoke.baseline_ran_ok(records) is True


def test_baseline_ran_ok_false_when_baseline_failed_even_if_other_ok():
    # The quality anchor (bf16/VANILLA) failed but a later config passed: the
    # sweep is NOT meaningful, so the success gate must fail.
    matrix = smoke.build_config_matrix()
    records = [
        smoke.build_record(matrix[0], "error", "baseline boom", None, 1.0),  # baseline
        smoke.build_record(matrix[1], "ok", None, None, 1.0),  # later config ok
    ]
    assert smoke.baseline_ran_ok(records) is False
    # And the summary the exit code is derived from agrees.
    assert smoke.summarize_matrix(records)["baseline_ok"] is False


def test_baseline_ran_ok_false_when_no_baseline_record():
    matrix = smoke.build_config_matrix()
    records = [smoke.build_record(matrix[1], "ok", None, None, 1.0)]  # no baseline
    assert smoke.baseline_ran_ok(records) is False


def test_runner_has_no_plan_process_labels():
    # PLAN.md forbids plan-process terms (AC-*, Milestone, Step, Phase) in
    # implementation code / runtime strings.
    text = _SMOKE_PATH.read_text()
    assert "AC-" not in text
    for label in ("Milestone", "Phase "):
        assert label not in text


# --------------------------------------------------------------------------- #
# stdlib-only: the pure helpers import + run without numpy/torch (Round 25)
# --------------------------------------------------------------------------- #


def test_smoke_pure_helpers_run_without_numpy(tmp_path):
    """The runner's module import + pure helpers must not require numpy/torch.

    (Heavy deps live inside ``main()``.) Spawn a subprocess whose import system
    raises on numpy/torch/tensorrt_llm, load the module by path, and run the pure
    helpers; assert a clean exit.
    """
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
        f"spec = importlib.util.spec_from_file_location('ltx2_retake_smoke', {str(_SMOKE_PATH)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "matrix = m.build_config_matrix()\n"
        "assert len(matrix) == 4\n"
        "recs = [m.build_record(c, 'ok', None, None, 1.0) for c in matrix]\n"
        "assert m.summarize_matrix(recs)['ok'] == 4\n"
        "ok, reason = m.precheck_capability({'attention_backend': 'FA4'}, {'fa4_available': False})\n"
        "assert ok is False\n"
        "print('SMOKE_PURE_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", bootstrap],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "SMOKE_PURE_OK" in proc.stdout
    assert "blocked for test" not in proc.stderr
