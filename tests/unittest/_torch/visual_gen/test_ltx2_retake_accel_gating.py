# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side unit tests for the pure helpers in ``ltx2_retake_accel_gating``.

Only the stdlib-only helpers (axis matrix, capability precheck, status
classification, fastest-attention selection, stack construction, incremental
deltas, record assembly, gating summary, JSON sanitization) are exercised; the
heavy resident measured loop lives inside ``main()`` and reuses the oracle /
timing modules. The runner lives under ``examples/`` so it is loaded by path via
``importlib``.
"""

import importlib.util
from pathlib import Path

import pytest

_ACCEL_PATH = (
    Path(__file__).resolve().parents[4] / "examples" / "visual_gen" / "ltx2_retake_accel_gating.py"
)


def _load_accel():
    spec = importlib.util.spec_from_file_location("ltx2_retake_accel_gating", _ACCEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


accel = _load_accel()

_ORACLE_PATH = (
    Path(__file__).resolve().parents[4] / "examples" / "visual_gen" / "ltx2_retake_oracle.py"
)


def _load_oracle():
    spec = importlib.util.spec_from_file_location("ltx2_retake_oracle", _ORACLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = _load_oracle()


class _FakeCompilePipe:
    """Records whether the torch.compile setup was invoked."""

    def __init__(self):
        self.compile_calls = 0

    def torch_compile(self):
        self.compile_calls += 1


def test_apply_torch_compile_invokes_compile_only_when_enabled():
    # The torch_compile acceleration axis is a real compiled measurement only if
    # build_pipeline actually calls pipe.torch_compile() (setting the config flag
    # alone never compiles). Guarded import keeps this runnable without torch.
    disabled = _FakeCompilePipe()
    assert oracle._apply_torch_compile(disabled, False) is disabled
    assert disabled.compile_calls == 0

    enabled = _FakeCompilePipe()
    assert oracle._apply_torch_compile(enabled, True) is enabled
    assert enabled.compile_calls == 1


def _steady(p50):
    """A minimal steady_warm dict for constructing test records."""
    return {"p50": p50, "p90": p50, "min": p50, "count": 1}


def _rec(accel_mod, config, status="ok", p50=None):
    """Build a record with an optional steady_warm p50."""
    timing = {"steady_warm": _steady(p50)} if p50 is not None else None
    return accel_mod.build_record(config, status, None, timing, None, None)


# --------------------------------------------------------------------------- #
# build_axis_matrix
# --------------------------------------------------------------------------- #


def test_matrix_exactly_one_baseline():
    matrix = accel.build_axis_matrix()
    baselines = [c for c in matrix if c.get("baseline")]
    assert len(baselines) == 1
    b = baselines[0]
    assert b["label"] == "bf16/VANILLA"
    assert b["axis"] == "baseline" and b["kind"] == "baseline"
    assert b["attention_backend"] == "VANILLA"
    assert b["cuda_graph"] is False and b["torch_compile"] is False
    assert b["quant_algo"] is None


def test_matrix_labels_unique():
    labels = [c["label"] for c in accel.build_axis_matrix()]
    assert len(labels) == len(set(labels))


def test_matrix_single_axis_rows_toggle_exactly_one_axis():
    matrix = accel.build_axis_matrix()
    baseline = next(c for c in matrix if c.get("baseline"))
    single = [c for c in matrix if c.get("kind") == "single-axis"]
    # The four independent axes: torch_compile, cuda_graph, two attention rows, quant.
    axes = sorted(c["axis"] for c in single)
    assert axes == ["attn", "attn", "cuda_graph", "quant", "torch_compile"]

    def _diff_axes(cfg):
        diffs = set()
        if cfg["attention_backend"] != baseline["attention_backend"]:
            diffs.add("attn")
        if bool(cfg["cuda_graph"]) != bool(baseline["cuda_graph"]):
            diffs.add("cuda_graph")
        if bool(cfg["torch_compile"]) != bool(baseline["torch_compile"]):
            diffs.add("torch_compile")
        if cfg["quant_algo"] != baseline["quant_algo"]:
            diffs.add("quant")
        return diffs

    for cfg in single:
        assert len(_diff_axes(cfg)) == 1, f"{cfg['label']} toggles more than one axis"


def test_matrix_no_static_stack_rows():
    # The stack is built dynamically after measurement, never statically.
    assert all(c.get("kind") != "stack" for c in accel.build_axis_matrix())


# --------------------------------------------------------------------------- #
# precheck_capability
# --------------------------------------------------------------------------- #


def test_precheck_fa4_gated_on_caps():
    cfg = {"attention_backend": "FA4"}
    ok, reason = accel.precheck_capability(cfg, {"fa4": False})
    assert ok is False and "FA4" in reason
    ok, _ = accel.precheck_capability(cfg, {"fa4": True})
    assert ok is True


def test_precheck_cutedsl_gated_on_caps():
    cfg = {"attention_backend": "CUTEDSL"}
    ok, reason = accel.precheck_capability(cfg, {"cutedsl": False})
    assert ok is False and "CuTeDSL" in reason
    ok, _ = accel.precheck_capability(cfg, {"cutedsl": True})
    assert ok is True


def test_precheck_nvfp4_gated_on_caps():
    cfg = {"attention_backend": "VANILLA", "quant_algo": "NVFP4"}
    ok, reason = accel.precheck_capability(cfg, {"nvfp4": False})
    assert ok is False and "NVFP4" in reason
    ok, _ = accel.precheck_capability(cfg, {"nvfp4": True})
    assert ok is True


def test_precheck_torch_compile_and_cuda_graph_always_supported():
    tc = {"attention_backend": "VANILLA", "torch_compile": True}
    cg = {"attention_backend": "VANILLA", "cuda_graph": True}
    # No caps at all: both are always attempted.
    assert accel.precheck_capability(tc, {}) == (True, None)
    assert accel.precheck_capability(cg, {}) == (True, None)


# --------------------------------------------------------------------------- #
# classify_status / regresses_vs_baseline
# --------------------------------------------------------------------------- #


def test_classify_unsupported_carries_precheck_reason():
    status, reason = accel.classify_status(False, "no FA4", False, None)
    assert status == "unsupported" and reason == "no FA4"


def test_classify_not_applicable_when_prerequisite_absent():
    status, reason = accel.classify_status(True, "baseline failed", False, None, applicable=False)
    assert status == "not-applicable" and reason == "baseline failed"


def test_classify_error_when_run_failed():
    status, reason = accel.classify_status(True, None, False, "RuntimeError: boom")
    assert status == "error" and "boom" in reason


def test_classify_ok_when_ran():
    assert accel.classify_status(True, None, True, None) == ("ok", None)


def test_classify_relays_run_condition_status():
    status, reason = accel.classify_status(
        True, None, True, None, run_condition=("regresses", "slower")
    )
    assert status == "regresses" and reason == "slower"


def test_classify_rejects_bogus_run_condition_status():
    with pytest.raises(ValueError):
        accel.classify_status(True, None, True, None, run_condition=("bogus", "x"))


def test_regresses_vs_baseline_tolerance():
    assert accel.regresses_vs_baseline(2.0, 1.0) is True
    assert accel.regresses_vs_baseline(1.0, 2.0) is False
    # tol is a multiplicative margin: a 5% slowdown is within a 10% tolerance.
    assert accel.regresses_vs_baseline(1.05, 1.0, tol=1.1) is False
    assert accel.regresses_vs_baseline(1.2, 1.0, tol=1.1) is True
    # Missing values never regress.
    assert accel.regresses_vs_baseline(None, 1.0) is False
    assert accel.regresses_vs_baseline(1.0, None) is False


# --------------------------------------------------------------------------- #
# fastest_attention
# --------------------------------------------------------------------------- #


def test_fastest_attention_picks_lowest_steady_p50():
    matrix = {c["label"]: c for c in accel.build_axis_matrix()}
    records = [
        _rec(accel, matrix["bf16/VANILLA"], "ok", p50=3.0),
        _rec(accel, matrix["bf16/FA4"], "ok", p50=1.5),
        _rec(accel, matrix["bf16/CUTEDSL"], "ok", p50=2.0),
        # A non-attention ok row must be ignored.
        _rec(accel, matrix["bf16/cuda_graph"], "ok", p50=0.5),
    ]
    assert accel.fastest_attention(records) == "FA4"


def test_fastest_attention_ignores_non_ok_rows():
    matrix = {c["label"]: c for c in accel.build_axis_matrix()}
    records = [
        _rec(accel, matrix["bf16/VANILLA"], "ok", p50=3.0),
        _rec(accel, matrix["bf16/FA4"], "error", p50=0.1),  # not ok -> ignored
    ]
    assert accel.fastest_attention(records) == "VANILLA"


def test_fastest_attention_deterministic_tie_break():
    matrix = {c["label"]: c for c in accel.build_axis_matrix()}
    # Same p50 for FA4 and CUTEDSL: FA4 wins by fixed backend order.
    records = [
        _rec(accel, matrix["bf16/CUTEDSL"], "ok", p50=2.0),
        _rec(accel, matrix["bf16/FA4"], "ok", p50=2.0),
    ]
    assert accel.fastest_attention(records) == "FA4"


def test_fastest_attention_none_when_no_eligible_rows():
    assert accel.fastest_attention([]) is None


# --------------------------------------------------------------------------- #
# build_stack_configs
# --------------------------------------------------------------------------- #


def test_stack_fixed_order_with_nvfp4():
    configs = accel.build_stack_configs("FA4", {"nvfp4": True})
    steps = [(c["step"], c["axis"], c["kind"]) for c in configs]
    assert steps == [
        (1, "stack", "stack"),
        (2, "stack", "stack"),
        (3, "stack", "stack"),
        (4, "stack", "stack"),
    ]
    # Cumulative wiring: torch_compile -> +cuda_graph -> +fastest_attn -> +NVFP4.
    assert configs[0]["torch_compile"] is True and configs[0]["cuda_graph"] is False
    assert configs[1]["torch_compile"] is True and configs[1]["cuda_graph"] is True
    assert configs[2]["attention_backend"] == "FA4"
    assert configs[2]["cuda_graph"] is True and configs[2]["torch_compile"] is True
    assert configs[3]["quant_algo"] == "NVFP4" and configs[3]["dtype"] == "nvfp4"
    assert configs[3]["attention_backend"] == "FA4"


def test_stack_omits_nvfp4_when_caps_disallow():
    configs = accel.build_stack_configs("CUTEDSL", {"nvfp4": False})
    assert len(configs) == 3
    assert all(c["quant_algo"] is None for c in configs)
    assert [c["step"] for c in configs] == [1, 2, 3]
    assert configs[2]["attention_backend"] == "CUTEDSL"


def test_stack_defaults_attention_to_vanilla_when_none():
    configs = accel.build_stack_configs(None, {"nvfp4": False})
    assert configs[2]["attention_backend"] == "VANILLA"


# --------------------------------------------------------------------------- #
# incremental_deltas
# --------------------------------------------------------------------------- #


def test_incremental_deltas_hand_example():
    stack_configs = accel.build_stack_configs("FA4", {"nvfp4": False})
    p50s = [2.0, 1.0, 0.5]
    stack_records = [
        accel.build_record(cfg, "ok", None, {"steady_warm": _steady(p)}, None, None)
        for cfg, p in zip(stack_configs, p50s)
    ]
    deltas = accel.incremental_deltas(stack_records, baseline_p50=4.0)
    # step 1: prev=baseline 4.0 -> 4/2 = 2.0; vs baseline 4/2 = 2.0
    assert deltas[0]["delta_vs_prev"] == pytest.approx(2.0)
    assert deltas[0]["delta_vs_baseline"] == pytest.approx(2.0)
    # step 2: prev=2.0 -> 2/1 = 2.0; vs baseline 4/1 = 4.0
    assert deltas[1]["delta_vs_prev"] == pytest.approx(2.0)
    assert deltas[1]["delta_vs_baseline"] == pytest.approx(4.0)
    # step 3: prev=1.0 -> 1/0.5 = 2.0; vs baseline 4/0.5 = 8.0
    assert deltas[2]["delta_vs_prev"] == pytest.approx(2.0)
    assert deltas[2]["delta_vs_baseline"] == pytest.approx(8.0)


def test_incremental_deltas_handles_missing_p50():
    stack_configs = accel.build_stack_configs("FA4", {"nvfp4": False})
    stack_records = [
        accel.build_record(stack_configs[0], "error", "boom", None, None, None),
        accel.build_record(stack_configs[1], "ok", None, {"steady_warm": _steady(1.0)}, None, None),
    ]
    deltas = accel.incremental_deltas(stack_records, baseline_p50=4.0)
    assert deltas[0]["delta_vs_prev"] is None
    # The failed step does not advance prev, so step 2 still compares to baseline.
    assert deltas[1]["delta_vs_prev"] == pytest.approx(4.0)
    assert deltas[1]["delta_vs_baseline"] == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# baseline_required_ok / summarize_gating
# --------------------------------------------------------------------------- #


def test_baseline_required_ok():
    matrix = {c["label"]: c for c in accel.build_axis_matrix()}
    ok_records = [_rec(accel, matrix["bf16/VANILLA"], "ok", p50=2.0)]
    assert accel.baseline_required_ok(ok_records) is True
    bad_records = [_rec(accel, matrix["bf16/VANILLA"], "error")]
    assert accel.baseline_required_ok(bad_records) is False
    # No baseline row at all.
    assert accel.baseline_required_ok([_rec(accel, matrix["bf16/FA4"], "ok", p50=1.0)]) is False


def test_summarize_gating_counts_and_speedup():
    matrix = {c["label"]: c for c in accel.build_axis_matrix()}
    stack_cfg = accel.build_stack_configs("FA4", {"nvfp4": False})
    records = [
        _rec(accel, matrix["bf16/VANILLA"], "ok", p50=4.0),
        _rec(accel, matrix["bf16/FA4"], "ok", p50=2.0),
        _rec(accel, matrix["bf16/CUTEDSL"], "error"),
        accel.build_record(stack_cfg[2], "ok", None, {"steady_warm": _steady(1.0)}, None, None),
    ]
    summary = accel.summarize_gating(records)
    assert summary["total"] == 4
    assert summary["by_status"] == {"ok": 3, "error": 1}
    assert summary["baseline_ok"] is True
    assert summary["fastest_attention"] == "FA4"
    # Full stack p50 1.0 vs baseline 4.0 -> 4x.
    assert summary["stack_speedup_vs_baseline"] == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# build_record / _json_safe
# --------------------------------------------------------------------------- #


def test_build_record_full_schema():
    cfg = next(c for c in accel.build_axis_matrix() if c["label"] == "bf16/FA4")
    timing = {
        "raw_samples": [2.0, 1.9, 2.1],
        "first_served": 2.0,
        "steady_warm": _steady(1.95),
        "per_stage": {"denoise_total": {"p50": 1.0}},
        "cold_model_build_load": 30.0,
    }
    rec = accel.build_record(
        cfg, "ok", None, timing, {"psnr": 30.0}, {"allocated": 1, "reserved": 2}
    )
    expected_keys = {
        "label",
        "dtype",
        "attention_backend",
        "quant_algo",
        "cuda_graph",
        "torch_compile",
        "axis",
        "kind",
        "baseline",
        "step",
        "status",
        "reason",
        "raw_samples",
        "first_served",
        "steady_warm",
        "per_stage",
        "quality_informational",
        "peak_memory",
        "cold_model_build_load",
    }
    assert set(rec.keys()) == expected_keys
    assert rec["label"] == "bf16/FA4"
    assert rec["attention_backend"] == "FA4"
    assert rec["axis"] == "attn" and rec["kind"] == "single-axis"
    assert rec["quality_informational"] == {"psnr": 30.0}
    assert rec["peak_memory"] == {"allocated": 1, "reserved": 2}
    assert rec["cold_model_build_load"] == 30.0


def test_build_record_rejects_invalid_status():
    cfg = accel.build_axis_matrix()[0]
    with pytest.raises(ValueError):
        accel.build_record(cfg, "totally-bogus", None, None, None, None)


def test_build_record_none_timing_yields_empty_samples():
    cfg = accel.build_axis_matrix()[0]
    rec = accel.build_record(cfg, "not-applicable", "no baseline", None, None, None)
    assert rec["raw_samples"] == []
    assert rec["steady_warm"] is None
    assert rec["peak_memory"] is None


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
    safe = accel._json_safe(raw)
    assert safe["psnr"] == "inf"
    assert safe["neg"] == "-inf"
    assert safe["nan"] == "nan"
    assert safe["ok"] == 30.5
    assert safe["nested"][0]["w"] == "inf"
    text = json.dumps(safe, allow_nan=False)  # strict JSON must not raise
    assert json.loads(text)["ok"] == 30.5


# --------------------------------------------------------------------------- #
# stdlib-only: the pure helpers import + run without numpy/torch
# --------------------------------------------------------------------------- #


def test_accel_pure_helpers_run_without_numpy():
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
        f"spec = importlib.util.spec_from_file_location('ltx2_retake_accel_gating', {str(_ACCEL_PATH)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "matrix = m.build_axis_matrix()\n"
        "assert len([c for c in matrix if c['baseline']]) == 1\n"
        "recs = [m.build_record(c, 'ok', None, {'steady_warm': {'p50': float(i + 1)}}, None, None)\n"
        "        for i, c in enumerate(matrix)]\n"
        "assert m.summarize_gating(recs)['baseline_ok'] is True\n"
        "ok, reason = m.precheck_capability({'attention_backend': 'FA4'}, {'fa4': False})\n"
        "assert ok is False\n"
        "stack = m.build_stack_configs('FA4', {'nvfp4': False})\n"
        "assert len(stack) == 3\n"
        "assert m.incremental_deltas([], 4.0) == []\n"
        "print('ACCEL_PURE_OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", bootstrap], capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "ACCEL_PURE_OK" in proc.stdout
    assert "blocked for test" not in proc.stderr


# --------------------------------------------------------------------------- #
# no plan-process labels leak into the checked-in runner
# --------------------------------------------------------------------------- #


def test_accel_gating_runner_has_no_plan_process_labels():
    # Bans plan-process terminology in checked-in implementation text. The banned
    # tokens are assembled from fragments so this guard file does not itself
    # contain them literally (its own check would otherwise flag them).
    text = _ACCEL_PATH.read_text()
    banned = [
        "A" + "C-",
        "Stage" + " 1",
        "Stage" + " 2",
        "Stage" + "-2",
        "Mile" + "stone",
        "Ph" + "ase ",
    ]
    for label in banned:
        assert label not in text, f"plan-process label {label!r} leaked into the gating runner"
