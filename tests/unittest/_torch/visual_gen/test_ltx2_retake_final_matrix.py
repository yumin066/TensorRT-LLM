# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side unit tests for ``ltx2_retake_final_matrix`` (stdlib-only aggregator)."""

import importlib.util
import json
from pathlib import Path

import pytest

_FM_PATH = (
    Path(__file__).resolve().parents[4] / "examples" / "visual_gen" / "ltx2_retake_final_matrix.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("ltx2_retake_final_matrix", _FM_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fm = _load()


# --------------------------------------------------------------------------- #
# speedup + amortization
# --------------------------------------------------------------------------- #


def test_speedup():
    assert fm.speedup(83.3, 1.83) == pytest.approx(45.5, abs=0.5)
    assert fm.speedup(None, 1.0) is None
    assert fm.speedup(1.0, 0.0) is None


def test_amortization_break_even_and_ratios():
    # cold 90s, warm 1.83s, every-rebuild 83.3s.
    a = fm.amortization(90.0, 1.83, 83.3)
    # break-even = ceil(90 / (83.3 - 1.83)) = ceil(1.10) = 2.
    assert a["break_even_calls"] == 2
    assert a["per_call_speedup"] == pytest.approx(83.3 / 1.83, abs=0.1)
    # 100 calls: mode_a=8330, mode_b=90+183=273 -> ~30.5x.
    assert a["ratio_100_calls"] == pytest.approx(8330 / 273, abs=0.5)


def test_amortization_break_even_strict_on_integer_ratio():
    # cold 100, warm 1, rebuild 51 -> ratio = 100/50 = 2.0 exactly. At N=2 the two
    # are a TIE, so the first STRICT win is N=3 (floor(2.0)+1), not ceil(2.0)=2.
    a = fm.amortization(100.0, 1.0, 51.0)
    assert a["break_even_calls"] == 3


def test_amortization_none_on_missing():
    a = fm.amortization(None, 1.0, 83.0)
    assert a["break_even_calls"] is None


def test_amortization_no_break_even_when_warm_slower():
    a = fm.amortization(90.0, 100.0, 83.3)  # warm slower than rebuild (degenerate)
    assert a["break_even_calls"] is None


# --------------------------------------------------------------------------- #
# section builders + gaps
# --------------------------------------------------------------------------- #


def test_build_latency_section_extracts_and_records_gaps():
    mode_a = {"summary": {"total": {"p50": 83.3}, "model_build_load": {"p50": 68.0}}}
    serve = {"steady_warm": {"wall": {"p50": 2.12}, "generation": {"p50": 1.73}}}
    timing = {
        "timeline": {"steady_warm": {"wall": {"p50": 1.83}}, "cold_model_build_load_seconds": 90.0}
    }
    sec = fm.build_latency_section(mode_a, serve, timing)
    assert sec["mode_a_every_rebuild_total_p50_s"] == 83.3
    assert sec["mode_b_serve_http_wall_p50_s"] == 2.12
    assert sec["mode_b_cold_load_s"] == 90.0
    assert sec["amortization"]["break_even_calls"] == 2
    assert sec["gaps"] == []


def test_build_latency_amortization_prefers_wall_over_generation():
    # No pipeline-direct timing; serve has both wall + generation. The amortization
    # must use the serve WALL (incl. HTTP overhead), not the optimistic generation.
    mode_a = {"summary": {"total": {"p50": 83.3}, "model_build_load": {"p50": 68.0}}}
    serve = {"steady_warm": {"wall": {"p50": 2.12}, "generation": {"p50": 1.73}}}
    sec = fm.build_latency_section(mode_a, serve, None)
    # per_call_speedup = 83.3 / warm; wall(2.12) -> 39.3, generation(1.73) -> 48.2.
    assert sec["amortization"]["per_call_speedup"] == pytest.approx(83.3 / 2.12, abs=0.1)


def test_build_latency_section_falls_back_to_mode_a_load_and_flags_gaps():
    mode_a = {"summary": {"total": {"p50": 83.3}, "model_build_load": {"p50": 68.0}}}
    sec = fm.build_latency_section(mode_a, None, None)
    assert sec["mode_b_cold_load_s"] == 68.0  # fell back to Mode A load
    assert set(sec["gaps"]) == {"serve", "timing"}


def test_build_quant_section():
    quant = {
        "records": [
            {
                "mode": "bf16",
                "status": "ok",
                "denoise": {"p50": 1.23},
                "latency_delta": {"speedup_vs_bf16": 1.0},
                "resident_allocated_gib": 65.4,
                "memory_delta": {"saved_gib": 0.0},
            },
            {
                "mode": "nvfp4",
                "status": "ok",
                "denoise": {"p50": 0.62},
                "latency_delta": {"speedup_vs_bf16": 1.99},
                "resident_allocated_gib": 40.0,
                "memory_delta": {"saved_gib": 25.4},
                "quality_informational": {"window": {"psnr": 10.1, "ssim": 0.228}},
            },
        ]
    }
    sec = fm.build_quant_section(quant)
    assert sec["modes"][1]["mode"] == "nvfp4"
    assert sec["modes"][1]["speedup_vs_bf16"] == 1.99
    assert sec["modes"][1]["window_psnr"] == 10.1


def test_build_device_section():
    resolution = {
        "device_query": {"name": "RTX PRO 6000"},
        "records": [
            {
                "label": "1080p",
                "status": "ok",
                "token_ratio_vs_baseline": 12.75,
                "cold": {"model_build_load": 94.7, "first_inference": 34.7},
                "warm_denoise": {"p50": 29.5},
                "inference_peak": {"reserved_bytes": 98 * 1024**3},
            },
        ],
    }
    sec = fm.build_device_section(resolution)
    assert sec["device"] == "RTX PRO 6000"
    assert sec["resolutions"][0]["warm_denoise_p50_s"] == 29.5
    assert sec["resolutions"][0]["infer_reserved_gib"] == 98.0


def test_build_matrix_records_all_gaps_when_dir_empty(tmp_path):
    matrix = fm.build_matrix(str(tmp_path))
    assert set(matrix["gaps"]) == {
        "mode_a",
        "serve",
        "timing",
        "smoke",
        "attn",
        "quant",
        "resolution",
    }
    # render must not crash on an all-gaps matrix.
    md = fm.render_markdown(matrix)
    assert "customer matrix" in md


def test_render_markdown_includes_recommendation(tmp_path):
    matrix = fm.build_matrix(str(tmp_path), recommendation="Use FP8 in production.")
    md = fm.render_markdown(matrix)
    assert "Use FP8 in production." in md


# --------------------------------------------------------------------------- #
# load_json + json_safe + no-label
# --------------------------------------------------------------------------- #


def test_load_json_missing_and_valid(tmp_path):
    assert fm.load_json(str(tmp_path / "nope.json")) is None
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"a": 1}))
    assert fm.load_json(str(p)) == {"a": 1}


def test_json_safe():
    out = fm._json_safe({"a": float("inf"), "b": [float("nan")]})
    assert out["a"] == "inf" and out["b"][0] == "nan"


def test_final_matrix_has_no_plan_process_labels():
    text = _FM_PATH.read_text()
    assert "AC-" not in text
    for label in ("Milestone", "Phase "):
        assert label not in text
