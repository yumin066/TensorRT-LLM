# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from scripts.visualgen_eval import qwen_image_layered_quality_gate as gate


def _metrics(*, qat_psnr: float = 39.0, dynamic_psnr: float = 32.0) -> dict:
    return {
        "samples": [{"id": "sample_a"}, {"id": "sample_b"}],
        "variants": {
            "bf16": {},
            "k12": {},
            "all_layer_dynamic_mxfp8": {},
            "qat_all_layer_mxfp8": {},
        },
        "comparisons": [
            {
                "sample_id": sample_id,
                "variant": variant,
                "psnr": value,
                "ssim": 0.99,
            }
            for sample_id in ("sample_a", "sample_b")
            for variant, value in (
                ("k12", 38.5),
                ("all_layer_dynamic_mxfp8", dynamic_psnr),
                ("qat_all_layer_mxfp8", qat_psnr),
            )
        ],
    }


def _coverage(**overrides) -> dict:
    coverage = {
        "status": "passed",
        "target_count": 840,
        "static_mxfp8_target_count": 840,
        "bf16_exclusion_count": 4,
        "total_linear_count": 844,
        "pipeline_quant_algo": "FP8_BLOCK_SCALES",
        "pipeline_dynamic_weight_quant": False,
        "pipeline_force_dynamic_quantization": False,
        "failures": {},
    }
    coverage.update(overrides)
    return coverage


def test_quality_gate_accepts_qat_above_threshold_and_dynamic():
    report = gate.evaluate_quality_gate(_metrics(), coverage=_coverage())

    assert report["status"] == "passed"
    assert report["sample_count"] == 2
    assert report["loader_coverage"]["static_mxfp8_target_count"] == 840
    assert all(row["qat_psnr_improvement_over_dynamic"] > 0 for row in report["per_sample"])


def test_quality_gate_rejects_qat_below_threshold():
    report = gate.evaluate_quality_gate(_metrics(qat_psnr=37.5), coverage=_coverage())

    assert report["status"] == "failed"
    assert any("PSNR 37.5000 < 38.0000" in failure for failure in report["failures"])


def test_quality_gate_rejects_qat_not_better_than_dynamic():
    report = gate.evaluate_quality_gate(
        _metrics(qat_psnr=39.0, dynamic_psnr=39.5),
        coverage=_coverage(),
    )

    assert report["status"] == "failed"
    assert any("<= all_layer_dynamic_mxfp8" in failure for failure in report["failures"])


def test_quality_gate_rejects_bad_loader_contract():
    report = gate.evaluate_quality_gate(
        _metrics(),
        coverage=_coverage(static_mxfp8_target_count=839),
    )

    assert report["status"] == "failed"
    assert any("static_mxfp8_target_count=839" in failure for failure in report["failures"])


def test_quality_gate_rejects_missing_loader_coverage_by_default():
    report = gate.evaluate_quality_gate(_metrics())

    assert report["status"] == "failed"
    assert report["loader_coverage"]["status"] == "missing"
    assert any("loader coverage JSON is required" in failure for failure in report["failures"])


def test_quality_gate_can_explicitly_skip_loader_coverage_for_diagnostics():
    report = gate.evaluate_quality_gate(_metrics(), require_coverage=False)

    assert report["status"] == "passed"
    assert report["loader_coverage"]["status"] == "skipped"


def test_quality_gate_rejects_844_slash_844_wording():
    metrics = _metrics()
    metrics["variants"]["qat_all_layer_mxfp8"]["note"] = "not allowed: 844/844 MXFP8"

    report = gate.evaluate_quality_gate(metrics, coverage=_coverage())

    assert report["status"] == "failed"
    assert "forbidden 844/844 wording" in "\n".join(report["failures"])
