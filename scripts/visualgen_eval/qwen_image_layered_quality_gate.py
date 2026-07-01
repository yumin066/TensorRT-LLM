# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Enforce final Qwen-Image-Layered QAT MXFP8 quality gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REFERENCE_VARIANT = "bf16"
K12_VARIANT = "k12"
DYNAMIC_VARIANT = "all_layer_dynamic_mxfp8"
QAT_VARIANT = "qat_all_layer_mxfp8"
DEFAULT_PSNR_THRESHOLD = 38.0
EXPECTED_STATIC_TARGET_COUNT = 840
EXPECTED_BF16_EXCLUSION_COUNT = 4


def load_json(path: str | Path) -> dict[str, Any]:
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return loaded


def evaluate_quality_gate(
    metrics: dict[str, Any],
    *,
    coverage: dict[str, Any] | None = None,
    require_coverage: bool = True,
    psnr_threshold: float = DEFAULT_PSNR_THRESHOLD,
    dynamic_variant: str = DYNAMIC_VARIANT,
    qat_variant: str = QAT_VARIANT,
    k12_variant: str = K12_VARIANT,
) -> dict[str, Any]:
    failures: list[str] = []
    comparisons = _comparison_index(metrics)
    sample_ids = _sample_ids(metrics)
    variants = metrics.get("variants")
    if not isinstance(variants, dict):
        failures.append("metrics.variants must be a mapping")
        variants = {}

    required_variants = (REFERENCE_VARIANT, k12_variant, dynamic_variant, qat_variant)
    for variant in required_variants:
        if variant not in variants:
            failures.append(f"missing required variant: {variant}")

    per_sample = []
    for sample_id in sample_ids:
        sample_report = {"sample_id": sample_id}
        qat = comparisons.get((sample_id, qat_variant))
        dynamic = comparisons.get((sample_id, dynamic_variant))
        k12 = comparisons.get((sample_id, k12_variant))
        if qat is None:
            failures.append(f"{sample_id}: missing {qat_variant} comparison")
            per_sample.append(sample_report)
            continue
        if dynamic is None:
            failures.append(f"{sample_id}: missing {dynamic_variant} comparison")
        if k12 is None:
            failures.append(f"{sample_id}: missing {k12_variant} comparison")

        qat_psnr = _metric_to_float(qat.get("psnr"), field=f"{sample_id}.{qat_variant}.psnr")
        sample_report["qat_psnr"] = _json_metric(qat_psnr)
        sample_report["qat_ssim"] = qat.get("ssim")
        if qat_psnr < psnr_threshold:
            failures.append(
                f"{sample_id}: {qat_variant} PSNR {qat_psnr:.4f} < {psnr_threshold:.4f}"
            )

        if dynamic is not None:
            dynamic_psnr = _metric_to_float(
                dynamic.get("psnr"),
                field=f"{sample_id}.{dynamic_variant}.psnr",
            )
            sample_report["dynamic_psnr"] = _json_metric(dynamic_psnr)
            sample_report["qat_psnr_improvement_over_dynamic"] = _json_metric(
                qat_psnr - dynamic_psnr
            )
            if not qat_psnr > dynamic_psnr:
                failures.append(
                    f"{sample_id}: {qat_variant} PSNR {qat_psnr:.4f} <= "
                    f"{dynamic_variant} PSNR {dynamic_psnr:.4f}"
                )

        if k12 is not None:
            sample_report["k12_psnr"] = k12.get("psnr")
            sample_report["k12_ssim"] = k12.get("ssim")
        per_sample.append(sample_report)

    coverage_report = _coverage_report(coverage, require_coverage=require_coverage)
    if coverage_report is not None:
        failures.extend(coverage_report["failures"])

    encoded_inputs = json.dumps(metrics, sort_keys=True)
    if coverage is not None:
        encoded_inputs += json.dumps(coverage, sort_keys=True)
    if "844/844" in encoded_inputs:
        failures.append("forbidden 844/844 wording found in metrics or coverage JSON")

    return {
        "status": "failed" if failures else "passed",
        "psnr_threshold": psnr_threshold,
        "reference_variant": REFERENCE_VARIANT,
        "k12_variant": k12_variant,
        "dynamic_variant": dynamic_variant,
        "qat_variant": qat_variant,
        "sample_count": len(sample_ids),
        "per_sample": per_sample,
        "loader_coverage": coverage_report,
        "failures": failures,
    }


def _comparison_index(metrics: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    comparisons = metrics.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("metrics.comparisons must be a list")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for item in comparisons:
        if not isinstance(item, dict):
            raise ValueError("metrics.comparisons entries must be mappings")
        sample_id = item.get("sample_id")
        variant = item.get("variant")
        if not isinstance(sample_id, str) or not isinstance(variant, str):
            raise ValueError("comparison entries require string sample_id and variant")
        key = (sample_id, variant)
        if key in indexed:
            raise ValueError(f"duplicate comparison entry for {sample_id}/{variant}")
        indexed[key] = item
    return indexed


def _sample_ids(metrics: dict[str, Any]) -> list[str]:
    samples = metrics.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("metrics.samples must be a non-empty list")
    sample_ids = []
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("id"), str):
            raise ValueError("metrics.samples entries require string id")
        sample_ids.append(sample["id"])
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"metrics.samples contains duplicate ids: {sample_ids}")
    return sample_ids


def _evaluate_loader_coverage(coverage: dict[str, Any]) -> dict[str, Any]:
    failures = []
    expected = {
        "target_count": EXPECTED_STATIC_TARGET_COUNT,
        "static_mxfp8_target_count": EXPECTED_STATIC_TARGET_COUNT,
        "bf16_exclusion_count": EXPECTED_BF16_EXCLUSION_COUNT,
    }
    if coverage.get("status") != "passed":
        failures.append(f"loader coverage status={coverage.get('status')!r}, expected 'passed'")
    for key, value in expected.items():
        if coverage.get(key) != value:
            failures.append(f"loader coverage {key}={coverage.get(key)!r}, expected {value}")
    if coverage.get("pipeline_dynamic_weight_quant") is not False:
        failures.append("loader coverage pipeline_dynamic_weight_quant must be False")
    if coverage.get("pipeline_force_dynamic_quantization") is not False:
        failures.append("loader coverage pipeline_force_dynamic_quantization must be False")

    coverage_failures = coverage.get("failures")
    if isinstance(coverage_failures, dict):
        non_empty = {key: value for key, value in coverage_failures.items() if value}
        if non_empty:
            failures.append(f"loader coverage has failure lists: {sorted(non_empty)}")
    elif coverage_failures is not None:
        failures.append("loader coverage failures must be a mapping when present")

    return {
        "status": "failed" if failures else "passed",
        "target_count": coverage.get("target_count"),
        "static_mxfp8_target_count": coverage.get("static_mxfp8_target_count"),
        "bf16_exclusion_count": coverage.get("bf16_exclusion_count"),
        "total_linear_count": coverage.get("total_linear_count"),
        "pipeline_quant_algo": coverage.get("pipeline_quant_algo"),
        "failures": failures,
    }


def _coverage_report(
    coverage: dict[str, Any] | None,
    *,
    require_coverage: bool,
) -> dict[str, Any]:
    if coverage is not None:
        return _evaluate_loader_coverage(coverage)
    if not require_coverage:
        return {
            "status": "skipped",
            "target_count": None,
            "static_mxfp8_target_count": None,
            "bf16_exclusion_count": None,
            "total_linear_count": None,
            "pipeline_quant_algo": None,
            "failures": [],
        }
    return {
        "status": "missing",
        "target_count": None,
        "static_mxfp8_target_count": None,
        "bf16_exclusion_count": None,
        "total_linear_count": None,
        "pipeline_quant_algo": None,
        "failures": ["loader coverage JSON is required for a QAT quality gate"],
    }


def _metric_to_float(value: Any, *, field: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("inf", "+inf", "infinity", "+infinity"):
            return math.inf
        if normalized in ("-inf", "-infinity"):
            return -math.inf
        try:
            return float(normalized)
        except ValueError as exc:
            raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    raise ValueError(f"{field} must be numeric, got {type(value).__name__}")


def _json_metric(value: float) -> float | str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "nan"
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, help="Quality metrics JSON path.")
    parser.add_argument("--output-json", required=True, help="Gate result JSON path.")
    parser.add_argument("--coverage-json", help="Task10 loader coverage JSON path.")
    parser.add_argument(
        "--skip-coverage-check",
        action="store_true",
        help="Explicitly skip loader coverage enforcement for diagnostics only.",
    )
    parser.add_argument(
        "--psnr-threshold",
        type=float,
        default=DEFAULT_PSNR_THRESHOLD,
        help="Per-sample QAT PSNR threshold versus BF16.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coverage = load_json(args.coverage_json) if args.coverage_json else None
    report = evaluate_quality_gate(
        load_json(args.metrics),
        coverage=coverage,
        require_coverage=not args.skip_coverage_check,
        psnr_threshold=args.psnr_threshold,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
