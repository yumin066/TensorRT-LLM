#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Summarize Qwen-Image-Layered MXFP8 K-sweep sensitivity artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.visualgen_eval.qwen_image_layered_mxfp8_k_sweep import (  # noqa: E402
    DEFAULT_TOTAL_BLOCKS,
    NON_BACKBONE_IGNORE,
    QUANT_ALGO,
    REFERENCE_VARIANT,
    block_index,
    build_edge_ignore_list,
    module_role,
    preserved_reason,
    write_json,
)

RECIPE_RE = re.compile(r"^fp8_block_scales_edge_bf16_(\d+)$")
DEFAULT_REQUIRED_K = (12, 8, 4, 0)
TRANSITION_ORDER = ("k12", "k8", "k4", "k0")


def recipe_to_variant(recipe: str) -> str:
    if recipe == REFERENCE_VARIANT:
        return REFERENCE_VARIANT
    if recipe.startswith("k") and recipe[1:].isdigit():
        return recipe
    match = RECIPE_RE.match(recipe)
    if match is not None:
        return f"k{match.group(1)}"
    return recipe


def _variant_k(variant: str) -> int:
    if not variant.startswith("k") or not variant[1:].isdigit():
        raise ValueError(f"Expected K variant name, got {variant!r}")
    return int(variant[1:])


def _metric_to_float(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "inf":
            return float("inf")
        if lowered == "-inf":
            return float("-inf")
        if lowered == "nan":
            return float("nan")
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Metric {field!r} is not numeric: {value!r}") from exc
    raise ValueError(f"Metric {field!r} has unsupported type: {type(value).__name__}")


def _jsonify(value: Any) -> Any:
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
        return value
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    return value


def _format_metric(value: Any) -> str:
    value = _metric_to_float(value, field="markdown")
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_summary_rows(summary_json: Path) -> list[dict[str, Any]]:
    payload = read_json(summary_json)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
        rows = payload["results"]
    else:
        raise ValueError(f"Expected list or {{'results': [...]}} summary in {summary_json}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"All summary rows must be objects: {summary_json}")
    return rows


def select_required_rows(
    rows: list[dict[str, Any]],
    *,
    model: str,
    required_k: tuple[int, ...] = DEFAULT_REQUIRED_K,
) -> dict[str, dict[str, Any]]:
    required_variants = [REFERENCE_VARIANT, *[f"k{k_value}" for k_value in required_k]]
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("model") != model:
            continue
        recipe = row.get("recipe")
        if not isinstance(recipe, str):
            raise ValueError(f"Summary row for model {model!r} lacks string recipe: {row}")
        variant = recipe_to_variant(recipe)
        if variant not in required_variants:
            continue
        if variant in selected:
            raise ValueError(f"Duplicate summary rows for variant {variant!r}")
        if row.get("status") != "ok":
            raise ValueError(f"Variant {variant!r} status is not ok: {row.get('error')}")
        selected[variant] = row

    missing = [variant for variant in required_variants if variant not in selected]
    if missing:
        raise ValueError(f"Missing required K-sweep variants: {missing}")
    return selected


def _resolve_manifest_path(raw_path: Any, *, summary_dir: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Summary row must include non-empty linear_manifest_path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = summary_dir / path
    if not path.exists():
        raise FileNotFoundError(f"Linear manifest does not exist: {path}")
    return path


def load_linear_manifest(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, dict):
        for key in ("linear_modules", "modules", "linears"):
            modules = payload.get(key)
            if isinstance(modules, list):
                payload = modules
                break
    if not isinstance(payload, list):
        raise ValueError(f"Linear manifest must be a list or contain a module list: {path}")
    if not all(isinstance(module, dict) for module in payload):
        raise ValueError(f"Linear manifest entries must be objects: {path}")
    return payload


def expected_linear_state(variant: str, name: str, *, total_blocks: int) -> tuple[str, str | None]:
    if variant == REFERENCE_VARIANT:
        return "bf16", "bf16_reference"
    k_edge_bf16 = _variant_k(variant)
    ignore = build_edge_ignore_list(k_edge_bf16, total_blocks=total_blocks)
    reason = preserved_reason(name, ignore)
    return ("bf16", reason) if reason is not None else ("mxfp8", None)


def observed_linear_state(module: dict[str, Any]) -> str:
    quant_algo = str(module.get("quant_algo"))
    quant_method = str(module.get("quant_method"))
    weight_dtype = str(module.get("weight_dtype"))
    if quant_algo == QUANT_ALGO:
        return "mxfp8"
    if "FP8BlockScales" in quant_method:
        return "mxfp8"
    if "float8" in weight_dtype:
        return "mxfp8"
    return "bf16"


def _empty_block_counts(total_blocks: int) -> list[dict[str, Any]]:
    return [
        {
            "block_index": idx,
            "expected_mxfp8_linear": 0,
            "expected_bf16_linear": 0,
            "observed_mxfp8_linear": 0,
            "observed_bf16_linear": 0,
        }
        for idx in range(total_blocks)
    ]


def build_observed_coverage(
    *,
    variant: str,
    row: dict[str, Any],
    summary_dir: Path,
    total_blocks: int = DEFAULT_TOTAL_BLOCKS,
) -> dict[str, Any]:
    manifest_path = _resolve_manifest_path(row.get("linear_manifest_path"), summary_dir=summary_dir)
    modules = load_linear_manifest(manifest_path)
    by_block = _empty_block_counts(total_blocks)
    counts = {
        "linear_total": 0,
        "expected_mxfp8_linear": 0,
        "expected_bf16_linear": 0,
        "observed_mxfp8_linear": 0,
        "observed_bf16_linear": 0,
        "transformer_linear": 0,
        "observed_mxfp8_transformer_linear": 0,
        "observed_bf16_transformer_linear": 0,
        "non_backbone_linear": 0,
        "other_linear": 0,
        "transformer_mismatch": 0,
    }
    coverage_errors = []
    coverage_modules = []

    for module in modules:
        name = module.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Linear manifest entry lacks non-empty name: {module}")

        idx = block_index(name)
        role = module_role(name)
        expected, reason = expected_linear_state(variant, name, total_blocks=total_blocks)
        observed = observed_linear_state(module)
        mismatch = expected != observed
        scale_shape = module.get("weight_scale_shape")
        scale_dtype = module.get("weight_scale_dtype")

        counts["linear_total"] += 1
        counts[f"expected_{expected}_linear"] += 1
        counts[f"observed_{observed}_linear"] += 1
        if role == "transformer_block":
            counts["transformer_linear"] += 1
            counts[f"observed_{observed}_transformer_linear"] += 1
        elif role == "non_backbone":
            counts["non_backbone_linear"] += 1
        else:
            counts["other_linear"] += 1

        if idx is not None and 0 <= idx < total_blocks:
            by_block[idx][f"expected_{expected}_linear"] += 1
            by_block[idx][f"observed_{observed}_linear"] += 1

        if role == "transformer_block" and mismatch:
            counts["transformer_mismatch"] += 1
            coverage_errors.append(
                {
                    "name": name,
                    "block_index": idx,
                    "expected": expected,
                    "observed": observed,
                    "reason": "expected_observed_mismatch",
                }
            )
        if role == "transformer_block" and expected == "mxfp8" and observed == "mxfp8":
            if scale_dtype in (None, "None") or scale_shape in (None, []):
                coverage_errors.append(
                    {
                        "name": name,
                        "block_index": idx,
                        "expected": expected,
                        "observed": observed,
                        "reason": "missing_mxfp8_weight_scale",
                    }
                )
        if variant == "k0" and role == "transformer_block" and observed != "mxfp8":
            coverage_errors.append(
                {
                    "name": name,
                    "block_index": idx,
                    "expected": "mxfp8",
                    "observed": observed,
                    "reason": "k0_transformer_linear_not_mxfp8",
                }
            )

        coverage_modules.append(
            {
                "name": name,
                "class": module.get("class", module.get("module_class", "Linear")),
                "block_index": idx,
                "role": role,
                "expected": expected,
                "observed": observed,
                "preserved_reason": reason,
                "observed_quant_method": module.get("quant_method"),
                "observed_quant_algo": module.get("quant_algo"),
                "observed_weight_dtype": module.get("weight_dtype"),
                "observed_weight_scale_dtype": scale_dtype,
                "observed_weight_scale_shape": scale_shape,
                "mismatch": mismatch,
            }
        )

    return {
        "schema_version": 1,
        "variant": variant,
        "recipe": row.get("recipe"),
        "linear_manifest_path": str(manifest_path),
        "counts": counts,
        "coverage_errors": coverage_errors,
        "linear_modules": coverage_modules,
        "by_block": by_block,
    }


def _variant_metrics(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "psnr_vs_bf16",
        "ssim_vs_bf16",
        "composite_psnr_vs_bf16",
        "composite_ssim_vs_bf16",
        "rmse_vs_bf16",
        "mae_vs_bf16",
    )
    return {field: _metric_to_float(row.get(field), field=field) for field in fields}


def summarize_variants(
    selected_rows: dict[str, dict[str, Any]],
    coverages: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    variants = {}
    for variant, row in selected_rows.items():
        variants[variant] = {
            "recipe": row.get("recipe"),
            "status": row.get("status"),
            "steps": row.get("steps"),
            "resolution": row.get("resolution"),
            "height": row.get("height"),
            "width": row.get("width"),
            "seed": row.get("seed"),
            "metrics": _variant_metrics(row),
            "linear_quant_counts": row.get("linear_quant_counts") or {},
            "observed_coverage_counts": coverages[variant]["counts"],
            "linear_manifest_path": coverages[variant]["linear_manifest_path"],
        }
    return variants


def _quantized_blocks(variant: str, *, total_blocks: int) -> set[int]:
    k_edge_bf16 = _variant_k(variant)
    return set(range(k_edge_bf16, total_blocks - k_edge_bf16))


def build_sensitivity(
    variants: dict[str, dict[str, Any]],
    *,
    total_blocks: int,
) -> list[dict[str, Any]]:
    results = []
    for less_quantized, more_quantized in zip(TRANSITION_ORDER, TRANSITION_ORDER[1:]):
        if less_quantized not in variants or more_quantized not in variants:
            continue
        added_blocks = sorted(
            _quantized_blocks(more_quantized, total_blocks=total_blocks)
            - _quantized_blocks(less_quantized, total_blocks=total_blocks)
        )
        from_metrics = variants[less_quantized]["metrics"]
        to_metrics = variants[more_quantized]["metrics"]
        psnr_delta = None
        ssim_delta = None
        composite_psnr_delta = None
        if from_metrics["psnr_vs_bf16"] is not None and to_metrics["psnr_vs_bf16"] is not None:
            psnr_delta = to_metrics["psnr_vs_bf16"] - from_metrics["psnr_vs_bf16"]
        if from_metrics["ssim_vs_bf16"] is not None and to_metrics["ssim_vs_bf16"] is not None:
            ssim_delta = to_metrics["ssim_vs_bf16"] - from_metrics["ssim_vs_bf16"]
        if (
            from_metrics["composite_psnr_vs_bf16"] is not None
            and to_metrics["composite_psnr_vs_bf16"] is not None
        ):
            composite_psnr_delta = (
                to_metrics["composite_psnr_vs_bf16"] - from_metrics["composite_psnr_vs_bf16"]
            )
        results.append(
            {
                "from_variant": less_quantized,
                "to_variant": more_quantized,
                "added_mxfp8_blocks": added_blocks,
                "psnr_delta": psnr_delta,
                "ssim_delta": ssim_delta,
                "composite_psnr_delta": composite_psnr_delta,
                "interpretation": (
                    "negative delta means the newly quantized block group is sensitive"
                ),
            }
        )
    return results


def build_recommendation(
    *,
    coverages: dict[str, dict[str, Any]],
    sensitivity: list[dict[str, Any]],
) -> dict[str, Any]:
    k0_counts = coverages["k0"]["counts"]
    ranked_groups = sorted(
        sensitivity,
        key=lambda item: (
            float("inf") if item["psnr_delta"] is None else float(item["psnr_delta"]),
            item["to_variant"],
        ),
    )
    priority_groups = [
        {
            "blocks": group["added_mxfp8_blocks"],
            "transition": f"{group['from_variant']}->{group['to_variant']}",
            "psnr_delta": group["psnr_delta"],
            "reason": "largest PSNR drop after adding this MXFP8 block group",
        }
        for group in ranked_groups
    ]
    return {
        "target_layer_set": "all transformer block Linear modules",
        "target_variant": "k0",
        "target_observed_mxfp8_transformer_linear": k0_counts["observed_mxfp8_transformer_linear"],
        "non_backbone_bf16_exclusions": list(NON_BACKBONE_IGNORE),
        "qat_priority_groups": priority_groups,
        "summary": (
            "Use QAT on all transformer-block Linear layers so K=0 can keep the MXFP8 "
            "runtime path; use the ranked edge groups as the first LoRA or partial-unfreeze "
            "focus if training capacity is limited."
        ),
    }


def validate_coverage_or_raise(coverages: dict[str, dict[str, Any]]) -> None:
    errors = []
    for variant, coverage in coverages.items():
        for error in coverage["coverage_errors"]:
            errors.append({"variant": variant, **error})
    if errors:
        preview = errors[:5]
        raise ValueError(f"Observed Linear coverage validation failed: {preview}")


def generate_sensitivity_summary(
    *,
    summary_json: Path,
    model: str,
    required_k: tuple[int, ...] = DEFAULT_REQUIRED_K,
    total_blocks: int = DEFAULT_TOTAL_BLOCKS,
    provenance_json: Path | None = None,
    strict_coverage: bool = True,
) -> dict[str, Any]:
    rows = load_summary_rows(summary_json)
    selected_rows = select_required_rows(rows, model=model, required_k=required_k)
    summary_dir = summary_json.resolve().parent
    coverages = {
        variant: build_observed_coverage(
            variant=variant,
            row=row,
            summary_dir=summary_dir,
            total_blocks=total_blocks,
        )
        for variant, row in selected_rows.items()
    }
    if strict_coverage:
        validate_coverage_or_raise(coverages)
    variants = summarize_variants(selected_rows, coverages)
    sensitivity = build_sensitivity(variants, total_blocks=total_blocks)
    recommendation = build_recommendation(coverages=coverages, sensitivity=sensitivity)
    provenance = read_json(provenance_json) if provenance_json is not None else None
    return _jsonify(
        {
            "schema_version": 1,
            "summary_json": str(summary_json),
            "model": model,
            "required_variants": [REFERENCE_VARIANT, *[f"k{k_value}" for k_value in required_k]],
            "provenance": provenance,
            "variants": variants,
            "observed_coverage": coverages,
            "sensitivity": sensitivity,
            "recommendation": recommendation,
        }
    )


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    variants = summary["variants"]
    coverage = summary["observed_coverage"]
    lines = [
        "<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->",
        "",
        "# Qwen-Image-Layered MXFP8 Sensitivity Summary",
        "",
        "## Variant Metrics",
        "",
        "| Variant | Recipe | PSNR | SSIM | Composite PSNR | BF16 Linear | MXFP8 Linear | Errors |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in summary["required_variants"]:
        item = variants[variant]
        counts = coverage[variant]["counts"]
        lines.append(
            "| {variant} | {recipe} | {psnr} | {ssim} | {composite_psnr} | {bf16} | "
            "{mxfp8} | {errors} |".format(
                variant=variant,
                recipe=item["recipe"],
                psnr=_format_metric(item["metrics"]["psnr_vs_bf16"]),
                ssim=_format_metric(item["metrics"]["ssim_vs_bf16"]),
                composite_psnr=_format_metric(item["metrics"]["composite_psnr_vs_bf16"]),
                bf16=counts["observed_bf16_linear"],
                mxfp8=counts["observed_mxfp8_linear"],
                errors=len(coverage[variant]["coverage_errors"]),
            )
        )

    lines.extend(
        [
            "",
            "## Block-Group Sensitivity",
            "",
            "| Transition | Added MXFP8 Blocks | PSNR Delta | SSIM Delta | Composite PSNR Delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for item in summary["sensitivity"]:
        blocks = item["added_mxfp8_blocks"]
        block_text = f"{blocks[0]}-{blocks[-1]}" if blocks else "none"
        if blocks and len(blocks) > 1 and blocks != list(range(blocks[0], blocks[-1] + 1)):
            block_text = ",".join(str(block) for block in blocks)
        lines.append(
            "| {from_variant}->{to_variant} | {blocks} | {psnr} | {ssim} | {composite} |".format(
                from_variant=item["from_variant"],
                to_variant=item["to_variant"],
                blocks=block_text,
                psnr=_format_metric(item["psnr_delta"]),
                ssim=_format_metric(item["ssim_delta"]),
                composite=_format_metric(item["composite_psnr_delta"]),
            )
        )

    recommendation = summary["recommendation"]
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- Target layer set: "
            f"{recommendation['target_layer_set']} ({recommendation['target_variant']}).",
            "- Observed target MXFP8 transformer Linear count: "
            f"{recommendation['target_observed_mxfp8_transformer_linear']}.",
            "- Keep non-backbone exclusions BF16: "
            f"{', '.join(recommendation['non_backbone_bf16_exclusions'])}.",
            f"- Training priority: {recommendation['summary']}",
            "",
            "## QAT Priority Groups",
            "",
        ]
    )
    for index, group in enumerate(recommendation["qat_priority_groups"], start=1):
        blocks = group["blocks"]
        block_text = f"{blocks[0]}-{blocks[-1]}" if blocks else "none"
        lines.append(
            f"{index}. {group['transition']}: blocks {block_text}, "
            f"PSNR delta {_format_metric(group['psnr_delta'])}."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_required_k(values: list[str] | None) -> tuple[int, ...]:
    if values is None:
        return DEFAULT_REQUIRED_K
    parsed = tuple(int(value) for value in values)
    if set(parsed) != set(DEFAULT_REQUIRED_K):
        raise ValueError(f"Formal sensitivity requires K set {DEFAULT_REQUIRED_K}, got {parsed}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--model", default="qwen_image_layered")
    parser.add_argument("--required-k", nargs="*")
    parser.add_argument("--total-blocks", type=int, default=DEFAULT_TOTAL_BLOCKS)
    parser.add_argument("--provenance-json", type=Path)
    parser.add_argument(
        "--allow-coverage-mismatch",
        action="store_true",
        help="Write summaries even when observed transformer Linear coverage mismatches expected.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_sensitivity_summary(
        summary_json=args.summary_json,
        model=args.model,
        required_k=_parse_required_k(args.required_k),
        total_blocks=args.total_blocks,
        provenance_json=args.provenance_json,
        strict_coverage=not args.allow_coverage_mismatch,
    )
    write_json(args.output_json, summary)
    write_markdown_summary(args.output_md, summary)


if __name__ == "__main__":
    main()
