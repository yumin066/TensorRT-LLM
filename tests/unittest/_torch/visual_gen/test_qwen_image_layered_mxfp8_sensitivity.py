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

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.visualgen_eval.qwen_image_layered_mxfp8_sensitivity import (
    generate_sensitivity_summary,
    recipe_to_variant,
    write_markdown_summary,
)

LINEAR_NAMES = [
    "img_in",
    "transformer_blocks.0.attn.to_q",
    "transformer_blocks.8.attn.to_q",
    "transformer_blocks.12.attn.to_q",
    "transformer_blocks.47.attn.to_q",
    "transformer_blocks.51.attn.to_q",
    "transformer_blocks.56.attn.to_q",
    "proj_out",
]


def _observed_state(variant: str, name: str) -> str:
    if not name.startswith("transformer_blocks."):
        return "bf16"
    block = int(name.split(".")[1])
    if variant == "bf16":
        return "bf16"
    k_edge_bf16 = int(variant[1:])
    return "mxfp8" if k_edge_bf16 <= block < 60 - k_edge_bf16 else "bf16"


def _module(name: str, state: str) -> dict:
    if state == "mxfp8":
        return {
            "name": name,
            "quant_algo": "FP8_BLOCK_SCALES",
            "quant_method": "FP8BlockScalesLinearMethod",
            "weight_dtype": "torch.float8_e4m3fn",
            "weight_scale_dtype": "torch.float32",
            "weight_scale_shape": [1, 1],
        }
    return {
        "name": name,
        "quant_algo": "None",
        "quant_method": "UnquantizedLinearMethod",
        "weight_dtype": "torch.bfloat16",
        "weight_scale_dtype": None,
        "weight_scale_shape": None,
    }


def _write_manifest(path: Path, variant: str, *, force_bf16_name: str | None = None) -> None:
    rows = []
    for name in LINEAR_NAMES:
        state = "bf16" if name == force_bf16_name else _observed_state(variant, name)
        rows.append(_module(name, state))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def _write_summary(tmp_path: Path, *, omit_variant: str | None = None) -> Path:
    recipes = {
        "bf16": "bf16",
        "k12": "fp8_block_scales_edge_bf16_12",
        "k8": "fp8_block_scales_edge_bf16_8",
        "k4": "fp8_block_scales_edge_bf16_4",
        "k0": "fp8_block_scales_edge_bf16_0",
    }
    metrics = {
        "bf16": float("inf"),
        "k12": 39.0,
        "k8": 38.0,
        "k4": 35.0,
        "k0": 24.0,
    }
    rows = []
    for variant, recipe in recipes.items():
        if variant == omit_variant:
            continue
        manifest_path = tmp_path / variant / "linear_manifest.json"
        _write_manifest(manifest_path, variant)
        rows.append(
            {
                "model": "qwen_image_layered",
                "recipe": recipe,
                "status": "ok",
                "steps": 50,
                "resolution": 1024,
                "seed": 44,
                "psnr_vs_bf16": metrics[variant],
                "ssim_vs_bf16": 1.0 if variant == "bf16" else 0.9,
                "composite_psnr_vs_bf16": metrics[variant] + 1.0,
                "composite_ssim_vs_bf16": 1.0 if variant == "bf16" else 0.91,
                "linear_manifest_path": str(manifest_path),
                "linear_quant_counts": {},
            }
        )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(rows), encoding="utf-8")
    return summary_path


def test_recipe_to_variant_supports_runner_recipe_names():
    assert recipe_to_variant("bf16") == "bf16"
    assert recipe_to_variant("k12") == "k12"
    assert recipe_to_variant("fp8_block_scales_edge_bf16_8") == "k8"


def test_generate_sensitivity_summary_records_observed_coverage_and_recommendation(tmp_path):
    summary_path = _write_summary(tmp_path)

    summary = generate_sensitivity_summary(
        summary_json=summary_path,
        model="qwen_image_layered",
    )

    assert summary["required_variants"] == ["bf16", "k12", "k8", "k4", "k0"]
    assert summary["observed_coverage"]["k0"]["counts"]["observed_mxfp8_transformer_linear"] == 6
    assert summary["observed_coverage"]["k0"]["counts"]["observed_bf16_linear"] == 2
    assert summary["observed_coverage"]["k0"]["coverage_errors"] == []
    assert summary["sensitivity"][0]["from_variant"] == "k12"
    assert summary["sensitivity"][0]["to_variant"] == "k8"
    assert summary["sensitivity"][0]["added_mxfp8_blocks"] == [
        8,
        9,
        10,
        11,
        48,
        49,
        50,
        51,
    ]
    assert summary["recommendation"]["target_layer_set"] == ("all transformer block Linear modules")
    assert summary["recommendation"]["target_observed_mxfp8_transformer_linear"] == 6

    markdown_path = tmp_path / "sensitivity.md"
    write_markdown_summary(markdown_path, summary)
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Qwen-Image-Layered MXFP8 Sensitivity Summary" in markdown
    assert "| k0 | fp8_block_scales_edge_bf16_0 | 24.0000" in markdown


def test_missing_required_k0_fails(tmp_path):
    summary_path = _write_summary(tmp_path, omit_variant="k0")

    with pytest.raises(ValueError, match="Missing required K-sweep variants"):
        generate_sensitivity_summary(
            summary_json=summary_path,
            model="qwen_image_layered",
        )


def test_expected_observed_transformer_mismatch_fails(tmp_path):
    summary_path = _write_summary(tmp_path)
    k0_manifest = tmp_path / "k0" / "linear_manifest.json"
    _write_manifest(
        k0_manifest,
        "k0",
        force_bf16_name="transformer_blocks.0.attn.to_q",
    )

    with pytest.raises(ValueError, match="Observed Linear coverage validation failed"):
        generate_sensitivity_summary(
            summary_json=summary_path,
            model="qwen_image_layered",
        )
