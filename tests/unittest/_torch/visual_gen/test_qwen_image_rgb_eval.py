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

from pathlib import Path

import pytest

from scripts.visualgen_eval.qwen_image_rgb_eval import (
    compute_rgb_values_metrics,
    run_qwen_image_rgb_split_eval,
)


def _record(prompt_id: str = "qwen_image_fast_0000") -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "split": "fast_calibration",
        "prompt": "a ceramic mug on a desk",
        "negative_prompt": "",
        "seed": 1234,
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "guidance_scale": 4.0,
        "max_sequence_length": 512,
    }


def _write_placeholder_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder png")


def test_compute_rgb_values_metrics_exact_match() -> None:
    values = [10.0 / 255.0, 20.0 / 255.0, 30.0 / 255.0] * 4
    metrics = compute_rgb_values_metrics(values, values)

    assert metrics["mse"] == 0.0
    assert metrics["psnr"] == float("inf")
    assert metrics["ssim"] == pytest.approx(1.0)


def test_run_qwen_image_rgb_split_eval_with_fake_pipeline(tmp_path: Path) -> None:
    records = [_record()]
    reference_root = tmp_path / "references" / "bf16_sage_fp8"
    output_root = tmp_path / "outputs" / "baseline"
    metrics_json = tmp_path / "metrics.json"
    _write_placeholder_png(reference_root / "fast_calibration" / "qwen_image_fast_0000.png")

    result = run_qwen_image_rgb_split_eval(
        records=records,
        split="fast_calibration",
        model="Qwen/Qwen-Image",
        visual_gen_args=tmp_path / "qwen-image.yaml",
        reference_root=reference_root,
        output_root=output_root,
        metrics_json=metrics_json,
        variant="original_all_layer_dynamic_mxfp8",
        reference_variant="bf16_sage_fp8",
        device="cuda:0",
        provenance={"git_head": "test-git-head"},
        config_metadata={"pipeline_quant_algo": "FP8_BLOCK_SCALES"},
        pipeline=object(),
        infer_fn=lambda _pipeline, record: record,
        save_fn=lambda _output, path: _write_placeholder_png(path),
        metrics_fn=lambda _candidate, _reference: {
            "mse": 1.0 / 3.0,
            "psnr": 4.771212547196624,
            "ssim": 0.8,
        },
    )

    comparison = result["comparisons"][0]
    aggregate = result["aggregates"]
    assert result["format"] == "qwen_image_rgb_split_metrics_v1"
    assert result["prompt_count"] == 1
    assert comparison["prompt_id"] == "qwen_image_fast_0000"
    assert comparison["psnr"] == pytest.approx(4.771212547196624)
    assert aggregate["psnr_mean"] == pytest.approx(comparison["psnr"])
    assert Path(comparison["candidate_image"]).is_file()
    assert metrics_json.is_file()


def test_run_qwen_image_rgb_split_eval_rejects_missing_reference(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing BF16 reference image"):
        run_qwen_image_rgb_split_eval(
            records=[_record()],
            split="fast_calibration",
            model="Qwen/Qwen-Image",
            visual_gen_args=tmp_path / "qwen-image.yaml",
            reference_root=tmp_path / "missing_references",
            output_root=tmp_path / "outputs",
            metrics_json=tmp_path / "metrics.json",
            variant="original_all_layer_dynamic_mxfp8",
            reference_variant="bf16_sage_fp8",
            device="cuda:0",
            provenance={},
            pipeline=object(),
            infer_fn=lambda _pipeline, record: record,
            save_fn=lambda _output, path: _write_placeholder_png(path),
        )
