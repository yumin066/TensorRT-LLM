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

import pytest
import torch

from scripts.visualgen_eval.qwen_image_layered_quality import (
    LayeredManifest,
    LayeredSample,
    capture_transformer_tuples,
    compute_layered_metrics,
    evaluate_manifest,
    load_manifest,
    normalize_layer_stack,
)


def _write_pt(path, video: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"video": video}, path)


def _write_safetensors(path, video: torch.Tensor) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"video": video.contiguous()}, str(path))


def _manifest(tmp_path, *, candidate_path, reference_path, layers=2) -> dict:
    return {
        "output_root": "out",
        "samples": [
            {
                "id": "sample_a",
                "prompt": "split this graphic into layers",
                "image": "inputs/sample.png",
                "seed": 123,
                "num_inference_steps": 4,
                "resolution": 640,
                "layers": layers,
                "height": 4,
                "width": 4,
                "negative_prompt": "blur",
                "guidance_scale": 4.0,
                "layer_metadata": {"alpha_policy": "premultiplied"},
            }
        ],
        "variants": [
            {
                "name": "bf16",
                "model": "qwen-image-layered",
                "visual_gen_args": "bf16.yaml",
                "artifact_paths": {"sample_a": str(reference_path)},
            },
            {
                "name": "all_layer_dynamic_mxfp8",
                "model": "qwen-image-layered",
                "visual_gen_args": "mxfp8.yaml",
                "artifact_paths": {"sample_a": str(candidate_path)},
                "metadata": {"preset": "all"},
            },
        ],
    }


def test_manifest_validation_requires_bf16_reference(tmp_path):
    path = tmp_path / "manifest.json"
    data = _manifest(
        tmp_path,
        candidate_path=tmp_path / "candidate.pt",
        reference_path=tmp_path / "reference.pt",
    )
    data["variants"][0]["name"] = "baseline"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="bf16"):
        load_manifest(path)


def test_sample_requires_schedule_and_matching_height_width():
    with pytest.raises(ValueError, match="num_inference_steps or sigmas"):
        LayeredSample(
            id="x",
            prompt="prompt",
            image="image.png",
            seed=1,
            resolution=640,
            layers=2,
        )

    with pytest.raises(ValueError, match="height and width"):
        LayeredSample(
            id="x",
            prompt="prompt",
            image="image.png",
            seed=1,
            resolution=640,
            layers=2,
            sigmas=[1.0],
            height=640,
        )


def test_evaluate_existing_pt_and_safetensors_artifacts(tmp_path):
    reference = torch.zeros(2, 4, 4, 4, dtype=torch.uint8)
    candidate = reference.clone()
    candidate[0, 0, 0, 0] = 1
    reference_path = tmp_path / "reference.pt"
    candidate_path = tmp_path / "candidate.safetensors"
    _write_pt(reference_path, reference)
    _write_safetensors(candidate_path, candidate)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _manifest(
                tmp_path,
                candidate_path=candidate_path,
                reference_path=reference_path,
            )
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path)
    output_json = tmp_path / "metrics.json"
    metrics = evaluate_manifest(
        manifest,
        manifest_path=manifest_path,
        output_json=output_json,
        artifact_format="pt",
        load_existing=True,
        overwrite=False,
        capture_transformer_tuples_enabled=False,
        save_audit_pngs=False,
    )

    expected = compute_layered_metrics(
        normalize_layer_stack(candidate),
        normalize_layer_stack(reference),
        candidate_path=candidate_path,
        reference_path=reference_path,
    )
    comparison = metrics["comparisons"][0]
    aggregate = metrics["aggregates"]["all_layer_dynamic_mxfp8"]
    assert comparison["sample_id"] == "sample_a"
    assert comparison["psnr"] == pytest.approx(expected["psnr"])
    assert comparison["ssim"] == pytest.approx(expected["ssim"])
    assert aggregate["psnr_min"] == pytest.approx(expected["psnr"])
    assert aggregate["psnr_mean"] == pytest.approx(expected["psnr"])

    persisted = json.loads(output_json.read_text(encoding="utf-8"))
    assert persisted["variants"]["bf16"]["artifacts"]["sample_a"] == str(reference_path)
    assert persisted["variants"]["all_layer_dynamic_mxfp8"]["metadata"] == {"preset": "all"}


def test_layer_count_mismatch_fails_before_metrics(tmp_path):
    reference_path = tmp_path / "reference.pt"
    candidate_path = tmp_path / "candidate.pt"
    _write_pt(reference_path, torch.zeros(2, 4, 4, 4, dtype=torch.uint8))
    _write_pt(candidate_path, torch.zeros(3, 4, 4, 4, dtype=torch.uint8))
    manifest = LayeredManifest.model_validate(
        _manifest(
            tmp_path,
            candidate_path=candidate_path,
            reference_path=reference_path,
        )
    )

    with pytest.raises(ValueError, match="expected 2"):
        evaluate_manifest(
            manifest,
            manifest_path=tmp_path / "manifest.json",
            output_json=tmp_path / "metrics.json",
            artifact_format="pt",
            load_existing=True,
            overwrite=False,
            capture_transformer_tuples_enabled=False,
            save_audit_pngs=False,
        )


def test_capture_transformer_tuples_saves_roles_and_restores_forward(tmp_path):
    class DummyTransformer(torch.nn.Module):
        def forward(self, hidden_states, *, timestep=None, return_dict=False):
            del timestep, return_dict
            return (hidden_states + 1.0,)

    class DummyPipeline:
        transformer = DummyTransformer()

    pipeline = DummyPipeline()
    original_forward = pipeline.transformer.forward
    with capture_transformer_tuples(
        pipeline,
        output_dir=tmp_path,
        sample_id="sample_a",
        variant_name="bf16",
        has_negative_prompt=True,
    ):
        pipeline.transformer(torch.zeros(1, 2), timestep=torch.ones(1), return_dict=False)
        pipeline.transformer(torch.ones(1, 2), timestep=torch.ones(1), return_dict=False)

    assert pipeline.transformer.forward == original_forward
    saved = sorted(tmp_path.glob("tuple_*.pt"))
    assert [path.name for path in saved] == [
        "tuple_0000_cond.pt",
        "tuple_0001_negative.pt",
    ]
    first = torch.load(saved[0], map_location="cpu", weights_only=True)
    second = torch.load(saved[1], map_location="cpu", weights_only=True)
    assert first["role"] == "cond"
    assert second["role"] == "negative"
    assert torch.equal(first["target_output"], torch.ones(1, 2))
