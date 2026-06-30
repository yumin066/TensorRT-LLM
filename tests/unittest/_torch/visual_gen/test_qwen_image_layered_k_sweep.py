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

from scripts.visualgen_eval.qwen_image_layered_mxfp8_k_sweep import (
    NON_BACKBONE_IGNORE,
    build_coverage,
    build_edge_ignore_list,
    materialize_configs,
    read_yaml,
    write_manifest,
    write_provenance,
    write_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
TEMPLATE = REPO_ROOT / (
    "examples/visual_gen/configs/qwen-image-layered-fp8-blockscale-edge-bf16-sage-fp8-1gpu.yaml"
)


def test_edge_ignore_list_matches_edge_blocks():
    assert build_edge_ignore_list(0) == list(NON_BACKBONE_IGNORE)

    ignore = build_edge_ignore_list(4)
    assert ignore[:4] == list(NON_BACKBONE_IGNORE)
    assert ignore[4:] == [
        "transformer_blocks.0",
        "transformer_blocks.1",
        "transformer_blocks.2",
        "transformer_blocks.3",
        "transformer_blocks.56",
        "transformer_blocks.57",
        "transformer_blocks.58",
        "transformer_blocks.59",
    ]

    with pytest.raises(ValueError, match="overlaps"):
        build_edge_ignore_list(31)


def test_materialize_configs_preserves_attention_and_updates_k(tmp_path):
    output_dir = tmp_path / "configs"
    paths = materialize_configs(
        template_path=TEMPLATE,
        output_dir=output_dir,
        k_values=[12, 8, 4, 0],
    )

    template = read_yaml(TEMPLATE)
    bf16 = read_yaml(paths["bf16"])
    k8 = read_yaml(paths["k8"])
    k0 = read_yaml(paths["k0"])

    assert "quant_config" not in bf16
    assert bf16["attention_config"] == template["attention_config"]
    assert k8["attention_config"] == template["attention_config"]
    assert k8["quant_config"]["quant_algo"] == "FP8_BLOCK_SCALES"
    assert k8["quant_config"]["dynamic"] is True
    assert "transformer_blocks.7" in k8["quant_config"]["ignore"]
    assert "transformer_blocks.8" not in k8["quant_config"]["ignore"]
    assert "transformer_blocks.51" not in k8["quant_config"]["ignore"]
    assert "transformer_blocks.52" in k8["quant_config"]["ignore"]
    assert k0["quant_config"]["ignore"] == list(NON_BACKBONE_IGNORE)


def test_build_coverage_records_expected_linear_sets(tmp_path):
    config_path = tmp_path / "k4.yaml"
    config = {
        "quant_config": {
            "quant_algo": "FP8_BLOCK_SCALES",
            "dynamic": True,
            "ignore": build_edge_ignore_list(4),
        },
        "attention_config": {"backend": "TRTLLM"},
    }
    write_yaml(config_path, config)
    linear_names = [
        "img_in",
        "txt_in",
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.4.attn.to_q",
        "transformer_blocks.55.ff.net.0.proj",
        "transformer_blocks.56.ff.net.0.proj",
        "proj_out",
    ]

    coverage = build_coverage(
        variant="k4",
        config_path=config_path,
        config=config,
        linear_names=linear_names,
        git_sha="abc123",
    )

    by_name = {entry["name"]: entry for entry in coverage["linear_modules"]}
    assert coverage["counts"]["linear_total"] == 7
    assert coverage["counts"]["mxfp8_linear"] == 2
    assert coverage["counts"]["bf16_preserved_linear"] == 5
    assert by_name["transformer_blocks.4.attn.to_q"]["expected"] == "mxfp8"
    assert by_name["transformer_blocks.55.ff.net.0.proj"]["expected"] == "mxfp8"
    assert by_name["transformer_blocks.56.ff.net.0.proj"]["preserved_reason"] == (
        "transformer_blocks.56"
    )
    assert coverage["by_block"][4]["mxfp8_linear"] == 1
    assert coverage["by_block"][56]["bf16_preserved_linear"] == 1


def test_manifest_and_provenance_outputs_are_reproducible(tmp_path):
    config_dir = tmp_path / "configs"
    config_paths = materialize_configs(
        template_path=TEMPLATE,
        output_dir=config_dir,
        k_values=[12, 8, 4, 0],
    )
    samples = [
        {
            "id": "sample_a",
            "prompt": "split this graphic into layers",
            "image": "inputs/sample.png",
            "seed": 123,
            "num_inference_steps": 4,
            "resolution": 640,
            "layers": 2,
        }
    ]
    samples_path = tmp_path / "samples.json"
    samples_path.write_text(json.dumps(samples), encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    manifest = write_manifest(
        samples_path=samples_path,
        config_paths=config_paths,
        output_json=manifest_path,
        output_root=tmp_path / "outputs",
        model="/models/qwen-image-layered",
        artifact_root=tmp_path / "artifacts",
        checkpoint_path="/models/qwen-image-layered",
    )
    variants = {variant["name"]: variant for variant in manifest["variants"]}
    assert set(variants) == {"bf16", "k12", "k8", "k4", "k0"}
    assert variants["k0"]["metadata"]["k_edge_bf16"] == 0
    assert variants["k12"]["artifact_paths"]["sample_a"].endswith("artifacts/k12/sample_a.pt")

    provenance_path = tmp_path / "run.json"
    provenance = write_provenance(
        output_json=provenance_path,
        run_root=tmp_path / "run",
        checkout="/remote/TensorRT-LLM",
        allocation="sc-2886562",
        node="viking-dvt-151",
        docker_image="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20",
        manifest_path=manifest_path,
        metrics_path=tmp_path / "metrics.json",
        project_root=Path.cwd(),
    )
    assert provenance["allocation"] == "sc-2886562"
    assert provenance["node"] == "viking-dvt-151"
    assert "ssh-gw exec sc-2886562 --no-enroot" in provenance["quality_command"]
    assert "docker run --rm --gpus all" in provenance["quality_command"]
