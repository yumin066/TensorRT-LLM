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

from scripts.visualgen_eval.qwen_image_prompt_manifest import (
    DEFAULT_ENROOT_IMAGE,
    DEFAULT_SPLIT_COUNTS,
    REQUIRED_CATEGORIES,
    build_default_prompt_records,
    build_summary,
    read_json,
    read_jsonl,
    validate_prompt_records,
    validate_summary,
    write_json,
    write_jsonl,
)

REMOTE_CHECKOUT = (
    "/lustre/fsw/portfolios/coreai/projects/coreai_devtech_all/minyu/ssh-gw/TensorRT-LLM"
)
REMOTE_RUN_ROOT = f"{REMOTE_CHECKOUT}/runs/qwen_image_qat_data/test_manifest"
REMOTE_CACHE_ROOT = (
    "/lustre/fsw/portfolios/coreai/projects/coreai_devtech_all/minyu/.cache/huggingface"
)


def _summary_kwargs(tmp_path: Path) -> dict:
    return {
        "manifest_jsonl": tmp_path / "qwen_image_qat_prompts_v1.jsonl",
        "checkout_root": REMOTE_CHECKOUT,
        "run_root": REMOTE_RUN_ROOT,
        "cache_root": REMOTE_CACHE_ROOT,
    }


def test_default_prompt_manifest_counts_and_summary(tmp_path):
    records = build_default_prompt_records()
    counts = validate_prompt_records(records)

    assert counts == DEFAULT_SPLIT_COUNTS
    assert len(records) == 193
    assert records[0]["split"] == "smoke"
    assert records[0]["expected_tuple_count"] == 100
    assert records[0]["cfg_branches"] == ["cond", "negative"]

    categories = {category for record in records for category in record["categories"]}
    assert REQUIRED_CATEGORIES <= categories

    manifest_jsonl = tmp_path / "prompts.jsonl"
    summary_json = tmp_path / "summary.json"
    write_jsonl(manifest_jsonl, records)
    summary = build_summary(records=records, **_summary_kwargs(tmp_path))
    write_json(summary_json, summary)

    assert read_jsonl(manifest_jsonl) == records
    assert read_json(summary_json)["enroot_image"] == DEFAULT_ENROOT_IMAGE
    assert summary["expected_total_teacher_tuples"] == 19_300
    assert summary["durable_paths"]["prompt_manifest"].endswith(
        "manifests/qwen_image_qat_prompts_v1.jsonl"
    )


def test_prompt_manifest_rejects_duplicate_prompt_id():
    records = build_default_prompt_records()
    records[1] = {**records[1], "prompt_id": records[0]["prompt_id"]}

    with pytest.raises(ValueError, match="duplicate prompt_id"):
        validate_prompt_records(records)


def test_prompt_manifest_rejects_missing_source_and_categories():
    records = build_default_prompt_records()
    records[0] = {**records[0], "source": ""}

    with pytest.raises(ValueError, match="non-empty source"):
        validate_prompt_records(records)

    records = build_default_prompt_records()
    records[0] = {**records[0], "categories": []}

    with pytest.raises(ValueError, match="at least one category"):
        validate_prompt_records(records)


def test_prompt_manifest_rejects_held_out_overlap():
    records = build_default_prompt_records()
    records[-1] = {**records[-1], "prompt": records[0]["prompt"]}

    with pytest.raises(ValueError, match="held-out prompts must not overlap"):
        validate_prompt_records(records)


def test_prompt_manifest_rejects_split_count_mismatch():
    records = build_default_prompt_records()

    with pytest.raises(ValueError, match="split counts"):
        validate_prompt_records(records[:-1])


def test_prompt_manifest_rejects_invalid_diffusion_settings():
    records = build_default_prompt_records()
    records[0] = {**records[0], "height": 1025}

    with pytest.raises(ValueError, match="height must be positive and divisible by 8"):
        validate_prompt_records(records)

    records = build_default_prompt_records()
    records[0] = {**records[0], "expected_tuple_count": 50}

    with pytest.raises(ValueError, match="expected_tuple_count"):
        validate_prompt_records(records)


def test_prompt_summary_rejects_docker_and_wrong_enroot(tmp_path):
    records = build_default_prompt_records()
    summary = build_summary(records=records, **_summary_kwargs(tmp_path))

    bad_runtime = {**summary, "container_runtime": "docker"}
    with pytest.raises(ValueError, match="container_runtime"):
        validate_summary(bad_runtime, records=records)

    bad_image = {**summary, "enroot_image": "nvcr.io/nvidia/tensorrt-llm/release:wrong"}
    with pytest.raises(ValueError, match="enroot_image"):
        validate_summary(bad_image, records=records)

    bad_command = {**summary, "remote_command": "docker run --rm --gpus all"}
    with pytest.raises(ValueError, match="Docker provenance"):
        validate_summary(bad_command, records=records)


def test_prompt_summary_rejects_tmp_durable_path(tmp_path):
    records = build_default_prompt_records()
    summary = build_summary(records=records, **_summary_kwargs(tmp_path))
    summary["durable_paths"] = {
        **summary["durable_paths"],
        "run_root": "/tmp/qwen_image_qat",
    }

    with pytest.raises(ValueError, match="must not be under /tmp"):
        validate_summary(summary, records=records)
