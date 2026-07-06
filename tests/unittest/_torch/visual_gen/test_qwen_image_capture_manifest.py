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

from scripts.visualgen_eval.qwen_image_capture_manifest import (
    REQUIRED_CAPTURE_FIELDS,
    build_capture_plan,
    build_tuple_index,
    timestep_bin_name,
    validate_capture_summary,
    validate_prompt_handoff,
    validate_tuple_index,
)
from scripts.visualgen_eval.qwen_image_prompt_manifest import (
    LOCAL_SMOKE_ARTIFACT_POLICY,
    build_default_prompt_records,
    build_summary,
)

REMOTE_CHECKOUT = (
    "/lustre/fsw/portfolios/coreai/projects/coreai_devtech_all/minyu/ssh-gw/TensorRT-LLM"
)
REMOTE_RUN_ROOT = f"{REMOTE_CHECKOUT}/runs/qwen_image_qat_data/test_capture"
REMOTE_CACHE_ROOT = (
    "/lustre/fsw/portfolios/coreai/projects/coreai_devtech_all/minyu/.cache/huggingface"
)
PROMPT_MANIFEST_PATH = Path(REMOTE_RUN_ROOT) / "manifests" / "qwen_image_qat_prompts_v1.jsonl"
CAPTURE_SUMMARY_PATH = Path(REMOTE_RUN_ROOT) / "manifests" / "qwen_image_teacher_capture_v1.json"
TUPLE_INDEX_PATH = Path(REMOTE_RUN_ROOT) / "manifests" / "qwen_image_teacher_tuple_index_v1.jsonl"


def _prompt_records_and_summary():
    records = build_default_prompt_records()
    summary = build_summary(
        records=records,
        manifest_jsonl=PROMPT_MANIFEST_PATH,
        checkout_root=REMOTE_CHECKOUT,
        run_root=REMOTE_RUN_ROOT,
        cache_root=REMOTE_CACHE_ROOT,
    )
    return records, summary


def test_build_smoke_capture_plan_from_formal_prompt_summary():
    records, prompt_summary = _prompt_records_and_summary()
    capture_summary, tuple_entries = build_capture_plan(
        records=records,
        prompt_summary=prompt_summary,
        prompt_manifest_path=PROMPT_MANIFEST_PATH,
        capture_summary_path=CAPTURE_SUMMARY_PATH,
        tuple_index_path=TUPLE_INDEX_PATH,
        splits=("smoke",),
        project_root=Path.cwd(),
    )

    assert capture_summary["format"] == "qwen_image_teacher_capture_manifest_v1"
    assert capture_summary["split_counts"] == {"smoke": 1}
    assert capture_summary["expected_total_tuples"] == 100
    assert capture_summary["task3_input_ready"] is True
    assert len(tuple_entries) == 100
    assert tuple_entries[0]["prompt_id"] == "qwen_image_smoke_0000"
    assert tuple_entries[0]["cfg_branch"] == "cond"
    assert tuple_entries[1]["cfg_branch"] == "negative"
    assert tuple_entries[0]["timestep_bin"] == "early"
    assert tuple_entries[-1]["timestep_bin"] == "late"
    assert tuple_entries[0]["tuple_path"].startswith(f"{REMOTE_RUN_ROOT}/teacher_tuples/smoke/")
    assert tuple_entries[0]["required_fields"] == list(REQUIRED_CAPTURE_FIELDS)


def test_capture_rejects_local_smoke_prompt_summary(tmp_path):
    records = build_default_prompt_records()
    prompt_summary = build_summary(
        records=records,
        manifest_jsonl=tmp_path / "prompts.jsonl",
        checkout_root=REMOTE_CHECKOUT,
        run_root=REMOTE_RUN_ROOT,
        cache_root=REMOTE_CACHE_ROOT,
        artifact_policy=LOCAL_SMOKE_ARTIFACT_POLICY,
    )

    with pytest.raises(ValueError, match="formal durable prompt summary"):
        validate_prompt_handoff(
            records=records,
            prompt_summary=prompt_summary,
            prompt_manifest_path=tmp_path / "prompts.jsonl",
        )


def test_capture_rejects_prompt_manifest_path_mismatch(tmp_path):
    records, prompt_summary = _prompt_records_and_summary()

    with pytest.raises(ValueError, match="prompt_manifest_path"):
        validate_prompt_handoff(
            records=records,
            prompt_summary=prompt_summary,
            prompt_manifest_path=tmp_path / "prompts.jsonl",
        )


def test_capture_rejects_layered_prompt_fields():
    records, prompt_summary = _prompt_records_and_summary()
    records[0] = {**records[0], "image": "layered-input.png"}

    with pytest.raises(ValueError, match="layered fields"):
        validate_prompt_handoff(
            records=records,
            prompt_summary=prompt_summary,
            prompt_manifest_path=PROMPT_MANIFEST_PATH,
        )


def test_tuple_index_rejects_wrong_true_cfg_count():
    records, prompt_summary = _prompt_records_and_summary()
    smoke_records = [record for record in records if record["split"] == "smoke"]
    tuple_entries = build_tuple_index(smoke_records, prompt_summary=prompt_summary)

    with pytest.raises(ValueError, match="true CFG counts"):
        validate_tuple_index(tuple_entries[:-1], records=smoke_records)


def test_tuple_index_rejects_missing_required_fields():
    records, prompt_summary = _prompt_records_and_summary()
    smoke_records = [record for record in records if record["split"] == "smoke"]
    tuple_entries = build_tuple_index(smoke_records, prompt_summary=prompt_summary)
    tuple_entries[0] = {**tuple_entries[0], "required_fields": ["prompt_id"]}

    with pytest.raises(ValueError, match="required_fields"):
        validate_tuple_index(tuple_entries, records=smoke_records)


def test_tuple_index_rejects_bad_timestep_bin():
    records, prompt_summary = _prompt_records_and_summary()
    smoke_records = [record for record in records if record["split"] == "smoke"]
    tuple_entries = build_tuple_index(smoke_records, prompt_summary=prompt_summary)
    tuple_entries[0] = {**tuple_entries[0], "timestep_bin": "bad"}

    with pytest.raises(ValueError, match="timestep_bin"):
        validate_tuple_index(tuple_entries, records=smoke_records)


def test_capture_summary_rejects_tmp_and_docker_provenance():
    records, prompt_summary = _prompt_records_and_summary()
    smoke_records = [record for record in records if record["split"] == "smoke"]
    capture_summary, tuple_entries = build_capture_plan(
        records=records,
        prompt_summary=prompt_summary,
        prompt_manifest_path=PROMPT_MANIFEST_PATH,
        capture_summary_path=CAPTURE_SUMMARY_PATH,
        tuple_index_path=TUPLE_INDEX_PATH,
        splits=("smoke",),
        project_root=Path.cwd(),
    )

    bad_tmp = {
        **capture_summary,
        "durable_paths": {
            **capture_summary["durable_paths"],
            "teacher_tuples": "/tmp/teacher_tuples",
        },
    }
    with pytest.raises(ValueError, match="teacher_tuples"):
        validate_capture_summary(
            bad_tmp,
            records=smoke_records,
            prompt_summary=prompt_summary,
            tuple_entries=tuple_entries,
        )

    bad_runtime = {**capture_summary, "remote_command": "docker run --rm --gpus all"}
    with pytest.raises(ValueError, match="Docker provenance"):
        validate_capture_summary(
            bad_runtime,
            records=smoke_records,
            prompt_summary=prompt_summary,
            tuple_entries=tuple_entries,
        )


def test_timestep_bin_boundaries():
    assert timestep_bin_name(0, 50) == "early"
    assert timestep_bin_name(9, 50) == "early"
    assert timestep_bin_name(10, 50) == "early_mid"
    assert timestep_bin_name(20, 50) == "mid"
    assert timestep_bin_name(30, 50) == "late_mid"
    assert timestep_bin_name(40, 50) == "late"
