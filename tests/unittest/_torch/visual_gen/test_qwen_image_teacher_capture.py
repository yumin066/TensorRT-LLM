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

import pickle
import shutil
import uuid
from pathlib import Path

import pytest

import scripts.visualgen_eval.qwen_image_teacher_capture as teacher_capture
from scripts.visualgen_eval.qwen_image_capture_manifest import (
    build_capture_plan,
    read_tuple_index_jsonl,
    write_json,
    write_jsonl,
)
from scripts.visualgen_eval.qwen_image_prompt_manifest import (
    DEFAULT_ENROOT_IMAGE,
    build_default_prompt_records,
    build_summary,
    read_json,
)
from scripts.visualgen_eval.qwen_image_teacher_capture import (
    build_captured_summary,
    capture_records_with_pipeline,
    validate_captured_artifacts,
)

CaptureCase = tuple[
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]


class FakeTensor:
    def __init__(self, dtype: str = "bfloat16") -> None:
        self.dtype = dtype

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> list[int]:
        return [0]


class FakeTorch:
    Tensor = FakeTensor
    bfloat16 = "bfloat16"

    @staticmethod
    def save(payload: object, path: str | Path) -> None:
        with Path(path).open("wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load(
        path: str | Path, map_location: str | None = None, weights_only: bool = False
    ) -> object:
        del map_location, weights_only
        with Path(path).open("rb") as f:
            return pickle.load(f)


class DummyTransformer:
    def forward(self, **kwargs: object) -> tuple[FakeTensor]:
        del kwargs
        return (FakeTensor(),)


class DummyScheduler:
    config = {"name": "dummy-flow-match"}
    timesteps = [1000, 500]


class DummyPipeline:
    def __init__(self, *, include_txt_seq_lens: bool = True) -> None:
        self.transformer = DummyTransformer()
        self.scheduler = DummyScheduler()
        self.include_txt_seq_lens = include_txt_seq_lens

    def infer(self, record: dict[str, object]) -> object:
        num_steps = record["num_inference_steps"]
        if not isinstance(num_steps, int):
            raise TypeError("num_inference_steps must be an int")
        for _ in range(num_steps):
            self._forward_branch()
            self._forward_branch()
        return object()

    def _forward_branch(self) -> None:
        kwargs: dict[str, object] = {
            "hidden_states": FakeTensor(),
            "timestep": FakeTensor(),
            "encoder_hidden_states": FakeTensor(),
            "encoder_hidden_states_mask": FakeTensor(dtype="bool"),
            "img_shapes": [[(1, 64, 64)]],
            "return_dict": False,
        }
        if self.include_txt_seq_lens:
            kwargs["txt_seq_lens"] = [16]
        else:
            kwargs["encoder_hidden_states_mask"] = [[True] * 16]
        self.transformer.forward(**kwargs)


@pytest.fixture
def capture_case(request: pytest.FixtureRequest) -> CaptureCase:
    root = (
        Path.cwd()
        / ".pytest_cache"
        / "qwen_image_teacher_capture"
        / f"{request.node.name}_{uuid.uuid4().hex}"
    )
    shutil.rmtree(root, ignore_errors=True)
    records = build_default_prompt_records()
    prompt_manifest_path = root / "manifests" / "qwen_image_qat_prompts_v1.jsonl"
    capture_summary_path = root / "manifests" / "qwen_image_teacher_capture_v1.json"
    tuple_index_path = root / "manifests" / "qwen_image_teacher_tuple_index_v1.jsonl"
    prompt_summary = build_summary(
        records=records,
        manifest_jsonl=prompt_manifest_path,
        checkout_root=str(Path.cwd()),
        run_root=str(root),
        cache_root=str(root / "hf_cache"),
    )
    capture_summary, tuple_entries = build_capture_plan(
        records=records,
        prompt_summary=prompt_summary,
        prompt_manifest_path=prompt_manifest_path,
        capture_summary_path=capture_summary_path,
        tuple_index_path=tuple_index_path,
        splits=("smoke",),
        project_root=Path.cwd(),
    )
    smoke_records = [record for record in records if record["split"] == "smoke"]

    def cleanup() -> None:
        shutil.rmtree(root, ignore_errors=True)

    request.addfinalizer(cleanup)
    return records, prompt_summary, capture_summary, tuple_entries, smoke_records


def _write_reference(_output: object, path: Path) -> None:
    path.write_bytes(b"fake png")


def _capture_with_dummy_pipeline(
    capture_case: CaptureCase,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object], list[dict[str, object]]]:
    records, prompt_summary, capture_summary, tuple_entries, smoke_records = capture_case
    pipeline = DummyPipeline()
    capture_records_with_pipeline(
        pipeline,
        records=smoke_records,
        tuple_entries=tuple_entries,
        save_reference_fn=_write_reference,
        infer_fn=lambda dummy_pipeline, record: dummy_pipeline.infer(record),
        torch_module=FakeTorch,
    )
    captured_summary = build_captured_summary(
        capture_summary,
        tuple_entries=tuple_entries,
        provenance={
            "container_runtime": "enroot",
            "enroot_image": DEFAULT_ENROOT_IMAGE,
            "command": "ssh-gw task submit alloc -- python capture-smoke",
            "git_head": "captured-git-head",
        },
        pipeline=pipeline,
    )
    return records, prompt_summary, captured_summary, tuple_entries


def test_capture_records_with_dummy_pipeline_and_validate(capture_case: CaptureCase) -> None:
    records, prompt_summary, captured_summary, tuple_entries = _capture_with_dummy_pipeline(
        capture_case
    )

    validate_captured_artifacts(
        captured_summary,
        records=records,
        prompt_summary=prompt_summary,
        tuple_entries=tuple_entries,
        torch_module=FakeTorch,
    )

    assert tuple_entries[0]["status"] == "captured"
    assert Path(tuple_entries[0]["tuple_path"]).is_file()
    assert Path(tuple_entries[0]["reference_image_path"]).is_file()
    assert captured_summary["git_head"] == "captured-git-head"
    assert captured_summary["capture_provenance"]["git_head"] == "captured-git-head"


def test_capture_derives_missing_txt_seq_lens(capture_case: CaptureCase) -> None:
    records, prompt_summary, capture_summary, tuple_entries, smoke_records = capture_case
    pipeline = DummyPipeline(include_txt_seq_lens=False)
    capture_records_with_pipeline(
        pipeline,
        records=smoke_records,
        tuple_entries=tuple_entries,
        save_reference_fn=_write_reference,
        infer_fn=lambda dummy_pipeline, record: dummy_pipeline.infer(record),
        torch_module=FakeTorch,
    )
    captured_summary = build_captured_summary(
        capture_summary,
        tuple_entries=tuple_entries,
        provenance={
            "container_runtime": "enroot",
            "enroot_image": DEFAULT_ENROOT_IMAGE,
            "command": "ssh-gw task submit alloc -- python capture-smoke",
            "git_head": "captured-git-head",
        },
        pipeline=pipeline,
    )

    validate_captured_artifacts(
        captured_summary,
        records=records,
        prompt_summary=prompt_summary,
        tuple_entries=tuple_entries,
        torch_module=FakeTorch,
    )

    payload = FakeTorch.load(tuple_entries[0]["tuple_path"])
    if not isinstance(payload, dict):
        raise TypeError("expected fake tuple payload to be a mapping")
    assert payload["txt_seq_lens"] == [16]


def test_capture_split_cli_selects_fast_calibration(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    root = (
        Path.cwd()
        / ".pytest_cache"
        / "qwen_image_teacher_capture"
        / f"{request.node.name}_{uuid.uuid4().hex}"
    )
    shutil.rmtree(root, ignore_errors=True)
    request.addfinalizer(lambda: shutil.rmtree(root, ignore_errors=True))

    records = build_default_prompt_records()
    prompt_manifest_path = root / "manifests" / "qwen_image_qat_prompts_v1.jsonl"
    prompt_summary_path = root / "manifests" / "qwen_image_qat_prompts_summary_v1.json"
    capture_summary_path = root / "manifests" / "qwen_image_teacher_capture_fast_plan_v1.json"
    tuple_index_path = root / "manifests" / "qwen_image_teacher_tuple_index_fast_plan_v1.jsonl"
    output_summary_path = root / "manifests" / "qwen_image_teacher_capture_fast_captured_v1.json"
    output_index_path = root / "manifests" / "qwen_image_teacher_tuple_index_fast_captured_v1.jsonl"
    prompt_summary = build_summary(
        records=records,
        manifest_jsonl=prompt_manifest_path,
        checkout_root=str(Path.cwd()),
        run_root=str(root),
        cache_root=str(root / "hf_cache"),
    )
    capture_summary, tuple_entries = build_capture_plan(
        records=records,
        prompt_summary=prompt_summary,
        prompt_manifest_path=prompt_manifest_path,
        capture_summary_path=capture_summary_path,
        tuple_index_path=tuple_index_path,
        splits=("fast_calibration",),
        project_root=Path.cwd(),
    )
    write_jsonl(prompt_manifest_path, records)
    write_json(prompt_summary_path, prompt_summary)
    write_json(capture_summary_path, capture_summary)
    write_jsonl(tuple_index_path, tuple_entries)

    monkeypatch.setattr(
        teacher_capture, "load_single_worker_pipeline", lambda **_kwargs: DummyPipeline()
    )
    monkeypatch.setattr(
        teacher_capture, "infer_record", lambda pipeline, record: pipeline.infer(record)
    )
    monkeypatch.setattr(teacher_capture, "_load_torch", lambda: FakeTorch)
    monkeypatch.setattr(teacher_capture, "save_reference_image", _write_reference)
    monkeypatch.setattr(teacher_capture, "git_commit", lambda _path: "captured-git-head")

    teacher_capture.main(
        [
            "capture-split",
            "--split",
            "fast_calibration",
            "--prompt-manifest-jsonl",
            str(prompt_manifest_path),
            "--prompt-summary-json",
            str(prompt_summary_path),
            "--capture-summary-json",
            str(capture_summary_path),
            "--tuple-index-jsonl",
            str(tuple_index_path),
            "--output-capture-summary-json",
            str(output_summary_path),
            "--output-tuple-index-jsonl",
            str(output_index_path),
            "--visual-gen-args",
            str(root / "qwen-image-bf16-sage-fp8-1gpu.yaml"),
            "--model",
            "Qwen/Qwen-Image",
            "--command",
            "ssh-gw task submit alloc -- python capture-split --split fast_calibration",
        ]
    )

    captured_summary = read_json(output_summary_path)
    captured_entries = read_tuple_index_jsonl(output_index_path)
    assert captured_summary["prompt_count"] == 32
    assert captured_summary["captured_total_tuples"] == 3200
    assert {entry["split"] for entry in captured_entries} == {"fast_calibration"}
    assert all(Path(entry["tuple_path"]).is_file() for entry in captured_entries)
    assert all(Path(entry["reference_image_path"]).is_file() for entry in captured_entries)


def test_captured_validator_rejects_missing_tuple_file(capture_case: CaptureCase) -> None:
    records, prompt_summary, captured_summary, tuple_entries = _capture_with_dummy_pipeline(
        capture_case
    )
    Path(tuple_entries[0]["tuple_path"]).unlink()

    with pytest.raises(ValueError, match="file does not exist"):
        validate_captured_artifacts(
            captured_summary,
            records=records,
            prompt_summary=prompt_summary,
            tuple_entries=tuple_entries,
            torch_module=FakeTorch,
        )


def test_captured_validator_rejects_missing_field_and_wrong_dtype(
    capture_case: CaptureCase,
) -> None:
    records, prompt_summary, captured_summary, tuple_entries = _capture_with_dummy_pipeline(
        capture_case
    )
    tuple_path = Path(tuple_entries[0]["tuple_path"])
    payload = FakeTorch.load(tuple_path)
    payload.pop("hidden_states")
    FakeTorch.save(payload, tuple_path)
    with pytest.raises(ValueError, match="missing required fields"):
        validate_captured_artifacts(
            captured_summary,
            records=records,
            prompt_summary=prompt_summary,
            tuple_entries=tuple_entries,
            torch_module=FakeTorch,
        )

    records, prompt_summary, captured_summary, tuple_entries = _capture_with_dummy_pipeline(
        capture_case
    )
    tuple_path = Path(tuple_entries[0]["tuple_path"])
    payload = FakeTorch.load(tuple_path)
    payload["hidden_states"] = FakeTensor(dtype="float32")
    FakeTorch.save(payload, tuple_path)
    with pytest.raises(ValueError, match="bfloat16"):
        validate_captured_artifacts(
            captured_summary,
            records=records,
            prompt_summary=prompt_summary,
            tuple_entries=tuple_entries,
            torch_module=FakeTorch,
        )


def test_captured_validator_rejects_wrong_tuple_count_and_forbidden_runtime(
    capture_case: CaptureCase,
) -> None:
    records, prompt_summary, captured_summary, tuple_entries = _capture_with_dummy_pipeline(
        capture_case
    )
    bad_count = {**captured_summary, "captured_total_tuples": len(tuple_entries) - 1}
    with pytest.raises(ValueError, match="captured_total_tuples"):
        validate_captured_artifacts(
            bad_count,
            records=records,
            prompt_summary=prompt_summary,
            tuple_entries=tuple_entries,
            torch_module=FakeTorch,
        )

    bad_runtime = {
        **captured_summary,
        "capture_provenance": {
            **captured_summary["capture_provenance"],
            "command": "docker run --rm",
        },
    }
    with pytest.raises(ValueError, match="Docker"):
        validate_captured_artifacts(
            bad_runtime,
            records=records,
            prompt_summary=prompt_summary,
            tuple_entries=tuple_entries,
            torch_module=FakeTorch,
        )

    bad_git_head = {
        **captured_summary,
        "git_head": "planned-stale-git-head",
    }
    with pytest.raises(ValueError, match="git_head"):
        validate_captured_artifacts(
            bad_git_head,
            records=records,
            prompt_summary=prompt_summary,
            tuple_entries=tuple_entries,
            torch_module=FakeTorch,
        )
