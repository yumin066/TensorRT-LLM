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

"""Plan and validate ordinary Qwen-Image BF16 teacher tuple capture artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal, cast

from scripts.visualgen_eval.qwen_image_prompt_manifest import (
    DEFAULT_ARTIFACT_POLICY,
    DEFAULT_CFG_BRANCHES,
    DEFAULT_CLUSTER_ALIAS,
    DEFAULT_CONTAINER_RUNTIME,
    DEFAULT_ENROOT_IMAGE,
    REQUIRED_CAPTURE_FIELDS,
    read_json,
    validate_prompt_records,
)
from scripts.visualgen_eval.qwen_image_prompt_manifest import read_jsonl as read_prompt_jsonl
from scripts.visualgen_eval.qwen_image_prompt_manifest import (
    validate_summary as validate_prompt_summary,
)

CaptureSplit = Literal["smoke", "fast_calibration", "main_calibration", "held_out"]

CAPTURE_MANIFEST_FORMAT = "qwen_image_teacher_capture_manifest_v1"
TUPLE_INDEX_FORMAT = "qwen_image_teacher_tuple_index_v1"
DEFAULT_CAPTURE_SUMMARY_NAME = "qwen_image_teacher_capture_v1.json"
DEFAULT_TUPLE_INDEX_NAME = "qwen_image_teacher_tuple_index_v1.jsonl"
TIMESTEP_BIN_NAMES = ("early", "early_mid", "mid", "late_mid", "late")
TIMESTEP_BIN_EDGES = (0.2, 0.4, 0.6, 0.8)
LAYERED_ONLY_FIELDS = {
    "image",
    "layers",
    "layer_count",
    "layer_rgba",
    "alpha_mask",
    "composite",
    "composite_white",
    "resolution",
}
REQUIRED_CAPTURE_DURABLE_PATH_FIELDS = (
    "run_root",
    "bf16_references",
    "teacher_tuples",
    "capture_summary",
    "tuple_index",
)


def build_capture_plan(
    *,
    records: list[dict[str, object]],
    prompt_summary: dict[str, object],
    prompt_manifest_path: Path,
    capture_summary_path: Path,
    tuple_index_path: Path,
    splits: tuple[CaptureSplit, ...] | None = None,
    project_root: Path | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Build a BF16 reference and teacher tuple capture plan for selected prompt splits."""
    validate_prompt_handoff(
        records=records,
        prompt_summary=prompt_summary,
        prompt_manifest_path=prompt_manifest_path,
    )
    selected_records = _select_records(records, splits=splits)
    tuple_entries = build_tuple_index(selected_records, prompt_summary=prompt_summary)
    summary = build_capture_summary(
        records=selected_records,
        prompt_summary=prompt_summary,
        prompt_manifest_path=prompt_manifest_path,
        capture_summary_path=capture_summary_path,
        tuple_index_path=tuple_index_path,
        tuple_entries=tuple_entries,
        project_root=project_root,
    )
    validate_capture_summary(
        summary,
        records=selected_records,
        prompt_summary=prompt_summary,
        tuple_entries=tuple_entries,
    )
    return summary, tuple_entries


def validate_prompt_handoff(
    *,
    records: list[dict[str, object]],
    prompt_summary: dict[str, object],
    prompt_manifest_path: Path,
) -> None:
    """Validate that the prompt manifest is a formal task2-ready B300 handoff."""
    validate_prompt_records(records)
    _reject_layered_prompt_fields(records)
    validate_prompt_summary(prompt_summary, records=records)
    if prompt_summary.get("artifact_policy") != DEFAULT_ARTIFACT_POLICY:
        raise ValueError("task2 capture requires a formal durable prompt summary")
    if prompt_summary.get("task2_input_ready") is not True:
        raise ValueError("task2 capture requires prompt summary task2_input_ready=true")

    durable_paths = _expect_mapping(prompt_summary, "durable_paths")
    durable_manifest = _expect_string(durable_paths, "prompt_manifest")
    if str(prompt_manifest_path) != durable_manifest:
        raise ValueError("prompt_manifest_path must match durable_paths.prompt_manifest")


def build_tuple_index(
    records: list[dict[str, object]],
    *,
    prompt_summary: dict[str, object],
) -> list[dict[str, object]]:
    """Build planned teacher tuple entries keyed by prompt, timestep, and CFG branch."""
    _validate_selected_prompt_records(records)
    _reject_layered_prompt_fields(records)
    durable_paths = _expect_mapping(prompt_summary, "durable_paths")
    reference_root = _expect_string(durable_paths, "bf16_references")
    tuple_root = _expect_string(durable_paths, "teacher_tuples")

    entries: list[dict[str, object]] = []
    for record in records:
        prompt_id = _expect_string(record, "prompt_id")
        split = _expect_split(record)
        num_steps = _expect_positive_int(record, "num_inference_steps")
        cfg_branches = _expect_string_list(record, "cfg_branches")
        if tuple(cfg_branches) != DEFAULT_CFG_BRANCHES:
            raise ValueError(
                f"prompt {prompt_id} cfg_branches must be {list(DEFAULT_CFG_BRANCHES)}"
            )
        reference_image_path = f"{reference_root}/{split}/{prompt_id}.png"
        for timestep_index in range(num_steps):
            timestep_bin = timestep_bin_name(timestep_index, num_steps)
            for cfg_branch in cfg_branches:
                tuple_id = f"{prompt_id}_step{timestep_index:03d}_{cfg_branch}"
                entries.append(
                    {
                        "format": TUPLE_INDEX_FORMAT,
                        "tuple_id": tuple_id,
                        "prompt_id": prompt_id,
                        "split": split,
                        "reference_image_path": reference_image_path,
                        "tuple_path": (
                            f"{tuple_root}/{split}/{prompt_id}/"
                            f"tuple_step{timestep_index:03d}_{cfg_branch}.pt"
                        ),
                        "timestep_index": timestep_index,
                        "timestep_bin": timestep_bin,
                        "cfg_branch": cfg_branch,
                        "required_fields": list(REQUIRED_CAPTURE_FIELDS),
                        "trajectory_source": "bf16_teacher",
                        "status": "planned",
                    }
                )
    validate_tuple_index(entries, records=records)
    return entries


def build_capture_summary(
    *,
    records: list[dict[str, object]],
    prompt_summary: dict[str, object],
    prompt_manifest_path: Path,
    capture_summary_path: Path,
    tuple_index_path: Path,
    tuple_entries: list[dict[str, object]],
    project_root: Path | None = None,
) -> dict[str, object]:
    """Build the top-level capture summary/provenance payload."""
    durable_paths = _expect_mapping(prompt_summary, "durable_paths")
    run_root = _expect_string(durable_paths, "run_root").rstrip("/")
    capture_summary = str(capture_summary_path)
    tuple_index = str(tuple_index_path)
    payload: dict[str, object] = {
        "format": CAPTURE_MANIFEST_FORMAT,
        "prompt_manifest_path": str(prompt_manifest_path),
        "prompt_summary_format": prompt_summary["format"],
        "prompt_count": len(records),
        "split_counts": _split_counts(records),
        "model": prompt_summary["model"],
        "height": prompt_summary["height"],
        "width": prompt_summary["width"],
        "num_inference_steps": prompt_summary["num_inference_steps"],
        "guidance_scale": prompt_summary["guidance_scale"],
        "max_sequence_length": prompt_summary["max_sequence_length"],
        "cfg_branches": list(DEFAULT_CFG_BRANCHES),
        "required_capture_fields": list(REQUIRED_CAPTURE_FIELDS),
        "timestep_bins": _timestep_bin_specs(),
        "expected_tuples_per_prompt": prompt_summary["expected_tuples_per_prompt"],
        "expected_total_tuples": len(tuple_entries),
        "cluster_alias": DEFAULT_CLUSTER_ALIAS,
        "container_runtime": DEFAULT_CONTAINER_RUNTIME,
        "enroot_image": DEFAULT_ENROOT_IMAGE,
        "artifact_policy": DEFAULT_ARTIFACT_POLICY,
        "task3_input_ready": True,
        "git_head": git_commit(project_root or Path.cwd()),
        "durable_paths": {
            "run_root": run_root,
            "bf16_references": _expect_string(durable_paths, "bf16_references"),
            "teacher_tuples": _expect_string(durable_paths, "teacher_tuples"),
            "capture_summary": capture_summary,
            "tuple_index": tuple_index,
        },
    }
    return payload


def validate_capture_summary(
    summary: dict[str, object],
    *,
    records: list[dict[str, object]],
    prompt_summary: dict[str, object],
    tuple_entries: list[dict[str, object]],
) -> None:
    _validate_selected_prompt_records(records)
    _reject_layered_prompt_fields(records)
    if prompt_summary.get("artifact_policy") != DEFAULT_ARTIFACT_POLICY:
        raise ValueError("capture summary requires a formal durable prompt summary")
    if prompt_summary.get("task2_input_ready") is not True:
        raise ValueError("capture summary requires prompt summary task2_input_ready=true")
    prompt_durable_paths = _expect_mapping(prompt_summary, "durable_paths")
    if summary.get("prompt_manifest_path") != _expect_string(
        prompt_durable_paths, "prompt_manifest"
    ):
        raise ValueError("capture summary prompt_manifest_path must match prompt summary")
    validate_tuple_index(tuple_entries, records=records)
    if summary.get("format") != CAPTURE_MANIFEST_FORMAT:
        raise ValueError(f"capture summary format must be {CAPTURE_MANIFEST_FORMAT}")
    for field_name in (
        "model",
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "max_sequence_length",
        "expected_tuples_per_prompt",
    ):
        if summary.get(field_name) != prompt_summary.get(field_name):
            raise ValueError(f"capture summary {field_name} does not match prompt summary")
    if summary.get("split_counts") != _split_counts(records):
        raise ValueError("capture summary split_counts do not match selected prompt records")
    if summary.get("expected_total_tuples") != len(tuple_entries):
        raise ValueError("capture summary expected_total_tuples does not match tuple index")
    if summary.get("cfg_branches") != list(DEFAULT_CFG_BRANCHES):
        raise ValueError("capture summary cfg_branches do not match true CFG branches")
    if set(_expect_string_list(summary, "required_capture_fields")) != set(REQUIRED_CAPTURE_FIELDS):
        raise ValueError("capture summary required_capture_fields are incomplete")
    if summary.get("cluster_alias") != DEFAULT_CLUSTER_ALIAS:
        raise ValueError(f"capture summary cluster_alias must be {DEFAULT_CLUSTER_ALIAS}")
    if summary.get("container_runtime") != DEFAULT_CONTAINER_RUNTIME:
        raise ValueError(f"capture summary container_runtime must be {DEFAULT_CONTAINER_RUNTIME}")
    if summary.get("enroot_image") != DEFAULT_ENROOT_IMAGE:
        raise ValueError(f"capture summary enroot_image must be {DEFAULT_ENROOT_IMAGE}")
    if summary.get("artifact_policy") != DEFAULT_ARTIFACT_POLICY:
        raise ValueError("capture summary artifact_policy must be durable_b300_enroot_only")
    if _contains_forbidden_runtime(summary):
        raise ValueError("capture summary must not include Docker provenance")
    _validate_capture_durable_paths(summary)


def validate_tuple_index(
    entries: list[dict[str, object]],
    *,
    records: list[dict[str, object]],
) -> None:
    if not entries:
        raise ValueError("tuple index must contain at least one entry")

    records_by_prompt = {_expect_string(record, "prompt_id"): record for record in records}
    expected_counts = {
        prompt_id: _expect_positive_int(record, "num_inference_steps")
        * len(_expect_string_list(record, "cfg_branches"))
        for prompt_id, record in records_by_prompt.items()
    }
    observed_counts: Counter[str] = Counter()
    seen_tuple_ids: set[str] = set()

    for entry in entries:
        if entry.get("format") != TUPLE_INDEX_FORMAT:
            raise ValueError(f"tuple entry format must be {TUPLE_INDEX_FORMAT}")
        tuple_id = _expect_string(entry, "tuple_id")
        if tuple_id in seen_tuple_ids:
            raise ValueError(f"duplicate tuple_id in tuple index: {tuple_id}")
        seen_tuple_ids.add(tuple_id)
        prompt_id = _expect_string(entry, "prompt_id")
        if prompt_id not in records_by_prompt:
            raise ValueError(f"tuple entry references unknown prompt_id: {prompt_id}")
        record = records_by_prompt[prompt_id]
        num_steps = _expect_positive_int(record, "num_inference_steps")
        timestep_index = _expect_non_negative_int(entry, "timestep_index")
        if timestep_index >= num_steps:
            raise ValueError(f"tuple {tuple_id} timestep_index exceeds prompt num_inference_steps")
        expected_bin = timestep_bin_name(timestep_index, num_steps)
        if _expect_string(entry, "timestep_bin") != expected_bin:
            raise ValueError(f"tuple {tuple_id} timestep_bin does not match timestep_index")
        cfg_branch = _expect_string(entry, "cfg_branch")
        if cfg_branch not in _expect_string_list(record, "cfg_branches"):
            raise ValueError(f"tuple {tuple_id} cfg_branch is not in prompt cfg_branches")
        if set(_expect_string_list(entry, "required_fields")) != set(REQUIRED_CAPTURE_FIELDS):
            raise ValueError(f"tuple {tuple_id} required_fields are incomplete")
        for path_field in ("reference_image_path", "tuple_path"):
            path = _expect_string(entry, path_field)
            if path.startswith("/tmp"):
                raise ValueError(f"tuple {tuple_id} {path_field} must not be under /tmp")
        observed_counts[prompt_id] += 1

    if dict(observed_counts) != expected_counts:
        raise ValueError(
            f"tuple counts do not match expected true CFG counts: "
            f"observed={dict(observed_counts)}, expected={expected_counts}"
        )


def timestep_bin_name(timestep_index: int, num_inference_steps: int) -> str:
    if timestep_index < 0:
        raise ValueError("timestep_index must be non-negative")
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if timestep_index >= num_inference_steps:
        raise ValueError("timestep_index must be smaller than num_inference_steps")
    fraction = (timestep_index + 1) / float(num_inference_steps)
    for edge, name in zip(TIMESTEP_BIN_EDGES, TIMESTEP_BIN_NAMES[:-1], strict=True):
        if fraction <= edge:
            return name
    return TIMESTEP_BIN_NAMES[-1]


def _timestep_bin_specs() -> list[dict[str, object]]:
    starts = (0.0, *TIMESTEP_BIN_EDGES)
    ends = (*TIMESTEP_BIN_EDGES, 1.0)
    return [
        {"name": name, "start_fraction": start, "end_fraction": end}
        for name, start, end in zip(TIMESTEP_BIN_NAMES, starts, ends, strict=True)
    ]


def _select_records(
    records: list[dict[str, object]],
    *,
    splits: tuple[CaptureSplit, ...] | None,
) -> list[dict[str, object]]:
    if splits is None:
        return list(records)
    selected = [record for record in records if _expect_split(record) in splits]
    if not selected:
        raise ValueError(f"no prompt records selected for splits: {splits}")
    return selected


def _validate_selected_prompt_records(records: list[dict[str, object]]) -> None:
    if not records:
        raise ValueError("capture prompt selection must contain at least one record")
    seen_prompt_ids: set[str] = set()
    common_settings: dict[str, object] = {}
    for record in records:
        prompt_id = _expect_string(record, "prompt_id")
        if prompt_id in seen_prompt_ids:
            raise ValueError(f"duplicate prompt_id in capture prompt selection: {prompt_id}")
        seen_prompt_ids.add(prompt_id)
        _expect_split(record)
        if not _expect_string(record, "source"):
            raise ValueError(f"prompt {prompt_id} must include a non-empty source")
        if not _expect_string_list(record, "categories"):
            raise ValueError(f"prompt {prompt_id} must include at least one category")
        if not _expect_string(record, "prompt"):
            raise ValueError(f"prompt {prompt_id} must include prompt text")
        if not _expect_string(record, "negative_prompt"):
            raise ValueError(f"prompt {prompt_id} must include a negative prompt")
        num_steps = _expect_positive_int(record, "num_inference_steps")
        cfg_branches = _expect_string_list(record, "cfg_branches")
        if tuple(cfg_branches) != DEFAULT_CFG_BRANCHES:
            raise ValueError(
                f"prompt {prompt_id} cfg_branches must be {list(DEFAULT_CFG_BRANCHES)}"
            )
        if record.get("expected_tuple_count") != num_steps * len(cfg_branches):
            raise ValueError(f"prompt {prompt_id} expected_tuple_count does not match CFG count")
        for field_name in (
            "model",
            "negative_prompt",
            "height",
            "width",
            "num_inference_steps",
            "guidance_scale",
            "max_sequence_length",
            "cfg_branches",
        ):
            field_value = record.get(field_name)
            if field_name not in common_settings:
                common_settings[field_name] = field_value
                continue
            if common_settings[field_name] != field_value:
                raise ValueError(
                    f"prompt {prompt_id} {field_name} does not match selection settings"
                )


def _split_counts(records: list[dict[str, object]]) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for record in records:
        counts[_expect_split(record)] += 1
    return dict(sorted(counts.items()))


def _reject_layered_prompt_fields(records: list[dict[str, object]]) -> None:
    for record in records:
        prompt_id = _expect_string(record, "prompt_id")
        layered_fields = sorted(LAYERED_ONLY_FIELDS & set(record))
        if layered_fields:
            raise ValueError(
                f"ordinary Qwen-Image prompt {prompt_id} must not include layered fields: "
                f"{layered_fields}"
            )


def _validate_capture_durable_paths(summary: dict[str, object]) -> None:
    durable_paths = _expect_mapping(summary, "durable_paths")
    for field_name in REQUIRED_CAPTURE_DURABLE_PATH_FIELDS:
        value = _expect_string(durable_paths, field_name)
        if value.startswith("/tmp"):
            raise ValueError(f"capture durable_paths.{field_name} must not be under /tmp")
    run_root = _expect_string(durable_paths, "run_root").rstrip("/")
    for field_name in ("bf16_references", "teacher_tuples", "capture_summary", "tuple_index"):
        value = _expect_string(durable_paths, field_name)
        if not _path_is_under(value, run_root):
            raise ValueError(f"capture durable_paths.{field_name} must be under durable run_root")


def _path_is_under(path: str, root: str) -> bool:
    normalized_root = root.rstrip("/")
    return path == normalized_root or path.startswith(f"{normalized_root}/")


def _contains_forbidden_runtime(value: object, *, key_name: str = "") -> bool:
    if "docker" in key_name.casefold():
        return True
    if isinstance(value, str):
        return "docker" in value.casefold()
    if isinstance(value, dict):
        return any(
            _contains_forbidden_runtime(child_value, key_name=str(child_key))
            for child_key, child_value in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_runtime(child_value) for child_value in value)
    return False


def _expect_mapping(value: dict[str, object], field_name: str) -> dict[str, object]:
    field = value.get(field_name)
    if not isinstance(field, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return field


def _expect_string(value: dict[str, object], field_name: str) -> str:
    field = value.get(field_name)
    if not isinstance(field, str) or not field:
        raise ValueError(f"{field_name} must be a non-empty string")
    return field


def _expect_split(record: dict[str, object]) -> CaptureSplit:
    split = _expect_string(record, "split")
    if split not in ("smoke", "fast_calibration", "main_calibration", "held_out"):
        raise ValueError(f"invalid capture split: {split}")
    return cast(CaptureSplit, split)


def _expect_string_list(value: dict[str, object], field_name: str) -> list[str]:
    field = value.get(field_name)
    if not isinstance(field, list) or not all(isinstance(item, str) for item in field):
        raise ValueError(f"{field_name} must be a list of strings")
    return field


def _expect_positive_int(value: dict[str, object], field_name: str) -> int:
    field = value.get(field_name)
    if not isinstance(field, int) or field <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return field


def _expect_non_negative_int(value: dict[str, object], field_name: str) -> int:
    field = value.get(field_name)
    if not isinstance(field, int) or field < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return field


def git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            cwd=project_root,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_capture_plan_command(args: argparse.Namespace) -> None:
    records = read_prompt_jsonl(Path(args.prompt_manifest_jsonl))
    prompt_summary = read_json(Path(args.prompt_summary_json))
    splits = tuple(args.splits) if args.splits else None
    summary, tuple_entries = build_capture_plan(
        records=records,
        prompt_summary=prompt_summary,
        prompt_manifest_path=Path(args.prompt_manifest_jsonl),
        capture_summary_path=Path(args.capture_summary_json),
        tuple_index_path=Path(args.tuple_index_jsonl),
        splits=splits,
        project_root=Path(args.project_root),
    )
    write_json(Path(args.capture_summary_json), summary)
    write_jsonl(Path(args.tuple_index_jsonl), tuple_entries)


def _validate_capture_plan_command(args: argparse.Namespace) -> None:
    records = read_prompt_jsonl(Path(args.prompt_manifest_jsonl))
    prompt_summary = read_json(Path(args.prompt_summary_json))
    capture_summary = read_json(Path(args.capture_summary_json))
    tuple_entries = read_tuple_index_jsonl(Path(args.tuple_index_jsonl))
    validate_capture_summary(
        capture_summary,
        records=records,
        prompt_summary=prompt_summary,
        tuple_entries=tuple_entries,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_parser = subparsers.add_parser("write-capture-plan")
    write_parser.add_argument("--prompt-manifest-jsonl", required=True)
    write_parser.add_argument("--prompt-summary-json", required=True)
    write_parser.add_argument("--capture-summary-json", required=True)
    write_parser.add_argument("--tuple-index-jsonl", required=True)
    write_parser.add_argument("--project-root", default=str(Path.cwd()))
    write_parser.add_argument(
        "--splits",
        nargs="+",
        choices=("smoke", "fast_calibration", "main_calibration", "held_out"),
    )
    write_parser.set_defaults(func=_write_capture_plan_command)

    validate_parser = subparsers.add_parser("validate-capture-plan")
    validate_parser.add_argument("--prompt-manifest-jsonl", required=True)
    validate_parser.add_argument("--prompt-summary-json", required=True)
    validate_parser.add_argument("--capture-summary-json", required=True)
    validate_parser.add_argument("--tuple-index-jsonl", required=True)
    validate_parser.set_defaults(func=_validate_capture_plan_command)

    return parser


def read_tuple_index_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        loaded = json.loads(line)
        if not isinstance(loaded, dict):
            raise ValueError(f"tuple index line {line_number} must be a JSON object")
        records.append(loaded)
    return records


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
