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

"""Capture ordinary Qwen-Image BF16 references and teacher tuples."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterator, Mapping

from scripts.visualgen_eval.qwen_image_capture_manifest import (
    BF16_TEACHER_TRAJECTORY_SOURCE,
    CAPTURE_MANIFEST_FORMAT,
    DEFAULT_CLUSTER_ALIAS,
    DEFAULT_CONTAINER_RUNTIME,
    DEFAULT_ENROOT_IMAGE,
    FORBIDDEN_RUNTIME_TOKENS,
    TUPLE_INDEX_FORMAT,
    git_commit,
    read_tuple_index_jsonl,
    timestep_bin_name,
    validate_capture_summary,
    validate_prompt_handoff,
    write_json,
    write_jsonl,
)
from scripts.visualgen_eval.qwen_image_prompt_manifest import (
    DEFAULT_CFG_BRANCHES,
    REQUIRED_CAPTURE_FIELDS,
    read_json,
)
from scripts.visualgen_eval.qwen_image_prompt_manifest import read_jsonl as read_prompt_jsonl

CAPTURED_TUPLE_STATUS = "captured"
CAPTURED_STATUS = "captured"
ROLLOUT_METADATA_ENTRY_FORMAT = "qwen_image_rollout_metadata_entry_v1"
ROLLOUT_METADATA_SUMMARY_FORMAT = "qwen_image_rollout_metadata_manifest_summary_v1"
ROLLOUT_METADATA_DERIVATION_METHOD = "bf16_teacher_tuple_cfg_scheduler_step_v1"
BF16_TENSOR_FIELDS = (
    "hidden_states",
    "timestep",
    "encoder_hidden_states",
    "target_output",
)
CAPTURE_SPLIT_CHOICES = ("smoke", "fast_calibration", "main_calibration", "held_out")


def _load_torch() -> Any:
    return importlib.import_module("torch")


def _is_tensor(value: object, torch_module: Any) -> bool:
    tensor_type = getattr(torch_module, "Tensor", None)
    return tensor_type is not None and isinstance(value, tensor_type)


def tensor_to_cpu(value: Any, torch_module: Any | None = None) -> Any:
    """Detach tensors and move nested values to CPU for durable tuple storage."""
    torch_module = torch_module or _load_torch()
    if _is_tensor(value, torch_module):
        return value.detach().cpu()
    if isinstance(value, list):
        return [tensor_to_cpu(item, torch_module) for item in value]
    if isinstance(value, tuple):
        return tuple(tensor_to_cpu(item, torch_module) for item in value)
    if isinstance(value, dict):
        return {key: tensor_to_cpu(item, torch_module) for key, item in value.items()}
    return value


def first_tensor_output(output: Any, torch_module: Any | None = None) -> Any | None:
    """Return the first tensor-like output from a transformer forward result."""
    torch_module = torch_module or _load_torch()
    if _is_tensor(output, torch_module):
        return output
    if isinstance(output, (list, tuple)) and output and _is_tensor(output[0], torch_module):
        return output[0]
    sample = getattr(output, "sample", None)
    if _is_tensor(sample, torch_module):
        return sample
    return None


@contextmanager
def capture_transformer_teacher_tuples(
    pipeline: Any,
    *,
    record: dict[str, object],
    tuple_entries: list[dict[str, object]],
    torch_module: Any | None = None,
) -> Iterator[None]:
    """Wrap ``pipeline.transformer.forward`` and write one tuple per true-CFG call."""
    torch_module = torch_module or _load_torch()
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise ValueError("Pipeline does not expose a transformer for tuple capture")

    prompt_id = _expect_string(record, "prompt_id")
    split = _expect_string(record, "split")
    num_steps = _expect_positive_int(record, "num_inference_steps")
    cfg_branches = _expect_string_list(record, "cfg_branches")
    if tuple(cfg_branches) != DEFAULT_CFG_BRANCHES:
        raise ValueError(f"prompt {prompt_id} cfg_branches must be {list(DEFAULT_CFG_BRANCHES)}")

    entries_by_key = {
        (
            _expect_non_negative_int(entry, "timestep_index"),
            _expect_string(entry, "cfg_branch"),
        ): entry
        for entry in tuple_entries
    }
    expected_call_count = num_steps * len(cfg_branches)
    original_forward = transformer.forward
    call_index = 0

    def wrapped_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal call_index
        output = original_forward(*args, **kwargs)
        timestep_index = call_index // len(cfg_branches)
        branch_index = call_index % len(cfg_branches)
        if timestep_index >= num_steps:
            raise ValueError(f"captured too many transformer calls for prompt {prompt_id}")
        cfg_branch = cfg_branches[branch_index]
        tuple_entry = entries_by_key.get((timestep_index, cfg_branch))
        if tuple_entry is None:
            raise ValueError(
                f"missing tuple index entry for prompt {prompt_id}, "
                f"timestep {timestep_index}, branch {cfg_branch}"
            )
        tuple_path = Path(_expect_string(tuple_entry, "tuple_path"))
        tuple_path.parent.mkdir(parents=True, exist_ok=True)
        timestep_bin = timestep_bin_name(timestep_index, num_steps)
        target = first_tensor_output(output, torch_module)
        payload = build_teacher_tuple_payload(
            kwargs=kwargs,
            target_output=target,
            prompt_id=prompt_id,
            split=split,
            timestep_index=timestep_index,
            timestep_bin=timestep_bin,
            cfg_branch=cfg_branch,
            torch_module=torch_module,
        )
        torch_module.save(payload, tuple_path)
        tuple_entry["status"] = CAPTURED_TUPLE_STATUS
        tuple_entry["trajectory_source"] = BF16_TEACHER_TRAJECTORY_SOURCE
        call_index += 1
        return output

    transformer.forward = MethodType(wrapped_forward, transformer)
    try:
        yield
    finally:
        transformer.forward = original_forward
        if call_index != expected_call_count:
            raise ValueError(
                f"captured {call_index} transformer calls for prompt {prompt_id}, "
                f"expected {expected_call_count}"
            )


def build_teacher_tuple_payload(
    *,
    kwargs: dict[str, Any],
    target_output: Any | None,
    prompt_id: str,
    split: str,
    timestep_index: int,
    timestep_bin: str,
    cfg_branch: str,
    torch_module: Any | None = None,
) -> dict[str, object]:
    torch_module = torch_module or _load_torch()
    kwargs_cpu_raw = tensor_to_cpu(kwargs, torch_module)
    if not isinstance(kwargs_cpu_raw, dict):
        raise ValueError("transformer kwargs did not serialize to a mapping")
    kwargs_cpu = _with_derived_capture_fields(kwargs_cpu_raw, torch_module)
    payload: dict[str, object] = {}
    for field_name in REQUIRED_CAPTURE_FIELDS:
        if field_name == "target_output":
            if target_output is None:
                raise ValueError("transformer output did not contain a tensor target")
            payload[field_name] = tensor_to_cpu(target_output, torch_module)
        elif field_name == "prompt_id":
            payload[field_name] = prompt_id
        elif field_name == "timestep_index":
            payload[field_name] = timestep_index
        elif field_name == "cfg_branch":
            payload[field_name] = cfg_branch
        elif field_name == "timestep_bin":
            payload[field_name] = timestep_bin
        else:
            if field_name not in kwargs_cpu:
                raise ValueError(f"transformer kwargs missing required field: {field_name}")
            payload[field_name] = kwargs_cpu[field_name]
    payload["split"] = split
    payload["status"] = CAPTURED_TUPLE_STATUS
    payload["trajectory_source"] = BF16_TEACHER_TRAJECTORY_SOURCE
    return payload


def _with_derived_capture_fields(
    kwargs_cpu: dict[str, object], torch_module: Any
) -> dict[str, object]:
    normalized = dict(kwargs_cpu)
    if "txt_seq_lens" not in normalized:
        normalized["txt_seq_lens"] = _derive_txt_seq_lens(normalized, torch_module)
    return normalized


def _derive_txt_seq_lens(kwargs_cpu: dict[str, object], torch_module: Any) -> list[int]:
    lengths = _txt_seq_lens_from_mask(kwargs_cpu.get("encoder_hidden_states_mask"), torch_module)
    if lengths:
        return lengths
    lengths = _txt_seq_lens_from_encoder_hidden_states(
        kwargs_cpu.get("encoder_hidden_states"), torch_module
    )
    if lengths:
        return lengths
    raise ValueError(
        "transformer kwargs missing required field: txt_seq_lens and it could not be "
        "derived from encoder_hidden_states_mask or encoder_hidden_states"
    )


def _txt_seq_lens_from_mask(mask: object, torch_module: Any) -> list[int] | None:
    if mask is None:
        return None
    if _is_tensor(mask, torch_module):
        try:
            int_dtype = getattr(torch_module, "int64", None)
            mask_for_sum = mask.to(dtype=int_dtype) if int_dtype is not None else mask
            return _coerce_int_list(mask_for_sum.sum(dim=-1).tolist())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
    if isinstance(mask, (list, tuple)):
        return _txt_seq_lens_from_sequence_mask(mask)
    return None


def _txt_seq_lens_from_sequence_mask(mask: list[object] | tuple[object, ...]) -> list[int]:
    if not mask:
        return []
    if all(not isinstance(row, (list, tuple)) for row in mask):
        return [sum(1 for value in mask if bool(value))]
    lengths = []
    for row in mask:
        if not isinstance(row, (list, tuple)):
            return []
        lengths.append(sum(1 for value in row if bool(value)))
    return lengths


def _txt_seq_lens_from_encoder_hidden_states(
    encoder_hidden_states: object, torch_module: Any
) -> list[int] | None:
    if encoder_hidden_states is None or not _is_tensor(encoder_hidden_states, torch_module):
        return None
    shape = getattr(encoder_hidden_states, "shape", None)
    if shape is None:
        return None
    dims = tuple(int(dim) for dim in shape)
    if len(dims) >= 3:
        return [dims[1]] * dims[0]
    if len(dims) == 2:
        return [dims[0]]
    return None


def _coerce_int_list(value: object) -> list[int]:
    if isinstance(value, (int, float, bool)):
        return [int(value)]
    if isinstance(value, list):
        if all(isinstance(item, (int, float, bool)) for item in value):
            return [int(item) for item in value]
        if len(value) == 1 and isinstance(value[0], list):
            return _coerce_int_list(value[0])
    return []


def capture_record_with_pipeline(
    pipeline: Any,
    *,
    record: dict[str, object],
    tuple_entries: list[dict[str, object]],
    save_reference_fn: Callable[[Any, Path], None] | None = None,
    infer_fn: Callable[[Any, dict[str, object]], Any] | None = None,
    torch_module: Any | None = None,
) -> None:
    """Capture one prompt record with an already-loaded ordinary Qwen-Image pipeline."""
    prompt_id = _expect_string(record, "prompt_id")
    split = _expect_string(record, "split")
    reference_path = Path(_expect_string(tuple_entries[0], "reference_image_path"))
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    if not all(_expect_string(entry, "prompt_id") == prompt_id for entry in tuple_entries):
        raise ValueError("tuple_entries for capture_record_with_pipeline must share prompt_id")
    if not all(_expect_string(entry, "split") == split for entry in tuple_entries):
        raise ValueError("tuple_entries for capture_record_with_pipeline must share split")

    with capture_transformer_teacher_tuples(
        pipeline,
        record=record,
        tuple_entries=tuple_entries,
        torch_module=torch_module,
    ):
        output = (
            infer_fn(pipeline, record) if infer_fn is not None else infer_record(pipeline, record)
        )

    if save_reference_fn is None:
        save_reference_image(output, reference_path)
    else:
        save_reference_fn(output, reference_path)


def infer_record(pipeline: Any, record: dict[str, object]) -> Any:
    """Run one ordinary Qwen-Image prompt through a local in-process pipeline."""
    from tensorrt_llm._torch.visual_gen import DiffusionRequest

    params = build_visual_gen_params(record, local_pipeline_defaults(pipeline))
    req = DiffusionRequest(
        request_id=0,
        prompt=[_expect_string(record, "prompt")],
        params=params,
    )
    return pipeline.infer(req)


def build_visual_gen_params(record: dict[str, object], defaults: Any) -> Any:
    from tensorrt_llm.visual_gen.params import VisualGenParams

    params = (
        defaults.model_copy(deep=True) if hasattr(defaults, "model_copy") else VisualGenParams()
    )
    params.seed = _expect_positive_int(record, "seed")
    params.height = _expect_positive_int(record, "height")
    params.width = _expect_positive_int(record, "width")
    params.num_inference_steps = _expect_positive_int(record, "num_inference_steps")
    params.guidance_scale = float(_expect_number(record, "guidance_scale"))
    params.max_sequence_length = _expect_positive_int(record, "max_sequence_length")
    params.negative_prompt = _expect_string(record, "negative_prompt")
    params.num_images_per_prompt = 1
    return params


def local_pipeline_defaults(pipeline: Any) -> Any:
    from tensorrt_llm.visual_gen.params import VisualGenParams

    kwargs = dict(getattr(pipeline, "default_generation_params", {}))
    extra_specs = getattr(pipeline, "extra_param_specs", {})
    extra = {key: spec.default for key, spec in extra_specs.items()}
    if extra:
        kwargs["extra_params"] = extra
    return VisualGenParams(**kwargs)


def save_reference_image(output: Any, path: Path) -> None:
    from tensorrt_llm.visual_gen.output import VisualGenOutput

    visual_output = VisualGenOutput(
        request_id=0,
        image=getattr(output, "image", None),
        video=getattr(output, "video", None),
        audio=getattr(output, "audio", None),
        frame_rate=getattr(output, "frame_rate", None),
        audio_sample_rate=getattr(output, "audio_sample_rate", None),
    )
    saved = visual_output.save(path, format="png", frame_rate=1.0)
    if saved != path:
        raise ValueError(f"expected reference image save path {path}, got {saved}")


def build_captured_summary(
    planned_summary: dict[str, object],
    *,
    tuple_entries: list[dict[str, object]],
    provenance: dict[str, object],
    pipeline: Any | None = None,
) -> dict[str, object]:
    summary = dict(planned_summary)
    if provenance.get("git_head") is not None:
        summary["git_head"] = provenance["git_head"]
    summary["capture_status"] = CAPTURED_STATUS
    summary["task3_input_ready"] = True
    summary["captured_total_tuples"] = len(tuple_entries)
    summary["capture_provenance"] = dict(provenance)
    if pipeline is not None:
        summary["scheduler_provenance"] = collect_scheduler_provenance(pipeline)
    return summary


def validate_captured_artifacts(
    summary: dict[str, object],
    *,
    records: list[dict[str, object]],
    prompt_summary: dict[str, object],
    tuple_entries: list[dict[str, object]],
    torch_module: Any | None = None,
) -> None:
    """Validate captured ordinary Qwen-Image BF16 references and teacher tuple files."""
    torch_module = torch_module or _load_torch()
    if summary.get("format") != CAPTURE_MANIFEST_FORMAT:
        raise ValueError(f"capture summary format must be {CAPTURE_MANIFEST_FORMAT}")
    if summary.get("capture_status") != CAPTURED_STATUS:
        raise ValueError("captured summary capture_status must be captured")
    if summary.get("task3_input_ready") is not True:
        raise ValueError("captured summary task3_input_ready must be true")
    if _contains_forbidden_runtime(summary):
        raise ValueError("captured summary must not include Docker or no-enroot provenance")
    provenance = _expect_mapping(summary, "capture_provenance")
    if provenance.get("git_head") is None:
        raise ValueError("capture_provenance.git_head must be recorded")
    if summary.get("git_head") != provenance["git_head"]:
        raise ValueError("captured summary git_head must match capture_provenance.git_head")

    validate_prompt_handoff(
        records=records,
        prompt_summary=prompt_summary,
        prompt_manifest_path=Path(_expect_string(summary, "prompt_manifest_path")),
    )
    durable_paths = _expect_mapping(summary, "durable_paths")
    reference_root = _expect_string(durable_paths, "bf16_references")
    tuple_root = _expect_string(durable_paths, "teacher_tuples")
    selected_records = _selected_records_from_tuple_entries(records, tuple_entries)
    records_by_prompt = {_expect_string(record, "prompt_id"): record for record in selected_records}
    expected_keys = _expected_tuple_keys(records_by_prompt)
    observed_keys: set[tuple[str, int, str]] = set()
    reference_paths: set[str] = set()

    for entry in tuple_entries:
        if entry.get("format") != TUPLE_INDEX_FORMAT:
            raise ValueError(f"tuple entry format must be {TUPLE_INDEX_FORMAT}")
        prompt_id = _expect_string(entry, "prompt_id")
        record = records_by_prompt[prompt_id]
        tuple_id = _expect_string(entry, "tuple_id")
        split = _expect_string(entry, "split")
        if split != _expect_string(record, "split"):
            raise ValueError(f"tuple {tuple_id} split does not match prompt record split")
        timestep_index = _expect_non_negative_int(entry, "timestep_index")
        num_steps = _expect_positive_int(record, "num_inference_steps")
        if timestep_index >= num_steps:
            raise ValueError(f"tuple {tuple_id} timestep_index exceeds prompt num_inference_steps")
        cfg_branch = _expect_string(entry, "cfg_branch")
        if cfg_branch not in _expect_string_list(record, "cfg_branches"):
            raise ValueError(f"tuple {tuple_id} cfg_branch is not in prompt cfg_branches")
        tuple_key = (prompt_id, timestep_index, cfg_branch)
        observed_keys.add(tuple_key)
        if _expect_string(entry, "timestep_bin") != timestep_bin_name(timestep_index, num_steps):
            raise ValueError(f"tuple {tuple_id} timestep_bin does not match timestep_index")
        if _expect_string(entry, "status") != CAPTURED_TUPLE_STATUS:
            raise ValueError(f"tuple {tuple_id} status must be {CAPTURED_TUPLE_STATUS}")
        if _expect_string(entry, "trajectory_source") != BF16_TEACHER_TRAJECTORY_SOURCE:
            raise ValueError(
                f"tuple {tuple_id} trajectory_source must be {BF16_TEACHER_TRAJECTORY_SOURCE}"
            )

        reference_path = Path(_expect_string(entry, "reference_image_path"))
        tuple_path = Path(_expect_string(entry, "tuple_path"))
        _validate_path_under(
            reference_path, reference_root, f"tuple {tuple_id} reference_image_path"
        )
        _validate_path_under(tuple_path, tuple_root, f"tuple {tuple_id} tuple_path")
        if not reference_path.is_file():
            raise ValueError(f"tuple {tuple_id} reference image does not exist: {reference_path}")
        if not tuple_path.is_file():
            raise ValueError(f"tuple {tuple_id} file does not exist: {tuple_path}")
        reference_paths.add(str(reference_path))
        payload = _torch_load(tuple_path, torch_module)
        validate_teacher_tuple_payload(payload, entry=entry, torch_module=torch_module)

    missing_keys = expected_keys - observed_keys
    extra_keys = observed_keys - expected_keys
    if missing_keys or extra_keys:
        raise ValueError(
            "captured tuple key coverage does not match expected true CFG counts: "
            f"missing={sorted(missing_keys)}, extra={sorted(extra_keys)}"
        )
    if summary.get("captured_total_tuples") != len(tuple_entries):
        raise ValueError("captured summary captured_total_tuples does not match tuple index")
    if summary.get("prompt_count") != len(selected_records):
        raise ValueError("captured summary prompt_count does not match selected prompt records")
    if not reference_paths:
        raise ValueError("captured summary must include at least one reference image")


def validate_teacher_tuple_payload(
    payload: object,
    *,
    entry: dict[str, object],
    torch_module: Any | None = None,
) -> None:
    torch_module = torch_module or _load_torch()
    if not isinstance(payload, dict):
        raise ValueError("teacher tuple payload must be a mapping")
    missing = [field for field in REQUIRED_CAPTURE_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"teacher tuple payload missing required fields: {missing}")
    for field_name in BF16_TENSOR_FIELDS:
        value = payload[field_name]
        if not _is_tensor(value, torch_module):
            raise ValueError(f"teacher tuple field {field_name} must be a tensor")
        if not _is_bfloat16_tensor(value, torch_module):
            raise ValueError(f"teacher tuple field {field_name} must be bfloat16")
    for field_name in ("prompt_id", "timestep_index", "cfg_branch", "timestep_bin", "split"):
        if payload.get(field_name) != entry.get(field_name):
            raise ValueError(f"teacher tuple payload {field_name} does not match tuple index")
    if payload.get("trajectory_source") != BF16_TEACHER_TRAJECTORY_SOURCE:
        raise ValueError("teacher tuple payload trajectory_source must be bf16_teacher")


def write_rollout_metadata_from_captured_tuples(
    *,
    records: list[dict[str, object]],
    capture_summary: dict[str, object],
    tuple_entries: list[dict[str, object]],
    output_metadata_root: Path,
    output_metadata_jsonl: Path,
    summary_json: Path | None = None,
    provenance: Mapping[str, object] | None = None,
    torch_module: Any | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Derive rollout latent metadata from captured BF16 teacher tuples.

    Existing teacher tuples already contain the packed latent before the scheduler step
    as ``hidden_states`` plus BF16 cond/negative transformer outputs. This helper rebuilds
    the Qwen-Image true-CFG output and applies the same first-order FlowMatch/Euler
    scheduler step used by the rollout trainer, producing durable metadata that can be
    consumed by ``augment-rollout-tuples``.
    """
    torch_module = torch_module or _load_torch()
    runtime_provenance = dict(provenance or {})
    _validate_rollout_metadata_runtime_provenance(runtime_provenance)
    _validate_rollout_capture_summary(capture_summary)

    records_by_prompt = {_expect_string(record, "prompt_id"): record for record in records}
    grouped_entries = _group_rollout_tuple_entries(tuple_entries)
    scheduler = _rollout_scheduler_spec(capture_summary)
    scheduler_config_signature = _stable_hash(scheduler["scheduler_config"])
    sigmas = scheduler["sigmas"]
    sigmas_hash = _stable_hash(sigmas)

    metadata_records: list[dict[str, object]] = []
    output_metadata_root.mkdir(parents=True, exist_ok=True)
    for (prompt_id, timestep_index), branch_entries in sorted(grouped_entries.items()):
        record = records_by_prompt.get(prompt_id)
        if record is None:
            raise ValueError(f"rollout metadata tuple references unknown prompt_id {prompt_id}")
        cond_entry = branch_entries.get("cond")
        negative_entry = branch_entries.get("negative")
        if cond_entry is None or negative_entry is None:
            raise ValueError(
                "rollout metadata requires paired cond/negative tuple entries for "
                f"{prompt_id} timestep {timestep_index}"
            )
        cond_payload = _load_and_validate_tuple(cond_entry, torch_module)
        negative_payload = _load_and_validate_tuple(negative_entry, torch_module)
        latent_before = _expect_tensor_field(cond_payload, "hidden_states", torch_module)
        negative_latent_before = _expect_tensor_field(
            negative_payload, "hidden_states", torch_module
        )
        if not bool(torch_module.equal(latent_before, negative_latent_before)):
            raise ValueError(
                "rollout metadata requires cond/negative hidden_states to match for "
                f"{prompt_id} timestep {timestep_index}"
            )
        cond_output = _expect_tensor_field(cond_payload, "target_output", torch_module)
        negative_output = _expect_tensor_field(negative_payload, "target_output", torch_module)
        guidance_scale = float(_expect_number(record, "guidance_scale"))
        guided_output = _combine_qwen_image_cfg_outputs(
            cond_output,
            negative_output,
            guidance_scale=guidance_scale,
            torch_module=torch_module,
        )
        scheduler_state = _rollout_scheduler_state_for_step(
            scheduler=scheduler,
            timestep_index=timestep_index,
            scheduler_config_signature=scheduler_config_signature,
            sigmas_hash=sigmas_hash,
        )
        sigma_delta = float(scheduler_state["sigma_delta"])
        latent_after = (latent_before.float() + guided_output.float() * sigma_delta).to(
            dtype=latent_before.dtype
        )

        step_dir = output_metadata_root / _expect_string(cond_entry, "split") / prompt_id
        step_dir.mkdir(parents=True, exist_ok=True)
        latent_before_path = step_dir / f"step{timestep_index:03d}_latent_before.pt"
        latent_after_path = step_dir / f"step{timestep_index:03d}_latent_after.pt"
        torch_module.save(latent_before.detach().cpu(), latent_before_path)
        torch_module.save(latent_after.detach().cpu(), latent_after_path)

        for entry in (cond_entry, negative_entry):
            reference_image_path = _expect_string(entry, "reference_image_path")
            metadata_records.append(
                {
                    "format": ROLLOUT_METADATA_ENTRY_FORMAT,
                    "tuple_id": _expect_string(entry, "tuple_id"),
                    "prompt_id": prompt_id,
                    "split": _expect_string(entry, "split"),
                    "timestep_index": timestep_index,
                    "timestep_bin": _expect_string(entry, "timestep_bin"),
                    "cfg_branch": _expect_string(entry, "cfg_branch"),
                    "latent_before_step_path": str(latent_before_path),
                    "latent_after_step_path": str(latent_after_path),
                    "scheduler_state": dict(scheduler_state),
                    "rollout_provenance": _build_rollout_metadata_provenance(
                        record=record,
                        entry=entry,
                        capture_summary=capture_summary,
                        runtime_provenance=runtime_provenance,
                        scheduler_config_signature=scheduler_config_signature,
                        sigmas_hash=sigmas_hash,
                        reference_image_path=reference_image_path,
                    ),
                    "guidance_scale": guidance_scale,
                    "reference_image_path": reference_image_path,
                }
            )

    output_metadata_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_metadata_jsonl, metadata_records)
    summary = _build_rollout_metadata_summary(
        records=records,
        tuple_entries=tuple_entries,
        metadata_records=metadata_records,
        capture_summary=capture_summary,
        output_metadata_root=output_metadata_root,
        output_metadata_jsonl=output_metadata_jsonl,
        summary_json=summary_json,
        runtime_provenance=runtime_provenance,
        scheduler_config_signature=scheduler_config_signature,
        sigmas_hash=sigmas_hash,
    )
    if summary_json is not None:
        write_json(summary_json, summary)
    return metadata_records, summary


def _load_and_validate_tuple(entry: dict[str, object], torch_module: Any) -> dict[str, object]:
    tuple_path = Path(_expect_string(entry, "tuple_path"))
    payload = _torch_load(tuple_path, torch_module)
    validate_teacher_tuple_payload(payload, entry=entry, torch_module=torch_module)
    if not isinstance(payload, dict):
        raise ValueError(f"teacher tuple payload must be a mapping: {tuple_path}")
    return payload


def _expect_tensor_field(payload: dict[str, object], field_name: str, torch_module: Any) -> Any:
    value = payload.get(field_name)
    if not _is_tensor(value, torch_module):
        raise ValueError(f"teacher tuple field {field_name} must be a tensor")
    return value


def _group_rollout_tuple_entries(
    tuple_entries: list[dict[str, object]],
) -> dict[tuple[str, int], dict[str, dict[str, object]]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, object]]] = defaultdict(dict)
    for entry in tuple_entries:
        if _expect_string(entry, "status") != CAPTURED_TUPLE_STATUS:
            raise ValueError("rollout metadata requires captured tuple entries")
        prompt_id = _expect_string(entry, "prompt_id")
        timestep_index = _expect_non_negative_int(entry, "timestep_index")
        cfg_branch = _expect_string(entry, "cfg_branch")
        if cfg_branch not in DEFAULT_CFG_BRANCHES:
            raise ValueError(f"unsupported cfg_branch for rollout metadata: {cfg_branch}")
        branch_entries = grouped[(prompt_id, timestep_index)]
        if cfg_branch in branch_entries:
            raise ValueError(
                "duplicate rollout metadata tuple entry for "
                f"{prompt_id} timestep {timestep_index} branch {cfg_branch}"
            )
        branch_entries[cfg_branch] = entry
    if not grouped:
        raise ValueError("rollout metadata requires at least one tuple entry")
    return grouped


def _rollout_scheduler_spec(capture_summary: dict[str, object]) -> dict[str, object]:
    scheduler_provenance = _capture_scheduler_provenance(capture_summary)
    scheduler_config = _expect_mapping(scheduler_provenance, "scheduler_config")
    scheduler_class = _expect_string(scheduler_provenance, "scheduler_class")
    num_train_timesteps = int(scheduler_config.get("num_train_timesteps", 1000))
    timesteps = _expect_number_list(scheduler_provenance, "timesteps")
    if len(timesteps) != _expect_positive_int(capture_summary, "num_inference_steps"):
        raise ValueError("scheduler timesteps length must match num_inference_steps")
    sigmas = [float(timestep) / float(num_train_timesteps) for timestep in timesteps]
    if not sigmas:
        raise ValueError("scheduler timesteps must not be empty")
    if any(left < right for left, right in zip(sigmas, sigmas[1:])):
        raise ValueError("scheduler sigmas must be monotonically non-increasing")
    return {
        "scheduler_class": scheduler_class,
        "scheduler_config": scheduler_config,
        "num_train_timesteps": num_train_timesteps,
        "timesteps": timesteps,
        "sigmas": sigmas,
    }


def _capture_scheduler_provenance(capture_summary: dict[str, object]) -> dict[str, object]:
    scheduler = capture_summary.get("scheduler_provenance")
    if isinstance(scheduler, dict):
        return scheduler
    capture_provenance = capture_summary.get("capture_provenance")
    if isinstance(capture_provenance, dict) and isinstance(
        capture_provenance.get("scheduler"), dict
    ):
        return capture_provenance["scheduler"]
    raise ValueError("capture summary missing scheduler_provenance")


def _rollout_scheduler_state_for_step(
    *,
    scheduler: dict[str, object],
    timestep_index: int,
    scheduler_config_signature: str,
    sigmas_hash: str,
) -> dict[str, object]:
    timesteps = scheduler["timesteps"]
    sigmas = scheduler["sigmas"]
    if not isinstance(timesteps, list) or not isinstance(sigmas, list):
        raise ValueError("scheduler spec timesteps/sigmas must be lists")
    if timestep_index >= len(sigmas):
        raise ValueError(f"timestep_index exceeds scheduler sigmas: {timestep_index}")
    sigma = float(sigmas[timestep_index])
    next_sigma = float(sigmas[timestep_index + 1]) if timestep_index + 1 < len(sigmas) else 0.0
    return {
        "scheduler_class": str(scheduler["scheduler_class"]),
        "timestep": float(timesteps[timestep_index]),
        "step_index": timestep_index,
        "timestep_index": timestep_index,
        "num_inference_steps": len(sigmas),
        "num_train_timesteps": int(scheduler["num_train_timesteps"]),
        "sigma": sigma,
        "next_sigma": next_sigma,
        "sigma_next": next_sigma,
        "sigma_delta": next_sigma - sigma,
        "dt": next_sigma - sigma,
        "order": 1,
        "begin_index": 0,
        "scheduler_config_signature": scheduler_config_signature,
        "sigmas_hash": sigmas_hash,
        "stochastic_sampling": False,
        "per_token_sigmas": False,
        "per_token_timesteps": False,
    }


def _combine_qwen_image_cfg_outputs(
    cond_output: Any,
    negative_output: Any,
    *,
    guidance_scale: float,
    torch_module: Any,
) -> Any:
    if tuple(cond_output.shape) != tuple(negative_output.shape):
        raise ValueError("cond and negative target_output tensors must have matching shapes")
    guided = negative_output.float() + float(guidance_scale) * (
        cond_output.float() - negative_output.float()
    )
    cond_norm = torch_module.norm(cond_output.float(), dim=-1, keepdim=True)
    guided_norm = torch_module.norm(guided, dim=-1, keepdim=True).clamp_min(1.0e-8)
    return guided * (cond_norm / guided_norm)


def _build_rollout_metadata_provenance(
    *,
    record: dict[str, object],
    entry: dict[str, object],
    capture_summary: dict[str, object],
    runtime_provenance: dict[str, object],
    scheduler_config_signature: str,
    sigmas_hash: str,
    reference_image_path: str,
) -> dict[str, object]:
    prompt_id = _expect_string(record, "prompt_id")
    return {
        "derivation_method": ROLLOUT_METADATA_DERIVATION_METHOD,
        "scheduler_config_signature": scheduler_config_signature,
        "sigmas_hash": sigmas_hash,
        "git_head": _expect_string(runtime_provenance, "git_head"),
        "prompt_id": prompt_id,
        "seed": _expect_positive_int(record, "seed"),
        "height": _expect_positive_int(record, "height"),
        "width": _expect_positive_int(record, "width"),
        "num_inference_steps": _expect_positive_int(record, "num_inference_steps"),
        "guidance_scale": float(_expect_number(record, "guidance_scale")),
        "reference_image_path": reference_image_path,
        "source_tuple_id": _expect_string(entry, "tuple_id"),
        "source_capture_git_head": capture_summary.get("git_head"),
        "source_capture_status": capture_summary.get("capture_status"),
        "closed_set_target_manifest": runtime_provenance.get("closed_set_target_manifest"),
        "scheduler_metadata_source": runtime_provenance.get("scheduler_metadata_source"),
        "cfg_normalize": True,
        "scheduler_state_equivalent": False,
    }


def _build_rollout_metadata_summary(
    *,
    records: list[dict[str, object]],
    tuple_entries: list[dict[str, object]],
    metadata_records: list[dict[str, object]],
    capture_summary: dict[str, object],
    output_metadata_root: Path,
    output_metadata_jsonl: Path,
    summary_json: Path | None,
    runtime_provenance: dict[str, object],
    scheduler_config_signature: str,
    sigmas_hash: str,
) -> dict[str, object]:
    prompt_ids = sorted({_expect_string(entry, "prompt_id") for entry in tuple_entries})
    return {
        "format": ROLLOUT_METADATA_SUMMARY_FORMAT,
        "derivation_method": ROLLOUT_METADATA_DERIVATION_METHOD,
        "prompt_count": len(prompt_ids),
        "tuple_count": len(tuple_entries),
        "metadata_record_count": len(metadata_records),
        "prompt_ids": prompt_ids,
        "source_capture_summary_status": capture_summary.get("capture_status"),
        "source_capture_git_head": capture_summary.get("git_head"),
        "output_metadata_root": str(output_metadata_root),
        "output_metadata_jsonl": str(output_metadata_jsonl),
        "summary_json": str(summary_json) if summary_json is not None else None,
        "scheduler_config_signature": scheduler_config_signature,
        "sigmas_hash": sigmas_hash,
        "runtime_provenance": dict(runtime_provenance),
        "record_prompt_count": len(records),
    }


def _validate_rollout_capture_summary(capture_summary: dict[str, object]) -> None:
    if capture_summary.get("format") != CAPTURE_MANIFEST_FORMAT:
        raise ValueError(f"capture summary format must be {CAPTURE_MANIFEST_FORMAT}")
    if capture_summary.get("capture_status") != CAPTURED_STATUS:
        raise ValueError("rollout metadata requires a captured teacher capture summary")
    if _contains_forbidden_runtime(capture_summary):
        raise ValueError("rollout metadata capture summary must not include Docker/no-enroot")
    _rollout_scheduler_spec(capture_summary)


def _validate_rollout_metadata_runtime_provenance(
    provenance: dict[str, object],
) -> None:
    required_fields = (
        "cluster_alias",
        "container_runtime",
        "enroot_image",
        "command",
        "model_snapshot_path",
        "git_head",
        "closed_set_target_manifest",
        "scheduler_metadata_source",
        "reference_image_root",
        "node_list",
    )
    missing = [
        field_name for field_name in required_fields if provenance.get(field_name) in (None, "")
    ]
    if missing:
        raise ValueError(f"rollout metadata provenance missing required fields {missing}")
    if provenance["cluster_alias"] != DEFAULT_CLUSTER_ALIAS:
        raise ValueError(
            f"rollout metadata provenance cluster_alias must be {DEFAULT_CLUSTER_ALIAS}"
        )
    if provenance["container_runtime"] != DEFAULT_CONTAINER_RUNTIME:
        raise ValueError(
            f"rollout metadata provenance container_runtime must be {DEFAULT_CONTAINER_RUNTIME}"
        )
    if provenance["enroot_image"] != DEFAULT_ENROOT_IMAGE:
        raise ValueError(f"rollout metadata provenance enroot_image must be {DEFAULT_ENROOT_IMAGE}")
    if provenance.get("allocation_id") in (None, "") and provenance.get("job_id") in (None, ""):
        raise ValueError("rollout metadata provenance requires allocation_id or job_id")
    if _contains_forbidden_runtime(provenance):
        raise ValueError("rollout metadata provenance must not include Docker/no-enroot")


def _expect_number_list(value: dict[str, object], field_name: str) -> list[float]:
    field = value.get(field_name)
    if not isinstance(field, list) or not all(isinstance(item, (int, float)) for item in field):
        raise ValueError(f"{field_name} must be a list of numbers")
    return [float(item) for item in field]


def _stable_hash(value: object) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_bfloat16_tensor(value: object, torch_module: Any) -> bool:
    dtype = getattr(value, "dtype", None)
    bfloat16 = getattr(torch_module, "bfloat16", None)
    return dtype == bfloat16 or str(dtype) in ("bfloat16", "torch.bfloat16")


def _torch_load(path: Path, torch_module: Any) -> object:
    try:
        return torch_module.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch_module.load(path, map_location="cpu")


def capture_records_with_pipeline(
    pipeline: Any,
    *,
    records: list[dict[str, object]],
    tuple_entries: list[dict[str, object]],
    save_reference_fn: Callable[[Any, Path], None] | None = None,
    infer_fn: Callable[[Any, dict[str, object]], Any] | None = None,
    torch_module: Any | None = None,
) -> None:
    entries_by_prompt: dict[str, list[dict[str, object]]] = defaultdict(list)
    for entry in tuple_entries:
        entries_by_prompt[_expect_string(entry, "prompt_id")].append(entry)
    for record in records:
        prompt_id = _expect_string(record, "prompt_id")
        capture_record_with_pipeline(
            pipeline,
            record=record,
            tuple_entries=entries_by_prompt[prompt_id],
            save_reference_fn=save_reference_fn,
            infer_fn=infer_fn,
            torch_module=torch_module,
        )


def load_single_worker_pipeline(
    *,
    model: str,
    visual_gen_args: Path,
    device: str,
) -> Any:
    from tensorrt_llm._torch.visual_gen import PipelineLoader
    from tensorrt_llm.visual_gen import VisualGenArgs

    args = VisualGenArgs.from_yaml(visual_gen_args, model=model)
    if args.parallel_config.n_workers != 1:
        raise ValueError("teacher capture requires parallel_config.n_workers == 1")
    loader = PipelineLoader(args, device=device)
    return loader.load(skip_warmup=args.compilation_config.skip_warmup)


def cleanup_pipeline(pipeline: Any) -> None:
    cleanup = getattr(pipeline, "cleanup", None)
    if cleanup is not None:
        cleanup()


def collect_scheduler_provenance(pipeline: Any) -> dict[str, object]:
    scheduler = getattr(pipeline, "scheduler", None)
    if scheduler is None:
        return {}
    config = getattr(scheduler, "config", {})
    timesteps = getattr(scheduler, "timesteps", None)
    return {
        "scheduler_class": scheduler.__class__.__name__,
        "scheduler_config": _jsonable(config),
        "timesteps": _jsonable(timesteps),
    }


def build_runtime_provenance(
    args: argparse.Namespace, pipeline: Any | None = None
) -> dict[str, object]:
    provenance = {
        "cluster_alias": args.cluster_alias,
        "allocation_id": args.allocation_id or os.environ.get("SSH_GW_ALLOC_ID"),
        "job_id": args.job_id or os.environ.get("SLURM_JOB_ID"),
        "node_list": args.node_list or os.environ.get("SLURM_NODELIST"),
        "container_runtime": DEFAULT_CONTAINER_RUNTIME,
        "enroot_image": args.enroot_image,
        "command": args.command or " ".join(sys.argv),
        "model_snapshot_path": args.model_snapshot_path,
        "visual_gen_args": str(args.visual_gen_args),
        "git_head": git_commit(Path.cwd()),
    }
    if pipeline is not None:
        provenance["scheduler"] = collect_scheduler_provenance(pipeline)
    return provenance


def build_rollout_metadata_runtime_provenance(
    args: argparse.Namespace,
    *,
    capture_summary: dict[str, object],
) -> dict[str, object]:
    durable_paths = _expect_mapping(capture_summary, "durable_paths")
    return {
        "cluster_alias": args.cluster_alias,
        "allocation_id": args.allocation_id or os.environ.get("SSH_GW_ALLOC_ID"),
        "job_id": args.job_id or os.environ.get("SLURM_JOB_ID"),
        "node_list": args.node_list or os.environ.get("SLURM_NODELIST"),
        "container_runtime": DEFAULT_CONTAINER_RUNTIME,
        "enroot_image": args.enroot_image,
        "command": args.command or " ".join(sys.argv),
        "model_snapshot_path": args.model_snapshot_path,
        "git_head": git_commit(Path.cwd()),
        "closed_set_target_manifest": str(args.closed_set_target_manifest),
        "scheduler_metadata_source": str(args.capture_summary_json),
        "reference_image_root": _expect_string(durable_paths, "bf16_references"),
    }


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
        return _jsonable(value)
    return repr(value)


def _selected_records(
    records: list[dict[str, object]],
    *,
    split: str,
) -> list[dict[str, object]]:
    selected = [record for record in records if _expect_string(record, "split") == split]
    if not selected:
        raise ValueError(f"no prompt records selected for split {split}")
    return selected


def _selected_tuple_entries(
    tuple_entries: list[dict[str, object]],
    selected_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    prompt_ids = {_expect_string(record, "prompt_id") for record in selected_records}
    selected = [
        entry for entry in tuple_entries if _expect_string(entry, "prompt_id") in prompt_ids
    ]
    if not selected:
        raise ValueError("no tuple entries selected for prompt records")
    return selected


def _validate_split_scoped_plan(
    capture_summary: dict[str, object],
    *,
    selected_records: list[dict[str, object]],
    selected_entries: list[dict[str, object]],
    split: str,
) -> None:
    if capture_summary.get("prompt_count") != len(selected_records):
        raise ValueError(
            "capture summary prompt_count must match selected split prompt records; "
            "build a split-scoped capture plan first"
        )
    if capture_summary.get("expected_total_tuples") != len(selected_entries):
        raise ValueError(
            "capture summary expected_total_tuples must match selected split tuple entries; "
            "build a split-scoped capture plan first"
        )
    entry_splits = {_expect_string(entry, "split") for entry in selected_entries}
    if entry_splits != {split}:
        raise ValueError(f"tuple entries must contain only split {split}")


def _selected_records_from_tuple_entries(
    records: list[dict[str, object]],
    tuple_entries: list[dict[str, object]],
) -> list[dict[str, object]]:
    prompt_ids = {_expect_string(entry, "prompt_id") for entry in tuple_entries}
    records_by_prompt = {_expect_string(record, "prompt_id"): record for record in records}
    missing = prompt_ids - set(records_by_prompt)
    if missing:
        raise ValueError(f"tuple entries reference unknown prompt ids: {sorted(missing)}")
    return [record for record in records if _expect_string(record, "prompt_id") in prompt_ids]


def _expected_tuple_keys(
    records_by_prompt: dict[str, dict[str, object]],
) -> set[tuple[str, int, str]]:
    keys: set[tuple[str, int, str]] = set()
    for prompt_id, record in records_by_prompt.items():
        num_steps = _expect_positive_int(record, "num_inference_steps")
        for timestep_index in range(num_steps):
            for cfg_branch in _expect_string_list(record, "cfg_branches"):
                keys.add((prompt_id, timestep_index, cfg_branch))
    return keys


def _validate_path_under(path: Path, root: str, field_name: str) -> None:
    path_text = str(path)
    root_text = root.rstrip("/")
    if path_text.startswith("/tmp"):
        raise ValueError(f"{field_name} must not be under /tmp")
    if not (path_text == root_text or path_text.startswith(f"{root_text}/")):
        raise ValueError(f"{field_name} must be under {root_text}")


def _contains_forbidden_runtime(value: object, *, key_name: str = "") -> bool:
    if any(token in key_name.casefold() for token in FORBIDDEN_RUNTIME_TOKENS):
        return True
    if isinstance(value, str):
        return any(token in value.casefold() for token in FORBIDDEN_RUNTIME_TOKENS)
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


def _expect_number(value: dict[str, object], field_name: str) -> int | float:
    field = value.get(field_name)
    if not isinstance(field, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    return field


def _capture_split_command(args: argparse.Namespace) -> None:
    split = _expect_string(vars(args), "split")
    records = read_prompt_jsonl(Path(args.prompt_manifest_jsonl))
    prompt_summary = read_json(Path(args.prompt_summary_json))
    capture_summary = read_json(Path(args.capture_summary_json))
    tuple_entries = read_tuple_index_jsonl(Path(args.tuple_index_jsonl))
    validate_capture_summary(
        capture_summary,
        records=records,
        prompt_summary=prompt_summary,
        tuple_entries=tuple_entries,
        capture_summary_path=Path(args.capture_summary_json),
        tuple_index_path=Path(args.tuple_index_jsonl),
    )
    selected_records = _selected_records(records, split=split)
    selected_entries = _selected_tuple_entries(tuple_entries, selected_records)
    _validate_split_scoped_plan(
        capture_summary,
        selected_records=selected_records,
        selected_entries=selected_entries,
        split=split,
    )
    pipeline = load_single_worker_pipeline(
        model=args.model or _expect_string(capture_summary, "model"),
        visual_gen_args=Path(args.visual_gen_args),
        device=args.device,
    )
    try:
        capture_records_with_pipeline(
            pipeline,
            records=selected_records,
            tuple_entries=selected_entries,
        )
        captured_summary = build_captured_summary(
            capture_summary,
            tuple_entries=selected_entries,
            provenance=build_runtime_provenance(args, pipeline=pipeline),
            pipeline=pipeline,
        )
        validate_captured_artifacts(
            captured_summary,
            records=records,
            prompt_summary=prompt_summary,
            tuple_entries=selected_entries,
        )
        write_json(Path(args.output_capture_summary_json), captured_summary)
        write_jsonl(Path(args.output_tuple_index_jsonl), selected_entries)
    finally:
        cleanup_pipeline(pipeline)


def _capture_smoke_command(args: argparse.Namespace) -> None:
    args.split = "smoke"
    _capture_split_command(args)


def _validate_captured_command(args: argparse.Namespace) -> None:
    records = read_prompt_jsonl(Path(args.prompt_manifest_jsonl))
    prompt_summary = read_json(Path(args.prompt_summary_json))
    capture_summary = read_json(Path(args.capture_summary_json))
    tuple_entries = read_tuple_index_jsonl(Path(args.tuple_index_jsonl))
    validate_captured_artifacts(
        capture_summary,
        records=records,
        prompt_summary=prompt_summary,
        tuple_entries=tuple_entries,
    )


def _generate_rollout_metadata_command(args: argparse.Namespace) -> None:
    records = read_rollout_metadata_prompt_jsonl(Path(args.prompt_manifest_jsonl))
    capture_summary = read_json(Path(args.capture_summary_json))
    tuple_entries = read_tuple_index_jsonl(Path(args.tuple_index_jsonl))
    provenance = build_rollout_metadata_runtime_provenance(
        args,
        capture_summary=capture_summary,
    )
    write_rollout_metadata_from_captured_tuples(
        records=records,
        capture_summary=capture_summary,
        tuple_entries=tuple_entries,
        output_metadata_root=Path(args.output_metadata_root),
        output_metadata_jsonl=Path(args.output_metadata_jsonl),
        summary_json=Path(args.summary_json),
        provenance=provenance,
    )


def read_rollout_metadata_prompt_jsonl(path: Path) -> list[dict[str, object]]:
    """Read closed-set prompt records without requiring calibration split names."""
    records: list[dict[str, object]] = []
    prompt_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"prompt manifest line {line_number} must be a mapping")
        prompt_id = _expect_string(record, "prompt_id")
        if prompt_id in prompt_ids:
            raise ValueError(
                f"duplicate prompt_id in rollout metadata prompt manifest: {prompt_id}"
            )
        prompt_ids.add(prompt_id)
        for field_name in (
            "seed",
            "height",
            "width",
            "num_inference_steps",
            "guidance_scale",
        ):
            if record.get(field_name) is None:
                raise ValueError(
                    "rollout metadata prompt manifest missing required field "
                    f"{field_name} for prompt_id {prompt_id}"
                )
        records.append(record)
    if not records:
        raise ValueError(f"rollout metadata prompt manifest is empty: {path}")
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_split_parser = subparsers.add_parser("capture-split")
    capture_split_parser.add_argument("--split", required=True, choices=CAPTURE_SPLIT_CHOICES)
    _add_capture_parser_arguments(capture_split_parser)
    capture_split_parser.set_defaults(func=_capture_split_command)

    capture_parser = subparsers.add_parser("capture-smoke")
    _add_capture_parser_arguments(capture_parser)
    capture_parser.set_defaults(func=_capture_smoke_command)

    validate_parser = subparsers.add_parser("validate-captured")
    validate_parser.add_argument("--prompt-manifest-jsonl", required=True)
    validate_parser.add_argument("--prompt-summary-json", required=True)
    validate_parser.add_argument("--capture-summary-json", required=True)
    validate_parser.add_argument("--tuple-index-jsonl", required=True)
    validate_parser.set_defaults(func=_validate_captured_command)

    rollout_metadata_parser = subparsers.add_parser("generate-rollout-metadata")
    rollout_metadata_parser.add_argument("--prompt-manifest-jsonl", required=True)
    rollout_metadata_parser.add_argument("--capture-summary-json", required=True)
    rollout_metadata_parser.add_argument("--tuple-index-jsonl", required=True)
    rollout_metadata_parser.add_argument("--output-metadata-root", required=True)
    rollout_metadata_parser.add_argument("--output-metadata-jsonl", required=True)
    rollout_metadata_parser.add_argument("--summary-json", required=True)
    rollout_metadata_parser.add_argument("--closed-set-target-manifest", required=True)
    rollout_metadata_parser.add_argument("--cluster-alias", default=DEFAULT_CLUSTER_ALIAS)
    rollout_metadata_parser.add_argument("--allocation-id")
    rollout_metadata_parser.add_argument("--job-id")
    rollout_metadata_parser.add_argument("--node-list")
    rollout_metadata_parser.add_argument("--enroot-image", default=DEFAULT_ENROOT_IMAGE)
    rollout_metadata_parser.add_argument("--model-snapshot-path", required=True)
    rollout_metadata_parser.add_argument("--command")
    rollout_metadata_parser.set_defaults(func=_generate_rollout_metadata_command)
    return parser


def _add_capture_parser_arguments(capture_parser: argparse.ArgumentParser) -> None:
    capture_parser.add_argument("--prompt-manifest-jsonl", required=True)
    capture_parser.add_argument("--prompt-summary-json", required=True)
    capture_parser.add_argument("--capture-summary-json", required=True)
    capture_parser.add_argument("--tuple-index-jsonl", required=True)
    capture_parser.add_argument("--output-capture-summary-json", required=True)
    capture_parser.add_argument("--output-tuple-index-jsonl", required=True)
    capture_parser.add_argument("--visual-gen-args", required=True)
    capture_parser.add_argument("--model")
    capture_parser.add_argument("--device", default="cuda:0")
    capture_parser.add_argument("--cluster-alias", default=DEFAULT_CLUSTER_ALIAS)
    capture_parser.add_argument("--allocation-id")
    capture_parser.add_argument("--job-id")
    capture_parser.add_argument("--node-list")
    capture_parser.add_argument("--enroot-image", default=DEFAULT_ENROOT_IMAGE)
    capture_parser.add_argument("--model-snapshot-path")
    capture_parser.add_argument("--command")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
