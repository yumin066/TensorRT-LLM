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

"""Build ordinary Qwen-Image all-layer MXFP8 block Linear coverage manifests."""

from __future__ import annotations

import argparse
import datetime
import importlib
import importlib.util
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from scripts.visualgen_eval.qwen_image_capture_manifest import (
    DEFAULT_CLUSTER_ALIAS,
    DEFAULT_CONTAINER_RUNTIME,
    DEFAULT_ENROOT_IMAGE,
    git_commit,
)
from scripts.visualgen_eval.qwen_image_teacher_capture import (
    cleanup_pipeline,
    load_single_worker_pipeline,
)

COVERAGE_MANIFEST_FORMAT = "qwen_image_block_linear_coverage_manifest_v1"
QWEN_BLOCK_LINEAR_POLICY = "qwen_block_linears_840"
QWEN_IMAGE_LAYER_COUNT = 60
QWEN_IMAGE_BLOCK_LINEAR_SUFFIXES = (
    "img_mod.1",
    "txt_mod.1",
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "img_mlp.up_proj",
    "img_mlp.down_proj",
    "txt_mlp.up_proj",
    "txt_mlp.down_proj",
)
QWEN_IMAGE_BLOCK_LINEAR_COUNT = len(QWEN_IMAGE_BLOCK_LINEAR_SUFFIXES)
QWEN_IMAGE_BLOCK_LINEAR_TARGET_COUNT = QWEN_IMAGE_LAYER_COUNT * QWEN_IMAGE_BLOCK_LINEAR_COUNT
QWEN_NON_BLOCK_LINEAR_EXCLUSIONS = ("img_in", "txt_in", "norm_out.linear", "proj_out")


@dataclass(frozen=True)
class BlockLinearCoverageRecord:
    """One Linear module recorded in the all-layer MXFP8 target manifest."""

    module_name: str
    normalized_name: str
    block_index: int | None
    role: str | None
    weight_shape: tuple[int, ...] | None
    weight_dtype: str | None
    is_target: bool
    is_non_block_exclusion: bool


def analyze_transformer_block_linear_coverage(
    transformer: nn.Module,
    *,
    expected_num_layers: int = QWEN_IMAGE_LAYER_COUNT,
    expected_target_count: int = QWEN_IMAGE_BLOCK_LINEAR_TARGET_COUNT,
    linear_cls: type[nn.Module] | None = None,
) -> dict[str, object]:
    """Analyze Qwen-Image block Linear target coverage and fail on contract drift."""
    linear_cls = linear_cls or _default_linear_cls()
    helper = _load_fake_mxfp8_helper()
    targets = helper.select_qwen_image_block_linears(
        transformer,
        expected_count=expected_target_count,
        linear_cls=linear_cls,
    )
    target_by_name = {target.normalized_name: target for target in targets}
    records = collect_linear_coverage_records(
        transformer,
        target_by_name=target_by_name,
        linear_cls=linear_cls,
    )
    report = build_coverage_report(
        records=records,
        expected_num_layers=expected_num_layers,
        expected_target_count=expected_target_count,
    )
    failures = report["failures"]
    if isinstance(failures, dict) and any(failures.values()):
        raise ValueError(_format_failures(failures))
    return report


def collect_linear_coverage_records(
    transformer: nn.Module,
    *,
    target_by_name: dict[str, Any],
    linear_cls: type[nn.Module],
) -> tuple[BlockLinearCoverageRecord, ...]:
    """Collect all Linear modules, marking Qwen block targets and non-block exclusions."""
    seen: set[str] = set()
    records: list[BlockLinearCoverageRecord] = []
    exclusion_set = set(QWEN_NON_BLOCK_LINEAR_EXCLUSIONS)
    helper = _load_fake_mxfp8_helper()
    for module_name, module in transformer.named_modules():
        if not module_name or not isinstance(module, linear_cls):
            continue
        normalized_name = helper.normalize_qwen_module_name(module_name)
        if normalized_name in seen:
            raise ValueError(f"Duplicate Linear module after normalization: {normalized_name}")
        seen.add(normalized_name)
        target = target_by_name.get(normalized_name)
        records.append(
            BlockLinearCoverageRecord(
                module_name=module_name,
                normalized_name=normalized_name,
                block_index=getattr(target, "block_index", None),
                role=getattr(target, "role", None),
                weight_shape=_tensor_shape(getattr(module, "weight", None)),
                weight_dtype=_tensor_dtype_name(getattr(module, "weight", None)),
                is_target=target is not None,
                is_non_block_exclusion=normalized_name in exclusion_set,
            )
        )
    return tuple(sorted(records, key=lambda record: record.normalized_name))


def build_coverage_report(
    *,
    records: tuple[BlockLinearCoverageRecord, ...],
    expected_num_layers: int,
    expected_target_count: int,
) -> dict[str, object]:
    """Build a JSON-serializable coverage manifest for ordinary Qwen-Image."""
    target_records = [record for record in records if record.is_target]
    target_names = {record.normalized_name for record in target_records}
    non_block_exclusions = [record for record in records if record.is_non_block_exclusion]
    non_block_exclusion_names = {record.normalized_name for record in non_block_exclusions}
    role_counts = {
        role: sum(record.role == role for record in target_records)
        for role in QWEN_IMAGE_BLOCK_LINEAR_SUFFIXES
    }
    observed_block_indices = sorted(
        {record.block_index for record in target_records if isinstance(record.block_index, int)}
    )
    all_names = {record.normalized_name for record in records}
    expected_exclusions = set(QWEN_NON_BLOCK_LINEAR_EXCLUSIONS)
    failures: dict[str, list[str]] = {
        "target_count": [],
        "missing_roles": [],
        "missing_block_indices": [],
        "target_exclusion_overlap": sorted(target_names & expected_exclusions),
        "missing_non_block_exclusions": sorted(expected_exclusions - non_block_exclusion_names),
        "unexpected_non_target_linears": sorted(all_names - target_names - expected_exclusions),
    }
    if len(target_records) != expected_target_count:
        failures["target_count"].append(
            f"found {len(target_records)} target Linears, expected {expected_target_count}"
        )
    for role, count in role_counts.items():
        if count != expected_num_layers:
            failures["missing_roles"].append(
                f"{role}: found {count}, expected {expected_num_layers}"
            )
    expected_block_indices = list(range(expected_num_layers))
    if observed_block_indices != expected_block_indices:
        failures["missing_block_indices"].append(
            f"found {observed_block_indices}, expected {expected_block_indices}"
        )

    return {
        "format": COVERAGE_MANIFEST_FORMAT,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "failed" if any(failures.values()) else "passed",
        "target_policy": QWEN_BLOCK_LINEAR_POLICY,
        "expected_num_layers": expected_num_layers,
        "expected_target_count": expected_target_count,
        "target_count": len(target_records),
        "target_roles": list(QWEN_IMAGE_BLOCK_LINEAR_SUFFIXES),
        "target_role_counts": role_counts,
        "target_block_indices": observed_block_indices,
        "non_block_exclusions": list(QWEN_NON_BLOCK_LINEAR_EXCLUSIONS),
        "non_block_exclusion_count": len(non_block_exclusions),
        "total_linear_count": len(records),
        "failures": failures,
        "records": [asdict(record) for record in records],
    }


def run_coverage_manifest(args: argparse.Namespace) -> dict[str, object]:
    """Load Qwen-Image and write the all-layer block Linear coverage manifest."""
    pipeline = load_single_worker_pipeline(
        model=args.model,
        visual_gen_args=Path(args.visual_gen_args),
        device=args.device,
    )
    try:
        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            raise ValueError("Loaded pipeline does not expose a transformer component")
        report = analyze_transformer_block_linear_coverage(
            transformer,
            expected_num_layers=args.expected_num_layers,
            expected_target_count=args.expected_target_count,
        )
        report["provenance"] = build_runtime_provenance(args)
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report
    finally:
        cleanup_pipeline(pipeline)


def build_runtime_provenance(args: argparse.Namespace) -> dict[str, object]:
    """Build provenance for the coverage manifest run."""
    return {
        "cluster_alias": args.cluster_alias,
        "allocation_id": args.allocation_id or os.environ.get("SSH_GW_ALLOC_ID"),
        "job_id": args.job_id or os.environ.get("SLURM_JOB_ID"),
        "node_list": args.node_list or os.environ.get("SLURM_NODELIST"),
        "container_runtime": DEFAULT_CONTAINER_RUNTIME,
        "enroot_image": args.enroot_image,
        "command": args.command or " ".join(sys.argv),
        "git_head": git_commit(Path.cwd()),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
        "model": args.model,
        "model_snapshot_path": args.model_snapshot_path,
        "visual_gen_args": str(args.visual_gen_args),
        "output_json": str(args.output_json),
    }


def _tensor_shape(value: object) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _tensor_dtype_name(value: object) -> str | None:
    dtype = getattr(value, "dtype", None)
    return None if dtype is None else str(dtype)


def _format_failures(failures: dict[str, list[str]]) -> str:
    parts = []
    for key, items in failures.items():
        if items:
            parts.append(f"{key}: {items[:8]}")
    return "Qwen-Image block Linear coverage failed; " + "; ".join(parts)


def _default_linear_cls() -> type[nn.Module]:
    from tensorrt_llm._torch.modules.linear import Linear

    return Linear


def _load_fake_mxfp8_helper() -> Any:
    module_name = "_qwen_image_fake_mxfp8"
    cached_module = sys.modules.get(module_name)
    if cached_module is not None:
        return cached_module
    helper_path = (
        Path(__file__).resolve().parents[2]
        / "tensorrt_llm/_torch/visual_gen/quantization/fake_mxfp8.py"
    )
    if helper_path.is_file():
        spec = importlib.util.spec_from_file_location(module_name, helper_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load fake MXFP8 helper from {helper_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module("tensorrt_llm._torch.visual_gen.quantization.fake_mxfp8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-gen-args", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-num-layers", type=int, default=QWEN_IMAGE_LAYER_COUNT)
    parser.add_argument(
        "--expected-target-count",
        type=int,
        default=QWEN_IMAGE_BLOCK_LINEAR_TARGET_COUNT,
    )
    parser.add_argument("--cluster-alias", default=DEFAULT_CLUSTER_ALIAS)
    parser.add_argument("--allocation-id")
    parser.add_argument("--job-id")
    parser.add_argument("--node-list")
    parser.add_argument("--enroot-image", default=DEFAULT_ENROOT_IMAGE)
    parser.add_argument("--model-snapshot-path")
    parser.add_argument("--command")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_coverage_manifest(args)


if __name__ == "__main__":
    main()
