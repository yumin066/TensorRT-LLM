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

"""Evaluate ordinary Qwen-Image RGB outputs against BF16 reference images."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from scripts.visualgen_eval.qwen_image_capture_manifest import (
    DEFAULT_CLUSTER_ALIAS,
    DEFAULT_CONTAINER_RUNTIME,
    DEFAULT_ENROOT_IMAGE,
    git_commit,
    write_json,
)
from scripts.visualgen_eval.qwen_image_prompt_manifest import read_jsonl as read_prompt_jsonl
from scripts.visualgen_eval.qwen_image_teacher_capture import (
    cleanup_pipeline,
    infer_record,
    load_single_worker_pipeline,
    save_reference_image,
)

RGB_METRICS_FORMAT = "qwen_image_rgb_split_metrics_v1"
SPLIT_CHOICES = ("smoke", "fast_calibration", "main_calibration", "held_out")


def compute_rgb_image_metrics(candidate_path: Path, reference_path: Path) -> dict[str, float]:
    candidate_shape, candidate = _load_rgb_unit_image(candidate_path)
    reference_shape, reference = _load_rgb_unit_image(reference_path)
    if candidate_shape != reference_shape:
        raise ValueError(
            f"RGB image shape mismatch: {candidate_path} has {candidate_shape}, "
            f"{reference_path} has {reference_shape}"
        )
    return compute_rgb_values_metrics(candidate, reference)


def compute_rgb_values_metrics(
    candidate: Sequence[float], reference: Sequence[float]
) -> dict[str, float]:
    if len(candidate) != len(reference):
        raise ValueError(
            f"RGB value count mismatch: candidate has {len(candidate)}, reference has {len(reference)}"
        )
    if not candidate:
        raise ValueError("RGB metric input must not be empty")
    mse = sum(
        (candidate_value - reference_value) ** 2
        for candidate_value, reference_value in zip(candidate, reference)
    ) / len(candidate)
    psnr = float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
    ssim = _global_ssim(candidate, reference)
    return {"mse": mse, "psnr": psnr, "ssim": ssim}


def run_qwen_image_rgb_split_eval(
    *,
    records: list[dict[str, object]],
    split: str,
    model: str,
    visual_gen_args: Path,
    reference_root: Path,
    output_root: Path,
    metrics_json: Path,
    variant: str,
    reference_variant: str,
    device: str,
    provenance: dict[str, object],
    config_metadata: dict[str, object] | None = None,
    pipeline: Any | None = None,
    infer_fn: Callable[[Any, dict[str, object]], Any] = infer_record,
    save_fn: Callable[[Any, Path], None] = save_reference_image,
    metrics_fn: Callable[[Path, Path], dict[str, float]] = compute_rgb_image_metrics,
) -> dict[str, object]:
    selected_records = _selected_records(records, split=split)
    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    owns_pipeline = pipeline is None
    if pipeline is None:
        pipeline = load_single_worker_pipeline(
            model=model,
            visual_gen_args=visual_gen_args,
            device=device,
        )
    comparisons: list[dict[str, object]] = []
    try:
        for record in selected_records:
            prompt_id = _expect_string(record, "prompt_id")
            candidate_path = output_dir / f"{prompt_id}.png"
            reference_path = reference_root / split / f"{prompt_id}.png"
            if not reference_path.is_file():
                raise ValueError(f"missing BF16 reference image for {prompt_id}: {reference_path}")
            output = infer_fn(pipeline, record)
            save_fn(output, candidate_path)
            metrics = metrics_fn(candidate_path, reference_path)
            comparisons.append(
                {
                    "prompt_id": prompt_id,
                    "split": split,
                    "variant": variant,
                    "reference_variant": reference_variant,
                    "candidate_image": str(candidate_path),
                    "reference_image": str(reference_path),
                    "mse": _json_metric(metrics["mse"]),
                    "psnr": _json_metric(metrics["psnr"]),
                    "ssim": _json_metric(metrics["ssim"]),
                    "seed": record.get("seed"),
                    "height": record.get("height"),
                    "width": record.get("width"),
                    "num_inference_steps": record.get("num_inference_steps"),
                    "guidance_scale": record.get("guidance_scale"),
                    "max_sequence_length": record.get("max_sequence_length"),
                }
            )
    finally:
        if owns_pipeline:
            cleanup_pipeline(pipeline)

    aggregates = _aggregate_metrics(comparisons)
    result: dict[str, object] = {
        "format": RGB_METRICS_FORMAT,
        "variant": variant,
        "reference_variant": reference_variant,
        "split": split,
        "prompt_count": len(selected_records),
        "visual_gen_args": str(visual_gen_args),
        "model": model,
        "reference_root": str(reference_root),
        "output_root": str(output_root),
        "metrics_json": str(metrics_json),
        "config_metadata": dict(config_metadata or {}),
        "provenance": dict(provenance),
        "elapsed_seconds": time.monotonic() - start,
        "comparisons": comparisons,
        "aggregates": aggregates,
    }
    write_json(metrics_json, result)
    return result


def collect_visual_gen_config_metadata(visual_gen_args: Path, *, model: str) -> dict[str, object]:
    from tensorrt_llm.visual_gen import VisualGenArgs

    args = VisualGenArgs.from_yaml(visual_gen_args, model=model)
    return build_visual_gen_config_metadata(args)


def build_visual_gen_config_metadata(args: Any) -> dict[str, object]:
    attention_config = getattr(args, "attention_config", None)
    quant_config = getattr(args, "quant_config", None)
    quant_dynamic = _config_field(quant_config, "dynamic")
    force_dynamic_quantization = _config_field(quant_config, "force_dynamic_quantization")
    return {
        "attention_backend": _enum_value(getattr(attention_config, "backend", None)),
        "quant_attention_config": repr(getattr(attention_config, "quant_attention_config", None)),
        "pipeline_quant_algo": _enum_value(_config_field(quant_config, "quant_algo")),
        "pipeline_dynamic_weight_quant": bool(
            quant_dynamic
            if quant_dynamic is not None
            else getattr(args, "dynamic_weight_quant", False)
        ),
        "pipeline_force_dynamic_quantization": bool(
            force_dynamic_quantization
            if force_dynamic_quantization is not None
            else getattr(args, "force_dynamic_quantization", False)
        ),
    }


def build_runtime_provenance(args: argparse.Namespace) -> dict[str, object]:
    return {
        "cluster_alias": args.cluster_alias,
        "allocation_id": args.allocation_id or os.environ.get("SSH_GW_ALLOC_ID"),
        "job_id": args.job_id or os.environ.get("SLURM_JOB_ID"),
        "node_list": args.node_list or os.environ.get("SLURM_NODELIST"),
        "container_runtime": DEFAULT_CONTAINER_RUNTIME,
        "enroot_image": args.enroot_image,
        "command": args.command or " ".join(os.sys.argv),
        "model_snapshot_path": args.model_snapshot_path,
        "git_head": git_commit(Path.cwd()),
    }


def _load_rgb_unit_image(path: Path) -> tuple[tuple[int, int], list[float]]:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    width, height = image.size
    values: list[float] = []
    for red, green, blue in image.getdata():
        values.extend((red / 255.0, green / 255.0, blue / 255.0))
    return (height, width), values


def _global_ssim(candidate: Sequence[float], reference: Sequence[float]) -> float:
    mu_x = _mean(list(candidate))
    mu_y = _mean(list(reference))
    var_x = _mean([(value - mu_x) ** 2 for value in candidate])
    var_y = _mean([(value - mu_y) ** 2 for value in reference])
    cov_xy = _mean(
        [
            (candidate_value - mu_x) * (reference_value - mu_y)
            for candidate_value, reference_value in zip(candidate, reference)
        ]
    )
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return float(max(-1.0, min(1.0, numerator / denominator)))


def _aggregate_metrics(comparisons: list[dict[str, object]]) -> dict[str, object]:
    if not comparisons:
        raise ValueError("RGB eval requires at least one comparison")
    mse_values = [_metric_to_float(item["mse"]) for item in comparisons]
    psnr_values = [_metric_to_float(item["psnr"]) for item in comparisons]
    ssim_values = [_metric_to_float(item["ssim"]) for item in comparisons]
    return {
        "num_samples": len(comparisons),
        "mse_mean": _json_metric(_mean(mse_values)),
        "mse_max": _json_metric(max(mse_values)),
        "psnr_min": _json_metric(min(psnr_values)),
        "psnr_mean": _json_metric(_mean(psnr_values)),
        "ssim_min": _json_metric(min(ssim_values)),
        "ssim_mean": _json_metric(_mean(ssim_values)),
    }


def _selected_records(records: list[dict[str, object]], *, split: str) -> list[dict[str, object]]:
    if split not in SPLIT_CHOICES:
        raise ValueError(f"invalid split: {split}")
    selected = [record for record in records if _expect_string(record, "split") == split]
    if not selected:
        raise ValueError(f"no prompt records selected for split {split}")
    return selected


def _metric_to_float(value: object) -> float:
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    if value == "nan":
        return float("nan")
    if not isinstance(value, (int, float)):
        raise ValueError(f"metric value must be numeric or nonfinite sentinel: {value!r}")
    return float(value)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _json_metric(value: float) -> float | str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "nan"
    return value


def _enum_value(value: object) -> object:
    if value is None:
        return None
    return getattr(value, "name", None) or getattr(value, "value", None) or str(value)


def _config_field(config: object, field_name: str) -> object:
    if isinstance(config, dict):
        return config.get(field_name)
    return getattr(config, field_name, None)


def _expect_string(value: dict[str, object], field_name: str) -> str:
    field = value.get(field_name)
    if not isinstance(field, str) or not field:
        raise ValueError(f"{field_name} must be a non-empty string")
    return field


def _run_split_command(args: argparse.Namespace) -> None:
    records = read_prompt_jsonl(Path(args.prompt_manifest_jsonl))
    run_qwen_image_rgb_split_eval(
        records=records,
        split=args.split,
        model=args.model,
        visual_gen_args=Path(args.visual_gen_args),
        reference_root=Path(args.reference_root),
        output_root=Path(args.output_root),
        metrics_json=Path(args.metrics_json),
        variant=args.variant,
        reference_variant=args.reference_variant,
        device=args.device,
        provenance=build_runtime_provenance(args),
        config_metadata=collect_visual_gen_config_metadata(
            Path(args.visual_gen_args), model=args.model
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run-split")
    run_parser.add_argument("--prompt-manifest-jsonl", required=True)
    run_parser.add_argument("--split", required=True, choices=SPLIT_CHOICES)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--visual-gen-args", required=True)
    run_parser.add_argument("--reference-root", required=True)
    run_parser.add_argument("--output-root", required=True)
    run_parser.add_argument("--metrics-json", required=True)
    run_parser.add_argument("--variant", required=True)
    run_parser.add_argument("--reference-variant", default="bf16_sage_fp8")
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--cluster-alias", default=DEFAULT_CLUSTER_ALIAS)
    run_parser.add_argument("--allocation-id")
    run_parser.add_argument("--job-id")
    run_parser.add_argument("--node-list")
    run_parser.add_argument("--enroot-image", default=DEFAULT_ENROOT_IMAGE)
    run_parser.add_argument("--model-snapshot-path")
    run_parser.add_argument("--command")
    run_parser.set_defaults(func=_run_split_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
