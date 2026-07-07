# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a Qwen-Image static MXFP8 torch.compile/CUDA Graph smoke validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from scripts.visualgen_eval.qwen_image_capture_manifest import git_commit, write_json
from scripts.visualgen_eval.qwen_image_prompt_manifest import read_jsonl
from scripts.visualgen_eval.qwen_image_teacher_capture import (
    cleanup_pipeline,
    infer_record,
    save_reference_image,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.args import (
    CompilationConfig,
    CudaGraphConfig,
    TorchCompileConfig,
    VisualGenArgs,
)

CUDA_GRAPH_SMOKE_FORMAT = "qwen_image_static_cuda_graph_smoke_v1"
REQUIRED_ENROOT_IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20"


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    prompt_manifest = Path(args.prompt_manifest_jsonl)
    report_path = Path(args.report_json)
    output_path = Path(args.output_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(prompt_manifest)
    record = _select_prompt(records, prompt_id=args.prompt_id, split=args.split)

    visual_args = VisualGenArgs.from_yaml(args.visual_gen_args, model=args.model)
    visual_args = visual_args.model_copy(
        update={
            "compilation_config": CompilationConfig(
                resolutions=[tuple(args.warmup_resolution)],
                num_frames=[1],
                skip_warmup=False,
            ),
            "torch_compile_config": TorchCompileConfig(
                enable=not args.disable_torch_compile,
                enable_fullgraph=args.enable_fullgraph,
                enable_autotune=args.enable_autotune,
            ),
            "cuda_graph_config": CudaGraphConfig(enable=True),
        }
    )

    pipeline = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    try:
        loader = PipelineLoader(visual_args, device=args.device)
        pipeline = loader.load()
        load_elapsed = time.monotonic() - start
        pipeline_cuda_graph_config = _model_dump(
            getattr(pipeline.pipeline_config, "cuda_graph", None)
        )
        pipeline_torch_compile_config = _model_dump(
            getattr(pipeline.pipeline_config, "torch_compile", None)
        )
        transformer_components = [str(component) for component in pipeline.transformer_components]
        runner_counts_after_load = _runner_graph_counts(pipeline)

        infer_start = time.monotonic()
        output = infer_record(pipeline, record)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_elapsed = time.monotonic() - infer_start
        save_reference_image(output, output_path)
        runner_counts_after_infer = _runner_graph_counts(pipeline)

        status = "passed"
        failures: list[str] = []
        if not runner_counts_after_infer:
            status = "failed"
            failures.append("no CUDA graph runners were registered")
        if sum(runner_counts_after_infer.values()) <= 0:
            status = "failed"
            failures.append("no CUDA graphs were captured")
        if not pipeline_cuda_graph_config.get("enable", False):
            status = "failed"
            failures.append("pipeline_config.cuda_graph.enable is false")

        report: dict[str, object] = {
            "format": CUDA_GRAPH_SMOKE_FORMAT,
            "status": status,
            "failures": failures,
            "model": args.model,
            "visual_gen_args": args.visual_gen_args,
            "prompt_manifest_jsonl": str(prompt_manifest),
            "output_image": str(output_path),
            "report_json": str(report_path),
            "torch_compile_config": visual_args.torch_compile_config.model_dump(),
            "cuda_graph_config": visual_args.cuda_graph_config.model_dump(),
            "compilation_config": visual_args.compilation_config.model_dump(),
            "pipeline_cuda_graph_config": pipeline_cuda_graph_config,
            "pipeline_torch_compile_config": pipeline_torch_compile_config,
            "transformer_components": transformer_components,
            "runner_counts_after_load": runner_counts_after_load,
            "runner_counts_after_infer": runner_counts_after_infer,
            "cuda_graph_runner_count": len(runner_counts_after_infer),
            "captured_graph_count_after_infer": sum(runner_counts_after_infer.values()),
            "load_elapsed_seconds": load_elapsed,
            "infer_elapsed_seconds": infer_elapsed,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            ),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_capability": (
                list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
            ),
            "nvidia_smi_clocks": _nvidia_smi_clocks(),
            "cluster_alias": args.cluster_alias,
            "allocation_id": args.allocation_id,
            "enroot_image": args.enroot_image,
            "git_head": git_commit(Path(args.project_root)),
            "prompt_id": record["prompt_id"],
            "split": record["split"],
            "height": record["height"],
            "width": record["width"],
            "num_inference_steps": record["num_inference_steps"],
        }
        write_json(report_path, report)
        if status != "passed":
            raise RuntimeError("; ".join(failures))
        return report
    finally:
        if pipeline is not None:
            cleanup_pipeline(pipeline)


def _select_prompt(
    records: list[dict[str, object]], *, prompt_id: str | None, split: str
) -> dict[str, object]:
    for record in records:
        if prompt_id is not None and record.get("prompt_id") != prompt_id:
            continue
        if prompt_id is None and record.get("split") != split:
            continue
        return record
    selector = f"prompt_id={prompt_id}" if prompt_id is not None else f"split={split}"
    raise ValueError(f"no prompt record found for {selector}")


def _runner_graph_counts(pipeline: Any) -> dict[str, int]:
    return {
        str(name): len(runner.graphs)
        for name, runner in getattr(pipeline, "_cuda_graph_runners", {}).items()
    }


def _model_dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if isinstance(value, dict):
        return dict(value)
    return {"repr": repr(value)}


def _nvidia_smi_clocks() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,clocks.sm,clocks.mem,persistence_mode",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--visual-gen-args", required=True)
    parser.add_argument("--prompt-manifest-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--output-image", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--split", default="smoke")
    parser.add_argument("--prompt-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-resolution", nargs=2, type=int, default=[1024, 1024])
    parser.add_argument("--disable-torch-compile", action="store_true")
    parser.add_argument("--enable-fullgraph", action="store_true")
    parser.add_argument("--enable-autotune", action="store_true")
    parser.add_argument("--cluster-alias", default="B300-mars")
    parser.add_argument("--allocation-id")
    parser.add_argument("--enroot-image", default=REQUIRED_ENROOT_IMAGE)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_smoke(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
