# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify loaded Qwen-Image-Layered static MXFP8 Linear coverage."""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import yaml
from pydantic import Field, PositiveInt, model_validator

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineComponent, PipelineLoader
from tensorrt_llm.llmapi.utils import StrictBaseModel
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import (
    CompilationConfig,
    CudaGraphConfig,
    TorchCompileConfig,
    VisualGenArgs,
)

LOADER_COVERAGE_FORMAT = "qwen_image_layered_static_mxfp8_loader_coverage_v1"
QWEN_BLOCK_LINEAR_POLICY = "qwen_block_linears_840"
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
QWEN_LAYER_COUNT = 60
QWEN_BLOCK_LINEAR_TARGET_COUNT = QWEN_LAYER_COUNT * len(QWEN_IMAGE_BLOCK_LINEAR_SUFFIXES)
QWEN_STATIC_BF16_EXCLUSIONS = ("img_in", "txt_in", "norm_out.linear", "proj_out")
REQUIRED_CONTAINER_IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20"

TargetPolicyName = Literal["qwen_block_linears_840"]


class QwenImageLayeredLoaderCoverageConfig(StrictBaseModel):
    """Strict config for static MXFP8 loader coverage."""

    model: str = Field(description="Static Qwen-Image-Layered checkpoint root to load.")
    output_dir: str = Field(description="Directory for loader coverage JSON artifacts.")
    expected_target_policy: TargetPolicyName = Field(
        default=QWEN_BLOCK_LINEAR_POLICY,
        description="Expected Qwen static MXFP8 target policy.",
    )
    expected_num_layers: PositiveInt = Field(
        default=QWEN_LAYER_COUNT,
        description="Expected Qwen transformer block count.",
    )
    expected_target_count: PositiveInt = Field(
        default=QWEN_BLOCK_LINEAR_TARGET_COUNT,
        description="Expected number of static MXFP8 block Linear modules.",
    )
    expected_bf16_exclusions: list[str] = Field(
        default_factory=lambda: list(QWEN_STATIC_BF16_EXCLUSIONS),
        description="Expected non-block Linear modules that must remain BF16.",
    )
    device: str = Field(default="cuda", description="Device used by PipelineLoader.")
    skip_warmup: bool = Field(default=True, description="Skip VisualGen warmup.")
    container_image: str = Field(
        default=REQUIRED_CONTAINER_IMAGE,
        description="Container image used for the formal remote validation.",
    )
    remote_checkout: str | None = Field(
        default=None,
        description="Remote TensorRT-LLM checkout path recorded in provenance.",
    )
    qwen_weight_cache_root: str | None = Field(
        default=None,
        description="Qwen-Image-Layered HuggingFace cache root recorded in provenance.",
    )
    qwen_weight_snapshot: str | None = Field(
        default=None,
        description="Resolved Qwen-Image-Layered snapshot path recorded in provenance.",
    )

    @model_validator(mode="after")
    def _validate_static_loader_contract(self) -> "QwenImageLayeredLoaderCoverageConfig":
        expected = int(self.expected_num_layers) * len(QWEN_IMAGE_BLOCK_LINEAR_SUFFIXES)
        if self.expected_target_count != expected:
            raise ValueError(
                f"{self.expected_target_policy} with {self.expected_num_layers} layers requires "
                f"expected_target_count={expected}"
            )
        if self.expected_target_policy == QWEN_BLOCK_LINEAR_POLICY:
            if self.expected_num_layers != QWEN_LAYER_COUNT:
                raise ValueError(
                    f"{QWEN_BLOCK_LINEAR_POLICY} requires expected_num_layers={QWEN_LAYER_COUNT}"
                )
            if self.expected_target_count != QWEN_BLOCK_LINEAR_TARGET_COUNT:
                raise ValueError(
                    f"{QWEN_BLOCK_LINEAR_POLICY} requires "
                    f"expected_target_count={QWEN_BLOCK_LINEAR_TARGET_COUNT}"
                )
        if tuple(self.expected_bf16_exclusions) != QWEN_STATIC_BF16_EXCLUSIONS:
            raise ValueError(
                "static Qwen-Image-Layered coverage requires BF16 exclusions "
                f"{QWEN_STATIC_BF16_EXCLUSIONS}"
            )
        return self


@dataclass(frozen=True)
class LinearModuleRecord:
    """Loaded Linear module state relevant to static MXFP8 coverage."""

    name: str
    normalized_name: str
    weight_dtype: str | None
    weight_shape: tuple[int, ...] | None
    quant_algo: str | None
    quant_method: str | None
    has_weight_scale: bool
    weight_scale_dtype: str | None
    weight_scale_shape: tuple[int, ...] | None
    weight_scale_numel: int
    has_input_scale: bool
    has_inv_input_scale: bool
    is_target: bool
    is_bf16_exclusion: bool


@dataclass(frozen=True)
class LoaderCoverageResult:
    """Paths and counts produced by one coverage run."""

    output_dir: Path
    coverage_path: Path
    provenance_path: Path
    target_count: int
    static_mxfp8_target_count: int
    bf16_exclusion_count: int
    total_linear_count: int


def load_loader_coverage_config(path: str | Path) -> QwenImageLayeredLoaderCoverageConfig:
    """Load a strict loader coverage config from JSON or YAML."""
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Loader coverage config must contain a mapping: {config_path}")
    return QwenImageLayeredLoaderCoverageConfig(**data)


def build_qwen_block_linear_targets(num_layers: int = QWEN_LAYER_COUNT) -> tuple[str, ...]:
    """Build normalized Qwen transformer-block Linear target names."""
    return tuple(
        f"transformer_blocks.{layer_idx}.{suffix}"
        for layer_idx in range(num_layers)
        for suffix in QWEN_IMAGE_BLOCK_LINEAR_SUFFIXES
    )


def normalize_qwen_module_name(name: str) -> str:
    """Normalize module names emitted before or after torch.compile wrapping."""
    normalized = name
    while "._orig_mod." in normalized:
        normalized = normalized.replace("._orig_mod.", ".")
    if normalized.startswith("_orig_mod."):
        normalized = normalized[len("_orig_mod.") :]
    return normalized


def load_static_mxfp8_pipeline(config: QwenImageLayeredLoaderCoverageConfig):
    """Load the static checkpoint through the existing VisualGen PipelineLoader path."""
    args = VisualGenArgs(
        model=config.model,
        compilation_config=CompilationConfig(skip_warmup=config.skip_warmup),
        torch_compile_config=TorchCompileConfig(enable=False, enable_autotune=False),
        cuda_graph_config=CudaGraphConfig(enable=False),
    )
    return PipelineLoader(args, device=config.device).load(
        skip_warmup=config.skip_warmup,
        skip_components=_non_transformer_skip_components(),
    )


def _non_transformer_skip_components() -> list[PipelineComponent | str]:
    component_values = {
        "TEXT_ENCODER": "text_encoder",
        "TEXT_ENCODER_2": "text_encoder_2",
        "VAE": "vae",
        "TOKENIZER": "tokenizer",
        "TOKENIZER_2": "tokenizer_2",
        "SCHEDULER": "scheduler",
        "IMAGE_ENCODER": "image_encoder",
        "IMAGE_PROCESSOR": "image_processor",
        "PROCESSOR": "processor",
    }
    skip_components: list[PipelineComponent | str] = []
    for name, value in component_values.items():
        component = getattr(PipelineComponent, name, None)
        skip_components.append(component if component is not None else value)
    return skip_components


def analyze_loaded_transformer_coverage(
    pipeline,
    config: QwenImageLayeredLoaderCoverageConfig,
    *,
    linear_cls: type[nn.Module] = Linear,
) -> dict[str, object]:
    """Analyze a loaded VisualGen pipeline and fail on coverage mismatches."""
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise ValueError("Loaded pipeline does not expose a transformer component")

    target_layers = build_qwen_block_linear_targets(int(config.expected_num_layers))
    bf16_exclusions = tuple(config.expected_bf16_exclusions)
    records = collect_linear_module_records(
        transformer,
        target_layers=target_layers,
        bf16_exclusions=bf16_exclusions,
        linear_cls=linear_cls,
    )
    report = build_coverage_report(
        pipeline_config=getattr(pipeline, "pipeline_config", None),
        config=config,
        records=records,
        target_layers=target_layers,
        bf16_exclusions=bf16_exclusions,
    )
    failures = report["failures"]
    if isinstance(failures, dict) and any(failures.values()):
        raise ValueError(_format_failures(failures))
    return report


def collect_linear_module_records(
    transformer: nn.Module,
    *,
    target_layers: tuple[str, ...],
    bf16_exclusions: tuple[str, ...],
    linear_cls: type[nn.Module] = Linear,
) -> tuple[LinearModuleRecord, ...]:
    """Collect loaded Linear module state, normalizing torch.compile wrapper names."""
    target_set = set(target_layers)
    exclusion_set = set(bf16_exclusions)
    seen: set[str] = set()
    records: list[LinearModuleRecord] = []
    for name, module in transformer.named_modules():
        if not name or not isinstance(module, linear_cls):
            continue
        normalized_name = normalize_qwen_module_name(name)
        if normalized_name in seen:
            raise ValueError(
                f"Duplicate Linear module after Qwen name normalization: {normalized_name}"
            )
        seen.add(normalized_name)
        weight = getattr(module, "weight", None)
        weight_scale = getattr(module, "weight_scale", None)
        records.append(
            LinearModuleRecord(
                name=name,
                normalized_name=normalized_name,
                weight_dtype=_tensor_dtype_name(weight),
                weight_shape=_tensor_shape(weight),
                quant_algo=_quant_algo_name(module),
                quant_method=_quant_method_name(module),
                has_weight_scale=isinstance(weight_scale, torch.Tensor),
                weight_scale_dtype=_tensor_dtype_name(weight_scale),
                weight_scale_shape=_tensor_shape(weight_scale),
                weight_scale_numel=_tensor_numel(weight_scale),
                has_input_scale=isinstance(getattr(module, "input_scale", None), torch.Tensor),
                has_inv_input_scale=isinstance(
                    getattr(module, "inv_input_scale", None), torch.Tensor
                ),
                is_target=normalized_name in target_set,
                is_bf16_exclusion=normalized_name in exclusion_set,
            )
        )
    return tuple(sorted(records, key=lambda record: record.normalized_name))


def build_coverage_report(
    *,
    pipeline_config: object,
    config: QwenImageLayeredLoaderCoverageConfig,
    records: tuple[LinearModuleRecord, ...],
    target_layers: tuple[str, ...],
    bf16_exclusions: tuple[str, ...],
) -> dict[str, object]:
    """Build a JSON-serializable coverage report and collect failures."""
    target_set = set(target_layers)
    exclusion_set = set(bf16_exclusions)
    names = {record.normalized_name for record in records}
    target_records = [record for record in records if record.normalized_name in target_set]
    exclusion_records = [record for record in records if record.normalized_name in exclusion_set]
    static_target_records = [record for record in target_records if _is_static_mxfp8(record)]
    bf16_exclusion_records = [
        record for record in exclusion_records if _is_bf16_unquantized(record)
    ]

    failures: dict[str, list[str]] = {
        "pipeline_config": [],
        "missing_targets": sorted(target_set - names),
        "missing_bf16_exclusions": sorted(exclusion_set - names),
        "unexpected_linears": sorted(names - target_set - exclusion_set),
        "target_not_static_mxfp8": [],
        "bf16_exclusion_not_bf16": [],
        "unexpected_quantized_non_targets": [],
    }

    quant_algo = _pipeline_quant_algo_name(pipeline_config)
    exclude_modules = _pipeline_exclude_modules(pipeline_config)
    dynamic_weight_quant = bool(getattr(pipeline_config, "dynamic_weight_quant", False))
    force_dynamic_quantization = bool(getattr(pipeline_config, "force_dynamic_quantization", False))
    if quant_algo != "FP8_BLOCK_SCALES":
        failures["pipeline_config"].append(
            f"quant_algo={quant_algo!r}, expected 'FP8_BLOCK_SCALES'"
        )
    if dynamic_weight_quant:
        failures["pipeline_config"].append("dynamic_weight_quant=True, expected False")
    if force_dynamic_quantization:
        failures["pipeline_config"].append("force_dynamic_quantization=True, expected False")
    if sorted(exclude_modules) != sorted(bf16_exclusions):
        failures["pipeline_config"].append(
            f"exclude_modules={exclude_modules!r}, expected {list(bf16_exclusions)!r}"
        )

    if len(target_records) != config.expected_target_count:
        failures["target_not_static_mxfp8"].append(
            f"found {len(target_records)} target Linear modules, "
            f"expected {config.expected_target_count}"
        )
    for record in target_records:
        if not _is_static_mxfp8(record):
            failures["target_not_static_mxfp8"].append(_record_failure_summary(record))
    for record in exclusion_records:
        if not _is_bf16_unquantized(record):
            failures["bf16_exclusion_not_bf16"].append(_record_failure_summary(record))
    for record in records:
        if record.normalized_name not in target_set and _is_quantized(record):
            failures["unexpected_quantized_non_targets"].append(_record_failure_summary(record))

    return {
        "format": LOADER_COVERAGE_FORMAT,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "failed" if any(failures.values()) else "passed",
        "config": config.model_dump(),
        "target_policy": config.expected_target_policy,
        "target_count": config.expected_target_count,
        "static_mxfp8_target_count": len(static_target_records),
        "bf16_exclusions": list(bf16_exclusions),
        "bf16_exclusion_count": len(bf16_exclusion_records),
        "total_linear_count": len(records),
        "pipeline_quant_algo": quant_algo,
        "pipeline_dynamic_weight_quant": dynamic_weight_quant,
        "pipeline_force_dynamic_quantization": force_dynamic_quantization,
        "pipeline_exclude_modules": exclude_modules,
        "failures": failures,
        "records": [asdict(record) for record in records],
    }


def write_loader_coverage_outputs(
    report: dict[str, object],
    config: QwenImageLayeredLoaderCoverageConfig,
) -> LoaderCoverageResult:
    """Write coverage and provenance JSON artifacts under the configured output directory."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = output_dir / "loader_coverage.json"
    provenance_path = output_dir / "loader_coverage_provenance.json"
    coverage_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = build_loader_coverage_provenance(config, coverage_path)
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return LoaderCoverageResult(
        output_dir=output_dir,
        coverage_path=coverage_path,
        provenance_path=provenance_path,
        target_count=int(report["target_count"]),
        static_mxfp8_target_count=int(report["static_mxfp8_target_count"]),
        bf16_exclusion_count=int(report["bf16_exclusion_count"]),
        total_linear_count=int(report["total_linear_count"]),
    )


def run_loader_coverage(config: QwenImageLayeredLoaderCoverageConfig) -> LoaderCoverageResult:
    """Load the checkpoint, analyze coverage, and write durable artifacts."""
    pipeline = load_static_mxfp8_pipeline(config)
    report = analyze_loaded_transformer_coverage(pipeline, config)
    return write_loader_coverage_outputs(report, config)


def build_loader_coverage_provenance(
    config: QwenImageLayeredLoaderCoverageConfig,
    coverage_path: Path,
) -> dict[str, object]:
    """Build a JSON-serializable provenance record for a loader coverage run."""
    cuda_device = None
    if torch.cuda.is_available():
        cuda_device = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "format": LOADER_COVERAGE_FORMAT,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": cuda_device,
        "container_image": config.container_image,
        "remote_checkout": config.remote_checkout,
        "qwen_weight_cache_root": config.qwen_weight_cache_root,
        "qwen_weight_snapshot": config.qwen_weight_snapshot,
        "config": config.model_dump(),
        "coverage_path": str(coverage_path),
    }


def _is_static_mxfp8(record: LinearModuleRecord) -> bool:
    return (
        record.weight_dtype == "torch.float8_e4m3fn"
        and record.quant_algo == "FP8_BLOCK_SCALES"
        and record.quant_method == "FP8BlockScalesLinearMethod"
        and record.has_weight_scale
        and record.weight_scale_numel > 0
        and record.has_input_scale
        and record.has_inv_input_scale
    )


def _is_bf16_unquantized(record: LinearModuleRecord) -> bool:
    return (
        record.weight_dtype == "torch.bfloat16"
        and record.quant_algo is None
        and record.quant_method != "FP8BlockScalesLinearMethod"
        and not record.has_weight_scale
    )


def _is_quantized(record: LinearModuleRecord) -> bool:
    return (
        record.weight_dtype == "torch.float8_e4m3fn"
        or record.quant_algo == "FP8_BLOCK_SCALES"
        or record.quant_method == "FP8BlockScalesLinearMethod"
        or record.has_weight_scale
    )


def _record_failure_summary(record: LinearModuleRecord) -> str:
    return (
        f"{record.normalized_name}: weight_dtype={record.weight_dtype}, "
        f"quant_algo={record.quant_algo}, quant_method={record.quant_method}, "
        f"weight_scale_shape={record.weight_scale_shape}, "
        f"weight_scale_dtype={record.weight_scale_dtype}, "
        f"has_input_scale={record.has_input_scale}, "
        f"has_inv_input_scale={record.has_inv_input_scale}"
    )


def _format_failures(failures: dict[str, list[str]]) -> str:
    parts = []
    for key, items in failures.items():
        if items:
            parts.append(f"{key}: {items[:8]}")
    return "Static MXFP8 loader coverage failed; " + "; ".join(parts)


def _pipeline_quant_algo_name(pipeline_config: object) -> str | None:
    quant_config = getattr(pipeline_config, "quant_config", None)
    return _quant_algo_value_name(getattr(quant_config, "quant_algo", None))


def _pipeline_exclude_modules(pipeline_config: object) -> list[str]:
    quant_config = getattr(pipeline_config, "quant_config", None)
    exclude_modules = getattr(quant_config, "exclude_modules", None)
    if exclude_modules is None:
        return []
    return [str(item) for item in exclude_modules]


def _quant_algo_name(module: nn.Module) -> str | None:
    quant_config = getattr(module, "quant_config", None)
    return _quant_algo_value_name(getattr(quant_config, "quant_algo", None))


def _quant_algo_value_name(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, QuantAlgo):
        return value.name
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value)


def _quant_method_name(module: nn.Module) -> str | None:
    quant_method = getattr(module, "quant_method", None)
    if quant_method is None:
        return None
    return type(quant_method).__name__


def _tensor_dtype_name(value: object) -> str | None:
    if not isinstance(value, torch.Tensor):
        return None
    return str(value.dtype)


def _tensor_shape(value: object) -> tuple[int, ...] | None:
    if not isinstance(value, torch.Tensor):
        return None
    return tuple(int(dim) for dim in value.shape)


def _tensor_numel(value: object) -> int:
    if not isinstance(value, torch.Tensor):
        return 0
    return int(value.numel())


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify static MXFP8 loader layer coverage for Qwen-Image-Layered."
    )
    parser.add_argument("--config", required=True, help="Loader coverage config JSON/YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_loader_coverage_config(args.config)
    result = run_loader_coverage(config)
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "coverage_path": str(result.coverage_path),
                "provenance_path": str(result.provenance_path),
                "target_count": result.target_count,
                "static_mxfp8_target_count": result.static_mxfp8_target_count,
                "bf16_exclusion_count": result.bf16_exclusion_count,
                "total_linear_count": result.total_linear_count,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
