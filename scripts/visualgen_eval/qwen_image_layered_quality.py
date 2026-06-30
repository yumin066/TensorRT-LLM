#!/usr/bin/env python3
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

"""Evaluate Qwen-Image-Layered variants against a BF16 tensor reference.

The manifest is intentionally explicit: every sample records prompt,
conditioning image, seed, denoising schedule, resolution bucket, and layer
count. Variants point at VisualGenArgs YAML files and may either provide
pre-generated tensor artifacts or let this tool generate missing artifacts.

Example manifest:

.. code-block:: json

  {
    "output_root": "outputs/qwen_layered_quality",
    "samples": [
      {
        "id": "logo_overlay",
        "prompt": "separate the foreground logo into editable layers",
        "image": "inputs/logo.png",
        "seed": 1234,
        "num_inference_steps": 50,
        "resolution": 640,
        "layers": 4,
        "height": 640,
        "width": 640,
        "negative_prompt": "low quality"
      }
    ],
    "variants": [
      {
        "name": "bf16",
        "model": "/models/qwen-image-layered",
        "visual_gen_args": "configs/qwen-image-layered-bf16.yaml"
      },
      {
        "name": "all_layer_dynamic_mxfp8",
        "model": "/models/qwen-image-layered",
        "visual_gen_args": "configs/qwen-image-layered-all-mxfp8.yaml"
      }
    ]
  }
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import re
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType
from typing import Any, TypeAlias

import torch
from pydantic import Field, PositiveInt, model_validator

from tensorrt_llm.llmapi.utils import StrictBaseModel
from tensorrt_llm.visual_gen.output import VisualGenOutput
from tensorrt_llm.visual_gen.params import VisualGenParams

REFERENCE_VARIANT = "bf16"
LINEAR_BLOCK_RE = re.compile(r"(?:^|\.)transformer_blocks\.(\d+)(?:\.|$)")
LINEAR_TRANSFORMER_BLOCKS = 60
LINEAR_NON_BACKBONE_IGNORE = ("img_in", "txt_in", "norm_out", "proj_out")
LINEAR_BLOCK_SCALE_ALGO = "FP8_BLOCK_SCALES"
SUPPORTED_VARIANT_NAMES = frozenset(
    {
        REFERENCE_VARIANT,
        "k12",
        "k8",
        "k4",
        "k0",
        "all_layer_dynamic_mxfp8",
        "qat_all_layer_mxfp8",
    }
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonMetadataValue: TypeAlias = JsonScalar | list[JsonScalar] | dict[str, JsonScalar]


class LayeredSample(StrictBaseModel):
    """One deterministic Qwen-Image-Layered prompt and conditioning input."""

    id: str = Field(description="Stable sample id used for artifact paths.")
    prompt: str = Field(description="Text prompt. Empty string enables image captioning.")
    image: str = Field(description="Conditioning image path, relative to the manifest.")
    seed: int = Field(description="Diffusion RNG seed.")
    resolution: int = Field(description="Qwen-Image-Layered resolution bucket: 640 or 1024.")
    layers: PositiveInt = Field(description="Expected output layer count.")
    num_inference_steps: PositiveInt | None = Field(
        default=None,
        description="Number of denoising steps. Required when generation is needed.",
    )
    sigmas: list[float] | None = Field(
        default=None,
        description="Explicit denoising sigmas for provenance or pre-generated artifacts.",
    )
    height: PositiveInt | None = Field(default=None, description="Optional output height.")
    width: PositiveInt | None = Field(default=None, description="Optional output width.")
    negative_prompt: str | None = Field(default=None, description="Optional negative prompt.")
    guidance_scale: float | None = Field(default=None, description="Classifier-free guidance.")
    max_sequence_length: PositiveInt | None = Field(
        default=None,
        description="Maximum text sequence length.",
    )
    cfg_normalize: bool = Field(
        default=False,
        description="Normalize CFG prediction by conditional norm.",
    )
    use_en_prompt: bool = Field(
        default=False,
        description="Use English image captioning when prompt is empty.",
    )
    layer_metadata: dict[str, JsonMetadataValue] = Field(
        default_factory=dict,
        description="Optional alpha, mask, or layer annotations for auditability.",
    )

    @model_validator(mode="after")
    def _validate_schedule_and_shape(self) -> "LayeredSample":
        if not self.id:
            raise ValueError("sample id must be non-empty")
        if self.resolution not in (640, 1024):
            raise ValueError(f"resolution must be 640 or 1024, got {self.resolution}")
        if self.num_inference_steps is None and not self.sigmas:
            raise ValueError("sample requires num_inference_steps or sigmas")
        if self.sigmas is not None and len(self.sigmas) == 0:
            raise ValueError("sigmas must not be empty when provided")
        if (self.height is None) != (self.width is None):
            raise ValueError("height and width must be set together")
        return self


class LayeredVariant(StrictBaseModel):
    """One BF16, K-sweep, MXFP8, or QAT checkpoint/config variant."""

    name: str = Field(description="Variant name, for example bf16 or k12.")
    model: str = Field(description="VisualGen model path or HuggingFace id.")
    visual_gen_args: str = Field(description="VisualGenArgs YAML path.")
    checkpoint_path: str | None = Field(
        default=None,
        description="Optional explicit checkpoint path for provenance.",
    )
    artifact_paths: dict[str, str] = Field(
        default_factory=dict,
        description="Optional sample id -> pre-generated .pt/.safetensors path.",
    )
    metadata: dict[str, JsonMetadataValue] = Field(
        default_factory=dict,
        description="Optional free-form provenance, such as quant preset or LoRA rank.",
    )

    @model_validator(mode="after")
    def _validate_name_and_paths(self) -> "LayeredVariant":
        if not self.name:
            raise ValueError("variant name must be non-empty")
        if not self.model:
            raise ValueError(f"variant {self.name!r} requires a model path or id")
        if not self.visual_gen_args:
            raise ValueError(f"variant {self.name!r} requires visual_gen_args")
        return self


class LayeredManifest(StrictBaseModel):
    """Top-level quality manifest."""

    output_root: str = Field(description="Root directory for generated artifacts and metrics.")
    samples: list[LayeredSample] = Field(min_length=1, description="Evaluation samples.")
    variants: list[LayeredVariant] = Field(
        min_length=2,
        description="Variants. Must include bf16 plus at least one candidate.",
    )

    @model_validator(mode="after")
    def _validate_manifest(self) -> "LayeredManifest":
        if not self.output_root:
            raise ValueError("output_root must be non-empty")
        sample_ids = [sample.id for sample in self.samples]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample ids must be unique")
        variant_names = [variant.name for variant in self.variants]
        if len(set(variant_names)) != len(variant_names):
            raise ValueError("variant names must be unique")
        if REFERENCE_VARIANT not in variant_names:
            raise ValueError(f"manifest must include a {REFERENCE_VARIANT!r} reference variant")
        return self


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate/load Qwen-Image-Layered tensor artifacts and compute PSNR/SSIM "
            "against the BF16 reference variant."
        )
    )
    parser.add_argument("--manifest", required=True, help="JSON quality manifest.")
    parser.add_argument(
        "--output-json",
        help="Metrics JSON path. Defaults to <output_root>/metrics.json.",
    )
    parser.add_argument(
        "--artifact-format",
        choices=("pt", "safetensors"),
        default="pt",
        help="Tensor artifact format for generated outputs.",
    )
    parser.add_argument(
        "--load-existing",
        action="store_true",
        help="Require all artifacts to exist; do not generate missing outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate default artifacts even when they already exist.",
    )
    parser.add_argument(
        "--capture-transformer-tuples",
        action="store_true",
        help=(
            "Use an in-process single-worker pipeline for generated artifacts and save "
            "transformer input/output tuples for BF16 teacher distillation."
        ),
    )
    parser.add_argument(
        "--emit-linear-manifests",
        action="store_true",
        help=(
            "Use the in-process single-worker pipeline and save observed Linear module "
            "quantization manifests for each generated variant."
        ),
    )
    parser.add_argument(
        "--save-audit-pngs",
        action="store_true",
        help="Save per-layer PNGs and an alpha composite next to each tensor artifact.",
    )
    parser.add_argument(
        "--only-variants",
        nargs="+",
        help="Restrict candidate variants. The BF16 reference is always included.",
    )
    return parser.parse_args()


def load_manifest(manifest_path: str | Path) -> LayeredManifest:
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest root must be a JSON object: {path}")
    return LayeredManifest.model_validate(data)


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def default_artifact_path(
    output_root: Path,
    variant_name: str,
    sample_id: str,
    artifact_format: str,
) -> Path:
    return output_root / "artifacts" / variant_name / f"{sample_id}.{artifact_format}"


def _load_safetensors(path: Path) -> Mapping[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path), device="cpu")


def load_tensor_payload(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        payload = dict(_load_safetensors(path))
    elif path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=True)
    else:
        raise ValueError(f"Unsupported tensor artifact suffix for {path}")
    if not isinstance(payload, dict):
        raise ValueError(f"Tensor artifact must load to a mapping: {path}")
    tensor_payload: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            tensor_payload[key] = value
    return tensor_payload


def load_layer_stack(path: Path) -> torch.Tensor:
    payload = load_tensor_payload(path)
    if "video" not in payload:
        raise ValueError(f"Layered artifact must contain a 'video' tensor: {path}")
    return normalize_layer_stack(payload["video"], path=str(path))


def normalize_layer_stack(layer_stack: torch.Tensor, *, path: str = "<memory>") -> torch.Tensor:
    if not isinstance(layer_stack, torch.Tensor):
        raise ValueError(f"Layer stack must be a torch.Tensor: {path}")
    if layer_stack.dim() == 4:
        layer_stack = layer_stack.unsqueeze(0)
    if layer_stack.dim() != 5:
        raise ValueError(
            f"Layer stack must have shape (layers,H,W,C) or (B,layers,H,W,C), "
            f"got {tuple(layer_stack.shape)} from {path}"
        )
    if layer_stack.shape[-1] not in (3, 4):
        raise ValueError(
            f"Layer stack channel dimension must be RGB/RGBA, got {tuple(layer_stack.shape)}"
        )
    if min(int(dim) for dim in layer_stack.shape) <= 0:
        raise ValueError(f"Layer stack has an empty dimension: {tuple(layer_stack.shape)}")
    return layer_stack.contiguous().cpu()


def _to_unit_float(layer_stack: torch.Tensor, *, path: str) -> torch.Tensor:
    if layer_stack.is_floating_point():
        tensor = layer_stack.to(torch.float32)
        min_value = float(tensor.amin().item())
        max_value = float(tensor.amax().item())
        if min_value >= 0.0 and max_value <= 1.0:
            return tensor
        if min_value >= 0.0 and max_value <= 255.0:
            return tensor / 255.0
        raise ValueError(
            f"Floating artifact values must be in [0,1] or [0,255], "
            f"got range [{min_value}, {max_value}] from {path}"
        )
    if not torch.is_floating_point(layer_stack):
        return layer_stack.to(torch.float32) / float(torch.iinfo(layer_stack.dtype).max)
    raise ValueError(f"Unsupported tensor dtype for {path}: {layer_stack.dtype}")


def validate_stack_against_sample(
    stack: torch.Tensor,
    sample: LayeredSample,
    *,
    variant_name: str,
    path: Path,
) -> None:
    if int(stack.shape[1]) != int(sample.layers):
        raise ValueError(
            f"Variant {variant_name!r} sample {sample.id!r} has {stack.shape[1]} layers "
            f"in {path}, expected {sample.layers}"
        )
    if sample.height is not None and int(stack.shape[2]) != int(sample.height):
        raise ValueError(
            f"Variant {variant_name!r} sample {sample.id!r} has height {stack.shape[2]} "
            f"in {path}, expected {sample.height}"
        )
    if sample.width is not None and int(stack.shape[3]) != int(sample.width):
        raise ValueError(
            f"Variant {variant_name!r} sample {sample.id!r} has width {stack.shape[3]} "
            f"in {path}, expected {sample.width}"
        )


def compute_psnr(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    mse = torch.mean((candidate - reference) ** 2).item()
    if mse == 0.0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def compute_global_ssim(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    candidate = candidate.reshape(-1)
    reference = reference.reshape(-1)
    mu_x = candidate.mean()
    mu_y = reference.mean()
    var_x = torch.mean((candidate - mu_x) ** 2)
    var_y = torch.mean((reference - mu_y) ** 2)
    cov_xy = torch.mean((candidate - mu_x) * (reference - mu_y))
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return float((numerator / denominator).clamp(min=-1.0, max=1.0).item())


def compute_layered_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    candidate_path: Path,
    reference_path: Path,
) -> dict[str, float]:
    if tuple(candidate.shape) != tuple(reference.shape):
        raise ValueError(
            f"Artifact shape mismatch: {candidate_path} has {tuple(candidate.shape)}, "
            f"{reference_path} has {tuple(reference.shape)}"
        )
    candidate_unit = _to_unit_float(candidate, path=str(candidate_path))
    reference_unit = _to_unit_float(reference, path=str(reference_path))
    return {
        "psnr": compute_psnr(candidate_unit, reference_unit),
        "ssim": compute_global_ssim(candidate_unit, reference_unit),
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _jsonify_metric(value: float) -> float | str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "nan"
    return value


def _jsonify_nonfinite(value: JsonMetadataValue | float) -> JsonMetadataValue:
    if isinstance(value, float):
        return _jsonify_metric(value)
    if isinstance(value, list):
        return [_jsonify_nonfinite(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonify_nonfinite(item) for key, item in value.items()}
    return value


def git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def linear_block_index(name: str) -> int | None:
    match = LINEAR_BLOCK_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def _module_matches_prefix(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def linear_role(name: str) -> str:
    if any(_module_matches_prefix(name, entry) for entry in LINEAR_NON_BACKBONE_IGNORE):
        return "non_backbone"
    if linear_block_index(name) is not None:
        return "transformer_block"
    return "other"


def _variant_k_edge_bf16(variant: LayeredVariant) -> int | None:
    value = variant.metadata.get("k_edge_bf16")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    if variant.name.startswith("k") and variant.name[1:].isdigit():
        return int(variant.name[1:])
    return None


def _linear_preserved_reason(name: str, variant: LayeredVariant) -> str | None:
    if variant.name == REFERENCE_VARIANT:
        return "bf16_reference"
    ignore = list(LINEAR_NON_BACKBONE_IGNORE)
    k_edge_bf16 = _variant_k_edge_bf16(variant)
    if k_edge_bf16 is not None:
        ignore.extend(f"transformer_blocks.{idx}" for idx in range(k_edge_bf16))
        ignore.extend(
            f"transformer_blocks.{idx}"
            for idx in range(
                LINEAR_TRANSFORMER_BLOCKS - k_edge_bf16,
                LINEAR_TRANSFORMER_BLOCKS,
            )
        )
    for entry in ignore:
        if _module_matches_prefix(name, entry):
            return entry
    return None


def _shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _dtype_of(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(dtype)


def _quant_method_name(module: Any) -> str:
    quant_method = getattr(module, "quant_method", None)
    if quant_method is None:
        quant_method = getattr(module, "linear_method", None)
    if quant_method is None:
        return "None"
    return quant_method.__class__.__name__


def _quant_algo_name(module: Any, quant_method_name: str) -> str:
    quant_config = getattr(module, "quant_config", None)
    quant_algo = getattr(quant_config, "quant_algo", None)
    if quant_algo is None and "FP8BlockScales" in quant_method_name:
        return LINEAR_BLOCK_SCALE_ALGO
    if quant_algo is None:
        return "None"
    name = getattr(quant_algo, "name", None)
    if isinstance(name, str):
        return name
    return str(quant_algo).split(".")[-1]


def is_linear_manifest_module(module: Any) -> bool:
    return module.__class__.__name__ == "Linear" and hasattr(module, "weight")


def linear_manifest_record(name: str, module: Any, variant: LayeredVariant) -> dict[str, Any]:
    quant_method = _quant_method_name(module)
    role = linear_role(name)
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    weight_scale = getattr(module, "weight_scale", None)
    return {
        "name": name,
        "class": module.__class__.__name__,
        "block_index": linear_block_index(name),
        "role": role,
        "is_backbone_block": role == "transformer_block",
        "preserved_reason": _linear_preserved_reason(name, variant),
        "quant_method": quant_method,
        "quant_algo": _quant_algo_name(module, quant_method),
        "weight_dtype": _dtype_of(weight),
        "weight_shape": _shape_of(weight),
        "weight_scale_dtype": _dtype_of(weight_scale),
        "weight_scale_shape": _shape_of(weight_scale),
        "bias_dtype": _dtype_of(bias),
        "bias_shape": _shape_of(bias),
        "in_features": getattr(module, "in_features", None),
        "out_features": getattr(module, "out_features", None),
        "tp_mode": str(getattr(module, "tp_mode", None)),
        "disable_deep_gemm": bool(getattr(module, "disable_deep_gemm", False)),
        "use_cute_dsl_blockscaling_mm": bool(
            getattr(module, "use_cute_dsl_blockscaling_mm", False)
        ),
    }


def collect_linear_manifest(
    named_modules: Iterator[tuple[str, Any]],
    variant: LayeredVariant,
) -> list[dict[str, Any]]:
    records = [
        linear_manifest_record(name, module, variant)
        for name, module in named_modules
        if is_linear_manifest_module(module)
    ]
    return sorted(records, key=lambda record: record["name"])


def write_linear_manifest(
    path: Path,
    named_modules: Iterator[tuple[str, Any]],
    variant: LayeredVariant,
) -> list[dict[str, Any]]:
    records = collect_linear_manifest(named_modules, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return records


def linear_quant_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = f"{record['weight_dtype']}|{record['quant_algo']}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _linear_manifest_metadata(output_root: Path, variant: LayeredVariant) -> dict[str, Any]:
    path = output_root / "linear_manifests" / variant.name / "linear_manifest.json"
    if not path.exists():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Linear manifest must contain a list: {path}")
    return {
        "linear_manifest_path": str(path),
        "linear_quant_counts": linear_quant_counts(records),
    }


def load_single_worker_pipeline(
    variant: LayeredVariant,
    *,
    manifest_dir: Path,
    purpose: str,
) -> Any:
    from tensorrt_llm._torch.visual_gen import PipelineLoader
    from tensorrt_llm.visual_gen import VisualGenArgs

    args = VisualGenArgs.from_yaml(
        resolve_path(variant.visual_gen_args, manifest_dir),
        model=variant.model,
    )
    if args.parallel_config.n_workers != 1:
        raise ValueError(
            f"{purpose} uses an in-process pipeline and requires "
            f"parallel_config.n_workers == 1, got {args.parallel_config.n_workers}"
        )

    loader = PipelineLoader(args, device="cuda:0")
    return loader.load(skip_warmup=args.compilation_config.skip_warmup)


def cleanup_pipeline(pipeline: Any) -> None:
    cleanup = getattr(pipeline, "cleanup", None)
    if cleanup is not None:
        cleanup()


def write_linear_manifest_with_local_pipeline(
    variant: LayeredVariant,
    *,
    manifest_dir: Path,
    linear_manifest_path: Path,
) -> None:
    pipeline = load_single_worker_pipeline(
        variant,
        manifest_dir=manifest_dir,
        purpose="Linear manifest emission",
    )
    try:
        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            raise ValueError("Pipeline does not expose a transformer for Linear manifest emission")
        write_linear_manifest(
            linear_manifest_path,
            transformer.named_modules(),
            variant,
        )
    finally:
        cleanup_pipeline(pipeline)


def _tensor_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_tensor_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tensor_to_cpu(item) for item in value)
    if isinstance(value, dict):
        return {key: _tensor_to_cpu(item) for key, item in value.items()}
    return value


def _first_tensor_output(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if hasattr(output, "sample") and isinstance(output.sample, torch.Tensor):
        return output.sample
    return None


@contextmanager
def capture_transformer_tuples(
    pipeline: Any,
    *,
    output_dir: Path,
    sample_id: str,
    variant_name: str,
    has_negative_prompt: bool,
) -> Iterator[None]:
    """Save transformer forward inputs and BF16 targets for QAT distillation."""
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise ValueError("Pipeline does not expose a transformer for tuple capture")

    output_dir.mkdir(parents=True, exist_ok=True)
    original_forward = transformer.forward
    call_index = 0

    def wrapped_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal call_index
        output = original_forward(*args, **kwargs)
        role = "cond"
        if has_negative_prompt and call_index % 2 == 1:
            role = "negative"
        target = _first_tensor_output(output)
        payload = {
            "sample_id": sample_id,
            "variant": variant_name,
            "call_index": call_index,
            "role": role,
            "args": _tensor_to_cpu(args),
            "kwargs": _tensor_to_cpu(kwargs),
            "target_output": None if target is None else target.detach().cpu(),
        }
        torch.save(payload, output_dir / f"tuple_{call_index:04d}_{role}.pt")
        call_index += 1
        return output

    transformer.forward = MethodType(wrapped_forward, transformer)
    try:
        yield
    finally:
        transformer.forward = original_forward


def build_visual_gen_params(sample: LayeredSample, defaults: VisualGenParams) -> VisualGenParams:
    if sample.num_inference_steps is None:
        raise ValueError(
            f"Sample {sample.id!r} has sigmas but no num_inference_steps. "
            "Current VisualGen generation path cannot pass sigmas through params; "
            "pre-generate this artifact or add num_inference_steps."
        )
    params = defaults.model_copy(deep=True)
    params.image = sample.image
    params.seed = sample.seed
    params.num_inference_steps = int(sample.num_inference_steps)
    params.negative_prompt = sample.negative_prompt
    if sample.height is not None and sample.width is not None:
        params.height = int(sample.height)
        params.width = int(sample.width)
    if sample.guidance_scale is not None:
        params.guidance_scale = float(sample.guidance_scale)
    if sample.max_sequence_length is not None:
        params.max_sequence_length = int(sample.max_sequence_length)
    params.num_images_per_prompt = 1

    extra_params = dict(params.extra_params or {})
    extra_params["layers"] = int(sample.layers)
    extra_params["resolution"] = int(sample.resolution)
    extra_params["cfg_normalize"] = bool(sample.cfg_normalize)
    extra_params["use_en_prompt"] = bool(sample.use_en_prompt)
    params.extra_params = extra_params
    return params


def visual_gen_defaults(visual_gen: Any) -> VisualGenParams:
    return visual_gen.default_params


def local_pipeline_defaults(pipeline: Any) -> VisualGenParams:
    kwargs = dict(pipeline.default_generation_params)
    extra = {key: spec.default for key, spec in pipeline.extra_param_specs.items()}
    if extra:
        kwargs["extra_params"] = extra
    return VisualGenParams(**kwargs)


def generate_with_visual_gen(
    sample: LayeredSample,
    variant: LayeredVariant,
    *,
    artifact_path: Path,
    manifest_dir: Path,
    artifact_format: str,
) -> Path:
    from tensorrt_llm.visual_gen import VisualGen, VisualGenArgs

    args = VisualGenArgs.from_yaml(
        resolve_path(variant.visual_gen_args, manifest_dir),
        model=variant.model,
    )
    sample_for_generation = sample.model_copy(
        update={"image": str(resolve_path(sample.image, manifest_dir))}
    )
    with VisualGen(model=variant.model, args=args) as visual_gen:
        params = build_visual_gen_params(sample_for_generation, visual_gen_defaults(visual_gen))
        output = visual_gen.generate(inputs=sample.prompt, params=params)
        if isinstance(output, list):
            raise ValueError("Qwen-Image-Layered quality samples must generate one output")
        saved = output.save(artifact_path, format=artifact_format, frame_rate=1.0)
        if not isinstance(saved, Path):
            raise ValueError("Expected VisualGenOutput.save to return a single path")
        return saved


def generate_with_local_pipeline(
    sample: LayeredSample,
    variant: LayeredVariant,
    *,
    artifact_path: Path,
    manifest_dir: Path,
    artifact_format: str,
    capture_root: Path | None,
    linear_manifest_path: Path | None,
) -> Path:
    from tensorrt_llm._torch.visual_gen import DiffusionRequest

    pipeline = load_single_worker_pipeline(
        variant,
        manifest_dir=manifest_dir,
        purpose="Transformer tuple capture and Linear manifest emission",
    )
    try:
        if linear_manifest_path is not None:
            transformer = getattr(pipeline, "transformer", None)
            if transformer is None:
                raise ValueError(
                    "Pipeline does not expose a transformer for Linear manifest emission"
                )
            write_linear_manifest(
                linear_manifest_path,
                transformer.named_modules(),
                variant,
            )
        sample_for_generation = sample.model_copy(
            update={"image": str(resolve_path(sample.image, manifest_dir))}
        )
        params = build_visual_gen_params(sample_for_generation, local_pipeline_defaults(pipeline))
        req = DiffusionRequest(request_id=0, prompt=[sample.prompt], params=params)

        context = (
            capture_transformer_tuples(
                pipeline,
                output_dir=capture_root,
                sample_id=sample.id,
                variant_name=variant.name,
                has_negative_prompt=sample.negative_prompt is not None
                and (params.guidance_scale or 0.0) > 1.0,
            )
            if capture_root is not None
            else nullcontext()
        )
        with context:
            output = pipeline.infer(req)
        visual_output = VisualGenOutput(
            request_id=0,
            image=output.image,
            video=output.video,
            audio=output.audio,
            frame_rate=output.frame_rate,
            audio_sample_rate=output.audio_sample_rate,
        )
        saved = visual_output.save(artifact_path, format=artifact_format, frame_rate=1.0)
        if not isinstance(saved, Path):
            raise ValueError("Expected VisualGenOutput.save to return a single path")
        return saved
    finally:
        cleanup_pipeline(pipeline)


def resolve_or_generate_artifact(
    sample: LayeredSample,
    variant: LayeredVariant,
    *,
    manifest_dir: Path,
    output_root: Path,
    artifact_format: str,
    load_existing: bool,
    overwrite: bool,
    capture_transformer_tuples_enabled: bool,
    emit_linear_manifests: bool,
) -> Path:
    explicit = variant.artifact_paths.get(sample.id)
    if explicit is not None:
        artifact_path = resolve_path(explicit, manifest_dir)
    else:
        artifact_path = default_artifact_path(
            output_root,
            variant.name,
            sample.id,
            artifact_format,
        )

    linear_manifest_path = None
    if emit_linear_manifests:
        linear_manifest_path = (
            output_root / "linear_manifests" / variant.name / "linear_manifest.json"
        )

    if artifact_path.exists() and not overwrite:
        if linear_manifest_path is not None and not linear_manifest_path.exists():
            write_linear_manifest_with_local_pipeline(
                variant,
                manifest_dir=manifest_dir,
                linear_manifest_path=linear_manifest_path,
            )
        return artifact_path
    if load_existing:
        raise FileNotFoundError(f"Missing required artifact: {artifact_path}")
    if explicit is not None and overwrite:
        raise ValueError(
            f"Refusing to overwrite explicit artifact path for variant {variant.name!r}, "
            f"sample {sample.id!r}: {artifact_path}"
        )

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    capture_root = None
    if capture_transformer_tuples_enabled and variant.name == REFERENCE_VARIANT:
        capture_root = output_root / "teacher_tuples" / variant.name / sample.id
    if capture_transformer_tuples_enabled or emit_linear_manifests:
        return generate_with_local_pipeline(
            sample,
            variant,
            artifact_path=artifact_path,
            manifest_dir=manifest_dir,
            artifact_format=artifact_format,
            capture_root=capture_root,
            linear_manifest_path=linear_manifest_path,
        )
    return generate_with_visual_gen(
        sample,
        variant,
        artifact_path=artifact_path,
        manifest_dir=manifest_dir,
        artifact_format=artifact_format,
    )


def _save_audit_pngs(layer_stack: torch.Tensor, output_dir: Path) -> None:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    stack = _to_unit_float(layer_stack, path=str(output_dir))[0]
    pixels = (stack.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
    images = []
    for layer_index, layer in enumerate(pixels):
        mode = "RGBA" if layer.shape[-1] == 4 else "RGB"
        image = Image.fromarray(layer, mode=mode)
        image.save(output_dir / f"layer_{layer_index:02d}.png")
        images.append(image.convert("RGBA"))
    if images:
        composite = Image.new("RGBA", images[0].size, (0, 0, 0, 0))
        for image in reversed(images):
            composite.alpha_composite(image)
        composite.save(output_dir / "composite.png")


def _variant_selection(
    variants: list[LayeredVariant],
    only_variants: list[str] | None,
) -> list[LayeredVariant]:
    if only_variants is None:
        return variants
    requested = set(only_variants)
    requested.add(REFERENCE_VARIANT)
    selected = [variant for variant in variants if variant.name in requested]
    missing = requested - {variant.name for variant in selected}
    if missing:
        raise ValueError(f"Requested variants are not in manifest: {sorted(missing)}")
    if len(selected) < 2:
        raise ValueError("Variant selection must include bf16 and at least one candidate")
    return selected


def evaluate_manifest(
    manifest: LayeredManifest,
    *,
    manifest_path: Path,
    output_json: Path | None,
    artifact_format: str,
    load_existing: bool,
    overwrite: bool,
    capture_transformer_tuples_enabled: bool,
    save_audit_pngs: bool,
    only_variants: list[str] | None = None,
    emit_linear_manifests: bool = False,
) -> dict[str, JsonMetadataValue]:
    manifest_dir = manifest_path.resolve().parent
    output_root = resolve_path(manifest.output_root, manifest_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    selected_variants = _variant_selection(manifest.variants, only_variants)
    artifact_records: dict[str, dict[str, Path]] = {
        variant.name: {} for variant in selected_variants
    }
    stack_cache: dict[tuple[str, str], torch.Tensor] = {}

    for variant in selected_variants:
        for sample in manifest.samples:
            artifact_path = resolve_or_generate_artifact(
                sample,
                variant,
                manifest_dir=manifest_dir,
                output_root=output_root,
                artifact_format=artifact_format,
                load_existing=load_existing,
                overwrite=overwrite,
                capture_transformer_tuples_enabled=capture_transformer_tuples_enabled,
                emit_linear_manifests=emit_linear_manifests,
            )
            stack = load_layer_stack(artifact_path)
            validate_stack_against_sample(
                stack,
                sample,
                variant_name=variant.name,
                path=artifact_path,
            )
            artifact_records[variant.name][sample.id] = artifact_path
            stack_cache[(variant.name, sample.id)] = stack
            if save_audit_pngs:
                _save_audit_pngs(
                    stack,
                    output_root / "audit_pngs" / variant.name / sample.id,
                )

    comparisons: list[dict[str, JsonMetadataValue]] = []
    aggregates: dict[str, dict[str, JsonMetadataValue]] = {}
    for variant in selected_variants:
        if variant.name == REFERENCE_VARIANT:
            continue
        psnr_values: list[float] = []
        ssim_values: list[float] = []
        for sample in manifest.samples:
            candidate_path = artifact_records[variant.name][sample.id]
            reference_path = artifact_records[REFERENCE_VARIANT][sample.id]
            metrics = compute_layered_metrics(
                stack_cache[(variant.name, sample.id)],
                stack_cache[(REFERENCE_VARIANT, sample.id)],
                candidate_path=candidate_path,
                reference_path=reference_path,
            )
            psnr_values.append(metrics["psnr"])
            ssim_values.append(metrics["ssim"])
            comparisons.append(
                {
                    "sample_id": sample.id,
                    "variant": variant.name,
                    "reference_variant": REFERENCE_VARIANT,
                    "artifact_path": str(candidate_path),
                    "reference_artifact_path": str(reference_path),
                    "psnr": _jsonify_metric(metrics["psnr"]),
                    "ssim": _jsonify_metric(metrics["ssim"]),
                }
            )
        aggregates[variant.name] = {
            "num_samples": len(psnr_values),
            "psnr_min": _jsonify_metric(min(psnr_values)),
            "psnr_mean": _jsonify_metric(_mean(psnr_values)),
            "ssim_min": _jsonify_metric(min(ssim_values)),
            "ssim_mean": _jsonify_metric(_mean(ssim_values)),
        }

    metrics_json = {
        "manifest_path": str(manifest_path.resolve()),
        "output_root": str(output_root),
        "artifact_format": artifact_format,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_commit(Path.cwd()),
        "supported_variant_names": sorted(SUPPORTED_VARIANT_NAMES),
        "samples": [
            {
                **sample.model_dump(),
                "image": str(resolve_path(sample.image, manifest_dir)),
            }
            for sample in manifest.samples
        ],
        "variants": {
            variant.name: {
                "model": variant.model,
                "visual_gen_args": str(resolve_path(variant.visual_gen_args, manifest_dir)),
                "checkpoint_path": variant.checkpoint_path,
                "metadata": variant.metadata,
                "artifacts": {
                    sample_id: str(path)
                    for sample_id, path in artifact_records[variant.name].items()
                },
                **_linear_manifest_metadata(output_root, variant),
            }
            for variant in selected_variants
        },
        "comparisons": comparisons,
        "aggregates": aggregates,
    }

    target_json = output_json or (output_root / "metrics.json")
    target_json.parent.mkdir(parents=True, exist_ok=True)
    with target_json.open("w", encoding="utf-8") as f:
        json.dump(_jsonify_nonfinite(metrics_json), f, indent=2, sort_keys=True)
        f.write("\n")
    return metrics_json


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    output_json = Path(args.output_json).expanduser() if args.output_json else None
    metrics = evaluate_manifest(
        manifest,
        manifest_path=manifest_path,
        output_json=output_json,
        artifact_format=args.artifact_format,
        load_existing=args.load_existing,
        overwrite=args.overwrite,
        capture_transformer_tuples_enabled=args.capture_transformer_tuples,
        save_audit_pngs=args.save_audit_pngs,
        only_variants=args.only_variants,
        emit_linear_manifests=args.emit_linear_manifests,
    )
    print(json.dumps(_jsonify_nonfinite(metrics["aggregates"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
