# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export Qwen-Image-Layered QAT checkpoints as static MXFP8 transformer weights."""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

import torch
import torch.nn.functional as F
import yaml
from pydantic import Field, PositiveInt, model_validator

from tensorrt_llm.llmapi.utils import StrictBaseModel

EXPORT_FORMAT = "qwen_image_layered_static_mxfp8_export_v1"
LAYERED_TRAINING_FORMAT = "qwen_image_layered_mxfp8_qat_adapter_v1"
QWEN_IMAGE_TRAINING_FORMAT = "qwen_image_mxfp8_qat_adapter_v1"
SUPPORTED_QAT_CHECKPOINT_FORMATS = (LAYERED_TRAINING_FORMAT, QWEN_IMAGE_TRAINING_FORMAT)
_FAKE_MXFP8_MODULE_NAMES = {
    "tensorrt_llm.bindings",
    "tensorrt_llm._torch.visual_gen.quantization",
    "tensorrt_llm._torch.visual_gen.quantization.fake_mxfp8",
}
QWEN_BLOCK_LINEAR_POLICY = "qwen_block_linears_840"
EXPLICIT_TARGET_POLICY = "explicit"
QWEN_LAYER_COUNT = 60
QWEN_STATIC_BF16_EXCLUSIONS = ("img_in", "txt_in", "norm_out.linear", "proj_out")
STATIC_INPUT_SCALE = 1.0
SCALE_EPS = 1.0e-12


def _load_source_fake_mxfp8() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "tensorrt_llm"
        / "_torch"
        / "visual_gen"
        / "quantization"
        / "fake_mxfp8.py"
    )
    spec = importlib.util.spec_from_file_location("_qwen_image_export_fake_mxfp8", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load fake_mxfp8 module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


try:
    from tensorrt_llm._torch.visual_gen.quantization.fake_mxfp8 import (
        FP8_E4M3_MAX,
        MXFP8_BLOCK_SIZE,
        QWEN_IMAGE_BLOCK_LINEAR_SUFFIXES,
        normalize_qwen_module_name,
    )
except ModuleNotFoundError as error:
    if error.name not in _FAKE_MXFP8_MODULE_NAMES:
        raise
    _fake_mxfp8 = _load_source_fake_mxfp8()
    FP8_E4M3_MAX = _fake_mxfp8.FP8_E4M3_MAX
    MXFP8_BLOCK_SIZE = _fake_mxfp8.MXFP8_BLOCK_SIZE
    QWEN_IMAGE_BLOCK_LINEAR_SUFFIXES = _fake_mxfp8.QWEN_IMAGE_BLOCK_LINEAR_SUFFIXES
    normalize_qwen_module_name = _fake_mxfp8.normalize_qwen_module_name


QWEN_BLOCK_LINEAR_TARGET_COUNT = QWEN_LAYER_COUNT * len(QWEN_IMAGE_BLOCK_LINEAR_SUFFIXES)

TargetPolicyName = Literal["qwen_block_linears_840", "explicit"]
ScaleModeName = Literal["fp32_block_scale"]


class QwenImageLayeredQatExportConfig(StrictBaseModel):
    """Strict config for offline Qwen-Image-Layered static MXFP8 export."""

    model: str = Field(description="BF16 Qwen-Image-Layered checkpoint root.")
    qat_checkpoint: str = Field(description="QAT training checkpoint produced by the native loop.")
    output_dir: str = Field(description="Output diffusers-style checkpoint directory.")
    target_policy: TargetPolicyName = Field(
        default=QWEN_BLOCK_LINEAR_POLICY,
        description="Static MXFP8 target policy.",
    )
    target_layers: list[str] | None = Field(
        default=None,
        description="Explicit normalized Linear module names when target_policy is explicit.",
    )
    expected_target_count: PositiveInt | None = Field(
        default=QWEN_BLOCK_LINEAR_TARGET_COUNT,
        description="Expected number of exported MXFP8 Linear weights.",
    )
    block_size: PositiveInt = Field(
        default=MXFP8_BLOCK_SIZE,
        description="MXFP8 block size for exported weight scales.",
    )
    scale_mode: ScaleModeName = Field(
        default="fp32_block_scale",
        description="Static block-scale mode for exported weights.",
    )
    weight_filename: str = Field(
        default="diffusion_pytorch_model.safetensors",
        description="Output transformer safetensors file name.",
    )
    copy_model_index: bool = Field(
        default=True,
        description="Copy source model_index.json when present; otherwise write a minimal one.",
    )

    @model_validator(mode="after")
    def _validate_export_contract(self) -> "QwenImageLayeredQatExportConfig":
        if self.block_size != MXFP8_BLOCK_SIZE:
            raise ValueError(f"MXFP8 static export requires block_size={MXFP8_BLOCK_SIZE}")
        if self.scale_mode != "fp32_block_scale":
            raise ValueError("static export currently supports only fp32_block_scale")
        if self.target_policy == EXPLICIT_TARGET_POLICY:
            if not self.target_layers:
                raise ValueError("explicit target_policy requires target_layers")
            if self.expected_target_count is None:
                raise ValueError("explicit target_policy requires expected_target_count")
            if self.expected_target_count != len(self.target_layers):
                raise ValueError("expected_target_count must match explicit target_layers")
        elif self.target_layers:
            raise ValueError("target_layers are only valid with target_policy=explicit")
        if self.target_policy == QWEN_BLOCK_LINEAR_POLICY:
            if self.expected_target_count != QWEN_BLOCK_LINEAR_TARGET_COUNT:
                raise ValueError(
                    f"{QWEN_BLOCK_LINEAR_POLICY} requires "
                    f"expected_target_count={QWEN_BLOCK_LINEAR_TARGET_COUNT}"
                )
        return self


@dataclass(frozen=True)
class TargetPolicy:
    """Resolved static quantization target policy."""

    name: str
    target_layers: tuple[str, ...]
    bf16_exclusions: tuple[str, ...]
    expected_target_count: int


@dataclass(frozen=True)
class QatCheckpoint:
    """Validated QAT checkpoint payload."""

    path: Path
    format: str
    config: dict[str, object]
    trainable_state_dict: dict[str, torch.Tensor]
    injections: tuple[dict[str, object], ...]
    best_validation_loss: float | None


@dataclass(frozen=True)
class ExportResult:
    """Files and counts produced by one export."""

    output_dir: Path
    weight_path: Path
    transformer_config_path: Path
    provenance_path: Path
    quantized_weight_count: int
    merged_parameter_count: int
    target_policy: TargetPolicy


def load_qat_export_config(path: str | Path) -> QwenImageLayeredQatExportConfig:
    """Load a strict static export config from JSON or YAML."""
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"QAT export config must contain a mapping: {config_path}")
    return QwenImageLayeredQatExportConfig(**data)


def build_target_policy(config: QwenImageLayeredQatExportConfig) -> TargetPolicy:
    """Resolve the configured static MXFP8 target policy."""
    if config.target_policy == QWEN_BLOCK_LINEAR_POLICY:
        target_layers = tuple(
            f"transformer_blocks.{layer_idx}.{suffix}"
            for layer_idx in range(QWEN_LAYER_COUNT)
            for suffix in QWEN_IMAGE_BLOCK_LINEAR_SUFFIXES
        )
        return TargetPolicy(
            name=QWEN_BLOCK_LINEAR_POLICY,
            target_layers=target_layers,
            bf16_exclusions=QWEN_STATIC_BF16_EXCLUSIONS,
            expected_target_count=QWEN_BLOCK_LINEAR_TARGET_COUNT,
        )
    if config.target_policy == EXPLICIT_TARGET_POLICY:
        assert config.target_layers is not None
        assert config.expected_target_count is not None
        normalized = tuple(_normalize_module_name(layer) for layer in config.target_layers)
        if len(set(normalized)) != len(normalized):
            raise ValueError("target_layers contain duplicate module names")
        return TargetPolicy(
            name=EXPLICIT_TARGET_POLICY,
            target_layers=normalized,
            bf16_exclusions=(),
            expected_target_count=int(config.expected_target_count),
        )
    raise ValueError(f"Unsupported target_policy: {config.target_policy}")


def export_qwen_image_layered_qat(config: QwenImageLayeredQatExportConfig) -> ExportResult:
    """Export a QAT checkpoint to a static MXFP8 transformer checkpoint."""
    source_model = Path(config.model)
    source_transformer_dir = _source_transformer_dir(source_model)
    output_dir = Path(config.output_dir)
    output_transformer_dir = output_dir / "transformer"
    output_transformer_dir.mkdir(parents=True, exist_ok=True)

    target_policy = build_target_policy(config)
    source_state = load_transformer_state_dict(source_transformer_dir)
    qat_checkpoint = load_qat_checkpoint(Path(config.qat_checkpoint))
    merged_state, merged_parameter_names = merge_qat_trainable_state(
        source_state,
        qat_checkpoint.trainable_state_dict,
        qat_config=qat_checkpoint.config,
        injections=qat_checkpoint.injections,
    )
    exported_state, quantized_layers = export_static_mxfp8_state_dict(
        merged_state,
        target_layers=target_policy.target_layers,
        block_size=int(config.block_size),
    )
    if len(quantized_layers) != target_policy.expected_target_count:
        raise ValueError(
            f"Expected {target_policy.expected_target_count} static MXFP8 weights for "
            f"{target_policy.name}, exported {len(quantized_layers)}."
        )

    transformer_config = _load_transformer_config(source_transformer_dir)
    transformer_config["quantization_config"] = build_static_quantization_config(target_policy)
    _write_model_index(source_model, output_dir, copy_source=config.copy_model_index)
    transformer_config_path = output_transformer_dir / "config.json"
    transformer_config_path.write_text(
        json.dumps(transformer_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    weight_path = output_transformer_dir / config.weight_filename
    _save_safetensors(exported_state, weight_path)
    provenance = build_export_provenance(
        config=config,
        source_model=source_model,
        source_transformer_dir=source_transformer_dir,
        qat_checkpoint=qat_checkpoint,
        target_policy=target_policy,
        quantized_layers=quantized_layers,
        merged_parameter_names=merged_parameter_names,
        weight_path=weight_path,
        transformer_config_path=transformer_config_path,
    )
    provenance_path = output_dir / "export_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ExportResult(
        output_dir=output_dir,
        weight_path=weight_path,
        transformer_config_path=transformer_config_path,
        provenance_path=provenance_path,
        quantized_weight_count=len(quantized_layers),
        merged_parameter_count=len(merged_parameter_names),
        target_policy=target_policy,
    )


def load_transformer_state_dict(transformer_dir: Path) -> dict[str, torch.Tensor]:
    """Load all safetensors shards from a transformer directory."""
    state_dict: dict[str, torch.Tensor] = {}
    for path in _find_safetensor_weight_files(transformer_dir):
        from safetensors.torch import load_file

        state_dict.update(load_file(str(path), device="cpu"))
    if not state_dict:
        raise ValueError(f"No transformer weights were loaded from {transformer_dir}")
    return _remap_qwen_checkpoint_keys(state_dict)


def load_qat_checkpoint(path: Path) -> QatCheckpoint:
    """Load and validate a native QAT training checkpoint."""
    if not path.exists():
        raise ValueError(f"QAT checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"QAT checkpoint must be a dict: {path}")
    checkpoint_format = payload.get("format")
    if checkpoint_format not in SUPPORTED_QAT_CHECKPOINT_FORMATS:
        raise ValueError(
            f"Unsupported QAT checkpoint format {checkpoint_format!r}; "
            f"expected one of {SUPPORTED_QAT_CHECKPOINT_FORMATS!r}."
        )
    trainable_state = payload.get("trainable_state_dict")
    if not isinstance(trainable_state, dict) or not trainable_state:
        raise ValueError("QAT checkpoint is missing a non-empty trainable_state_dict")
    checked_state: dict[str, torch.Tensor] = {}
    for name, tensor in trainable_state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("QAT trainable_state_dict must map names to tensors")
        checked_state[name] = tensor.detach().cpu()
    injections = payload.get("injections", ())
    if not isinstance(injections, list):
        raise ValueError("QAT checkpoint injections must be a list")
    best_validation_loss = payload.get("best_validation_loss")
    if best_validation_loss is not None:
        best_validation_loss = float(best_validation_loss)
    qat_config = payload.get("config", {})
    if qat_config is None:
        qat_config = {}
    if not isinstance(qat_config, dict):
        raise ValueError("QAT checkpoint config must be a mapping when present")
    return QatCheckpoint(
        path=path,
        format=str(checkpoint_format),
        config=qat_config,
        trainable_state_dict=checked_state,
        injections=tuple(dict(item) for item in injections),
        best_validation_loss=best_validation_loss,
    )


def merge_qat_trainable_state(
    source_state: dict[str, torch.Tensor],
    trainable_state: dict[str, torch.Tensor],
    *,
    qat_config: Mapping[str, object] | None = None,
    injections: tuple[dict[str, object], ...] = (),
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Merge partial-unfreeze or LoRA QAT trainables into a BF16 transformer state dict."""
    merged = dict(source_state)
    merged_names: list[str] = []
    lora_state: dict[str, dict[str, torch.Tensor]] = {}
    for qat_name, tensor in trainable_state.items():
        lora_name = _split_lora_trainable_name(qat_name)
        if lora_name is not None:
            module_name, parameter_name = lora_name
            lora_state.setdefault(module_name, {})[parameter_name] = tensor
            continue
        target_name = _qat_trainable_name_to_checkpoint_name(qat_name)
        if target_name not in source_state:
            raise ValueError(
                f"QAT trainable tensor {qat_name!r} maps to missing checkpoint key {target_name!r}."
            )
        source_tensor = source_state[target_name]
        if tuple(source_tensor.shape) != tuple(tensor.shape):
            raise ValueError(
                f"QAT tensor {qat_name!r} shape {tuple(tensor.shape)} does not match "
                f"checkpoint key {target_name!r} shape {tuple(source_tensor.shape)}."
            )
        if not torch.is_floating_point(tensor):
            raise ValueError(f"QAT tensor {qat_name!r} must be floating point")
        merged[target_name] = tensor.to(dtype=source_tensor.dtype).contiguous()
        merged_names.append(target_name)
    if lora_state:
        _validate_lora_injections(lora_state, injections)
        merged_names.extend(_merge_lora_adapter_state(merged, source_state, lora_state, qat_config))
    return merged, tuple(sorted(set(merged_names)))


def export_static_mxfp8_state_dict(
    source_state: dict[str, torch.Tensor],
    *,
    target_layers: tuple[str, ...],
    block_size: int = MXFP8_BLOCK_SIZE,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Convert selected Linear weights to static FP8 block-scale tensors."""
    targets = set(target_layers)
    exported: dict[str, torch.Tensor] = {}
    quantized_layers: list[str] = []
    for key, tensor in source_state.items():
        module_name, suffix = _split_parameter_name(key)
        if suffix == "weight" and module_name in targets:
            qweight, weight_scale = quantize_static_mxfp8_weight(
                tensor,
                block_size=block_size,
                tensor_name=key,
            )
            exported[key] = qweight.cpu().contiguous()
            exported[f"{module_name}.weight_scale"] = weight_scale.cpu().contiguous()
            exported[f"{module_name}.input_scale"] = torch.tensor(
                STATIC_INPUT_SCALE, dtype=torch.float32
            )
            exported[f"{module_name}.inv_input_scale"] = torch.tensor(
                1.0 / STATIC_INPUT_SCALE, dtype=torch.float32
            )
            quantized_layers.append(module_name)
        elif key not in exported:
            exported[key] = tensor.detach().cpu().contiguous()

    missing = sorted(targets - set(quantized_layers))
    if missing:
        raise ValueError(f"Static export target layers are missing weights: {missing[:5]}")
    return exported, tuple(sorted(quantized_layers))


def quantize_static_mxfp8_weight(
    weight: torch.Tensor,
    *,
    block_size: int = MXFP8_BLOCK_SIZE,
    tensor_name: str = "weight",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight tensor with FP8 E4M3 and 128x128 block scales."""
    _validate_weight_matrix(weight, block_size=block_size, tensor_name=tensor_name)
    out_features, in_features = weight.shape
    num_blocks_out = (out_features + block_size - 1) // block_size
    num_blocks_in = (in_features + block_size - 1) // block_size
    pad_out = num_blocks_out * block_size - out_features
    pad_in = num_blocks_in * block_size - in_features
    source = weight.detach().to(torch.float32)
    if pad_out or pad_in:
        source = F.pad(source, (0, pad_in, 0, pad_out))

    rows_per_block = (
        source.reshape(num_blocks_out, block_size, num_blocks_in, block_size)
        .permute(0, 2, 1, 3)
        .reshape(num_blocks_out * num_blocks_in, block_size * block_size)
        .contiguous()
    )
    scales = (rows_per_block.abs().amax(dim=1) / FP8_E4M3_MAX).clamp_min(SCALE_EPS)
    qrows = (rows_per_block / scales.unsqueeze(1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    qrows = qrows.to(torch.float8_e4m3fn)
    qweight = (
        qrows.reshape(num_blocks_out, num_blocks_in, block_size, block_size)
        .permute(0, 2, 1, 3)
        .reshape(num_blocks_out * block_size, num_blocks_in * block_size)
    )[:out_features, :in_features].contiguous()
    return qweight, scales.reshape(num_blocks_out, num_blocks_in).to(torch.float32)


def build_static_quantization_config(target_policy: TargetPolicy) -> dict[str, object]:
    """Build ModelOpt-style metadata for static FP8 block-scale weights."""
    return {
        "quant_algo": "FP8_BLOCK_SCALES",
        "config_groups": {
            "default": {
                "weights": {
                    "dynamic": False,
                    "group_size": MXFP8_BLOCK_SIZE,
                },
                "input_activations": {
                    "dynamic": False,
                },
            }
        },
        "ignore": list(target_policy.bf16_exclusions),
        "target_policy": target_policy.name,
        "target_count": target_policy.expected_target_count,
    }


def build_export_provenance(
    *,
    config: QwenImageLayeredQatExportConfig,
    source_model: Path,
    source_transformer_dir: Path,
    qat_checkpoint: QatCheckpoint,
    target_policy: TargetPolicy,
    quantized_layers: tuple[str, ...],
    merged_parameter_names: tuple[str, ...],
    weight_path: Path,
    transformer_config_path: Path,
) -> dict[str, object]:
    """Build a JSON-serializable provenance record for the export."""
    return {
        "format": EXPORT_FORMAT,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "config": config.model_dump(),
        "source_model": str(source_model),
        "source_transformer_dir": str(source_transformer_dir),
        "qat_checkpoint": str(qat_checkpoint.path),
        "qat_checkpoint_format": qat_checkpoint.format,
        "qat_best_validation_loss": qat_checkpoint.best_validation_loss,
        "qat_injection_count": len(qat_checkpoint.injections),
        "target_policy": target_policy.name,
        "target_count": target_policy.expected_target_count,
        "bf16_exclusions": list(target_policy.bf16_exclusions),
        "quantized_weight_count": len(quantized_layers),
        "quantized_layers": list(quantized_layers),
        "merged_parameter_count": len(merged_parameter_names),
        "merged_parameter_names": list(merged_parameter_names),
        "weight_path": str(weight_path),
        "transformer_config_path": str(transformer_config_path),
        "quantization_config": build_static_quantization_config(target_policy),
    }


def _source_transformer_dir(model: Path) -> Path:
    if (model / "transformer" / "config.json").exists():
        return model / "transformer"
    if (model / "config.json").exists():
        return model
    raise ValueError(f"Could not find transformer/config.json under {model}")


def _load_transformer_config(transformer_dir: Path) -> dict[str, object]:
    config_path = transformer_dir / "config.json"
    if not config_path.exists():
        raise ValueError(f"Transformer config does not exist: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Transformer config must be a JSON object: {config_path}")
    return data


def _find_safetensor_weight_files(transformer_dir: Path) -> tuple[Path, ...]:
    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    if not index_path.exists():
        index_path = transformer_dir / "model.safetensors.index.json"
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid safetensors weight index: {index_path}")
        return tuple(sorted({transformer_dir / str(name) for name in weight_map.values()}))
    files = tuple(sorted(transformer_dir.glob("*.safetensors")))
    if not files:
        raise ValueError(f"No safetensors files found in {transformer_dir}")
    if len(files) > 1:
        files = tuple(path for path in files if "consolidated" not in path.name)
    return files


def _save_safetensors(state_dict: dict[str, torch.Tensor], path: Path) -> None:
    from safetensors.torch import save_file

    save_file(state_dict, str(path))


def _remap_qwen_checkpoint_keys(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    from tensorrt_llm._torch.visual_gen.models.qwen_image.transformer_qwen_image import (
        _remap_checkpoint_keys,
    )

    return _remap_checkpoint_keys(weights)


def _write_model_index(source_model: Path, output_dir: Path, *, copy_source: bool) -> None:
    source_model_index = source_model / "model_index.json"
    output_model_index = output_dir / "model_index.json"
    if copy_source and source_model_index.exists():
        shutil.copyfile(source_model_index, output_model_index)
        return
    output_model_index.write_text(
        json.dumps(
            {
                "_class_name": "QwenImageLayeredPipeline",
                "transformer": ["diffusers", "QwenImageTransformer2DModel"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _split_lora_trainable_name(qat_name: str) -> tuple[str, str] | None:
    normalized = _normalize_module_name(qat_name)
    for parameter_name in ("lora_down.weight", "lora_up.weight"):
        suffix = f".{parameter_name}"
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)], parameter_name
    return None


def _validate_lora_injections(
    lora_state: Mapping[str, Mapping[str, torch.Tensor]],
    injections: tuple[dict[str, object], ...],
) -> None:
    if not injections:
        return
    injected_lora_modules = {
        _normalize_module_name(str(injection.get("module_name")))
        for injection in injections
        if _injection_has_lora_trainables(injection)
    }
    if not injected_lora_modules:
        return
    unexpected_modules = sorted(set(lora_state) - injected_lora_modules)
    if unexpected_modules:
        raise ValueError(
            "LoRA trainable tensors are missing matching injection records: "
            f"{unexpected_modules[:5]}"
        )


def _injection_has_lora_trainables(injection: Mapping[str, object]) -> bool:
    trainable_names = injection.get("trainable_parameter_names")
    if not isinstance(trainable_names, list):
        return False
    return "lora_down.weight" in trainable_names and "lora_up.weight" in trainable_names


def _merge_lora_adapter_state(
    merged_state: dict[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor],
    lora_state: Mapping[str, Mapping[str, torch.Tensor]],
    qat_config: Mapping[str, object] | None,
) -> tuple[str, ...]:
    merged_names: list[str] = []
    for module_name, tensors in lora_state.items():
        target_name = f"{module_name}.weight"
        if target_name not in source_state:
            raise ValueError(
                f"LoRA module {module_name!r} maps to missing checkpoint key {target_name!r}."
            )
        lora_down = tensors.get("lora_down.weight")
        lora_up = tensors.get("lora_up.weight")
        if lora_down is None or lora_up is None:
            raise ValueError(
                f"LoRA module {module_name!r} requires lora_down.weight and lora_up.weight"
            )
        source_tensor = source_state[target_name]
        _validate_lora_pair(
            module_name=module_name,
            source_tensor=source_tensor,
            lora_down=lora_down,
            lora_up=lora_up,
        )
        rank = int(lora_down.shape[0])
        scaling = _lora_scaling_from_config(qat_config, rank=rank, module_name=module_name)
        delta = torch.matmul(lora_up.float(), lora_down.float()) * scaling
        merged_state[target_name] = (
            source_tensor.float().add(delta).to(dtype=source_tensor.dtype).contiguous()
        )
        merged_names.append(target_name)
    return tuple(merged_names)


def _validate_lora_pair(
    *,
    module_name: str,
    source_tensor: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
) -> None:
    if not torch.is_floating_point(lora_down) or not torch.is_floating_point(lora_up):
        raise ValueError(f"LoRA tensors for {module_name!r} must be floating point")
    if lora_down.ndim != 2 or lora_up.ndim != 2:
        raise ValueError(f"LoRA tensors for {module_name!r} must be 2D")
    rank, in_features = lora_down.shape
    out_features, up_rank = lora_up.shape
    if rank <= 0 or up_rank != rank:
        raise ValueError(
            f"LoRA tensors for {module_name!r} have incompatible ranks: "
            f"down={tuple(lora_down.shape)}, up={tuple(lora_up.shape)}"
        )
    if tuple(source_tensor.shape) != (out_features, in_features):
        raise ValueError(
            f"LoRA tensors for {module_name!r} map to shape {(out_features, in_features)}, "
            f"but checkpoint weight has shape {tuple(source_tensor.shape)}."
        )


def _lora_scaling_from_config(
    qat_config: Mapping[str, object] | None,
    *,
    rank: int,
    module_name: str,
) -> float:
    if qat_config is None:
        raise ValueError(
            "LoRA adapter QAT merge requires checkpoint config with lora_rank and lora_alpha"
        )
    config_rank = qat_config.get("lora_rank")
    config_alpha = qat_config.get("lora_alpha")
    if config_rank is None or config_alpha is None:
        raise ValueError(
            "LoRA adapter QAT merge requires checkpoint config with lora_rank and lora_alpha"
        )
    if int(config_rank) != rank:
        raise ValueError(
            f"LoRA rank mismatch for {module_name!r}: checkpoint config has {config_rank}, "
            f"adapter tensor rank is {rank}."
        )
    return float(config_alpha) / float(rank)


def _qat_trainable_name_to_checkpoint_name(qat_name: str) -> str:
    if ".lora_down." in qat_name or ".lora_up." in qat_name:
        raise ValueError(
            f"QAT tensor {qat_name!r} is a LoRA adapter parameter; static export currently "
            "supports partial-unfreeze checkpoints with merged Linear weights."
        )
    normalized = _normalize_module_name(qat_name)
    if ".linear." in normalized:
        normalized = normalized.replace(".linear.", ".")
    if not normalized.endswith((".weight", ".bias")):
        raise ValueError(f"Unsupported QAT trainable tensor name: {qat_name!r}")
    return normalized


def _normalize_module_name(name: str) -> str:
    return normalize_qwen_module_name(name)


def _split_parameter_name(name: str) -> tuple[str, str]:
    module_name, sep, suffix = name.rpartition(".")
    if not sep:
        raise ValueError(f"Parameter name has no module prefix: {name!r}")
    return module_name, suffix


def _validate_weight_matrix(weight: torch.Tensor, *, block_size: int, tensor_name: str) -> None:
    if block_size != MXFP8_BLOCK_SIZE:
        raise ValueError(f"MXFP8 static export requires block_size={MXFP8_BLOCK_SIZE}")
    if not torch.is_floating_point(weight):
        raise TypeError(f"{tensor_name} must be floating point before MXFP8 export")
    if weight.ndim != 2:
        raise ValueError(f"{tensor_name} must be 2D, got shape {tuple(weight.shape)}")
    if weight.shape[0] == 0 or weight.shape[1] == 0:
        raise ValueError(f"{tensor_name} dimensions must be non-empty")
    if weight.shape[1] < block_size:
        raise ValueError(
            f"{tensor_name} in_features must be at least {block_size}, got {weight.shape[1]}"
        )


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
        description="Export Qwen-Image-Layered QAT weights as static MXFP8."
    )
    parser.add_argument("--config", required=True, help="QAT export config JSON/YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_qat_export_config(args.config)
    result = export_qwen_image_layered_qat(config)
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "weight_path": str(result.weight_path),
                "transformer_config_path": str(result.transformer_config_path),
                "provenance_path": str(result.provenance_path),
                "quantized_weight_count": result.quantized_weight_count,
                "merged_parameter_count": result.merged_parameter_count,
                "target_policy": result.target_policy.name,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
