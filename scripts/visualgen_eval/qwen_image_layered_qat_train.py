# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native PyTorch fake-MXFP8 QAT trainer for Qwen-Image-Layered tuples."""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pydantic import Field, NonNegativeFloat, PositiveFloat, PositiveInt, model_validator

from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent
from tensorrt_llm._torch.visual_gen.quantization.fake_mxfp8 import (
    MXFP8_BLOCK_SIZE,
    FakeMxfp8Linear,
    Mxfp8ScaleMode,
    QwenImageBlockLinearTarget,
    select_qwen_image_block_linears,
)
from tensorrt_llm.llmapi.utils import StrictBaseModel
from tensorrt_llm.visual_gen.args import VisualGenArgs

QWEN_BLOCK_LINEAR_TARGET = "qwen_block_linears"
SUPPORTED_TUPLE_KEYS = ("tuples", "tuple_paths", "samples")
TARGET_OUTPUT_KEY = "target_output"
TRAINING_FORMAT = "qwen_image_layered_mxfp8_qat_adapter_v1"
BF16_TARGET_DTYPE = torch.bfloat16

RecipeName = Literal["lora_adapter", "partial_unfreeze"]
ScheduleName = Literal["smoke", "pilot", "formal", "fallback"]
TrainingFrameworkName = Literal["native_pytorch"]
AttentionQuantRecipe = Literal["none", "sage_attention", "cute_dsl", "fa4", "qk16pv8"]
DistributedMode = Literal["none", "ddp", "fsdp"]
PriorityName = Literal["adapter", "partial_unfreeze", "full_unfreeze"]

_STEP_BOUNDS: dict[ScheduleName, tuple[int, int]] = {
    "smoke": (50, 100),
    "pilot": (500, 1000),
    "formal": (2000, 5000),
    "fallback": (500, 1500),
}
_UNSUPPORTED_ATTENTION_BACKEND_TOKENS = ("sage", "cute", "fa4", "qk16pv8")


class OptimizerConfig(StrictBaseModel):
    """Optimizer settings for the native PyTorch loop."""

    name: Literal["adamw"] = Field(default="adamw", description="Optimizer implementation.")
    learning_rate: PositiveFloat = Field(default=1.0e-4, description="Optimizer learning rate.")
    weight_decay: NonNegativeFloat = Field(default=0.0, description="AdamW weight decay.")
    betas: tuple[float, float] = Field(
        default=(0.9, 0.999),
        description="AdamW beta coefficients.",
    )
    eps: PositiveFloat = Field(default=1.0e-8, description="AdamW epsilon.")

    @model_validator(mode="after")
    def _validate_betas(self) -> "OptimizerConfig":
        beta1, beta2 = self.betas
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("optimizer betas must be in [0, 1)")
        return self


class DistributedConfig(StrictBaseModel):
    """Optional distributed wrapper settings."""

    mode: DistributedMode = Field(
        default="none",
        description="No wrapper, torch DDP, or torch FSDP.",
    )
    device_id: int | None = Field(
        default=None,
        description="CUDA device id for DDP. Uses the current CUDA device when omitted.",
    )


class QwenImageLayeredQatConfig(StrictBaseModel):
    """Strict config for offline Qwen-Image-Layered fake-MXFP8 QAT."""

    tuple_manifest: str = Field(description="JSON manifest, .pt tuple, or tuple directory.")
    output_dir: str = Field(description="Directory for metrics, checkpoints, and provenance.")
    target_layers: list[str] = Field(
        default_factory=lambda: [QWEN_BLOCK_LINEAR_TARGET],
        min_length=1,
        description=(
            "Qwen block Linear target selector. Use qwen_block_linears for all supported "
            "block Linears, or role/module names for a subset."
        ),
    )
    max_steps: PositiveInt = Field(description="Step-based training budget.")
    validation_interval_steps: PositiveInt = Field(description="Validation cadence in steps.")
    early_stop_patience: PositiveInt = Field(description="Validation plateaus before stopping.")
    recipe: RecipeName = Field(description="Trainable-parameter recipe.")
    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig,
        description="Native PyTorch optimizer settings.",
    )
    checkpoint_interval_steps: PositiveInt = Field(description="Checkpoint cadence in steps.")
    schedule: ScheduleName = Field(default="smoke", description="Training schedule tier.")
    training_framework: TrainingFrameworkName = Field(
        default="native_pytorch",
        description="The first-stage trainer only supports a native PyTorch loop.",
    )
    model: str | None = Field(
        default=None,
        description="Qwen-Image-Layered model path or id for the VisualGen loader path.",
    )
    visual_gen_args: str | None = Field(
        default=None,
        description="VisualGenArgs YAML for loading the BF16 transformer.",
    )
    device: str = Field(default="cuda", description="Torch device for training.")
    expected_num_layers: PositiveInt | None = Field(
        default=None,
        description="Expected Qwen transformer block count.",
    )
    expected_target_count: PositiveInt | None = Field(
        default=None,
        description="Expected selected target Linear count.",
    )
    block_size: PositiveInt = Field(
        default=MXFP8_BLOCK_SIZE,
        description="MXFP8 fake-quant block size.",
    )
    scale_mode: Mxfp8ScaleMode = Field(
        default="fp32_block_scale",
        description="Fake MXFP8 scale mode.",
    )
    lora_rank: PositiveInt = Field(default=8, description="Adapter rank.")
    lora_alpha: PositiveFloat = Field(default=16.0, description="Adapter scaling alpha.")
    lora_dropout: NonNegativeFloat = Field(default=0.0, description="Adapter dropout.")
    validation_fraction: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Fraction of tuples used for validation when more than one tuple exists.",
    )
    attention_qat: bool = Field(
        default=False,
        description="Rejected first-stage switch for attention-kernel QAT.",
    )
    attention_quant_recipe: AttentionQuantRecipe = Field(
        default="none",
        description="Rejected first-stage attention-kernel quantization recipe.",
    )
    attention_backend: str | None = Field(
        default=None,
        description="Optional attention backend provenance; Sage/CuteDSL/FA4 are rejected.",
    )
    allow_full_weight_unfreeze: bool = Field(
        default=False,
        description="Rejected unless train_priority explicitly chooses the trainable path.",
    )
    train_priority: PriorityName | None = Field(
        default=None,
        description="Priority when more than one trainable-parameter strategy is requested.",
    )
    enable_partial_unfreeze: bool = Field(
        default=False,
        description="Explicit opt-in for the sensitivity-guided partial-unfreeze fallback.",
    )
    sensitivity_path: str | None = Field(
        default=None,
        description="Sensitivity summary used by the partial-unfreeze fallback.",
    )
    distributed: DistributedConfig = Field(
        default_factory=DistributedConfig,
        description="Optional distributed wrapper settings.",
    )
    debug_allow_short_run: bool = Field(
        default=False,
        description="Allows sub-smoke step counts only for unit tests and local smoke fixtures.",
    )

    @model_validator(mode="after")
    def _validate_training_contract(self) -> "QwenImageLayeredQatConfig":
        if self.block_size != MXFP8_BLOCK_SIZE:
            raise ValueError(f"MXFP8 QAT requires block_size={MXFP8_BLOCK_SIZE}")
        if self.validation_interval_steps > self.max_steps:
            raise ValueError("validation_interval_steps must not exceed max_steps")
        if self.checkpoint_interval_steps > self.max_steps:
            raise ValueError("checkpoint_interval_steps must not exceed max_steps")
        if self.attention_qat or self.attention_quant_recipe != "none":
            raise ValueError("first-stage QAT rejects attention-kernel quantization")
        if self.attention_backend:
            attention_backend = self.attention_backend.lower()
            if any(token in attention_backend for token in _UNSUPPORTED_ATTENTION_BACKEND_TOKENS):
                raise ValueError("SageAttention, CuteDSL, FA4, and QK16PV8 training are rejected")
        if self.allow_full_weight_unfreeze and self.train_priority is None:
            raise ValueError("full-weight unfreeze requires an explicit train_priority")
        if self.recipe == "lora_adapter" and self.train_priority in (
            "partial_unfreeze",
            "full_unfreeze",
        ):
            raise ValueError("lora_adapter recipe cannot prioritize another trainable strategy")
        min_steps, max_steps = _STEP_BOUNDS[self.schedule]
        if self.debug_allow_short_run and self.schedule == "smoke":
            min_steps = 1
        if not min_steps <= self.max_steps <= max_steps:
            raise ValueError(
                f"{self.schedule} schedule requires max_steps in [{min_steps}, {max_steps}]"
            )
        if self.recipe == "partial_unfreeze":
            if not self.enable_partial_unfreeze:
                raise ValueError("partial_unfreeze is disabled until explicitly enabled")
            if self.schedule != "fallback":
                raise ValueError("partial_unfreeze must use the fallback schedule")
            if not self.sensitivity_path:
                raise ValueError("partial_unfreeze requires sensitivity_path")
            if self.optimizer.learning_rate > 1.0e-5:
                raise ValueError("partial_unfreeze requires learning_rate <= 1e-5")
            if self.train_priority not in (None, "partial_unfreeze"):
                raise ValueError("partial_unfreeze requires train_priority=partial_unfreeze")
        elif self.enable_partial_unfreeze or self.sensitivity_path:
            raise ValueError("sensitivity fallback settings require recipe=partial_unfreeze")
        return self


@dataclass(frozen=True)
class TransformerTupleSample:
    """One cached transformer forward sample and BF16 target."""

    path: Path
    args: tuple[object, ...]
    kwargs: dict[str, object]
    target_output: torch.Tensor
    hidden_states: torch.Tensor
    timestep: object


@dataclass(frozen=True)
class QatInjectionInfo:
    """Trainable wrapper inserted into one selected Linear."""

    module_name: str
    block_index: int
    role: str
    recipe: RecipeName
    trainable_parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class QatTrainingResult:
    """Paths and metrics emitted by one training run."""

    output_dir: Path
    checkpoint_path: Path
    metrics_path: Path
    provenance_path: Path
    train_steps: int
    best_validation_loss: float
    injections: tuple[QatInjectionInfo, ...]


@dataclass(frozen=True)
class LoadedTransformer:
    """Transformer loaded from a VisualGen pipeline with a cleanup callback."""

    transformer: nn.Module
    cleanup: Callable[[], None]


class Mxfp8LoraAdapterLinear(nn.Module):
    """Low-rank adapter residual around a frozen fake-MXFP8 Linear wrapper."""

    def __init__(
        self,
        base: FakeMxfp8Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        factory_kwargs = {
            "device": base.weight.device,
            "dtype": base.weight.dtype,
        }
        self.lora_down = nn.Linear(base.in_features, rank, bias=False, **factory_kwargs)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False, **factory_kwargs)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        base_output = self.base(activation)
        adapter_output = self.lora_up(self.lora_down(self.dropout(activation))) * self.scaling
        return base_output + adapter_output


class TransformerTupleDataset(torch.utils.data.Dataset):
    """Dataset for cached BF16 transformer tuples saved by the quality script."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.paths = _resolve_tuple_paths(self.manifest_path)
        if not self.paths:
            raise ValueError("tuple manifest contains no tuple payloads")
        self.samples = [_load_tuple_sample(path) for path in self.paths]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TransformerTupleSample:
        return self.samples[index]


def load_qat_config(path: str | Path) -> QwenImageLayeredQatConfig:
    """Load a strict QAT config from JSON or YAML."""
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"QAT config must contain a mapping at the document root: {config_path}")
    return QwenImageLayeredQatConfig(**data)


def load_bf16_qwen_transformer(config: QwenImageLayeredQatConfig) -> LoadedTransformer:
    """Load the BF16 Qwen transformer through the existing VisualGen loader path."""
    if not config.model or not config.visual_gen_args:
        raise ValueError("model and visual_gen_args are required when no transformer is provided")

    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader

    args = VisualGenArgs.from_yaml(config.visual_gen_args, model=config.model)
    loader = PipelineLoader(args, device=config.device)
    pipeline = loader.load(
        skip_warmup=True,
        skip_components=[
            PipelineComponent.TEXT_ENCODER,
            PipelineComponent.VAE,
            PipelineComponent.TOKENIZER,
            PipelineComponent.SCHEDULER,
        ],
    )
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        cleanup = getattr(pipeline, "cleanup", None)
        if cleanup is not None:
            cleanup()
        raise ValueError("VisualGen pipeline does not expose a transformer")

    def cleanup_pipeline() -> None:
        cleanup = getattr(pipeline, "cleanup", None)
        if cleanup is not None:
            cleanup()

    return LoadedTransformer(transformer=transformer, cleanup=cleanup_pipeline)


def prepare_qat_model(
    transformer: nn.Module,
    config: QwenImageLayeredQatConfig,
    *,
    linear_cls: type[nn.Module] | None = None,
) -> tuple[QatInjectionInfo, ...]:
    """Inject fake-MXFP8 Linear wrappers and configure trainable parameters."""
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)

    targets = select_qwen_image_block_linears(
        transformer,
        expected_num_layers=config.expected_num_layers,
        expected_count=config.expected_target_count,
        linear_cls=linear_cls,
    )
    selected_targets = _filter_targets(targets, config.target_layers)
    if not selected_targets:
        raise ValueError(f"target_layers selected no Qwen block Linears: {config.target_layers}")

    if config.recipe == "partial_unfreeze":
        allowed_names = _load_partial_unfreeze_names(Path(str(config.sensitivity_path)))
    else:
        allowed_names = set()

    injections: list[QatInjectionInfo] = []
    for target in selected_targets:
        fake = FakeMxfp8Linear(
            target.module,
            module_name=target.normalized_name,
            block_size=config.block_size,
            scale_mode=config.scale_mode,
        )
        if config.recipe == "lora_adapter":
            replacement = Mxfp8LoraAdapterLinear(
                fake,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=float(config.lora_dropout),
            )
            trainable_names = ("lora_down.weight", "lora_up.weight")
        else:
            if target.normalized_name not in allowed_names and target.role not in allowed_names:
                continue
            for parameter in fake.parameters():
                parameter.requires_grad_(True)
            replacement = fake
            trainable_names = tuple(name for name, _ in fake.named_parameters())
        _replace_module(transformer, target.module_name, replacement)
        injections.append(
            QatInjectionInfo(
                module_name=target.normalized_name,
                block_index=target.block_index,
                role=target.role,
                recipe=config.recipe,
                trainable_parameter_names=trainable_names,
            )
        )

    if not injections:
        raise ValueError("no trainable QAT targets were injected")
    return tuple(injections)


def train_qwen_image_layered_qat(
    config: QwenImageLayeredQatConfig,
    *,
    transformer: nn.Module | None = None,
    linear_cls: type[nn.Module] | None = None,
) -> QatTrainingResult:
    """Run fake-MXFP8 QAT over cached transformer tuples."""
    dataset = TransformerTupleDataset(config.tuple_manifest)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded: LoadedTransformer | None = None
    if transformer is None:
        loaded = load_bf16_qwen_transformer(config)
        transformer = loaded.transformer

    try:
        device = torch.device(config.device)
        transformer.to(device)
        transformer.train()
        injections = prepare_qat_model(transformer, config, linear_cls=linear_cls)
        trained_model = _maybe_wrap_distributed(transformer, config)
        trainable_parameters = [
            parameter for parameter in trained_model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("QAT training has no trainable parameters")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=float(config.optimizer.learning_rate),
            betas=config.optimizer.betas,
            eps=float(config.optimizer.eps),
            weight_decay=float(config.optimizer.weight_decay),
        )

        train_indices, validation_indices = _split_dataset_indices(
            dataset, config.validation_fraction
        )
        metrics: list[dict[str, float | int | str]] = []
        best_validation_loss = float("inf")
        stale_validation_count = 0
        checkpoint_path = output_dir / "checkpoint_last.pt"
        metrics_path = output_dir / "metrics.json"
        metrics_jsonl_path = output_dir / "metrics.jsonl"
        provenance_path = output_dir / "provenance.json"

        step = 0
        with metrics_jsonl_path.open("w", encoding="utf-8") as metrics_jsonl:
            for step in range(1, int(config.max_steps) + 1):
                sample = dataset[train_indices[(step - 1) % len(train_indices)]]
                optimizer.zero_grad(set_to_none=True)
                output = _forward_sample(trained_model, sample, device)
                loss = _mse_loss(output, sample.target_output.to(device=device))
                loss.backward()
                optimizer.step()

                train_record = {
                    "phase": "train",
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "sample_index": int(train_indices[(step - 1) % len(train_indices)]),
                }
                metrics.append(train_record)
                metrics_jsonl.write(json.dumps(train_record, sort_keys=True) + "\n")

                should_validate = step % int(config.validation_interval_steps) == 0
                should_checkpoint = step % int(config.checkpoint_interval_steps) == 0
                if should_validate or step == int(config.max_steps):
                    validation_loss = _validate(trained_model, dataset, validation_indices, device)
                    validation_record = {
                        "phase": "validation",
                        "step": step,
                        "loss": validation_loss,
                        "sample_count": len(validation_indices),
                    }
                    metrics.append(validation_record)
                    metrics_jsonl.write(json.dumps(validation_record, sort_keys=True) + "\n")
                    if validation_loss < best_validation_loss - 1.0e-8:
                        best_validation_loss = validation_loss
                        stale_validation_count = 0
                        _save_checkpoint(
                            checkpoint_path,
                            trained_model,
                            config,
                            injections,
                            metrics,
                            best_validation_loss,
                        )
                    else:
                        stale_validation_count += 1
                    if stale_validation_count >= int(config.early_stop_patience):
                        break
                elif should_checkpoint:
                    _save_checkpoint(
                        checkpoint_path,
                        trained_model,
                        config,
                        injections,
                        metrics,
                        best_validation_loss,
                    )

        if not checkpoint_path.exists():
            _save_checkpoint(
                checkpoint_path,
                trained_model,
                config,
                injections,
                metrics,
                best_validation_loss,
            )
        provenance = _build_provenance(config, dataset, injections, step)
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        return QatTrainingResult(
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
            provenance_path=provenance_path,
            train_steps=step,
            best_validation_loss=best_validation_loss,
            injections=injections,
        )
    finally:
        if loaded is not None:
            loaded.cleanup()


def _resolve_tuple_paths(manifest_path: Path) -> list[Path]:
    if manifest_path.is_dir():
        return sorted(manifest_path.glob("*.pt"))
    if manifest_path.suffix == ".pt":
        return [manifest_path]

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_dir = manifest_path.parent
    entries: object | None = None
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        for key in SUPPORTED_TUPLE_KEYS:
            if key in data:
                entries = data[key]
                break
    if not isinstance(entries, list):
        raise ValueError(
            f"tuple manifest must be a list or contain one of {SUPPORTED_TUPLE_KEYS}: "
            f"{manifest_path}"
        )

    paths: list[Path] = []
    for entry in entries:
        if isinstance(entry, str):
            path_text = entry
        elif isinstance(entry, dict) and isinstance(entry.get("path"), str):
            path_text = str(entry["path"])
        else:
            raise ValueError("tuple manifest entries must be paths or objects with a path field")
        path = Path(path_text)
        if not path.is_absolute():
            path = manifest_dir / path
        paths.append(path)
    return paths


def _load_tuple_sample(path: Path) -> TransformerTupleSample:
    if not path.exists():
        raise ValueError(f"tuple payload does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"tuple payload must be a dict: {path}")

    target_output = payload.get(TARGET_OUTPUT_KEY)
    if not isinstance(target_output, torch.Tensor):
        raise ValueError(f"tuple payload missing BF16 target_output: {path}")
    if target_output.dtype != BF16_TARGET_DTYPE:
        raise ValueError(
            f"tuple target_output must be {BF16_TARGET_DTYPE}, got {target_output.dtype}: {path}"
        )

    args = _normalize_args(payload.get("args", ()), path)
    kwargs = _normalize_kwargs(payload, path)
    hidden_states = _extract_hidden_states(args, kwargs, path)
    timestep = _extract_timestep(kwargs, path)
    if hidden_states.shape != target_output.shape:
        raise ValueError(
            f"tuple hidden_states and target_output shapes must match, got "
            f"{tuple(hidden_states.shape)} vs {tuple(target_output.shape)}: {path}"
        )
    return TransformerTupleSample(
        path=path,
        args=args,
        kwargs=kwargs,
        target_output=target_output,
        hidden_states=hidden_states,
        timestep=timestep,
    )


def _normalize_args(value: object, path: Path) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise ValueError(f"tuple args must be a list or tuple: {path}")


def _normalize_kwargs(payload: Mapping[str, object], path: Path) -> dict[str, object]:
    raw_kwargs = payload.get("kwargs")
    if raw_kwargs is None:
        kwargs = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "args",
                "kwargs",
                TARGET_OUTPUT_KEY,
                "sample_id",
                "variant",
                "call_index",
                "role",
            }
        }
    elif isinstance(raw_kwargs, dict):
        kwargs = dict(raw_kwargs)
    else:
        raise ValueError(f"tuple kwargs must be a mapping: {path}")
    if "latents" in kwargs and "hidden_states" not in kwargs:
        kwargs["hidden_states"] = kwargs.pop("latents")
    return kwargs


def _extract_hidden_states(
    args: Sequence[object],
    kwargs: Mapping[str, object],
    path: Path,
) -> torch.Tensor:
    hidden_states = kwargs.get("hidden_states")
    if hidden_states is None and args and isinstance(args[0], torch.Tensor):
        hidden_states = args[0]
    if not isinstance(hidden_states, torch.Tensor):
        raise ValueError(f"tuple payload missing hidden_states or latents: {path}")
    if not torch.is_floating_point(hidden_states):
        raise ValueError(f"tuple hidden_states must be floating point: {path}")
    return hidden_states


def _extract_timestep(kwargs: Mapping[str, object], path: Path) -> object:
    if "timestep" not in kwargs:
        raise ValueError(f"tuple payload missing timestep: {path}")
    timestep = kwargs["timestep"]
    if isinstance(timestep, torch.Tensor) and timestep.numel() == 0:
        raise ValueError(f"tuple timestep tensor must not be empty: {path}")
    return timestep


def _filter_targets(
    targets: Sequence[QwenImageBlockLinearTarget],
    selectors: Sequence[str],
) -> list[QwenImageBlockLinearTarget]:
    selector_set = set(selectors)
    if QWEN_BLOCK_LINEAR_TARGET in selector_set:
        return list(targets)
    selected: list[QwenImageBlockLinearTarget] = []
    for target in targets:
        if target.normalized_name in selector_set or target.role in selector_set:
            selected.append(target)
    return selected


def _load_partial_unfreeze_names(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    if isinstance(data, dict):
        raw_names = data.get("target_layers") or data.get("layers") or data.get("module_names")
        if isinstance(raw_names, list):
            names.update(str(name) for name in raw_names)
        priority_groups = data.get("qat_priority_groups")
        if isinstance(priority_groups, list):
            for group in priority_groups:
                if isinstance(group, dict):
                    group_names = group.get("layers") or group.get("module_names")
                    if isinstance(group_names, list):
                        names.update(str(name) for name in group_names)
    elif isinstance(data, list):
        names.update(str(name) for name in data)
    if not names:
        raise ValueError(f"sensitivity_path does not name any partial-unfreeze targets: {path}")
    return names


def _get_child_module(module: nn.Module, child_name: str) -> nn.Module:
    if child_name.isdigit() and isinstance(module, (nn.ModuleList, nn.Sequential)):
        return module[int(child_name)]
    return getattr(module, child_name)


def _set_child_module(module: nn.Module, child_name: str, child: nn.Module) -> None:
    if child_name.isdigit() and isinstance(module, (nn.ModuleList, nn.Sequential)):
        module[int(child_name)] = child
    else:
        setattr(module, child_name, child)


def _replace_module(root: nn.Module, module_name: str, replacement: nn.Module) -> None:
    parent = root
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = _get_child_module(parent, part)
    _set_child_module(parent, parts[-1], replacement)


def _maybe_wrap_distributed(
    transformer: nn.Module,
    config: QwenImageLayeredQatConfig,
) -> nn.Module:
    if config.distributed.mode == "none":
        return transformer
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise ValueError("distributed training requires an initialized torch.distributed process")
    if config.distributed.mode == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        device_ids = None
        if torch.device(config.device).type == "cuda":
            device_id = config.distributed.device_id
            if device_id is None:
                device_id = torch.cuda.current_device()
            device_ids = [device_id]
        return DistributedDataParallel(transformer, device_ids=device_ids)

    from torch.distributed.fsdp import FullyShardedDataParallel

    return FullyShardedDataParallel(transformer)


def _split_dataset_indices(
    dataset: TransformerTupleDataset,
    validation_fraction: float,
) -> tuple[list[int], list[int]]:
    indices = list(range(len(dataset)))
    if len(indices) == 1:
        return indices, indices
    validation_count = max(1, int(round(len(indices) * validation_fraction)))
    validation_count = min(validation_count, len(indices) - 1)
    return indices[:-validation_count], indices[-validation_count:]


def _move_value(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_move_value(item, device) for item in value)
    if isinstance(value, list):
        return [_move_value(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_value(item, device) for key, item in value.items()}
    return value


def _first_tensor_output(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    sample = getattr(output, "sample", None)
    if isinstance(sample, torch.Tensor):
        return sample
    raise ValueError("transformer output does not contain a tensor target")


def _forward_sample(
    transformer: nn.Module,
    sample: TransformerTupleSample,
    device: torch.device,
) -> torch.Tensor:
    args = _move_value(sample.args, device)
    kwargs = _move_value(sample.kwargs, device)
    if not isinstance(args, tuple) or not isinstance(kwargs, dict):
        raise TypeError("internal tuple sample normalization failed")
    return _first_tensor_output(transformer(*args, **kwargs))


def _mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if output.shape != target.shape:
        raise ValueError(
            f"transformer output and target_output shapes must match, got "
            f"{tuple(output.shape)} vs {tuple(target.shape)}"
        )
    return F.mse_loss(output.float(), target.to(dtype=output.dtype).float())


def _validate(
    transformer: nn.Module,
    dataset: TransformerTupleDataset,
    validation_indices: Sequence[int],
    device: torch.device,
) -> float:
    transformer.eval()
    losses: list[float] = []
    with torch.no_grad():
        for index in validation_indices:
            sample = dataset[index]
            output = _forward_sample(transformer, sample, device)
            loss = _mse_loss(output, sample.target_output.to(device=device))
            losses.append(float(loss.detach().cpu()))
    transformer.train()
    return sum(losses) / len(losses)


def _unwrap_model(model: nn.Module) -> nn.Module:
    module = getattr(model, "module", None)
    if isinstance(module, nn.Module):
        return module
    return model


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    config: QwenImageLayeredQatConfig,
    injections: Sequence[QatInjectionInfo],
    metrics: Sequence[Mapping[str, float | int | str]],
    best_validation_loss: float,
) -> None:
    unwrapped = _unwrap_model(model)
    trainable_state_dict = {
        name: parameter.detach().cpu()
        for name, parameter in unwrapped.named_parameters()
        if parameter.requires_grad
    }
    checkpoint = {
        "format": TRAINING_FORMAT,
        "config": config.model_dump(),
        "trainable_state_dict": trainable_state_dict,
        "trainable_parameter_names": sorted(trainable_state_dict),
        "injections": [_injection_to_dict(injection) for injection in injections],
        "metrics": list(metrics),
        "best_validation_loss": best_validation_loss,
    }
    torch.save(checkpoint, path)


def _injection_to_dict(injection: QatInjectionInfo) -> dict[str, object]:
    return {
        "module_name": injection.module_name,
        "block_index": injection.block_index,
        "role": injection.role,
        "recipe": injection.recipe,
        "trainable_parameter_names": list(injection.trainable_parameter_names),
    }


def _build_provenance(
    config: QwenImageLayeredQatConfig,
    dataset: TransformerTupleDataset,
    injections: Sequence[QatInjectionInfo],
    train_steps: int,
) -> dict[str, object]:
    return {
        "format": TRAINING_FORMAT,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "config": config.model_dump(),
        "tuple_manifest": str(dataset.manifest_path),
        "tuple_count": len(dataset),
        "tuple_paths": [str(path) for path in dataset.paths],
        "train_steps": train_steps,
        "injections": [_injection_to_dict(injection) for injection in injections],
    }


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
        description="Train Qwen-Image-Layered fake-MXFP8 adapters from cached transformer tuples."
    )
    parser.add_argument("--config", required=True, help="QAT training config JSON/YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_qat_config(args.config)
    result = train_qwen_image_layered_qat(config)
    print(
        json.dumps(
            {
                "checkpoint_path": str(result.checkpoint_path),
                "metrics_path": str(result.metrics_path),
                "provenance_path": str(result.provenance_path),
                "train_steps": result.train_steps,
                "best_validation_loss": result.best_validation_loss,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
