# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native PyTorch fake-MXFP8 QAT trainer for Qwen-Image-Layered tuples."""

from __future__ import annotations

import argparse
import datetime
import functools
import inspect
import json
import math
import os
import platform
import random
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent
from tensorrt_llm._torch.visual_gen.quantization.fake_mxfp8 import (
    MXFP8_BLOCK_SIZE,
    SUPPORTED_FAKE_MXFP8_DTYPES,
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
OPTIONAL_TARGET_KEYS = (
    "layered_rgba_target",
    "layered_rgba_prediction",
    "composite_target",
    "composite_prediction",
    "alpha_mask_target",
    "alpha_mask_prediction",
    "perceptual_target",
    "perceptual_prediction",
)

RecipeName = Literal["lora_adapter", "partial_unfreeze"]
OptimizerName = Literal["adamw", "sgd"]
ScheduleName = Literal[
    "smoke",
    "pilot",
    "formal",
    "fallback",
    "calibration_probe",
    "calibration_formal",
]
TrainingFrameworkName = Literal["native_pytorch"]
AttentionQuantRecipe = Literal["none", "sage_attention", "cute_dsl", "fa4", "qk16pv8"]
DistributedMode = Literal["none", "ddp", "fsdp"]
PriorityName = Literal["adapter", "partial_unfreeze", "full_unfreeze"]

_STEP_BOUNDS: dict[ScheduleName, tuple[int, int]] = {
    "smoke": (50, 100),
    "pilot": (500, 1000),
    "formal": (2000, 5000),
    "fallback": (500, 1500),
    "calibration_probe": (100, 300),
    "calibration_formal": (500, 1500),
}
_VALIDATION_INTERVAL_BOUNDS: dict[ScheduleName, tuple[int, int]] = {
    "pilot": (100, 200),
    "formal": (200, 500),
    "fallback": (100, 500),
}
_VALIDATION_DRIVEN_SCHEDULES = ("pilot", "formal", "fallback")
_UNSUPPORTED_ATTENTION_BACKEND_TOKENS = ("sage", "cute", "fa4", "qk16pv8")
_UNSUPPORTED_VISUALGEN_ATTENTION_BACKENDS = {"FA4", "CUTEDSL"}
_OPTIONAL_LOSS_TARGETS = {
    "layered_rgba": "layered_rgba_target",
    "composite": "composite_target",
    "alpha_mask": "alpha_mask_target",
    "perceptual": "perceptual_target",
}
_DIFFERENTIABLE_OPTIONAL_LOSSES = {
    "layered_rgba": "layered_rgba_weight",
    "composite": "composite_weight",
    "alpha_mask": "alpha_mask_weight",
}


class OptimizerConfig(StrictBaseModel):
    """Optimizer settings for the native PyTorch loop."""

    name: OptimizerName = Field(
        default="adamw",
        description="Optimizer implementation. SGD is state-free when momentum is zero.",
    )
    learning_rate: PositiveFloat = Field(default=1.0e-4, description="Optimizer learning rate.")
    weight_decay: NonNegativeFloat = Field(default=0.0, description="AdamW weight decay.")
    betas: tuple[float, float] = Field(
        default=(0.9, 0.999),
        description="AdamW beta coefficients.",
    )
    eps: PositiveFloat = Field(default=1.0e-8, description="AdamW epsilon.")
    foreach: bool | None = Field(
        default=None,
        description="Optional AdamW foreach implementation switch for memory-sensitive runs.",
    )

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


class LossConfig(StrictBaseModel):
    """Loss weights for latent-first Qwen-Image-Layered distillation."""

    latent_weight: NonNegativeFloat = Field(
        default=1.0,
        description="Weight for transformer latent reconstruction loss.",
    )
    layered_rgba_weight: NonNegativeFloat = Field(
        default=0.0,
        description="Weight for layered RGBA reconstruction loss.",
    )
    composite_weight: NonNegativeFloat = Field(
        default=0.0,
        description="Weight for composite image reconstruction loss.",
    )
    alpha_mask_weight: NonNegativeFloat = Field(
        default=0.0,
        description="Weight for alpha or mask quality loss.",
    )
    perceptual_weight: NonNegativeFloat = Field(
        default=0.0,
        description="Weight for an optional perceptual reconstruction loss.",
    )

    @model_validator(mode="after")
    def _validate_at_least_one_component(self) -> "LossConfig":
        if (
            self.latent_weight
            + self.layered_rgba_weight
            + self.composite_weight
            + self.alpha_mask_weight
            + self.perceptual_weight
            == 0.0
        ):
            raise ValueError("loss config must enable at least one component")
        return self


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
    loss: LossConfig = Field(
        default_factory=LossConfig,
        description="Weighted reconstruction losses used by the native PyTorch loop.",
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
    train_all_tuples: bool = Field(
        default=False,
        description="Use every cached tuple for training instead of holding out validation tuples.",
    )
    shuffle_train_tuples: bool = Field(
        default=False,
        description="Shuffle the training tuple order independently for each full tuple pass.",
    )
    train_shuffle_seed: NonNegativeInt = Field(
        default=0,
        description="Seed for deterministic per-epoch training tuple shuffles.",
    )
    monitor_subset_size: PositiveInt | None = Field(
        default=None,
        description="Fixed random tuple subset size for calibration monitoring.",
    )
    monitor_subset_seed: NonNegativeInt = Field(
        default=44,
        description="Seed used to choose the fixed monitor subset.",
    )
    monitor_all_tuples: bool = Field(
        default=False,
        description="Evaluate all cached tuples at the monitor cadence.",
    )
    disable_early_stop: bool = Field(
        default=False,
        description="Disable plateau early stop for explicit calibration runs.",
    )
    update_norm_interval_steps: NonNegativeInt = Field(
        default=0,
        description=(
            "Cadence for recording grad_norm, weight_norm, and update norm telemetry. "
            "A value of 0 disables the extra telemetry."
        ),
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
    activation_checkpointing: bool = Field(
        default=False,
        description="Enable block-level activation checkpointing for memory-heavy QAT pilots.",
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
        if self.train_all_tuples:
            if not self.disable_early_stop:
                raise ValueError("train_all_tuples calibration requires disable_early_stop=true")
            if self.monitor_subset_size is None and not self.monitor_all_tuples:
                raise ValueError(
                    "train_all_tuples calibration requires monitor_subset_size or "
                    "monitor_all_tuples"
                )
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
        validation_bounds = _VALIDATION_INTERVAL_BOUNDS.get(self.schedule)
        if validation_bounds is not None:
            min_interval, max_interval = validation_bounds
            if not min_interval <= self.validation_interval_steps <= max_interval:
                raise ValueError(
                    f"{self.schedule} schedule requires validation_interval_steps in "
                    f"[{min_interval}, {max_interval}]"
                )
        if self.schedule in _VALIDATION_DRIVEN_SCHEDULES and not (
            2 <= int(self.early_stop_patience) <= 3
        ):
            raise ValueError(f"{self.schedule} schedule requires early_stop_patience in [2, 3]")
        if self.recipe == "partial_unfreeze":
            if not self.enable_partial_unfreeze:
                raise ValueError("partial_unfreeze is disabled until explicitly enabled")
            calibration_schedules = ("calibration_probe", "calibration_formal")
            if self.schedule not in ("fallback", *calibration_schedules):
                raise ValueError(
                    "partial_unfreeze must use the fallback schedule or a calibration schedule"
                )
            if self.schedule in calibration_schedules and not self.train_all_tuples:
                raise ValueError("partial_unfreeze calibration schedule requires train_all_tuples")
            if not self.sensitivity_path:
                raise ValueError("partial_unfreeze requires sensitivity_path")
            max_learning_rate = 1.0e-4 if self.schedule in calibration_schedules else 1.0e-5
            if self.optimizer.learning_rate > max_learning_rate:
                raise ValueError(
                    f"partial_unfreeze requires learning_rate <= {max_learning_rate:g}"
                )
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
    encoder_hidden_states: torch.Tensor
    timestep: object
    img_shapes: object
    txt_seq_lens: object
    additional_t_cond: object | None
    optional_targets: dict[str, torch.Tensor]


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
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    metrics_path: Path
    provenance_path: Path
    train_steps: int
    best_validation_loss: float
    best_validation_step: int | None
    injections: tuple[QatInjectionInfo, ...]


@dataclass(frozen=True)
class LoadedTransformer:
    """Transformer loaded from a VisualGen pipeline with a cleanup callback."""

    transformer: nn.Module
    cleanup: Callable[[], None]


@dataclass(frozen=True)
class ValidationMetrics:
    """Aggregated validation losses and optional image-quality metrics."""

    loss: float
    loss_components: dict[str, float]
    quality_metrics: dict[str, float]


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

    _debug_log("loading VisualGenArgs")
    args = VisualGenArgs.from_yaml(config.visual_gen_args, model=config.model)
    _validate_visual_gen_args_for_linear_qat(args)
    _validate_resolved_pipeline_config_for_linear_qat(args, config.model)

    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader

    _debug_log("creating PipelineLoader")
    loader = PipelineLoader(args, device=config.device)
    _debug_log("loading BF16 transformer pipeline")
    pipeline = loader.load(
        skip_warmup=True,
        skip_components=[
            PipelineComponent.TEXT_ENCODER,
            PipelineComponent.VAE,
            PipelineComponent.TOKENIZER,
            PipelineComponent.SCHEDULER,
        ],
    )
    _debug_log("loaded BF16 transformer pipeline")
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


def _validate_visual_gen_args_for_linear_qat(args: VisualGenArgs) -> None:
    quant_config = args.quant_config
    if _visual_gen_quant_config_enables_linear_quantization(quant_config):
        raise ValueError(
            "QAT training must load a BF16 transformer; visual_gen_args cannot enable "
            "Linear quantization"
        )

    attention_config = args.attention_config
    if attention_config.quant_attention_config is not None:
        raise ValueError(
            "QAT training must load a BF16 transformer; attention quantization is rejected"
        )
    if attention_config.backend in _UNSUPPORTED_VISUALGEN_ATTENTION_BACKENDS:
        raise ValueError(
            "QAT training rejects FA4 and CuteDSL attention backends for the first-stage loader"
        )


def _visual_gen_quant_config_enables_linear_quantization(quant_config: object) -> bool:
    if isinstance(quant_config, dict):
        if quant_config.get("quant_algo") is not None:
            return True
        if quant_config.get("config_groups"):
            return True
        if bool(quant_config.get("dynamic")) or bool(quant_config.get("dynamic_weight_quant")):
            return True
        return False
    return getattr(quant_config, "quant_algo", None) is not None


def _validate_resolved_pipeline_config_for_linear_qat(args: VisualGenArgs, model: str) -> None:
    pipeline_config = DiffusionPipelineConfig.from_pretrained(model, args=args)
    _validate_quant_fields_for_linear_qat(pipeline_config, "resolved VisualGen pipeline config")
    model_configs = getattr(pipeline_config, "model_configs", None)
    if isinstance(model_configs, Mapping):
        for component_name, model_config in model_configs.items():
            _validate_quant_fields_for_linear_qat(
                model_config,
                f"resolved VisualGen model config {component_name}",
            )


def _validate_quant_fields_for_linear_qat(config_object: object, source_name: str) -> None:
    quant_reason = _resolved_quantization_reason(config_object)
    if quant_reason is not None:
        raise ValueError(
            "QAT training must load a BF16 transformer; "
            f"{source_name} enables Linear quantization through {quant_reason}"
        )


def _resolved_quantization_reason(config_object: object) -> str | None:
    quant_config = getattr(config_object, "quant_config", None)
    if _visual_gen_quant_config_enables_linear_quantization(quant_config):
        return "quant_config.quant_algo"
    quant_config_dict = getattr(config_object, "quant_config_dict", None)
    if isinstance(quant_config_dict, Mapping):
        if quant_config_dict:
            return "quant_config_dict"
    elif quant_config_dict:
        return "quant_config_dict"
    if bool(getattr(config_object, "dynamic_weight_quant", False)):
        return "dynamic_weight_quant"
    if bool(getattr(config_object, "force_dynamic_quantization", False)):
        return "dynamic activation quantization"
    if bool(getattr(config_object, "dynamic_activation_quant", False)):
        return "dynamic activation quantization"
    if bool(getattr(config_object, "dynamic_activation_quantization", False)):
        return "dynamic activation quantization"
    return None


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
    _assert_qat_target_weights_are_floating(selected_targets)

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


def _assert_qat_target_weights_are_floating(
    targets: Sequence[QwenImageBlockLinearTarget],
) -> None:
    for target in targets:
        weight = getattr(target.module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"selected QAT target {target.normalized_name} has no weight tensor")
        if weight.dtype not in SUPPORTED_FAKE_MXFP8_DTYPES:
            raise TypeError(
                f"selected QAT target {target.normalized_name} must have a BF16/FP16/FP32 "
                f"weight tensor before fake-MXFP8 injection, got {weight.dtype}"
            )
        bias = getattr(target.module, "bias", None)
        if bias is not None:
            if not isinstance(bias, torch.Tensor):
                raise TypeError(
                    f"selected QAT target {target.normalized_name} has a non-tensor bias"
                )
            if bias.dtype not in SUPPORTED_FAKE_MXFP8_DTYPES:
                raise TypeError(
                    f"selected QAT target {target.normalized_name} must have a BF16/FP16/FP32 "
                    f"bias tensor before fake-MXFP8 injection, got {bias.dtype}"
                )


def train_qwen_image_layered_qat(
    config: QwenImageLayeredQatConfig,
    *,
    transformer: nn.Module | None = None,
    linear_cls: type[nn.Module] | None = None,
) -> QatTrainingResult:
    """Run fake-MXFP8 QAT over cached transformer tuples."""
    dataset = TransformerTupleDataset(config.tuple_manifest)
    _validate_loss_payloads(config.loss, dataset)
    output_dir = Path(config.output_dir)
    if _is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    _distributed_barrier()
    best_checkpoint_path = output_dir / "checkpoint_best.pt"
    last_checkpoint_path = output_dir / "checkpoint_last.pt"
    preexisting_checkpoints = [
        checkpoint_path
        for checkpoint_path in (best_checkpoint_path, last_checkpoint_path)
        if checkpoint_path.exists()
    ]
    if preexisting_checkpoints:
        formatted_paths = ", ".join(str(path) for path in preexisting_checkpoints)
        raise RuntimeError(f"checkpoint output already exists: {formatted_paths}")
    metrics_path = output_dir / "metrics.json"
    metrics_jsonl_path = output_dir / "metrics.jsonl"
    provenance_path = output_dir / "provenance.json"

    loaded: LoadedTransformer | None = None
    try:
        if transformer is None:
            _debug_log("loading transformer")
            loaded = load_bf16_qwen_transformer(config)
            transformer = loaded.transformer
            _debug_log("transformer loaded")
        _debug_log("initializing distributed process group")
        _maybe_init_distributed_from_env(config)
        _debug_log("distributed process group ready")
        device = torch.device(config.device)
        _debug_log(f"moving transformer to {device}")
        transformer.to(device)
        transformer.train()
        _debug_log("preparing QAT injections")
        injections = prepare_qat_model(transformer, config, linear_cls=linear_cls)
        _debug_log(f"prepared {len(injections)} QAT injections")
        activation_checkpointed_blocks = 0
        if config.activation_checkpointing:
            _debug_log("enabling activation checkpointing")
            activation_checkpointed_blocks = _enable_qat_activation_checkpointing(transformer)
            _debug_log(f"activation checkpointed blocks: {activation_checkpointed_blocks}")
        _debug_log("wrapping distributed model")
        trained_model = _maybe_wrap_distributed(transformer, config)
        _debug_log("distributed model ready")
        trainable_parameters = [
            parameter for parameter in trained_model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("QAT training has no trainable parameters")
        _debug_log(f"building optimizer for {len(trainable_parameters)} trainable tensors")
        optimizer = _build_optimizer(trainable_parameters, config.optimizer)

        train_indices, validation_indices = _training_and_validation_indices(dataset, config)
        monitor_subset_indices, monitor_all_indices = _monitor_indices(dataset, config)
        metrics: list[dict[str, object]] = []
        best_validation_loss = float("inf")
        best_validation_step: int | None = None
        best_metric_name = "validation_loss"
        best_metric_phase = "validation"
        stale_validation_count = 0
        wrote_best_checkpoint = False
        epoch_orders: dict[int, list[int]] = {}

        step = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        metrics_jsonl = (
            metrics_jsonl_path.open("w", encoding="utf-8") if _is_main_process() else None
        )
        try:
            for step in range(1, int(config.max_steps) + 1):
                _debug_log(f"train step {step} start")
                _synchronize_device(device)
                step_started_at = time.perf_counter()
                sample_index = _training_sample_index_for_step(
                    train_indices,
                    step=step,
                    shuffle=config.shuffle_train_tuples,
                    seed=int(config.train_shuffle_seed),
                    epoch_orders=epoch_orders,
                )
                sample = dataset[sample_index]
                optimizer.zero_grad(set_to_none=True)
                parameter_snapshots = None
                should_record_update_norms = (
                    config.update_norm_interval_steps > 0
                    and step % int(config.update_norm_interval_steps) == 0
                )
                if should_record_update_norms:
                    parameter_snapshots = _clone_trainable_parameters(trainable_parameters)
                output = _forward_sample(trained_model, sample, device)
                loss, loss_components = _compute_loss(output, sample, config.loss, device)
                loss_value = float(loss.detach().cpu())
                if not math.isfinite(loss_value):
                    raise RuntimeError(f"non-finite training loss at step {step}: {loss_value}")
                loss.backward()
                update_metrics: dict[str, float] = {}
                if should_record_update_norms:
                    update_metrics["grad_norm"] = _gradient_global_norm(trainable_parameters)
                optimizer.step()
                if should_record_update_norms and parameter_snapshots is not None:
                    weight_norm = _parameter_global_norm(trainable_parameters)
                    weight_delta_norm = _parameter_update_global_norm(
                        trainable_parameters,
                        parameter_snapshots,
                    )
                    update_metrics["weight_norm"] = weight_norm
                    update_metrics["weight_delta_norm"] = weight_delta_norm
                    update_metrics["weight_delta_to_weight_norm"] = (
                        weight_delta_norm / weight_norm if weight_norm > 0.0 else 0.0
                    )
                _synchronize_device(device)

                train_record = {
                    "phase": "train",
                    "step": step,
                    "loss": loss_value,
                    "loss_components": _loss_components_to_float(loss_components),
                    "sample_index": sample_index,
                    "train_epoch": (step - 1) // len(train_indices),
                    "step_time_seconds": time.perf_counter() - step_started_at,
                    "peak_cuda_memory_bytes": _peak_cuda_memory_bytes(device),
                    "rank": _distributed_rank(),
                    "world_size": _distributed_world_size(),
                }
                train_record.update(update_metrics)
                metrics.append(train_record)
                _write_jsonl_record(metrics_jsonl, train_record)
                _debug_log(f"train step {step} wrote metrics")

                should_validate = step % int(config.validation_interval_steps) == 0
                should_checkpoint = step % int(config.checkpoint_interval_steps) == 0
                if should_validate or step == int(config.max_steps):
                    if should_checkpoint:
                        # Preserve the completed training step before monitor/eval can fail.
                        _debug_log(f"train step {step} saving pre-monitor checkpoint")
                        _save_checkpoint(
                            last_checkpoint_path,
                            trained_model,
                            config,
                            injections,
                            metrics,
                            best_validation_loss,
                            best_validation_step,
                            best_checkpoint_path,
                            last_checkpoint_path,
                            best_metric_name,
                            best_metric_phase,
                        )
                        _distributed_barrier()
                        _debug_log(f"train step {step} saved pre-monitor checkpoint")
                    selection_loss: float | None = None
                    selection_metric_name: str | None = None
                    selection_metric_phase: str | None = None
                    if validation_indices:
                        validation_loss = _evaluate_and_record(
                            trained_model,
                            dataset,
                            validation_indices,
                            config.loss,
                            device,
                            metrics,
                            metrics_jsonl,
                            phase="validation",
                            metric_name="validation_loss",
                            step=step,
                        )
                        selection_loss = validation_loss
                        selection_metric_name = "validation_loss"
                        selection_metric_phase = "validation"
                    if monitor_subset_indices:
                        monitor_subset_loss = _evaluate_and_record(
                            trained_model,
                            dataset,
                            monitor_subset_indices,
                            config.loss,
                            device,
                            metrics,
                            metrics_jsonl,
                            phase="monitor_subset",
                            metric_name="monitor_subset_loss",
                            step=step,
                        )
                        if config.train_all_tuples and not monitor_all_indices:
                            selection_loss = monitor_subset_loss
                            selection_metric_name = "monitor_subset_loss"
                            selection_metric_phase = "monitor_subset"
                    if monitor_all_indices:
                        monitor_all_loss = _evaluate_and_record(
                            trained_model,
                            dataset,
                            monitor_all_indices,
                            config.loss,
                            device,
                            metrics,
                            metrics_jsonl,
                            phase="monitor_all",
                            metric_name="monitor_all_100_loss",
                            step=step,
                        )
                        if config.train_all_tuples:
                            selection_loss = monitor_all_loss
                            selection_metric_name = "monitor_all_100_loss"
                            selection_metric_phase = "monitor_all"
                    if selection_loss is None:
                        raise RuntimeError(
                            "QAT training has no validation or monitor metric for "
                            "best-checkpoint selection"
                        )
                    if selection_loss < best_validation_loss - 1.0e-8:
                        best_validation_loss = selection_loss
                        best_validation_step = step
                        best_metric_name = str(selection_metric_name)
                        best_metric_phase = str(selection_metric_phase)
                        stale_validation_count = 0
                        _save_checkpoint(
                            best_checkpoint_path,
                            trained_model,
                            config,
                            injections,
                            metrics,
                            best_validation_loss,
                            best_validation_step,
                            best_checkpoint_path,
                            last_checkpoint_path,
                            best_metric_name,
                            best_metric_phase,
                        )
                        wrote_best_checkpoint = True
                    else:
                        stale_validation_count += 1
                    if not config.disable_early_stop and stale_validation_count >= int(
                        config.early_stop_patience
                    ):
                        break
                elif should_checkpoint:
                    _save_checkpoint(
                        last_checkpoint_path,
                        trained_model,
                        config,
                        injections,
                        metrics,
                        best_validation_loss,
                        best_validation_step,
                        best_checkpoint_path,
                        last_checkpoint_path,
                        best_metric_name,
                        best_metric_phase,
                    )
        finally:
            if metrics_jsonl is not None:
                metrics_jsonl.close()

        if not wrote_best_checkpoint:
            raise RuntimeError("QAT training completed without a finite best checkpoint")
        _save_checkpoint(
            last_checkpoint_path,
            trained_model,
            config,
            injections,
            metrics,
            best_validation_loss,
            best_validation_step,
            best_checkpoint_path,
            last_checkpoint_path,
            best_metric_name,
            best_metric_phase,
        )
        provenance = _build_provenance(
            config,
            dataset,
            injections,
            step,
            activation_checkpointed_blocks=activation_checkpointed_blocks,
            best_checkpoint_path=best_checkpoint_path,
            last_checkpoint_path=last_checkpoint_path,
            best_validation_loss=best_validation_loss,
            best_validation_step=best_validation_step,
            best_metric_name=best_metric_name,
            best_metric_phase=best_metric_phase,
        )
        if _is_main_process():
            provenance_path.write_text(
                json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
            )
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        _distributed_barrier()
        return QatTrainingResult(
            output_dir=output_dir,
            checkpoint_path=best_checkpoint_path,
            best_checkpoint_path=best_checkpoint_path,
            last_checkpoint_path=last_checkpoint_path,
            metrics_path=metrics_path,
            provenance_path=provenance_path,
            train_steps=step,
            best_validation_loss=best_validation_loss,
            best_validation_step=best_validation_step,
            injections=injections,
        )
    finally:
        if loaded is not None:
            loaded.cleanup()


def _build_optimizer(
    trainable_parameters: list[nn.Parameter],
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    if config.name == "adamw":
        return torch.optim.AdamW(
            trainable_parameters,
            lr=float(config.learning_rate),
            betas=config.betas,
            eps=float(config.eps),
            weight_decay=float(config.weight_decay),
            foreach=config.foreach,
        )
    if config.name == "sgd":
        return torch.optim.SGD(
            trainable_parameters,
            lr=float(config.learning_rate),
            momentum=0.0,
            weight_decay=float(config.weight_decay),
            foreach=config.foreach,
        )
    raise ValueError(f"unsupported optimizer: {config.name}")


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
    optional_targets = _extract_optional_targets(payload, kwargs, path)
    hidden_states = _extract_hidden_states(args, kwargs, path)
    encoder_hidden_states = _extract_required_forward_tensor(kwargs, "encoder_hidden_states", path)
    timestep = _extract_timestep(kwargs, path)
    img_shapes = _extract_required_sequence(kwargs, "img_shapes", path)
    txt_seq_lens = _extract_txt_seq_lens(kwargs, path)
    if hidden_states.shape != target_output.shape:
        raise ValueError(
            f"tuple hidden_states and target_output shapes must match, got "
            f"{tuple(hidden_states.shape)} vs {tuple(target_output.shape)}: {path}"
        )
    if hidden_states.ndim >= 1 and encoder_hidden_states.ndim >= 1:
        if hidden_states.shape[0] != encoder_hidden_states.shape[0]:
            raise ValueError(
                f"tuple hidden_states and encoder_hidden_states batch sizes must match, got "
                f"{hidden_states.shape[0]} vs {encoder_hidden_states.shape[0]}: {path}"
            )
    return TransformerTupleSample(
        path=path,
        args=args,
        kwargs=kwargs,
        target_output=target_output,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_shapes=img_shapes,
        txt_seq_lens=txt_seq_lens,
        additional_t_cond=kwargs.get("additional_t_cond"),
        optional_targets=optional_targets,
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
                *OPTIONAL_TARGET_KEYS,
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


def _extract_optional_targets(
    payload: Mapping[str, object],
    kwargs: dict[str, object],
    path: Path,
) -> dict[str, torch.Tensor]:
    optional_targets: dict[str, torch.Tensor] = {}
    for key in OPTIONAL_TARGET_KEYS:
        value = payload.get(key)
        if value is None and key in kwargs:
            value = kwargs.pop(key)
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"tuple optional target {key} must be a tensor when present: {path}")
        optional_targets[key] = value
    return optional_targets


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


def _extract_required_forward_tensor(
    kwargs: Mapping[str, object],
    key: str,
    path: Path,
) -> torch.Tensor:
    value = kwargs.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"tuple payload missing {key}: {path}")
    if not torch.is_floating_point(value):
        raise ValueError(f"tuple {key} must be floating point: {path}")
    if value.numel() == 0:
        raise ValueError(f"tuple {key} tensor must not be empty: {path}")
    return value


def _extract_timestep(kwargs: Mapping[str, object], path: Path) -> object:
    if "timestep" not in kwargs:
        raise ValueError(f"tuple payload missing timestep: {path}")
    timestep = kwargs["timestep"]
    if isinstance(timestep, torch.Tensor) and timestep.numel() == 0:
        raise ValueError(f"tuple timestep tensor must not be empty: {path}")
    return timestep


def _extract_required_sequence(
    kwargs: Mapping[str, object],
    key: str,
    path: Path,
) -> object:
    if key not in kwargs:
        raise ValueError(f"tuple payload missing {key}: {path}")
    value = kwargs[key]
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            raise ValueError(f"tuple {key} tensor must not be empty: {path}")
        return value
    if isinstance(value, (list, tuple)) and value:
        return value
    raise ValueError(f"tuple {key} must be a non-empty list, tuple, or tensor: {path}")


def _extract_txt_seq_lens(kwargs: Mapping[str, object], path: Path) -> object:
    value = _extract_required_sequence(kwargs, "txt_seq_lens", path)
    if isinstance(value, torch.Tensor):
        if value.ndim > 1:
            raise ValueError(f"tuple txt_seq_lens tensor must be 1D or scalar: {path}")
        values = value.reshape(-1).tolist()
    else:
        values = list(value)
    if not values:
        raise ValueError(f"tuple txt_seq_lens must not be empty: {path}")
    for item in values:
        if not isinstance(item, int) or item <= 0:
            raise ValueError(f"tuple txt_seq_lens entries must be positive integers: {path}")
    return values


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


def _get_module_by_name(root: nn.Module, module_name: str) -> nn.Module:
    module = _unwrap_model(root)
    for part in module_name.split("."):
        module = _unwrap_model(_get_child_module(module, part))
    return module


def _get_parameter_by_name(module: nn.Module, parameter_name: str) -> nn.Parameter:
    parent = _unwrap_model(module)
    parts = parameter_name.split(".")
    for part in parts[:-1]:
        parent = _unwrap_model(_get_child_module(parent, part))
    parameter = getattr(parent, parts[-1])
    if not isinstance(parameter, nn.Parameter):
        raise TypeError(f"{parameter_name} is not a Parameter")
    return parameter


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

    fsdp_kwargs: dict[str, object] = {"use_orig_params": True}
    trainable_parameters = [
        parameter for parameter in transformer.parameters() if parameter.requires_grad
    ]
    trainables_are_replicated = False
    if (
        trainable_parameters
        and "ignored_states" in inspect.signature(FullyShardedDataParallel).parameters
    ):
        fsdp_kwargs["ignored_states"] = trainable_parameters
        trainables_are_replicated = True
    auto_wrap_policy = _fsdp_transformer_block_auto_wrap_policy(transformer)
    if auto_wrap_policy is not None:
        fsdp_kwargs["auto_wrap_policy"] = auto_wrap_policy
    if torch.device(config.device).type == "cuda":
        device_id = config.distributed.device_id
        if device_id is None:
            device_id = torch.cuda.current_device()
        fsdp_kwargs["device_id"] = torch.device("cuda", device_id)
    wrapped = FullyShardedDataParallel(transformer, **fsdp_kwargs)
    setattr(wrapped, "_qat_fsdp_trainables_are_replicated", trainables_are_replicated)
    return wrapped


def _fsdp_transformer_block_auto_wrap_policy(
    transformer: nn.Module,
) -> Callable[..., bool] | None:
    blocks = getattr(transformer, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) == 0:
        return None

    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    block_classes = {type(block) for block in blocks}
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=block_classes,
    )


def _maybe_init_distributed_from_env(config: QwenImageLayeredQatConfig) -> bool:
    if config.distributed.mode == "none":
        return False
    if not torch.distributed.is_available():
        raise ValueError("distributed training requires torch.distributed")
    if torch.distributed.is_initialized():
        return False

    required_env = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing_env = [name for name in required_env if name not in os.environ]
    if missing_env:
        formatted = ", ".join(missing_env)
        raise ValueError(
            f"distributed training requires torchrun environment variables: {formatted}"
        )

    _bind_distributed_device_from_env(config)
    backend = "nccl" if torch.device(config.device).type == "cuda" else "gloo"
    torch.distributed.init_process_group(backend=backend)
    return True


def _bind_distributed_device_from_env(config: QwenImageLayeredQatConfig) -> None:
    if config.distributed.mode == "none":
        return
    if torch.device(config.device).type != "cuda":
        return
    local_rank_text = os.environ.get("LOCAL_RANK")
    if local_rank_text is None:
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(int(local_rank_text))


def _destroy_distributed_if_initialized(initialized_here: bool) -> None:
    if initialized_here and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _distributed_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _distributed_rank() -> int:
    if not _distributed_is_initialized():
        return 0
    return int(torch.distributed.get_rank())


def _distributed_world_size() -> int:
    if not _distributed_is_initialized():
        return 1
    return int(torch.distributed.get_world_size())


def _is_main_process() -> bool:
    return _distributed_rank() == 0


def _distributed_barrier() -> None:
    if _distributed_is_initialized():
        torch.distributed.barrier()


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_cuda_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _clone_trainable_parameters(
    trainable_parameters: Sequence[nn.Parameter],
) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in trainable_parameters]


def _gradient_global_norm(trainable_parameters: Sequence[nn.Parameter]) -> float:
    total = 0.0
    for parameter in trainable_parameters:
        if parameter.grad is None:
            continue
        total += _tensor_l2_squared(parameter.grad)
    return math.sqrt(total)


def _parameter_global_norm(trainable_parameters: Sequence[nn.Parameter]) -> float:
    total = 0.0
    for parameter in trainable_parameters:
        total += _tensor_l2_squared(parameter)
    return math.sqrt(total)


def _parameter_update_global_norm(
    trainable_parameters: Sequence[nn.Parameter],
    parameter_snapshots: Sequence[torch.Tensor],
) -> float:
    total = 0.0
    for parameter, snapshot in zip(trainable_parameters, parameter_snapshots):
        total += _tensor_l2_squared(parameter.detach() - snapshot)
    return math.sqrt(total)


def _tensor_l2_squared(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().square().sum().cpu())


def _write_jsonl_record(
    metrics_jsonl: TextIO | None,
    record: Mapping[str, object],
) -> None:
    if metrics_jsonl is None:
        return
    metrics_jsonl.write(json.dumps(record, sort_keys=True) + "\n")
    metrics_jsonl.flush()


def _debug_log(message: str) -> None:
    if not os.environ.get("QWEN_QAT_DEBUG"):
        return
    rank = os.environ.get("RANK", "0")
    local_rank = os.environ.get("LOCAL_RANK", "0")
    print(
        f"[qat-debug rank={rank} local_rank={local_rank}] {message}",
        flush=True,
    )


def _enable_qat_activation_checkpointing(transformer: nn.Module) -> int:
    """Checkpoint transformer block forwards without changing module names."""
    from torch.utils.checkpoint import checkpoint

    blocks = getattr(transformer, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise ValueError("activation_checkpointing requires transformer.transformer_blocks")

    enabled = 0
    for block in blocks:
        if bool(getattr(block, "_qat_activation_checkpointing_enabled", False)):
            continue
        original_forward = block.forward

        def _checkpointed_forward(*args, _original_forward=original_forward, **kwargs):
            if not torch.is_grad_enabled():
                return _original_forward(*args, **kwargs)
            return checkpoint(
                _original_forward,
                *args,
                use_reentrant=False,
                **kwargs,
            )

        block.forward = _checkpointed_forward
        setattr(block, "_qat_activation_checkpointing_enabled", True)
        enabled += 1

    if enabled == 0:
        raise ValueError("activation_checkpointing found no new transformer blocks to wrap")
    return enabled


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


def _training_and_validation_indices(
    dataset: TransformerTupleDataset,
    config: QwenImageLayeredQatConfig,
) -> tuple[list[int], list[int]]:
    if config.train_all_tuples:
        return list(range(len(dataset))), []
    return _split_dataset_indices(dataset, config.validation_fraction)


def _monitor_indices(
    dataset: TransformerTupleDataset,
    config: QwenImageLayeredQatConfig,
) -> tuple[list[int], list[int]]:
    subset_indices: list[int] = []
    if config.monitor_subset_size is not None:
        subset_indices = _select_monitor_subset_indices(
            dataset,
            int(config.monitor_subset_size),
            int(config.monitor_subset_seed),
        )
    all_indices = list(range(len(dataset))) if config.monitor_all_tuples else []
    return subset_indices, all_indices


def _select_monitor_subset_indices(
    dataset: TransformerTupleDataset,
    subset_size: int,
    seed: int,
) -> list[int]:
    indices = list(range(len(dataset)))
    if subset_size > len(indices):
        raise ValueError(f"monitor_subset_size={subset_size} exceeds tuple count {len(indices)}")
    rng = random.Random(seed)
    rng.shuffle(indices)
    return sorted(indices[:subset_size])


def _shuffled_train_indices_for_epoch(
    train_indices: Sequence[int],
    *,
    seed: int,
    epoch: int,
) -> list[int]:
    shuffled = list(train_indices)
    rng = random.Random(seed + epoch)
    rng.shuffle(shuffled)
    return shuffled


def _training_sample_index_for_step(
    train_indices: Sequence[int],
    *,
    step: int,
    shuffle: bool,
    seed: int,
    epoch_orders: dict[int, list[int]],
) -> int:
    epoch = (step - 1) // len(train_indices)
    position = (step - 1) % len(train_indices)
    if not shuffle:
        return int(train_indices[position])
    if epoch not in epoch_orders:
        epoch_orders[epoch] = _shuffled_train_indices_for_epoch(
            train_indices,
            seed=seed,
            epoch=epoch,
        )
    return int(epoch_orders[epoch][position])


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


def _validate_loss_payloads(
    loss_config: LossConfig,
    dataset: TransformerTupleDataset,
) -> None:
    if loss_config.latent_weight > 0.0:
        for sample in dataset.samples:
            if not isinstance(sample.target_output, torch.Tensor):
                raise ValueError(
                    f"latent reconstruction loss requires target_output: {sample.path}"
                )

    for component_name, weight_name in _DIFFERENTIABLE_OPTIONAL_LOSSES.items():
        weight = getattr(loss_config, weight_name)
        if weight == 0.0:
            continue
        target_key = _OPTIONAL_LOSS_TARGETS[component_name]
        missing_targets = [
            str(sample.path)
            for sample in dataset.samples
            if target_key not in sample.optional_targets
        ]
        if missing_targets:
            raise ValueError(
                f"{component_name} loss requires optional tensor {target_key}; "
                f"missing in {missing_targets[:3]}"
            )

    if loss_config.perceptual_weight > 0.0:
        target_key = _OPTIONAL_LOSS_TARGETS["perceptual"]
        missing_targets = [
            str(sample.path)
            for sample in dataset.samples
            if target_key not in sample.optional_targets
        ]
        if missing_targets:
            raise ValueError(
                f"perceptual loss requires optional tensor {target_key}; "
                f"missing in {missing_targets[:3]}"
            )
        raise ValueError(
            "perceptual loss requires a concrete perceptual evaluator; "
            "the current trainer supports differentiable tensor reconstruction losses only"
        )


def _mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if output.shape != target.shape:
        raise ValueError(
            f"transformer output and target_output shapes must match, got "
            f"{tuple(output.shape)} vs {tuple(target.shape)}"
        )
    return F.mse_loss(output.float(), target.to(dtype=output.dtype).float())


def _compute_loss(
    output: torch.Tensor,
    sample: TransformerTupleSample,
    loss_config: LossConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_components: dict[str, torch.Tensor] = {}
    total_loss = output.float().new_zeros(())

    if loss_config.latent_weight > 0.0:
        latent_loss = _mse_loss(output, sample.target_output.to(device=device))
        loss_components["latent_reconstruction"] = latent_loss
        total_loss = total_loss + latent_loss * float(loss_config.latent_weight)

    for component_name, weight_name in _DIFFERENTIABLE_OPTIONAL_LOSSES.items():
        weight = getattr(loss_config, weight_name)
        if weight == 0.0:
            continue
        target_key = _OPTIONAL_LOSS_TARGETS[component_name]
        target = sample.optional_targets[target_key].to(device=device)
        prediction = _derive_optional_reconstruction(output, target, component_name, sample.path)
        reconstruction_loss = F.mse_loss(prediction.float(), target.float())
        loss_components[f"{component_name}_reconstruction"] = reconstruction_loss
        total_loss = total_loss + reconstruction_loss * float(weight)

    return total_loss, loss_components


def _derive_optional_reconstruction(
    output: torch.Tensor,
    target: torch.Tensor,
    component_name: str,
    sample_path: Path,
) -> torch.Tensor:
    if target.numel() == 0:
        raise ValueError(f"{component_name} target must not be empty: {sample_path}")
    if not torch.is_floating_point(target):
        raise ValueError(f"{component_name} target must be floating point: {sample_path}")
    if output.numel() == 0:
        raise ValueError(
            f"transformer output must not be empty for {component_name}: {sample_path}"
        )
    output_float = output.float()
    target_float = target.float()
    if output_float.shape == target_float.shape:
        return output_float
    if output_float.numel() == target_float.numel():
        return output_float.reshape_as(target_float)

    output_batch = int(output_float.shape[0]) if output_float.ndim > 0 else 1
    target_batch = int(target_float.shape[0]) if target_float.ndim > 0 else 1
    if output_batch == target_batch and target_float.ndim > 0:
        flat_output = output_float.reshape(output_batch, -1)
        target_width = int(target_float.numel() // target_batch)
        return _resize_flat_features(flat_output, target_width).reshape_as(target_float)

    flat_output = output_float.reshape(1, -1)
    return _resize_flat_features(flat_output, int(target_float.numel())).reshape_as(target_float)


def _resize_flat_features(flat_output: torch.Tensor, target_width: int) -> torch.Tensor:
    if target_width <= 0:
        raise ValueError("target reconstruction width must be positive")
    if flat_output.shape[-1] == target_width:
        return flat_output
    return F.interpolate(
        flat_output.unsqueeze(1),
        size=target_width,
        mode="linear",
        align_corners=False,
    ).squeeze(1)


def _loss_components_to_float(
    loss_components: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in loss_components.items()}


def _validate(
    transformer: nn.Module,
    dataset: TransformerTupleDataset,
    validation_indices: Sequence[int],
    loss_config: LossConfig,
    device: torch.device,
) -> ValidationMetrics:
    transformer.eval()
    losses: list[float] = []
    component_totals: dict[str, float] = {}
    quality_totals: dict[str, float] = {}
    quality_counts: dict[str, int] = {}
    with torch.inference_mode():
        for index in validation_indices:
            sample = dataset[index]
            output = _forward_sample(transformer, sample, device)
            loss, loss_components = _compute_loss(output, sample, loss_config, device)
            losses.append(float(loss.detach().cpu()))
            for name, value in _loss_components_to_float(loss_components).items():
                component_totals[name] = component_totals.get(name, 0.0) + value
            for name, value in _optional_quality_metrics(output, sample).items():
                quality_totals[name] = quality_totals.get(name, 0.0) + value
                quality_counts[name] = quality_counts.get(name, 0) + 1
    transformer.train()
    sample_count = len(losses)
    loss_components_avg = {
        name: value / sample_count for name, value in sorted(component_totals.items())
    }
    quality_metrics_avg = {
        name: quality_totals[name] / quality_counts[name] for name in sorted(quality_totals)
    }
    return ValidationMetrics(
        loss=sum(losses) / sample_count,
        loss_components=loss_components_avg,
        quality_metrics=quality_metrics_avg,
    )


def _evaluate_and_record(
    transformer: nn.Module,
    dataset: TransformerTupleDataset,
    indices: Sequence[int],
    loss_config: LossConfig,
    device: torch.device,
    metrics: list[dict[str, object]],
    metrics_jsonl: TextIO | None,
    *,
    phase: str,
    metric_name: str,
    step: int,
) -> float:
    evaluation_metrics = _validate(transformer, dataset, indices, loss_config, device)
    evaluation_loss = float(evaluation_metrics.loss)
    if not math.isfinite(evaluation_loss):
        formatted_metric_name = metric_name.replace("_", " ")
        raise RuntimeError(f"non-finite {formatted_metric_name} at step {step}: {evaluation_loss}")
    evaluation_record = {
        "phase": phase,
        "step": step,
        "loss": evaluation_loss,
        metric_name: evaluation_loss,
        "metric_name": metric_name,
        "loss_components": evaluation_metrics.loss_components,
        "quality_metrics": evaluation_metrics.quality_metrics,
        "sample_count": len(indices),
        "rank": _distributed_rank(),
        "world_size": _distributed_world_size(),
    }
    metrics.append(evaluation_record)
    _write_jsonl_record(metrics_jsonl, evaluation_record)
    return evaluation_loss


def _optional_quality_metrics(
    output: torch.Tensor,
    sample: TransformerTupleSample,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for prefix in ("layered_rgba", "composite", "alpha_mask"):
        target = sample.optional_targets.get(f"{prefix}_target")
        if target is None:
            continue
        target = target.to(device=output.device)
        prediction = _derive_optional_reconstruction(output, target, prefix, sample.path)
        metrics[f"{prefix}_psnr"] = _psnr(prediction, target)
        metrics[f"{prefix}_ssim"] = _simple_ssim(prediction, target)
    return metrics


def _psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(prediction.float(), target.float()).item()
    if mse == 0.0:
        return 120.0
    return 20.0 * math.log10(1.0) - 10.0 * math.log10(mse)


def _simple_ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    x = prediction.float().reshape(-1)
    y = target.float().reshape(-1)
    mean_x = x.mean()
    mean_y = y.mean()
    var_x = ((x - mean_x) ** 2).mean()
    var_y = ((y - mean_y) ** 2).mean()
    covariance = ((x - mean_x) * (y - mean_y)).mean()
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2)
    denominator = (mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2)
    return float((numerator / denominator).cpu())


def _unwrap_model(model: nn.Module) -> nn.Module:
    unwrapped = model
    seen: set[int] = set()
    while id(unwrapped) not in seen:
        seen.add(id(unwrapped))
        module = getattr(unwrapped, "module", None)
        if isinstance(module, nn.Module):
            unwrapped = module
            continue
        fsdp_wrapped_module = getattr(unwrapped, "_fsdp_wrapped_module", None)
        if isinstance(fsdp_wrapped_module, nn.Module):
            unwrapped = fsdp_wrapped_module
            continue
        break
    return unwrapped


def _contains_fsdp(model: nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except ImportError:
        return False
    return any(isinstance(module, FullyShardedDataParallel) for module in model.modules())


def _normalize_state_name(name: str) -> str:
    prefixes = ("module.", "_fsdp_wrapped_module.", "_checkpoint_wrapped_module.")
    normalized = name
    changed = True
    while changed:
        changed = False
        if "._checkpoint_wrapped_module." in normalized:
            normalized = normalized.replace("._checkpoint_wrapped_module.", ".")
            changed = True
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
    return normalized


def _trainable_parameter_names_from_injections(
    injections: Sequence[QatInjectionInfo],
) -> set[str]:
    return {
        _normalize_state_name(f"{injection.module_name}.{parameter_name}")
        for injection in injections
        for parameter_name in injection.trainable_parameter_names
    }


def _requires_grad_parameter_names(model: nn.Module) -> set[str]:
    return {
        _normalize_state_name(name)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _trainable_state_dict_from_model(
    model: nn.Module,
    trainable_names: set[str],
) -> dict[str, torch.Tensor]:
    unwrapped = _unwrap_model(model)
    return {
        _normalize_state_name(name): parameter.detach().cpu()
        for name, parameter in unwrapped.named_parameters()
        if _normalize_state_name(name) in trainable_names
    }


def _trainable_state_dict_from_injections(
    model: nn.Module,
    injections: Sequence[QatInjectionInfo],
    trainable_names: set[str],
) -> dict[str, torch.Tensor]:
    trainable_state_dict: dict[str, torch.Tensor] = {}
    for injection in injections:
        module = _get_module_by_name(model, injection.module_name)
        for parameter_name in injection.trainable_parameter_names:
            state_name = _normalize_state_name(f"{injection.module_name}.{parameter_name}")
            if state_name not in trainable_names:
                continue
            parameter = _get_parameter_by_name(module, parameter_name)
            trainable_state_dict[state_name] = parameter.detach().cpu()
    return trainable_state_dict


def _trainable_state_dict_from_fsdp(
    model: nn.Module,
    trainable_names: set[str],
) -> dict[str, torch.Tensor]:
    from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel, StateDictType

    state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FullyShardedDataParallel.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        state_config,
    ):
        full_state_dict = model.state_dict()
    if not _is_main_process():
        return {}
    return {
        _normalize_state_name(name): tensor.detach().cpu()
        for name, tensor in full_state_dict.items()
        if _normalize_state_name(name) in trainable_names
    }


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    config: QwenImageLayeredQatConfig,
    injections: Sequence[QatInjectionInfo],
    metrics: Sequence[Mapping[str, object]],
    best_validation_loss: float,
    best_validation_step: int | None,
    best_checkpoint_path: Path,
    last_checkpoint_path: Path,
    best_metric_name: str,
    best_metric_phase: str,
) -> None:
    trainable_names = _trainable_parameter_names_from_injections(injections)
    if not trainable_names:
        trainable_names = _requires_grad_parameter_names(model)
    if _contains_fsdp(model):
        if config.recipe == "lora_adapter" or bool(
            getattr(model, "_qat_fsdp_trainables_are_replicated", False)
        ):
            if not _is_main_process():
                return
            _debug_log(f"saving {len(trainable_names)} FSDP trainable tensors directly")
            trainable_state_dict = _trainable_state_dict_from_injections(
                model,
                injections,
                trainable_names,
            )
        else:
            _debug_log("saving FSDP trainable tensors through full state dict")
            trainable_state_dict = _trainable_state_dict_from_fsdp(model, trainable_names)
    else:
        if not _is_main_process():
            return
        trainable_state_dict = _trainable_state_dict_from_model(model, trainable_names)
    if not _is_main_process():
        return
    if trainable_names and not trainable_state_dict:
        raise RuntimeError(
            "QAT checkpoint contains no trainable tensors after state-dict filtering"
        )
    _debug_log(f"writing QAT checkpoint {path} with {len(trainable_state_dict)} tensors")
    checkpoint = {
        "format": TRAINING_FORMAT,
        "config": config.model_dump(),
        "trainable_state_dict": trainable_state_dict,
        "trainable_parameter_names": sorted(trainable_state_dict),
        "injections": [_injection_to_dict(injection) for injection in injections],
        "metrics": list(metrics),
        "best_validation_loss": best_validation_loss,
        "best_validation_step": best_validation_step,
        "best_metric_name": best_metric_name,
        "best_metric_phase": best_metric_phase,
        "checkpoint_path": str(path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
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
    *,
    activation_checkpointed_blocks: int,
    best_checkpoint_path: Path,
    last_checkpoint_path: Path,
    best_validation_loss: float,
    best_validation_step: int | None,
    best_metric_name: str,
    best_metric_phase: str,
) -> dict[str, object]:
    return {
        "format": TRAINING_FORMAT,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "distributed": {
            "mode": config.distributed.mode,
            "rank": _distributed_rank(),
            "world_size": _distributed_world_size(),
            "local_rank": os.environ.get("LOCAL_RANK"),
        },
        "config": config.model_dump(),
        "tuple_manifest": str(dataset.manifest_path),
        "tuple_count": len(dataset),
        "tuple_paths": [str(path) for path in dataset.paths],
        "train_steps": train_steps,
        "activation_checkpointed_blocks": activation_checkpointed_blocks,
        "peak_cuda_memory_bytes": (
            _peak_cuda_memory_bytes(torch.device(config.device))
            if torch.cuda.is_available()
            else None
        ),
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "best_validation_loss": best_validation_loss,
        "best_validation_step": best_validation_step,
        "best_metric_name": best_metric_name,
        "best_metric_phase": best_metric_phase,
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
    _bind_distributed_device_from_env(config)
    try:
        result = train_qwen_image_layered_qat(config)
        if _is_main_process():
            print(
                json.dumps(
                    {
                        "checkpoint_path": str(result.checkpoint_path),
                        "best_checkpoint_path": str(result.best_checkpoint_path),
                        "last_checkpoint_path": str(result.last_checkpoint_path),
                        "metrics_path": str(result.metrics_path),
                        "provenance_path": str(result.provenance_path),
                        "train_steps": result.train_steps,
                        "best_validation_loss": result.best_validation_loss,
                        "best_validation_step": result.best_validation_step,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    finally:
        if config.distributed.mode != "none":
            _destroy_distributed_if_initialized(initialized_here=True)


if __name__ == "__main__":
    main()
