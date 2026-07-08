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

"""Ordinary Qwen-Image tuple replay and loss helpers for MXFP8 QAT."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from scripts.visualgen_eval.qwen_image_capture_manifest import (
    BF16_TEACHER_TRAJECTORY_SOURCE,
    DEFAULT_CLUSTER_ALIAS,
    DEFAULT_CONTAINER_RUNTIME,
    DEFAULT_ENROOT_IMAGE,
    git_commit,
    write_json,
)
from scripts.visualgen_eval.qwen_image_teacher_capture import (
    CAPTURED_TUPLE_STATUS,
    cleanup_pipeline,
    load_single_worker_pipeline,
    validate_teacher_tuple_payload,
)


def _load_source_fake_mxfp8() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "tensorrt_llm"
        / "_torch"
        / "visual_gen"
        / "quantization"
        / "fake_mxfp8.py"
    )
    spec = importlib.util.spec_from_file_location("_qwen_image_source_fake_mxfp8", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load fake_mxfp8 module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


try:
    from tensorrt_llm._torch.visual_gen.quantization.fake_mxfp8 import (
        MXFP8_BLOCK_SIZE,
        QWEN_IMAGE_BLOCK_LINEAR_COUNT,
        SUPPORTED_FAKE_MXFP8_DTYPES,
        FakeMxfp8Linear,
        Mxfp8ScaleMode,
        QwenImageBlockLinearTarget,
        fake_mxfp8_weight_quantize,
        normalize_qwen_module_name,
        select_qwen_image_block_linears,
    )
except ModuleNotFoundError as error:
    source_fallback_names = {
        "tensorrt_llm.bindings",
        "tensorrt_llm._torch.visual_gen.quantization",
        "tensorrt_llm._torch.visual_gen.quantization.fake_mxfp8",
    }
    if error.name not in source_fallback_names:
        raise
    _fake_mxfp8 = _load_source_fake_mxfp8()
    MXFP8_BLOCK_SIZE = _fake_mxfp8.MXFP8_BLOCK_SIZE
    SUPPORTED_FAKE_MXFP8_DTYPES = _fake_mxfp8.SUPPORTED_FAKE_MXFP8_DTYPES
    FakeMxfp8Linear = _fake_mxfp8.FakeMxfp8Linear
    Mxfp8ScaleMode = _fake_mxfp8.Mxfp8ScaleMode
    QWEN_IMAGE_BLOCK_LINEAR_COUNT = _fake_mxfp8.QWEN_IMAGE_BLOCK_LINEAR_COUNT
    QwenImageBlockLinearTarget = _fake_mxfp8.QwenImageBlockLinearTarget
    fake_mxfp8_weight_quantize = _fake_mxfp8.fake_mxfp8_weight_quantize
    normalize_qwen_module_name = _fake_mxfp8.normalize_qwen_module_name
    select_qwen_image_block_linears = _fake_mxfp8.select_qwen_image_block_linears


TRAINING_FORMAT = "qwen_image_mxfp8_qat_adapter_v1"
PROBE_SUMMARY_FORMAT = "qwen_image_mxfp8_qat_probe_summary_v1"
MONITOR_SUMMARY_FORMAT = "qwen_image_mxfp8_qat_checkpoint_monitor_v1"
ROLLOUT_MONITOR_SUMMARY_FORMAT = "qwen_image_mxfp8_qat_rollout_monitor_v1"
ROLLOUT_METADATA_MANIFEST_FORMAT = "qwen_image_rollout_metadata_manifest_v1"
ROLLOUT_AUGMENTATION_SUMMARY_FORMAT = "qwen_image_rollout_tuple_augmentation_summary_v1"
ROLLOUT_QAT_SUMMARY_FORMAT = "qwen_image_closed_set_rollout_qat_summary_v1"
ROLLOUT_TUPLE_SCHEMA_VERSION = "qwen_image_closed_set_rollout_tuple_v1"
ROLLOUT_SCHEDULER_STEP_IMPLEMENTATION = "qwen_image_flowmatch_euler_sigma_delta_v1"
FORMAL_QWEN_IMAGE_BLOCK_LAYER_COUNT = 60
FORMAL_QWEN_IMAGE_BLOCK_LINEAR_TARGET_COUNT = (
    FORMAL_QWEN_IMAGE_BLOCK_LAYER_COUNT * QWEN_IMAGE_BLOCK_LINEAR_COUNT
)
SUPPORTED_ROLLOUT_SCHEDULER_NAMES = frozenset(
    {
        "flowmatcheulerdiscrete",
        "flowmatcheulerdiscretescheduler",
        "qwenimageflowmatcheulerdiscretescheduler",
    }
)
UNSUPPORTED_ROLLOUT_SCHEDULER_FLAGS = (
    "per_token_sigmas",
    "per_token_timesteps",
    "stochastic_sampling",
    "use_stochastic_sampling",
)
QWEN_BLOCK_LINEAR_TARGET = "qwen_block_linears"
TARGET_OUTPUT_KEY = "target_output"
ROLLOUT_COND_BRANCH = "cond"
ROLLOUT_NEGATIVE_BRANCH = "negative"
ROLLOUT_CFG_BRANCHES = (ROLLOUT_COND_BRANCH, ROLLOUT_NEGATIVE_BRANCH)
QWEN_IMAGE_QAT_PROBE_RECIPES = (
    "mse_only",
    "timestep_weighted",
    "direction_aware",
    "scale_aware_lora",
)
DEFAULT_TIMESTEP_WEIGHTS = {
    "early": 1.0,
    "early_mid": 1.25,
    "mid": 1.5,
    "late_mid": 2.0,
    "late": 3.0,
}
CLOSED_SET_ROLLOUT_TIMESTEP_WEIGHTS = {
    "early": 1.0,
    "early_mid": 2.0,
    "mid": 3.0,
    "late_mid": 4.0,
    "late": 8.0,
}
ROLLOUT_REQUIRED_FIELDS = (
    "latent_before_step",
    "latent_after_step",
    "scheduler_state",
    "rollout_tuple_schema",
    "rollout_provenance",
    "guidance_scale",
    "reference_image_path",
)
ROLLOUT_PROVENANCE_REQUIRED_FIELDS = (
    "derivation_method",
    "scheduler_config_signature",
    "sigmas_hash",
    "git_head",
    "prompt_id",
    "seed",
    "height",
    "width",
    "num_inference_steps",
    "guidance_scale",
    "reference_image_path",
)
ROLLOUT_PER_STEP_SCHEDULER_FIELDS = frozenset(
    (
        "begin_index",
        "index",
        "sigma",
        "sigma_delta",
        "sigma_next",
        "sigmas",
        "dt",
        "next_sigma",
        "step_index",
        "timestep",
        "timestep_index",
        "timesteps",
    )
)
_LAYERED_ONLY_FIELDS = frozenset(
    (
        "alpha_mask",
        "alpha_mask_target",
        "composite",
        "composite_target",
        "image",
        "layered_rgba",
        "layered_rgba_target",
        "layers",
    )
)


@dataclass(frozen=True)
class QwenImageTupleSample:
    """One ordinary Qwen-Image transformer tuple and BF16 teacher target."""

    path: Path
    entry: Mapping[str, object]
    target_output: torch.Tensor
    hidden_states: torch.Tensor
    latent_before_step: torch.Tensor | None
    latent_after_step: torch.Tensor | None
    scheduler_state: Mapping[str, object] | None
    rollout_tuple_schema: str | None
    rollout_provenance: Mapping[str, object] | None
    guidance_scale: float | None
    reference_image_path: str | None
    timestep: torch.Tensor
    additional_t_cond: torch.Tensor | None
    encoder_hidden_states: torch.Tensor
    encoder_hidden_states_mask: torch.Tensor | None
    img_shapes: object
    txt_seq_lens: object
    prompt_id: str
    split: str
    timestep_index: int
    timestep_bin: str
    cfg_branch: str


@dataclass(frozen=True)
class QwenImageTupleLossConfig:
    """Loss weights for teacher-tuple distillation."""

    lambda_mse: float = 1.0
    lambda_dir: float = 0.0
    timestep_weights: Mapping[str, float] | None = None


@dataclass(frozen=True)
class QwenImageRolloutLossConfig:
    """K-step latent rollout loss config for closed-set trajectory repair."""

    rollout_k: int = 4
    lambda_output_mse: float = 1.0
    lambda_dir: float = 0.1
    lambda_latent: float = 1.0
    lambda_anchor: float = 0.2
    timestep_weights: Mapping[str, float] | None = field(
        default_factory=lambda: dict(CLOSED_SET_ROLLOUT_TIMESTEP_WEIGHTS)
    )
    teacher_no_grad: bool = True
    student_latent_detach: bool = False
    cfg_normalize: bool = True
    student_activation_checkpoint: bool = False


@dataclass(frozen=True)
class QwenImageRolloutStepSamples:
    """Paired CFG branch samples for one denoise step."""

    timestep_index: int
    timestep_bin: str
    cond: QwenImageTupleSample
    negative: QwenImageTupleSample
    guidance_scale: float


@dataclass(frozen=True)
class QwenImageQatInjectionInfo:
    """Trainable LoRA fake-MXFP8 wrapper inserted into one selected Linear."""

    module_name: str
    block_index: int
    role: str
    trainable_parameter_names: tuple[str, ...]
    lora_rank: int
    lora_alpha: float
    scale_multiplier: float


@dataclass(frozen=True)
class QwenImageQatTrainingConfig:
    """Minimal native PyTorch training loop config for tuple-level QAT probes."""

    tuple_index_jsonl: str | Path
    output_dir: str | Path
    max_steps: int
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    device: str = "cuda"
    target_layers: tuple[str, ...] = (QWEN_BLOCK_LINEAR_TARGET,)
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    expected_num_layers: int | None = None
    expected_target_count: int | None = None
    loss: QwenImageTupleLossConfig = field(default_factory=QwenImageTupleLossConfig)
    log_interval_steps: int = 1
    checkpoint_name: str = "qwen_image_qat_lora_last.pt"
    metrics_name: str = "qwen_image_qat_train_metrics.jsonl"
    compute_lora_delta_norm: bool = False
    scale_multipliers: Mapping[str, float] | None = None
    warmup_steps: int = 0
    warmup_target_layers: tuple[str, ...] = ()
    optimizer_foreach: bool | None = False
    sample_stride: int = 1
    sample_start_index: int = 0


@dataclass(frozen=True)
class QwenImageQatTrainingResult:
    """Artifacts emitted by one tuple-level QAT training run."""

    output_dir: Path
    metrics_path: Path
    checkpoint_path: Path
    train_steps: int
    injections: tuple[QwenImageQatInjectionInfo, ...]


@dataclass(frozen=True)
class QwenImageRolloutTupleAugmentationConfig:
    """Write rollout-schema tuples from captured transformer tuples plus trajectory metadata."""

    source_tuple_index_jsonl: str | Path
    output_tuple_index_jsonl: str | Path
    output_tuple_root: str | Path
    metadata_by_tuple_id: Mapping[str, Mapping[str, object]]
    summary_json: str | Path | None = None
    validate_prompt_continuity: bool = True
    provenance: Mapping[str, object] | None = None


@dataclass(frozen=True)
class QwenImageRolloutQatTrainingConfig:
    """Minimal closed-set K-step rollout QAT training config."""

    tuple_index_jsonl: str | Path
    output_dir: str | Path
    max_steps: int
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    grad_clip_norm: float = 1.0
    device: str = "cuda"
    target_layers: tuple[str, ...] = (QWEN_BLOCK_LINEAR_TARGET,)
    lora_rank: int = 64
    lora_alpha: float = 128.0
    lora_dropout: float = 0.0
    expected_num_layers: int | None = FORMAL_QWEN_IMAGE_BLOCK_LAYER_COUNT
    expected_target_count: int | None = FORMAL_QWEN_IMAGE_BLOCK_LINEAR_TARGET_COUNT
    diagnostic_only: bool = False
    teacher_target_mode: str = "on_policy"
    loss: QwenImageRolloutLossConfig = field(default_factory=QwenImageRolloutLossConfig)
    log_interval_steps: int = 1
    checkpoint_name: str = "qwen_image_closed_set_rollout_qat_lora_last.pt"
    checkpoint_interval_steps: int | None = None
    metrics_name: str = "qwen_image_closed_set_rollout_qat_train_metrics.jsonl"
    compute_lora_delta_norm: bool = False
    scale_multipliers: Mapping[str, float] | None = None
    optimizer_foreach: bool | None = False
    window_stride: int = 1
    window_start_index: int = 0


@dataclass(frozen=True)
class QwenImageRolloutQatTrainingResult:
    """Artifacts emitted by one rollout QAT training smoke/run."""

    output_dir: Path
    metrics_path: Path
    checkpoint_path: Path
    train_steps: int
    injections: tuple[QwenImageQatInjectionInfo, ...]
    rollout_window_count: int


@dataclass(frozen=True)
class QwenImageQatMonitorConfig:
    """Fixed tuple-monitor config for one trained QAT adapter checkpoint."""

    checkpoint_path: str | Path
    tuple_index_jsonl: str | Path
    output_json: str | Path
    records_jsonl: str | Path | None = None
    max_samples: int | None = None
    sample_stride: int = 1
    sample_start_index: int = 0
    device: str = "cuda"
    loss: QwenImageTupleLossConfig | None = None


class Mxfp8LoraAdapterLinear(nn.Module):
    """Low-rank trainable residual around a frozen fake-MXFP8 Linear."""

    def __init__(
        self,
        base: FakeMxfp8Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        scale_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0.0:
            raise ValueError("LoRA alpha must be positive")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if scale_multiplier <= 0.0:
            raise ValueError("LoRA scale_multiplier must be positive")

        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale_multiplier = float(scale_multiplier)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout))
        factory_kwargs = {
            "device": base.weight.device,
            "dtype": base.weight.dtype,
        }
        self.lora_down = nn.Linear(base.in_features, self.rank, bias=False, **factory_kwargs)
        self.lora_up = nn.Linear(self.rank, base.out_features, bias=False, **factory_kwargs)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def lora_delta_weight(self) -> torch.Tensor:
        """Return the dense LoRA delta that would be merged into the base weight."""
        delta = self.lora_up.weight.float() @ self.lora_down.weight.float()
        return (delta * self.scaling * self.scale_multiplier).to(
            dtype=self.weight.dtype,
            device=self.weight.device,
        )

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        base_output = self.base(activation)
        adapter_output = (
            self.lora_up(self.lora_down(self.dropout(activation)))
            * self.scaling
            * self.scale_multiplier
        )
        return base_output + adapter_output


class QwenImageTupleDataset(torch.utils.data.Dataset):
    """Dataset for ordinary Qwen-Image BF16 teacher tuple indexes."""

    def __init__(self, tuple_index_jsonl: str | Path) -> None:
        self.tuple_index_jsonl = Path(tuple_index_jsonl)
        self.entries = _read_tuple_index(self.tuple_index_jsonl)
        self.samples = [
            load_qwen_image_tuple_sample(
                _tuple_path_from_entry(entry, manifest_dir=self.tuple_index_jsonl.parent),
                entry=entry,
            )
            for entry in self.entries
        ]
        if not self.samples:
            raise ValueError(f"tuple index contains no entries: {self.tuple_index_jsonl}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> QwenImageTupleSample:
        return self.samples[index]


class QwenImageRolloutWindowDataset(torch.utils.data.Dataset):
    """Dataset of consecutive paired-CFG rollout windows."""

    def __init__(
        self,
        tuple_index_jsonl: str | Path,
        *,
        rollout_k: int,
    ) -> None:
        if rollout_k <= 0:
            raise ValueError("rollout_k must be positive")
        self.tuple_index_jsonl = Path(tuple_index_jsonl)
        self.rollout_k = int(rollout_k)
        self.base_dataset = QwenImageTupleDataset(self.tuple_index_jsonl)
        self.windows = _build_rollout_windows_from_samples(
            self.base_dataset.samples,
            rollout_k=self.rollout_k,
        )
        if not self.windows:
            raise ValueError(f"tuple index contains no rollout windows: {self.tuple_index_jsonl}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[QwenImageTupleSample, ...]:
        return self.windows[index]


def augment_qwen_image_rollout_tuple_dataset(
    config: QwenImageRolloutTupleAugmentationConfig,
) -> list[dict[str, object]]:
    """Write rollout-schema tuple payloads and index entries from captured tuple payloads."""
    source_index_path = Path(config.source_tuple_index_jsonl)
    output_index_path = Path(config.output_tuple_index_jsonl)
    output_tuple_root = Path(config.output_tuple_root)
    source_entries = _read_tuple_index(source_index_path)
    output_entries: list[dict[str, object]] = []

    for source_entry in source_entries:
        tuple_id = _tuple_id_for_entry(source_entry)
        metadata = config.metadata_by_tuple_id.get(tuple_id)
        if metadata is None:
            raise ValueError(f"missing rollout metadata for tuple_id={tuple_id}")
        source_tuple_path = _tuple_path_from_entry(
            source_entry,
            manifest_dir=source_index_path.parent,
        )
        payload = torch.load(source_tuple_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"source tuple payload must be a mapping: {source_tuple_path}")
        rollout_payload = build_qwen_image_rollout_tuple_payload(
            payload,
            entry=source_entry,
            metadata=metadata,
            path=source_tuple_path,
        )
        output_tuple_path = _rollout_output_tuple_path(
            output_tuple_root,
            source_entry,
            source_tuple_path=source_tuple_path,
        )
        output_tuple_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(rollout_payload, output_tuple_path)
        output_entry = _build_rollout_tuple_index_entry(
            source_entry,
            metadata=metadata,
            output_tuple_path=output_tuple_path,
            output_index_dir=output_index_path.parent,
        )
        load_sample = load_qwen_image_tuple_sample(output_tuple_path, entry=output_entry)
        validate_qwen_image_rollout_tuple_sample(load_sample)
        output_entries.append(output_entry)

    output_index_path.parent.mkdir(parents=True, exist_ok=True)
    output_index_path.write_text(
        "\n".join(json.dumps(entry, sort_keys=True) for entry in output_entries) + "\n",
        encoding="utf-8",
    )
    if config.validate_prompt_continuity and output_entries:
        _validate_rollout_prompt_continuity(
            tuple(
                load_qwen_image_tuple_sample(
                    _tuple_path_from_entry(entry, manifest_dir=output_index_path.parent),
                    entry=entry,
                )
                for entry in output_entries
            )
        )
    if config.summary_json is not None:
        write_json(
            Path(config.summary_json),
            _build_rollout_augmentation_summary(
                config=config,
                output_entries=output_entries,
                source_index_path=source_index_path,
                output_index_path=output_index_path,
                output_tuple_root=output_tuple_root,
            ),
        )
    return output_entries


def load_qwen_image_rollout_metadata_manifest(
    path: str | Path,
) -> dict[str, dict[str, object]]:
    """Load durable rollout metadata keyed by tuple_id from a JSONL manifest."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise ValueError(f"rollout metadata manifest does not exist: {manifest_path}")
    metadata_by_tuple_id: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"metadata manifest line {line_number} must be a mapping")
        tuple_id = record.get("tuple_id")
        if not isinstance(tuple_id, str) or not tuple_id:
            raise ValueError(f"metadata manifest line {line_number} missing tuple_id")
        if tuple_id in metadata_by_tuple_id:
            raise ValueError(f"duplicate rollout metadata tuple_id={tuple_id}")
        metadata_payload = _load_rollout_metadata_payload(record, manifest_path.parent, tuple_id)
        metadata_by_tuple_id[tuple_id] = {
            "latent_before_step": _rollout_metadata_tensor_from_record(
                record,
                metadata_payload,
                "latent_before_step",
                manifest_path.parent,
                tuple_id,
            ),
            "latent_after_step": _rollout_metadata_tensor_from_record(
                record,
                metadata_payload,
                "latent_after_step",
                manifest_path.parent,
                tuple_id,
            ),
            "scheduler_state": _rollout_metadata_mapping_from_record(
                record,
                metadata_payload,
                "scheduler_state",
                tuple_id,
            ),
            "rollout_provenance": _rollout_metadata_mapping_from_record(
                record,
                metadata_payload,
                "rollout_provenance",
                tuple_id,
            ),
            "guidance_scale": _rollout_metadata_float_from_record(
                record,
                metadata_payload,
                "guidance_scale",
                tuple_id,
            ),
            "reference_image_path": _rollout_metadata_string_from_record(
                record,
                metadata_payload,
                "reference_image_path",
                tuple_id,
            ),
        }
    if not metadata_by_tuple_id:
        raise ValueError(f"rollout metadata manifest is empty: {manifest_path}")
    return metadata_by_tuple_id


def build_qwen_image_rollout_tuple_payload(
    payload: Mapping[str, object],
    *,
    entry: Mapping[str, object],
    metadata: Mapping[str, object],
    path: Path,
) -> dict[str, object]:
    """Add closed-set rollout fields to one ordinary captured tuple payload."""
    rollout_payload = dict(payload)
    tuple_id = _tuple_id_for_entry(entry)
    latent_before_step = _metadata_tensor(metadata, "latent_before_step", tuple_id)
    latent_after_step = _metadata_tensor(metadata, "latent_after_step", tuple_id)
    hidden_states = _expect_tensor(rollout_payload, "hidden_states", path)
    if tuple(latent_before_step.shape) != tuple(hidden_states.shape):
        raise ValueError(
            f"rollout latent_before_step shape must match hidden_states for tuple_id={tuple_id}"
        )
    if tuple(latent_after_step.shape) != tuple(hidden_states.shape):
        raise ValueError(
            f"rollout latent_after_step shape must match hidden_states for tuple_id={tuple_id}"
        )
    scheduler_state = _metadata_mapping(metadata, "scheduler_state", tuple_id)
    rollout_provenance = _metadata_mapping(metadata, "rollout_provenance", tuple_id)
    guidance_scale = _metadata_float(metadata, "guidance_scale", tuple_id)
    reference_image_path = _metadata_string(metadata, "reference_image_path", tuple_id)

    rollout_payload["latent_before_step"] = latent_before_step.detach().cpu()
    rollout_payload["latent_after_step"] = latent_after_step.detach().cpu()
    rollout_payload["scheduler_state"] = dict(scheduler_state)
    rollout_payload["rollout_tuple_schema"] = ROLLOUT_TUPLE_SCHEMA_VERSION
    rollout_payload["rollout_provenance"] = dict(rollout_provenance)
    rollout_payload["guidance_scale"] = guidance_scale
    rollout_payload["reference_image_path"] = reference_image_path
    return rollout_payload


def load_qwen_image_tuple_sample(
    path: str | Path,
    *,
    entry: Mapping[str, object] | None = None,
) -> QwenImageTupleSample:
    """Load and validate one ordinary Qwen-Image teacher tuple."""
    tuple_path = Path(path)
    if not tuple_path.exists():
        raise ValueError(f"tuple payload does not exist: {tuple_path}")
    payload = torch.load(tuple_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"teacher tuple payload must be a mapping: {tuple_path}")
    _reject_layered_payload_fields(payload, tuple_path)
    tuple_entry = dict(entry) if entry is not None else _entry_from_payload(payload, tuple_path)
    validate_teacher_tuple_payload(payload, entry=tuple_entry)

    return QwenImageTupleSample(
        path=tuple_path,
        entry=tuple_entry,
        target_output=_expect_tensor(payload, TARGET_OUTPUT_KEY, tuple_path),
        hidden_states=_expect_tensor(payload, "hidden_states", tuple_path),
        latent_before_step=_optional_tensor(payload, "latent_before_step", tuple_path),
        latent_after_step=_optional_tensor(payload, "latent_after_step", tuple_path),
        scheduler_state=_optional_mapping_from_payload_or_entry(
            payload,
            tuple_entry,
            "scheduler_state",
            tuple_path,
        ),
        rollout_tuple_schema=_optional_string_from_payload_or_entry(
            payload,
            tuple_entry,
            "rollout_tuple_schema",
            tuple_path,
        ),
        rollout_provenance=_optional_mapping_from_payload_or_entry(
            payload,
            tuple_entry,
            "rollout_provenance",
            tuple_path,
        ),
        guidance_scale=_optional_float_from_payload_or_entry(
            payload,
            tuple_entry,
            "guidance_scale",
            tuple_path,
        ),
        reference_image_path=_optional_string_from_payload_or_entry(
            payload,
            tuple_entry,
            "reference_image_path",
            tuple_path,
        ),
        timestep=_expect_tensor(payload, "timestep", tuple_path),
        additional_t_cond=_optional_tensor(payload, "additional_t_cond", tuple_path),
        encoder_hidden_states=_expect_tensor(payload, "encoder_hidden_states", tuple_path),
        encoder_hidden_states_mask=_optional_tensor(
            payload,
            "encoder_hidden_states_mask",
            tuple_path,
        ),
        img_shapes=_expect_field(payload, "img_shapes", tuple_path),
        txt_seq_lens=_expect_field(payload, "txt_seq_lens", tuple_path),
        prompt_id=str(tuple_entry["prompt_id"]),
        split=str(tuple_entry["split"]),
        timestep_index=int(tuple_entry["timestep_index"]),
        timestep_bin=str(tuple_entry["timestep_bin"]),
        cfg_branch=str(tuple_entry["cfg_branch"]),
    )


def forward_qwen_image_tuple(
    transformer: nn.Module,
    sample: QwenImageTupleSample,
    device: torch.device | str,
) -> torch.Tensor:
    """Replay one captured tuple through ``QwenImageTransformer2DModel.forward``."""
    return _forward_qwen_image_tuple_with_hidden_states(
        transformer,
        sample,
        device,
        hidden_states=sample.hidden_states,
    )


def _forward_qwen_image_tuple_with_hidden_states(
    transformer: nn.Module,
    sample: QwenImageTupleSample,
    device: torch.device | str,
    *,
    hidden_states: torch.Tensor,
    activation_checkpoint: bool = False,
) -> torch.Tensor:
    """Replay one tuple while overriding the latent/hidden state."""
    target_device = torch.device(device)
    forward_kwargs = {
        "hidden_states": hidden_states.to(device=target_device),
        "encoder_hidden_states": sample.encoder_hidden_states.to(device=target_device),
        "encoder_hidden_states_mask": _move_value(sample.encoder_hidden_states_mask, target_device),
        "timestep": sample.timestep.to(device=target_device),
        "img_shapes": _move_value(sample.img_shapes, target_device),
        "txt_seq_lens": _move_value(sample.txt_seq_lens, target_device),
        "return_dict": False,
    }
    if sample.additional_t_cond is not None:
        forward_kwargs["additional_t_cond"] = sample.additional_t_cond.to(device=target_device)
    if activation_checkpoint:

        def _forward(checkpoint_hidden_states: torch.Tensor) -> torch.Tensor:
            checkpoint_kwargs = dict(forward_kwargs)
            checkpoint_kwargs["hidden_states"] = checkpoint_hidden_states
            return first_tensor_output(transformer(**checkpoint_kwargs))

        return torch_checkpoint(
            _forward,
            forward_kwargs["hidden_states"],
            use_reentrant=False,
        )
    output = transformer(**forward_kwargs)
    return first_tensor_output(output)


def first_tensor_output(output: object) -> torch.Tensor:
    """Return the first tensor from supported transformer output containers."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    sample = getattr(output, "sample", None)
    if isinstance(sample, torch.Tensor):
        return sample
    raise ValueError("transformer output does not contain a tensor")


def compute_qwen_image_tuple_loss(
    output: torch.Tensor,
    sample: QwenImageTupleSample,
    config: QwenImageTupleLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute weighted MSE/direction teacher distillation loss."""
    loss_config = config or QwenImageTupleLossConfig()
    if loss_config.lambda_mse < 0.0 or loss_config.lambda_dir < 0.0:
        raise ValueError("loss weights must be non-negative")
    if loss_config.lambda_mse == 0.0 and loss_config.lambda_dir == 0.0:
        raise ValueError("at least one loss weight must be positive")
    target = sample.target_output.to(device=output.device, dtype=output.dtype)
    if output.shape != target.shape:
        raise ValueError(
            "transformer output and target_output shapes must match, got "
            f"{tuple(output.shape)} vs {tuple(target.shape)}"
        )

    components: dict[str, torch.Tensor] = {}
    total = output.float().new_zeros(())
    if loss_config.lambda_mse > 0.0:
        mse = F.mse_loss(output.float(), target.float())
        components["mse"] = mse
        total = total + mse * float(loss_config.lambda_mse)
    if loss_config.lambda_dir > 0.0:
        direction = (
            1.0
            - F.cosine_similarity(
                output.float().reshape(1, -1),
                target.float().reshape(1, -1),
                dim=1,
                eps=1.0e-8,
            ).mean()
        )
        components["direction"] = direction
        total = total + direction * float(loss_config.lambda_dir)

    timestep_weight = _timestep_weight(sample, loss_config)
    components["unweighted_total"] = total
    components["timestep_weight"] = output.float().new_tensor(timestep_weight)
    return total * timestep_weight, components


def validate_qwen_image_rollout_tuple_sample(sample: QwenImageTupleSample) -> None:
    """Validate one sample has the latent fields required for rollout training."""
    if sample.rollout_tuple_schema != ROLLOUT_TUPLE_SCHEMA_VERSION:
        raise ValueError(
            f"rollout tuple schema must be {ROLLOUT_TUPLE_SCHEMA_VERSION}: {sample.path}"
        )
    if sample.latent_before_step is None:
        raise ValueError(f"rollout tuple missing latent_before_step: {sample.path}")
    if sample.latent_after_step is None:
        raise ValueError(f"rollout tuple missing latent_after_step: {sample.path}")
    if sample.rollout_provenance is None:
        raise ValueError(f"rollout tuple missing rollout_provenance: {sample.path}")
    _validate_rollout_provenance(sample)
    if sample.scheduler_state is None and not _has_equivalent_scheduler_provenance(sample):
        raise ValueError(f"rollout tuple missing scheduler_state provenance: {sample.path}")
    if sample.guidance_scale is None or sample.guidance_scale <= 0.0:
        raise ValueError(f"rollout tuple missing positive guidance_scale: {sample.path}")
    if sample.reference_image_path is None:
        raise ValueError(f"rollout tuple missing reference_image_path: {sample.path}")
    if tuple(sample.latent_before_step.shape) != tuple(sample.hidden_states.shape):
        raise ValueError(
            "rollout tuple latent_before_step shape must match hidden_states, got "
            f"{tuple(sample.latent_before_step.shape)} vs {tuple(sample.hidden_states.shape)}: "
            f"{sample.path}"
        )
    if tuple(sample.latent_after_step.shape) != tuple(sample.hidden_states.shape):
        raise ValueError(
            "rollout tuple latent_after_step shape must match hidden_states, got "
            f"{tuple(sample.latent_after_step.shape)} vs {tuple(sample.hidden_states.shape)}: "
            f"{sample.path}"
        )


def compute_qwen_image_rollout_loss(
    student_transformer: nn.Module,
    samples: Sequence[QwenImageTupleSample],
    *,
    scheduler_step_fn: Callable[
        [torch.Tensor, torch.Tensor, QwenImageRolloutStepSamples],
        torch.Tensor,
    ]
    | None = None,
    device: torch.device | str,
    config: QwenImageRolloutLossConfig | None = None,
    teacher_transformer: nn.Module | None = None,
    teacher_forward_fn: Callable[[torch.Tensor, QwenImageTupleSample], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[str, object]]]:
    """Compute K-step latent rollout loss with differentiable student trajectory."""
    loss_config = config or QwenImageRolloutLossConfig()
    _validate_rollout_loss_config(loss_config)
    if (teacher_transformer is None) == (teacher_forward_fn is None):
        raise ValueError("exactly one of teacher_transformer or teacher_forward_fn must be set")
    scheduler_step = scheduler_step_fn or qwen_image_rollout_scheduler_step
    window = _build_rollout_window(samples, loss_config.rollout_k)
    target_device = torch.device(device)

    first_latent = window[0].cond.latent_before_step
    if first_latent is None:
        raise ValueError(f"rollout tuple missing latent_before_step: {window[0].cond.path}")
    current_latent = first_latent.to(device=target_device)
    total = current_latent.float().new_zeros(())
    aggregate: dict[str, torch.Tensor] = {
        "output_mse": total.clone(),
        "cond_output_mse": total.clone(),
        "negative_output_mse": total.clone(),
        "direction": total.clone(),
        "cond_direction": total.clone(),
        "negative_direction": total.clone(),
        "latent_mse": total.clone(),
        "anchor_mse": total.clone(),
        "unweighted_total": total.clone(),
        "timestep_weighted_total": total.clone(),
    }
    records: list[dict[str, object]] = []

    for rollout_index, step in enumerate(window):
        cond_student_output = _forward_qwen_image_tuple_with_hidden_states(
            student_transformer,
            step.cond,
            target_device,
            hidden_states=current_latent,
            activation_checkpoint=loss_config.student_activation_checkpoint,
        )
        negative_student_output = _forward_qwen_image_tuple_with_hidden_states(
            student_transformer,
            step.negative,
            target_device,
            hidden_states=current_latent,
            activation_checkpoint=loss_config.student_activation_checkpoint,
        )
        teacher_input = current_latent.detach()
        with torch.no_grad():
            cond_teacher_output = _forward_rollout_teacher_branch(
                teacher_input,
                step.cond,
                target_device,
                teacher_transformer=teacher_transformer,
                teacher_forward_fn=teacher_forward_fn,
            )
            negative_teacher_output = _forward_rollout_teacher_branch(
                teacher_input,
                step.negative,
                target_device,
                teacher_transformer=teacher_transformer,
                teacher_forward_fn=teacher_forward_fn,
            )

        cond_teacher_output = cond_teacher_output.to(
            device=cond_student_output.device,
            dtype=cond_student_output.dtype,
        )
        negative_teacher_output = negative_teacher_output.to(
            device=negative_student_output.device,
            dtype=negative_student_output.dtype,
        )
        guided_student_output = combine_qwen_image_cfg_outputs(
            cond_student_output,
            negative_student_output,
            guidance_scale=step.guidance_scale,
            normalize=loss_config.cfg_normalize,
        )
        guided_teacher_output = combine_qwen_image_cfg_outputs(
            cond_teacher_output,
            negative_teacher_output,
            guidance_scale=step.guidance_scale,
            normalize=loss_config.cfg_normalize,
        )
        student_next_latent = scheduler_step(current_latent, guided_student_output, step)
        with torch.no_grad():
            teacher_next_latent = scheduler_step(teacher_input, guided_teacher_output, step)
        step_loss, step_components = compute_qwen_image_rollout_step_loss(
            cond_student_output=cond_student_output,
            cond_teacher_output=cond_teacher_output,
            negative_student_output=negative_student_output,
            negative_teacher_output=negative_teacher_output,
            student_next_latent=student_next_latent,
            teacher_next_latent=teacher_next_latent,
            teacher_reference_next_latent=step.cond.latent_after_step,
            step=step,
            config=loss_config,
        )
        total = total + step_loss
        for name in aggregate:
            aggregate[name] = aggregate[name] + step_components[name]
        records.append(
            _build_rollout_step_record(
                rollout_index=rollout_index,
                step=step,
                components=step_components,
            )
        )
        current_latent = student_next_latent

    aggregate["loss_total"] = total
    aggregate["rollout_steps"] = total.new_tensor(float(len(window)))
    return total, aggregate, records


def compute_qwen_image_rollout_step_loss(
    *,
    cond_student_output: torch.Tensor,
    cond_teacher_output: torch.Tensor,
    negative_student_output: torch.Tensor,
    negative_teacher_output: torch.Tensor,
    student_next_latent: torch.Tensor,
    teacher_next_latent: torch.Tensor,
    teacher_reference_next_latent: torch.Tensor | None,
    step: QwenImageRolloutStepSamples,
    config: QwenImageRolloutLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute one rollout step's output, direction, latent, and anchor losses."""
    if teacher_reference_next_latent is None:
        raise ValueError(f"rollout tuple missing latent_after_step: {step.cond.path}")
    _validate_output_shape_pair(cond_student_output, cond_teacher_output, "cond")
    _validate_output_shape_pair(negative_student_output, negative_teacher_output, "negative")
    if tuple(student_next_latent.shape) != tuple(teacher_next_latent.shape):
        raise ValueError(
            "student and teacher next latent shapes must match, got "
            f"{tuple(student_next_latent.shape)} vs {tuple(teacher_next_latent.shape)}"
        )
    anchor = teacher_reference_next_latent.to(
        device=student_next_latent.device,
        dtype=student_next_latent.dtype,
    )
    if tuple(anchor.shape) != tuple(student_next_latent.shape):
        raise ValueError(
            "student next latent and teacher reference latent shapes must match, got "
            f"{tuple(student_next_latent.shape)} vs {tuple(anchor.shape)}"
        )

    cond_output_mse = F.mse_loss(cond_student_output.float(), cond_teacher_output.float())
    negative_output_mse = F.mse_loss(
        negative_student_output.float(),
        negative_teacher_output.float(),
    )
    output_mse = (cond_output_mse + negative_output_mse) * 0.5
    cond_direction = _tensor_direction_loss(cond_student_output, cond_teacher_output)
    negative_direction = _tensor_direction_loss(negative_student_output, negative_teacher_output)
    direction = (cond_direction + negative_direction) * 0.5
    latent_mse = F.mse_loss(student_next_latent.float(), teacher_next_latent.float())
    anchor_mse = F.mse_loss(student_next_latent.float(), anchor.float())
    unweighted = (
        output_mse * float(config.lambda_output_mse)
        + direction * float(config.lambda_dir)
        + latent_mse * float(config.lambda_latent)
        + anchor_mse * float(config.lambda_anchor)
    )
    timestep_weight = _rollout_timestep_weight(step.cond, config)
    weighted = unweighted * timestep_weight
    components = {
        "output_mse": output_mse,
        "cond_output_mse": cond_output_mse,
        "negative_output_mse": negative_output_mse,
        "direction": direction,
        "cond_direction": cond_direction,
        "negative_direction": negative_direction,
        "latent_mse": latent_mse,
        "anchor_mse": anchor_mse,
        "unweighted_total": unweighted,
        "timestep_weight": cond_student_output.float().new_tensor(timestep_weight),
        "timestep_weighted_total": weighted,
    }
    return weighted, components


def combine_qwen_image_cfg_outputs(
    cond_output: torch.Tensor,
    negative_output: torch.Tensor,
    *,
    guidance_scale: float,
    normalize: bool = True,
) -> torch.Tensor:
    """Combine Qwen-Image true-CFG branch outputs."""
    if tuple(cond_output.shape) != tuple(negative_output.shape):
        raise ValueError(
            "cond and negative CFG outputs must have matching shapes, got "
            f"{tuple(cond_output.shape)} vs {tuple(negative_output.shape)}"
        )
    guided = negative_output + float(guidance_scale) * (cond_output - negative_output)
    if not normalize:
        return guided
    cond_norm = torch.norm(cond_output, dim=-1, keepdim=True)
    guided_norm = torch.norm(guided, dim=-1, keepdim=True).clamp_min(1.0e-8)
    return guided * (cond_norm / guided_norm)


def qwen_image_rollout_scheduler_step(
    latent: torch.Tensor,
    guided_output: torch.Tensor,
    step: QwenImageRolloutStepSamples,
) -> torch.Tensor:
    """Side-effect-free FlowMatch/Euler step from captured Qwen-Image scheduler metadata."""
    scheduler_state = step.cond.scheduler_state
    if not isinstance(scheduler_state, Mapping):
        raise ValueError("rollout scheduler_state must be a mapping")
    _validate_rollout_scheduler_step_contract(scheduler_state, step)
    current_sigma = _scheduler_state_float(
        scheduler_state,
        step,
        ("sigma", "current_sigma"),
    )
    sigma_delta = _scheduler_state_optional_float(
        scheduler_state,
        ("sigma_delta", "dt"),
    )
    if sigma_delta is None:
        next_sigma = _scheduler_state_optional_float(
            scheduler_state,
            ("next_sigma", "sigma_next"),
        )
        if next_sigma is None:
            raise ValueError(
                "rollout scheduler_state must include next_sigma/sigma_next or sigma_delta/dt"
            )
        sigma_delta = next_sigma - current_sigma
    delta = guided_output.new_tensor(float(sigma_delta))
    next_latent = latent + guided_output.to(dtype=latent.dtype) * delta.to(dtype=latent.dtype)
    return next_latent.to(dtype=latent.dtype)


def monitor_qwen_image_rollout_no_grad(
    student_transformer: nn.Module,
    samples: Sequence[QwenImageTupleSample],
    *,
    scheduler_step_fn: Callable[
        [torch.Tensor, torch.Tensor, QwenImageRolloutStepSamples],
        torch.Tensor,
    ]
    | None = None,
    output_json: str | Path,
    device: torch.device | str,
    config: QwenImageRolloutLossConfig | None = None,
    teacher_transformer: nn.Module | None = None,
    teacher_forward_fn: Callable[[torch.Tensor, QwenImageTupleSample], torch.Tensor] | None = None,
    records_jsonl: str | Path | None = None,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run a no-grad rollout monitor and write JSON summary artifacts."""
    output_path = Path(output_json)
    records_path = Path(records_jsonl) if records_jsonl is not None else None
    if records_path is not None:
        records_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        loss, components, records = compute_qwen_image_rollout_loss(
            student_transformer,
            samples,
            scheduler_step_fn=scheduler_step_fn,
            device=device,
            config=config,
            teacher_transformer=teacher_transformer,
            teacher_forward_fn=teacher_forward_fn,
        )
    if records_path is not None:
        records_path.write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
            encoding="utf-8",
        )
    summary = {
        "format": ROLLOUT_MONITOR_SUMMARY_FORMAT,
        "output_json": str(output_path),
        "records_jsonl": str(records_path) if records_path is not None else None,
        "rollout_config": _rollout_loss_config_to_dict(config or QwenImageRolloutLossConfig()),
        "sample_count": len(samples),
        "metrics_summary": {
            "loss_total": _tensor_scalar(loss),
            "rollout_steps": _tensor_scalar(components["rollout_steps"]),
            "output_mse_sum": _tensor_scalar(components["output_mse"]),
            "cond_output_mse_sum": _tensor_scalar(components["cond_output_mse"]),
            "negative_output_mse_sum": _tensor_scalar(components["negative_output_mse"]),
            "direction_sum": _tensor_scalar(components["direction"]),
            "cond_direction_sum": _tensor_scalar(components["cond_direction"]),
            "negative_direction_sum": _tensor_scalar(components["negative_direction"]),
            "latent_mse_sum": _tensor_scalar(components["latent_mse"]),
            "anchor_mse_sum": _tensor_scalar(components["anchor_mse"]),
        },
        "provenance": dict(provenance or {}),
    }
    write_json(output_path, summary)
    return summary


def prepare_qwen_image_qat_model(
    transformer: nn.Module,
    *,
    target_layers: tuple[str, ...] = (QWEN_BLOCK_LINEAR_TARGET,),
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    expected_num_layers: int | None = None,
    expected_target_count: int | None = None,
    block_size: int = MXFP8_BLOCK_SIZE,
    scale_mode: Mxfp8ScaleMode = "fp32_block_scale",
    scale_multipliers: Mapping[str, float] | None = None,
    linear_cls: type[nn.Module] | None = None,
) -> tuple[QwenImageQatInjectionInfo, ...]:
    """Freeze base weights and inject LoRA fake-MXFP8 wrappers into Qwen block Linears."""
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)

    if expected_num_layers is not None and expected_target_count is not None:
        expected_from_layers = int(expected_num_layers) * int(QWEN_IMAGE_BLOCK_LINEAR_COUNT)
        if int(expected_target_count) != expected_from_layers:
            raise ValueError(
                "expected_target_count must equal expected_num_layers * "
                f"{QWEN_IMAGE_BLOCK_LINEAR_COUNT}, got {expected_target_count} vs "
                f"{expected_from_layers}"
            )
    targets = select_qwen_image_block_linears(
        transformer,
        expected_num_layers=None if expected_target_count is not None else expected_num_layers,
        expected_count=expected_target_count,
        linear_cls=linear_cls,
    )
    selected_targets = _filter_targets(targets, target_layers)
    if not selected_targets:
        raise ValueError(f"target_layers selected no Qwen block Linears: {target_layers}")
    _assert_qat_target_weights_are_floating(selected_targets)

    injections: list[QwenImageQatInjectionInfo] = []
    for target in selected_targets:
        scale_multiplier = _scale_multiplier_for_target(target, scale_multipliers)
        fake = FakeMxfp8Linear(
            target.module,
            module_name=target.normalized_name,
            block_size=block_size,
            scale_mode=scale_mode,
        )
        replacement = Mxfp8LoraAdapterLinear(
            fake,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            scale_multiplier=scale_multiplier,
        )
        _replace_module(transformer, target.module_name, replacement)
        injections.append(
            QwenImageQatInjectionInfo(
                module_name=target.normalized_name,
                block_index=target.block_index,
                role=target.role,
                trainable_parameter_names=("lora_down.weight", "lora_up.weight"),
                lora_rank=int(lora_rank),
                lora_alpha=float(lora_alpha),
                scale_multiplier=scale_multiplier,
            )
        )

    return tuple(injections)


def train_qwen_image_qat(
    transformer: nn.Module,
    config: QwenImageQatTrainingConfig,
    *,
    linear_cls: type[nn.Module] | None = None,
) -> QwenImageQatTrainingResult:
    """Train LoRA fake-MXFP8 adapters against captured BF16 teacher tuples."""
    _validate_training_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = QwenImageTupleDataset(config.tuple_index_jsonl)
    device = torch.device(config.device)
    transformer.to(device=device)
    transformer.train()
    injections = prepare_qwen_image_qat_model(
        transformer,
        target_layers=config.target_layers,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        expected_num_layers=config.expected_num_layers,
        expected_target_count=config.expected_target_count,
        scale_multipliers=config.scale_multipliers,
        linear_cls=linear_cls,
    )
    trainable_parameters = [
        parameter for parameter in transformer.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("QAT model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        betas=config.betas,
        eps=float(config.eps),
        foreach=config.optimizer_foreach,
    )

    metrics_path = output_dir / config.metrics_name
    checkpoint_path = output_dir / config.checkpoint_name
    start_time = time.perf_counter()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, config.max_steps + 1):
            warmup_active = _apply_progressive_warmup(
                transformer,
                step=step,
                warmup_steps=config.warmup_steps,
                warmup_target_layers=config.warmup_target_layers,
            )
            sample_index = _sample_index_for_step(
                step,
                dataset_size=len(dataset),
                sample_stride=config.sample_stride,
                sample_start_index=config.sample_start_index,
            )
            sample = dataset[sample_index]
            optimizer.zero_grad(set_to_none=True)
            output = forward_qwen_image_tuple(transformer, sample, device)
            loss, components = compute_qwen_image_tuple_loss(output, sample, config.loss)
            loss.backward()
            grad_norm = _parameter_global_norm(
                trainable_parameters,
                grad=True,
            )
            optimizer.step()
            if step % config.log_interval_steps == 0 or step == config.max_steps:
                record = _build_training_record(
                    step=step,
                    sample_index=sample_index,
                    sample=sample,
                    loss=loss,
                    components=components,
                    grad_norm=grad_norm,
                    trainable_parameters=trainable_parameters,
                    transformer=transformer,
                    elapsed_seconds=time.perf_counter() - start_time,
                    compute_lora_delta_norm=config.compute_lora_delta_norm,
                    warmup_active=warmup_active,
                )
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_file.flush()
    _save_qwen_image_qat_checkpoint(
        checkpoint_path,
        transformer=transformer,
        config=config,
        injections=injections,
        train_steps=config.max_steps,
    )
    return QwenImageQatTrainingResult(
        output_dir=output_dir,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        train_steps=config.max_steps,
        injections=injections,
    )


def train_qwen_image_rollout_qat(
    student_transformer: nn.Module,
    config: QwenImageRolloutQatTrainingConfig,
    *,
    scheduler_step_fn: Callable[
        [torch.Tensor, torch.Tensor, QwenImageRolloutStepSamples],
        torch.Tensor,
    ]
    | None = None,
    teacher_transformer: nn.Module | None = None,
    teacher_forward_fn: Callable[[torch.Tensor, QwenImageTupleSample], torch.Tensor] | None = None,
    linear_cls: type[nn.Module] | None = None,
) -> QwenImageRolloutQatTrainingResult:
    """Train LoRA fake-MXFP8 adapters with closed-set K-step latent rollout loss."""
    _validate_rollout_training_config(config)
    if (teacher_transformer is None) == (teacher_forward_fn is None):
        raise ValueError("exactly one of teacher_transformer or teacher_forward_fn must be set")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = QwenImageRolloutWindowDataset(
        config.tuple_index_jsonl,
        rollout_k=config.loss.rollout_k,
    )
    device = torch.device(config.device)
    student_transformer.to(device=device)
    student_transformer.train()
    if teacher_transformer is not None:
        teacher_transformer.to(device=device)
        teacher_transformer.eval()
        for parameter in teacher_transformer.parameters():
            parameter.requires_grad_(False)
    injections = prepare_qwen_image_qat_model(
        student_transformer,
        target_layers=config.target_layers,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        expected_num_layers=config.expected_num_layers,
        expected_target_count=config.expected_target_count,
        scale_multipliers=config.scale_multipliers,
        linear_cls=linear_cls,
    )
    trainable_parameters = [
        parameter for parameter in student_transformer.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("QAT model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        betas=config.betas,
        eps=float(config.eps),
        foreach=config.optimizer_foreach,
    )

    metrics_path = output_dir / config.metrics_name
    checkpoint_path = output_dir / config.checkpoint_name
    start_time = time.perf_counter()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, config.max_steps + 1):
            window_index = _sample_index_for_step(
                step,
                dataset_size=len(dataset),
                sample_stride=config.window_stride,
                sample_start_index=config.window_start_index,
            )
            window = dataset[window_index]
            optimizer.zero_grad(set_to_none=True)
            loss, components, rollout_records = compute_qwen_image_rollout_loss(
                student_transformer,
                window,
                scheduler_step_fn=scheduler_step_fn,
                device=device,
                config=config.loss,
                teacher_transformer=teacher_transformer,
                teacher_forward_fn=teacher_forward_fn,
            )
            loss.backward()
            grad_norm = _parameter_global_norm(
                trainable_parameters,
                grad=True,
            )
            if config.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=float(config.grad_clip_norm),
                )
            optimizer.step()
            if step % config.log_interval_steps == 0 or step == config.max_steps:
                record = _build_rollout_training_record(
                    step=step,
                    window_index=window_index,
                    window=window,
                    loss=loss,
                    components=components,
                    rollout_records=rollout_records,
                    grad_norm=grad_norm,
                    trainable_parameters=trainable_parameters,
                    transformer=student_transformer,
                    elapsed_seconds=time.perf_counter() - start_time,
                    compute_lora_delta_norm=config.compute_lora_delta_norm,
                )
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_file.flush()
            if (
                config.checkpoint_interval_steps is not None
                and step % config.checkpoint_interval_steps == 0
            ):
                _save_qwen_image_qat_checkpoint_from_config(
                    _rollout_interval_checkpoint_path(
                        output_dir=output_dir,
                        checkpoint_name=config.checkpoint_name,
                        step=step,
                    ),
                    transformer=student_transformer,
                    config=_rollout_training_config_to_dict(config),
                    injections=injections,
                    train_steps=step,
                )

    _save_qwen_image_qat_checkpoint_from_config(
        checkpoint_path,
        transformer=student_transformer,
        config=_rollout_training_config_to_dict(config),
        injections=injections,
        train_steps=config.max_steps,
    )
    return QwenImageRolloutQatTrainingResult(
        output_dir=output_dir,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        train_steps=config.max_steps,
        injections=injections,
        rollout_window_count=len(dataset),
    )


def build_qwen_image_qat_probe_config(
    *,
    recipe: str,
    tuple_index_jsonl: str | Path,
    output_dir: str | Path,
    max_steps: int,
    learning_rate: float = 1.0e-5,
    weight_decay: float = 0.0,
    device: str = "cuda:0",
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0,
    expected_num_layers: int | None = 60,
    expected_target_count: int | None = 840,
    log_interval_steps: int = 10,
    compute_lora_delta_norm: bool = True,
    scale_multipliers: Mapping[str, float] | None = None,
    checkpoint_name: str = "qwen_image_qat_lora_last.pt",
    metrics_name: str = "qwen_image_qat_train_metrics.jsonl",
    sample_stride: int = 17,
    sample_start_index: int = 0,
) -> QwenImageQatTrainingConfig:
    """Build the literature-guided 32-prompt probe training config."""
    recipe_name = validate_qwen_image_qat_probe_recipe(recipe)
    return QwenImageQatTrainingConfig(
        tuple_index_jsonl=tuple_index_jsonl,
        output_dir=output_dir,
        max_steps=max_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        expected_num_layers=expected_num_layers,
        expected_target_count=expected_target_count,
        loss=_probe_loss_config(recipe_name),
        log_interval_steps=log_interval_steps,
        checkpoint_name=checkpoint_name,
        metrics_name=metrics_name,
        compute_lora_delta_norm=compute_lora_delta_norm,
        scale_multipliers=scale_multipliers if recipe_name == "scale_aware_lora" else None,
        optimizer_foreach=False,
        sample_stride=sample_stride,
        sample_start_index=sample_start_index,
    )


def validate_qwen_image_qat_probe_recipe(recipe: str) -> str:
    """Return a supported probe recipe name or raise a precise error."""
    if recipe not in QWEN_IMAGE_QAT_PROBE_RECIPES:
        raise ValueError(
            "unsupported Qwen-Image QAT probe recipe "
            f"{recipe!r}; expected one of {QWEN_IMAGE_QAT_PROBE_RECIPES}"
        )
    return recipe


def load_qwen_image_rollout_qat_config(
    path: str | Path,
) -> QwenImageRolloutQatTrainingConfig:
    """Load a closed-set rollout QAT training config from JSON or YAML."""
    config_path = Path(path)
    data = _read_mapping_config(config_path)
    loss_data = data.get("loss", {})
    if loss_data is not None and not isinstance(loss_data, Mapping):
        raise ValueError(f"rollout QAT config loss must be a mapping: {config_path}")
    betas = _optimizer_betas_from_config(data.get("betas", (0.9, 0.999)), config_path)
    loss = QwenImageRolloutLossConfig(
        rollout_k=int(loss_data.get("rollout_k", 4)),
        lambda_output_mse=float(loss_data.get("lambda_output_mse", 1.0)),
        lambda_dir=float(loss_data.get("lambda_dir", 0.1)),
        lambda_latent=float(loss_data.get("lambda_latent", 1.0)),
        lambda_anchor=float(loss_data.get("lambda_anchor", 0.2)),
        timestep_weights=dict(
            loss_data.get("timestep_weights", CLOSED_SET_ROLLOUT_TIMESTEP_WEIGHTS)
        ),
        teacher_no_grad=bool(loss_data.get("teacher_no_grad", True)),
        student_latent_detach=bool(loss_data.get("student_latent_detach", False)),
        cfg_normalize=bool(loss_data.get("cfg_normalize", True)),
        student_activation_checkpoint=bool(loss_data.get("student_activation_checkpoint", False)),
    )
    config = QwenImageRolloutQatTrainingConfig(
        tuple_index_jsonl=_required_config_value(data, "tuple_index_jsonl", config_path),
        output_dir=_required_config_value(data, "output_dir", config_path),
        max_steps=int(_required_config_value(data, "max_steps", config_path)),
        learning_rate=float(data.get("learning_rate", 2.0e-5)),
        weight_decay=float(data.get("weight_decay", 0.0)),
        betas=betas,
        eps=float(data.get("eps", 1.0e-8)),
        grad_clip_norm=float(data.get("grad_clip_norm", 1.0)),
        device=str(data.get("device", "cuda")),
        target_layers=tuple(data.get("target_layers", (QWEN_BLOCK_LINEAR_TARGET,))),
        lora_rank=int(data.get("lora_rank", 64)),
        lora_alpha=float(data.get("lora_alpha", 128.0)),
        lora_dropout=float(data.get("lora_dropout", 0.0)),
        expected_num_layers=int(
            data.get("expected_num_layers", FORMAL_QWEN_IMAGE_BLOCK_LAYER_COUNT)
        ),
        expected_target_count=int(
            data.get(
                "expected_target_count",
                FORMAL_QWEN_IMAGE_BLOCK_LINEAR_TARGET_COUNT,
            )
        ),
        diagnostic_only=bool(data.get("diagnostic_only", False)),
        teacher_target_mode=str(data.get("teacher_target_mode", "on_policy")),
        loss=loss,
        log_interval_steps=int(data.get("log_interval_steps", 1)),
        checkpoint_name=str(
            data.get("checkpoint_name", "qwen_image_closed_set_rollout_qat_lora_last.pt")
        ),
        checkpoint_interval_steps=(
            int(data["checkpoint_interval_steps"])
            if data.get("checkpoint_interval_steps") is not None
            else None
        ),
        metrics_name=str(
            data.get("metrics_name", "qwen_image_closed_set_rollout_qat_train_metrics.jsonl")
        ),
        compute_lora_delta_norm=bool(data.get("compute_lora_delta_norm", False)),
        scale_multipliers=_optional_mapping_config_value(data, "scale_multipliers"),
        optimizer_foreach=data.get("optimizer_foreach", False),
        window_stride=int(data.get("window_stride", 1)),
        window_start_index=int(data.get("window_start_index", 0)),
    )
    _validate_rollout_training_config(config)
    return config


def run_qwen_image_rollout_tuple_augmentation(
    *,
    source_tuple_index_jsonl: str | Path,
    rollout_metadata_jsonl: str | Path,
    output_tuple_root: str | Path,
    output_tuple_index_jsonl: str | Path,
    summary_json: str | Path,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run durable rollout tuple augmentation from a JSONL metadata manifest."""
    validated_provenance = _validate_rollout_command_provenance(
        provenance,
        command_name="augment-rollout-tuples",
    )
    metadata = load_qwen_image_rollout_metadata_manifest(rollout_metadata_jsonl)
    augmentation_config = QwenImageRolloutTupleAugmentationConfig(
        source_tuple_index_jsonl=source_tuple_index_jsonl,
        output_tuple_index_jsonl=output_tuple_index_jsonl,
        output_tuple_root=output_tuple_root,
        metadata_by_tuple_id=metadata,
        summary_json=None,
        validate_prompt_continuity=True,
        provenance={
            **validated_provenance,
            "rollout_metadata_jsonl": str(rollout_metadata_jsonl),
        },
    )
    output_entries = augment_qwen_image_rollout_tuple_dataset(augmentation_config)
    summary = _build_rollout_augmentation_summary(
        config=QwenImageRolloutTupleAugmentationConfig(
            source_tuple_index_jsonl=source_tuple_index_jsonl,
            output_tuple_index_jsonl=output_tuple_index_jsonl,
            output_tuple_root=output_tuple_root,
            metadata_by_tuple_id=metadata,
            summary_json=summary_json,
            validate_prompt_continuity=True,
            provenance=augmentation_config.provenance,
        ),
        output_entries=output_entries,
        source_index_path=Path(source_tuple_index_jsonl),
        output_index_path=Path(output_tuple_index_jsonl),
        output_tuple_root=Path(output_tuple_root),
    )
    summary["rollout_metadata_jsonl"] = str(rollout_metadata_jsonl)
    write_json(Path(summary_json), summary)
    return summary


def run_qwen_image_rollout_qat(
    *,
    config_path: str | Path,
    model: str,
    visual_gen_args: str | Path,
    teacher_model: str | None = None,
    teacher_visual_gen_args: str | Path | None = None,
    summary_json: str | Path | None = None,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Load Qwen-Image student/teacher transformers and run closed-set rollout QAT."""
    validated_provenance = _validate_rollout_command_provenance(
        provenance,
        command_name="run-rollout-qat",
    )
    config = load_qwen_image_rollout_qat_config(config_path)
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    student_pipeline = load_single_worker_pipeline(
        model=model,
        visual_gen_args=Path(visual_gen_args),
        device=config.device,
    )
    teacher_pipeline = None
    try:
        student_transformer = getattr(student_pipeline, "transformer", None)
        if student_transformer is None:
            raise ValueError("Loaded Qwen-Image student pipeline must expose a transformer")
        teacher_transformer = None
        teacher_forward_fn = None
        if config.teacher_target_mode == "on_policy":
            teacher_pipeline = load_single_worker_pipeline(
                model=teacher_model or model,
                visual_gen_args=Path(teacher_visual_gen_args or visual_gen_args),
                device=config.device,
            )
            teacher_transformer = getattr(teacher_pipeline, "transformer", None)
            if teacher_transformer is None:
                raise ValueError("Loaded Qwen-Image teacher pipeline must expose a transformer")
        elif config.teacher_target_mode == "captured_tuple":
            teacher_forward_fn = _captured_rollout_teacher_target
        else:
            raise ValueError(f"unsupported teacher_target_mode: {config.teacher_target_mode}")
        result = train_qwen_image_rollout_qat(
            student_transformer,
            config,
            teacher_transformer=teacher_transformer,
            teacher_forward_fn=teacher_forward_fn,
        )
    finally:
        if teacher_pipeline is not None:
            cleanup_pipeline(teacher_pipeline)
        cleanup_pipeline(student_pipeline)

    summary_path = (
        Path(summary_json) if summary_json is not None else output_path / "rollout_qat_summary.json"
    )
    summary = {
        "format": ROLLOUT_QAT_SUMMARY_FORMAT,
        "recipe": "closed_set_rollout_qat_v1",
        "model": model,
        "teacher_model": teacher_model or model,
        "visual_gen_args": str(visual_gen_args),
        "teacher_visual_gen_args": str(teacher_visual_gen_args or visual_gen_args),
        "config_path": str(config_path),
        "tuple_index_jsonl": str(config.tuple_index_jsonl),
        "output_dir": str(output_path),
        "summary_json": str(summary_path),
        "checkpoint_path": str(result.checkpoint_path),
        "metrics_path": str(result.metrics_path),
        "train_steps": result.train_steps,
        "rollout_window_count": result.rollout_window_count,
        "injection_count": len(result.injections),
        "training_config": _rollout_training_config_to_dict(config),
        "formal_task12c_eligible": not config.diagnostic_only,
        "scheduler_step_implementation": ROLLOUT_SCHEDULER_STEP_IMPLEMENTATION,
        "metrics_summary": summarize_qwen_image_qat_training_metrics(result.metrics_path),
        "provenance": validated_provenance,
    }
    write_json(summary_path, summary)
    return summary


def run_qwen_image_qat_probe(
    *,
    recipe: str,
    model: str,
    visual_gen_args: str | Path,
    tuple_index_jsonl: str | Path,
    output_dir: str | Path,
    max_steps: int,
    device: str = "cuda:0",
    learning_rate: float = 1.0e-5,
    weight_decay: float = 0.0,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0,
    expected_num_layers: int | None = 60,
    expected_target_count: int | None = 840,
    log_interval_steps: int = 10,
    scale_clip_min: float = 0.25,
    scale_clip_max: float = 4.0,
    sample_stride: int = 17,
    sample_start_index: int = 0,
    summary_json: str | Path | None = None,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Load Qwen-Image and run one fake-MXFP8 LoRA QAT probe recipe."""
    recipe_name = validate_qwen_image_qat_probe_recipe(recipe)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    scale_multipliers = _maybe_build_probe_scale_multipliers(
        recipe=recipe_name,
        model=model,
        visual_gen_args=Path(visual_gen_args),
        output_dir=output_path,
        device=device,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        expected_num_layers=expected_num_layers,
        expected_target_count=expected_target_count,
        scale_clip_min=scale_clip_min,
        scale_clip_max=scale_clip_max,
    )
    config = build_qwen_image_qat_probe_config(
        recipe=recipe_name,
        tuple_index_jsonl=tuple_index_jsonl,
        output_dir=output_path,
        max_steps=max_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        expected_num_layers=expected_num_layers,
        expected_target_count=expected_target_count,
        log_interval_steps=log_interval_steps,
        scale_multipliers=scale_multipliers,
        sample_stride=sample_stride,
        sample_start_index=sample_start_index,
    )
    pipeline = load_single_worker_pipeline(
        model=model,
        visual_gen_args=Path(visual_gen_args),
        device=device,
    )
    try:
        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            raise ValueError("Loaded pipeline does not expose a transformer component")
        result = train_qwen_image_qat(transformer, config)
    finally:
        cleanup_pipeline(pipeline)

    summary_path = (
        Path(summary_json) if summary_json is not None else output_path / "probe_summary.json"
    )
    summary = {
        "format": PROBE_SUMMARY_FORMAT,
        "recipe": recipe_name,
        "model": model,
        "visual_gen_args": str(visual_gen_args),
        "tuple_index_jsonl": str(tuple_index_jsonl),
        "output_dir": str(output_path),
        "summary_json": str(summary_path),
        "checkpoint_path": str(result.checkpoint_path),
        "metrics_path": str(result.metrics_path),
        "train_steps": result.train_steps,
        "injection_count": len(result.injections),
        "training_config": _training_config_to_dict(config),
        "metrics_summary": summarize_qwen_image_qat_training_metrics(result.metrics_path),
        "provenance": dict(provenance or {}),
    }
    write_json(summary_path, summary)
    return summary


def run_qwen_image_qat_checkpoint_monitor(
    *,
    model: str,
    visual_gen_args: str | Path,
    checkpoint_path: str | Path,
    tuple_index_jsonl: str | Path,
    output_json: str | Path,
    records_jsonl: str | Path | None = None,
    max_samples: int | None = None,
    sample_stride: int = 1,
    sample_start_index: int = 0,
    device: str = "cuda:0",
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Load Qwen-Image and evaluate one trained fake-MXFP8 LoRA checkpoint on fixed tuples."""
    pipeline = load_single_worker_pipeline(
        model=model,
        visual_gen_args=Path(visual_gen_args),
        device=device,
    )
    try:
        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            raise ValueError("Loaded pipeline does not expose a transformer component")
        return monitor_qwen_image_qat_checkpoint(
            transformer,
            QwenImageQatMonitorConfig(
                checkpoint_path=checkpoint_path,
                tuple_index_jsonl=tuple_index_jsonl,
                output_json=output_json,
                records_jsonl=records_jsonl,
                max_samples=max_samples,
                sample_stride=sample_stride,
                sample_start_index=sample_start_index,
                device=device,
            ),
            model=model,
            visual_gen_args=visual_gen_args,
            provenance=provenance,
        )
    finally:
        cleanup_pipeline(pipeline)


def run_qwen_image_qat_checkpoint_rgb_eval(
    *,
    prompt_manifest_jsonl: str | Path,
    split: str,
    model: str,
    visual_gen_args: str | Path,
    checkpoint_path: str | Path,
    reference_root: str | Path,
    output_root: str | Path,
    metrics_json: str | Path,
    variant: str,
    reference_variant: str = "bf16_sage_fp8",
    device: str = "cuda:0",
    max_prompts: int | None = None,
    provenance: Mapping[str, object] | None = None,
    config_metadata: Mapping[str, object] | None = None,
    records: list[dict[str, object]] | None = None,
    pipeline: Any | None = None,
    infer_fn: Callable[[Any, dict[str, object]], Any] | None = None,
    save_fn: Callable[[Any, Path], None] | None = None,
    metrics_fn: Callable[[Path, Path], dict[str, float]] | None = None,
    linear_cls: type[nn.Module] | None = None,
) -> dict[str, object]:
    """Run probe-only RGB eval with a trained fake-MXFP8 LoRA checkpoint applied."""
    from scripts.visualgen_eval.qwen_image_prompt_manifest import read_jsonl as read_prompt_jsonl
    from scripts.visualgen_eval.qwen_image_rgb_eval import (
        collect_visual_gen_config_metadata,
        compute_rgb_image_metrics,
        infer_record,
        run_qwen_image_rgb_split_eval,
        save_reference_image,
    )

    visual_args_path = Path(visual_gen_args)
    owns_pipeline = pipeline is None
    if pipeline is None:
        pipeline = load_single_worker_pipeline(
            model=model,
            visual_gen_args=visual_args_path,
            device=device,
        )
    try:
        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            raise ValueError("Loaded pipeline does not expose a transformer component")
        checkpoint = load_qwen_image_qat_checkpoint(checkpoint_path)
        injections = apply_qwen_image_qat_checkpoint(
            transformer,
            checkpoint=checkpoint,
            linear_cls=linear_cls,
        )
        metadata = (
            dict(config_metadata)
            if config_metadata is not None
            else collect_visual_gen_config_metadata(visual_args_path, model=model)
        )
        metadata.update(
            {
                "qat_checkpoint_path": str(checkpoint_path),
                "qat_checkpoint_train_steps": int(checkpoint["train_steps"]),
                "qat_injection_count": len(injections),
                "qat_eval_mode": "probe_fake_mxfp8_lora",
                "qat_eval_max_prompts": max_prompts,
            }
        )
        prompt_records = (
            list(records) if records is not None else read_prompt_jsonl(Path(prompt_manifest_jsonl))
        )
        if max_prompts is not None:
            if max_prompts <= 0:
                raise ValueError("max_prompts must be positive when set")
            prompt_records = [
                record for record in prompt_records if str(record.get("split")) == split
            ][:max_prompts]
        result = run_qwen_image_rgb_split_eval(
            records=prompt_records,
            split=split,
            model=model,
            visual_gen_args=visual_args_path,
            reference_root=Path(reference_root),
            output_root=Path(output_root),
            metrics_json=Path(metrics_json),
            variant=variant,
            reference_variant=reference_variant,
            device=device,
            provenance=dict(provenance or {}),
            config_metadata=metadata,
            pipeline=pipeline,
            infer_fn=infer_fn or infer_record,
            save_fn=save_fn or save_reference_image,
            metrics_fn=metrics_fn or compute_rgb_image_metrics,
        )
        result["qat_checkpoint_path"] = str(checkpoint_path)
        result["qat_checkpoint_train_steps"] = int(checkpoint["train_steps"])
        result["qat_injection_count"] = len(injections)
        write_json(Path(metrics_json), result)
        return result
    finally:
        if owns_pipeline:
            cleanup_pipeline(pipeline)


def monitor_qwen_image_qat_checkpoint(
    transformer: nn.Module,
    config: QwenImageQatMonitorConfig,
    *,
    model: str | None = None,
    visual_gen_args: str | Path | None = None,
    provenance: Mapping[str, object] | None = None,
    linear_cls: type[nn.Module] | None = None,
) -> dict[str, object]:
    """Evaluate a trained QAT adapter checkpoint on a fixed tuple subset."""
    _validate_monitor_config(config)
    dataset = QwenImageTupleDataset(config.tuple_index_jsonl)
    device = torch.device(config.device)
    transformer.to(device=device)
    checkpoint = load_qwen_image_qat_checkpoint(config.checkpoint_path)
    injections = apply_qwen_image_qat_checkpoint(
        transformer,
        checkpoint=checkpoint,
        linear_cls=linear_cls,
    )
    transformer.eval()
    loss_config = config.loss or _loss_config_from_checkpoint(checkpoint)
    sample_count = len(dataset)
    if config.max_samples is not None:
        sample_count = min(sample_count, int(config.max_samples))
    records_path = Path(config.records_jsonl) if config.records_jsonl is not None else None
    if records_path is not None:
        records_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    records: list[dict[str, object]] = []
    records_file = records_path.open("w", encoding="utf-8") if records_path is not None else None
    try:
        with torch.no_grad():
            for monitor_step in range(1, sample_count + 1):
                sample_index = _sample_index_for_step(
                    monitor_step,
                    dataset_size=len(dataset),
                    sample_stride=config.sample_stride,
                    sample_start_index=config.sample_start_index,
                )
                sample = dataset[sample_index]
                output = forward_qwen_image_tuple(transformer, sample, device)
                loss, components = compute_qwen_image_tuple_loss(output, sample, loss_config)
                record = _build_monitor_record(
                    monitor_step=monitor_step,
                    sample_index=sample_index,
                    sample=sample,
                    output=output,
                    loss=loss,
                    components=components,
                    elapsed_seconds=time.perf_counter() - start_time,
                )
                records.append(record)
                if records_file is not None:
                    records_file.write(json.dumps(record, sort_keys=True) + "\n")
                    records_file.flush()
    finally:
        if records_file is not None:
            records_file.close()

    output_path = Path(config.output_json)
    checkpoint_config = checkpoint["config"]
    summary: dict[str, object] = {
        "format": MONITOR_SUMMARY_FORMAT,
        "model": model,
        "visual_gen_args": str(visual_gen_args) if visual_gen_args is not None else None,
        "checkpoint_path": str(config.checkpoint_path),
        "tuple_index_jsonl": str(config.tuple_index_jsonl),
        "output_json": str(output_path),
        "records_jsonl": str(records_path) if records_path is not None else None,
        "checkpoint_train_steps": int(checkpoint["train_steps"]),
        "checkpoint_training_config": checkpoint_config,
        "monitor_config": _monitor_config_to_dict(config),
        "sample_count": sample_count,
        "dataset_size": len(dataset),
        "injection_count": len(injections),
        "metrics_summary": _summarize_monitor_records(records),
        "provenance": dict(provenance or {}),
    }
    write_json(output_path, summary)
    return summary


def load_qwen_image_qat_checkpoint(path: str | Path) -> dict[str, object]:
    """Load and validate a LoRA-only Qwen-Image QAT checkpoint."""
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"QAT checkpoint must be a mapping: {checkpoint_path}")
    if checkpoint.get("format") != TRAINING_FORMAT:
        raise ValueError(
            f"unsupported QAT checkpoint format {checkpoint.get('format')!r}: {checkpoint_path}"
        )
    required_fields = (
        "config",
        "train_steps",
        "trainable_state_dict",
        "trainable_parameter_names",
        "injections",
    )
    missing = [field for field in required_fields if field not in checkpoint]
    if missing:
        raise ValueError(f"QAT checkpoint missing fields {missing}: {checkpoint_path}")
    if not isinstance(checkpoint["config"], dict):
        raise ValueError(f"QAT checkpoint config must be a mapping: {checkpoint_path}")
    if not isinstance(checkpoint["trainable_state_dict"], dict):
        raise ValueError(
            f"QAT checkpoint trainable_state_dict must be a mapping: {checkpoint_path}"
        )
    if not isinstance(checkpoint["injections"], list):
        raise ValueError(f"QAT checkpoint injections must be a list: {checkpoint_path}")
    return checkpoint


def apply_qwen_image_qat_checkpoint(
    transformer: nn.Module,
    *,
    checkpoint: Mapping[str, object],
    linear_cls: type[nn.Module] | None = None,
) -> tuple[QwenImageQatInjectionInfo, ...]:
    """Inject fake-MXFP8 LoRA wrappers and restore trainable adapter weights."""
    checkpoint_config = _checkpoint_training_config(checkpoint)
    injections = prepare_qwen_image_qat_model(
        transformer,
        target_layers=tuple(str(item) for item in checkpoint_config.get("target_layers", [])),
        lora_rank=int(checkpoint_config["lora_rank"]),
        lora_alpha=float(checkpoint_config["lora_alpha"]),
        lora_dropout=float(checkpoint_config["lora_dropout"]),
        expected_num_layers=_optional_int(checkpoint_config.get("expected_num_layers")),
        expected_target_count=_optional_int(checkpoint_config.get("expected_target_count")),
        scale_multipliers=_optional_float_mapping(checkpoint_config.get("scale_multipliers")),
        linear_cls=linear_cls,
    )
    trainable_state = checkpoint.get("trainable_state_dict")
    if not isinstance(trainable_state, dict):
        raise ValueError("QAT checkpoint trainable_state_dict must be a mapping")
    named_parameters = {
        normalize_qwen_image_qat_parameter_name(name): parameter
        for name, parameter in transformer.named_parameters()
    }
    expected_names = qwen_image_qat_trainable_parameter_names(injections)
    actual_names = {str(name) for name in trainable_state}
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise ValueError(
            "QAT checkpoint trainable names do not match injected adapters: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    for name, tensor in trainable_state.items():
        normalized_name = str(name)
        parameter = named_parameters.get(normalized_name)
        if parameter is None:
            raise ValueError(f"QAT checkpoint tensor has no matching parameter: {normalized_name}")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"QAT checkpoint value must be a tensor: {normalized_name}")
        if tuple(tensor.shape) != tuple(parameter.shape):
            raise ValueError(
                f"QAT checkpoint tensor {normalized_name} shape {tuple(tensor.shape)} "
                f"does not match parameter shape {tuple(parameter.shape)}"
            )
        with torch.no_grad():
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    return injections


def summarize_qwen_image_qat_training_metrics(metrics_path: str | Path) -> dict[str, object]:
    """Summarize JSONL training metrics for recipe selection."""
    path = Path(metrics_path)
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not records:
        raise ValueError(f"training metrics contain no records: {path}")
    first_record = records[0]
    last_record = records[-1]
    best_record = min(records, key=lambda record: float(record["loss_total"]))
    return {
        "record_count": len(records),
        "first_step": int(first_record["step"]),
        "last_step": int(last_record["step"]),
        "first_loss_total": float(first_record["loss_total"]),
        "last_loss_total": float(last_record["loss_total"]),
        "loss_total_delta": float(last_record["loss_total"]) - float(first_record["loss_total"]),
        "best_loss_total": float(best_record["loss_total"]),
        "best_loss_step": int(best_record["step"]),
        "last_loss_mse": _optional_float_metric(last_record.get("loss_mse")),
        "last_loss_direction": _optional_float_metric(last_record.get("loss_direction")),
        "last_grad_norm": _optional_float_metric(last_record.get("grad_norm")),
        "last_lora_delta_norm": _optional_float_metric(last_record.get("lora_delta_norm")),
        "last_elapsed_seconds": _optional_float_metric(last_record.get("elapsed_seconds")),
        "timestep_bins": _summarize_metrics_by_timestep_bin(records),
    }


def qwen_image_qat_trainable_parameter_names(
    injections: tuple[QwenImageQatInjectionInfo, ...],
) -> set[str]:
    """Return fully qualified LoRA parameter names expected to require gradients."""
    return {
        normalize_qwen_image_qat_parameter_name(f"{injection.module_name}.{parameter_name}")
        for injection in injections
        for parameter_name in injection.trainable_parameter_names
    }


def normalize_qwen_image_qat_parameter_name(name: str) -> str:
    """Normalize parameter names across torch.compile/DDP/FSDP wrappers."""
    normalized = name
    prefixes = ("module.", "_fsdp_wrapped_module.", "_checkpoint_wrapped_module.")
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
    return normalize_qwen_module_name(normalized)


def build_qwen_image_qat_scale_sensitivity_manifest(
    transformer: nn.Module,
    *,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Summarize per-block/role MXFP8 weight scale stats for wrapped LoRA targets."""
    records: list[dict[str, object]] = []
    for module_name, module in transformer.named_modules():
        if not isinstance(module, Mxfp8LoraAdapterLinear):
            continue
        normalized_name = normalize_qwen_image_qat_parameter_name(module_name)
        _, scales = fake_mxfp8_weight_quantize(
            module.weight.detach(),
            block_size=module.base.block_size,
            scale_mode=module.base.scale_mode,
        )
        flat = scales.detach().float().reshape(-1).cpu()
        block_index, role = _parse_qwen_block_name(normalized_name)
        records.append(
            {
                "module_name": normalized_name,
                "block_index": block_index,
                "role": role,
                "scale_min": float(flat.min()),
                "scale_mean": float(flat.mean()),
                "scale_p95": float(torch.quantile(flat, 0.95)),
                "scale_max": float(flat.max()),
                "scale_num_blocks": int(flat.numel()),
                "scale_multiplier": float(module.scale_multiplier),
            }
        )

    role_summary: dict[str, dict[str, float | int]] = {}
    for role in sorted({str(record["role"]) for record in records}):
        role_records = [record for record in records if record["role"] == role]
        role_summary[role] = {
            "count": len(role_records),
            "scale_mean": sum(float(record["scale_mean"]) for record in role_records)
            / max(1, len(role_records)),
            "scale_max": max(float(record["scale_max"]) for record in role_records),
        }
    manifest = {
        "format": "qwen_image_qat_scale_sensitivity_manifest_v1",
        "status": "passed" if records else "failed",
        "record_count": len(records),
        "role_summary": role_summary,
        "records": records,
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def compute_scale_aware_lora_multipliers(
    scale_manifest: Mapping[str, object],
    *,
    clip_min: float = 0.25,
    clip_max: float = 4.0,
) -> dict[str, float]:
    """Build per-module LoRA multipliers from normalized MXFP8 scale means."""
    if clip_min <= 0.0 or clip_max < clip_min:
        raise ValueError("scale multiplier clip bounds must satisfy 0 < clip_min <= clip_max")
    records = scale_manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("scale manifest contains no records")
    scale_means = [float(record["scale_mean"]) for record in records if isinstance(record, dict)]
    if not scale_means:
        raise ValueError("scale manifest contains no scale_mean values")
    global_mean = sum(scale_means) / len(scale_means)
    if global_mean <= 0.0:
        raise ValueError("scale manifest global scale mean must be positive")
    multipliers: dict[str, float] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        module_name = str(record["module_name"])
        raw_multiplier = float(record["scale_mean"]) / global_mean
        multipliers[module_name] = min(clip_max, max(clip_min, raw_multiplier))
    return multipliers


def _checkpoint_training_config(checkpoint: Mapping[str, object]) -> Mapping[str, object]:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("QAT checkpoint config must be a mapping")
    return config


def _loss_config_from_checkpoint(checkpoint: Mapping[str, object]) -> QwenImageTupleLossConfig:
    checkpoint_config = _checkpoint_training_config(checkpoint)
    loss = checkpoint_config.get("loss")
    if not isinstance(loss, dict):
        return QwenImageTupleLossConfig()
    timestep_weights = loss.get("timestep_weights")
    if isinstance(timestep_weights, dict) and timestep_weights:
        parsed_timestep_weights = {
            str(key): float(value) for key, value in timestep_weights.items()
        }
    else:
        parsed_timestep_weights = None
    return QwenImageTupleLossConfig(
        lambda_mse=float(loss.get("lambda_mse", 1.0)),
        lambda_dir=float(loss.get("lambda_dir", 0.0)),
        timestep_weights=parsed_timestep_weights,
    )


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_float_mapping(value: object) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("scale_multipliers must be a mapping in QAT checkpoint config")
    return {str(key): float(multiplier) for key, multiplier in value.items()}


def _validate_monitor_config(config: QwenImageQatMonitorConfig) -> None:
    if config.max_samples is not None and config.max_samples <= 0:
        raise ValueError("max_samples must be positive when set")
    if config.sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    if config.sample_start_index < 0:
        raise ValueError("sample_start_index must be non-negative")


def _monitor_config_to_dict(config: QwenImageQatMonitorConfig) -> dict[str, object]:
    loss = config.loss
    return {
        "checkpoint_path": str(config.checkpoint_path),
        "tuple_index_jsonl": str(config.tuple_index_jsonl),
        "output_json": str(config.output_json),
        "records_jsonl": str(config.records_jsonl) if config.records_jsonl is not None else None,
        "max_samples": config.max_samples,
        "sample_stride": config.sample_stride,
        "sample_start_index": config.sample_start_index,
        "device": config.device,
        "loss": None
        if loss is None
        else {
            "lambda_mse": loss.lambda_mse,
            "lambda_dir": loss.lambda_dir,
            "timestep_weights": dict(loss.timestep_weights or {}),
        },
    }


def _build_monitor_record(
    *,
    monitor_step: int,
    sample_index: int,
    sample: QwenImageTupleSample,
    output: torch.Tensor,
    loss: torch.Tensor,
    components: Mapping[str, torch.Tensor],
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "monitor_step": monitor_step,
        "sample_index": sample_index,
        "prompt_id": sample.prompt_id,
        "split": sample.split,
        "timestep_index": sample.timestep_index,
        "timestep_bin": sample.timestep_bin,
        "cfg_branch": sample.cfg_branch,
        "loss_total": _tensor_scalar(loss),
        "loss_unweighted_total": _tensor_scalar(components["unweighted_total"]),
        "loss_mse": _optional_component_scalar(components, "mse"),
        "loss_direction": _optional_component_scalar(components, "direction"),
        "direction_loss": _tuple_direction_loss(output, sample),
        "timestep_weight": _tensor_scalar(components["timestep_weight"]),
        "elapsed_seconds": float(elapsed_seconds),
    }


def _tuple_direction_loss(output: torch.Tensor, sample: QwenImageTupleSample) -> float:
    target = sample.target_output.to(device=output.device, dtype=output.dtype)
    value = (
        1.0
        - F.cosine_similarity(
            output.float().reshape(1, -1),
            target.float().reshape(1, -1),
            dim=1,
            eps=1.0e-8,
        ).mean()
    )
    return _tensor_scalar(value)


def _summarize_monitor_records(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        raise ValueError("monitor records must not be empty")
    prompt_ids = {str(record["prompt_id"]) for record in records}
    sample_indices = {int(record["sample_index"]) for record in records}
    return {
        "record_count": len(records),
        "prompt_count": len(prompt_ids),
        "sample_index_unique": len(sample_indices),
        "first_monitor_step": int(records[0]["monitor_step"]),
        "last_monitor_step": int(records[-1]["monitor_step"]),
        "loss_total_mean": _mean_record_metric(records, "loss_total"),
        "loss_mse_mean": _mean_record_metric(records, "loss_mse"),
        "loss_mse_max": _max_record_metric(records, "loss_mse"),
        "direction_loss_mean": _mean_record_metric(records, "direction_loss"),
        "direction_loss_max": _max_record_metric(records, "direction_loss"),
        "timestep_bins": _summarize_metrics_by_timestep_bin(records),
        "cfg_branches": _summarize_records_by_field(records, "cfg_branch"),
    }


def _probe_loss_config(recipe: str) -> QwenImageTupleLossConfig:
    if recipe == "mse_only":
        return QwenImageTupleLossConfig(lambda_mse=1.0, lambda_dir=0.0)
    if recipe == "timestep_weighted":
        return QwenImageTupleLossConfig(
            lambda_mse=1.0,
            lambda_dir=0.0,
            timestep_weights=DEFAULT_TIMESTEP_WEIGHTS,
        )
    if recipe in ("direction_aware", "scale_aware_lora"):
        return QwenImageTupleLossConfig(
            lambda_mse=1.0,
            lambda_dir=0.1,
            timestep_weights=DEFAULT_TIMESTEP_WEIGHTS,
        )
    raise ValueError(f"unsupported Qwen-Image QAT probe recipe {recipe!r}")


def _maybe_build_probe_scale_multipliers(
    *,
    recipe: str,
    model: str,
    visual_gen_args: Path,
    output_dir: Path,
    device: str,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    expected_num_layers: int | None,
    expected_target_count: int | None,
    scale_clip_min: float,
    scale_clip_max: float,
) -> dict[str, float] | None:
    if recipe != "scale_aware_lora":
        return None
    scale_dir = output_dir / "scale_sensitivity"
    scale_manifest_path = scale_dir / "scale_sensitivity_manifest.json"
    scale_multipliers_path = scale_dir / "scale_multipliers.json"
    pipeline = load_single_worker_pipeline(
        model=model,
        visual_gen_args=visual_gen_args,
        device=device,
    )
    try:
        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            raise ValueError("Loaded pipeline does not expose a transformer component")
        prepare_qwen_image_qat_model(
            transformer,
            target_layers=(QWEN_BLOCK_LINEAR_TARGET,),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            expected_num_layers=expected_num_layers,
            expected_target_count=expected_target_count,
        )
        manifest = build_qwen_image_qat_scale_sensitivity_manifest(
            transformer,
            output_path=scale_manifest_path,
        )
        multipliers = compute_scale_aware_lora_multipliers(
            manifest,
            clip_min=scale_clip_min,
            clip_max=scale_clip_max,
        )
        write_json(
            scale_multipliers_path,
            {
                "format": "qwen_image_qat_scale_aware_multipliers_v1",
                "clip_min": scale_clip_min,
                "clip_max": scale_clip_max,
                "scale_manifest": str(scale_manifest_path),
                "multipliers": multipliers,
            },
        )
        return multipliers
    finally:
        cleanup_pipeline(pipeline)


def _optional_float_metric(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"metric value must be numeric or null, got {value!r}")
    return float(value)


def _summarize_metrics_by_timestep_bin(records: list[dict[str, object]]) -> dict[str, object]:
    by_bin: dict[str, list[dict[str, object]]] = {}
    for record in records:
        timestep_bin = str(record.get("timestep_bin"))
        by_bin.setdefault(timestep_bin, []).append(record)
    return {
        timestep_bin: {
            "count": len(bin_records),
            "loss_total_mean": _mean_record_metric(bin_records, "loss_total"),
            "loss_mse_mean": _mean_record_metric(bin_records, "loss_mse"),
            "loss_direction_mean": _mean_record_metric(bin_records, "loss_direction"),
            "direction_loss_mean": _mean_record_metric(bin_records, "direction_loss"),
            "last_loss_total": _optional_float_metric(bin_records[-1].get("loss_total")),
        }
        for timestep_bin, bin_records in sorted(by_bin.items())
    }


def _mean_record_metric(records: list[dict[str, object]], field_name: str) -> float | None:
    values = [
        value
        for record in records
        if (value := _optional_float_metric(record.get(field_name))) is not None
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def _max_record_metric(records: list[dict[str, object]], field_name: str) -> float | None:
    values = [
        value
        for record in records
        if (value := _optional_float_metric(record.get(field_name))) is not None
    ]
    if not values:
        return None
    return float(max(values))


def _summarize_records_by_field(
    records: list[dict[str, object]],
    field_name: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field_name))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _read_tuple_index(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise ValueError(f"tuple index does not exist: {path}")
    entries = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        entry = json.loads(line)
        if not isinstance(entry, dict):
            raise ValueError(f"tuple index line {line_number} must be a mapping")
        if entry.get("status") != CAPTURED_TUPLE_STATUS:
            raise ValueError(f"tuple index line {line_number} must have status=captured")
        if entry.get("trajectory_source") != BF16_TEACHER_TRAJECTORY_SOURCE:
            raise ValueError(
                f"tuple index line {line_number} must have trajectory_source=bf16_teacher"
            )
        entries.append(entry)
    return entries


def _read_mapping_config(path: Path) -> dict[str, object]:
    if not path.exists():
        raise ValueError(f"config does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(f"PyYAML is required to load YAML config: {path}") from exc
        loaded = yaml.safe_load(text) or {}
    else:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return loaded


def _required_config_value(
    data: Mapping[str, object],
    field_name: str,
    path: Path,
) -> object:
    value = data.get(field_name)
    if value in (None, ""):
        raise ValueError(f"config missing required field {field_name}: {path}")
    return value


def _optional_mapping_config_value(
    data: Mapping[str, object],
    field_name: str,
) -> Mapping[str, float] | None:
    value = data.get(field_name)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"config field {field_name} must be a mapping")
    return {str(key): float(item) for key, item in value.items()}


def _optimizer_betas_from_config(value: object, path: Path) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"config field betas must contain two numeric values: {path}")
    beta1, beta2 = value
    return (float(beta1), float(beta2))


def _build_rollout_windows_from_samples(
    samples: Sequence[QwenImageTupleSample],
    *,
    rollout_k: int,
) -> tuple[tuple[QwenImageTupleSample, ...], ...]:
    grouped: dict[tuple[str, str], dict[int, dict[str, QwenImageTupleSample]]] = {}
    for sample in samples:
        key = (sample.prompt_id, sample.split)
        by_step = grouped.setdefault(key, {})
        branch_map = by_step.setdefault(sample.timestep_index, {})
        if sample.cfg_branch in branch_map:
            raise ValueError(
                "duplicate rollout tuple for prompt "
                f"{sample.prompt_id}, timestep {sample.timestep_index}, branch {sample.cfg_branch}"
            )
        branch_map[sample.cfg_branch] = sample

    windows: list[tuple[QwenImageTupleSample, ...]] = []
    for (_prompt_id, _split), by_step in sorted(grouped.items()):
        timestep_indices = sorted(by_step)
        for start_index in timestep_indices:
            window_samples: list[QwenImageTupleSample] = []
            complete = True
            for timestep_index in range(start_index, start_index + rollout_k):
                branch_map = by_step.get(timestep_index)
                if branch_map is None:
                    complete = False
                    break
                for cfg_branch in ROLLOUT_CFG_BRANCHES:
                    sample = branch_map.get(cfg_branch)
                    if sample is None:
                        complete = False
                        break
                    window_samples.append(sample)
                if not complete:
                    break
            if complete:
                _build_rollout_window(window_samples, rollout_k)
                windows.append(tuple(window_samples))
    return tuple(windows)


def _tuple_id_for_entry(entry: Mapping[str, object]) -> str:
    value = entry.get("tuple_id")
    if isinstance(value, str) and value:
        return value
    prompt_id = entry.get("prompt_id")
    timestep_index = entry.get("timestep_index")
    cfg_branch = entry.get("cfg_branch")
    if prompt_id is None or timestep_index is None or cfg_branch is None:
        raise ValueError("tuple index entry missing tuple_id and tuple key fields")
    return f"{prompt_id}_step{int(timestep_index):03d}_{cfg_branch}"


def _tuple_path_from_entry(entry: Mapping[str, object], *, manifest_dir: Path) -> Path:
    value = entry.get("tuple_path")
    if not isinstance(value, str):
        raise ValueError("tuple index entry missing tuple_path")
    path = Path(value)
    return path if path.is_absolute() else manifest_dir / path


def _rollout_output_tuple_path(
    output_tuple_root: Path,
    entry: Mapping[str, object],
    *,
    source_tuple_path: Path,
) -> Path:
    split = str(entry.get("split", "unknown_split"))
    prompt_id = str(entry.get("prompt_id", "unknown_prompt"))
    filename = source_tuple_path.name
    return output_tuple_root / split / prompt_id / filename


def _build_rollout_tuple_index_entry(
    source_entry: Mapping[str, object],
    *,
    metadata: Mapping[str, object],
    output_tuple_path: Path,
    output_index_dir: Path,
) -> dict[str, object]:
    entry = dict(source_entry)
    tuple_id = _tuple_id_for_entry(entry)
    rollout_provenance = _metadata_mapping(metadata, "rollout_provenance", tuple_id)
    reference_image_path = _metadata_string(metadata, "reference_image_path", tuple_id)
    scheduler_state = _metadata_mapping(metadata, "scheduler_state", tuple_id)
    guidance_scale = _metadata_float(metadata, "guidance_scale", tuple_id)
    required_fields = list(entry.get("required_fields") or [])
    for field_name in ROLLOUT_REQUIRED_FIELDS:
        if field_name not in required_fields:
            required_fields.append(field_name)
    entry.update(
        {
            "tuple_id": tuple_id,
            "tuple_path": _manifest_relative_path(output_tuple_path, output_index_dir),
            "required_fields": required_fields,
            "rollout_tuple_schema": ROLLOUT_TUPLE_SCHEMA_VERSION,
            "rollout_provenance": _json_safe_metadata(dict(rollout_provenance)),
            "scheduler_state": _json_safe_metadata(dict(scheduler_state)),
            "guidance_scale": guidance_scale,
            "reference_image_path": reference_image_path,
            "status": CAPTURED_TUPLE_STATUS,
            "trajectory_source": BF16_TEACHER_TRAJECTORY_SOURCE,
        }
    )
    return entry


def _manifest_relative_path(path: Path, manifest_dir: Path) -> str:
    return os.path.relpath(path.resolve(), start=manifest_dir.resolve())


def _metadata_tensor(
    metadata: Mapping[str, object],
    field_name: str,
    tuple_id: str,
) -> torch.Tensor:
    value = metadata.get(field_name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"rollout metadata {tuple_id}.{field_name} must be a tensor")
    return value


def _metadata_mapping(
    metadata: Mapping[str, object],
    field_name: str,
    tuple_id: str,
) -> Mapping[str, object]:
    value = metadata.get(field_name)
    if not isinstance(value, Mapping):
        raise ValueError(f"rollout metadata {tuple_id}.{field_name} must be a mapping")
    return value


def _metadata_string(
    metadata: Mapping[str, object],
    field_name: str,
    tuple_id: str,
) -> str:
    value = metadata.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"rollout metadata {tuple_id}.{field_name} must be a non-empty string")
    return value


def _metadata_float(
    metadata: Mapping[str, object],
    field_name: str,
    tuple_id: str,
) -> float:
    value = metadata.get(field_name)
    if not isinstance(value, (int, float)):
        raise ValueError(f"rollout metadata {tuple_id}.{field_name} must be numeric")
    return float(value)


def _json_safe_metadata(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_metadata(item) for item in value]
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return tensor.item()
        if tensor.numel() <= 16:
            return tensor.tolist()
        return {
            "dtype": str(tensor.dtype),
            "shape": [int(dim) for dim in tensor.shape],
        }
    if isinstance(value, Path):
        return str(value)
    return value


def _load_rollout_metadata_payload(
    record: Mapping[str, object],
    manifest_dir: Path,
    tuple_id: str,
) -> Mapping[str, object]:
    value = record.get("metadata_path")
    if value is None:
        return {}
    if not isinstance(value, str) or not value:
        raise ValueError(f"rollout metadata {tuple_id}.metadata_path must be a non-empty string")
    payload_path = _resolve_manifest_path(manifest_dir, value)
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"rollout metadata payload must be a mapping: {payload_path}")
    return payload


def _resolve_manifest_path(manifest_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_dir / path


def _rollout_metadata_tensor_from_record(
    record: Mapping[str, object],
    payload: Mapping[str, object],
    field_name: str,
    manifest_dir: Path,
    tuple_id: str,
) -> torch.Tensor:
    value = payload.get(field_name, record.get(field_name))
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    path_key = f"{field_name}_path"
    path_value = record.get(path_key, payload.get(path_key))
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"rollout metadata {tuple_id}.{path_key} must be a non-empty string")
    tensor_path = _resolve_manifest_path(manifest_dir, path_value)
    loaded = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if isinstance(loaded, torch.Tensor):
        return loaded.detach().cpu()
    if isinstance(loaded, Mapping):
        tensor_key = str(
            record.get(f"{field_name}_key", payload.get(f"{field_name}_key", field_name))
        )
        tensor_value = loaded.get(tensor_key)
        if isinstance(tensor_value, torch.Tensor):
            return tensor_value.detach().cpu()
    raise ValueError(f"rollout metadata {tuple_id}.{field_name} tensor not found in {tensor_path}")


def _rollout_metadata_mapping_from_record(
    record: Mapping[str, object],
    payload: Mapping[str, object],
    field_name: str,
    tuple_id: str,
) -> Mapping[str, object]:
    value = payload.get(field_name, record.get(field_name))
    if not isinstance(value, Mapping):
        raise ValueError(f"rollout metadata {tuple_id}.{field_name} must be a mapping")
    return value


def _rollout_metadata_float_from_record(
    record: Mapping[str, object],
    payload: Mapping[str, object],
    field_name: str,
    tuple_id: str,
) -> float:
    value = payload.get(field_name, record.get(field_name))
    if not isinstance(value, (int, float)):
        raise ValueError(f"rollout metadata {tuple_id}.{field_name} must be numeric")
    return float(value)


def _rollout_metadata_string_from_record(
    record: Mapping[str, object],
    payload: Mapping[str, object],
    field_name: str,
    tuple_id: str,
) -> str:
    value = payload.get(field_name, record.get(field_name))
    if not isinstance(value, str) or not value:
        raise ValueError(f"rollout metadata {tuple_id}.{field_name} must be a non-empty string")
    return value


def _scheduler_state_float(
    scheduler_state: Mapping[str, object],
    step: QwenImageRolloutStepSamples,
    field_names: tuple[str, ...],
) -> float:
    value = _scheduler_state_optional_float(scheduler_state, field_names)
    if value is None:
        raise ValueError(
            "rollout scheduler_state missing "
            f"{'/'.join(field_names)} for timestep {step.timestep_index}"
        )
    return value


def _scheduler_state_optional_float(
    scheduler_state: Mapping[str, object],
    field_names: tuple[str, ...],
) -> float | None:
    for field_name in field_names:
        value = scheduler_state.get(field_name)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return float(value.detach().float().cpu())
    return None


def _validate_rollout_scheduler_step_contract(
    scheduler_state: Mapping[str, object],
    step: QwenImageRolloutStepSamples,
) -> None:
    required_groups = (
        ("scheduler_class", "scheduler_name"),
        ("timestep", "timestep_index"),
        ("step_index", "timestep_index"),
        ("num_inference_steps", "total_steps"),
    )
    missing_groups = [
        "/".join(field_names)
        for field_names in required_groups
        if not any(scheduler_state.get(field_name) not in (None, "") for field_name in field_names)
    ]
    if missing_groups:
        raise ValueError(
            "rollout scheduler_state missing required scheduler contract fields "
            f"{missing_groups} for timestep {step.timestep_index}"
        )
    scheduler_name = _rollout_scheduler_identifier(scheduler_state)
    if scheduler_name not in SUPPORTED_ROLLOUT_SCHEDULER_NAMES:
        raise ValueError(
            "rollout scheduler_step supports only FlowMatch/Euler scheduler metadata, "
            f"got {scheduler_state.get('scheduler_class') or scheduler_state.get('scheduler_name')}"
        )
    unsupported_flags = [
        flag_name
        for flag_name in UNSUPPORTED_ROLLOUT_SCHEDULER_FLAGS
        if scheduler_state.get(flag_name) is True
    ]
    if unsupported_flags:
        raise ValueError(
            "rollout scheduler_step does not support stochastic/per-token scheduler modes "
            f"{unsupported_flags}"
        )
    provenance = step.cond.rollout_provenance
    if not isinstance(provenance, Mapping):
        raise ValueError("rollout scheduler contract requires rollout_provenance")
    missing_provenance = [
        field_name
        for field_name in ("scheduler_config_signature", "sigmas_hash")
        if provenance.get(field_name) in (None, "")
    ]
    if missing_provenance:
        raise ValueError(
            "rollout scheduler contract missing provenance fields "
            f"{missing_provenance} for timestep {step.timestep_index}"
        )


def _rollout_scheduler_identifier(scheduler_state: Mapping[str, object]) -> str:
    raw_name = scheduler_state.get("scheduler_class") or scheduler_state.get("scheduler_name")
    return "".join(ch for ch in str(raw_name).lower() if ch.isalnum())


def _build_rollout_augmentation_summary(
    *,
    config: QwenImageRolloutTupleAugmentationConfig,
    output_entries: list[dict[str, object]],
    source_index_path: Path,
    output_index_path: Path,
    output_tuple_root: Path,
) -> dict[str, object]:
    prompt_ids = {str(entry.get("prompt_id")) for entry in output_entries}
    return {
        "format": ROLLOUT_AUGMENTATION_SUMMARY_FORMAT,
        "source_tuple_index_jsonl": str(source_index_path),
        "output_tuple_index_jsonl": str(output_index_path),
        "output_tuple_root": str(output_tuple_root),
        "summary_json": str(config.summary_json) if config.summary_json is not None else None,
        "rollout_tuple_schema": ROLLOUT_TUPLE_SCHEMA_VERSION,
        "tuple_count": len(output_entries),
        "prompt_count": len(prompt_ids),
        "validate_prompt_continuity": bool(config.validate_prompt_continuity),
        "status": "passed",
        "provenance": dict(config.provenance or {}),
    }


def _entry_from_payload(payload: Mapping[str, object], path: Path) -> dict[str, object]:
    entry = {
        field: payload.get(field)
        for field in (
            "prompt_id",
            "split",
            "timestep_index",
            "timestep_bin",
            "cfg_branch",
        )
    }
    entry["trajectory_source"] = payload.get("trajectory_source")
    missing = [field for field, value in entry.items() if value is None]
    if missing:
        raise ValueError(f"teacher tuple payload missing metadata {missing}: {path}")
    return entry


def _reject_layered_payload_fields(payload: Mapping[str, object], path: Path) -> None:
    layered_fields = sorted(set(payload) & _LAYERED_ONLY_FIELDS)
    if layered_fields:
        raise ValueError(
            "ordinary Qwen-Image QAT tuple cannot contain layered-only fields "
            f"{layered_fields}: {path}"
        )


def _expect_tensor(payload: Mapping[str, object], field_name: str, path: Path) -> torch.Tensor:
    value = _expect_field(payload, field_name, path)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"teacher tuple field {field_name} must be a tensor: {path}")
    return value


def _optional_tensor(
    payload: Mapping[str, object],
    field_name: str,
    path: Path,
) -> torch.Tensor | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"teacher tuple field {field_name} must be a tensor when present: {path}")
    return value


def _optional_mapping(
    payload: Mapping[str, object],
    field_name: str,
    path: Path,
) -> Mapping[str, object] | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"teacher tuple field {field_name} must be a mapping when present: {path}")
    return value


def _optional_mapping_from_payload_or_entry(
    payload: Mapping[str, object],
    entry: Mapping[str, object],
    field_name: str,
    path: Path,
) -> Mapping[str, object] | None:
    value = payload.get(field_name, entry.get(field_name))
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"teacher tuple field {field_name} must be a mapping when present: {path}")
    return value


def _optional_string_from_payload_or_entry(
    payload: Mapping[str, object],
    entry: Mapping[str, object],
    field_name: str,
    path: Path,
) -> str | None:
    value = payload.get(field_name, entry.get(field_name))
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"teacher tuple field {field_name} must be a string when present: {path}")
    return value


def _optional_float_from_payload_or_entry(
    payload: Mapping[str, object],
    entry: Mapping[str, object],
    field_name: str,
    path: Path,
) -> float | None:
    value = payload.get(field_name, entry.get(field_name))
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"teacher tuple field {field_name} must be numeric when present: {path}")
    return float(value)


def _expect_field(payload: Mapping[str, object], field_name: str, path: Path) -> object:
    if field_name not in payload:
        raise ValueError(f"teacher tuple payload missing field {field_name}: {path}")
    return payload[field_name]


def _move_value(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, list):
        return [_move_value(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_value(item, device) for key, item in value.items()}
    return value


def _timestep_weight(
    sample: QwenImageTupleSample,
    config: QwenImageTupleLossConfig,
) -> float:
    if config.timestep_weights is None:
        return 1.0
    return float(config.timestep_weights.get(sample.timestep_bin, 1.0))


def _rollout_timestep_weight(
    sample: QwenImageTupleSample,
    config: QwenImageRolloutLossConfig,
) -> float:
    if config.timestep_weights is None:
        return 1.0
    return float(config.timestep_weights.get(sample.timestep_bin, 1.0))


def _validate_rollout_loss_config(config: QwenImageRolloutLossConfig) -> None:
    if config.rollout_k <= 0:
        raise ValueError("rollout_k must be positive")
    weights = (
        config.lambda_output_mse,
        config.lambda_dir,
        config.lambda_latent,
        config.lambda_anchor,
    )
    if any(float(weight) < 0.0 for weight in weights):
        raise ValueError("rollout loss weights must be non-negative")
    if all(float(weight) == 0.0 for weight in weights):
        raise ValueError("at least one rollout loss weight must be positive")
    if not config.teacher_no_grad:
        raise ValueError("closed-set rollout teacher targets must use no_grad")
    if config.student_latent_detach:
        raise ValueError("closed-set rollout must not detach student intermediate latents")


def _build_rollout_window(
    samples: Sequence[QwenImageTupleSample],
    rollout_k: int,
) -> tuple[QwenImageRolloutStepSamples, ...]:
    candidates = tuple(samples)[: rollout_k * len(ROLLOUT_CFG_BRANCHES)]
    if len(candidates) < rollout_k * len(ROLLOUT_CFG_BRANCHES):
        raise ValueError(
            "rollout window must contain paired CFG branch samples for at least "
            f"{rollout_k} denoise steps"
        )
    for sample in candidates:
        validate_qwen_image_rollout_tuple_sample(sample)

    by_timestep: dict[int, dict[str, QwenImageTupleSample]] = {}
    for sample in candidates:
        if sample.cfg_branch not in ROLLOUT_CFG_BRANCHES:
            raise ValueError(f"rollout tuple has unsupported cfg_branch: {sample.cfg_branch}")
        branch_map = by_timestep.setdefault(sample.timestep_index, {})
        if sample.cfg_branch in branch_map:
            raise ValueError(
                "rollout window contains duplicate CFG branch "
                f"{sample.cfg_branch} for timestep {sample.timestep_index}"
            )
        branch_map[sample.cfg_branch] = sample

    timestep_indices = sorted(by_timestep)
    if len(timestep_indices) != rollout_k:
        raise ValueError(
            "rollout window must contain exactly "
            f"{rollout_k} paired denoise steps, got {len(timestep_indices)}"
        )
    for previous, timestep_index in zip(timestep_indices, timestep_indices[1:]):
        if timestep_index != previous + 1:
            raise ValueError("rollout window timestep_index values must be consecutive")

    steps = []
    first_step: QwenImageRolloutStepSamples | None = None
    for timestep_index in timestep_indices:
        branch_map = by_timestep[timestep_index]
        missing = set(ROLLOUT_CFG_BRANCHES) - set(branch_map)
        if missing:
            raise ValueError(
                f"rollout window missing CFG branch(es) {sorted(missing)} "
                f"for timestep {timestep_index}"
            )
        step = _build_rollout_step_samples(
            branch_map[ROLLOUT_COND_BRANCH],
            branch_map[ROLLOUT_NEGATIVE_BRANCH],
        )
        if first_step is None:
            first_step = step
        else:
            _validate_rollout_step_matches_first(step, first_step)
        steps.append(step)
    _validate_rollout_adjacent_step_continuity(tuple(steps))
    return tuple(steps)


def _build_rollout_step_samples(
    cond: QwenImageTupleSample,
    negative: QwenImageTupleSample,
) -> QwenImageRolloutStepSamples:
    if cond.prompt_id != negative.prompt_id:
        raise ValueError("rollout CFG branch pair must have the same prompt_id")
    if cond.split != negative.split:
        raise ValueError("rollout CFG branch pair must have the same split")
    if cond.timestep_index != negative.timestep_index:
        raise ValueError("rollout CFG branch pair must have the same timestep_index")
    if cond.timestep_bin != negative.timestep_bin:
        raise ValueError("rollout CFG branch pair must have the same timestep_bin")
    if cond.guidance_scale != negative.guidance_scale:
        raise ValueError("rollout CFG branch pair must have the same guidance_scale")
    if cond.reference_image_path != negative.reference_image_path:
        raise ValueError("rollout CFG branch pair must have the same reference_image_path")
    if _cfg_pair_scheduler_signature(cond) != _cfg_pair_scheduler_signature(negative):
        raise ValueError("rollout CFG branch pair must have matching scheduler metadata")
    if cond.latent_before_step is None or negative.latent_before_step is None:
        raise ValueError("rollout CFG branch pair missing latent_before_step")
    if cond.latent_after_step is None or negative.latent_after_step is None:
        raise ValueError("rollout CFG branch pair missing latent_after_step")
    if tuple(cond.latent_before_step.shape) != tuple(negative.latent_before_step.shape):
        raise ValueError("rollout CFG branch pair latent_before_step shapes must match")
    if tuple(cond.latent_after_step.shape) != tuple(negative.latent_after_step.shape):
        raise ValueError("rollout CFG branch pair latent_after_step shapes must match")
    if not torch.equal(cond.latent_before_step, negative.latent_before_step):
        raise ValueError("rollout CFG branch pair latent_before_step values must match")
    if not torch.equal(cond.latent_after_step, negative.latent_after_step):
        raise ValueError("rollout CFG branch pair latent_after_step values must match")
    if cond.guidance_scale is None:
        raise ValueError("rollout CFG branch pair missing guidance_scale")
    return QwenImageRolloutStepSamples(
        timestep_index=cond.timestep_index,
        timestep_bin=cond.timestep_bin,
        cond=cond,
        negative=negative,
        guidance_scale=cond.guidance_scale,
    )


def _validate_rollout_step_matches_first(
    step: QwenImageRolloutStepSamples,
    first: QwenImageRolloutStepSamples,
) -> None:
    if step.cond.prompt_id != first.cond.prompt_id:
        raise ValueError("rollout window samples must have the same prompt_id")
    if step.cond.split != first.cond.split:
        raise ValueError("rollout window samples must have the same split")
    if step.guidance_scale != first.guidance_scale:
        raise ValueError("rollout window samples must have the same guidance_scale")
    if step.cond.reference_image_path != first.cond.reference_image_path:
        raise ValueError("rollout window samples must have the same reference_image_path")
    if _rollout_schedule_signature(step.cond) != _rollout_schedule_signature(first.cond):
        raise ValueError("rollout window samples must have matching scheduler schedule provenance")


def _validate_rollout_adjacent_step_continuity(
    steps: tuple[QwenImageRolloutStepSamples, ...],
) -> None:
    for current, next_step in zip(steps, steps[1:]):
        _validate_rollout_latent_continuity(
            current.cond,
            next_step.cond,
            branch_name=ROLLOUT_COND_BRANCH,
        )
        _validate_rollout_latent_continuity(
            current.negative,
            next_step.negative,
            branch_name=ROLLOUT_NEGATIVE_BRANCH,
        )


def _validate_rollout_latent_continuity(
    current: QwenImageTupleSample,
    next_sample: QwenImageTupleSample,
    *,
    branch_name: str,
) -> None:
    if current.latent_after_step is None or next_sample.latent_before_step is None:
        raise ValueError("rollout adjacent latent continuity requires latent tensors")
    if tuple(current.latent_after_step.shape) != tuple(next_sample.latent_before_step.shape):
        raise ValueError(
            "rollout adjacent latent continuity shapes must match for "
            f"{branch_name} step {current.timestep_index}->{next_sample.timestep_index}"
        )
    if not torch.equal(current.latent_after_step, next_sample.latent_before_step):
        raise ValueError(
            "rollout adjacent latent continuity mismatch for "
            f"{branch_name} step {current.timestep_index}->{next_sample.timestep_index}"
        )


def _validate_rollout_prompt_continuity(samples: Sequence[QwenImageTupleSample]) -> None:
    grouped: dict[tuple[str, str], dict[int, dict[str, QwenImageTupleSample]]] = {}
    for sample in samples:
        branch_map = grouped.setdefault((sample.prompt_id, sample.split), {}).setdefault(
            sample.timestep_index,
            {},
        )
        branch_map[sample.cfg_branch] = sample
    for (_prompt_id, _split), by_step in sorted(grouped.items()):
        timestep_indices = sorted(by_step)
        for current_index, next_index in zip(timestep_indices, timestep_indices[1:]):
            if next_index != current_index + 1:
                continue
            for cfg_branch in ROLLOUT_CFG_BRANCHES:
                current = by_step[current_index].get(cfg_branch)
                next_sample = by_step[next_index].get(cfg_branch)
                if current is not None and next_sample is not None:
                    _validate_rollout_latent_continuity(
                        current,
                        next_sample,
                        branch_name=cfg_branch,
                    )


def _forward_rollout_teacher_branch(
    teacher_input: torch.Tensor,
    sample: QwenImageTupleSample,
    device: torch.device,
    *,
    teacher_transformer: nn.Module | None,
    teacher_forward_fn: Callable[[torch.Tensor, QwenImageTupleSample], torch.Tensor] | None,
) -> torch.Tensor:
    if teacher_forward_fn is not None:
        return teacher_forward_fn(teacher_input, sample)
    if teacher_transformer is None:
        raise ValueError("teacher_transformer unexpectedly missing")
    return _forward_qwen_image_tuple_with_hidden_states(
        teacher_transformer,
        sample,
        device,
        hidden_states=teacher_input,
    )


def _captured_rollout_teacher_target(
    teacher_input: torch.Tensor,
    sample: QwenImageTupleSample,
) -> torch.Tensor:
    return sample.target_output.to(device=teacher_input.device, dtype=teacher_input.dtype)


def _validate_output_shape_pair(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    branch_name: str,
) -> None:
    if tuple(student_output.shape) != tuple(teacher_output.shape):
        raise ValueError(
            f"{branch_name} student and teacher output shapes must match, got "
            f"{tuple(student_output.shape)} vs {tuple(teacher_output.shape)}"
        )


def _tensor_direction_loss(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
) -> torch.Tensor:
    return (
        1.0
        - F.cosine_similarity(
            student_output.float().reshape(1, -1),
            teacher_output.float().reshape(1, -1),
            dim=1,
            eps=1.0e-8,
        ).mean()
    )


def _has_equivalent_scheduler_provenance(sample: QwenImageTupleSample) -> bool:
    provenance = sample.rollout_provenance
    if not isinstance(provenance, Mapping):
        return False
    if provenance.get("scheduler_state_equivalent") is not True:
        return False
    missing = [
        field_name
        for field_name in ROLLOUT_PROVENANCE_REQUIRED_FIELDS
        if provenance.get(field_name) in (None, "")
    ]
    return not missing


def _validate_rollout_provenance(sample: QwenImageTupleSample) -> None:
    provenance = sample.rollout_provenance
    if not isinstance(provenance, Mapping):
        raise ValueError(f"rollout tuple missing rollout_provenance: {sample.path}")
    missing = [
        field_name
        for field_name in ROLLOUT_PROVENANCE_REQUIRED_FIELDS
        if provenance.get(field_name) in (None, "")
    ]
    if missing:
        raise ValueError(
            f"rollout tuple rollout_provenance missing fields {missing}: {sample.path}"
        )
    if str(provenance["prompt_id"]) != sample.prompt_id:
        raise ValueError(f"rollout tuple provenance prompt_id mismatch: {sample.path}")
    if str(provenance["reference_image_path"]) != sample.reference_image_path:
        raise ValueError(f"rollout tuple provenance reference_image_path mismatch: {sample.path}")
    if float(provenance["guidance_scale"]) != float(sample.guidance_scale or 0.0):
        raise ValueError(f"rollout tuple provenance guidance_scale mismatch: {sample.path}")


def _rollout_schedule_signature(sample: QwenImageTupleSample) -> object:
    provenance = sample.rollout_provenance
    if isinstance(provenance, Mapping):
        stable_provenance = {
            key: provenance.get(key)
            for key in ROLLOUT_PROVENANCE_REQUIRED_FIELDS
            if key not in ("prompt_id", "reference_image_path")
        }
        stable_provenance["prompt_id"] = sample.prompt_id
        stable_provenance["reference_image_path"] = sample.reference_image_path
        return _metadata_signature(stable_provenance)
    scheduler_state = sample.scheduler_state or {}
    if isinstance(scheduler_state, Mapping):
        stable_state = {
            str(key): value
            for key, value in scheduler_state.items()
            if str(key) not in ROLLOUT_PER_STEP_SCHEDULER_FIELDS
        }
        return _metadata_signature(stable_state)
    return None


def _cfg_pair_scheduler_signature(sample: QwenImageTupleSample) -> object:
    if sample.scheduler_state is not None:
        return ("scheduler_state", _metadata_signature(sample.scheduler_state))
    return ("rollout_schedule", _rollout_schedule_signature(sample))


def _metadata_signature(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _metadata_signature(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_metadata_signature(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.numel() <= 16:
            return {
                "dtype": str(value.dtype),
                "shape": tuple(int(dim) for dim in value.shape),
                "values": value.detach().cpu().tolist(),
            }
        return {
            "dtype": str(value.dtype),
            "shape": tuple(int(dim) for dim in value.shape),
        }
    return value


def _rollout_loss_config_to_dict(config: QwenImageRolloutLossConfig) -> dict[str, object]:
    return {
        "rollout_k": config.rollout_k,
        "lambda_output_mse": config.lambda_output_mse,
        "lambda_dir": config.lambda_dir,
        "lambda_latent": config.lambda_latent,
        "lambda_anchor": config.lambda_anchor,
        "timestep_weights": dict(config.timestep_weights or {}),
        "teacher_no_grad": config.teacher_no_grad,
        "student_latent_detach": config.student_latent_detach,
        "cfg_normalize": config.cfg_normalize,
        "student_activation_checkpoint": config.student_activation_checkpoint,
    }


def _build_rollout_step_record(
    *,
    rollout_index: int,
    step: QwenImageRolloutStepSamples,
    components: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    return {
        "rollout_index": int(rollout_index),
        "prompt_id": step.cond.prompt_id,
        "split": step.cond.split,
        "timestep_index": step.timestep_index,
        "timestep_bin": step.timestep_bin,
        "cfg_branches": list(ROLLOUT_CFG_BRANCHES),
        "guidance_scale": float(step.guidance_scale),
        "loss_total": _tensor_scalar(components["timestep_weighted_total"]),
        "loss_unweighted_total": _tensor_scalar(components["unweighted_total"]),
        "loss_output_mse": _tensor_scalar(components["output_mse"]),
        "loss_cond_output_mse": _tensor_scalar(components["cond_output_mse"]),
        "loss_negative_output_mse": _tensor_scalar(components["negative_output_mse"]),
        "loss_direction": _tensor_scalar(components["direction"]),
        "loss_cond_direction": _tensor_scalar(components["cond_direction"]),
        "loss_negative_direction": _tensor_scalar(components["negative_direction"]),
        "loss_latent_mse": _tensor_scalar(components["latent_mse"]),
        "loss_anchor_mse": _tensor_scalar(components["anchor_mse"]),
        "timestep_weight": _tensor_scalar(components["timestep_weight"]),
    }


def _filter_targets(
    targets: list[QwenImageBlockLinearTarget],
    selectors: tuple[str, ...],
) -> list[QwenImageBlockLinearTarget]:
    selector_set = set(selectors)
    if QWEN_BLOCK_LINEAR_TARGET in selector_set:
        return list(targets)
    return [
        target
        for target in targets
        if target.normalized_name in selector_set or target.role in selector_set
    ]


def _scale_multiplier_for_target(
    target: QwenImageBlockLinearTarget,
    scale_multipliers: Mapping[str, float] | None,
) -> float:
    if scale_multipliers is None:
        return 1.0
    value = scale_multipliers.get(target.normalized_name)
    if value is None:
        value = scale_multipliers.get(target.role)
    if value is None:
        return 1.0
    multiplier = float(value)
    if multiplier <= 0.0:
        raise ValueError(
            f"scale multiplier for {target.normalized_name} must be positive, got {multiplier}"
        )
    return multiplier


def _sample_index_for_step(
    step: int,
    *,
    dataset_size: int,
    sample_stride: int,
    sample_start_index: int,
) -> int:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    return (int(sample_start_index) + (int(step) - 1) * int(sample_stride)) % int(dataset_size)


def _apply_progressive_warmup(
    transformer: nn.Module,
    *,
    step: int,
    warmup_steps: int,
    warmup_target_layers: tuple[str, ...],
) -> bool:
    warmup_active = warmup_steps > 0 and step <= warmup_steps
    if warmup_steps > 0 and not warmup_target_layers:
        raise ValueError("warmup_target_layers must be non-empty when warmup_steps > 0")
    selectors = set(warmup_target_layers)
    for module_name, module in transformer.named_modules():
        if not isinstance(module, Mxfp8LoraAdapterLinear):
            continue
        normalized_name = normalize_qwen_image_qat_parameter_name(module_name)
        _, role = _parse_qwen_block_name(normalized_name)
        enabled = not warmup_active or normalized_name in selectors or role in selectors
        module.lora_down.weight.requires_grad_(enabled)
        module.lora_up.weight.requires_grad_(enabled)
    return warmup_active


def _parse_qwen_block_name(normalized_name: str) -> tuple[int | None, str]:
    prefix = "transformer_blocks."
    if not normalized_name.startswith(prefix):
        return None, normalized_name
    remainder = normalized_name[len(prefix) :]
    block_text, sep, role = remainder.partition(".")
    if not sep or not block_text.isdigit():
        return None, normalized_name
    return int(block_text), role


def _assert_qat_target_weights_are_floating(
    targets: list[QwenImageBlockLinearTarget],
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
                raise TypeError(f"selected QAT target {target.normalized_name} has invalid bias")
            if bias.dtype not in SUPPORTED_FAKE_MXFP8_DTYPES:
                raise TypeError(
                    f"selected QAT target {target.normalized_name} must have a BF16/FP16/FP32 "
                    f"bias tensor before fake-MXFP8 injection, got {bias.dtype}"
                )


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


def _validate_training_config(config: QwenImageQatTrainingConfig) -> None:
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    beta1, beta2 = config.betas
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("optimizer betas must be in [0, 1)")
    if config.eps <= 0.0:
        raise ValueError("optimizer eps must be positive")
    if config.log_interval_steps <= 0:
        raise ValueError("log_interval_steps must be positive")
    if config.sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    if config.sample_start_index < 0:
        raise ValueError("sample_start_index must be non-negative")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if config.warmup_steps > 0 and not config.warmup_target_layers:
        raise ValueError("warmup_target_layers must be non-empty when warmup_steps > 0")
    if config.scale_multipliers is not None:
        invalid = [
            name
            for name, multiplier in config.scale_multipliers.items()
            if float(multiplier) <= 0.0
        ]
        if invalid:
            raise ValueError(f"scale_multipliers must be positive for {invalid[:8]}")


def _validate_rollout_training_config(config: QwenImageRolloutQatTrainingConfig) -> None:
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    beta1, beta2 = config.betas
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("optimizer betas must be in [0, 1)")
    if config.eps <= 0.0:
        raise ValueError("optimizer eps must be positive")
    if config.grad_clip_norm < 0.0:
        raise ValueError("grad_clip_norm must be non-negative")
    if config.log_interval_steps <= 0:
        raise ValueError("log_interval_steps must be positive")
    if config.checkpoint_interval_steps is not None and config.checkpoint_interval_steps <= 0:
        raise ValueError("checkpoint_interval_steps must be positive when set")
    if config.window_stride <= 0:
        raise ValueError("window_stride must be positive")
    if config.window_start_index < 0:
        raise ValueError("window_start_index must be non-negative")
    if config.lora_rank <= 0:
        raise ValueError("lora_rank must be positive")
    if config.lora_alpha <= 0.0:
        raise ValueError("lora_alpha must be positive")
    if config.teacher_target_mode not in ("on_policy", "captured_tuple"):
        raise ValueError("teacher_target_mode must be 'on_policy' or 'captured_tuple'")
    expected_num_layers = _optional_int(config.expected_num_layers)
    expected_target_count = _optional_int(config.expected_target_count)
    if (
        expected_num_layers != FORMAL_QWEN_IMAGE_BLOCK_LAYER_COUNT
        or expected_target_count != FORMAL_QWEN_IMAGE_BLOCK_LINEAR_TARGET_COUNT
    ) and not config.diagnostic_only:
        raise ValueError(
            "closed-set rollout QAT formal runs must target all "
            f"{FORMAL_QWEN_IMAGE_BLOCK_LAYER_COUNT} layers and "
            f"{FORMAL_QWEN_IMAGE_BLOCK_LINEAR_TARGET_COUNT} block Linears; "
            "set diagnostic_only=true "
            "for explicit tiny/partial test runs"
        )
    _validate_rollout_loss_config(config.loss)
    _validate_closed_set_rollout_timestep_weights(config.loss.timestep_weights)
    if config.scale_multipliers is not None:
        invalid = [
            name
            for name, multiplier in config.scale_multipliers.items()
            if float(multiplier) <= 0.0
        ]
        if invalid:
            raise ValueError(f"scale_multipliers must be positive for {invalid[:8]}")


def _validate_closed_set_rollout_timestep_weights(
    timestep_weights: Mapping[str, float] | None,
) -> None:
    if timestep_weights is None:
        raise ValueError("closed-set rollout timestep_weights must be set")
    late_mid_weight = float(timestep_weights.get("late_mid", 0.0))
    late_weight = float(timestep_weights.get("late", 0.0))
    if late_mid_weight < 4.0 or late_mid_weight > 8.0 or late_weight < 4.0 or late_weight > 8.0:
        raise ValueError(
            "closed-set rollout late timestep weights must keep late bins in the 4x-8x range"
        )


def _build_training_record(
    *,
    step: int,
    sample_index: int,
    sample: QwenImageTupleSample,
    loss: torch.Tensor,
    components: Mapping[str, torch.Tensor],
    grad_norm: float,
    trainable_parameters: list[nn.Parameter],
    transformer: nn.Module,
    elapsed_seconds: float,
    compute_lora_delta_norm: bool,
    warmup_active: bool,
) -> dict[str, object]:
    record: dict[str, object] = {
        "step": step,
        "sample_index": sample_index,
        "prompt_id": sample.prompt_id,
        "split": sample.split,
        "timestep_index": sample.timestep_index,
        "timestep_bin": sample.timestep_bin,
        "cfg_branch": sample.cfg_branch,
        "loss_total": _tensor_scalar(loss),
        "loss_unweighted_total": _tensor_scalar(components["unweighted_total"]),
        "loss_mse": _optional_component_scalar(components, "mse"),
        "loss_direction": _optional_component_scalar(components, "direction"),
        "timestep_weight": _tensor_scalar(components["timestep_weight"]),
        "grad_norm": grad_norm,
        "trainable_parameter_norm": _parameter_global_norm(trainable_parameters),
        "elapsed_seconds": float(elapsed_seconds),
        "warmup_active": warmup_active,
    }
    if compute_lora_delta_norm:
        record["lora_delta_norm"] = _lora_delta_global_norm(transformer)
    return record


def _build_rollout_training_record(
    *,
    step: int,
    window_index: int,
    window: tuple[QwenImageTupleSample, ...],
    loss: torch.Tensor,
    components: Mapping[str, torch.Tensor],
    rollout_records: list[dict[str, object]],
    grad_norm: float,
    trainable_parameters: list[nn.Parameter],
    transformer: nn.Module,
    elapsed_seconds: float,
    compute_lora_delta_norm: bool,
) -> dict[str, object]:
    first_sample = window[0]
    last_sample = window[-1]
    record: dict[str, object] = {
        "step": step,
        "window_index": window_index,
        "prompt_id": first_sample.prompt_id,
        "split": first_sample.split,
        "start_timestep_index": first_sample.timestep_index,
        "end_timestep_index": last_sample.timestep_index,
        "rollout_steps": _tensor_scalar(components["rollout_steps"]),
        "loss_total": _tensor_scalar(loss),
        "loss_output_mse": _tensor_scalar(components["output_mse"]),
        "loss_cond_output_mse": _tensor_scalar(components["cond_output_mse"]),
        "loss_negative_output_mse": _tensor_scalar(components["negative_output_mse"]),
        "loss_direction": _tensor_scalar(components["direction"]),
        "loss_cond_direction": _tensor_scalar(components["cond_direction"]),
        "loss_negative_direction": _tensor_scalar(components["negative_direction"]),
        "loss_latent_mse": _tensor_scalar(components["latent_mse"]),
        "loss_anchor_mse": _tensor_scalar(components["anchor_mse"]),
        "loss_unweighted_total": _tensor_scalar(components["unweighted_total"]),
        "loss_timestep_weighted_total": _tensor_scalar(components["timestep_weighted_total"]),
        "first_timestep_weight": float(rollout_records[0]["timestep_weight"]),
        "grad_norm": grad_norm,
        "trainable_parameter_norm": _parameter_global_norm(trainable_parameters),
        "elapsed_seconds": float(elapsed_seconds),
        "rollout_records": rollout_records,
    }
    if compute_lora_delta_norm:
        record["lora_delta_norm"] = _lora_delta_global_norm(transformer)
    return record


def _save_qwen_image_qat_checkpoint(
    path: Path,
    *,
    transformer: nn.Module,
    config: QwenImageQatTrainingConfig,
    injections: tuple[QwenImageQatInjectionInfo, ...],
    train_steps: int,
) -> None:
    _save_qwen_image_qat_checkpoint_from_config(
        path,
        transformer=transformer,
        config=_training_config_to_dict(config),
        injections=injections,
        train_steps=train_steps,
    )


def _save_qwen_image_qat_checkpoint_from_config(
    path: Path,
    *,
    transformer: nn.Module,
    config: Mapping[str, object],
    injections: tuple[QwenImageQatInjectionInfo, ...],
    train_steps: int,
) -> None:
    trainable_names = qwen_image_qat_trainable_parameter_names(injections)
    trainable_state_dict = {
        normalize_qwen_image_qat_parameter_name(name): parameter.detach().cpu()
        for name, parameter in transformer.named_parameters()
        if normalize_qwen_image_qat_parameter_name(name) in trainable_names
    }
    if trainable_names and set(trainable_state_dict) != trainable_names:
        missing = sorted(trainable_names - set(trainable_state_dict))[:20]
        raise RuntimeError(f"QAT checkpoint is missing trainable tensors: {missing}")
    checkpoint = {
        "format": TRAINING_FORMAT,
        "config": dict(config),
        "train_steps": int(train_steps),
        "trainable_state_dict": trainable_state_dict,
        "trainable_parameter_names": sorted(trainable_state_dict),
        "injections": [_injection_to_dict(injection) for injection in injections],
    }
    torch.save(checkpoint, path)


def _training_config_to_dict(config: QwenImageQatTrainingConfig) -> dict[str, object]:
    return {
        "tuple_index_jsonl": str(config.tuple_index_jsonl),
        "output_dir": str(config.output_dir),
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "betas": list(config.betas),
        "eps": config.eps,
        "device": config.device,
        "target_layers": list(config.target_layers),
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "expected_num_layers": config.expected_num_layers,
        "expected_target_count": config.expected_target_count,
        "loss": {
            "lambda_mse": config.loss.lambda_mse,
            "lambda_dir": config.loss.lambda_dir,
            "timestep_weights": dict(config.loss.timestep_weights or {}),
        },
        "log_interval_steps": config.log_interval_steps,
        "checkpoint_name": config.checkpoint_name,
        "metrics_name": config.metrics_name,
        "compute_lora_delta_norm": config.compute_lora_delta_norm,
        "scale_multipliers": dict(config.scale_multipliers or {}),
        "warmup_steps": config.warmup_steps,
        "warmup_target_layers": list(config.warmup_target_layers),
        "optimizer_foreach": config.optimizer_foreach,
        "sample_stride": config.sample_stride,
        "sample_start_index": config.sample_start_index,
    }


def _rollout_training_config_to_dict(
    config: QwenImageRolloutQatTrainingConfig,
) -> dict[str, object]:
    return {
        "recipe": "closed_set_rollout_qat_v1",
        "tuple_index_jsonl": str(config.tuple_index_jsonl),
        "output_dir": str(config.output_dir),
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "betas": list(config.betas),
        "eps": config.eps,
        "grad_clip_norm": config.grad_clip_norm,
        "device": config.device,
        "target_layers": list(config.target_layers),
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "expected_num_layers": config.expected_num_layers,
        "expected_target_count": config.expected_target_count,
        "diagnostic_only": config.diagnostic_only,
        "teacher_target_mode": config.teacher_target_mode,
        "loss": _rollout_loss_config_to_dict(config.loss),
        "log_interval_steps": config.log_interval_steps,
        "checkpoint_name": config.checkpoint_name,
        "checkpoint_interval_steps": config.checkpoint_interval_steps,
        "metrics_name": config.metrics_name,
        "compute_lora_delta_norm": config.compute_lora_delta_norm,
        "scale_multipliers": dict(config.scale_multipliers or {}),
        "optimizer_foreach": config.optimizer_foreach,
        "window_stride": config.window_stride,
        "window_start_index": config.window_start_index,
    }


def _rollout_interval_checkpoint_path(
    *,
    output_dir: Path,
    checkpoint_name: str,
    step: int,
) -> Path:
    checkpoint_path = Path(checkpoint_name)
    return output_dir / f"{checkpoint_path.stem}_step{int(step):04d}{checkpoint_path.suffix}"


def _injection_to_dict(injection: QwenImageQatInjectionInfo) -> dict[str, object]:
    return {
        "module_name": injection.module_name,
        "block_index": injection.block_index,
        "role": injection.role,
        "trainable_parameter_names": list(injection.trainable_parameter_names),
        "lora_rank": injection.lora_rank,
        "lora_alpha": injection.lora_alpha,
        "scale_multiplier": injection.scale_multiplier,
    }


def _optional_component_scalar(
    components: Mapping[str, torch.Tensor],
    name: str,
) -> float | None:
    tensor = components.get(name)
    return None if tensor is None else _tensor_scalar(tensor)


def _tensor_scalar(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().cpu())


def _parameter_global_norm(
    parameters: list[nn.Parameter],
    *,
    grad: bool = False,
) -> float:
    total = 0.0
    for parameter in parameters:
        tensor = parameter.grad if grad else parameter
        if tensor is None:
            continue
        total += float(tensor.detach().float().square().sum().cpu())
    return total**0.5


def _lora_delta_global_norm(transformer: nn.Module) -> float:
    total = 0.0
    for module in transformer.modules():
        if not isinstance(module, Mxfp8LoraAdapterLinear):
            continue
        total += float(module.lora_delta_weight().detach().float().square().sum().cpu())
    return total**0.5


def _run_probe_command(args: argparse.Namespace) -> None:
    run_qwen_image_qat_probe(
        recipe=args.recipe,
        model=args.model,
        visual_gen_args=Path(args.visual_gen_args),
        tuple_index_jsonl=Path(args.tuple_index_jsonl),
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        expected_num_layers=args.expected_num_layers,
        expected_target_count=args.expected_target_count,
        log_interval_steps=args.log_interval_steps,
        scale_clip_min=args.scale_clip_min,
        scale_clip_max=args.scale_clip_max,
        sample_stride=args.sample_stride,
        sample_start_index=args.sample_start_index,
        summary_json=Path(args.summary_json) if args.summary_json else None,
        provenance=_build_probe_runtime_provenance(args),
    )


def _run_monitor_command(args: argparse.Namespace) -> None:
    max_samples = None if args.max_samples == 0 else args.max_samples
    run_qwen_image_qat_checkpoint_monitor(
        model=args.model,
        visual_gen_args=Path(args.visual_gen_args),
        checkpoint_path=Path(args.checkpoint_path),
        tuple_index_jsonl=Path(args.tuple_index_jsonl),
        output_json=Path(args.output_json),
        records_jsonl=Path(args.records_jsonl) if args.records_jsonl else None,
        max_samples=max_samples,
        sample_stride=args.sample_stride,
        sample_start_index=args.sample_start_index,
        device=args.device,
        provenance=_build_probe_runtime_provenance(args),
    )


def _run_rgb_eval_command(args: argparse.Namespace) -> None:
    max_prompts = None if args.max_prompts == 0 else args.max_prompts
    run_qwen_image_qat_checkpoint_rgb_eval(
        prompt_manifest_jsonl=Path(args.prompt_manifest_jsonl),
        split=args.split,
        model=args.model,
        visual_gen_args=Path(args.visual_gen_args),
        checkpoint_path=Path(args.checkpoint_path),
        reference_root=Path(args.reference_root),
        output_root=Path(args.output_root),
        metrics_json=Path(args.metrics_json),
        variant=args.variant,
        reference_variant=args.reference_variant,
        device=args.device,
        max_prompts=max_prompts,
        provenance=_build_probe_runtime_provenance(args),
    )


def _run_augment_rollout_tuples_command(args: argparse.Namespace) -> None:
    run_qwen_image_rollout_tuple_augmentation(
        source_tuple_index_jsonl=Path(args.source_tuple_index_jsonl),
        rollout_metadata_jsonl=Path(args.rollout_metadata_jsonl),
        output_tuple_root=Path(args.output_tuple_root),
        output_tuple_index_jsonl=Path(args.output_tuple_index_jsonl),
        summary_json=Path(args.summary_json),
        provenance=_build_probe_runtime_provenance(args),
    )


def _run_rollout_qat_command(args: argparse.Namespace) -> None:
    run_qwen_image_rollout_qat(
        config_path=Path(args.config),
        model=args.model,
        visual_gen_args=Path(args.visual_gen_args),
        teacher_model=args.teacher_model,
        teacher_visual_gen_args=Path(args.teacher_visual_gen_args)
        if args.teacher_visual_gen_args
        else None,
        summary_json=Path(args.summary_json) if args.summary_json else None,
        provenance=_build_probe_runtime_provenance(args),
    )


def _validate_rollout_command_provenance(
    provenance: Mapping[str, object] | None,
    *,
    command_name: str,
) -> dict[str, object]:
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{command_name} requires rollout command provenance")
    data = dict(provenance)
    required_fields = (
        "cluster_alias",
        "container_runtime",
        "enroot_image",
        "git_head",
        "model_snapshot_path",
        "closed_set_target_manifest",
        "scheduler_metadata_source",
        "reference_image_root",
        "node_list",
    )
    missing = [field_name for field_name in required_fields if data.get(field_name) in (None, "")]
    if missing:
        raise ValueError(f"{command_name} provenance missing required fields {missing}")
    if data["cluster_alias"] != DEFAULT_CLUSTER_ALIAS:
        raise ValueError(f"{command_name} provenance cluster_alias must be {DEFAULT_CLUSTER_ALIAS}")
    if data["container_runtime"] != DEFAULT_CONTAINER_RUNTIME:
        raise ValueError(
            f"{command_name} provenance container_runtime must be {DEFAULT_CONTAINER_RUNTIME}"
        )
    if data["enroot_image"] != DEFAULT_ENROOT_IMAGE:
        raise ValueError(f"{command_name} provenance enroot_image must be {DEFAULT_ENROOT_IMAGE}")
    if data.get("allocation_id") in (None, "") and data.get("job_id") in (None, ""):
        raise ValueError(f"{command_name} provenance requires allocation_id or job_id")
    return data


def _build_probe_runtime_provenance(args: argparse.Namespace) -> dict[str, object]:
    provenance = {
        "cluster_alias": args.cluster_alias,
        "allocation_id": args.allocation_id or os.environ.get("SSH_GW_ALLOC_ID"),
        "job_id": args.job_id or os.environ.get("SLURM_JOB_ID"),
        "node_list": args.node_list or os.environ.get("SLURM_NODELIST"),
        "container_runtime": DEFAULT_CONTAINER_RUNTIME,
        "enroot_image": args.enroot_image,
        "command": args.command or " ".join(sys.argv),
        "model_snapshot_path": args.model_snapshot_path,
        "git_head": git_commit(Path.cwd()),
        "teacher_attention_backend": args.teacher_attention_backend,
        "training_attention_backend": args.training_attention_backend,
        "evaluation_attention_backend": args.evaluation_attention_backend,
    }
    for attr_name in (
        "closed_set_target_manifest",
        "rollout_metadata_jsonl",
        "scheduler_metadata_source",
        "reference_image_root",
    ):
        value = getattr(args, attr_name, None)
        if value:
            provenance[attr_name] = str(value)
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    run_parser = subparsers.add_parser("run-probe")
    run_parser.add_argument("--recipe", required=True, choices=QWEN_IMAGE_QAT_PROBE_RECIPES)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--visual-gen-args", required=True)
    run_parser.add_argument("--tuple-index-jsonl", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--max-steps", required=True, type=int)
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    run_parser.add_argument("--weight-decay", type=float, default=0.0)
    run_parser.add_argument("--lora-rank", type=int, default=16)
    run_parser.add_argument("--lora-alpha", type=float, default=32.0)
    run_parser.add_argument("--lora-dropout", type=float, default=0.0)
    run_parser.add_argument("--expected-num-layers", type=int, default=60)
    run_parser.add_argument("--expected-target-count", type=int, default=840)
    run_parser.add_argument("--log-interval-steps", type=int, default=10)
    run_parser.add_argument("--scale-clip-min", type=float, default=0.25)
    run_parser.add_argument("--scale-clip-max", type=float, default=4.0)
    run_parser.add_argument("--sample-stride", type=int, default=17)
    run_parser.add_argument("--sample-start-index", type=int, default=0)
    run_parser.add_argument("--summary-json")
    run_parser.add_argument("--cluster-alias", default=DEFAULT_CLUSTER_ALIAS)
    run_parser.add_argument("--allocation-id")
    run_parser.add_argument("--job-id")
    run_parser.add_argument("--node-list")
    run_parser.add_argument("--enroot-image", default=DEFAULT_ENROOT_IMAGE)
    run_parser.add_argument("--model-snapshot-path")
    run_parser.add_argument("--teacher-attention-backend", default="TRTLLM_FP8")
    run_parser.add_argument("--training-attention-backend", default="TRTLLM_FP8")
    run_parser.add_argument("--evaluation-attention-backend", default="TRTLLM_FP8")
    run_parser.add_argument("--command")
    run_parser.set_defaults(func=_run_probe_command)

    monitor_parser = subparsers.add_parser("monitor-checkpoint")
    monitor_parser.add_argument("--model", required=True)
    monitor_parser.add_argument("--visual-gen-args", required=True)
    monitor_parser.add_argument("--checkpoint-path", required=True)
    monitor_parser.add_argument("--tuple-index-jsonl", required=True)
    monitor_parser.add_argument("--output-json", required=True)
    monitor_parser.add_argument("--records-jsonl")
    monitor_parser.add_argument("--max-samples", type=int, default=320)
    monitor_parser.add_argument("--sample-stride", type=int, default=17)
    monitor_parser.add_argument("--sample-start-index", type=int, default=0)
    monitor_parser.add_argument("--device", default="cuda:0")
    monitor_parser.add_argument("--cluster-alias", default=DEFAULT_CLUSTER_ALIAS)
    monitor_parser.add_argument("--allocation-id")
    monitor_parser.add_argument("--job-id")
    monitor_parser.add_argument("--node-list")
    monitor_parser.add_argument("--enroot-image", default=DEFAULT_ENROOT_IMAGE)
    monitor_parser.add_argument("--model-snapshot-path")
    monitor_parser.add_argument("--teacher-attention-backend", default="TRTLLM_FP8")
    monitor_parser.add_argument("--training-attention-backend", default="TRTLLM_FP8")
    monitor_parser.add_argument("--evaluation-attention-backend", default="TRTLLM_FP8")
    monitor_parser.add_argument("--command")
    monitor_parser.set_defaults(func=_run_monitor_command)

    rgb_parser = subparsers.add_parser("rgb-eval-checkpoint")
    rgb_parser.add_argument("--prompt-manifest-jsonl", required=True)
    rgb_parser.add_argument("--split", required=True)
    rgb_parser.add_argument("--model", required=True)
    rgb_parser.add_argument("--visual-gen-args", required=True)
    rgb_parser.add_argument("--checkpoint-path", required=True)
    rgb_parser.add_argument("--reference-root", required=True)
    rgb_parser.add_argument("--output-root", required=True)
    rgb_parser.add_argument("--metrics-json", required=True)
    rgb_parser.add_argument("--variant", required=True)
    rgb_parser.add_argument("--reference-variant", default="bf16_sage_fp8")
    rgb_parser.add_argument("--device", default="cuda:0")
    rgb_parser.add_argument("--max-prompts", type=int, default=0)
    rgb_parser.add_argument("--cluster-alias", default=DEFAULT_CLUSTER_ALIAS)
    rgb_parser.add_argument("--allocation-id")
    rgb_parser.add_argument("--job-id")
    rgb_parser.add_argument("--node-list")
    rgb_parser.add_argument("--enroot-image", default=DEFAULT_ENROOT_IMAGE)
    rgb_parser.add_argument("--model-snapshot-path")
    rgb_parser.add_argument("--teacher-attention-backend", default="TRTLLM_FP8")
    rgb_parser.add_argument("--training-attention-backend", default="TRTLLM_FP8")
    rgb_parser.add_argument("--evaluation-attention-backend", default="TRTLLM_FP8")
    rgb_parser.add_argument("--command")
    rgb_parser.set_defaults(func=_run_rgb_eval_command)

    augment_parser = subparsers.add_parser("augment-rollout-tuples")
    augment_parser.add_argument("--source-tuple-index-jsonl", required=True)
    augment_parser.add_argument("--rollout-metadata-jsonl", required=True)
    augment_parser.add_argument("--output-tuple-root", required=True)
    augment_parser.add_argument("--output-tuple-index-jsonl", required=True)
    augment_parser.add_argument("--summary-json", required=True)
    augment_parser.add_argument("--closed-set-target-manifest")
    augment_parser.add_argument("--scheduler-metadata-source")
    augment_parser.add_argument("--reference-image-root")
    augment_parser.add_argument("--cluster-alias", default=DEFAULT_CLUSTER_ALIAS)
    augment_parser.add_argument("--allocation-id")
    augment_parser.add_argument("--job-id")
    augment_parser.add_argument("--node-list")
    augment_parser.add_argument("--enroot-image", default=DEFAULT_ENROOT_IMAGE)
    augment_parser.add_argument("--model-snapshot-path")
    augment_parser.add_argument("--teacher-attention-backend", default="TRTLLM_FP8")
    augment_parser.add_argument("--training-attention-backend", default="TRTLLM_FP8")
    augment_parser.add_argument("--evaluation-attention-backend", default="TRTLLM_FP8")
    augment_parser.add_argument("--command")
    augment_parser.set_defaults(func=_run_augment_rollout_tuples_command)

    rollout_parser = subparsers.add_parser("run-rollout-qat")
    rollout_parser.add_argument("--config", required=True)
    rollout_parser.add_argument("--model", required=True)
    rollout_parser.add_argument("--visual-gen-args", required=True)
    rollout_parser.add_argument("--teacher-model")
    rollout_parser.add_argument("--teacher-visual-gen-args")
    rollout_parser.add_argument("--summary-json")
    rollout_parser.add_argument("--closed-set-target-manifest")
    rollout_parser.add_argument("--rollout-metadata-jsonl")
    rollout_parser.add_argument("--scheduler-metadata-source")
    rollout_parser.add_argument("--reference-image-root")
    rollout_parser.add_argument("--cluster-alias", default=DEFAULT_CLUSTER_ALIAS)
    rollout_parser.add_argument("--allocation-id")
    rollout_parser.add_argument("--job-id")
    rollout_parser.add_argument("--node-list")
    rollout_parser.add_argument("--enroot-image", default=DEFAULT_ENROOT_IMAGE)
    rollout_parser.add_argument("--model-snapshot-path")
    rollout_parser.add_argument("--teacher-attention-backend", default="TRTLLM_FP8")
    rollout_parser.add_argument("--training-attention-backend", default="TRTLLM_FP8")
    rollout_parser.add_argument("--evaluation-attention-backend", default="TRTLLM_FP8")
    rollout_parser.add_argument("--command")
    rollout_parser.set_defaults(func=_run_rollout_qat_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


__all__ = [
    "CLOSED_SET_ROLLOUT_TIMESTEP_WEIGHTS",
    "DEFAULT_TIMESTEP_WEIGHTS",
    "FakeMxfp8Linear",
    "MXFP8_BLOCK_SIZE",
    "Mxfp8LoraAdapterLinear",
    "MONITOR_SUMMARY_FORMAT",
    "PROBE_SUMMARY_FORMAT",
    "QWEN_BLOCK_LINEAR_TARGET",
    "QWEN_IMAGE_QAT_PROBE_RECIPES",
    "ROLLOUT_AUGMENTATION_SUMMARY_FORMAT",
    "ROLLOUT_METADATA_MANIFEST_FORMAT",
    "ROLLOUT_MONITOR_SUMMARY_FORMAT",
    "ROLLOUT_QAT_SUMMARY_FORMAT",
    "ROLLOUT_SCHEDULER_STEP_IMPLEMENTATION",
    "ROLLOUT_TUPLE_SCHEMA_VERSION",
    "QwenImageQatInjectionInfo",
    "QwenImageQatMonitorConfig",
    "QwenImageQatTrainingConfig",
    "QwenImageQatTrainingResult",
    "QwenImageRolloutLossConfig",
    "QwenImageRolloutQatTrainingConfig",
    "QwenImageRolloutQatTrainingResult",
    "QwenImageRolloutStepSamples",
    "QwenImageRolloutTupleAugmentationConfig",
    "QwenImageRolloutWindowDataset",
    "QwenImageTupleDataset",
    "QwenImageTupleLossConfig",
    "QwenImageTupleSample",
    "TRAINING_FORMAT",
    "build_parser",
    "build_qwen_image_qat_probe_config",
    "build_qwen_image_qat_scale_sensitivity_manifest",
    "augment_qwen_image_rollout_tuple_dataset",
    "build_qwen_image_rollout_tuple_payload",
    "combine_qwen_image_cfg_outputs",
    "compute_qwen_image_rollout_loss",
    "compute_qwen_image_rollout_step_loss",
    "compute_qwen_image_tuple_loss",
    "compute_scale_aware_lora_multipliers",
    "first_tensor_output",
    "forward_qwen_image_tuple",
    "load_qwen_image_qat_checkpoint",
    "load_qwen_image_rollout_metadata_manifest",
    "load_qwen_image_rollout_qat_config",
    "load_qwen_image_tuple_sample",
    "main",
    "monitor_qwen_image_rollout_no_grad",
    "monitor_qwen_image_qat_checkpoint",
    "normalize_qwen_image_qat_parameter_name",
    "prepare_qwen_image_qat_model",
    "qwen_image_rollout_scheduler_step",
    "qwen_image_qat_trainable_parameter_names",
    "run_qwen_image_rollout_qat",
    "run_qwen_image_rollout_tuple_augmentation",
    "run_qwen_image_qat_checkpoint_monitor",
    "run_qwen_image_qat_checkpoint_rgb_eval",
    "run_qwen_image_qat_probe",
    "summarize_qwen_image_qat_training_metrics",
    "train_qwen_image_rollout_qat",
    "train_qwen_image_qat",
    "validate_qwen_image_rollout_tuple_sample",
    "validate_qwen_image_qat_probe_recipe",
]


if __name__ == "__main__":
    main()
