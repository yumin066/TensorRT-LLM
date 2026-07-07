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

import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.visualgen_eval.qwen_image_capture_manifest import BF16_TEACHER_TRAJECTORY_SOURCE
from scripts.visualgen_eval.qwen_image_teacher_capture import (
    CAPTURED_TUPLE_STATUS,
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
    normalize_qwen_module_name = _fake_mxfp8.normalize_qwen_module_name
    select_qwen_image_block_linears = _fake_mxfp8.select_qwen_image_block_linears


TRAINING_FORMAT = "qwen_image_mxfp8_qat_adapter_v1"
QWEN_BLOCK_LINEAR_TARGET = "qwen_block_linears"
TARGET_OUTPUT_KEY = "target_output"
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
class QwenImageQatInjectionInfo:
    """Trainable LoRA fake-MXFP8 wrapper inserted into one selected Linear."""

    module_name: str
    block_index: int
    role: str
    trainable_parameter_names: tuple[str, ...]
    lora_rank: int
    lora_alpha: float


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


@dataclass(frozen=True)
class QwenImageQatTrainingResult:
    """Artifacts emitted by one tuple-level QAT training run."""

    output_dir: Path
    metrics_path: Path
    checkpoint_path: Path
    train_steps: int
    injections: tuple[QwenImageQatInjectionInfo, ...]


class Mxfp8LoraAdapterLinear(nn.Module):
    """Low-rank trainable residual around a frozen fake-MXFP8 Linear."""

    def __init__(
        self,
        base: FakeMxfp8Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0.0:
            raise ValueError("LoRA alpha must be positive")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")

        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
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
        return (delta * self.scaling).to(dtype=self.weight.dtype, device=self.weight.device)

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        base_output = self.base(activation)
        adapter_output = self.lora_up(self.lora_down(self.dropout(activation))) * self.scaling
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
    target_device = torch.device(device)
    forward_kwargs = {
        "hidden_states": sample.hidden_states.to(device=target_device),
        "encoder_hidden_states": sample.encoder_hidden_states.to(device=target_device),
        "encoder_hidden_states_mask": _move_value(sample.encoder_hidden_states_mask, target_device),
        "timestep": sample.timestep.to(device=target_device),
        "img_shapes": _move_value(sample.img_shapes, target_device),
        "txt_seq_lens": _move_value(sample.txt_seq_lens, target_device),
        "return_dict": False,
    }
    if sample.additional_t_cond is not None:
        forward_kwargs["additional_t_cond"] = sample.additional_t_cond.to(device=target_device)
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
    )

    metrics_path = output_dir / config.metrics_name
    checkpoint_path = output_dir / config.checkpoint_name
    start_time = time.perf_counter()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, config.max_steps + 1):
            sample = dataset[(step - 1) % len(dataset)]
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
                    sample=sample,
                    loss=loss,
                    components=components,
                    grad_norm=grad_norm,
                    trainable_parameters=trainable_parameters,
                    transformer=transformer,
                    elapsed_seconds=time.perf_counter() - start_time,
                    compute_lora_delta_norm=config.compute_lora_delta_norm,
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


def _tuple_path_from_entry(entry: Mapping[str, object], *, manifest_dir: Path) -> Path:
    value = entry.get("tuple_path")
    if not isinstance(value, str):
        raise ValueError("tuple index entry missing tuple_path")
    path = Path(value)
    return path if path.is_absolute() else manifest_dir / path


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


def _build_training_record(
    *,
    step: int,
    sample: QwenImageTupleSample,
    loss: torch.Tensor,
    components: Mapping[str, torch.Tensor],
    grad_norm: float,
    trainable_parameters: list[nn.Parameter],
    transformer: nn.Module,
    elapsed_seconds: float,
    compute_lora_delta_norm: bool,
) -> dict[str, object]:
    record: dict[str, object] = {
        "step": step,
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
        "config": _training_config_to_dict(config),
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
    }


def _injection_to_dict(injection: QwenImageQatInjectionInfo) -> dict[str, object]:
    return {
        "module_name": injection.module_name,
        "block_index": injection.block_index,
        "role": injection.role,
        "trainable_parameter_names": list(injection.trainable_parameter_names),
        "lora_rank": injection.lora_rank,
        "lora_alpha": injection.lora_alpha,
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


__all__ = [
    "FakeMxfp8Linear",
    "MXFP8_BLOCK_SIZE",
    "Mxfp8LoraAdapterLinear",
    "QWEN_BLOCK_LINEAR_TARGET",
    "QwenImageQatInjectionInfo",
    "QwenImageQatTrainingConfig",
    "QwenImageQatTrainingResult",
    "QwenImageTupleDataset",
    "QwenImageTupleLossConfig",
    "QwenImageTupleSample",
    "TRAINING_FORMAT",
    "compute_qwen_image_tuple_loss",
    "first_tensor_output",
    "forward_qwen_image_tuple",
    "load_qwen_image_tuple_sample",
    "normalize_qwen_image_qat_parameter_name",
    "prepare_qwen_image_qat_model",
    "qwen_image_qat_trainable_parameter_names",
    "train_qwen_image_qat",
]
