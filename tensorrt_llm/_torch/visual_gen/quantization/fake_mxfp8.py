# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline fake MXFP8 helpers for VisualGen transformer QAT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

Mxfp8ScaleMode = Literal["fp32_block_scale", "e8m0_power2"]

MXFP8_BLOCK_SIZE = 128
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
SUPPORTED_FAKE_MXFP8_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
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
_QWEN_IMAGE_BLOCK_LINEAR_ORDER = {
    role: index for index, role in enumerate(QWEN_IMAGE_BLOCK_LINEAR_SUFFIXES)
}

_SCALE_EPS = 1.0e-12


@dataclass(frozen=True)
class QwenImageBlockLinearTarget:
    """A selected Qwen-Image transformer-block Linear module."""

    module_name: str
    normalized_name: str
    block_index: int
    role: str
    module: nn.Module


@dataclass(frozen=True)
class FakeMxfp8Injection:
    """A replacement performed by fake MXFP8 injection."""

    module_name: str
    wrapper: "FakeMxfp8Linear"


def normalize_qwen_module_name(name: str) -> str:
    """Normalize module names emitted before or after torch.compile wrapping."""
    normalized = name
    while "._orig_mod." in normalized:
        normalized = normalized.replace("._orig_mod.", ".")
    if normalized.startswith("_orig_mod."):
        normalized = normalized[len("_orig_mod.") :]
    return normalized


def _check_block_size(block_size: int) -> None:
    if block_size != MXFP8_BLOCK_SIZE:
        raise ValueError(
            f"MXFP8 fake quantization requires block_size={MXFP8_BLOCK_SIZE}, got {block_size}."
        )


def _check_scale_mode(scale_mode: Mxfp8ScaleMode) -> None:
    if scale_mode not in ("fp32_block_scale", "e8m0_power2"):
        raise ValueError(f"Unsupported MXFP8 scale mode: {scale_mode!r}.")


def _check_floating_tensor(tensor: torch.Tensor, *, tensor_name: str) -> None:
    if tensor.dtype not in SUPPORTED_FAKE_MXFP8_DTYPES:
        raise TypeError(
            f"{tensor_name} must use one of {SUPPORTED_FAKE_MXFP8_DTYPES}, got {tensor.dtype}."
        )


def _check_matrix(matrix: torch.Tensor, *, tensor_name: str, block_size: int) -> None:
    _check_block_size(block_size)
    _check_floating_tensor(matrix, tensor_name=tensor_name)
    if matrix.ndim != 2:
        raise ValueError(f"{tensor_name} must be 2D, got shape {tuple(matrix.shape)}.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(
            f"{tensor_name} dimensions must be non-empty, got shape {tuple(matrix.shape)}."
        )
    if matrix.shape[1] < block_size:
        raise ValueError(
            f"{tensor_name} in_features must be at least {block_size} for MXFP8 fake quantization, "
            f"got {matrix.shape[1]}."
        )


def _dequant_scales(amax: torch.Tensor, scale_mode: Mxfp8ScaleMode) -> torch.Tensor:
    scale = (amax / FP8_E4M3_MAX).clamp_min(_SCALE_EPS)
    if scale_mode == "e8m0_power2":
        return torch.exp2(torch.ceil(torch.log2(scale)))
    return scale


def _fake_quantize_rows(
    matrix: torch.Tensor,
    *,
    block_size: int,
    scale_mode: Mxfp8ScaleMode,
    tensor_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_matrix(matrix, tensor_name=tensor_name, block_size=block_size)
    _check_scale_mode(scale_mode)

    rows, cols = matrix.shape
    num_blocks = (cols + block_size - 1) // block_size
    padded_cols = num_blocks * block_size
    pad_cols = padded_cols - cols
    if pad_cols:
        padded = F.pad(matrix, (0, pad_cols))
    else:
        padded = matrix

    blocks = padded.float().reshape(rows, num_blocks, block_size)
    scales = _dequant_scales(blocks.abs().amax(dim=2), scale_mode)
    qblocks = (blocks / scales.unsqueeze(2)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    dequantized = qblocks.to(torch.float8_e4m3fn).to(torch.float32) * scales.unsqueeze(2)
    dequantized = dequantized.reshape(rows, padded_cols)[:, :cols].to(matrix.dtype)
    return dequantized, scales.to(torch.float32)


def fake_mxfp8_activation_quantize(
    activation: torch.Tensor,
    *,
    block_size: int = MXFP8_BLOCK_SIZE,
    scale_mode: Mxfp8ScaleMode = "fp32_block_scale",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake-quantize activations with 1x128 E4M3 block semantics."""
    if activation.ndim == 0:
        raise ValueError("activation must have at least one dimension.")
    in_features = activation.shape[-1]
    source = activation.detach()
    flat = source.reshape(-1, in_features)
    dequantized, scales = _fake_quantize_rows(
        flat,
        block_size=block_size,
        scale_mode=scale_mode,
        tensor_name="activation",
    )
    fake = activation + (dequantized.reshape_as(activation) - source).detach()
    return fake, scales


def fake_mxfp8_weight_quantize(
    weight: torch.Tensor,
    *,
    block_size: int = MXFP8_BLOCK_SIZE,
    scale_mode: Mxfp8ScaleMode = "fp32_block_scale",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake-quantize Linear weights with 128x128 E4M3 block semantics."""
    _check_matrix(weight, tensor_name="weight", block_size=block_size)
    _check_scale_mode(scale_mode)

    source = weight.detach()
    out_features, in_features = source.shape
    num_blocks_out = (out_features + block_size - 1) // block_size
    num_blocks_in = (in_features + block_size - 1) // block_size
    pad_out = num_blocks_out * block_size - out_features
    pad_in = num_blocks_in * block_size - in_features
    if pad_out or pad_in:
        padded = F.pad(source, (0, pad_in, 0, pad_out))
    else:
        padded = source

    rows_per_block = (
        padded.reshape(num_blocks_out, block_size, num_blocks_in, block_size)
        .permute(0, 2, 1, 3)
        .reshape(num_blocks_out * num_blocks_in, block_size * block_size)
    )
    rows_float = rows_per_block.float()
    scales = _dequant_scales(rows_float.abs().amax(dim=1), scale_mode)
    qrows = (rows_float / scales.unsqueeze(1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    dequantized_rows = qrows.to(torch.float8_e4m3fn).to(torch.float32) * scales.unsqueeze(1)
    dequantized_rows = dequantized_rows.to(weight.dtype)
    dequantized = (
        dequantized_rows.reshape(num_blocks_out, num_blocks_in, block_size, block_size)
        .permute(0, 2, 1, 3)
        .reshape(num_blocks_out * block_size, num_blocks_in * block_size)
    )[:out_features, :in_features].contiguous()
    fake = weight + (dequantized - source).detach()
    return fake, scales.reshape(num_blocks_out, num_blocks_in).to(torch.float32)


class FakeMxfp8Linear(nn.Module):
    """Offline fake MXFP8 wrapper for a BF16/FP16/FP32 Linear-like module."""

    def __init__(
        self,
        linear: nn.Module,
        *,
        module_name: str = "",
        block_size: int = MXFP8_BLOCK_SIZE,
        scale_mode: Mxfp8ScaleMode = "fp32_block_scale",
    ) -> None:
        super().__init__()
        _check_block_size(block_size)
        _check_scale_mode(scale_mode)
        if not hasattr(linear, "weight"):
            raise TypeError("FakeMxfp8Linear requires a module with a weight tensor.")
        weight = linear.weight
        if not isinstance(weight, torch.Tensor):
            raise TypeError("FakeMxfp8Linear requires linear.weight to be a torch.Tensor.")
        _check_matrix(weight, tensor_name="linear.weight", block_size=block_size)
        bias = getattr(linear, "bias", None)
        if bias is not None:
            if not isinstance(bias, torch.Tensor):
                raise TypeError(
                    "FakeMxfp8Linear requires linear.bias to be a torch.Tensor or None."
                )
            if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
                raise ValueError(
                    "FakeMxfp8Linear bias must be 1D with out_features elements, "
                    f"got bias shape {tuple(bias.shape)} and weight shape {tuple(weight.shape)}."
                )
            _check_floating_tensor(bias, tensor_name="linear.bias")

        self.linear = linear
        self.module_name = module_name
        self.block_size = block_size
        self.scale_mode = scale_mode
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return getattr(self.linear, "bias", None)

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        _check_floating_tensor(activation, tensor_name="activation")
        if activation.shape[-1] != self.in_features:
            raise ValueError(
                f"activation last dimension must be {self.in_features}, got {activation.shape[-1]}."
            )
        fake_activation, _ = fake_mxfp8_activation_quantize(
            activation,
            block_size=self.block_size,
            scale_mode=self.scale_mode,
        )
        fake_weight, _ = fake_mxfp8_weight_quantize(
            self.weight,
            block_size=self.block_size,
            scale_mode=self.scale_mode,
        )
        return F.linear(fake_activation, fake_weight, self.bias)


def _parse_qwen_block_role(normalized_name: str) -> tuple[int, str] | None:
    prefix = "transformer_blocks."
    if not normalized_name.startswith(prefix):
        return None
    remainder = normalized_name[len(prefix) :]
    block_text, sep, role = remainder.partition(".")
    if not sep or not block_text.isdigit():
        return None
    if role not in QWEN_IMAGE_BLOCK_LINEAR_SUFFIXES:
        return None
    return int(block_text), role


def _default_linear_cls() -> Type[nn.Module]:
    from tensorrt_llm._torch.modules.linear import Linear

    return Linear


def select_qwen_image_block_linears(
    module: nn.Module,
    *,
    expected_num_layers: int | None = None,
    expected_count: int | None = None,
    linear_cls: Type[nn.Module] | None = None,
) -> list[QwenImageBlockLinearTarget]:
    """Select fake-QAT target Linears from Qwen-Image transformer blocks."""
    linear_cls = linear_cls or _default_linear_cls()
    if expected_count is not None and expected_num_layers is not None:
        raise ValueError("Pass only one of expected_count or expected_num_layers.")
    if expected_count is None:
        num_layers = expected_num_layers
        if num_layers is None:
            num_layers = getattr(module, "num_layers", None)
        if num_layers is not None:
            expected_count = int(num_layers) * QWEN_IMAGE_BLOCK_LINEAR_COUNT

    targets: list[QwenImageBlockLinearTarget] = []
    seen: set[str] = set()
    for module_name, child in module.named_modules():
        if not module_name:
            continue
        normalized_name = normalize_qwen_module_name(module_name)
        parsed = _parse_qwen_block_role(normalized_name)
        if parsed is None:
            continue
        if not isinstance(child, linear_cls):
            continue
        if normalized_name in seen:
            raise ValueError(
                f"Duplicate Qwen block Linear target after normalization: {normalized_name}."
            )
        seen.add(normalized_name)
        block_index, role = parsed
        targets.append(
            QwenImageBlockLinearTarget(
                module_name=module_name,
                normalized_name=normalized_name,
                block_index=block_index,
                role=role,
                module=child,
            )
        )

    targets.sort(
        key=lambda target: (
            target.block_index,
            _QWEN_IMAGE_BLOCK_LINEAR_ORDER[target.role],
        )
    )
    if expected_count is not None and len(targets) != expected_count:
        raise ValueError(
            f"Expected {expected_count} Qwen block Linear targets, found {len(targets)}."
        )
    return targets


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
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = _get_child_module(parent, part)
    _set_child_module(parent, parts[-1], replacement)


def inject_fake_mxfp8_linears(
    module: nn.Module,
    *,
    expected_num_layers: int | None = None,
    expected_count: int | None = None,
    linear_cls: Type[nn.Module] | None = None,
    block_size: int = MXFP8_BLOCK_SIZE,
    scale_mode: Mxfp8ScaleMode = "fp32_block_scale",
) -> list[FakeMxfp8Injection]:
    """Replace selected Qwen block Linears with offline fake MXFP8 wrappers."""
    targets = select_qwen_image_block_linears(
        module,
        expected_num_layers=expected_num_layers,
        expected_count=expected_count,
        linear_cls=linear_cls,
    )
    injections: list[FakeMxfp8Injection] = []
    for target in targets:
        wrapper = FakeMxfp8Linear(
            target.module,
            module_name=target.normalized_name,
            block_size=block_size,
            scale_mode=scale_mode,
        )
        _replace_module(module, target.module_name, wrapper)
        injections.append(FakeMxfp8Injection(module_name=target.normalized_name, wrapper=wrapper))
    return injections
