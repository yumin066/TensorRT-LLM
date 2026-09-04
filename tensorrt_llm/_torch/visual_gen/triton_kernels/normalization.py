# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fused tensor normalization and layout conversion for VisualGen video input."""

import torch
import triton
import triton.language as tl


@triton.jit
def _normalize_bthwc_to_ncthw_kernel(
    source,
    output,
    numel,
    frames: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    channels: tl.constexpr,
    block_size: tl.constexpr,
):
    output_offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = output_offsets < numel

    remaining = output_offsets
    pixel_x = remaining % width
    remaining //= width
    pixel_y = remaining % height
    remaining //= height
    frame = remaining % frames
    remaining //= frames
    channel = remaining % channels
    batch = remaining // channels

    source_offsets = (
        ((batch * frames + frame) * height + pixel_y) * width + pixel_x
    ) * channels + channel
    pixel = tl.load(source + source_offsets, mask=valid).to(tl.float32)
    normalized = pixel * (1.0 / 127.5) - 1.0
    tl.store(output + output_offsets, normalized, mask=valid)


def uint8_bthwc_to_normalized_ncthw(
    source: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert contiguous CUDA uint8 ``BTHWC`` video to normalized ``NCTHW``.

    The fused kernel maps values from ``[0, 255]`` to ``[-1, 1]`` and writes
    directly to the requested output dtype without a full-sized FP32 or layout
    conversion intermediate.
    """
    if source.device.type != "cuda":
        raise ValueError("uint8_bthwc_to_normalized_ncthw requires a CUDA input tensor.")
    if source.dtype != torch.uint8 or source.ndim != 5 or not source.is_contiguous():
        raise ValueError(
            "uint8_bthwc_to_normalized_ncthw expects contiguous uint8 [B, T, H, W, C] input."
        )
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"uint8_bthwc_to_normalized_ncthw does not support output dtype {dtype}.")

    batch, frames, height, width, channels = source.shape
    output = torch.empty(
        (batch, channels, frames, height, width),
        device=source.device,
        dtype=dtype,
    )
    block_size = 256
    _normalize_bthwc_to_ncthw_kernel[(triton.cdiv(output.numel(), block_size),)](
        source,
        output,
        output.numel(),
        frames=frames,
        height=height,
        width=width,
        channels=channels,
        block_size=block_size,
    )
    return output
