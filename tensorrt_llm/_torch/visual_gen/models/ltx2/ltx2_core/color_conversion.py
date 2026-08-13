# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""RGB -> YUV420p colour conversion for LTX-2 video writeout.

The encoder must be handed frames in the colour space it tags them with. Passing
RGB24 to PyAV and letting libswscale pick the conversion, while separately
tagging the stream BT.709, produces a file whose pixels and whose metadata
disagree -- worse than leaving the tags unset, because every downstream decoder
then trusts the wrong matrix.

So the conversion is done here explicitly and the same constants drive the stream
tags, which is how the reference implementation does it.
"""

from __future__ import annotations

import torch

# FFmpeg enum values, used for `codec_context.colorspace` / `.color_range`.
AVCOL_SPC_BT709 = 1
AVCOL_RANGE_MPEG = 1  # "tv" / limited range

# BT.709 full-range RGB -> YUV matrix. Y in [0,1], U/V centred on 0.
_BT709_RGB_TO_YUV = torch.tensor(
    [
        [0.2126, 0.7152, 0.0722],
        [-0.1146, -0.3854, 0.5000],
        [0.5000, -0.4542, -0.0458],
    ],
    dtype=torch.float32,
)


def rgb_to_yuv420p_bt709_limited(frames_uint8: torch.Tensor) -> torch.Tensor:
    """``(T, H, W, 3)`` uint8 RGB -> ``(T, H*3//2, W)`` uint8 packed I420.

    BT.709 primaries, MPEG (limited/"tv") range: Y scaled to [16, 235] and chroma
    to [16, 240]. Chroma is subsampled by averaging 2x2 blocks.

    ``H`` and ``W`` must be even, which the retake window (1280x704) satisfies.
    """
    if frames_uint8.dim() != 4 or frames_uint8.shape[-1] != 3:
        raise ValueError(f"expected (T, H, W, 3) RGB frames; got {tuple(frames_uint8.shape)}")
    height, width = frames_uint8.shape[1], frames_uint8.shape[2]
    if height % 2 or width % 2:
        raise ValueError(f"I420 needs even H and W; got {height}x{width}")

    # Convert in fp32 on whatever device the frames are on.
    rgb = frames_uint8.to(torch.float32).div_(255.0)  # (T, H, W, 3) in [0, 1]
    mat = _BT709_RGB_TO_YUV.to(device=rgb.device, dtype=rgb.dtype)
    yuv = rgb @ mat.T  # (T, H, W, 3)

    y = yuv[..., 0]  # (T, H, W), [0, 1]
    uv = yuv[..., 1:].permute(0, 3, 1, 2).contiguous()  # (T, 2, H, W), centred on 0
    uv = torch.nn.functional.avg_pool2d(uv, kernel_size=2, stride=2)  # (T, 2, H/2, W/2)

    # Limited-range scaling, matching the reference's `apply_color_range_`.
    y = y.mul(219.0).add_(16.0)
    uv = uv.mul(224.0).add_(128.0)

    # I420 packing: the Y plane, then U rows and V rows each packed two-per-row.
    uv_packed = uv.reshape(uv.shape[0], uv.shape[2], uv.shape[3] * 2)  # (T, H/2, W)
    packed = torch.cat([y, uv_packed], dim=1)  # (T, H*3//2, W)
    # Truncate, do not round. `.to(uint8)` truncates and that is what the
    # reference does; rounding here would be marginally more accurate in
    # isolation but would introduce a systematic half-level offset against every
    # frame the reference produces.
    return packed.clamp_(0, 255).to(torch.uint8)
