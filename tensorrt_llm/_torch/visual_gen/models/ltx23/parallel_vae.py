# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tile-parallel VAE encode for LTX-2.3 Retake.

The Retake encoder already evaluates overlapping spatiotemporal tiles as
independent forwards.  This module distributes those forwards across the VAE
process group and reduces only the final blend numerator and denominator.
Unlike layer-wise spatial sharding, it requires no Conv3d halo exchange.
"""

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from tensorrt_llm.logger import logger

from ..ltx2.parallel_vae import assign_tiles_lpt
from .ltx23_core.video_vae_ltx23 import VIDEO_SCALE_FACTORS, VideoLatentShape, _prepare_encode_tiles

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from ..ltx2.ltx2_core.video_vae.tiling import TilingConfig
    from .ltx23_core.video_vae_ltx23 import LTX23VideoEncoder


def tile_parallel_encode(
    video_encoder: "LTX23VideoEncoder",
    video: torch.Tensor,
    tiling_config: "TilingConfig",
    pg: "ProcessGroup",
) -> torch.Tensor:
    """Distributed equivalent of ``LTX23VideoEncoder.tiled_encode``.

    Every rank in ``pg`` must hold the full input video and an identical encoder
    and must enter this function collectively.  Tiles are assigned with the same
    longest-processing-time policy as LTX tile-parallel decode.  The returned
    full latent is available on every participating rank.
    """
    if pg is None:
        raise ValueError(
            "tile_parallel_encode requires a valid VAE process group, got pg=None "
            "(a None group would fall back to the world group)."
        )

    rank = dist.get_rank(pg)
    world = dist.get_world_size(pg)
    device = next(video_encoder.parameters()).device
    dtype = next(video_encoder.parameters()).dtype
    scales = VIDEO_SCALE_FACTORS
    batch, _, frames, height, width = video.shape
    remainder = (frames - 1) % scales.time
    if remainder:
        logger.warning("Cropping %d video frame(s) for causal VAE encode", remainder)
        video = video[:, :, :-remainder]
        frames = video.shape[2]

    tiles = _prepare_encode_tiles(video, tiling_config, scales)
    mine = assign_tiles_lpt(tiles, world, rank)
    latent_shape = VideoLatentShape(
        batch=batch,
        channels=video_encoder.out_channels,
        frames=(frames - 1) // scales.time + 1,
        height=height // scales.height,
        width=width // scales.width,
    ).to_torch_shape()
    latents = torch.zeros(latent_shape, device=device, dtype=dtype)
    weight_shape = list(latent_shape)
    weight_shape[1] = 1
    weights = torch.zeros(weight_shape, device=device, dtype=dtype)

    for tile in mine:
        latent_tile = video_encoder(video[tile.in_coords].to(device=device, dtype=dtype))
        mask = tile.blend_mask(device, dtype)
        latents[tile.out_coords] += latent_tile * mask
        weights[tile.out_coords] += mask

    dist.all_reduce(latents, group=pg)
    dist.all_reduce(weights, group=pg)
    return latents / weights.clamp_min(1.0e-8)
