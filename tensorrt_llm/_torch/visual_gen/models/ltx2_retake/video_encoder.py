# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tiled native video VAE encoder used by LTX-2 retake."""

import logging
from dataclasses import replace

import torch

from ..ltx2.ltx2_core.types import SpatioTemporalScaleFactors, VideoLatentShape
from ..ltx2.ltx2_core.video_vae.enums import NormLayerType, PaddingModeType
from ..ltx2.ltx2_core.video_vae.tiling import (
    DEFAULT_MAPPING_OPERATION,
    DEFAULT_SPLIT_OPERATION,
    DimensionIntervals,
    Tile,
    TilingConfig,
    compute_rectangular_mask_1d,
    create_tiles,
)
from ..ltx2.ltx2_core.video_vae.video_vae import (
    VideoEncoder,
    make_mapping_operation,
    split_with_symmetric_overlaps,
)
from .video_decoder import RetakePerChannelStatistics

logger = logging.getLogger(__name__)

_MIN_SPATIAL_OVERLAP_PIXELS = 64
_MIN_TEMPORAL_OVERLAP_FRAMES = 16


def _split_temporal_frames(tile_size: int, overlap: int):
    non_causal_split = split_with_symmetric_overlaps(tile_size, overlap)

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= tile_size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        intervals = non_causal_split(dimension_size)
        ends = list(intervals.ends)
        ends[:-1] = [end + 1 for end in ends[:-1]]
        return replace(intervals, ends=ends, right_ramps=[0] * len(ends))

    return split


def _map_temporal_interval(begin, end, left_ramp, right_ramp, scale):
    start = begin // scale
    stop = (end - 1) // scale + 1
    left = 0 if left_ramp == 0 else 1 + (left_ramp - 1) // scale
    right = right_ramp // scale
    if right:
        raise ValueError(f"Retake encode tiles require a zero right ramp; got {right_ramp}")
    return slice(start, stop), compute_rectangular_mask_1d(stop - start, left, right)


def _map_spatial_interval(begin, end, left_ramp, right_ramp, scale):
    start = begin // scale
    stop = end // scale
    return slice(start, stop), compute_rectangular_mask_1d(
        stop - start,
        max(0, left_ramp // scale - 1),
        0 if right_ramp == 0 else 1,
    )


def _prepare_encode_tiles(
    video: torch.Tensor,
    tiling_config: TilingConfig,
    scales: SpatioTemporalScaleFactors,
) -> list[Tile]:
    splitters = [DEFAULT_SPLIT_OPERATION] * video.ndim
    mappers = [DEFAULT_MAPPING_OPERATION] * video.ndim
    if tiling_config.spatial_config is not None:
        config = tiling_config.spatial_config
        overlap = max(config.tile_overlap_in_pixels, _MIN_SPATIAL_OVERLAP_PIXELS)
        for axis, scale in ((3, scales.height), (4, scales.width)):
            splitters[axis] = split_with_symmetric_overlaps(config.tile_size_in_pixels, overlap)
            mappers[axis] = make_mapping_operation(_map_spatial_interval, scale=scale)
    if tiling_config.temporal_config is not None:
        config = tiling_config.temporal_config
        overlap = max(config.tile_overlap_in_frames, _MIN_TEMPORAL_OVERLAP_FRAMES)
        splitters[2] = _split_temporal_frames(config.tile_size_in_frames, overlap)
        mappers[2] = make_mapping_operation(_map_temporal_interval, scale=scales.time)
    return create_tiles(video.shape, splitters, mappers)


class RetakeVideoEncoder(VideoEncoder):
    """VideoEncoder with overlap-aware tiled encoding for full retake clips."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.per_channel_statistics = RetakePerChannelStatistics(latent_channels=self.out_channels)

    def tiled_encode(
        self,
        video: torch.Tensor,
        tiling_config: TilingConfig | None = None,
    ) -> torch.Tensor:
        if tiling_config is None:
            return self.forward(video)

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        scales = SpatioTemporalScaleFactors(time=8, width=32, height=32)
        batch, _, frames, height, width = video.shape
        remainder = (frames - 1) % scales.time
        if remainder:
            logger.warning("Cropping %d video frame(s) for causal VAE encode", remainder)
            video = video[:, :, :-remainder]
            frames = video.shape[2]

        latent_shape = VideoLatentShape(
            batch=batch,
            channels=self.out_channels,
            frames=(frames - 1) // scales.time + 1,
            height=height // scales.height,
            width=width // scales.width,
        )
        latents = torch.zeros(latent_shape.to_torch_shape(), device=device, dtype=dtype)
        weights = torch.zeros_like(latents)
        for tile in _prepare_encode_tiles(video, tiling_config, scales):
            video_tile = video[tile.in_coords].to(device=device, dtype=dtype)
            latent_tile = self.forward(video_tile)
            mask = tile.blend_mask(device, dtype)
            latents[tile.out_coords] += latent_tile * mask
            weights[tile.out_coords] += mask
        return latents / weights.clamp(min=1e-8)


class RetakeVideoEncoderConfigurator:
    @classmethod
    def from_config(cls, config: dict) -> RetakeVideoEncoder:
        config = config.get("vae", {})
        return RetakeVideoEncoder(
            convolution_dimensions=config.get("dims", 3),
            in_channels=config.get("out_channels", 3),
            out_channels=config.get("latent_channels", 128),
            encoder_blocks=config.get("encoder_blocks", []),
            patch_size=config.get("patch_size", 4),
            norm_layer=NormLayerType(config.get("norm_layer", "pixel_norm")),
            causal=config.get("causal_encoder", True),
            timestep_conditioning=False,
            encoder_spatial_padding_mode=PaddingModeType(
                config.get(
                    "spatial_padding_mode",
                    config.get("encoder_spatial_padding_mode", "zeros"),
                )
            ),
        )
