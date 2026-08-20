# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LTX-2.3 video decoder extensions used by the retake workflow."""

from typing import Any

import torch
from torch import nn

from ..ltx2.ltx2_core.normalization import PixelNorm
from ..ltx2.ltx2_core.timestep_embedding import PixArtAlphaCombinedTimestepSizeEmbeddings
from ..ltx2.ltx2_core.types import SpatioTemporalScaleFactors
from ..ltx2.ltx2_core.video_vae.convolution import make_conv_nd
from ..ltx2.ltx2_core.video_vae.enums import NormLayerType, PaddingModeType
from ..ltx2.ltx2_core.video_vae.sampling import DepthToSpaceUpsample
from ..ltx2.ltx2_core.video_vae.video_vae import VideoDecoder, _make_decoder_block


class RetakePerChannelStatistics(nn.Module):
    """The two video latent statistics present in the LTX-2.3 checkpoint."""

    def __init__(self, latent_channels: int = 128):
        super().__init__()
        self.register_buffer("std-of-means", torch.empty(latent_channels))
        self.register_buffer("mean-of-means", torch.empty(latent_channels))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)
        std = self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)
        return (x - mean) / std

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)
        std = self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)
        return x * std + mean


def _make_retake_decoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    timestep_conditioning: bool,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> tuple[nn.Module, int]:
    if block_name in ("compress_time", "compress_space"):
        multiplier = block_config.get("multiplier", 1)
        stride = (2, 1, 1) if block_name == "compress_time" else (1, 2, 2)
        return (
            DepthToSpaceUpsample(
                dims=convolution_dimensions,
                in_channels=in_channels,
                stride=stride,
                out_channels_reduction_factor=multiplier,
                spatial_padding_mode=spatial_padding_mode,
            ),
            in_channels // multiplier,
        )
    return _make_decoder_block(
        block_name=block_name,
        block_config=block_config,
        in_channels=in_channels,
        convolution_dimensions=convolution_dimensions,
        norm_layer=norm_layer,
        timestep_conditioning=timestep_conditioning,
        norm_num_groups=norm_num_groups,
        spatial_padding_mode=spatial_padding_mode,
    )


class RetakeVideoDecoder(VideoDecoder):
    """Native decoder matching the LTX-2.3 retake checkpoint layout."""

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 128,
        out_channels: int = 3,
        decoder_blocks: list[tuple[str, int | dict]] | None = None,
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = False,
        timestep_conditioning: bool = False,
        decoder_spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
        base_channels: int = 128,
    ):
        nn.Module.__init__(self)
        decoder_blocks = decoder_blocks or []
        self.video_downscale_factors = SpatioTemporalScaleFactors(time=8, width=32, height=32)
        self.patch_size = patch_size
        out_channels = out_channels * patch_size**2
        self.causal = causal
        self.timestep_conditioning = timestep_conditioning
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS
        self.per_channel_statistics = RetakePerChannelStatistics(latent_channels=in_channels)
        self.decode_noise_scale = 0.025
        self.decode_timestep = 0.05

        feature_channels = base_channels * 8
        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )
        self.up_blocks = nn.ModuleList()
        for block_name, block_params in reversed(decoder_blocks):
            block_config = (
                {"num_layers": block_params} if isinstance(block_params, int) else block_params
            )
            block, feature_channels = _make_retake_decoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                timestep_conditioning=timestep_conditioning,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=decoder_spatial_padding_mode,
            )
            self.up_blocks.append(block)

        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=feature_channels,
                num_groups=self._norm_num_groups,
                eps=1e-6,
            )
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()
        else:
            raise ValueError(f"Unsupported decoder norm layer: {norm_layer}")
        self.conv_act = nn.SiLU()
        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )
        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(torch.tensor(1000.0))
            self.last_time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                embedding_dim=feature_channels * 2,
                size_emb_dim=0,
            )
            self.last_scale_shift_table = nn.Parameter(torch.empty(2, feature_channels))


class RetakeVideoDecoderConfigurator:
    @classmethod
    def from_config(cls, config: dict) -> RetakeVideoDecoder:
        config = config.get("vae", {})
        return RetakeVideoDecoder(
            convolution_dimensions=config.get("dims", 3),
            in_channels=config.get("latent_channels", 128),
            out_channels=config.get("out_channels", 3),
            decoder_blocks=config.get("decoder_blocks", []),
            patch_size=config.get("patch_size", 4),
            norm_layer=NormLayerType(config.get("norm_layer", "pixel_norm")),
            causal=config.get("causal_decoder", False),
            timestep_conditioning=config.get("timestep_conditioning", True),
            decoder_spatial_padding_mode=PaddingModeType(
                config.get(
                    "spatial_padding_mode",
                    config.get("decoder_spatial_padding_mode", "reflect"),
                )
            ),
            base_channels=config.get("decoder_base_channels", 128),
        )
