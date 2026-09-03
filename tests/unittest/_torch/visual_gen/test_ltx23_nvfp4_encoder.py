# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.video_vae.enums import (
    NormLayerType,
    PaddingModeType,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.video_vae.resnet import ResnetBlock3D
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.video_vae.sampling import (
    SpaceToDepthDownsample,
)
from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core import (
    video_vae_ltx23_nvfp4 as nvfp4_encoder,
)


class _ResnetContainer(nn.Module):
    def __init__(self, blocks: list[ResnetBlock3D]) -> None:
        super().__init__()
        self.res_blocks = nn.ModuleList(blocks)


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_in = nn.Identity()
        self.down_blocks = nn.ModuleList(
            [
                _ResnetContainer([_block(), _block()]),
                _downsample(),
                _ResnetContainer([_block()]),
            ]
        )
        self.conv_out = nn.Identity()


def _block() -> ResnetBlock3D:
    return ResnetBlock3D(
        dims=3,
        in_channels=128,
        out_channels=128,
        dropout=0.0,
        norm_layer=NormLayerType.PIXEL_NORM,
        spatial_padding_mode=PaddingModeType.ZEROS,
    ).eval()


def _downsample() -> SpaceToDepthDownsample:
    return SpaceToDepthDownsample(
        dims=3,
        in_channels=128,
        out_channels=128,
        stride=(2, 1, 1),
        spatial_padding_mode=PaddingModeType.ZEROS,
    ).eval()


def test_encoder_resnet_sites_are_in_execution_order() -> None:
    sites = nvfp4_encoder.encoder_resnet_sites(_Encoder())
    assert [site.name for site in sites] == [
        "down_blocks.0.res_blocks.0",
        "down_blocks.0.res_blocks.1",
        "down_blocks.2.res_blocks.0",
    ]


def test_graph_build_rejects_incomplete_scale_map_before_gpu_setup() -> None:
    with pytest.raises(ValueError, match="missing="):
        nvfp4_encoder._build_nvfp4_encoder_graph(_Encoder(), {})


def test_graph_is_final_before_weight_load_and_preserves_checkpoint_keys() -> None:
    encoder = _Encoder()
    conv_in = encoder.conv_in
    conv_out = encoder.conv_out
    downsample = encoder.down_blocks[1]
    checkpoint_keys = set(encoder.state_dict())
    names = [site.name for site in nvfp4_encoder.encoder_resnet_sites(encoder)]
    scales = {conv_name: 1.0 for name in names for conv_name in (f"{name}.conv1", f"{name}.conv2")}
    installed = nvfp4_encoder._build_nvfp4_encoder_graph(encoder, scales)

    assert installed == tuple(names)
    assert all(
        isinstance(block, nvfp4_encoder.LTX23Nvfp4ResnetBlock)
        for container in (encoder.down_blocks[0], encoder.down_blocks[2])
        for block in container.res_blocks
    )
    assert encoder.down_blocks[1] is downsample
    assert encoder.conv_in is conv_in
    assert encoder.conv_out is conv_out
    assert set(encoder.state_dict()) == checkpoint_keys


def test_graph_rejects_invalid_scale_before_gpu_setup() -> None:
    encoder = _Encoder()
    names = [site.name for site in nvfp4_encoder.encoder_resnet_sites(encoder)]
    scales = {conv_name: 1.0 for name in names for conv_name in (f"{name}.conv1", f"{name}.conv2")}
    scales[f"{names[0]}.conv1"] = 0.0
    with pytest.raises(ValueError, match="positive activation"):
        nvfp4_encoder._build_nvfp4_encoder_graph(encoder, scales)
