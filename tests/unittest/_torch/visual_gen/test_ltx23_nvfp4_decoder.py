# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.video_vae.enums import (
    NormLayerType,
    PaddingModeType,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.video_vae.resnet import ResnetBlock3D
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.video_vae.sampling import (
    DepthToSpaceUpsample,
)
from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core import (
    video_vae_ltx23_nvfp4 as nvfp4_decoder,
)


class _ResnetContainer(nn.Module):
    def __init__(self, blocks: list[ResnetBlock3D]) -> None:
        super().__init__()
        self.res_blocks = nn.ModuleList(blocks)


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_in = nn.Identity()
        self.up_blocks = nn.ModuleList(
            [
                _ResnetContainer([_block(), _block()]),
                _upsample(),
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


def _upsample() -> DepthToSpaceUpsample:
    return DepthToSpaceUpsample(
        dims=3,
        in_channels=128,
        stride=(2, 1, 1),
        spatial_padding_mode=PaddingModeType.ZEROS,
    ).eval()


def test_decoder_resnet_sites_are_in_execution_order() -> None:
    decoder = _Decoder()
    sites = nvfp4_decoder.decoder_resnet_sites(decoder)
    assert [site.name for site in sites] == [
        "up_blocks.0.res_blocks.0",
        "up_blocks.0.res_blocks.1",
        "up_blocks.2.res_blocks.0",
    ]


def test_decoder_upsample_sites_are_in_execution_order() -> None:
    decoder = _Decoder()
    sites = nvfp4_decoder.decoder_upsample_sites(decoder)
    assert [site.name for site in sites] == ["up_blocks.1.conv"]


def test_graph_build_rejects_incomplete_scale_map_before_gpu_setup() -> None:
    with pytest.raises(ValueError, match="missing="):
        nvfp4_decoder._build_nvfp4_decoder_graph(_Decoder(), {})


def test_graph_is_final_before_weight_load_and_preserves_checkpoint_keys() -> None:
    decoder = _Decoder()
    conv_in = decoder.conv_in
    conv_out = decoder.conv_out
    checkpoint_keys = set(decoder.state_dict())
    resnet_names = [site.name for site in nvfp4_decoder.decoder_resnet_sites(decoder)]
    upsample_names = [site.name for site in nvfp4_decoder.decoder_upsample_sites(decoder)]
    scales = {
        conv_name: 1.0 for name in resnet_names for conv_name in (f"{name}.conv1", f"{name}.conv2")
    }
    scales.update({name: 3.0 for name in upsample_names})
    installed = nvfp4_decoder._build_nvfp4_decoder_graph(decoder, scales)

    assert installed == tuple(resnet_names + upsample_names)
    assert all(
        isinstance(block, nvfp4_decoder.LTX23Nvfp4ResnetBlock)
        for container in (decoder.up_blocks[0], decoder.up_blocks[2])
        for block in container.res_blocks
    )
    assert isinstance(decoder.up_blocks[1].conv, nvfp4_decoder._PreparedNvfp4Conv3d)
    assert decoder.conv_in is conv_in
    assert decoder.conv_out is conv_out
    assert set(decoder.state_dict()) == checkpoint_keys


def test_graph_rejects_invalid_scale_before_gpu_setup() -> None:
    decoder = _Decoder()
    names = [site.name for site in nvfp4_decoder.decoder_resnet_sites(decoder)]
    upsample_names = [site.name for site in nvfp4_decoder.decoder_upsample_sites(decoder)]
    scales = {conv_name: 1.0 for name in names for conv_name in (f"{name}.conv1", f"{name}.conv2")}
    scales.update({name: 1.0 for name in upsample_names})
    scales[upsample_names[0]] = 0.0
    with pytest.raises(ValueError, match="positive activation"):
        nvfp4_decoder._build_nvfp4_decoder_graph(decoder, scales)


def test_retake_recipe_is_pipeline_scoped(tmp_path: Path) -> None:
    path = tmp_path / "video_vae_ltx23_retake_nvfp4_global_fp32_scales.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": "LTX-2.3",
                "pipeline": "retake",
                "encoder": {"encoder.conv": {"activation_quant_multiplier": 2.0}},
                "decoder": {"decoder.conv": {"activation_quant_multiplier": 3.0}},
            }
        )
    )
    recipe = nvfp4_decoder.load_ltx23_nvfp4_retake_recipe(path)
    assert recipe.encoder == {"encoder.conv": 2.0}
    assert recipe.decoder == {"decoder.conv": 3.0}


def test_retake_recipe_rejects_another_pipeline(tmp_path: Path) -> None:
    path = tmp_path / "video_vae_ltx23_nvfp4_text_to_video.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": "LTX-2.3",
                "pipeline": "text_to_video",
                "encoder": {"encoder.conv": {"activation_quant_multiplier": 2.0}},
                "decoder": {"decoder.conv": {"activation_quant_multiplier": 3.0}},
            }
        )
    )
    with pytest.raises(ValueError, match="not scoped"):
        nvfp4_decoder.load_ltx23_nvfp4_retake_recipe(path)


def test_fused_kernel_source_contains_layout_aware_contract() -> None:
    source = (
        Path(nvfp4_decoder.__file__)
        .with_name("video_vae_ltx23_nvfp4_preprocess.cu")
        .read_text(encoding="utf-8")
    )
    assert "input.stride(1)" in source
    assert "video_vae_ltx23_nvfp4_preprocess" in source
    assert "fp32_pair_to_e2m1" in source
