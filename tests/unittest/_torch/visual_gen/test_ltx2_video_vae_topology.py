# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Channel-topology parity tests for the LTX-2 native video VAE.

These tests build ``VideoDecoder`` / ``VideoEncoder`` from configurator
configs and assert the per-block convolution channel shapes. They cover two
VAE topologies:

* A ``compress_space`` / ``compress_time`` / ``compress_all`` decoder (and the
  matching ``*_res`` encoder) that exercises the mixed-compression channel
  arithmetic required by the larger VAE checkpoint.
* A ``compress_all``-only decoder/encoder that must build unchanged.

All construction is CPU-only and weight-free (shapes are read from freshly
initialized ``nn.Conv`` parameters), so no GPU or model checkpoint is needed.
"""

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.video_vae import (
    VideoDecoderConfigurator,
    VideoEncoderConfigurator,
)

# A base_channels=128 VAE whose decoder mixes compress_space / compress_time /
# compress_all and whose encoder uses the corresponding *_res downsamplers.
MIXED_COMPRESSION_VAE = {
    "vae": {
        "_class_name": "CausalVideoAutoencoder",
        "dims": 3,
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 128,
        "patch_size": 4,
        "norm_layer": "pixel_norm",
        "use_quant_conv": False,
        "causal_decoder": False,
        "timestep_conditioning": False,
        "normalize_latent_channels": False,
        "encoder_base_channels": 128,
        "decoder_base_channels": 128,
        "spatial_padding_mode": "zeros",
        "encoder_blocks": [
            ["res_x", {"num_layers": 4}],
            ["compress_space_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 6}],
            ["compress_time_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 4}],
            ["compress_all_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 2}],
            ["compress_all_res", {"multiplier": 1}],
            ["res_x", {"num_layers": 2}],
        ],
        "decoder_blocks": [
            ["res_x", {"num_layers": 4}],
            ["compress_space", {"multiplier": 2}],
            ["res_x", {"num_layers": 6}],
            ["compress_time", {"multiplier": 2}],
            ["res_x", {"num_layers": 4}],
            ["compress_all", {"multiplier": 1}],
            ["res_x", {"num_layers": 2}],
            ["compress_all", {"multiplier": 2}],
            ["res_x", {"num_layers": 2}],
        ],
    }
}

# A base_channels=128 VAE whose decoder uses compress_all only (the smaller
# checkpoint topology). Its build must be unaffected by the mixed-compression
# support.
COMPRESS_ALL_ONLY_VAE = {
    "vae": {
        "_class_name": "CausalVideoAutoencoder",
        "dims": 3,
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 128,
        "patch_size": 4,
        "norm_layer": "pixel_norm",
        "causal_decoder": False,
        "timestep_conditioning": False,
        "decoder_base_channels": 128,
        "spatial_padding_mode": "zeros",
        "encoder_blocks": [
            ["res_x", {"num_layers": 4}],
            ["compress_all_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 4}],
            ["compress_all_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 4}],
            ["compress_all_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 4}],
        ],
        "decoder_blocks": [
            ["res_x", {"num_layers": 4}],
            ["compress_all", {"multiplier": 2}],
            ["res_x", {"num_layers": 4}],
            ["compress_all", {"multiplier": 2}],
            ["res_x", {"num_layers": 4}],
            ["compress_all", {"multiplier": 2}],
            ["res_x", {"num_layers": 4}],
        ],
    }
}


def _conv_out_in(module) -> tuple[int, int]:
    """Return ``(out_channels, in_channels)`` of a convolution ``weight``.

    Works for ``CausalConv3d`` (whose ``weight`` property forwards to the
    wrapped ``nn.Conv3d``) as well as plain ``nn.Conv`` modules.
    """
    weight = module.weight
    return int(weight.shape[0]), int(weight.shape[1])


def _res_x_channels(block) -> int:
    """Channel count of a ``res_x`` (``UNetMidBlock3D``) block."""
    out_channels, in_channels = _conv_out_in(block.res_blocks[0].conv1)
    assert out_channels == in_channels
    return out_channels


def test_mixed_compression_decoder_channels():
    decoder = VideoDecoderConfigurator.from_config(MIXED_COMPRESSION_VAE)

    # conv_in lifts the 128 latent channels to base_channels * 8 = 1024.
    assert _conv_out_in(decoder.conv_in) == (1024, 128)

    up = decoder.up_blocks
    # up_blocks are built in reversed(decoder_blocks) order.
    # feature_channels: 1024 -> compress_all(m2) 512 -> compress_all(m1) 512
    #   -> compress_time(m2) 256 -> compress_space(m2) 128.
    assert _res_x_channels(up[0]) == 1024
    # compress_all(m2): DepthToSpace conv out = 8 * 1024 // 2 = 4096.
    assert _conv_out_in(up[1].conv) == (4096, 1024)
    assert _res_x_channels(up[2]) == 512
    # compress_all(m1): DepthToSpace conv out = 8 * 512 // 1 = 4096.
    assert _conv_out_in(up[3].conv) == (4096, 512)
    assert _res_x_channels(up[4]) == 512
    # compress_time(m2): stride (2,1,1) -> conv out = 2 * 512 // 2 = 512.
    assert _conv_out_in(up[5].conv) == (512, 512)
    # The mixed-compression fix keeps this res_x at 256 channels (previously
    # undercounted to 128, which broke checkpoint loading).
    assert _res_x_channels(up[6]) == 256
    # compress_space(m2): stride (1,2,2) -> conv out = 4 * 256 // 2 = 512.
    assert _conv_out_in(up[7].conv) == (512, 256)
    assert _res_x_channels(up[8]) == 128

    # conv_out maps 128 features back to out_channels * patch_size**2 = 48.
    assert _conv_out_in(decoder.conv_out) == (48, 128)


def test_mixed_compression_encoder_channels():
    encoder = VideoEncoderConfigurator.from_config(MIXED_COMPRESSION_VAE)

    # conv_in maps patchified pixels (3 * 4**2 = 48) to 128 latent-width feats.
    assert _conv_out_in(encoder.conv_in) == (128, 48)

    down = encoder.down_blocks
    # feature_channels: 128 -> compress_space_res(m2) 256 -> compress_time_res(m2)
    #   512 -> compress_all_res(m2) 1024 -> compress_all_res(m1) 1024.
    assert _res_x_channels(down[0]) == 128
    # compress_space_res(m2): SpaceToDepth conv out = 256 // 4 = 64.
    assert _conv_out_in(down[1].conv) == (64, 128)
    assert down[1].out_channels == 256
    assert _res_x_channels(down[2]) == 256
    # compress_time_res(m2): conv out = 512 // 2 = 256.
    assert _conv_out_in(down[3].conv) == (256, 256)
    assert down[3].out_channels == 512
    assert _res_x_channels(down[4]) == 512
    # compress_all_res(m2): conv out = 1024 // 8 = 128.
    assert _conv_out_in(down[5].conv) == (128, 512)
    assert down[5].out_channels == 1024
    assert _res_x_channels(down[6]) == 1024
    # compress_all_res(m1): conv out = 1024 // 8 = 128.
    assert _conv_out_in(down[7].conv) == (128, 1024)
    assert down[7].out_channels == 1024
    assert _res_x_channels(down[8]) == 1024

    # conv_out emits latent_channels + 1 (uniform log-variance channel) = 129.
    assert _conv_out_in(encoder.conv_out) == (129, 1024)


def test_compress_all_only_decoder_channels_unchanged():
    decoder = VideoDecoderConfigurator.from_config(COMPRESS_ALL_ONLY_VAE)

    # conv_in still lifts to base_channels * 8 = 1024 for the compress_all-only
    # topology, matching the pre-fix behavior.
    assert _conv_out_in(decoder.conv_in) == (1024, 128)

    up = decoder.up_blocks
    # feature_channels: 1024 -> 512 -> 256 -> 128 via three compress_all(m2).
    assert _res_x_channels(up[0]) == 1024
    assert _conv_out_in(up[1].conv) == (4096, 1024)  # 8 * 1024 // 2
    assert _res_x_channels(up[2]) == 512
    assert _conv_out_in(up[3].conv) == (2048, 512)  # 8 * 512 // 2
    assert _res_x_channels(up[4]) == 256
    assert _conv_out_in(up[5].conv) == (1024, 256)  # 8 * 256 // 2
    assert _res_x_channels(up[6]) == 128

    assert _conv_out_in(decoder.conv_out) == (48, 128)


def test_compress_all_only_encoder_channels_unchanged():
    encoder = VideoEncoderConfigurator.from_config(COMPRESS_ALL_ONLY_VAE)

    assert _conv_out_in(encoder.conv_in) == (128, 48)

    down = encoder.down_blocks
    # feature_channels: 128 -> 256 -> 512 -> 1024 via three compress_all_res(m2).
    assert _res_x_channels(down[0]) == 128
    assert _conv_out_in(down[1].conv) == (32, 128)  # 256 // 8
    assert down[1].out_channels == 256
    assert _res_x_channels(down[2]) == 256
    assert _conv_out_in(down[3].conv) == (64, 256)  # 512 // 8
    assert down[3].out_channels == 512
    assert _res_x_channels(down[4]) == 512
    assert _conv_out_in(down[5].conv) == (128, 512)  # 1024 // 8
    assert down[5].out_channels == 1024
    assert _res_x_channels(down[6]) == 1024

    assert _conv_out_in(encoder.conv_out) == (129, 1024)
