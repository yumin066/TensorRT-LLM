# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the LTX-2 video VAE tiled-encode geometry.

The tiled encode exists so the retake path matches the reference implementation,
which tiles both spatially and temporally for inputs larger than one tile. The
geometry -- which pixels each tile reads, where it lands in latent space, and how
tiles are blended -- is pure index arithmetic, so it is covered here without a
GPU or checkpoint. An error in that arithmetic would otherwise only surface as an
opaque end-to-end numeric difference.

Everything below runs on CPU in well under a second.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.video_vae.tiling import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.video_vae.video_vae import (
    prepare_tiles_for_encoding,
)

# The retake window the parity work is anchored on, and the reference tiling
# geometry the oracle encodes it with.
FRAMES, HEIGHT, WIDTH = 209, 1280, 704
LATENT_CHANNELS = 128
SCALE_T, SCALE_HW = 8, 32
LATENT_FRAMES = (FRAMES - 1) // SCALE_T + 1  # 27
LATENT_H, LATENT_W = HEIGHT // SCALE_HW, WIDTH // SCALE_HW  # 40, 22

ORACLE_TILING = TilingConfig(
    spatial_config=SpatialTilingConfig(tile_size_in_pixels=768, tile_overlap_in_pixels=64),
    temporal_config=TemporalTilingConfig(tile_size_in_frames=80, tile_overlap_in_frames=24),
)


def _video():
    # Only the shape matters: prepare_tiles_for_encoding is pure index arithmetic.
    return torch.zeros(1, 3, FRAMES, HEIGHT, WIDTH)


def test_retake_window_tile_geometry():
    """The 209-frame window splits into the reference 2x4 tile grid.

    Width 704 is below the 768 tile size so it is not split; height 1280 gives two
    spatial tiles and 209 frames give four temporal tiles.
    """
    tiles = prepare_tiles_for_encoding(_video(), ORACLE_TILING)
    assert len(tiles) == 8

    def span(tile, axis):
        return (tile.in_coords[axis].start, tile.in_coords[axis].stop)

    assert {span(t, 3) for t in tiles} == {(0, 768), (704, 1280)}
    assert {span(t, 4) for t in tiles} == {(0, WIDTH)}, "width must not be tiled at 704 < 768"
    assert {span(t, 2) for t in tiles} == {(0, 81), (56, 137), (112, 193), (168, 209)}


def test_every_tile_feeds_the_causal_encoder_a_legal_frame_count():
    """The causal encoder maps F frames to (F-1)/8 + 1 latents, so F must be 8k+1.

    A tile that is not 8k+1 would be silently cropped by the encoder and land in
    the wrong latent range.
    """
    for tile in prepare_tiles_for_encoding(_video(), ORACLE_TILING):
        frames = tile.in_coords[2].stop - tile.in_coords[2].start
        assert (frames - 1) % SCALE_T == 0, f"tile with {frames} frames is not 8k+1"


def test_tiles_cover_the_latent_without_gaps():
    """No latent position may be left unwritten.

    `tiled_encode` divides the accumulator by the accumulated weights, so a
    position no tile wrote would become 0/0 -> garbage rather than an error.
    """
    weights = torch.zeros(1, LATENT_CHANNELS, LATENT_FRAMES, LATENT_H, LATENT_W)
    for tile in prepare_tiles_for_encoding(_video(), ORACLE_TILING):
        weights[tile.out_coords] += tile.blend_mask(torch.device("cpu"), torch.float32)
    assert (weights > 0).all(), f"{int((weights <= 0).sum())} latent positions never written"


def test_output_slices_stay_inside_the_latent():
    latent_bounds = (1, LATENT_CHANNELS, LATENT_FRAMES, LATENT_H, LATENT_W)
    for tile in prepare_tiles_for_encoding(_video(), ORACLE_TILING):
        for axis, bound in enumerate(latent_bounds):
            sl = tile.out_coords[axis]
            # Untiled axes (batch, channel) keep the default whole-axis mapping,
            # which is `slice(0, None)` -- an open end, not an out-of-range one.
            stop = bound if sl.stop is None else sl.stop
            assert 0 <= sl.start < stop <= bound, f"axis {axis}: {sl} exceeds {bound}"


def test_input_below_tile_size_is_not_tiled():
    """A window smaller than one tile must stay a single tile on every axis."""
    small = torch.zeros(1, 3, 17, 256, 256)
    tiles = prepare_tiles_for_encoding(small, ORACLE_TILING)
    assert len(tiles) == 1
    assert tiles[0].in_coords[2] == slice(0, 17)
    assert tiles[0].in_coords[3] == slice(0, 256)


def test_no_tiling_config_produces_a_single_whole_tensor_tile():
    tiles = prepare_tiles_for_encoding(_video(), None)
    assert len(tiles) == 1
    assert tiles[0].in_coords[2] == slice(0, FRAMES)


def test_tiled_encode_without_a_config_does_not_take_the_tiled_path():
    """AC-5's reverse assertion: tiling must be opt-in.

    The image-to-video and two-stage paths call the encoder without a tiling
    config and must keep their previous numerics, i.e. a single `forward` call on
    the whole tensor -- not a tiled accumulate-and-normalize, which would change
    the result even with correct geometry.
    """
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.video_vae.video_vae import (
        VideoEncoder,
    )

    calls = []
    sentinel = torch.zeros(1, LATENT_CHANNELS, 3, 4, 4)

    class _Spy(VideoEncoder):
        def __init__(self):  # bypass the real (weight-carrying) constructor
            torch.nn.Module.__init__(self)
            self.out_channels = LATENT_CHANNELS

        def forward(self, sample):  # type: ignore[override]
            calls.append(tuple(sample.shape))
            return sentinel

    enc = _Spy()
    video = torch.zeros(1, 3, 17, 128, 128)
    out = enc.tiled_encode(video, None)

    assert calls == [tuple(video.shape)], "no-config encode must be one whole-tensor forward"
    assert out is sentinel, "no-config encode must return forward()'s result untouched"
