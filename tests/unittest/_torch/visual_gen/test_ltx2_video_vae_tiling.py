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


# --------------------------------------------------------------------------------------
# AC-5 / task8 — the image-to-video and two-stage callers must not request tiling.
#
# Both go through `LTX2Pipeline._encode_image`, which delegates to
# `_encode_video_window`. These call the real methods against a minimal stub rather
# than building a pipeline, so they run on CPU without weights while still
# exercising the production code path.
# --------------------------------------------------------------------------------------


class _EncoderSpy:
    """Stands in for the VAE encoder and records how it was called."""

    def __init__(self):
        self.calls = []
        self.result = torch.zeros(1, LATENT_CHANNELS, 1, 4, 4)

    def tiled_encode(self, video, tiling_config=None):
        self.calls.append({"shape": tuple(video.shape), "tiling_config": tiling_config})
        return self.result


class _PipelineStub:
    """Only what `_encode_image` / `_encode_video_window` actually touch."""

    def __init__(self, encoder):
        self.video_encoder = encoder

    _encode_video_window = None  # bound below
    _encode_image = None


def _stub_with_spy():
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

    spy = _EncoderSpy()
    stub = _PipelineStub(spy)
    # Bind the real, unmodified methods onto the stub.
    stub._encode_video_window = LTX2Pipeline._encode_video_window.__get__(stub)
    stub._encode_image = LTX2Pipeline._encode_image.__get__(stub)
    return stub, spy


def test_encode_image_requests_no_tiling():
    """i2v conditioning must reach the encoder with tiling_config=None."""
    stub, spy = _stub_with_spy()
    out = stub._encode_image(torch.zeros(1, 3, 1, 128, 128))

    assert len(spy.calls) == 1
    assert spy.calls[0]["tiling_config"] is None, (
        "the image-to-video path must not opt into tiled encoding; doing so would "
        "change its numerics"
    )
    assert out is spy.result


def test_encode_video_window_defaults_to_no_tiling():
    """The default must stay opt-in at the signature level.

    The two-stage path calls `_encode_image(image_5d)` with no tiling argument, so
    the default is what protects it; a change of default would silently alter both
    callers without touching either call site.
    """
    import inspect

    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

    sig = inspect.signature(LTX2Pipeline._encode_video_window)
    assert sig.parameters["tiling_config"].default is None

    stub, spy = _stub_with_spy()
    stub._encode_video_window(torch.zeros(1, 3, 9, 64, 64))
    assert spy.calls[0]["tiling_config"] is None


def test_encode_video_window_forwards_an_explicit_config():
    """Retake opts in explicitly, so the argument must actually reach the encoder."""
    stub, spy = _stub_with_spy()
    stub._encode_video_window(torch.zeros(1, 3, 9, 64, 64), ORACLE_TILING)
    assert spy.calls[0]["tiling_config"] is ORACLE_TILING
