# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the LTX-2 video VAE tiled-encode geometry.

The tiled encode exists so the retake path matches the reference implementation,
which tiles both spatially and temporally for inputs larger than one tile. The
geometry -- which pixels each tile reads, where it lands in latent space, and how
tiles are blended -- is pure index arithmetic, so it is covered here without a
GPU or checkpoint. An error in that arithmetic would otherwise only surface as an
opaque end-to-end numeric difference.

AC-5 coverage is deliberately a **two-part proof**, and the parts are named so
neither is mistaken for the other:

  * the CALLERS are checked structurally, by parsing their call nodes with `ast`
    (`*_call_site_*` tests). These do NOT execute `LTX2Pipeline.forward` or
    `LTX2TwoStagesPipeline` -- those branches sit inside long generation methods
    needing a full pipeline, weights and a device.
  * the shared CALLEE is executed (`test_encode_image_requests_no_tiling`,
    `test_i2v_encode_matches_a_plain_forward`), proving it reaches the encoder
    with `tiling_config=None` and returns the plain forward result.

Together they close the path: the call sites cannot introduce a tiling argument,
and the callee provably does not add one. `test_the_call_site_check_can_fail`
shows the structural half discriminates.

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


# --------------------------------------------------------------------------------------
# AC-9 / task13 — the retake DECODE must use the oracle's geometry too.
#
# The encode was aligned first; the decode kept calling native's `TilingConfig.default()`,
# which emits 5 temporal chunks where the oracle emits 4. Aligning it moved the measured
# decoder gap from relL2 1.297e-2 to 1.640e-3 on H200.
# --------------------------------------------------------------------------------------


def test_oracle_tiling_constant_matches_the_reference_geometry():
    """Pin the four numbers the reference decodes and encodes with.

    Upstream's `TilingConfig.default()` is 768/64 spatial and 80/24 temporal, and
    `RetakePipeline`'s callers pass exactly that. This constant is native's
    transcription of it; if upstream's default ever moves, this test is what says so.
    """
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_retake import (
        _ORACLE_TILING_CONFIG,
    )

    assert _ORACLE_TILING_CONFIG.spatial_config.tile_size_in_pixels == 768
    assert _ORACLE_TILING_CONFIG.spatial_config.tile_overlap_in_pixels == 64
    assert _ORACLE_TILING_CONFIG.temporal_config.tile_size_in_frames == 80
    assert _ORACLE_TILING_CONFIG.temporal_config.tile_overlap_in_frames == 24


def test_native_default_tiling_is_left_alone():
    """The reverse assertion: retake must not have moved the shared default.

    The generation and two-stage paths decode with native's `TilingConfig.default()`.
    Aligning retake by editing that default would have silently changed their
    numerics — which is why retake passes its geometry explicitly instead.
    """
    default = TilingConfig.default()
    assert default.spatial_config.tile_size_in_pixels == 512
    assert default.spatial_config.tile_overlap_in_pixels == 64
    assert default.temporal_config.tile_size_in_frames == 64
    assert default.temporal_config.tile_overlap_in_frames == 24


def test_the_two_tilings_really_are_different_work():
    """If both geometries produced the same tiles, aligning them would be a no-op.

    They do not: on the retake window the oracle geometry emits 8 encode tiles and
    native's default emits more, so this is a structurally different computation
    rather than a rounding difference.
    """
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_retake import (
        _ORACLE_TILING_CONFIG,
    )

    oracle_tiles = prepare_tiles_for_encoding(_video(), _ORACLE_TILING_CONFIG)
    default_tiles = prepare_tiles_for_encoding(_video(), TilingConfig.default())
    assert len(oracle_tiles) != len(default_tiles), (
        "the two tiling configs produce the same tiling, so nothing was actually aligned"
    )


def _tiling_args_at(module, func_name, callee):
    """The tiling argument expression passed to *callee* inside *func_name*, as source."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(module))
    found = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if func.name != func_name:
            continue
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == callee
            ):
                args = list(node.args) + [k.value for k in node.keywords]
                found.append([ast.unparse(a) for a in args])
    return found


def test_retake_decode_call_site_passes_the_oracle_geometry():
    """The retake decode must name the constant, not call `default()`.

    Checked structurally: `_run_native_pre_post_retake` needs a checkpoint, weights
    and a GPU to execute, so the executed proof for this row is the GPU A/B probe.
    What this pins is that the call site cannot quietly go back to `default()`.
    """
    from tensorrt_llm._torch.visual_gen.models.ltx2 import pipeline_ltx2_retake

    calls = _tiling_args_at(pipeline_ltx2_retake, "_run_native_pre_post_retake", "tiled_decode")
    assert len(calls) == 1, f"expected exactly one native decode call; got {calls}"
    args = calls[0]
    assert any("_ORACLE_TILING_CONFIG" in a for a in args), (
        f"retake decode must pass _ORACLE_TILING_CONFIG; got {args}"
    )
    assert not any("default()" in a for a in args), (
        f"retake decode must not fall back to a default() geometry; got {args}"
    )


def test_encode_and_decode_share_one_tiling_object():
    """Both retake call sites must reference the same constant, not two copies.

    Two hand-written copies of the same four numbers can drift apart silently —
    the failure mode that had an audit reporting a fixed loader's stale key counts.
    """
    from tensorrt_llm._torch.visual_gen.models.ltx2 import pipeline_ltx2_retake

    encode = _tiling_args_at(
        pipeline_ltx2_retake, "_run_native_pre_post_retake", "_encode_video_window"
    )
    assert len(encode) == 1, f"expected exactly one native encode call; got {encode}"
    assert any("_ORACLE_TILING_CONFIG" in a for a in encode[0]), (
        f"retake encode must pass the shared constant; got {encode[0]}"
    )


def test_generation_and_two_stage_decode_keep_the_native_default():
    """Retake's geometry must not leak into the paths that never asked for it.

    Those paths were characterized against native's default; silently decoding them
    with the oracle geometry would change their output with no call-site change.
    """
    from tensorrt_llm._torch.visual_gen.models.ltx2 import pipeline_ltx2, pipeline_ltx2_two_stages

    for module in (pipeline_ltx2, pipeline_ltx2_two_stages):
        source = __import__("inspect").getsource(module)
        assert "_ORACLE_TILING_CONFIG" not in source, (
            f"{module.__name__} must not use retake's tiling geometry"
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


def test_two_stage_pipeline_inherits_the_untiled_encode_path():
    """`LTX2TwoStagesPipeline` must not override the encode entry points.

    Its image-conditioning branch calls `self._encode_image(image_5d)`, so the
    no-tiling guarantee proven above only transfers if the subclass uses the same
    methods. An override would silently reintroduce tiling for two-stage.
    """
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_two_stages import (
        LTX2TwoStagesPipeline,
    )

    assert issubclass(LTX2TwoStagesPipeline, LTX2Pipeline)
    for name in ("_encode_image", "_encode_video_window"):
        assert name not in vars(LTX2TwoStagesPipeline), (
            f"{name} is overridden by LTX2TwoStagesPipeline; the tiling guarantee "
            "verified against LTX2Pipeline no longer covers the two-stage path"
        )
        assert getattr(LTX2TwoStagesPipeline, name) is getattr(LTX2Pipeline, name)


def _encode_call_sites(module, exclude_within=()):
    """Every `self._encode_image(...)` / `self._encode_video_window(...)` call node.

    `exclude_within` drops calls whose enclosing function has one of those names,
    keyed by the function they sit in rather than by line number: `_encode_image`
    delegating to `_encode_video_window` is the definition, not a caller, and a
    line-range filter would silently start excluding the wrong calls the moment
    the file shifts.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(module))
    found = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if func.name in exclude_within:
            continue
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("_encode_image", "_encode_video_window")
            ):
                found.append(node)
    return found


def _assert_no_tiling_argument(calls, where):
    """A conditioning call site must pass the tensor and nothing else.

    `_encode_video_window`'s second parameter is the tiling config, so an extra
    positional argument is as dangerous as an explicit keyword.
    """
    assert calls, f"expected {where} to encode a conditioning image"
    for call in calls:
        assert len(call.args) == 1, (
            f"{where}: {call.func.attr} at line {call.lineno} passes "
            f"{len(call.args)} positional arguments; the second parameter of "
            "_encode_video_window is the tiling config"
        )
        assert not call.keywords, (
            f"{where}: {call.func.attr} at line {call.lineno} passes keywords "
            f"{[k.arg for k in call.keywords]}; conditioning must not request tiling"
        )


def test_base_i2v_call_site_passes_no_tiling_config():
    """The base image-to-video call site must not request tiling.

    `LTX2Pipeline.forward`'s image branch calls `self._encode_image(image_5d)`.
    Pinning it here is what stops a future change from bypassing `_encode_image`
    and calling `_encode_video_window(image_5d, SOME_TILING)` directly, which the
    callee-level tests could not detect.
    """
    from tensorrt_llm._torch.visual_gen.models.ltx2 import pipeline_ltx2

    # `_encode_image` delegating to `_encode_video_window` is the definition
    # itself, not a caller; the execution tests above cover it.
    calls = _encode_call_sites(pipeline_ltx2, exclude_within={"_encode_image"})
    _assert_no_tiling_argument(calls, "pipeline_ltx2")


def test_two_stage_call_site_passes_no_tiling_config():
    """The two-stage image-conditioning call site must not request tiling."""
    from tensorrt_llm._torch.visual_gen.models.ltx2 import pipeline_ltx2_two_stages

    _assert_no_tiling_argument(
        _encode_call_sites(pipeline_ltx2_two_stages), "pipeline_ltx2_two_stages"
    )


def test_the_call_site_check_can_fail():
    """Prove the checker discriminates, on both failure shapes.

    A structural check that has only ever been run against passing source is not a
    verified check. These feed it source that a regression would actually look
    like.
    """
    import ast

    import pytest

    for bad, why in (
        ("self._encode_video_window(image_5d, ORACLE_TILING)", "extra positional"),
        ("self._encode_image(image_5d, tiling_config=ORACLE_TILING)", "explicit keyword"),
    ):
        calls = [
            n
            for n in ast.walk(ast.parse(bad))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]
        with pytest.raises(AssertionError):
            _assert_no_tiling_argument(calls, f"synthetic ({why})")


def test_i2v_encode_matches_a_plain_forward():
    """i2v zero-regression: the default path equals the pre-tiling behaviour.

    Before the tiled encode existed, `_encode_video_window` was a bare
    `self.video_encoder(video_5d)`. This asserts the default path still returns
    exactly that -- the same object a single `forward` produced -- so introducing
    tiling changed nothing for callers that do not opt in.
    """
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.video_vae.video_vae import (
        VideoEncoder,
    )
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline

    forwarded = []
    plain_result = torch.randn(1, LATENT_CHANNELS, 1, 4, 4)

    class _Encoder(VideoEncoder):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.out_channels = LATENT_CHANNELS

        def forward(self, sample):  # type: ignore[override]
            forwarded.append(tuple(sample.shape))
            return plain_result

    stub = _PipelineStub(_Encoder())
    stub._encode_video_window = LTX2Pipeline._encode_video_window.__get__(stub)
    stub._encode_image = LTX2Pipeline._encode_image.__get__(stub)

    image_5d = torch.zeros(1, 3, 1, 128, 128)
    out = stub._encode_image(image_5d)

    assert forwarded == [tuple(image_5d.shape)], "i2v must run one whole-tensor forward"
    assert out is plain_result, "i2v output must be the plain forward result, unblended"
