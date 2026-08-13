# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for LTX-2 BT.709 limited-range video and audio writeout."""

from __future__ import annotations

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.color_conversion import (
    AVCOL_RANGE_MPEG,
    AVCOL_SPC_BT709,
    rgb_to_yuv420p_bt709_limited,
)


def _solid(height, width, rgb, frames=1):
    t = torch.zeros(frames, height, width, 3, dtype=torch.uint8)
    t[..., 0], t[..., 1], t[..., 2] = rgb
    return t


def _planes(packed, height, width):
    """Split (T, H*3//2, W) packed I420 back into Y, U, V."""
    y = packed[:, :height, :]
    uv = packed[:, height:, :]
    u = uv[..., : width // 2]
    v = uv[..., width // 2 :]
    return y, u, v


def test_ffmpeg_constants_are_the_tv_bt709_pair():
    """These two integers are what end up in the container; a typo is silent."""
    assert AVCOL_SPC_BT709 == 1  # AVCOL_SPC_BT709
    assert AVCOL_RANGE_MPEG == 1  # AVCOL_RANGE_MPEG == limited / "tv"


def test_output_shape_is_packed_i420():
    out = rgb_to_yuv420p_bt709_limited(torch.zeros(3, 8, 6, 3, dtype=torch.uint8))
    assert out.shape == (3, 8 * 3 // 2, 6)
    assert out.dtype == torch.uint8


def test_black_and_white_land_on_the_limited_range_endpoints():
    """Limited range is the whole point: black is 16 and white is 235, not 0/255.

    Emitting full-range values under a `tv` tag is exactly the mislabelling this
    conversion exists to prevent, and it shows up as washed-out or crushed levels
    rather than as an error.
    """
    black_y, _, _ = _planes(rgb_to_yuv420p_bt709_limited(_solid(4, 4, (0, 0, 0))), 4, 4)
    white_y, _, _ = _planes(rgb_to_yuv420p_bt709_limited(_solid(4, 4, (255, 255, 255))), 4, 4)
    assert black_y.min() == 16 and black_y.max() == 16
    # 219 + 16 = 235; truncation can land one level low, never high.
    assert white_y.min() in (234, 235) and white_y.max() in (234, 235)


def test_neutral_grey_leaves_chroma_centred():
    """Any achromatic input must give U = V = 128; a matrix error tilts this."""
    packed = rgb_to_yuv420p_bt709_limited(_solid(4, 4, (128, 128, 128)))
    _, u, v = _planes(packed, 4, 4)
    assert u.min() in (127, 128) and u.max() in (127, 128)
    assert v.min() in (127, 128) and v.max() in (127, 128)


@pytest.mark.parametrize(
    ("rgb", "weight"),
    [((255, 0, 0), 0.2126), ((0, 255, 0), 0.7152), ((0, 0, 255), 0.0722)],
)
def test_bt709_luma_weights_are_used(rgb, weight):
    """Each primary's luma must be 16 + 219 * its BT.709 weight.

    BT.601 (0.299 / 0.587 / 0.114) is the matrix libswscale would plausibly have
    picked, and it fails every one of these by a wide margin — which is the
    wrong-matrix mistake this conversion exists to rule out.
    """
    y, _, _ = _planes(rgb_to_yuv420p_bt709_limited(_solid(4, 4, rgb)), 4, 4)
    value = float(y.float().mean())
    expected = 16.0 + 219.0 * weight
    assert abs(value - expected) <= 1.0, f"{rgb} -> Y {value}, expected ~{expected}"


def test_chroma_is_subsampled_by_averaging_2x2_blocks():
    """A 2x2 block of mixed colour must collapse to one averaged chroma sample.

    Taking a single sample of the block instead of the mean is a common shortcut
    and produces visible chroma aliasing on fine detail.
    """
    frame = torch.zeros(1, 2, 2, 3, dtype=torch.uint8)
    frame[0, 0, 0] = torch.tensor([255, 0, 0], dtype=torch.uint8)  # red
    frame[0, 0, 1] = torch.tensor([0, 0, 255], dtype=torch.uint8)  # blue
    frame[0, 1, 0] = torch.tensor([0, 0, 255], dtype=torch.uint8)
    frame[0, 1, 1] = torch.tensor([255, 0, 0], dtype=torch.uint8)

    packed = rgb_to_yuv420p_bt709_limited(frame)
    _, u, v = _planes(packed, 2, 2)
    assert u.shape == (1, 1, 1) and v.shape == (1, 1, 1)

    # Average the two colours' chroma by hand and compare.
    mat = torch.tensor([[-0.1146, -0.3854, 0.5], [0.5, -0.4542, -0.0458]])
    red = mat @ torch.tensor([1.0, 0.0, 0.0])
    blue = mat @ torch.tensor([0.0, 0.0, 1.0])
    expected_uv = (red + blue) / 2 * 224.0 + 128.0
    assert abs(float(u[0, 0, 0]) - float(expected_uv[0])) <= 1.0
    assert abs(float(v[0, 0, 0]) - float(expected_uv[1])) <= 1.0


def test_odd_dimensions_are_rejected():
    """I420 has no meaning for odd H/W; failing loudly beats emitting a bad plane."""
    with pytest.raises(ValueError, match="even"):
        rgb_to_yuv420p_bt709_limited(torch.zeros(1, 5, 4, 3, dtype=torch.uint8))


def test_wrong_rank_is_rejected():
    with pytest.raises(ValueError, match=r"\(T, H, W, 3\)"):
        rgb_to_yuv420p_bt709_limited(torch.zeros(1, 3, 8, 8, dtype=torch.uint8))


def test_input_is_not_mutated():
    """`_save_video` keeps the RGB frames for the frame count after converting."""
    frames = _solid(4, 4, (10, 200, 30), frames=2)
    before = frames.clone()
    rgb_to_yuv420p_bt709_limited(frames)
    assert torch.equal(frames, before)


def _retake_example():
    """Import the example script by path, or skip.

    Its module-level imports are only `argparse`, `SimpleNamespace` and `torch`,
    so loading it is cheap and side-effect free (`main()` is guarded). That lets
    the pure writeout helpers be tested by EXECUTION rather than by parsing.
    """
    import importlib.util
    import pathlib

    rel = pathlib.Path("examples/visual_gen/ltx2_retake_e2e.py")
    script = next(
        (p / rel for p in pathlib.Path(__file__).resolve().parents if (p / rel).is_file()),
        None,
    )
    if script is None:
        pytest.skip(f"{rel} not found above {__file__}; run from a repo checkout")
    spec = importlib.util.spec_from_file_location("ltx2_retake_e2e_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mono_audio_is_duplicated_to_both_channels():
    """Mono must be duplicated, not left as a silent right channel."""
    mod = _retake_example()
    out = mod._to_stereo_int16(torch.full((1, 4), 0.5))
    assert out.shape == (1, 8)
    assert (out[0, 0::2] == out[0, 1::2]).all(), "right channel must mirror mono input"


def test_stereo_channels_are_interleaved_in_order():
    """L/R interleaving is invisible in shape but audible if swapped."""
    mod = _retake_example()
    left = torch.tensor([0.1, 0.2, 0.3])
    right = torch.tensor([-0.1, -0.2, -0.3])
    out = mod._to_stereo_int16(torch.stack([left, right]))
    assert out.shape == (1, 6)
    assert (out[0, 0::2] > 0).all(), "even samples must be the left channel"
    assert (out[0, 1::2] < 0).all(), "odd samples must be the right channel"


def test_full_scale_and_overshoot_are_clamped_not_wrapped():
    """int16 wrap turns a peak into a full-scale click of the opposite sign.

    Decoded float samples can exceed [-1, 1] slightly, so the overshoot path is
    real. `+1.5 * 32767` overflows int16 and wraps to a large negative value if
    the clamp is missing — the assertion is on the exact values, because
    `max()`/`min()` alone would still pass with one wrapped sample present.
    """
    import numpy as np

    mod = _retake_example()
    # Mono input, so every value appears twice (once per duplicated channel).
    out = mod._to_stereo_int16(torch.tensor([[1.0, -1.0, 1.5, -1.5]]))
    expected = np.array([[32767, 32767, -32767, -32767, 32767, 32767, -32767, -32767]])
    assert np.array_equal(out, expected)
    assert out.dtype == np.int16


def test_batched_audio_is_accepted_and_unwrapped():
    mod = _retake_example()
    assert mod._to_stereo_int16(torch.zeros(1, 2, 5)).shape == (1, 10)


def test_multi_batch_audio_is_rejected():
    """Silently taking batch 0 would ship the wrong clip's audio."""
    mod = _retake_example()
    with pytest.raises(ValueError, match="one audio batch"):
        mod._to_stereo_int16(torch.zeros(2, 2, 5))


def test_more_than_two_channels_is_rejected():
    """Dropping a channel is invisible in the output file, so it must raise."""
    mod = _retake_example()
    with pytest.raises(ValueError, match="mono or stereo"):
        mod._to_stereo_int16(torch.zeros(6, 5))


def test_audio_is_trimmed_to_the_video_duration():
    """Audio samples beyond the encoded video duration are dropped."""
    mod = _retake_example()
    max_samples = int(round(209 / 30.0 * 48000))
    out = mod._to_stereo_int16(torch.zeros(2, 48000 * 10), max_samples=max_samples)
    assert out.shape == (1, max_samples * 2)


def test_untrimmed_audio_keeps_every_sample():
    mod = _retake_example()
    assert mod._to_stereo_int16(torch.zeros(2, 1234)).shape == (1, 1234 * 2)


def test_save_video_accepts_and_muxes_audio():
    """The signature and the mux path must both exist.

    Checked structurally because `_save_video` needs PyAV and a real encoder;
    the pure conversion half above is executed.
    """
    import inspect

    mod = _retake_example()
    params = inspect.signature(mod._save_video).parameters
    assert "audio" in params, "_save_video must accept the preserved source audio"
    assert "audio_sample_rate" in params
    assert params["audio"].default is None, "silent clips must still be writable"

    body = inspect.getsource(mod._save_video)
    assert "add_stream" in body and "aac" in body, "an AAC stream must be created"
    assert "_mux_audio" in body


def test_main_passes_the_pipeline_audio_to_the_writer():
    """The example passes preserved audio and its sample rate to the writer."""
    import ast
    import inspect

    mod = _retake_example()
    tree = ast.parse(inspect.getsource(mod.main))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_save_video"
    ]
    assert len(calls) == 1, "expected exactly one writeout call in main()"
    passed = ast.unparse(calls[0])
    assert "out.audio" in passed, "main() must pass the pipeline's preserved audio"
    assert "out.audio_sample_rate" in passed


def test_save_video_tags_the_stream_and_keeps_fractional_fps():
    """The writeout tags colour correctly and preserves fractional frame rates."""
    import ast
    import pathlib

    rel = pathlib.Path("examples/visual_gen/ltx2_retake_e2e.py")
    script = next(
        (p / rel for p in pathlib.Path(__file__).resolve().parents if (p / rel).is_file()),
        None,
    )
    if script is None:
        pytest.skip(f"{rel} not found above {__file__}; run from a repo checkout")

    tree = ast.parse(script.read_text())
    func = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_save_video"
    )
    statements = func.body
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
        and isinstance(statements[0].value.value, str)
    ):
        statements = statements[1:]
    body = "\n".join(ast.unparse(s) for s in statements)

    assert "AVCOL_SPC_BT709" in body, "stream must be tagged with the BT.709 colourspace"
    assert "AVCOL_RANGE_MPEG" in body, "stream must be tagged limited/tv range"
    assert "rgb_to_yuv420p_bt709_limited" in body, "conversion must be explicit, not libswscale's"
    assert "yuv420p" in body and "rgb24" not in body, "frames must be fed as yuv420p"
    assert "int(round(" not in body, "frame rate must not be rounded to an integer"
    assert "Fraction" in body, "fractional frame rates must be preserved"
