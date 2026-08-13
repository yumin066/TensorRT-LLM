# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the LTX-2 RGB -> YUV420p BT.709 writeout conversion.

The retake mp4 previously carried no colour metadata at all
(`color_range=unknown`, `color_space=unknown`, against the reference's
`tv`/`bt709`), because RGB24 was handed to PyAV and libswscale chose the matrix.
Tagging the stream without also controlling the conversion would have been worse
than leaving it untagged — the pixels and the metadata would disagree — so the
conversion is done explicitly and these tests pin it against the BT.709 limited
range definition rather than against a reimplementation of itself.

Everything here runs on CPU in well under a second.
"""

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


def test_save_video_tags_the_stream_and_keeps_fractional_fps():
    """The writeout must set both tags and must not round the frame rate.

    Checked structurally — `_save_video` needs PyAV and a real encoder to run —
    against the two regressions that actually happened: no colour tags at all,
    and `int(round(frame_rate))`.
    """
    import ast
    import pathlib

    # Walk up to find the repo root rather than assuming a fixed depth: a fixed
    # `parents[n]` silently resolves to the wrong directory the moment this file
    # is run from anywhere but its canonical path.
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
    # Scan CODE only. The docstring explains the `int(round(fps))` regression by
    # name, so unparsing the whole function makes the check match its own
    # explanation and fail on correct source.
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


def test_the_writeout_check_can_fail():
    """A structural check only ever run against passing source is not a check.

    Feed it each regression shape this test claims to catch and require every one
    to be rejected -- including the docstring case, where the explanation of a
    past bug must not be mistaken for the bug.
    """
    import ast

    def scan(src):
        func = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef))
        statements = func.body
        if (
            statements
            and isinstance(statements[0], ast.Expr)
            and isinstance(statements[0].value, ast.Constant)
            and isinstance(statements[0].value.value, str)
        ):
            statements = statements[1:]
        return "\n".join(ast.unparse(s) for s in statements)

    rounded = scan(
        "def _save_video():\n    stream = add_stream('libx264', rate=int(round(frame_rate)))\n"
    )
    assert "int(round(" in rounded, "checker must catch a rounded frame rate"
    assert "Fraction" not in rounded

    untagged = scan("def _save_video():\n    stream.pix_fmt = 'yuv420p'\n")
    assert "AVCOL_SPC_BT709" not in untagged, "checker must catch a missing colourspace tag"

    rgb_path = scan("def _save_video():\n    VideoFrame.from_ndarray(f, format='rgb24')\n")
    assert "rgb24" in rgb_path, "checker must catch frames still being fed as rgb24"

    # The docstring-only mention must NOT register as the defect.
    documented = scan(
        "def _save_video():\n"
        '    """Previously used int(round(fps)) and fed rgb24."""\n'
        "    rate = Fraction(frame_rate)\n"
    )
    assert "int(round(" not in documented and "rgb24" not in documented
