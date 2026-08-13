# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for native LTX-2 video source decoding."""

import ast
import inspect
from fractions import Fraction

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core import media_io
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.types import VideoPixelShape


class _FakeCodecContext:
    width = 8
    height = 4


class _FakeVideoStream:
    type = "video"
    average_rate = Fraction(30000, 1001)
    codec_context = _FakeCodecContext()

    def __init__(self, frames: int):
        self.frames = frames


class _FakeVideoFrame:
    def __init__(self, value: int):
        self._array = np.full((4, 8, 3), value, dtype=np.uint8)

    def to_rgb(self):
        return self

    def to_ndarray(self):
        return self._array


class _FakeContainer:
    def __init__(self, *, reported_frames: int, decoded_frames: int):
        self.video_stream = _FakeVideoStream(reported_frames)
        self.streams = [self.video_stream]
        self.frames = [_FakeVideoFrame(i) for i in range(decoded_frames)]
        self.decode_calls = 0
        self.closed = False

    def decode(self, stream):
        assert stream is self.video_stream
        self.decode_calls += 1
        return iter(self.frames)

    def close(self):
        self.closed = True


class _FakeAV:
    def __init__(self, container):
        self.container = container

    def open(self, path):
        assert path == "retake_input.mp4"
        return self.container


def _install_fake_av(monkeypatch, container):
    monkeypatch.setattr(media_io, "_require_av", lambda: _FakeAV(container))


def test_video_metadata_uses_container_frame_count_without_decode(monkeypatch):
    container = _FakeContainer(reported_frames=17, decoded_frames=19)
    _install_fake_av(monkeypatch, container)

    metadata = media_io.get_videostream_metadata("retake_input.mp4")

    assert metadata == VideoPixelShape(
        batch=1,
        frames=17,
        height=4,
        width=8,
        fps=float(Fraction(30000, 1001)),
    )
    assert container.decode_calls == 0
    assert container.closed


def test_video_metadata_decodes_exact_count_when_container_omits_it(monkeypatch):
    container = _FakeContainer(reported_frames=0, decoded_frames=5)
    _install_fake_av(monkeypatch, container)

    metadata = media_io.get_videostream_metadata("retake_input.mp4")

    assert metadata.frames == 5
    assert container.decode_calls == 1
    assert container.closed


def test_decode_video_by_frame_applies_index_and_cap(monkeypatch):
    container = _FakeContainer(reported_frames=4, decoded_frames=4)
    _install_fake_av(monkeypatch, container)

    frames = list(
        media_io.decode_video_by_frame(
            "retake_input.mp4",
            torch.device("cpu"),
            starting_frame=1,
            frame_cap=2,
        )
    )

    assert len(frames) == 2
    assert all(frame.shape == (1, 4, 8, 3) for frame in frames)
    assert all(frame.dtype == torch.uint8 for frame in frames)
    assert torch.all(frames[0] == 1)
    assert torch.all(frames[1] == 2)
    assert container.closed


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"starting_frame": -1}, "starting_frame"),
        ({"frame_cap": -1}, "frame_cap"),
    ],
)
def test_decode_video_by_frame_rejects_negative_indices(kwargs, message):
    with pytest.raises(ValueError, match=message):
        list(media_io.decode_video_by_frame("retake_input.mp4", "cpu", **kwargs))


def test_retake_runtime_has_no_external_ltx_imports():
    from tensorrt_llm._torch.visual_gen.models.ltx2 import pipeline_ltx2_retake

    tree = ast.parse(inspect.getsource(pipeline_ltx2_retake))
    external = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            external.extend(
                alias.name
                for alias in node.names
                if alias.name.split(".", 1)[0] in {"ltx_core", "ltx_pipelines"}
            )
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".", 1)[0] in {"ltx_core", "ltx_pipelines"}:
                external.append(node.module)
    assert not external, f"retake runtime imports external LTX packages: {external}"
