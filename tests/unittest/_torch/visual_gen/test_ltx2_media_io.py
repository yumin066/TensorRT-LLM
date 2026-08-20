# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke tests for native LTX-2 retake video decoding."""

from fractions import Fraction

import numpy as np
import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.types import VideoPixelShape
from tensorrt_llm._torch.visual_gen.models.ltx2_retake import media_io


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
    def __init__(self, frames: int):
        self.video_stream = _FakeVideoStream(frames)
        self.streams = [self.video_stream]
        self.frames = [_FakeVideoFrame(i) for i in range(frames)]
        self.decode_calls = 0
        self.closed = False

    def decode(self, stream):
        assert stream is self.video_stream
        self.decode_calls += 1
        return iter(self.frames)

    def close(self):
        self.closed = True


def _install_fake_av(monkeypatch, container):
    fake_av = type("FakeAV", (), {"open": lambda _self, _path: container})()
    monkeypatch.setattr(media_io, "_require_av", lambda: fake_av)


def test_native_video_metadata_does_not_decode_frames(monkeypatch):
    container = _FakeContainer(frames=17)
    _install_fake_av(monkeypatch, container)

    assert media_io.get_videostream_metadata("retake_input.mp4") == VideoPixelShape(
        batch=1,
        frames=17,
        height=4,
        width=8,
        fps=float(Fraction(30000, 1001)),
    )
    assert container.decode_calls == 0
    assert container.closed


def test_native_video_decode_applies_frame_window(monkeypatch):
    container = _FakeContainer(frames=4)
    _install_fake_av(monkeypatch, container)

    frames = list(
        media_io.decode_video_by_frame(
            "retake_input.mp4",
            torch.device("cpu"),
            starting_frame=1,
            frame_cap=2,
        )
    )

    assert [tuple(frame.shape) for frame in frames] == [(1, 4, 8, 3)] * 2
    assert all(frame.dtype == torch.uint8 for frame in frames)
    assert [int(frame[0, 0, 0, 0]) for frame in frames] == [1, 2]
    assert container.closed
