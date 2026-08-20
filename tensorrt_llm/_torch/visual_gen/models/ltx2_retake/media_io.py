# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""Deterministic source-media readers for the native LTX-2 retake pipeline.

Decoding is done with PyAV directly (no torchaudio), so this module is usable in
runtimes where only the audio VAE's mel front-end needs torchaudio.
"""

from collections.abc import Iterator

import numpy as np
import torch

from ..ltx2.ltx2_core.types import VideoPixelShape
from .types import Audio

# Offset and divisor that map integer sample formats to [-1, 1]. Float formats
# are already normalized and intentionally absent.
_INT_FORMAT_SCALE: dict[str, tuple[float, float]] = {
    "u8": (128.0, 128.0),
    "u8p": (128.0, 128.0),
    "s16": (0.0, 32768.0),
    "s16p": (0.0, 32768.0),
    "s32": (0.0, 2147483648.0),
    "s32p": (0.0, 2147483648.0),
}


def _require_av():
    """Import PyAV on demand, with an actionable error if it is absent."""
    try:
        import av
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "LTX-2 native source-media decode needs `av` (PyAV). Install it in "
            "the VisualGen runtime environment."
        ) from exc
    return av


def _audio_frame_to_float(frame) -> np.ndarray:
    """Convert an ``av.AudioFrame`` to float32 ``(channels, samples)`` in [-1, 1]."""
    fmt = frame.format.name
    arr = frame.to_ndarray().astype(np.float32)
    if fmt in _INT_FORMAT_SCALE:
        offset, divisor = _INT_FORMAT_SCALE[fmt]
        arr = (arr - offset) / divisor
    if not frame.format.is_planar:
        # Interleaved formats arrive as (1, samples * channels).
        channels = len(frame.layout.channels)
        arr = arr.reshape(-1, channels).T
    return arr


def get_videostream_metadata(path: str) -> VideoPixelShape:
    """Read the first video stream's shape and frame rate with PyAV.

    Container metadata does not always carry a frame count. In that case the
    stream is decoded once to obtain an exact count rather than estimating it
    from duration and frame rate.
    """
    av = _require_av()

    container = av.open(path)
    try:
        try:
            video_stream = next(s for s in container.streams if s.type == "video")
        except StopIteration as exc:
            raise ValueError(f"media file has no video stream: {path}") from exc
        if video_stream.average_rate is None:
            raise ValueError(f"video stream has no frame rate: {path}")
        fps = float(video_stream.average_rate)
        num_frames = int(video_stream.frames or 0)
        if num_frames == 0:
            num_frames = sum(1 for _ in container.decode(video_stream))
        return VideoPixelShape(
            batch=1,
            frames=num_frames,
            height=int(video_stream.codec_context.height),
            width=int(video_stream.codec_context.width),
            fps=fps,
        )
    finally:
        container.close()


def decode_video_by_frame(
    path: str,
    device: torch.device | str,
    starting_frame: int = 0,
    frame_cap: int | None = None,
) -> Iterator[torch.Tensor]:
    """Yield RGB frames by sequential frame index.

    Each yielded tensor has shape ``(1, H, W, C)`` and dtype ``uint8``. Frame
    indices are used instead of presentation timestamps so conditioning input
    selection remains deterministic for retake clips.
    """
    if starting_frame < 0:
        raise ValueError(f"starting_frame must be non-negative, got {starting_frame}")
    if frame_cap is not None and frame_cap < 0:
        raise ValueError(f"frame_cap must be non-negative, got {frame_cap}")

    av = _require_av()

    container = av.open(path)
    try:
        try:
            video_stream = next(s for s in container.streams if s.type == "video")
        except StopIteration as exc:
            raise ValueError(f"media file has no video stream: {path}") from exc
        yielded = 0
        for index, frame in enumerate(container.decode(video_stream)):
            if index < starting_frame:
                continue
            if frame_cap is not None and yielded >= frame_cap:
                break
            array = frame.to_rgb().to_ndarray()
            yield torch.tensor(array, dtype=torch.uint8, device=device).unsqueeze(0)
            yielded += 1
    finally:
        container.close()


def decode_audio_from_file(
    path: str,
    device: torch.device | str,
    start_time: float = 0.0,
    max_duration: float | None = None,
) -> Audio | None:
    """Decode an audio stream, optionally seeking and limiting duration.

    Returns an :class:`Audio` whose waveform is ``(1, channels, samples)``, or
    ``None`` when the file carries no audio stream.
    """
    if start_time < 0:
        raise ValueError(f"start_time must be non-negative, got {start_time}")
    if max_duration is not None and max_duration <= 0:
        raise ValueError(f"max_duration must be positive, got {max_duration}")

    av = _require_av()
    container = av.open(path)
    try:
        try:
            audio_stream = next(s for s in container.streams if s.type == "audio")
        except StopIteration:
            return None

        sample_rate = int(audio_stream.rate)
        start_pts = int(start_time / audio_stream.time_base)
        end_time = start_time + max_duration if max_duration is not None else float("inf")
        container.seek(start_pts, stream=audio_stream)

        samples = []
        first_frame_time = None
        for frame in container.decode(audio=0):
            if frame.pts is None:
                continue
            frame_time = float(frame.pts * audio_stream.time_base)
            frame_end = frame_time + frame.samples / frame.sample_rate
            if frame_end < start_time:
                continue
            if frame_time > end_time:
                break
            if first_frame_time is None:
                first_frame_time = frame_time
            samples.append(_audio_frame_to_float(frame))
    finally:
        container.close()

    if not samples:
        return None

    audio = np.concatenate(samples, axis=-1)

    # Codec frame boundaries need not align with the requested time range.
    skip_samples = round((start_time - first_frame_time) * sample_rate)
    if skip_samples > 0:
        audio = audio[..., skip_samples:]

    if max_duration is not None:
        max_samples = round(max_duration * sample_rate)
        audio = audio[..., :max_samples]

    waveform = torch.from_numpy(audio).to(device).unsqueeze(0)
    return Audio(waveform=waveform, sampling_rate=sample_rate)
