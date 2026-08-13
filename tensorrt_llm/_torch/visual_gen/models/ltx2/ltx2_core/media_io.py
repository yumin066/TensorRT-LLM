# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""Deterministic source-media readers for the LTX-2 native pipelines.

Decoding is done with PyAV directly (no torchaudio), so this module is usable in
runtimes where only the audio VAE's mel front-end needs torchaudio.
"""

import numpy as np
import torch

from .types import Audio

# Divisor that maps each integer sample format to [-1, 1]. Float formats ("flt",
# "fltp", "dbl", "dblp") are already in range and are absent by design.
_INT_FORMAT_MAX: dict[str, float] = {
    "u8": 128.0,
    "u8p": 128.0,
    "s16": 32768.0,
    "s16p": 32768.0,
    "s32": 2147483648.0,
    "s32p": 2147483648.0,
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
    if fmt in _INT_FORMAT_MAX:
        arr = arr / _INT_FORMAT_MAX[fmt]
    if not frame.format.is_planar:
        # Interleaved formats arrive as (1, samples * channels).
        channels = len(frame.layout.channels)
        arr = arr.reshape(-1, channels).T
    return arr


def decode_audio_from_file(
    path: str,
    device: torch.device,
    start_time: float = 0.0,
    max_duration: float | None = None,
) -> Audio | None:
    """Decode an audio stream, optionally seeking and limiting duration.

    Returns an :class:`Audio` whose waveform is ``(1, channels, samples)``, or
    ``None`` when the file carries no audio stream.
    """
    av = _require_av()

    container = av.open(path)
    try:
        audio_stream = next(s for s in container.streams if s.type == "audio")
    except StopIteration:
        container.close()
        return None

    sample_rate = audio_stream.rate
    start_pts = int(start_time / audio_stream.time_base)
    end_time = (
        start_time + max_duration
        if max_duration
        else audio_stream.duration * audio_stream.time_base
    )
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

    container.close()

    if not samples:
        return None

    audio = np.concatenate(samples, axis=-1)

    # Trim samples outside [start_time, start_time + max_duration]. Codecs decode
    # in fixed-size frames whose boundaries need not align with the requested
    # range, so the first frame can start early and the last can end late.
    skip_samples = round((start_time - first_frame_time) * sample_rate)
    if skip_samples > 0:
        audio = audio[..., skip_samples:]

    if max_duration is not None:
        max_samples = round(max_duration * sample_rate)
        audio = audio[..., :max_samples]

    waveform = torch.from_numpy(audio).to(device).unsqueeze(0)
    return Audio(waveform=waveform, sampling_rate=sample_rate)
