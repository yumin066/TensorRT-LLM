# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the native LTX-2 audio VAE encoder used by retake."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.audio_vae.causality_axis import (
    CausalityAxis,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.audio_vae.downsample import Downsample
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.audio_vae.model_configurator import (
    AudioEncoderConfigurator,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.media_io import _audio_frame_to_float
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.types import Audio, VideoPixelShape
from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_retake import _conform_latent_length


@pytest.mark.parametrize(
    ("axis", "expected_pad"),
    [
        (CausalityAxis.NONE, (0, 1, 0, 1)),
        (CausalityAxis.WIDTH, (2, 0, 0, 1)),
        (CausalityAxis.HEIGHT, (0, 1, 2, 0)),
        (CausalityAxis.WIDTH_COMPATIBILITY, (1, 0, 0, 1)),
    ],
)
def test_downsample_pads_asymmetrically_per_causality_axis(axis, expected_pad):
    """The whole point of the hand-rolled pad is that it is asymmetric.

    Conv2d cannot express "pad 2 on the left and 0 on the right", which is what
    keeps the convolution causal. A symmetric pad of the same total width yields
    the same output shape, so only the values reveal the error.
    """
    torch.manual_seed(0)
    down = Downsample(1, with_conv=True, causality_axis=axis)
    x = torch.randn(1, 1, 8, 8)

    expected = down.conv(torch.nn.functional.pad(x, expected_pad, mode="constant", value=0))
    assert torch.equal(down(x), expected)


def test_downsample_halves_both_axes():
    down = Downsample(3, with_conv=True, causality_axis=CausalityAxis.HEIGHT)
    out = down(torch.zeros(1, 3, 16, 32))
    assert out.shape == (1, 3, 8, 16)


def test_downsample_rejects_causality_without_conv():
    """Average pooling cannot be causal; silently pooling would break the encode."""
    with pytest.raises(ValueError, match="causality"):
        Downsample(3, with_conv=False, causality_axis=CausalityAxis.HEIGHT)


def test_conform_latent_length_crops_when_too_long():
    latent = torch.arange(4 * 6, dtype=torch.float32).reshape(1, 4, 6)[..., None]
    out = _conform_latent_length(latent, 4)
    assert out.shape[2] == 4
    assert torch.equal(out, latent[:, :, :4])


def test_conform_latent_length_zero_pads_when_too_short():
    latent = torch.ones(1, 4, 3, 2)
    out = _conform_latent_length(latent, 5)
    assert out.shape == (1, 4, 5, 2)
    assert torch.equal(out[:, :, :3], latent)
    assert torch.equal(out[:, :, 3:], torch.zeros(1, 4, 2, 2)), "pad must be zeros, not edge-repeat"
    assert out.dtype == latent.dtype


def test_conform_latent_length_is_identity_at_the_exact_length():
    latent = torch.ones(1, 4, 5, 2)
    assert _conform_latent_length(latent, 5) is latent


class _FakeFormat:
    def __init__(self, name, is_planar):
        self.name = name
        self.is_planar = is_planar


class _FakeLayout:
    def __init__(self, channels):
        self.channels = list(range(channels))


class _FakeFrame:
    def __init__(self, array, fmt_name, is_planar, channels):
        self._array = array
        self.format = _FakeFormat(fmt_name, is_planar)
        self.layout = _FakeLayout(channels)

    def to_ndarray(self):
        return self._array


def test_audio_frame_s16_is_scaled_to_unit_range():
    """s16 must divide by 32768, not 32767: a wrong divisor is a silent gain error."""
    frame = _FakeFrame(np.array([[-32768, 16384, -16384]], dtype=np.int16), "s16p", True, 1)
    out = _audio_frame_to_float(frame)
    assert np.allclose(out, [[-1.0, 0.5, -0.5]])
    assert out.dtype == np.float32


def test_audio_frame_u8_is_centered_before_scaling():
    frame = _FakeFrame(np.array([[0, 128, 255]], dtype=np.uint8), "u8p", True, 1)
    out = _audio_frame_to_float(frame)
    assert np.allclose(out, [[-1.0, 0.0, 127 / 128]])


def test_audio_frame_float_format_is_left_alone():
    frame = _FakeFrame(np.array([[0.25, -0.5]], dtype=np.float32), "fltp", True, 1)
    assert np.allclose(_audio_frame_to_float(frame), [[0.25, -0.5]])


def test_interleaved_frame_is_deinterleaved_to_channels_first():
    """Interleaved arrives as (1, samples*channels) and must become (channels, samples).

    Getting this transpose wrong swaps left/right content into a single channel's
    timeline rather than raising, so the shape alone cannot catch it.
    """
    interleaved = np.array([[1.0, 10.0, 2.0, 20.0, 3.0, 30.0]], dtype=np.float32)
    out = _audio_frame_to_float(_FakeFrame(interleaved, "flt", False, 2))
    assert out.shape == (2, 3)
    assert np.allclose(out[0], [1.0, 2.0, 3.0])
    assert np.allclose(out[1], [10.0, 20.0, 30.0])


def _encoder_config(**ddconfig_overrides):
    # `ch` sizes the per-channel statistics buffers, but those normalize the
    # *patchified* latent, whose channel count is `z_channels * output_mel_bins`.
    # The real checkpoint happens to satisfy that (128 == 8 * 16); a toy config has
    # to be chosen to satisfy it too. One downsampling level takes mel_bins 16 -> 8,
    # so 8 * 8 == 64.
    ddconfig = {
        "ch": 64,
        "ch_mult": [1, 2],
        "num_res_blocks": 1,
        "attn_resolutions": set(),
        "resolution": 32,
        "z_channels": 8,
        "in_channels": 2,
        "norm_type": "pixel",
        "causality_axis": "height",
        "mel_bins": 16,
    }
    ddconfig.update(ddconfig_overrides)
    return {
        "audio_vae": {
            "model": {"params": {"ddconfig": ddconfig, "sampling_rate": 16000}},
            "preprocessing": {"stft": {"hop_length": 160, "filter_length": 1024, "causal": True}},
        }
    }


def test_configurator_maps_the_stft_section_onto_the_mel_front_end():
    """`n_fft` comes from `stft.filter_length`, which is not a name match.

    A wrong FFT size still builds and still runs; it just produces a different
    spectrogram, so only an explicit mapping check catches it.
    """
    enc = AudioEncoderConfigurator.from_config(_encoder_config())
    assert enc.n_fft == 1024
    assert enc.mel_hop_length == 160
    assert enc.sample_rate == 16000
    assert enc.is_causal is True
    assert enc.mel_bins == 16


def test_configurator_falls_back_for_mel_bins():
    """`mel_bins` may live in any of three config sections depending on export."""
    config = _encoder_config()
    del config["audio_vae"]["model"]["params"]["ddconfig"]["mel_bins"]
    config["audio_vae"]["preprocessing"]["mel"] = {"n_mel_channels": 24}
    assert AudioEncoderConfigurator.from_config(config).mel_bins == 24

    config["audio_vae"]["preprocessing"]["mel"] = {}
    config["audio_vae"]["variables"] = {"mel_bins": 48}
    assert AudioEncoderConfigurator.from_config(config).mel_bins == 48


def test_configurator_rejects_an_unsupported_attention_type():
    """Unsupported attention must fail instead of silently changing the model."""
    with pytest.raises(ValueError, match="attn_type"):
        AudioEncoderConfigurator.from_config(_encoder_config(attn_type="none"))


@pytest.mark.parametrize("double_z", [True, False])
def test_encoder_halves_both_axes_per_level_and_emits_z_channels(double_z):
    """End-to-end forward of a tiny encoder: the shape contract the retake path assumes.

    Each downsampling level halves time and mel bins. Both deterministic
    ``double_z=False`` output and the mean half of ``double_z=True`` must expose
    exactly ``z_channels``.
    """
    enc = AudioEncoderConfigurator.from_config(_encoder_config(double_z=double_z))
    with torch.no_grad():
        enc.per_channel_statistics.get_buffer("std-of-means").fill_(1.0)
        enc.per_channel_statistics.get_buffer("mean-of-means").zero_()

    # (batch, in_channels, time, mel_bins); one downsampling level for ch_mult of length 2.
    latent = enc(torch.randn(1, 2, 64, 16))
    assert latent.shape == (1, 8, 32, 8)
    assert torch.isfinite(latent).all()


class _RetakeStub:
    """Only what `_encode_source_audio_latent` touches."""

    def __init__(self, audio, recorder):
        self._audio = audio
        self._recorder = recorder

    def _decode_audio_from_file(self, path, device, start_time=0.0, max_duration=None):
        self._recorder.append(
            {"path": path, "start_time": start_time, "max_duration": max_duration}
        )
        return self._audio


def _bound_encode(audio, recorder):
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_retake import LTX2RetakePipeline

    stub = _RetakeStub(audio, recorder)
    return LTX2RetakePipeline._encode_source_audio_latent.__get__(stub)


def test_audio_encode_reads_exactly_the_video_duration(monkeypatch):
    """Duration comes from the VIDEO shape, so the audio latent lines up token-wise.

    Reading the audio stream's own length instead would silently misalign the
    frozen audio against the video latent the transformer cross-attends to.
    """
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core import audio_vae as audio_vae_mod
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.types import AudioLatentShape

    pixel_shape = VideoPixelShape(batch=1, frames=209, height=1280, width=704, fps=25.0)
    expected_frames = AudioLatentShape.from_video_pixel_shape(pixel_shape).frames

    # Return a deliberately over-long latent so the conform step has to act.
    monkeypatch.setattr(
        audio_vae_mod,
        "encode_audio",
        lambda audio, encoder: torch.ones(1, 8, expected_frames + 7, 16),
    )

    recorder = []
    audio = Audio(waveform=torch.zeros(1, 2, 1000), sampling_rate=16000)
    out = _bound_encode(audio, recorder)(
        object(), "clip.mp4", pixel_shape, torch.device("cpu"), torch.bfloat16
    )

    assert recorder == [{"path": "clip.mp4", "start_time": 0.0, "max_duration": 209 / 25.0}]
    assert out.shape[2] == expected_frames, "latent must be conformed to the video-derived length"
    assert out.dtype == torch.bfloat16, "the frozen audio latent must carry the model dtype"


def test_audio_encode_returns_none_without_an_audio_stream(monkeypatch):
    """A silent source must disable audio conditioning, not crash or fabricate zeros."""
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core import audio_vae as audio_vae_mod

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("encode_audio must not run when there is no audio stream")

    monkeypatch.setattr(audio_vae_mod, "encode_audio", _must_not_run)

    pixel_shape = VideoPixelShape(batch=1, frames=209, height=1280, width=704, fps=25.0)
    out = _bound_encode(None, [])(
        object(), "silent.mp4", pixel_shape, torch.device("cpu"), torch.bfloat16
    )
    assert out is None
