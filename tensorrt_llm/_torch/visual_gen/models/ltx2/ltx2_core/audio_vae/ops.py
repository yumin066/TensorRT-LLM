# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import torch
from torch import nn

from ..types import Audio


def _require_torchaudio():
    """Import ``torchaudio`` on demand, with an actionable error if it is absent.

    Imported here rather than at module scope so that loading the LTX-2 native
    pipeline -- including the audio *decoder* and vocoder, which do not need it --
    does not hard-require torchaudio. Only the mel front-end of the audio encoder
    does.
    """
    try:
        import torchaudio
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The LTX-2 native audio encoder needs `torchaudio` for the mel "
            "spectrogram front-end. Install a torchaudio build matching the "
            "installed torch in the VisualGen runtime environment."
        ) from exc
    return torchaudio


class AudioProcessor(nn.Module):
    """Converts audio waveforms to log-mel spectrograms with optional resampling."""

    def __init__(
        self,
        target_sample_rate: int,
        mel_bins: int,
        mel_hop_length: int,
        n_fft: int,
    ) -> None:
        super().__init__()
        torchaudio = _require_torchaudio()
        self.target_sample_rate = target_sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=mel_hop_length,
            f_min=0.0,
            f_max=target_sample_rate / 2.0,
            n_mels=mel_bins,
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
        )

    def resample_audio(self, audio: Audio) -> Audio:
        """Resample audio to the processor's target sample rate if needed."""
        if audio.sampling_rate == self.target_sample_rate:
            return audio
        torchaudio = _require_torchaudio()
        resampled = torchaudio.functional.resample(
            audio.waveform, audio.sampling_rate, self.target_sample_rate
        )
        resampled = resampled.to(device=audio.waveform.device, dtype=audio.waveform.dtype)
        return Audio(waveform=resampled, sampling_rate=self.target_sample_rate)

    def waveform_to_mel(self, audio: Audio) -> torch.Tensor:
        """Convert waveform to log-mel spectrogram ``(batch, channels, time, n_mels)``."""
        waveform = self.resample_audio(audio).waveform

        mel = self.mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        mel = mel.to(device=waveform.device, dtype=waveform.dtype)
        return mel.permute(0, 1, 3, 2).contiguous()


class PerChannelStatistics(nn.Module):
    """Per-channel statistics for normalizing/denormalizing audio latents."""

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        self.register_buffer("std-of-means", torch.empty(latent_channels))
        self.register_buffer("mean-of-means", torch.empty(latent_channels))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.get_buffer("std-of-means").to(x)) + self.get_buffer("mean-of-means").to(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.get_buffer("mean-of-means").to(x)) / self.get_buffer("std-of-means").to(x)
