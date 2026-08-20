# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from typing import Any

import torch
from torch import nn

from tensorrt_llm.inputs.multimodal_data import AudioData


def _require_torchaudio() -> Any:
    """Import the optional mel-front-end dependency on demand."""
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

    def resample_audio(self, audio: AudioData) -> AudioData:
        """Resample audio to the processor's target sample rate if needed."""
        if audio.sample_rate == self.target_sample_rate:
            return audio
        torchaudio = _require_torchaudio()
        samples = torch.as_tensor(audio.samples)
        resampled = torchaudio.functional.resample(
            samples, audio.sample_rate, self.target_sample_rate
        )
        resampled = resampled.to(device=samples.device, dtype=samples.dtype)
        return AudioData(samples=resampled, sample_rate=self.target_sample_rate)

    def waveform_to_mel(self, audio: AudioData) -> torch.Tensor:
        """Convert waveform to log-mel spectrogram ``(batch, channels, time, n_mels)``."""
        waveform = torch.as_tensor(self.resample_audio(audio).samples)

        mel = self.mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        mel = mel.to(device=waveform.device, dtype=waveform.dtype)
        return mel.permute(0, 1, 3, 2).contiguous()
