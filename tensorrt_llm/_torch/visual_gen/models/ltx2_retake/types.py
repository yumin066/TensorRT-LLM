# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retake-specific media value types."""

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class Audio:
    """Decoded audio waveform and its sampling rate."""

    waveform: torch.Tensor
    sampling_rate: int

    def to(self, **kwargs: object) -> "Audio":
        return replace(self, waveform=self.waveform.to(**kwargs))
