# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Modality:
    """Input data for a single modality (video or audio) in the transformer.

    Bundles the latent tokens, timestep embeddings, positional information,
    and text conditioning context for processing by the diffusion transformer.
    """

    latent: torch.Tensor  # (B, T, D): packed latent tokens
    timesteps: torch.Tensor  # (B,) or (B, T): per-batch or per-token timesteps
    positions: torch.Tensor  # (B, n_dims, T) or (B, n_dims, T, 2): index grid
    context: torch.Tensor  # Text embeddings
    enabled: bool = True
    context_mask: torch.Tensor | None = None
    # (B,): current sigma (noise level). Only consumed by prompt AdaLN
    # (cross_attention_adaln=True checkpoints, e.g. LTX-2.3 22b distilled) to
    # derive the per-batch prompt timestep. None for models without prompt
    # AdaLN, where it is unused.
    sigma: torch.Tensor | None = None
