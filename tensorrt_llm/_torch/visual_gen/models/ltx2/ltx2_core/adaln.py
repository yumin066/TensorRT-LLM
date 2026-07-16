# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from typing import Optional, Tuple

import torch

from .timestep_embedding import PixArtAlphaCombinedTimestepSizeEmbeddings

# Number of AdaLN modulation parameters per transformer block.
# Base: 2 params (shift + scale) x 3 norms (self-attn, feed-forward, output).
ADALN_NUM_BASE_PARAMS = 6
# Cross-attention AdaLN adds 3 more (shift, scale, gate) for the text-CA norm.
ADALN_NUM_CROSS_ATTN_PARAMS = 3


def adaln_embedding_coefficient(cross_attention_adaln: bool) -> int:
    """Total number of AdaLN scale/shift/gate rows per transformer block.

    Mirrors ``ltx_core.model.transformer.adaln.adaln_embedding_coefficient``:
    6 for the base blocks (self-attn + FFN), 9 when the block also carries
    text cross-attention AdaLN modulation (rows ``[6:9]``).
    """
    return ADALN_NUM_BASE_PARAMS + (ADALN_NUM_CROSS_ATTN_PARAMS if cross_attention_adaln else 0)


class AdaLayerNormSingle(torch.nn.Module):
    """Adaptive layer norm (adaLN-single) from PixArt-Alpha.

    Produces scale/shift/gate modulation parameters from timestep embeddings.
    """

    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6, make_linear=None):
        super().__init__()
        if make_linear is None:
            make_linear = torch.nn.Linear
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
            make_linear=make_linear,
        )
        self.silu = torch.nn.SiLU()
        self.linear = make_linear(embedding_dim, embedding_coefficient * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep
