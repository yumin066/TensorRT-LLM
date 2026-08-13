# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import math

import torch

from .attention import Attention, FeedForward
from .rope import LTXRopeType, precompute_freqs_cis
from .rope import _generate_freq_grid_np as generate_freq_grid_np
from .rope import _generate_freq_grid_pytorch as generate_freq_grid_pytorch
from .utils_ltx2 import rms_norm


class _BasicTransformerBlock1D(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        apply_gated_attention: bool = False,
    ):
        super().__init__()
        self.attn1 = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            rope_type=rope_type,
            apply_gated_attention=apply_gated_attention,
        )
        self.ff = FeedForward(dim, dim_out=dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm_hidden_states = rms_norm(hidden_states)
        norm_hidden_states = norm_hidden_states.squeeze(1)
        attn_output = self.attn1(norm_hidden_states, mask=attention_mask, pe=pe)
        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        norm_hidden_states = rms_norm(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class Embeddings1DConnector(torch.nn.Module):
    """1D transformer-based connector for sequential embeddings."""

    def __init__(
        self,
        attention_head_dim: int = 128,
        num_attention_heads: int = 30,
        num_layers: int = 2,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        causal_temporal_positioning: bool = False,
        num_learnable_registers: int | None = 128,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.causal_temporal_positioning = causal_temporal_positioning
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = (
            positional_embedding_max_pos if positional_embedding_max_pos is not None else [1]
        )
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.transformer_1d_blocks = torch.nn.ModuleList(
            [
                _BasicTransformerBlock1D(
                    dim=self.inner_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    rope_type=rope_type,
                    apply_gated_attention=apply_gated_attention,
                )
                for _ in range(num_layers)
            ]
        )
        self.num_learnable_registers = num_learnable_registers
        if self.num_learnable_registers:
            self.learnable_registers = torch.nn.Parameter(
                torch.rand(self.num_learnable_registers, self.inner_dim, dtype=torch.bfloat16) * 2.0
                - 1.0
            )

    def _replace_padded_with_learnable_registers(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, D = hidden_states.shape
        assert S % self.num_learnable_registers == 0

        num_registers_duplications = S // self.num_learnable_registers
        learnable_registers = torch.tile(
            self.learnable_registers, (num_registers_duplications, 1)
        ).to(hidden_states.dtype)  # [S, D]

        # [B, S] binary: True for valid tokens, False for padding
        mask_2d = attention_mask.squeeze(1).squeeze(1) >= -9000.0

        results = []
        for b in range(B):
            valid_mask = mask_2d[b]  # [S]
            valid_tokens = hidden_states[b, valid_mask, :]  # [N_valid, D]
            pad_length = S - valid_tokens.shape[0]
            padded = torch.nn.functional.pad(
                valid_tokens, pad=(0, 0, 0, pad_length), value=0
            )  # [S, D]
            flipped = torch.flip(
                valid_mask.to(hidden_states.dtype).unsqueeze(-1), dims=[0]
            )  # [S, 1]
            results.append(flipped * padded + (1 - flipped) * learnable_registers)

        hidden_states = torch.stack(results, dim=0)
        attention_mask = torch.full_like(attention_mask, 0.0)
        return hidden_states, attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_learnable_registers:
            hidden_states, attention_mask = self._replace_padded_with_learnable_registers(
                hidden_states, attention_mask
            )

        indices_grid = torch.arange(
            hidden_states.shape[1],
            dtype=torch.float32,
            device=hidden_states.device,
        )
        indices_grid = indices_grid[None, None, :]
        freq_grid_generator = (
            generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        )
        freqs_cis = precompute_freqs_cis(
            indices_grid=indices_grid,
            dim=self.inner_dim,
            out_dtype=hidden_states.dtype,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )

        for block in self.transformer_1d_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask, pe=freqs_cis)

        hidden_states = rms_norm(hidden_states)
        return hidden_states, attention_mask


class Embeddings1DConnectorConfigurator:
    @classmethod
    def from_config(cls, config: dict) -> Embeddings1DConnector:
        config = config.get("transformer", {})
        rope_type = LTXRopeType(config.get("rope_type", "interleaved"))
        double_precision_rope = config.get("frequencies_precision", False) == "float64"
        pe_max_pos = config.get("connector_positional_embedding_max_pos", [1])
        # Thread the connector's own attention dimensions from the checkpoint
        # config. Checkpoints that enable the embeddings connector size it
        # independently of the text-projection width (e.g. the LTX-2.3 22b uses
        # 32 heads x 128 = 4096); without these the connector falls back to the
        # 30-head (3840) default and fails to load the connector weights.
        return Embeddings1DConnector(
            num_attention_heads=config.get("connector_num_attention_heads", 30),
            attention_head_dim=config.get("connector_attention_head_dim", 128),
            num_layers=config.get("connector_num_layers", 2),
            positional_embedding_max_pos=pe_max_pos,
            rope_type=rope_type,
            double_precision_rope=double_precision_rope,
            apply_gated_attention=config.get("connector_apply_gated_attention", False),
        )


class AudioEmbeddings1DConnectorConfigurator:
    @classmethod
    def from_config(cls, config: dict) -> Embeddings1DConnector:
        config = config.get("transformer", {})
        rope_type = LTXRopeType(config.get("rope_type", "interleaved"))
        double_precision_rope = config.get("frequencies_precision", False) == "float64"
        pe_max_pos = config.get("connector_positional_embedding_max_pos", [1])
        # The audio embeddings connector is sized independently of the video one
        # (the LTX-2.3 22b uses 32 heads x 64 = 2048 vs the video connector's
        # 32 x 128 = 4096); read the ``audio_connector_*`` dimensions and fall
        # back to the video connector fields for checkpoints that share them.
        return Embeddings1DConnector(
            num_attention_heads=config.get(
                "audio_connector_num_attention_heads",
                config.get("connector_num_attention_heads", 30),
            ),
            attention_head_dim=config.get(
                "audio_connector_attention_head_dim",
                config.get("connector_attention_head_dim", 128),
            ),
            num_layers=config.get(
                "audio_connector_num_layers",
                config.get("connector_num_layers", 2),
            ),
            positional_embedding_max_pos=pe_max_pos,
            rope_type=rope_type,
            double_precision_rope=double_precision_rope,
            apply_gated_attention=config.get("connector_apply_gated_attention", False),
        )


# ---------------------------------------------------------------------------
# Gemma feature extraction: normalization helpers
# ---------------------------------------------------------------------------

# Constants from the fixed Gemma-3-12b text config used by LTX-2. The stacked
# hidden states carry ``num_hidden_layers`` (48) transformer layers plus the
# embedding layer, hence 49 layers of width ``hidden_size`` (3840).
_GEMMA_EMBEDDING_DIM = 3840
_GEMMA_NUM_LAYERS = 49
_GEMMA_FLAT_DIM = _GEMMA_EMBEDDING_DIM * _GEMMA_NUM_LAYERS

_NORM_EPS = 1e-6
_V1_SCALE_FACTOR = 8


def _norm_and_concat_padded_batch(
    encoded_text: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """V1 per-batch/per-layer masked mean+range normalization.

    Normalizes ``[B, T, D, L]`` hidden states using a masked mean and range
    computed over the valid (batch, hidden) positions of each layer, then
    flattens the layer dimension to ``[B, T, D * L]`` and zeroes padded tokens.
    The binary ``attention_mask`` ([B, T], 1 for valid tokens) fully encodes
    the padding, so the result is padding-side agnostic.
    """
    b, _, d, num_layers = encoded_text.shape
    sequence_lengths = attention_mask.sum(dim=-1)
    mask = attention_mask.bool().view(b, -1, 1, 1)

    masked = encoded_text.masked_fill(~mask, 0.0)
    denom = (sequence_lengths * d).view(b, 1, 1, 1)
    mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + _NORM_EPS)

    x_min = encoded_text.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = encoded_text.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)

    normed = (encoded_text - mean) / (x_max - x_min + _NORM_EPS)
    normed = normed * _V1_SCALE_FACTOR
    normed = normed.reshape(b, -1, d * num_layers)

    mask_flattened = mask.view(b, -1, 1).expand(-1, -1, d * num_layers)
    return normed.masked_fill(~mask_flattened, 0.0)


def norm_and_concat_per_token_rms(
    encoded_text: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """V2 per-token RMSNorm over the hidden dimension.

    For ``[B, T, D, L]`` hidden states, computes the variance over the hidden
    dimension ``D`` (per token, per layer), normalizes, flattens to
    ``[B, T, D * L]`` and zeroes padded tokens.
    """
    b, t, d, num_layers = encoded_text.shape
    variance = torch.mean(encoded_text**2, dim=2, keepdim=True)  # [B, T, 1, L]
    normed = encoded_text * torch.rsqrt(variance + _NORM_EPS)
    normed = normed.reshape(b, t, d * num_layers)
    mask_3d = attention_mask.bool().unsqueeze(-1)  # [B, T, 1]
    return torch.where(mask_3d, normed, torch.zeros_like(normed))


def _rescale_norm(x: torch.Tensor, target_dim: int, source_dim: int) -> torch.Tensor:
    """Rescale normalization: ``x * sqrt(target_dim / source_dim)``."""
    return x * math.sqrt(target_dim / source_dim)


# ---------------------------------------------------------------------------
# Gemma feature extractor variants
# ---------------------------------------------------------------------------


class FeatureExtractorV1(torch.nn.Module):
    """Single-head feature extractor (19b/generation checkpoints).

    Applies masked mean+range normalization, then a single ``aggregate_embed``
    projection. When ``is_av`` is set the same projected features drive both the
    video and audio connectors.
    """

    def __init__(self, aggregate_embed: torch.nn.Module, is_av: bool = False):
        super().__init__()
        self.aggregate_embed = aggregate_embed
        self.is_av = is_av

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_side: str = "left",  # noqa: ARG002 - kept for API parity; norm is layout-agnostic
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoded = (
            torch.stack(hidden_states, dim=-1)
            if isinstance(hidden_states, (list, tuple))
            else hidden_states
        )
        dtype = encoded.dtype
        normed = _norm_and_concat_padded_batch(encoded, attention_mask)
        features = self.aggregate_embed(normed.to(dtype))
        if self.is_av:
            return features, features
        return features, None


class FeatureExtractorV2(torch.nn.Module):
    """Dual-head feature extractor (22b checkpoints).

    Applies per-token RMS normalization, then rescales and projects into
    separate video and (optional) audio inner dimensions.
    """

    def __init__(
        self,
        video_aggregate_embed: torch.nn.Linear,
        embedding_dim: int,
        audio_aggregate_embed: torch.nn.Linear | None = None,
    ):
        super().__init__()
        self.video_aggregate_embed = video_aggregate_embed
        self.audio_aggregate_embed = audio_aggregate_embed
        self.embedding_dim = embedding_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_side: str = "left",  # noqa: ARG002 - kept for API parity; norm is layout-agnostic
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoded = (
            torch.stack(hidden_states, dim=-1)
            if isinstance(hidden_states, (list, tuple))
            else hidden_states
        )
        normed = norm_and_concat_per_token_rms(encoded, attention_mask)
        normed = normed.to(encoded.dtype)
        v_dim = self.video_aggregate_embed.out_features
        video = self.video_aggregate_embed(_rescale_norm(normed, v_dim, self.embedding_dim))
        audio = None
        if self.audio_aggregate_embed is not None:
            a_dim = self.audio_aggregate_embed.out_features
            audio = self.audio_aggregate_embed(_rescale_norm(normed, a_dim, self.embedding_dim))
        return video, audio


_V2_EXPECTED_CONFIG = {
    "caption_proj_before_connector": True,
    "caption_projection_first_linear": False,
    "caption_proj_input_norm": False,
    "caption_projection_second_linear": False,
}


class GemmaFeaturesExtractorConfigurator:
    """Config-driven factory selecting the V1 or V2 Gemma feature extractor.

    Selection rules:
    - none of the V2 keys present -> ``FeatureExtractorV1`` (projection lives in
      the feature extractor, single ``aggregate_embed`` head, drives both
      connectors).
    - all V2 keys present with their exact expected values -> ``FeatureExtractorV2``
      (dual ``video_aggregate_embed`` / ``audio_aggregate_embed`` heads).
    - any partial or mismatched V2 config -> ``NotImplementedError``.
    """

    @classmethod
    def from_config(cls, config: dict) -> torch.nn.Module:
        transformer_config = config.get("transformer", {})

        overlapping_keys = transformer_config.keys() & _V2_EXPECTED_CONFIG.keys()
        if not overlapping_keys:
            aggregate_embed = torch.nn.Linear(_GEMMA_FLAT_DIM, _GEMMA_EMBEDDING_DIM, bias=False)
            return FeatureExtractorV1(aggregate_embed=aggregate_embed, is_av=True)

        missing_keys = _V2_EXPECTED_CONFIG.keys() - overlapping_keys
        if missing_keys:
            raise NotImplementedError(
                "Partial V2 config - missing keys: " + ", ".join(sorted(missing_keys))
            )

        unexpected_value_keys = {
            k for k in overlapping_keys if transformer_config[k] != _V2_EXPECTED_CONFIG[k]
        }
        if unexpected_value_keys:
            raise NotImplementedError(
                "Unknown config: "
                + ", ".join(
                    f"{k}={transformer_config[k]!r} (expected {_V2_EXPECTED_CONFIG[k]!r})"
                    for k in sorted(unexpected_value_keys)
                )
            )

        video_inner_dim = (
            transformer_config["num_attention_heads"] * transformer_config["attention_head_dim"]
        )
        audio_inner_dim = (
            transformer_config["audio_num_attention_heads"]
            * transformer_config["audio_attention_head_dim"]
        )
        return FeatureExtractorV2(
            video_aggregate_embed=torch.nn.Linear(_GEMMA_FLAT_DIM, video_inner_dim, bias=True),
            embedding_dim=_GEMMA_EMBEDDING_DIM,
            audio_aggregate_embed=torch.nn.Linear(_GEMMA_FLAT_DIM, audio_inner_dim, bias=True),
        )
