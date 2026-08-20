# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import math

import torch

from ..ltx2.ltx2_core.connector import Embeddings1DConnector
from ..ltx2.ltx2_core.rope import LTXRopeType


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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normed = _norm_and_concat_padded_batch(hidden_states, attention_mask)
        features = self.aggregate_embed(normed.to(hidden_states.dtype))
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normed = norm_and_concat_per_token_rms(hidden_states, attention_mask)
        normed = normed.to(hidden_states.dtype)
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
