# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the LTX-2 native Gemma feature extractor.

These tests cover the config-driven V1/V2 selection, the two normalization
formulas (V1 masked mean+range vs V2 per-token RMS), the projection head shapes
and bias flags, and the submodule names that must line up with the checkpoint
key suffixes so weights load without missing/unexpected keys.

All construction and math is CPU-only and uses tiny tensors (the norm helpers
are exercised directly, and the ``forward`` routing is checked with small
custom projection heads), so no GPU or model checkpoint is needed.
"""

import math

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.connector import (
    FeatureExtractorV1,
    FeatureExtractorV2,
    GemmaFeaturesExtractorConfigurator,
    _norm_and_concat_padded_batch,
    _rescale_norm,
    norm_and_concat_per_token_rms,
)

# Fixed Gemma-3-12b constants used by the LTX-2 feature extractor.
EMBEDDING_DIM = 3840
NUM_LAYERS = 49
FLAT_DIM = EMBEDDING_DIM * NUM_LAYERS  # 188160

# The four V2 gating keys with their exact expected values.
V2_KEYS = {
    "caption_proj_before_connector": True,
    "caption_projection_first_linear": False,
    "caption_proj_input_norm": False,
    "caption_projection_second_linear": False,
}

# 22b head dimensions: video 32 x 128 = 4096, audio 32 x 64 = 2048.
V2_HEAD_DIMS = {
    "num_attention_heads": 32,
    "attention_head_dim": 128,
    "audio_num_attention_heads": 32,
    "audio_attention_head_dim": 64,
}
V2_VIDEO_INNER_DIM = 4096
V2_AUDIO_INNER_DIM = 2048


def _v2_transformer_config(**overrides):
    cfg = {**V2_KEYS, **V2_HEAD_DIMS}
    cfg.update(overrides)
    return {"transformer": cfg}


# ---------------------------------------------------------------------------
# (a) V1 selection: single bias-free head, shared video/audio features
# ---------------------------------------------------------------------------


def test_v1_config_builds_single_aggregate_embed():
    fe = GemmaFeaturesExtractorConfigurator.from_config({"transformer": {}})

    assert isinstance(fe, FeatureExtractorV1)
    assert fe.is_av is True
    assert isinstance(fe.aggregate_embed, torch.nn.Linear)
    assert fe.aggregate_embed.in_features == FLAT_DIM
    assert fe.aggregate_embed.out_features == EMBEDDING_DIM
    assert fe.aggregate_embed.bias is None


def test_v1_config_with_empty_top_level_dict_builds_v1():
    # A checkpoint that carries no "transformer" section at all is still V1.
    fe = GemmaFeaturesExtractorConfigurator.from_config({})
    assert isinstance(fe, FeatureExtractorV1)


def test_v1_returns_same_features_for_video_and_audio():
    # Use an identity projection over a tiny flat dim so we can inspect the
    # normalized output directly and confirm both heads share one tensor.
    b, t, d, num_layers = 1, 3, 2, 2
    fe = FeatureExtractorV1(aggregate_embed=torch.nn.Identity(), is_av=True)

    encoded = torch.randn(b, t, d, num_layers)
    mask = torch.ones(b, t, dtype=torch.long)

    video, audio = fe(encoded, mask)
    assert video is audio
    expected = _norm_and_concat_padded_batch(encoded, mask)
    assert torch.equal(video, expected)


def test_v1_norm_matches_masked_mean_range_formula():
    # Known input, all tokens valid -> compare against the explicit formula.
    encoded = torch.tensor([[[[1.0], [3.0]], [[5.0], [7.0]]]])  # [B=1, T=2, D=2, L=1]
    mask = torch.ones(1, 2, dtype=torch.long)

    normed = _norm_and_concat_padded_batch(encoded, mask)

    eps = 1e-6
    mean = encoded.mean()
    rng = encoded.max() - encoded.min()
    expected = (encoded - mean) / (rng + eps) * 8
    expected = expected.reshape(1, 2, 2)
    assert torch.allclose(normed, expected, atol=1e-5)

    # And it must NOT match the per-token RMS output for this input.
    rms = norm_and_concat_per_token_rms(encoded, mask)
    assert not torch.allclose(normed, rms, atol=1e-3)


def test_v1_norm_zeroes_padded_tokens():
    encoded = torch.randn(1, 4, 2, 2)
    # Left-padded: first two tokens are padding.
    mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    normed = _norm_and_concat_padded_batch(encoded, mask)
    assert torch.count_nonzero(normed[0, 0]) == 0
    assert torch.count_nonzero(normed[0, 1]) == 0


# ---------------------------------------------------------------------------
# (b) V2 selection: dual heads with bias, per-token RMS + rescale
# ---------------------------------------------------------------------------


def test_v2_config_builds_dual_aggregate_embeds():
    fe = GemmaFeaturesExtractorConfigurator.from_config(_v2_transformer_config())

    assert isinstance(fe, FeatureExtractorV2)
    assert fe.embedding_dim == EMBEDDING_DIM

    assert fe.video_aggregate_embed.in_features == FLAT_DIM
    assert fe.video_aggregate_embed.out_features == V2_VIDEO_INNER_DIM
    assert fe.video_aggregate_embed.bias is not None

    assert fe.audio_aggregate_embed.in_features == FLAT_DIM
    assert fe.audio_aggregate_embed.out_features == V2_AUDIO_INNER_DIM
    assert fe.audio_aggregate_embed.bias is not None


def test_v2_norm_matches_per_token_rms_formula():
    encoded = torch.tensor([[[[1.0], [3.0]], [[5.0], [7.0]]]])  # [B=1, T=2, D=2, L=1]
    mask = torch.ones(1, 2, dtype=torch.long)

    normed = norm_and_concat_per_token_rms(encoded, mask)

    variance = torch.mean(encoded**2, dim=2, keepdim=True)
    expected = (encoded * torch.rsqrt(variance + 1e-6)).reshape(1, 2, 2)
    assert torch.allclose(normed, expected, atol=1e-5)

    # And it must NOT match the V1 mean/range output for this input.
    mean_range = _norm_and_concat_padded_batch(encoded, mask)
    assert not torch.allclose(normed, mean_range, atol=1e-3)


def test_v2_returns_separate_video_and_audio_features():
    b, t, d, num_layers = 1, 3, 2, 2
    flat = d * num_layers
    video_head = torch.nn.Linear(flat, 5, bias=True)
    audio_head = torch.nn.Linear(flat, 7, bias=True)
    fe = FeatureExtractorV2(
        video_aggregate_embed=video_head,
        embedding_dim=d,
        audio_aggregate_embed=audio_head,
    )

    encoded = torch.randn(b, t, d, num_layers)
    mask = torch.ones(b, t, dtype=torch.long)

    video, audio = fe(encoded, mask)
    assert video.shape == (b, t, 5)
    assert audio.shape == (b, t, 7)
    assert video is not audio


def test_v2_applies_rescale_before_projection():
    # An identity-weight video head with out_features == embedding_dim makes the
    # output exactly the rescaled RMS-normed tensor (rescale factor sqrt(1) == 1).
    b, t, d, num_layers = 1, 2, 3, 1
    flat = d * num_layers
    video_head = torch.nn.Linear(flat, flat, bias=False)
    torch.nn.init.eye_(video_head.weight)
    fe = FeatureExtractorV2(
        video_aggregate_embed=video_head,
        embedding_dim=flat,  # source_dim == target_dim
        audio_aggregate_embed=None,
    )

    encoded = torch.randn(b, t, d, num_layers)
    mask = torch.ones(b, t, dtype=torch.long)

    video, audio = fe(encoded, mask)
    assert audio is None

    normed = norm_and_concat_per_token_rms(encoded, mask).to(encoded.dtype)
    expected = _rescale_norm(normed, flat, flat)
    assert torch.allclose(video, expected, atol=1e-5)


def test_rescale_norm_scales_by_sqrt_ratio():
    x = torch.ones(2, 3)
    out = _rescale_norm(x, target_dim=8, source_dim=2)
    assert torch.allclose(out, x * math.sqrt(4.0))


# ---------------------------------------------------------------------------
# (c) Partial / mismatched V2 config -> NotImplementedError
# ---------------------------------------------------------------------------


def test_partial_v2_config_raises():
    with pytest.raises(NotImplementedError):
        GemmaFeaturesExtractorConfigurator.from_config(
            {"transformer": {"caption_proj_before_connector": True}}
        )


def test_mismatched_v2_value_raises():
    cfg = _v2_transformer_config(caption_proj_before_connector=False)
    with pytest.raises(NotImplementedError):
        GemmaFeaturesExtractorConfigurator.from_config(cfg)


# ---------------------------------------------------------------------------
# (d) Submodule names line up with checkpoint key suffixes
# ---------------------------------------------------------------------------


def test_v1_submodule_names_match_checkpoint_suffixes():
    # The native loader strips the "text_embedding_projection." prefix, so the
    # module parameter names must equal the checkpoint key suffixes.
    fe = GemmaFeaturesExtractorConfigurator.from_config({"transformer": {}})
    names = {name for name, _ in fe.named_parameters()}
    assert names == {"aggregate_embed.weight"}


def test_v2_submodule_names_match_checkpoint_suffixes():
    fe = GemmaFeaturesExtractorConfigurator.from_config(_v2_transformer_config())
    names = {name for name, _ in fe.named_parameters()}
    assert names == {
        "video_aggregate_embed.weight",
        "video_aggregate_embed.bias",
        "audio_aggregate_embed.weight",
        "audio_aggregate_embed.bias",
    }


def test_v2_load_state_dict_has_no_missing_or_unexpected_keys():
    fe = GemmaFeaturesExtractorConfigurator.from_config(_v2_transformer_config())
    # Simulate the stripped checkpoint state dict for the V2 dual heads.
    stripped = {
        "video_aggregate_embed.weight": torch.zeros(V2_VIDEO_INNER_DIM, FLAT_DIM),
        "video_aggregate_embed.bias": torch.zeros(V2_VIDEO_INNER_DIM),
        "audio_aggregate_embed.weight": torch.zeros(V2_AUDIO_INNER_DIM, FLAT_DIM),
        "audio_aggregate_embed.bias": torch.zeros(V2_AUDIO_INNER_DIM),
    }
    missing, unexpected = fe.load_state_dict(stripped, strict=False)
    assert missing == []
    assert unexpected == []
