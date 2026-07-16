# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LTX-2 text cross-attention AdaLN (``cross_attention_adaln``).

The LTX-2.3 22b distilled checkpoint sets ``cross_attention_adaln=True``, which:
- enlarges every block's ``scale_shift_table`` from 6 to 9 rows
  ([0:3]=self-attn, [3:6]=FFN, [6:9]=text cross-attn query modulation),
- sizes the root ``adaln_single`` with coefficient 9 and adds a
  ``prompt_adaln_single`` (coefficient 2),
- adds per-block ``prompt_scale_shift_table`` params that (with a per-batch
  prompt timestep derived from ``Modality.sigma``) modulate the text context.

The 19b (``cross_attention_adaln=False``) must be unchanged (6 rows, no prompt
tables). Structure/shape assertions run on CPU (no GPU, no weights); the forward
path is CUDA-gated.
"""

import unittest

import pytest
import torch

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.adaln import adaln_embedding_coefficient
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import AttentionConfig

# Reduced VideoOnly config (mirrors test_ltx2_transformer.py).
VIDEO_ONLY_CONFIG = dict(
    num_attention_heads=4,
    attention_head_dim=32,
    in_channels=16,
    out_channels=16,
    num_layers=1,
    cross_attention_dim=128,
    caption_channels=64,
    norm_eps=1e-6,
    positional_embedding_max_pos=[4, 32, 32],
    timestep_scale_multiplier=1000,
    use_middle_indices_grid=True,
)

AUDIO_VIDEO_CONFIG = dict(
    **VIDEO_ONLY_CONFIG,
    audio_num_attention_heads=4,
    audio_attention_head_dim=16,
    audio_in_channels=16,
    audio_out_channels=16,
    audio_cross_attention_dim=64,
    audio_positional_embedding_max_pos=[4],
    av_ca_timestep_scale_multiplier=1,
)

V_INNER = VIDEO_ONLY_CONFIG["num_attention_heads"] * VIDEO_ONLY_CONFIG["attention_head_dim"]
A_INNER = (
    AUDIO_VIDEO_CONFIG["audio_num_attention_heads"] * AUDIO_VIDEO_CONFIG["audio_attention_head_dim"]
)


def _create_model_config(backend: str = "VANILLA") -> DiffusionModelConfig:
    from types import SimpleNamespace

    return DiffusionModelConfig(
        pretrained_config=SimpleNamespace(),
        quant_config=QuantConfig(),
        mapping=Mapping(),
        attention=AttentionConfig(backend=backend),
        skip_create_weights_in_init=False,
    )


def _build_model(model_type, cross_attention_adaln, config, device="cpu"):
    from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTXModel

    return LTXModel(
        model_type=model_type,
        model_config=_create_model_config(),
        cross_attention_adaln=cross_attention_adaln,
        **config,
    ).to(device)


def _init_all_weights(model: torch.nn.Module, std: float = 0.02):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and "weight" in name:
                p.fill_(1.0)
            elif p.numel() > 0:
                torch.nn.init.normal_(p, mean=0.0, std=std)


def _make_video_positions(batch, n_patches, n_frames, grid_h, grid_w, device):
    positions = torch.zeros(batch, 3, n_patches, 2, device=device)
    idx = 0
    for f in range(n_frames):
        for h in range(grid_h):
            for w in range(grid_w):
                positions[:, 0, idx, :] = torch.tensor([f, f + 1], dtype=torch.float32)
                positions[:, 1, idx, :] = torch.tensor([h, h + 1], dtype=torch.float32)
                positions[:, 2, idx, :] = torch.tensor([w, w + 1], dtype=torch.float32)
                idx += 1
    return positions


class TestAdalnEmbeddingCoefficient(unittest.TestCase):
    def test_coefficient(self):
        self.assertEqual(adaln_embedding_coefficient(False), 6)
        self.assertEqual(adaln_embedding_coefficient(True), 9)


class TestCrossAttentionAdalnStructure(unittest.TestCase):
    """CPU-only structure/shape checks for both branches."""

    def test_video_only_false_is_six_rows_no_prompt(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTXModelType

        model = _build_model(LTXModelType.VideoOnly, False, VIDEO_ONLY_CONFIG)

        # Root AdaLN: coefficient 6.
        self.assertEqual(model.adaln_single.linear.weight.shape[0], 6 * V_INNER)
        self.assertIsNone(model.prompt_adaln_single)

        # Per-block table: 6 rows, no prompt table.
        block = model.transformer_blocks[0]
        self.assertFalse(block.cross_attention_adaln)
        self.assertEqual(tuple(block.scale_shift_table.shape), (6, V_INNER))
        self.assertFalse(hasattr(block, "prompt_scale_shift_table"))

    def test_video_only_true_is_nine_rows_with_prompt(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTXModelType

        model = _build_model(LTXModelType.VideoOnly, True, VIDEO_ONLY_CONFIG)

        # Root AdaLN: coefficient 9; prompt AdaLN coefficient 2.
        self.assertEqual(model.adaln_single.linear.weight.shape[0], 9 * V_INNER)
        self.assertIsNotNone(model.prompt_adaln_single)
        self.assertEqual(model.prompt_adaln_single.linear.weight.shape[0], 2 * V_INNER)

        # Per-block table: 9 rows + prompt table with 2 rows.
        block = model.transformer_blocks[0]
        self.assertTrue(block.cross_attention_adaln)
        self.assertEqual(tuple(block.scale_shift_table.shape), (9, V_INNER))
        self.assertTrue(hasattr(block, "prompt_scale_shift_table"))
        self.assertEqual(tuple(block.prompt_scale_shift_table.shape), (2, V_INNER))

    def test_audio_video_false(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTXModelType

        model = _build_model(LTXModelType.AudioVideo, False, AUDIO_VIDEO_CONFIG)

        self.assertEqual(model.adaln_single.linear.weight.shape[0], 6 * V_INNER)
        self.assertEqual(model.audio_adaln_single.linear.weight.shape[0], 6 * A_INNER)
        self.assertIsNone(model.prompt_adaln_single)
        self.assertIsNone(model.audio_prompt_adaln_single)

        block = model.transformer_blocks[0]
        self.assertEqual(tuple(block.scale_shift_table.shape), (6, V_INNER))
        self.assertEqual(tuple(block.audio_scale_shift_table.shape), (6, A_INNER))
        self.assertFalse(hasattr(block, "prompt_scale_shift_table"))
        self.assertFalse(hasattr(block, "audio_prompt_scale_shift_table"))

    def test_audio_video_true(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTXModelType

        model = _build_model(LTXModelType.AudioVideo, True, AUDIO_VIDEO_CONFIG)

        self.assertEqual(model.adaln_single.linear.weight.shape[0], 9 * V_INNER)
        self.assertEqual(model.audio_adaln_single.linear.weight.shape[0], 9 * A_INNER)
        self.assertIsNotNone(model.prompt_adaln_single)
        self.assertIsNotNone(model.audio_prompt_adaln_single)
        self.assertEqual(model.prompt_adaln_single.linear.weight.shape[0], 2 * V_INNER)
        self.assertEqual(model.audio_prompt_adaln_single.linear.weight.shape[0], 2 * A_INNER)

        block = model.transformer_blocks[0]
        self.assertEqual(tuple(block.scale_shift_table.shape), (9, V_INNER))
        self.assertEqual(tuple(block.audio_scale_shift_table.shape), (9, A_INNER))
        self.assertEqual(tuple(block.prompt_scale_shift_table.shape), (2, V_INNER))
        self.assertEqual(tuple(block.audio_prompt_scale_shift_table.shape), (2, A_INNER))


class TestFfnSliceSelection(unittest.TestCase):
    """The FFN AdaLN rows must be [3:6] regardless of table height.

    Under the old ``slice(3, None)``, a 9-row table selects rows 3..8 (six
    values) which unpacks to the wrong tuple; ``slice(3, 6)`` always yields
    exactly (row3, row4, row5).
    """

    def test_slice_3_6_selects_ffn_rows_of_nine_row_table(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import (
            BasicAVTransformerBlock,
        )

        dim = 4
        num_params = 9
        # Distinct, easily-identifiable per-row values.
        table = torch.arange(num_params, dtype=torch.float32).view(num_params, 1).repeat(1, dim)
        batch = 2
        # timestep zero -> ada_values equal the table rows exactly.
        timestep = torch.zeros(batch, 1, num_params * dim)

        vals = BasicAVTransformerBlock._get_ada_values(table, batch, timestep, slice(3, 6))
        self.assertEqual(len(vals), 3)  # shift, scale, gate for FFN
        for offset, v in enumerate(vals):
            expected_row = 3 + offset
            self.assertTrue(torch.allclose(v, torch.full_like(v, float(expected_row))))

    def test_slice_6_9_selects_cross_attn_rows(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import (
            BasicAVTransformerBlock,
        )

        dim = 4
        num_params = 9
        table = torch.arange(num_params, dtype=torch.float32).view(num_params, 1).repeat(1, dim)
        batch = 1
        timestep = torch.zeros(batch, 1, num_params * dim)

        vals = BasicAVTransformerBlock._get_ada_values(table, batch, timestep, slice(6, 9))
        self.assertEqual(len(vals), 3)  # shift_q, scale_q, gate
        for offset, v in enumerate(vals):
            expected_row = 6 + offset
            self.assertTrue(torch.allclose(v, torch.full_like(v, float(expected_row))))


class TestCrossAttentionAdalnForward(unittest.TestCase):
    """CUDA-gated forward through the ``cross_attention_adaln=True`` path.

    Would fail under the old 6-row assumption: the 9-row table + prompt AdaLN
    + per-step text-context modulation exercises the new code end to end.
    """

    DEVICE = "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_video_only_true_forward(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.modality import Modality
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTXModelType

        torch.manual_seed(42)
        dtype = torch.bfloat16
        model = _build_model(LTXModelType.VideoOnly, True, VIDEO_ONLY_CONFIG, device=self.DEVICE)
        model = model.to(dtype=dtype).eval()
        _init_all_weights(model)

        batch = 1
        n_frames, grid_h, grid_w = 1, 4, 4
        n_patches = n_frames * grid_h * grid_w
        in_channels = VIDEO_ONLY_CONFIG["in_channels"]
        caption_channels = VIDEO_ONLY_CONFIG["caption_channels"]
        text_len = 8

        v_context = (
            torch.randn(batch, text_len, caption_channels, device=self.DEVICE, dtype=dtype) * 0.02
        )
        v_positions = _make_video_positions(batch, n_patches, n_frames, grid_h, grid_w, self.DEVICE)

        video_modality = Modality(
            latent=torch.randn(batch, n_patches, in_channels, device=self.DEVICE, dtype=dtype)
            * 0.02,
            timesteps=torch.tensor([0.5], device=self.DEVICE),
            positions=v_positions,
            context=v_context,
            sigma=torch.tensor([0.5], device=self.DEVICE),
        )

        text_cache = model.prepare_text_cache(
            video_context=v_context,
            video_positions=v_positions,
            dtype=dtype,
        )
        # Prompt-AdaLN models must NOT pre-project text K/V (stale under per-step
        # context modulation); the block projects K/V from the modulated context.
        self.assertIsNone(text_cache.video_kv)

        with torch.no_grad():
            video_out, audio_out = model(video=video_modality, audio=None, text_cache=text_cache)

        self.assertIsNotNone(video_out)
        self.assertIsNone(audio_out)
        self.assertEqual(video_out.shape, (batch, n_patches, VIDEO_ONLY_CONFIG["out_channels"]))
        self.assertFalse(torch.isnan(video_out).any())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_missing_sigma_raises_for_prompt_adaln(self):
        from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.modality import Modality
        from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTXModelType

        torch.manual_seed(0)
        dtype = torch.bfloat16
        model = _build_model(LTXModelType.VideoOnly, True, VIDEO_ONLY_CONFIG, device=self.DEVICE)
        model = model.to(dtype=dtype).eval()
        _init_all_weights(model)

        batch = 1
        n_frames, grid_h, grid_w = 1, 4, 4
        n_patches = n_frames * grid_h * grid_w
        in_channels = VIDEO_ONLY_CONFIG["in_channels"]
        caption_channels = VIDEO_ONLY_CONFIG["caption_channels"]
        v_context = torch.randn(batch, 8, caption_channels, device=self.DEVICE, dtype=dtype) * 0.02
        v_positions = _make_video_positions(batch, n_patches, n_frames, grid_h, grid_w, self.DEVICE)

        video_modality = Modality(
            latent=torch.randn(batch, n_patches, in_channels, device=self.DEVICE, dtype=dtype)
            * 0.02,
            timesteps=torch.tensor([0.5], device=self.DEVICE),
            positions=v_positions,
            context=v_context,
            sigma=None,  # required by prompt AdaLN -> must fail fast
        )
        text_cache = model.prepare_text_cache(
            video_context=v_context, video_positions=v_positions, dtype=dtype
        )
        with self.assertRaises(ValueError):
            model(video=video_modality, audio=None, text_cache=text_cache)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
