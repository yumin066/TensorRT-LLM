# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused CPU tests for the native LTX-2 retake pipeline."""

from pathlib import Path

import pytest
import safetensors.torch
import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.patchifier import VideoLatentPatchifier
from tensorrt_llm._torch.visual_gen.models.ltx2_retake.ltx2_retake_core.attention import (
    FlashInferSelfAttention,
)
from tensorrt_llm._torch.visual_gen.models.ltx2_retake.ltx2_retake_core.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
)
from tensorrt_llm._torch.visual_gen.models.ltx2_retake.pipeline_ltx2_retake import (
    LTX2RetakePipeline,
    _fuse_lora_into_transformer_weights,
    _init_retake_patchified_latents,
    _normalize_fp8_step_indices,
    _retake_conditioned_latent_ranges,
    _retake_pixel_window,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY
from tensorrt_llm.visual_gen.args import VisualGenArgs


class _RecordingAdaLN:
    def __init__(self, width: int) -> None:
        self.width = width
        self.inputs: list[torch.Tensor] = []

    def __call__(
        self, timestep: torch.Tensor, *, hidden_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.inputs.append(timestep.clone())
        output = timestep.to(hidden_dtype).unsqueeze(-1).expand(-1, self.width).contiguous()
        return output, output


def test_retake_window_maps_to_unconditioned_latents() -> None:
    pixel_start, pixel_end = _retake_pixel_window(
        start_time=2.9667,
        end_time=3.9333,
        fps=30.0,
        num_frames=209,
    )

    assert (pixel_start, pixel_end) == (89, 118)
    assert _retake_conditioned_latent_ranges(
        pixel_start=pixel_start,
        pixel_end=pixel_end,
        num_frames=209,
        temporal_ratio=8,
    ) == [(0, 12), (16, 27)]


def test_retake_initial_noise_preserves_conditioned_tokens() -> None:
    patchifier = VideoLatentPatchifier(patch_size=1)
    source = patchifier.patchify(
        torch.arange(1 * 4 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 4, 3, 2, 2)
    )
    noise = torch.randn(source.shape, generator=torch.Generator().manual_seed(42))
    denoise_mask = torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]])

    initialized = _init_retake_patchified_latents(noise, source, denoise_mask)

    conditioned = denoise_mask[0] == 0
    regenerated = ~conditioned
    assert torch.equal(initialized[:, conditioned], source[:, conditioned])
    assert torch.equal(initialized[:, regenerated], noise[:, regenerated])


def test_retake_fp8_step_indices_are_validated() -> None:
    assert _normalize_fp8_step_indices([7, 4, 4]) == frozenset({4, 7})
    with pytest.raises(TypeError, match="sequence of integer"):
        _normalize_fp8_step_indices("4,7")
    with pytest.raises(ValueError, match=r"\[0, 8\)"):
        _normalize_fp8_step_indices([8])


def test_nvfp4_attention_defers_padding_to_flashinfer() -> None:
    attention = FlashInferSelfAttention.__new__(FlashInferSelfAttention)
    quantized_shape = None

    def fake_quantize(query, key, value, *, per_block_mean):
        nonlocal quantized_shape
        quantized_shape = tuple(query.shape)
        assert tuple(key.shape) == tuple(value.shape) == quantized_shape
        assert not per_block_mean
        return query, key, value, None, None, None, None

    def fake_forward(*_args, **_kwargs):
        return torch.zeros((1, 2, 128, 128), dtype=torch.bfloat16), None

    attention._nvfp4_quantize = fake_quantize
    attention._nvfp4_forward = fake_forward
    query = torch.zeros((1, 5, 256), dtype=torch.bfloat16)

    output = attention._run_nvfp4(query, query, query, num_heads=2, head_dim=128)

    assert quantized_shape == (1, 2, 5, 128)
    assert output.shape == query.shape


def test_cross_attention_gate_uses_cross_modality_sigma() -> None:
    scale_shift_adaln = _RecordingAdaLN(width=8)
    gate_adaln = _RecordingAdaLN(width=4)
    preprocessor = MultiModalTransformerArgsPreprocessor.__new__(
        MultiModalTransformerArgsPreprocessor
    )
    preprocessor.cross_scale_shift_adaln = scale_shift_adaln
    preprocessor.cross_gate_adaln = gate_adaln
    preprocessor.av_ca_timestep_scale_multiplier = 10

    modality_timesteps = torch.tensor(
        [[0.0, 0.25, 0.5], [0.1, 0.2, 0.3]],
        dtype=torch.bfloat16,
    )
    cross_modality_sigma = torch.tensor([0.25, 0.75], dtype=torch.bfloat16)
    _, gate = preprocessor._prepare_cross_attention_timestep(
        modality_timesteps=modality_timesteps,
        cross_modality_sigma=cross_modality_sigma,
        timestep_scale_multiplier=1000,
        batch_size=2,
        hidden_dtype=torch.bfloat16,
    )

    torch.testing.assert_close(gate_adaln.inputs[0], cross_modality_sigma * 10)
    assert gate.shape == (2, 3, 4)


def test_retake_lora_fusion_maps_checkpoint_prefix(tmp_path: Path) -> None:
    lora_path = tmp_path / "retake_lora.safetensors"
    down = torch.tensor([[1.0, 2.0, 3.0]])
    up = torch.tensor([[2.0], [4.0]])
    safetensors.torch.save_file(
        {
            "model.diffusion_model.block.lora_A.weight": down,
            "model.diffusion_model.block.lora_B.weight": up,
        },
        lora_path,
    )
    weights = {"block.weight": torch.zeros(2, 3)}

    fused = _fuse_lora_into_transformer_weights(weights, str(lora_path), strength=0.5)

    torch.testing.assert_close(fused["block.weight"], 0.5 * up @ down)


def test_retake_pipeline_registration_and_config_schema() -> None:
    entry = PIPELINE_REGISTRY["LTX2RetakePipeline"]
    assert entry.pipeline_cls is LTX2RetakePipeline
    assert set(entry.defaults) == {
        "text_encoder_path",
        "retake_lora_path",
        "retake_lora_strength",
        "retake_prompt_conditioning_path",
        "retake_start_time",
        "retake_end_time",
        "retake_seed",
        "retake_fp8_linear_steps",
        "retake_attention_backend",
    }

    args = VisualGenArgs(
        model="/tmp/ltx2-retake.safetensors",
        pipeline="LTX2RetakePipeline",
        pipeline_config={"retake_lora_strength": 0.5},
    )
    resolved = PipelineLoader(args)._resolve_pipeline_config(args.model)
    assert resolved["retake_lora_strength"] == 0.5

    config_dir = Path(__file__).resolve().parents[4] / "examples" / "visual_gen" / "configs"
    for filename in (
        "ltx2-retake-1gpu.yaml",
        "ltx2-retake-fp4-1gpu.yaml",
        "ltx2-retake-fp8-1gpu.yaml",
    ):
        recipe_args = VisualGenArgs.from_yaml(config_dir / filename)
        pipeline_config = recipe_args.pipeline_config
        assert pipeline_config["retake_start_time"] == 2.9667
        assert pipeline_config["retake_end_time"] == 3.9333
        assert pipeline_config["retake_seed"] == 42
        assert pipeline_config["retake_prompt_conditioning_path"] is None
        assert pipeline_config["retake_lora_path"] is None
        assert pipeline_config["retake_lora_strength"] == 1.0

    recipes = {
        "ltx2-retake-fp4-1gpu.yaml": ("NVFP4", "flashinfer_nvfp4", [4, 7]),
        "ltx2-retake-fp8-1gpu.yaml": ("FP8_BLOCK_SCALES", "flashinfer_fp8", None),
    }
    for filename, (quant_algo, attention_backend, fp8_steps) in recipes.items():
        recipe_args = VisualGenArgs.from_yaml(config_dir / filename)
        assert recipe_args.pipeline == "LTX2RetakePipeline"
        assert recipe_args.quant_config == {"quant_algo": quant_algo, "dynamic": True}
        assert recipe_args.pipeline_config["retake_attention_backend"] == attention_backend
        assert recipe_args.pipeline_config.get("retake_fp8_linear_steps") == fp8_steps
