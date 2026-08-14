# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for LTX-2 multimodal transformer argument preparation."""

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.modality import Modality
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
)


class _RecordingAdaLN:
    def __init__(self, output_width: int) -> None:
        self.output_width = output_width
        self.inputs: list[torch.Tensor] = []

    def __call__(
        self, timestep: torch.Tensor, *, hidden_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_dtype == torch.bfloat16
        self.inputs.append(timestep.clone())
        output = timestep.to(hidden_dtype).unsqueeze(-1).expand(-1, self.output_width).contiguous()
        return output, output


class _SimplePreprocessor:
    timestep_scale_multiplier = 1000

    def prepare(
        self,
        modality: Modality,
        static_context: torch.Tensor,
        static_mask: torch.Tensor | None,
        static_pe: tuple[torch.Tensor, torch.Tensor],
    ) -> TransformerArgs:
        batch_size, sequence_length, _ = modality.latent.shape
        return TransformerArgs(
            x=modality.latent,
            context=static_context,
            context_mask=static_mask,
            timesteps=torch.empty(batch_size, sequence_length, 1),
            embedded_timestep=torch.empty(batch_size, sequence_length, 1),
            positional_embeddings=static_pe,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=True,
        )


def _modality(timesteps: torch.Tensor, sigma: torch.Tensor) -> Modality:
    batch_size, sequence_length = timesteps.shape
    return Modality(
        latent=torch.zeros(batch_size, sequence_length, 4, dtype=torch.bfloat16),
        timesteps=timesteps,
        positions=torch.zeros(batch_size, 1, sequence_length),
        context=torch.zeros(batch_size, 1, 4, dtype=torch.bfloat16),
        sigma=sigma,
    )


def test_cross_attention_gate_uses_cross_modality_sigma() -> None:
    scale_shift_adaln = _RecordingAdaLN(output_width=8)
    gate_adaln = _RecordingAdaLN(output_width=4)
    preprocessor = MultiModalTransformerArgsPreprocessor.__new__(
        MultiModalTransformerArgsPreprocessor
    )
    preprocessor.simple_preprocessor = _SimplePreprocessor()
    preprocessor.cross_scale_shift_adaln = scale_shift_adaln
    preprocessor.cross_gate_adaln = gate_adaln
    preprocessor.av_ca_timestep_scale_multiplier = 10

    modality_timesteps = torch.tensor([[0.0, 0.25, 0.5], [0.1, 0.2, 0.3]], dtype=torch.bfloat16)
    modality = _modality(modality_timesteps, sigma=torch.tensor([0.9, 0.8]))
    cross_modality = _modality(
        torch.ones_like(modality_timesteps),
        sigma=torch.tensor([0.25, 0.75], dtype=torch.bfloat16),
    )
    static_context = torch.zeros(2, 1, 4, dtype=torch.bfloat16)
    static_pe = (torch.zeros(1), torch.zeros(1))

    result = preprocessor.prepare(
        modality,
        static_context=static_context,
        static_mask=None,
        static_pe=static_pe,
        static_cross_pe=static_pe,
        cross_modality=cross_modality,
    )

    torch.testing.assert_close(
        scale_shift_adaln.inputs[0],
        (modality_timesteps * 1000).flatten(),
    )
    torch.testing.assert_close(
        gate_adaln.inputs[0],
        cross_modality.sigma * 10,
    )
    assert result.cross_gate_timestep is not None
    assert result.cross_gate_timestep.shape == (2, 3, 4)
    assert result.cross_gate_timestep.is_contiguous()
    torch.testing.assert_close(
        result.cross_gate_timestep[:, 0],
        result.cross_gate_timestep[:, 2],
    )
