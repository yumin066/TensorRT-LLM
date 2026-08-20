# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-modality timestep parity test for native LTX-2 retake."""

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2_retake.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
)


class _RecordingAdaLN:
    def __init__(self, width: int):
        self.width = width
        self.inputs = []

    def __call__(self, timestep, *, hidden_dtype):
        self.inputs.append(timestep.clone())
        output = timestep.to(hidden_dtype).unsqueeze(-1).expand(-1, self.width).contiguous()
        return output, output


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
