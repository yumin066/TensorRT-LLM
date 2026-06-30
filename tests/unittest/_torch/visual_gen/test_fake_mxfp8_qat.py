# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for offline fake MXFP8 VisualGen QAT helpers."""

import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch.visual_gen.quantization.fake_mxfp8 import (
    MXFP8_BLOCK_SIZE,
    QWEN_IMAGE_BLOCK_LINEAR_SUFFIXES,
    FakeMxfp8Linear,
    fake_mxfp8_activation_quantize,
    fake_mxfp8_weight_quantize,
    inject_fake_mxfp8_linears,
    normalize_qwen_module_name,
    select_qwen_image_block_linears,
)

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="torch.float8_e4m3fn is required for fake MXFP8 tests.",
)


class _TinyQwenAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(128, 128)
        self.to_k = nn.Linear(128, 128)
        self.to_v = nn.Linear(128, 128)
        self.to_out = nn.Sequential(nn.Linear(128, 128))
        self.add_q_proj = nn.Linear(128, 128)
        self.add_k_proj = nn.Linear(128, 128)
        self.add_v_proj = nn.Linear(128, 128)
        self.to_add_out = nn.Linear(128, 128)


class _TinyQwenBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(128, 128))
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(128, 128))
        self.attn = _TinyQwenAttention()
        self.img_mlp = nn.Module()
        self.img_mlp.up_proj = nn.Linear(128, 256)
        self.img_mlp.down_proj = nn.Linear(256, 128)
        self.txt_mlp = nn.Module()
        self.txt_mlp.up_proj = nn.Linear(128, 256)
        self.txt_mlp.down_proj = nn.Linear(256, 128)


class _TinyQwenTransformer(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.img_in = nn.Linear(64, 128)
        self.txt_in = nn.Linear(128, 128)
        self.transformer_blocks = nn.ModuleList([_TinyQwenBlock() for _ in range(num_layers)])
        self.norm_out = nn.Module()
        self.norm_out.linear = nn.Linear(128, 128)
        self.proj_out = nn.Linear(128, 64)


def _is_power_of_two(tensor: torch.Tensor) -> torch.Tensor:
    log2 = torch.log2(tensor)
    return torch.isclose(log2, log2.round())


def test_normalize_qwen_module_name_removes_orig_mod_segments():
    assert (
        normalize_qwen_module_name("transformer_blocks.0._orig_mod.attn.to_q")
        == "transformer_blocks.0.attn.to_q"
    )
    assert (
        normalize_qwen_module_name("_orig_mod.transformer_blocks.1.attn.to_v")
        == "transformer_blocks.1.attn.to_v"
    )


def test_select_qwen_image_block_linears_excludes_non_block_linears():
    model = _TinyQwenTransformer(num_layers=2)

    targets = select_qwen_image_block_linears(
        model,
        expected_num_layers=2,
        linear_cls=nn.Linear,
    )

    assert len(targets) == 2 * len(QWEN_IMAGE_BLOCK_LINEAR_SUFFIXES)
    assert [target.role for target in targets[: len(QWEN_IMAGE_BLOCK_LINEAR_SUFFIXES)]] == list(
        QWEN_IMAGE_BLOCK_LINEAR_SUFFIXES
    )
    assert all(target.normalized_name.startswith("transformer_blocks.") for target in targets)
    assert "img_in" not in {target.normalized_name for target in targets}
    assert "txt_in" not in {target.normalized_name for target in targets}
    assert "norm_out.linear" not in {target.normalized_name for target in targets}
    assert "proj_out" not in {target.normalized_name for target in targets}


def test_select_qwen_image_block_linears_fails_on_unexpected_count():
    model = _TinyQwenTransformer(num_layers=1)

    with pytest.raises(ValueError, match="Expected 15"):
        select_qwen_image_block_linears(model, expected_count=15, linear_cls=nn.Linear)


def test_inject_fake_mxfp8_linears_replaces_only_targets():
    model = _TinyQwenTransformer(num_layers=1)

    injections = inject_fake_mxfp8_linears(model, expected_count=14, linear_cls=nn.Linear)

    assert len(injections) == 14
    assert isinstance(model.transformer_blocks[0].attn.to_q, FakeMxfp8Linear)
    assert isinstance(model.transformer_blocks[0].img_mod[1], FakeMxfp8Linear)
    assert isinstance(model.img_in, nn.Linear)
    assert isinstance(model.txt_in, nn.Linear)
    assert isinstance(model.norm_out.linear, nn.Linear)
    assert isinstance(model.proj_out, nn.Linear)

    activation = torch.randn(2, 3, 128)
    output = model.transformer_blocks[0].attn.to_q(activation)
    assert output.shape == (2, 3, 128)


def test_fake_mxfp8_weight_quantize_handles_padded_edges_and_scale_modes():
    weight = torch.linspace(-2.0, 2.0, steps=129 * 257, dtype=torch.float32).reshape(129, 257)

    fake, scales = fake_mxfp8_weight_quantize(weight)

    assert fake.shape == weight.shape
    assert fake.dtype == weight.dtype
    assert scales.shape == (2, 3)
    assert scales.dtype == torch.float32
    assert torch.isfinite(fake).all()
    assert torch.isfinite(scales).all()

    _, power_scales = fake_mxfp8_weight_quantize(weight, scale_mode="e8m0_power2")
    assert torch.all(_is_power_of_two(power_scales))


def test_fake_mxfp8_activation_quantize_uses_token_block_scales():
    activation = torch.randn(2, 5, 257, dtype=torch.float32)

    fake, scales = fake_mxfp8_activation_quantize(activation)

    assert fake.shape == activation.shape
    assert fake.dtype == activation.dtype
    assert scales.shape == (10, 3)
    assert scales.dtype == torch.float32
    assert torch.isfinite(fake).all()
    assert torch.isfinite(scales).all()


def test_fake_mxfp8_linear_preserves_shape_bias_and_ste_gradients():
    torch.manual_seed(0)
    linear = nn.Linear(130, 64, bias=True).to(torch.float32)
    wrapper = FakeMxfp8Linear(linear, module_name="transformer_blocks.0.attn.to_q")
    activation = torch.randn(4, 2, 130, dtype=torch.float32, requires_grad=True)

    output = wrapper(activation)
    loss = output.square().mean()
    loss.backward()

    assert output.shape == (4, 2, 64)
    assert activation.grad is not None
    assert linear.weight.grad is not None
    assert linear.bias is not None and linear.bias.grad is not None
    assert activation.grad.abs().sum() > 0
    assert linear.weight.grad.abs().sum() > 0
    assert linear.bias.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "factory, error_type, match",
    [
        (
            lambda: fake_mxfp8_weight_quantize(torch.ones(128, 128, dtype=torch.int32)),
            TypeError,
            "must use one of",
        ),
        (
            lambda: fake_mxfp8_weight_quantize(torch.ones(2, 128, 128)),
            ValueError,
            "must be 2D",
        ),
        (
            lambda: fake_mxfp8_weight_quantize(torch.empty(0, 128)),
            ValueError,
            "non-empty",
        ),
        (
            lambda: fake_mxfp8_weight_quantize(torch.ones(128, 128), block_size=64),
            ValueError,
            f"block_size={MXFP8_BLOCK_SIZE}",
        ),
        (
            lambda: fake_mxfp8_weight_quantize(torch.ones(128, 64)),
            ValueError,
            "at least 128",
        ),
        (
            lambda: fake_mxfp8_activation_quantize(torch.tensor(1.0)),
            ValueError,
            "at least one dimension",
        ),
        (
            lambda: FakeMxfp8Linear(nn.Linear(64, 32)),
            ValueError,
            "at least 128",
        ),
    ],
)
def test_fake_mxfp8_failure_modes(factory, error_type, match):
    with pytest.raises(error_type, match=match):
        factory()
