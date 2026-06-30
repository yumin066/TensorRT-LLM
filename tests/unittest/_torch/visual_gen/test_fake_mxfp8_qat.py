# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for offline fake MXFP8 VisualGen QAT helpers."""

import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch.visual_gen.quantization.fake_mxfp8 import (
    FP8_E4M3_MAX,
    MXFP8_BLOCK_SIZE,
    QWEN_IMAGE_BLOCK_LINEAR_SUFFIXES,
    FakeMxfp8Linear,
    fake_mxfp8_activation_quantize,
    fake_mxfp8_weight_quantize,
    inject_fake_mxfp8_linears,
    normalize_qwen_module_name,
    select_qwen_image_block_linears,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for runtime quantization parity tests.",
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


def _dequantize_blockwise_weight(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = MXFP8_BLOCK_SIZE,
) -> torch.Tensor:
    out_features, in_features = qweight.shape
    num_blocks_out, num_blocks_in = scales.shape
    pad_out = num_blocks_out * block_size - out_features
    pad_in = num_blocks_in * block_size - in_features
    if pad_out or pad_in:
        padded = torch.nn.functional.pad(qweight, (0, pad_in, 0, pad_out))
    else:
        padded = qweight

    blocks = (
        padded.to(torch.float32)
        .reshape(num_blocks_out, block_size, num_blocks_in, block_size)
        .permute(0, 2, 1, 3)
    )
    dequantized = (
        (blocks * scales[:, :, None, None])
        .permute(0, 2, 1, 3)
        .reshape(
            num_blocks_out * block_size,
            num_blocks_in * block_size,
        )
    )
    return dequantized[:out_features, :in_features].contiguous()


def _dequantize_activation_1x128(
    qactivation: torch.Tensor,
    scales: torch.Tensor,
    original_shape: tuple[int, ...],
    block_size: int = MXFP8_BLOCK_SIZE,
) -> torch.Tensor:
    in_features = original_shape[-1]
    flat = qactivation.reshape(-1, in_features)
    rows, cols = flat.shape
    num_blocks = scales.shape[-1]
    padded_cols = num_blocks * block_size
    pad_cols = padded_cols - cols
    if pad_cols:
        flat = torch.nn.functional.pad(flat, (0, pad_cols))

    dequantized = flat.to(torch.float32).reshape(rows, num_blocks, block_size) * scales.reshape(
        rows, num_blocks, 1
    )
    dequantized = dequantized.reshape(rows, padded_cols)[:, :cols]
    return dequantized.reshape(original_shape).contiguous()


def _boundary_weight(shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    values = torch.tensor(
        [
            -4.0 * FP8_E4M3_MAX,
            -FP8_E4M3_MAX,
            -1.0,
            -1.0e-7,
            0.0,
            1.0e-7,
            1.0,
            FP8_E4M3_MAX,
            4.0 * FP8_E4M3_MAX,
        ],
        dtype=torch.float32,
        device=device,
    )
    total = shape[0] * shape[1]
    repeats = (total + values.numel() - 1) // values.numel()
    return values.repeat(repeats)[:total].reshape(shape)


def _deterministic_tensor(
    shape: tuple[int, ...], device: torch.device, scale: float
) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= dim
    return (
        torch.linspace(-1.0, 1.0, steps=numel, dtype=torch.float32, device=device)
        .reshape(shape)
        .mul(scale)
    )


def _flat_rows(shape: tuple[int, ...]) -> int:
    rows = 1
    for dim in shape[:-1]:
        rows *= dim
    return rows


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


@requires_cuda
@pytest.mark.parametrize("shape", [(128, 128), (256, 384), (129, 257)])
@pytest.mark.parametrize(
    "scale_mode, use_e8m0_scales",
    [
        ("fp32_block_scale", False),
        ("e8m0_power2", True),
    ],
)
@pytest.mark.parametrize("case", ["zero", "tiny", "boundary", "random"])
def test_fake_mxfp8_weight_quantize_matches_runtime_blockwise_helper(
    shape,
    scale_mode,
    use_e8m0_scales,
    case,
):
    from tensorrt_llm._torch.visual_gen.quantization.ops import quantize_fp8_blockwise

    device = torch.device("cuda")
    torch.manual_seed(1234)
    if case == "zero":
        weight = torch.zeros(shape, dtype=torch.float32, device=device)
    elif case == "tiny":
        weight = torch.linspace(
            -1.0e-8,
            1.0e-8,
            steps=shape[0] * shape[1],
            device=device,
        ).reshape(shape)
    elif case == "boundary":
        weight = _boundary_weight(shape, device)
    else:
        weight = torch.randn(shape, dtype=torch.float32, device=device) * 3.0

    fake, fake_scales = fake_mxfp8_weight_quantize(weight, scale_mode=scale_mode)
    qweight, runtime_scales = quantize_fp8_blockwise(
        weight,
        block_size=MXFP8_BLOCK_SIZE,
        use_e8m0_scales=use_e8m0_scales,
    )
    runtime_dequant = _dequantize_blockwise_weight(qweight, runtime_scales)

    torch.testing.assert_close(fake_scales, runtime_scales, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(fake, runtime_dequant, rtol=1.0e-5, atol=1.0e-5)


@requires_cuda
@pytest.mark.parametrize("activation_shape", [(7, 256), (2, 5, 256)])
def test_fake_mxfp8_linear_matches_runtime_fp8_block_scales_linear(activation_shape):
    try:
        _ = torch.ops.trtllm.fp8_quantize_1x128
        _ = torch.ops.trtllm.fp8_block_scaling_gemm
    except (AttributeError, RuntimeError) as exc:
        pytest.skip(f"FP8 block-scale runtime ops are not available: {exc}")

    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm._torch.visual_gen.quantization.ops import quantize_fp8_blockwise
    from tensorrt_llm._utils import get_sm_version
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    sm_version = get_sm_version()
    if sm_version != 90:
        pytest.skip(
            f"This parity test targets the SM90 FP8 block-scale Linear path, got SM{sm_version}."
        )

    device = torch.device("cuda")
    in_features = 256
    out_features = 256
    weight = _deterministic_tensor((out_features, in_features), device, scale=0.25)
    bias = _deterministic_tensor((out_features,), device, scale=0.05).to(torch.bfloat16)
    activation = _deterministic_tensor(activation_shape, device, scale=0.5).to(torch.bfloat16)

    reference = nn.Linear(in_features, out_features, bias=True, dtype=torch.bfloat16).to(device)
    with torch.no_grad():
        reference.weight.copy_(weight.to(torch.bfloat16))
        reference.bias.copy_(bias)
    fake_linear = FakeMxfp8Linear(reference)

    qweight, weight_scale = quantize_fp8_blockwise(weight.to(torch.bfloat16))
    runtime_linear = Linear(
        in_features,
        out_features,
        bias=True,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
    ).to(device)
    runtime_linear.load_weights(
        [
            {
                "weight": qweight,
                "weight_scale": weight_scale,
                "bias": bias,
            }
        ]
    )
    runtime_linear.post_load_weights()

    assert runtime_linear.has_fp8_block_scales
    assert runtime_linear.weight.dtype == torch.float8_e4m3fn
    assert runtime_linear.weight_scale.dtype == torch.float32
    assert runtime_linear.weight_scale.numel() > 0

    runtime_output = runtime_linear(activation)
    fake_output = fake_linear(activation)

    assert runtime_output.shape == fake_output.shape
    torch.testing.assert_close(
        runtime_output.float(),
        fake_output.float(),
        rtol=1.5e-1,
        atol=1.5e-1,
    )


@requires_cuda
@pytest.mark.parametrize("activation_shape", [(5, 128), (2, 3, 257)])
@pytest.mark.parametrize("case", ["zero", "tiny", "boundary", "random"])
def test_fake_mxfp8_activation_quantize_matches_runtime_1x128_helper(
    activation_shape,
    case,
):
    try:
        _ = torch.ops.trtllm.fp8_quantize_1x128
    except (AttributeError, RuntimeError) as exc:
        pytest.skip(f"FP8 1x128 activation quantization op is not available: {exc}")

    from tensorrt_llm.quantization.utils.fp8_utils import fp8_quantize_1x128_sf_transpose

    device = torch.device("cuda")
    torch.manual_seed(4321)
    if case == "zero":
        activation = torch.zeros(activation_shape, dtype=torch.bfloat16, device=device)
    elif case == "tiny":
        activation = _deterministic_tensor(activation_shape, device, scale=1.0e-8).to(
            torch.bfloat16
        )
    elif case == "boundary":
        flat_shape = (_flat_rows(activation_shape), activation_shape[-1])
        activation = (
            _boundary_weight(flat_shape, device).reshape(activation_shape).to(torch.bfloat16)
        )
    else:
        activation = (torch.randn(activation_shape, dtype=torch.float32, device=device) * 3.0).to(
            torch.bfloat16
        )

    fake, fake_scales = fake_mxfp8_activation_quantize(activation)
    qactivation, runtime_scales = fp8_quantize_1x128_sf_transpose(
        activation.reshape(-1, activation_shape[-1]),
        use_ue8m0=False,
    )
    runtime_dequant = _dequantize_activation_1x128(
        qactivation,
        runtime_scales,
        tuple(activation_shape),
    ).to(activation.dtype)

    torch.testing.assert_close(fake_scales, runtime_scales, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(fake, runtime_dequant, rtol=1.0e-5, atol=1.0e-5)


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
