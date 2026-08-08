# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import tensorrt_llm._torch.modules.linear as linear_module
from tensorrt_llm._torch.modules.linear import NVFP4LinearMethod


@pytest.fixture(autouse=True)
def restore_reference_backend():
    NVFP4LinearMethod.configure_dynamic_absmax_backend("reference")
    yield
    NVFP4LinearMethod.configure_dynamic_absmax_backend("reference")


@pytest.mark.parametrize("shape", [(3, 17), (4, 8, 32)])
def test_optimized_absmax_is_bit_exact_for_bf16(shape) -> None:
    torch.manual_seed(42)
    value = torch.randn(shape, dtype=torch.bfloat16)
    reference = NVFP4LinearMethod._dynamic_absmax(value)
    assert NVFP4LinearMethod.configure_dynamic_absmax_backend("optimized") == "optimized"
    actual = NVFP4LinearMethod._dynamic_absmax(value)
    assert torch.equal(actual, reference)
    assert actual.dtype == torch.float32


def test_optimized_absmax_preserves_nan_and_inf_contract() -> None:
    NVFP4LinearMethod.configure_dynamic_absmax_backend("optimized")
    assert math.isinf(
        NVFP4LinearMethod._dynamic_absmax(
            torch.tensor([float("-inf"), 1.0], dtype=torch.bfloat16)
        ).item()
    )
    assert torch.isnan(
        NVFP4LinearMethod._dynamic_absmax(
            torch.tensor([float("nan"), 1.0], dtype=torch.bfloat16)
        )
    )


def test_invalid_backend_fails_closed() -> None:
    with pytest.raises(ValueError, match="reference, optimized, or auto"):
        NVFP4LinearMethod.configure_dynamic_absmax_backend("silent-fallback")


def test_backend_report_is_explicit() -> None:
    NVFP4LinearMethod.configure_dynamic_absmax_backend("optimized")
    assert NVFP4LinearMethod.dynamic_absmax_report() == {
        "requested": "optimized",
        "resolved": "optimized",
    }


def test_auto_backend_is_architecture_aware(monkeypatch) -> None:
    monkeypatch.setattr(linear_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(linear_module, "get_sm_version", lambda: 120)
    assert NVFP4LinearMethod.configure_dynamic_absmax_backend("auto") == "optimized"
    assert NVFP4LinearMethod.dynamic_absmax_report() == {
        "requested": "auto",
        "resolved": "optimized",
    }

    monkeypatch.setattr(linear_module, "get_sm_version", lambda: 100)
    assert NVFP4LinearMethod.configure_dynamic_absmax_backend("auto") == "reference"
    assert NVFP4LinearMethod.dynamic_absmax_report() == {
        "requested": "auto",
        "resolved": "reference",
    }
