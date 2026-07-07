# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from scripts.visualgen_eval.qwen_image_qat_train import (
    QWEN_BLOCK_LINEAR_TARGET,
    FakeMxfp8Linear,
    Mxfp8LoraAdapterLinear,
    QwenImageTupleDataset,
    QwenImageTupleLossConfig,
    compute_qwen_image_tuple_loss,
    forward_qwen_image_tuple,
    load_qwen_image_tuple_sample,
    normalize_qwen_image_qat_parameter_name,
    prepare_qwen_image_qat_model,
    qwen_image_qat_trainable_parameter_names,
)

requires_float8 = pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="torch.float8_e4m3fn is required for fake MXFP8 tests.",
)


class _TupleEchoTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_kwargs: dict[str, object] | None = None

    def forward(self, **kwargs: object) -> tuple[torch.Tensor]:
        self.seen_kwargs = kwargs
        hidden_states = kwargs["hidden_states"]
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a tensor")
        return (hidden_states + 1.0,)


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
    def __init__(self, num_layers: int = 1) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.img_in = nn.Linear(64, 128)
        self.txt_in = nn.Linear(128, 128)
        self.transformer_blocks = nn.ModuleList([_TinyQwenBlock() for _ in range(num_layers)])
        self.norm_out = nn.Module()
        self.norm_out.linear = nn.Linear(128, 128)
        self.proj_out = nn.Linear(128, 64)


def _tuple_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "prompt_id": "qwen_image_smoke_0000",
        "split": "smoke",
        "timestep_index": 7,
        "timestep_bin": "late",
        "cfg_branch": "cond",
        "trajectory_source": "bf16_teacher",
        "status": "captured",
        "hidden_states": torch.ones(1, 2, 4, dtype=torch.bfloat16),
        "timestep": torch.tensor([0.25], dtype=torch.bfloat16),
        "encoder_hidden_states": torch.ones(1, 3, 4, dtype=torch.bfloat16),
        "encoder_hidden_states_mask": torch.tensor([[True, True, False]]),
        "img_shapes": [[(1, 1, 2)]],
        "txt_seq_lens": [2],
        "target_output": torch.full((1, 2, 4), 2.0, dtype=torch.bfloat16),
    }
    payload.update(overrides)
    return payload


def _write_tuple(path: Path, **overrides: object) -> dict[str, object]:
    payload = _tuple_payload(**overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def _write_index(path: Path, tuple_path: Path, **overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "prompt_id": "qwen_image_smoke_0000",
        "split": "smoke",
        "timestep_index": 7,
        "timestep_bin": "late",
        "cfg_branch": "cond",
        "trajectory_source": "bf16_teacher",
        "status": "captured",
        "tuple_path": str(tuple_path),
    }
    entry.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    return entry


def test_qwen_image_tuple_dataset_loads_captured_schema(tmp_path: Path) -> None:
    tuple_path = tmp_path / "tuples" / "tuple.pt"
    index_path = tmp_path / "index.jsonl"
    _write_tuple(tuple_path)
    _write_index(index_path, tuple_path)

    dataset = QwenImageTupleDataset(index_path)

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample.prompt_id == "qwen_image_smoke_0000"
    assert sample.timestep_index == 7
    assert sample.timestep_bin == "late"
    assert sample.cfg_branch == "cond"
    assert sample.target_output.dtype == torch.bfloat16


def test_qwen_image_tuple_loader_rejects_layered_fields(tmp_path: Path) -> None:
    tuple_path = tmp_path / "tuple.pt"
    _write_tuple(tuple_path, layered_rgba_target=torch.zeros(1, 2, 4))

    with pytest.raises(ValueError, match="layered-only"):
        load_qwen_image_tuple_sample(tuple_path)


def test_qwen_image_tuple_dataset_rejects_uncaptured_index(tmp_path: Path) -> None:
    tuple_path = tmp_path / "tuple.pt"
    index_path = tmp_path / "index.jsonl"
    _write_tuple(tuple_path)
    _write_index(index_path, tuple_path, status="planned")

    with pytest.raises(ValueError, match="status=captured"):
        QwenImageTupleDataset(index_path)


def test_forward_qwen_image_tuple_replays_expected_kwargs(tmp_path: Path) -> None:
    tuple_path = tmp_path / "tuple.pt"
    _write_tuple(tuple_path)
    sample = load_qwen_image_tuple_sample(tuple_path)
    transformer = _TupleEchoTransformer()

    output = forward_qwen_image_tuple(transformer, sample, torch.device("cpu"))

    assert torch.allclose(output.float(), sample.hidden_states.float() + 1.0)
    assert transformer.seen_kwargs is not None
    assert set(transformer.seen_kwargs) == {
        "encoder_hidden_states",
        "encoder_hidden_states_mask",
        "hidden_states",
        "img_shapes",
        "return_dict",
        "timestep",
        "txt_seq_lens",
    }
    assert transformer.seen_kwargs["return_dict"] is False


def test_forward_qwen_image_tuple_forwards_additional_t_cond(tmp_path: Path) -> None:
    tuple_path = tmp_path / "tuple.pt"
    additional_t_cond = torch.ones(1, 6, dtype=torch.bfloat16)
    _write_tuple(tuple_path, additional_t_cond=additional_t_cond)
    sample = load_qwen_image_tuple_sample(tuple_path)
    transformer = _TupleEchoTransformer()

    forward_qwen_image_tuple(transformer, sample, torch.device("cpu"))

    assert transformer.seen_kwargs is not None
    assert "additional_t_cond" in transformer.seen_kwargs
    assert torch.equal(
        transformer.seen_kwargs["additional_t_cond"],
        additional_t_cond,
    )


def test_compute_qwen_image_tuple_loss_uses_direction_and_timestep_weight(
    tmp_path: Path,
) -> None:
    tuple_path = tmp_path / "tuple.pt"
    _write_tuple(tuple_path)
    sample = load_qwen_image_tuple_sample(tuple_path)
    output = torch.full_like(sample.target_output, 1.0)

    loss, components = compute_qwen_image_tuple_loss(
        output,
        sample,
        QwenImageTupleLossConfig(
            lambda_mse=1.0,
            lambda_dir=0.5,
            timestep_weights={"late": 3.0},
        ),
    )

    assert components["mse"].item() == pytest.approx(1.0)
    assert components["direction"].item() == pytest.approx(0.0, abs=1.0e-6)
    assert components["unweighted_total"].item() == pytest.approx(1.0)
    assert components["timestep_weight"].item() == pytest.approx(3.0)
    assert loss.item() == pytest.approx(3.0)


def test_compute_qwen_image_tuple_loss_rejects_shape_mismatch(tmp_path: Path) -> None:
    tuple_path = tmp_path / "tuple.pt"
    _write_tuple(tuple_path)
    sample = load_qwen_image_tuple_sample(tuple_path)

    with pytest.raises(ValueError, match="shapes must match"):
        compute_qwen_image_tuple_loss(torch.zeros(1, 1, 4), sample)


@requires_float8
def test_mxfp8_lora_adapter_starts_as_fake_base() -> None:
    base_linear = nn.Linear(128, 128, bias=False).to(dtype=torch.bfloat16)
    fake_base = FakeMxfp8Linear(base_linear)
    adapter = Mxfp8LoraAdapterLinear(fake_base, rank=4, alpha=8.0, dropout=0.0)
    activation = torch.randn(2, 3, 128, dtype=torch.bfloat16)

    adapter_output = adapter(activation)
    base_output = fake_base(activation)

    assert torch.allclose(adapter_output.float(), base_output.float())
    assert adapter.lora_delta_weight().shape == base_linear.weight.shape
    assert adapter.lora_delta_weight().abs().sum().item() == pytest.approx(0.0)


@requires_float8
def test_prepare_qwen_image_qat_model_replaces_all_block_linears_and_freezes_base() -> None:
    model = _TinyQwenTransformer(num_layers=2)

    injections = prepare_qwen_image_qat_model(
        model,
        target_layers=(QWEN_BLOCK_LINEAR_TARGET,),
        lora_rank=4,
        lora_alpha=8.0,
        expected_num_layers=2,
        linear_cls=nn.Linear,
    )

    assert len(injections) == 28
    replacement = model.transformer_blocks[0].attn.to_q
    assert isinstance(replacement, Mxfp8LoraAdapterLinear)
    assert isinstance(replacement.base, FakeMxfp8Linear)
    assert not replacement.base.weight.requires_grad
    assert replacement.lora_down.weight.requires_grad
    assert replacement.lora_up.weight.requires_grad
    assert isinstance(model.img_in, nn.Linear)
    assert isinstance(model.txt_in, nn.Linear)
    assert isinstance(model.norm_out.linear, nn.Linear)
    assert isinstance(model.proj_out, nn.Linear)
    assert not model.img_in.weight.requires_grad
    assert not model.proj_out.weight.requires_grad

    trainable_names = qwen_image_qat_trainable_parameter_names(injections)
    assert len(trainable_names) == 56
    assert "transformer_blocks.0.attn.to_q.lora_down.weight" in trainable_names
    assert "transformer_blocks.0.attn.to_q.lora_up.weight" in trainable_names
    actual_trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert actual_trainable_names == trainable_names


@requires_float8
def test_prepare_qwen_image_qat_model_filters_by_role() -> None:
    model = _TinyQwenTransformer(num_layers=2)

    injections = prepare_qwen_image_qat_model(
        model,
        target_layers=("attn.to_q",),
        lora_rank=4,
        lora_alpha=8.0,
        expected_num_layers=2,
        linear_cls=nn.Linear,
    )

    assert len(injections) == 2
    assert all(injection.role == "attn.to_q" for injection in injections)
    assert isinstance(model.transformer_blocks[0].attn.to_q, Mxfp8LoraAdapterLinear)
    assert isinstance(model.transformer_blocks[0].attn.to_k, nn.Linear)
    assert not model.transformer_blocks[0].attn.to_k.weight.requires_grad


@requires_float8
def test_prepare_qwen_image_qat_model_rejects_inconsistent_expected_counts() -> None:
    model = _TinyQwenTransformer(num_layers=2)

    with pytest.raises(ValueError, match="expected_target_count"):
        prepare_qwen_image_qat_model(
            model,
            expected_num_layers=2,
            expected_target_count=27,
            linear_cls=nn.Linear,
        )


def test_normalize_qwen_image_qat_parameter_name_strips_wrappers() -> None:
    assert (
        normalize_qwen_image_qat_parameter_name(
            "module.transformer_blocks.0._orig_mod.attn.to_q.lora_down.weight"
        )
        == "transformer_blocks.0.attn.to_q.lora_down.weight"
    )
    assert (
        normalize_qwen_image_qat_parameter_name(
            "_fsdp_wrapped_module.transformer_blocks.1._checkpoint_wrapped_module."
            "attn.to_v.lora_up.weight"
        )
        == "transformer_blocks.1.attn.to_v.lora_up.weight"
    )


@requires_float8
def test_prepare_qwen_image_qat_model_rejects_empty_selector() -> None:
    model = _TinyQwenTransformer()

    with pytest.raises(ValueError, match="selected no Qwen block Linears"):
        prepare_qwen_image_qat_model(
            model,
            target_layers=("not_a_real_role",),
            linear_cls=nn.Linear,
        )


@requires_float8
def test_prepare_qwen_image_qat_model_rejects_prequantized_weight() -> None:
    model = _TinyQwenTransformer()
    quantized_weight = model.transformer_blocks[0].attn.to_q.weight.detach().to(torch.float8_e4m3fn)
    model.transformer_blocks[0].attn.to_q.weight = nn.Parameter(
        quantized_weight,
        requires_grad=False,
    )

    with pytest.raises(TypeError, match="BF16/FP16/FP32"):
        prepare_qwen_image_qat_model(
            model,
            target_layers=("attn.to_q",),
            linear_cls=nn.Linear,
        )
