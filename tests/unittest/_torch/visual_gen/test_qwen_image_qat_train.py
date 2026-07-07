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
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from scripts.visualgen_eval.qwen_image_qat_train import (
    DEFAULT_TIMESTEP_WEIGHTS,
    QWEN_BLOCK_LINEAR_TARGET,
    FakeMxfp8Linear,
    Mxfp8LoraAdapterLinear,
    QwenImageQatMonitorConfig,
    QwenImageQatTrainingConfig,
    QwenImageTupleDataset,
    QwenImageTupleLossConfig,
    build_qwen_image_qat_probe_config,
    build_qwen_image_qat_scale_sensitivity_manifest,
    compute_qwen_image_tuple_loss,
    compute_scale_aware_lora_multipliers,
    forward_qwen_image_tuple,
    load_qwen_image_qat_checkpoint,
    load_qwen_image_tuple_sample,
    monitor_qwen_image_qat_checkpoint,
    normalize_qwen_image_qat_parameter_name,
    prepare_qwen_image_qat_model,
    qwen_image_qat_trainable_parameter_names,
    run_qwen_image_qat_checkpoint_rgb_eval,
    summarize_qwen_image_qat_training_metrics,
    train_qwen_image_qat,
    validate_qwen_image_qat_probe_recipe,
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        timestep: torch.Tensor,
        img_shapes: object,
        txt_seq_lens: object,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        additional_t_cond: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> tuple[torch.Tensor]:
        del (
            additional_t_cond,
            encoder_hidden_states,
            encoder_hidden_states_mask,
            img_shapes,
            return_dict,
            timestep,
            txt_seq_lens,
        )
        return (self.transformer_blocks[0].attn.to_q(hidden_states),)


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


def _write_prompt_manifest(path: Path, *, split: str = "fast_calibration") -> dict[str, object]:
    record: dict[str, object] = {
        "prompt_id": "qwen_image_smoke_0000",
        "split": split,
        "source": "unit_test",
        "prompt": "a ceramic mug on a desk",
        "negative_prompt": "",
        "seed": 1234,
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "guidance_scale": 4.0,
        "max_sequence_length": 512,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return record


def _write_placeholder_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder png")


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


def test_qwen_image_qat_probe_config_recipes(tmp_path: Path) -> None:
    tuple_index = tmp_path / "index.jsonl"
    output_dir = tmp_path / "probe"

    mse_config = build_qwen_image_qat_probe_config(
        recipe="mse_only",
        tuple_index_jsonl=tuple_index,
        output_dir=output_dir,
        max_steps=200,
    )
    assert mse_config.loss.lambda_mse == pytest.approx(1.0)
    assert mse_config.loss.lambda_dir == pytest.approx(0.0)
    assert mse_config.loss.timestep_weights is None
    assert mse_config.lora_rank == 16
    assert mse_config.lora_alpha == pytest.approx(32.0)
    assert mse_config.optimizer_foreach is False
    assert mse_config.sample_stride == 17
    assert mse_config.sample_start_index == 0

    timestep_config = build_qwen_image_qat_probe_config(
        recipe="timestep_weighted",
        tuple_index_jsonl=tuple_index,
        output_dir=output_dir,
        max_steps=200,
    )
    assert timestep_config.loss.lambda_dir == pytest.approx(0.0)
    assert timestep_config.loss.timestep_weights == DEFAULT_TIMESTEP_WEIGHTS

    direction_config = build_qwen_image_qat_probe_config(
        recipe="direction_aware",
        tuple_index_jsonl=tuple_index,
        output_dir=output_dir,
        max_steps=200,
    )
    assert direction_config.loss.lambda_dir == pytest.approx(0.1)
    assert direction_config.loss.timestep_weights == DEFAULT_TIMESTEP_WEIGHTS

    scale_config = build_qwen_image_qat_probe_config(
        recipe="scale_aware_lora",
        tuple_index_jsonl=tuple_index,
        output_dir=output_dir,
        max_steps=200,
        scale_multipliers={"transformer_blocks.0.attn.to_q": 2.0},
    )
    assert scale_config.loss.lambda_dir == pytest.approx(0.1)
    assert scale_config.scale_multipliers == {"transformer_blocks.0.attn.to_q": 2.0}


def test_qwen_image_qat_probe_config_rejects_unknown_recipe() -> None:
    with pytest.raises(ValueError, match="unsupported Qwen-Image QAT probe recipe"):
        validate_qwen_image_qat_probe_recipe("unknown")


def test_summarize_qwen_image_qat_training_metrics_groups_timestep_bins(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    records = [
        {
            "step": 1,
            "timestep_bin": "early",
            "loss_total": 3.0,
            "loss_mse": 3.0,
            "loss_direction": None,
            "grad_norm": 2.0,
            "elapsed_seconds": 10.0,
        },
        {
            "step": 2,
            "timestep_bin": "late",
            "loss_total": 1.0,
            "loss_mse": 0.5,
            "loss_direction": 0.25,
            "grad_norm": 1.5,
            "lora_delta_norm": 0.1,
            "elapsed_seconds": 20.0,
        },
    ]
    metrics_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    summary = summarize_qwen_image_qat_training_metrics(metrics_path)

    assert summary["record_count"] == 2
    assert summary["first_loss_total"] == pytest.approx(3.0)
    assert summary["last_loss_total"] == pytest.approx(1.0)
    assert summary["loss_total_delta"] == pytest.approx(-2.0)
    assert summary["best_loss_step"] == 2
    assert summary["last_lora_delta_norm"] == pytest.approx(0.1)
    assert summary["timestep_bins"]["early"]["loss_total_mean"] == pytest.approx(3.0)
    assert summary["timestep_bins"]["late"]["loss_direction_mean"] == pytest.approx(0.25)


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
def test_prepare_qwen_image_qat_model_applies_scale_multipliers() -> None:
    model = _TinyQwenTransformer()

    injections = prepare_qwen_image_qat_model(
        model,
        target_layers=("attn.to_q", "attn.to_k"),
        lora_rank=4,
        lora_alpha=8.0,
        scale_multipliers={"attn.to_q": 2.0},
        linear_cls=nn.Linear,
    )

    by_role = {injection.role: injection for injection in injections}
    assert by_role["attn.to_q"].scale_multiplier == pytest.approx(2.0)
    assert by_role["attn.to_k"].scale_multiplier == pytest.approx(1.0)
    assert model.transformer_blocks[0].attn.to_q.scale_multiplier == pytest.approx(2.0)


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
def test_scale_sensitivity_manifest_and_multipliers(tmp_path: Path) -> None:
    model = _TinyQwenTransformer()
    prepare_qwen_image_qat_model(
        model,
        target_layers=("attn.to_q", "attn.to_k"),
        linear_cls=nn.Linear,
    )
    output_path = tmp_path / "scale_manifest.json"

    manifest = build_qwen_image_qat_scale_sensitivity_manifest(model, output_path=output_path)
    multipliers = compute_scale_aware_lora_multipliers(manifest, clip_min=0.5, clip_max=2.0)

    assert output_path.exists()
    assert manifest["status"] == "passed"
    assert manifest["record_count"] == 2
    assert set(multipliers) == {
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_q",
    }
    assert all(0.5 <= value <= 2.0 for value in multipliers.values())


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


@requires_float8
def test_train_qwen_image_qat_logs_loss_components_and_checkpoint(tmp_path: Path) -> None:
    tuple_path = tmp_path / "tuple.pt"
    index_path = tmp_path / "index.jsonl"
    _write_tuple(
        tuple_path,
        hidden_states=torch.ones(1, 2, 128, dtype=torch.bfloat16),
        encoder_hidden_states=torch.ones(1, 3, 128, dtype=torch.bfloat16),
        target_output=torch.zeros(1, 2, 128, dtype=torch.bfloat16),
    )
    _write_index(index_path, tuple_path)
    model = _TinyQwenTransformer().to(dtype=torch.bfloat16)
    config = QwenImageQatTrainingConfig(
        tuple_index_jsonl=index_path,
        output_dir=tmp_path / "out",
        max_steps=2,
        learning_rate=1.0e-3,
        device="cpu",
        target_layers=("attn.to_q",),
        lora_rank=4,
        lora_alpha=8.0,
        expected_num_layers=1,
        loss=QwenImageTupleLossConfig(
            lambda_mse=1.0,
            lambda_dir=0.1,
            timestep_weights={"late": 2.0},
        ),
        compute_lora_delta_norm=True,
    )

    result = train_qwen_image_qat(model, config, linear_cls=nn.Linear)

    records = [
        json.loads(line)
        for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(records) == 2
    assert records[-1]["step"] == 2
    assert records[-1]["loss_total"] > 0.0
    assert records[-1]["loss_mse"] > 0.0
    assert records[-1]["loss_direction"] is not None
    assert records[-1]["timestep_weight"] == pytest.approx(2.0)
    assert records[-1]["sample_index"] == 0
    assert records[-1]["grad_norm"] > 0.0
    assert records[-1]["trainable_parameter_norm"] > 0.0
    assert records[-1]["lora_delta_norm"] >= 0.0

    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["format"] == "qwen_image_mxfp8_qat_adapter_v1"
    assert checkpoint["train_steps"] == 2
    assert sorted(checkpoint["trainable_parameter_names"]) == [
        "transformer_blocks.0.attn.to_q.lora_down.weight",
        "transformer_blocks.0.attn.to_q.lora_up.weight",
    ]


@requires_float8
def test_monitor_qwen_image_qat_checkpoint_restores_adapter_and_writes_summary(
    tmp_path: Path,
) -> None:
    tuple_path = tmp_path / "tuple.pt"
    index_path = tmp_path / "index.jsonl"
    _write_tuple(
        tuple_path,
        hidden_states=torch.ones(1, 2, 128, dtype=torch.bfloat16),
        encoder_hidden_states=torch.ones(1, 3, 128, dtype=torch.bfloat16),
        target_output=torch.zeros(1, 2, 128, dtype=torch.bfloat16),
    )
    _write_index(index_path, tuple_path)
    train_model = _TinyQwenTransformer().to(dtype=torch.bfloat16)
    train_result = train_qwen_image_qat(
        train_model,
        QwenImageQatTrainingConfig(
            tuple_index_jsonl=index_path,
            output_dir=tmp_path / "train",
            max_steps=1,
            learning_rate=1.0e-3,
            device="cpu",
            target_layers=("attn.to_q",),
            lora_rank=4,
            lora_alpha=8.0,
            expected_num_layers=1,
            loss=QwenImageTupleLossConfig(lambda_mse=1.0, lambda_dir=0.1),
        ),
        linear_cls=nn.Linear,
    )
    monitor_model = _TinyQwenTransformer().to(dtype=torch.bfloat16)
    output_json = tmp_path / "monitor" / "summary.json"
    records_jsonl = tmp_path / "monitor" / "records.jsonl"

    summary = monitor_qwen_image_qat_checkpoint(
        monitor_model,
        QwenImageQatMonitorConfig(
            checkpoint_path=train_result.checkpoint_path,
            tuple_index_jsonl=index_path,
            output_json=output_json,
            records_jsonl=records_jsonl,
            max_samples=1,
            device="cpu",
        ),
        linear_cls=nn.Linear,
    )

    checkpoint = load_qwen_image_qat_checkpoint(train_result.checkpoint_path)
    assert checkpoint["train_steps"] == 1
    assert output_json.is_file()
    assert records_jsonl.is_file()
    assert summary["format"] == "qwen_image_mxfp8_qat_checkpoint_monitor_v1"
    assert summary["sample_count"] == 1
    assert summary["dataset_size"] == 1
    assert summary["injection_count"] == 1
    metrics = summary["metrics_summary"]
    assert metrics["record_count"] == 1
    assert metrics["prompt_count"] == 1
    assert metrics["loss_mse_mean"] is not None
    assert metrics["direction_loss_mean"] is not None
    assert metrics["timestep_bins"]["late"]["count"] == 1
    records = [
        json.loads(line) for line in records_jsonl.read_text(encoding="utf-8").splitlines() if line
    ]
    assert records[0]["sample_index"] == 0
    assert records[0]["monitor_step"] == 1


@requires_float8
def test_run_qwen_image_qat_checkpoint_rgb_eval_applies_adapter(
    tmp_path: Path,
) -> None:
    tuple_path = tmp_path / "tuple.pt"
    index_path = tmp_path / "index.jsonl"
    _write_tuple(
        tuple_path,
        hidden_states=torch.ones(1, 2, 128, dtype=torch.bfloat16),
        encoder_hidden_states=torch.ones(1, 3, 128, dtype=torch.bfloat16),
        target_output=torch.zeros(1, 2, 128, dtype=torch.bfloat16),
    )
    _write_index(index_path, tuple_path)
    train_model = _TinyQwenTransformer().to(dtype=torch.bfloat16)
    train_result = train_qwen_image_qat(
        train_model,
        QwenImageQatTrainingConfig(
            tuple_index_jsonl=index_path,
            output_dir=tmp_path / "train",
            max_steps=1,
            learning_rate=1.0e-3,
            device="cpu",
            target_layers=("attn.to_q",),
            lora_rank=4,
            lora_alpha=8.0,
            expected_num_layers=1,
        ),
        linear_cls=nn.Linear,
    )
    prompt_manifest = tmp_path / "prompts.jsonl"
    prompt_record = _write_prompt_manifest(prompt_manifest)
    reference_root = tmp_path / "references" / "bf16_sage_fp8"
    _write_placeholder_png(reference_root / "fast_calibration" / "qwen_image_smoke_0000.png")
    pipeline = SimpleNamespace(transformer=_TinyQwenTransformer().to(dtype=torch.bfloat16))
    metrics_json = tmp_path / "rgb_metrics.json"

    result = run_qwen_image_qat_checkpoint_rgb_eval(
        prompt_manifest_jsonl=prompt_manifest,
        split="fast_calibration",
        model="Qwen/Qwen-Image",
        visual_gen_args=tmp_path / "qwen-image.yaml",
        checkpoint_path=train_result.checkpoint_path,
        reference_root=reference_root,
        output_root=tmp_path / "outputs" / "qat_mse_only",
        metrics_json=metrics_json,
        variant="qat_mse_only_probe_fake_mxfp8_lora",
        device="cpu",
        provenance={"git_head": "test"},
        config_metadata={"pipeline_quant_algo": "FP8_BLOCK_SCALES"},
        records=[prompt_record],
        pipeline=pipeline,
        infer_fn=lambda _pipeline, record: record,
        save_fn=lambda _output, path: _write_placeholder_png(path),
        metrics_fn=lambda _candidate, _reference: {
            "mse": 0.25,
            "psnr": 6.020599913279624,
            "ssim": 0.9,
        },
        linear_cls=nn.Linear,
    )

    assert isinstance(pipeline.transformer.transformer_blocks[0].attn.to_q, Mxfp8LoraAdapterLinear)
    assert result["qat_checkpoint_path"] == str(train_result.checkpoint_path)
    assert result["qat_injection_count"] == 1
    assert result["aggregates"]["psnr_mean"] == pytest.approx(6.020599913279624)
    assert result["config_metadata"]["qat_eval_mode"] == "probe_fake_mxfp8_lora"
    assert metrics_json.is_file()


@requires_float8
def test_train_qwen_image_qat_supports_strided_sampling(tmp_path: Path) -> None:
    tuple_paths = [tmp_path / "tuples" / f"tuple_{index}.pt" for index in range(3)]
    index_path = tmp_path / "index.jsonl"
    entries = []
    for index, tuple_path in enumerate(tuple_paths):
        _write_tuple(
            tuple_path,
            prompt_id=f"qwen_image_smoke_{index:04d}",
            timestep_index=index,
            timestep_bin="early",
            hidden_states=torch.ones(1, 2, 128, dtype=torch.bfloat16),
            encoder_hidden_states=torch.ones(1, 3, 128, dtype=torch.bfloat16),
            target_output=torch.zeros(1, 2, 128, dtype=torch.bfloat16),
        )
        entries.append(
            {
                "prompt_id": f"qwen_image_smoke_{index:04d}",
                "split": "smoke",
                "timestep_index": index,
                "timestep_bin": "early",
                "cfg_branch": "cond",
                "trajectory_source": "bf16_teacher",
                "status": "captured",
                "tuple_path": str(tuple_path),
            }
        )
    index_path.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    model = _TinyQwenTransformer().to(dtype=torch.bfloat16)

    result = train_qwen_image_qat(
        model,
        QwenImageQatTrainingConfig(
            tuple_index_jsonl=index_path,
            output_dir=tmp_path / "out",
            max_steps=3,
            learning_rate=1.0e-3,
            device="cpu",
            target_layers=("attn.to_q",),
            lora_rank=4,
            lora_alpha=8.0,
            expected_num_layers=1,
            log_interval_steps=1,
            sample_stride=2,
        ),
        linear_cls=nn.Linear,
    )

    records = [
        json.loads(line)
        for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [record["sample_index"] for record in records] == [0, 2, 1]
    assert [record["prompt_id"] for record in records] == [
        "qwen_image_smoke_0000",
        "qwen_image_smoke_0002",
        "qwen_image_smoke_0001",
    ]


@requires_float8
def test_train_qwen_image_qat_records_progressive_warmup(tmp_path: Path) -> None:
    tuple_path = tmp_path / "tuple.pt"
    index_path = tmp_path / "index.jsonl"
    _write_tuple(
        tuple_path,
        hidden_states=torch.ones(1, 2, 128, dtype=torch.bfloat16),
        encoder_hidden_states=torch.ones(1, 3, 128, dtype=torch.bfloat16),
        target_output=torch.zeros(1, 2, 128, dtype=torch.bfloat16),
    )
    _write_index(index_path, tuple_path)
    model = _TinyQwenTransformer().to(dtype=torch.bfloat16)

    result = train_qwen_image_qat(
        model,
        QwenImageQatTrainingConfig(
            tuple_index_jsonl=index_path,
            output_dir=tmp_path / "out",
            max_steps=2,
            learning_rate=1.0e-3,
            device="cpu",
            target_layers=("attn.to_q", "attn.to_k"),
            warmup_steps=1,
            warmup_target_layers=("attn.to_q",),
            expected_num_layers=1,
        ),
        linear_cls=nn.Linear,
    )

    records = [
        json.loads(line)
        for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert records[0]["warmup_active"] is True
    assert records[1]["warmup_active"] is False
