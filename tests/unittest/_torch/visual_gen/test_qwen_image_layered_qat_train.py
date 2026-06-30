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

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from scripts.visualgen_eval.qwen_image_layered_qat_train import (
    Mxfp8LoraAdapterLinear,
    QwenImageLayeredQatConfig,
    TransformerTupleDataset,
    load_qat_config,
    prepare_qat_model,
    train_qwen_image_layered_qat,
)
from tensorrt_llm._torch.visual_gen.quantization.fake_mxfp8 import FakeMxfp8Linear


class _TinyQwenAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(128, 128)
        self.to_k = nn.Linear(128, 128)


class _TinyQwenBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _TinyQwenAttention()


class _TinyQwenTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_TinyQwenBlock()])
        self.proj_out = nn.Linear(128, 128)

    def forward(self, hidden_states, *, timestep, return_dict=False):
        del return_dict
        timestep_tensor = timestep
        if not isinstance(timestep_tensor, torch.Tensor):
            timestep_tensor = torch.tensor(float(timestep), dtype=hidden_states.dtype)
        timestep_tensor = timestep_tensor.to(device=hidden_states.device, dtype=hidden_states.dtype)
        while timestep_tensor.ndim < hidden_states.ndim:
            timestep_tensor = timestep_tensor.unsqueeze(-1)
        output = self.transformer_blocks[0].attn.to_q(hidden_states)
        return (output + timestep_tensor * 0.0,)


def _base_config(tmp_path, **overrides) -> dict:
    data = {
        "tuple_manifest": str(tmp_path / "tuples.json"),
        "output_dir": str(tmp_path / "out"),
        "target_layers": ["attn.to_q"],
        "max_steps": 1,
        "validation_interval_steps": 1,
        "early_stop_patience": 2,
        "checkpoint_interval_steps": 1,
        "recipe": "lora_adapter",
        "schedule": "smoke",
        "device": "cpu",
        "debug_allow_short_run": True,
        "lora_rank": 4,
        "lora_alpha": 8.0,
        "optimizer": {"learning_rate": 1.0e-3},
    }
    data.update(overrides)
    return data


def _write_tuple(path, *, include_target=True, include_hidden=True, include_timestep=True):
    hidden_states = torch.randn(1, 2, 128)
    payload = {
        "args": [],
        "kwargs": {
            "return_dict": False,
        },
    }
    if include_hidden:
        payload["kwargs"]["hidden_states"] = hidden_states
    if include_timestep:
        payload["kwargs"]["timestep"] = torch.tensor([1.0])
    if include_target:
        payload["target_output"] = (hidden_states * 0.5).to(torch.bfloat16)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def _write_manifest(path, tuple_paths) -> None:
    path.write_text(
        json.dumps({"tuples": [str(tuple_path) for tuple_path in tuple_paths]}),
        encoding="utf-8",
    )


def test_config_rejects_epochs_only_and_attention_qat(tmp_path):
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        QwenImageLayeredQatConfig(**_base_config(tmp_path, epochs=1))

    with pytest.raises(ValidationError, match="attention-kernel"):
        QwenImageLayeredQatConfig(**_base_config(tmp_path, attention_qat=True))

    with pytest.raises(ValidationError, match="SageAttention"):
        QwenImageLayeredQatConfig(**_base_config(tmp_path, attention_backend="sage_attention"))

    with pytest.raises(ValidationError, match="native_pytorch"):
        QwenImageLayeredQatConfig(**_base_config(tmp_path, training_framework="diffusers_trainer"))


def test_config_rejects_step_bounds_and_unfreeze_ambiguity(tmp_path):
    with pytest.raises(ValidationError, match="smoke schedule"):
        QwenImageLayeredQatConfig(
            **_base_config(tmp_path, debug_allow_short_run=False, max_steps=1)
        )

    with pytest.raises(ValidationError, match="train_priority"):
        QwenImageLayeredQatConfig(**_base_config(tmp_path, allow_full_weight_unfreeze=True))


def test_config_rejects_partial_unfreeze_without_sensitivity(tmp_path):
    with pytest.raises(ValidationError, match="disabled"):
        QwenImageLayeredQatConfig(
            **_base_config(
                tmp_path,
                recipe="partial_unfreeze",
                schedule="fallback",
                max_steps=500,
                debug_allow_short_run=False,
                optimizer={"learning_rate": 1.0e-6},
            )
        )


def test_load_qat_config_accepts_yaml(tmp_path):
    tuple_path = tmp_path / "tuple.pt"
    _write_tuple(tuple_path)
    manifest_path = tmp_path / "tuples.json"
    _write_manifest(manifest_path, [tuple_path])
    config_path = tmp_path / "qat.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"tuple_manifest: {manifest_path}",
                f"output_dir: {tmp_path / 'out'}",
                "target_layers: [attn.to_q]",
                "max_steps: 1",
                "validation_interval_steps: 1",
                "early_stop_patience: 2",
                "checkpoint_interval_steps: 1",
                "recipe: lora_adapter",
                "schedule: smoke",
                "device: cpu",
                "debug_allow_short_run: true",
            ]
        ),
        encoding="utf-8",
    )

    config = load_qat_config(config_path)

    assert config.tuple_manifest == str(manifest_path)
    assert config.training_framework == "native_pytorch"


def test_tuple_dataset_rejects_empty_manifest(tmp_path):
    manifest_path = tmp_path / "tuples.json"
    manifest_path.write_text(json.dumps({"tuples": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="no tuple"):
        TransformerTupleDataset(manifest_path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"include_target": False}, "target_output"),
        ({"include_hidden": False}, "hidden_states"),
        ({"include_timestep": False}, "timestep"),
    ],
)
def test_tuple_dataset_rejects_missing_required_fields(tmp_path, kwargs, message):
    tuple_path = tmp_path / "tuple.pt"
    _write_tuple(tuple_path, **kwargs)
    manifest_path = tmp_path / "tuples.json"
    _write_manifest(manifest_path, [tuple_path])

    with pytest.raises(ValueError, match=message):
        TransformerTupleDataset(manifest_path)


def test_tuple_dataset_rejects_non_bf16_target(tmp_path):
    tuple_path = tmp_path / "tuple.pt"
    payload = _write_tuple(tuple_path)
    payload["target_output"] = payload["target_output"].float()
    torch.save(payload, tuple_path)
    manifest_path = tmp_path / "tuples.json"
    _write_manifest(manifest_path, [tuple_path])

    with pytest.raises(ValueError, match="torch.bfloat16"):
        TransformerTupleDataset(manifest_path)


def test_tuple_dataset_rejects_incompatible_shapes(tmp_path):
    tuple_path = tmp_path / "tuple.pt"
    payload = _write_tuple(tuple_path)
    payload["target_output"] = torch.randn(1, 3, 128).to(torch.bfloat16)
    torch.save(payload, tuple_path)
    manifest_path = tmp_path / "tuples.json"
    _write_manifest(manifest_path, [tuple_path])

    with pytest.raises(ValueError, match="shapes"):
        TransformerTupleDataset(manifest_path)


def test_prepare_qat_model_freezes_base_and_trains_adapters(tmp_path):
    model = _TinyQwenTransformer()
    config = QwenImageLayeredQatConfig(**_base_config(tmp_path))

    injections = prepare_qat_model(model, config, linear_cls=nn.Linear)

    assert len(injections) == 1
    replacement = model.transformer_blocks[0].attn.to_q
    assert isinstance(replacement, Mxfp8LoraAdapterLinear)
    assert not replacement.base.weight.requires_grad
    assert replacement.lora_down.weight.requires_grad
    assert replacement.lora_up.weight.requires_grad
    assert not model.transformer_blocks[0].attn.to_k.weight.requires_grad
    assert not model.proj_out.weight.requires_grad


def test_prepare_qat_model_partial_unfreeze_requires_sensitivity_target(tmp_path):
    sensitivity_path = tmp_path / "sensitivity.json"
    sensitivity_path.write_text(
        json.dumps({"target_layers": ["transformer_blocks.0.attn.to_q"]}),
        encoding="utf-8",
    )
    config = QwenImageLayeredQatConfig(
        **_base_config(
            tmp_path,
            recipe="partial_unfreeze",
            schedule="fallback",
            max_steps=500,
            validation_interval_steps=100,
            checkpoint_interval_steps=100,
            debug_allow_short_run=False,
            enable_partial_unfreeze=True,
            sensitivity_path=str(sensitivity_path),
            train_priority="partial_unfreeze",
            optimizer={"learning_rate": 1.0e-6},
        )
    )
    model = _TinyQwenTransformer()

    injections = prepare_qat_model(model, config, linear_cls=nn.Linear)

    assert len(injections) == 1
    replacement = model.transformer_blocks[0].attn.to_q
    assert isinstance(replacement, FakeMxfp8Linear)
    assert replacement.weight.requires_grad
    assert not model.transformer_blocks[0].attn.to_k.weight.requires_grad


def test_train_qat_one_synthetic_tuple_saves_checkpoint_metadata(tmp_path):
    teacher = _TinyQwenTransformer()
    hidden_states = torch.randn(1, 2, 128)
    timestep = torch.tensor([1.0])
    with torch.no_grad():
        target_output = teacher(
            hidden_states,
            timestep=timestep,
            return_dict=False,
        )[0].to(torch.bfloat16)
    tuple_path = tmp_path / "tuple.pt"
    torch.save(
        {
            "args": [],
            "kwargs": {
                "hidden_states": hidden_states,
                "timestep": timestep,
                "return_dict": False,
            },
            "target_output": target_output,
        },
        tuple_path,
    )
    manifest_path = tmp_path / "tuples.json"
    _write_manifest(manifest_path, [tuple_path])
    config = QwenImageLayeredQatConfig(**_base_config(tmp_path, tuple_manifest=str(manifest_path)))

    result = train_qwen_image_layered_qat(
        config,
        transformer=_TinyQwenTransformer(),
        linear_cls=nn.Linear,
    )

    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["format"] == "qwen_image_layered_mxfp8_qat_adapter_v1"
    assert checkpoint["config"]["recipe"] == "lora_adapter"
    assert checkpoint["injections"][0]["module_name"] == "transformer_blocks.0.attn.to_q"
    assert any("lora" in name for name in checkpoint["trainable_parameter_names"])
    assert result.metrics_path.exists()
    assert result.provenance_path.exists()
    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert provenance["tuple_count"] == 1
    assert provenance["injections"][0]["role"] == "attn.to_q"
