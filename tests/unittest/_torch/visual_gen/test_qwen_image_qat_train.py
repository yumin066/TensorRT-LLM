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
    CLOSED_SET_ROLLOUT_TIMESTEP_WEIGHTS,
    DEFAULT_TIMESTEP_WEIGHTS,
    QWEN_BLOCK_LINEAR_TARGET,
    ROLLOUT_TUPLE_SCHEMA_VERSION,
    FakeMxfp8Linear,
    Mxfp8LoraAdapterLinear,
    QwenImageQatMonitorConfig,
    QwenImageQatTrainingConfig,
    QwenImageRolloutLossConfig,
    QwenImageRolloutQatTrainingConfig,
    QwenImageRolloutStepSamples,
    QwenImageRolloutTupleAugmentationConfig,
    QwenImageRolloutWindowDataset,
    QwenImageTupleDataset,
    QwenImageTupleLossConfig,
    augment_qwen_image_rollout_tuple_dataset,
    build_parser,
    build_qwen_image_qat_probe_config,
    build_qwen_image_qat_scale_sensitivity_manifest,
    compute_qwen_image_rollout_loss,
    compute_qwen_image_tuple_loss,
    compute_scale_aware_lora_multipliers,
    forward_qwen_image_tuple,
    load_qwen_image_qat_checkpoint,
    load_qwen_image_rollout_metadata_manifest,
    load_qwen_image_rollout_qat_config,
    load_qwen_image_tuple_sample,
    monitor_qwen_image_qat_checkpoint,
    monitor_qwen_image_rollout_no_grad,
    normalize_qwen_image_qat_parameter_name,
    prepare_qwen_image_qat_model,
    qwen_image_qat_trainable_parameter_names,
    qwen_image_rollout_scheduler_step,
    run_qwen_image_qat_checkpoint_rgb_eval,
    run_qwen_image_rollout_tuple_augmentation,
    summarize_qwen_image_qat_training_metrics,
    train_qwen_image_qat,
    train_qwen_image_rollout_qat,
    validate_qwen_image_qat_probe_recipe,
    validate_qwen_image_rollout_tuple_sample,
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


class _IndexedRolloutTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.step0_scale = nn.Parameter(torch.tensor(1.0))
        self.step1_scale = nn.Parameter(torch.tensor(2.0))
        self.cond_branch_scale = nn.Parameter(torch.tensor(1.0))
        self.negative_branch_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, **kwargs: object) -> tuple[torch.Tensor]:
        hidden_states = kwargs["hidden_states"]
        timestep = kwargs["timestep"]
        encoder_hidden_states = kwargs["encoder_hidden_states"]
        if (
            not isinstance(hidden_states, torch.Tensor)
            or not isinstance(timestep, torch.Tensor)
            or not isinstance(encoder_hidden_states, torch.Tensor)
        ):
            raise TypeError("hidden_states, timestep, and encoder_hidden_states must be tensors")
        scale = self.step0_scale if int(timestep.flatten()[0].item()) == 0 else self.step1_scale
        branch_indicator = float(encoder_hidden_states.flatten()[0].item())
        branch_scale = (
            self.cond_branch_scale if branch_indicator > 0.0 else self.negative_branch_scale
        )
        return (hidden_states * scale * branch_scale,)


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


def _rollout_scheduler_step(
    latent: torch.Tensor,
    output: torch.Tensor,
    _step: object,
) -> torch.Tensor:
    return latent + output


def _rollout_tuple_fields(
    *,
    branch: str = "cond",
    timestep_index: int = 0,
    hidden_size: int = 1,
    image_seq_len: int = 1,
    text_seq_len: int = 1,
) -> dict[str, object]:
    branch_indicator = 1.0 if branch == "cond" else -1.0
    latent_before_value = 1.0 - float(timestep_index)
    latent_after_value = -float(timestep_index)
    return {
        "cfg_branch": branch,
        "timestep_index": timestep_index,
        "timestep_bin": "late",
        "timestep": torch.tensor([float(timestep_index)], dtype=torch.bfloat16),
        "hidden_states": torch.ones(1, image_seq_len, hidden_size, dtype=torch.bfloat16),
        "target_output": torch.zeros(1, image_seq_len, hidden_size, dtype=torch.bfloat16),
        "encoder_hidden_states": torch.full(
            (1, text_seq_len, hidden_size),
            branch_indicator,
            dtype=torch.bfloat16,
        ),
        "latent_before_step": torch.full(
            (1, image_seq_len, hidden_size),
            latent_before_value,
            dtype=torch.bfloat16,
        ),
        "latent_after_step": torch.full(
            (1, image_seq_len, hidden_size),
            latent_after_value,
            dtype=torch.bfloat16,
        ),
        "scheduler_state": {
            "scheduler_class": "FlowMatchEulerDiscreteScheduler",
            "timestep": float(timestep_index),
            "step_index": timestep_index,
            "num_inference_steps": 50,
            "sigma": float(timestep_index),
            "next_sigma": float(timestep_index) - 1.0,
            "order": 1,
        },
        "rollout_tuple_schema": ROLLOUT_TUPLE_SCHEMA_VERSION,
        "rollout_provenance": _rollout_provenance(),
        "guidance_scale": 4.0,
        "reference_image_path": "/durable/qwen_image_smoke_0000.png",
    }


def _rollout_provenance(**overrides: object) -> dict[str, object]:
    provenance: dict[str, object] = {
        "derivation_method": "unit_test_direct_latents",
        "scheduler_config_signature": "flowmatch_unit_config",
        "sigmas_hash": "unit_sigmas_hash",
        "git_head": "unit_git_head",
        "prompt_id": "qwen_image_smoke_0000",
        "seed": 1234,
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "guidance_scale": 4.0,
        "reference_image_path": "/durable/qwen_image_smoke_0000.png",
    }
    provenance.update(overrides)
    return provenance


def _rollout_command_provenance(**overrides: object) -> dict[str, object]:
    provenance: dict[str, object] = {
        "cluster_alias": "B300-mars",
        "allocation_id": "alloc-unit",
        "job_id": "job-unit",
        "node_list": "b300-unit-[0-7]",
        "container_runtime": "enroot",
        "enroot_image": "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20",
        "command": "unit rollout command",
        "model_snapshot_path": "/durable/Qwen-Image",
        "git_head": "unit_git_head",
        "teacher_attention_backend": "TRTLLM_FP8",
        "training_attention_backend": "TRTLLM_FP8",
        "evaluation_attention_backend": "TRTLLM_FP8",
        "closed_set_target_manifest": "/durable/closed_set_targets.jsonl",
        "scheduler_metadata_source": "/durable/scheduler_metadata.jsonl",
        "reference_image_root": "/durable/bf16_reference",
    }
    provenance.update(overrides)
    return provenance


def _rollout_metadata(
    *,
    timestep_index: int,
    hidden_size: int = 4,
    image_seq_len: int = 2,
) -> dict[str, object]:
    latent_before_value = 1.0 - float(timestep_index)
    latent_after_value = -float(timestep_index)
    return {
        "latent_before_step": torch.full(
            (1, image_seq_len, hidden_size),
            latent_before_value,
            dtype=torch.bfloat16,
        ),
        "latent_after_step": torch.full(
            (1, image_seq_len, hidden_size),
            latent_after_value,
            dtype=torch.bfloat16,
        ),
        "scheduler_state": {
            "scheduler_class": "FlowMatchEulerDiscreteScheduler",
            "timestep": float(timestep_index),
            "step_index": timestep_index,
            "num_inference_steps": 50,
            "sigma": float(timestep_index),
            "next_sigma": float(timestep_index) - 1.0,
            "order": 1,
        },
        "rollout_provenance": _rollout_provenance(),
        "guidance_scale": 4.0,
        "reference_image_path": "/durable/qwen_image_smoke_0000.png",
    }


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


def test_rollout_tuple_validation_requires_latent_fields(tmp_path: Path) -> None:
    tuple_path = tmp_path / "tuple.pt"
    _write_tuple(tuple_path)
    sample = load_qwen_image_tuple_sample(tuple_path)

    with pytest.raises(ValueError, match="schema"):
        validate_qwen_image_rollout_tuple_sample(sample)

    _write_tuple(
        tuple_path,
        **_rollout_tuple_fields(timestep_index=0),
    )
    sample = load_qwen_image_tuple_sample(tuple_path)

    validate_qwen_image_rollout_tuple_sample(sample)
    assert sample.latent_before_step is not None
    assert sample.latent_after_step is not None
    assert sample.scheduler_state is not None
    assert sample.scheduler_state["sigma"] == 0.0
    assert sample.scheduler_state["next_sigma"] == -1.0

    fields = _rollout_tuple_fields(timestep_index=0)
    fields["scheduler_state"] = None
    _write_tuple(tuple_path, **fields)
    sample = load_qwen_image_tuple_sample(tuple_path)

    with pytest.raises(ValueError, match="scheduler_state"):
        validate_qwen_image_rollout_tuple_sample(sample)


def test_compute_qwen_image_rollout_loss_requires_cfg_pairs(tmp_path: Path) -> None:
    tuple_path = tmp_path / "tuples" / "tuple_0_cond.pt"
    _write_tuple(tuple_path, **_rollout_tuple_fields(branch="cond", timestep_index=0))
    sample = load_qwen_image_tuple_sample(tuple_path)

    with pytest.raises(ValueError, match="paired CFG branch"):
        compute_qwen_image_rollout_loss(
            _IndexedRolloutTransformer(),
            [sample],
            scheduler_step_fn=_rollout_scheduler_step,
            device="cpu",
            config=QwenImageRolloutLossConfig(rollout_k=1),
            teacher_forward_fn=lambda latent, _sample: torch.zeros_like(latent),
        )


def test_compute_qwen_image_rollout_loss_rejects_duplicate_cfg_branch(
    tmp_path: Path,
) -> None:
    samples = []
    for suffix in ("a", "b"):
        tuple_path = tmp_path / "tuples" / f"tuple_0_cond_{suffix}.pt"
        _write_tuple(tuple_path, **_rollout_tuple_fields(branch="cond", timestep_index=0))
        samples.append(load_qwen_image_tuple_sample(tuple_path))

    with pytest.raises(ValueError, match="duplicate CFG branch"):
        compute_qwen_image_rollout_loss(
            _IndexedRolloutTransformer(),
            samples,
            scheduler_step_fn=_rollout_scheduler_step,
            device="cpu",
            config=QwenImageRolloutLossConfig(rollout_k=1),
            teacher_forward_fn=lambda latent, _sample: torch.zeros_like(latent),
        )


def test_compute_qwen_image_rollout_loss_rejects_cfg_pair_scheduler_mismatch(
    tmp_path: Path,
) -> None:
    samples = []
    for branch, scheduler_state in (
        ("cond", {"sigma": 0.0, "order": 1}),
        ("negative", {"sigma": 0.5, "order": 1}),
    ):
        tuple_path = tmp_path / "tuples" / f"tuple_0_{branch}.pt"
        fields = _rollout_tuple_fields(branch=branch, timestep_index=0)
        fields["scheduler_state"] = scheduler_state
        _write_tuple(tuple_path, **fields)
        samples.append(load_qwen_image_tuple_sample(tuple_path))

    with pytest.raises(ValueError, match="matching scheduler metadata"):
        compute_qwen_image_rollout_loss(
            _IndexedRolloutTransformer(),
            samples,
            scheduler_step_fn=_rollout_scheduler_step,
            device="cpu",
            config=QwenImageRolloutLossConfig(rollout_k=1),
            teacher_forward_fn=lambda latent, _sample: torch.zeros_like(latent),
        )


def test_compute_qwen_image_rollout_loss_rejects_cfg_pair_schedule_provenance_mismatch(
    tmp_path: Path,
) -> None:
    samples = []
    for branch, sigmas_hash in (
        ("cond", "unit_sigmas_hash_a"),
        ("negative", "unit_sigmas_hash_b"),
    ):
        tuple_path = tmp_path / "tuples" / f"tuple_0_{branch}.pt"
        fields = _rollout_tuple_fields(branch=branch, timestep_index=0)
        fields["scheduler_state"] = None
        fields["rollout_provenance"] = _rollout_provenance(
            scheduler_state_equivalent=True,
            sigmas_hash=sigmas_hash,
        )
        _write_tuple(tuple_path, **fields)
        samples.append(load_qwen_image_tuple_sample(tuple_path))

    with pytest.raises(ValueError, match="matching scheduler metadata"):
        compute_qwen_image_rollout_loss(
            _IndexedRolloutTransformer(),
            samples,
            scheduler_step_fn=_rollout_scheduler_step,
            device="cpu",
            config=QwenImageRolloutLossConfig(rollout_k=1),
            teacher_forward_fn=lambda latent, _sample: torch.zeros_like(latent),
        )


def test_qwen_image_rollout_scheduler_step_uses_captured_sigmas(tmp_path: Path) -> None:
    samples = []
    for branch in ("cond", "negative"):
        tuple_path = tmp_path / "tuples" / f"tuple_0_{branch}.pt"
        fields = _rollout_tuple_fields(branch=branch, timestep_index=0)
        fields["scheduler_state"] = {
            "scheduler_class": "FlowMatchEulerDiscreteScheduler",
            "timestep": 0.0,
            "sigma": 1.0,
            "next_sigma": 0.75,
            "step_index": 0,
            "num_inference_steps": 50,
        }
        _write_tuple(tuple_path, **fields)
        samples.append(load_qwen_image_tuple_sample(tuple_path))
    step = QwenImageRolloutStepSamples(
        timestep_index=0,
        timestep_bin="late",
        cond=samples[0],
        negative=samples[1],
        guidance_scale=4.0,
    )
    latent = torch.ones(1, 1, 1, dtype=torch.bfloat16)
    output = torch.full_like(latent, 2.0)

    next_latent = qwen_image_rollout_scheduler_step(latent, output, step)

    assert next_latent.item() == pytest.approx(0.5)


def test_qwen_image_rollout_scheduler_step_rejects_missing_delta(tmp_path: Path) -> None:
    samples = []
    for branch in ("cond", "negative"):
        tuple_path = tmp_path / "tuples" / f"tuple_0_{branch}.pt"
        fields = _rollout_tuple_fields(branch=branch, timestep_index=0)
        fields["scheduler_state"] = {
            "scheduler_class": "FlowMatchEulerDiscreteScheduler",
            "timestep": 0.0,
            "step_index": 0,
            "num_inference_steps": 50,
            "sigma": 1.0,
        }
        _write_tuple(tuple_path, **fields)
        samples.append(load_qwen_image_tuple_sample(tuple_path))
    step = QwenImageRolloutStepSamples(
        timestep_index=0,
        timestep_bin="late",
        cond=samples[0],
        negative=samples[1],
        guidance_scale=4.0,
    )

    with pytest.raises(ValueError, match="next_sigma"):
        qwen_image_rollout_scheduler_step(
            torch.ones(1, 1, 1, dtype=torch.bfloat16),
            torch.ones(1, 1, 1, dtype=torch.bfloat16),
            step,
        )


def test_qwen_image_rollout_scheduler_step_rejects_unsupported_scheduler(
    tmp_path: Path,
) -> None:
    samples = []
    for branch in ("cond", "negative"):
        tuple_path = tmp_path / "tuples" / f"tuple_0_{branch}.pt"
        fields = _rollout_tuple_fields(branch=branch, timestep_index=0)
        fields["scheduler_state"] = {
            "scheduler_class": "DDIMScheduler",
            "timestep": 0.0,
            "step_index": 0,
            "num_inference_steps": 50,
            "sigma": 1.0,
            "next_sigma": 0.75,
        }
        _write_tuple(tuple_path, **fields)
        samples.append(load_qwen_image_tuple_sample(tuple_path))
    step = QwenImageRolloutStepSamples(
        timestep_index=0,
        timestep_bin="late",
        cond=samples[0],
        negative=samples[1],
        guidance_scale=4.0,
    )

    with pytest.raises(ValueError, match="FlowMatch/Euler"):
        qwen_image_rollout_scheduler_step(
            torch.ones(1, 1, 1, dtype=torch.bfloat16),
            torch.ones(1, 1, 1, dtype=torch.bfloat16),
            step,
        )


def test_compute_qwen_image_rollout_loss_rejects_adjacent_latent_discontinuity(
    tmp_path: Path,
) -> None:
    samples = []
    for timestep_index in range(2):
        for branch in ("cond", "negative"):
            tuple_path = tmp_path / "tuples" / f"tuple_{timestep_index}_{branch}.pt"
            fields = _rollout_tuple_fields(branch=branch, timestep_index=timestep_index)
            if timestep_index == 1:
                fields["latent_before_step"] = torch.full(
                    (1, 1, 1),
                    7.0,
                    dtype=torch.bfloat16,
                )
            _write_tuple(tuple_path, **fields)
            samples.append(load_qwen_image_tuple_sample(tuple_path))

    with pytest.raises(ValueError, match="adjacent latent continuity mismatch"):
        compute_qwen_image_rollout_loss(
            _IndexedRolloutTransformer(),
            samples,
            scheduler_step_fn=_rollout_scheduler_step,
            device="cpu",
            config=QwenImageRolloutLossConfig(rollout_k=2),
            teacher_forward_fn=lambda latent, _sample: torch.zeros_like(latent),
        )


def test_compute_qwen_image_rollout_loss_backprops_through_previous_step(
    tmp_path: Path,
) -> None:
    samples = []
    for index in range(2):
        for branch in ("cond", "negative"):
            tuple_path = tmp_path / "tuples" / f"tuple_{index}_{branch}.pt"
            _write_tuple(tuple_path, **_rollout_tuple_fields(branch=branch, timestep_index=index))
            samples.append(load_qwen_image_tuple_sample(tuple_path))
    student = _IndexedRolloutTransformer()

    loss, components, records = compute_qwen_image_rollout_loss(
        student,
        samples,
        scheduler_step_fn=_rollout_scheduler_step,
        device="cpu",
        config=QwenImageRolloutLossConfig(
            rollout_k=2,
            lambda_output_mse=1.0,
            lambda_dir=0.0,
            lambda_latent=0.0,
            lambda_anchor=0.0,
            cfg_normalize=False,
        ),
        teacher_forward_fn=lambda latent, sample: latent
        if sample.timestep_index == 0
        else torch.zeros_like(latent),
    )
    loss.backward()

    assert records[0]["loss_total"] == pytest.approx(0.0)
    assert records[1]["loss_output_mse"] > 0.0
    assert records[1]["timestep_weight"] == pytest.approx(
        CLOSED_SET_ROLLOUT_TIMESTEP_WEIGHTS["late"]
    )
    assert components["rollout_steps"].item() == pytest.approx(2.0)
    assert student.step0_scale.grad is not None
    assert student.step0_scale.grad.abs().item() > 0.0
    assert student.step1_scale.grad is not None
    assert student.step1_scale.grad.abs().item() > 0.0
    assert student.cond_branch_scale.grad is not None
    assert student.cond_branch_scale.grad.abs().item() > 0.0
    assert student.negative_branch_scale.grad is not None
    assert student.negative_branch_scale.grad.abs().item() > 0.0


def test_monitor_qwen_image_rollout_no_grad_writes_summary(tmp_path: Path) -> None:
    samples = []
    for index in range(2):
        for branch in ("cond", "negative"):
            tuple_path = tmp_path / "tuples" / f"tuple_{index}_{branch}.pt"
            _write_tuple(tuple_path, **_rollout_tuple_fields(branch=branch, timestep_index=index))
            samples.append(load_qwen_image_tuple_sample(tuple_path))
    output_json = tmp_path / "rollout_monitor" / "summary.json"
    records_jsonl = tmp_path / "rollout_monitor" / "records.jsonl"

    summary = monitor_qwen_image_rollout_no_grad(
        _IndexedRolloutTransformer(),
        samples,
        scheduler_step_fn=_rollout_scheduler_step,
        output_json=output_json,
        records_jsonl=records_jsonl,
        device="cpu",
        config=QwenImageRolloutLossConfig(
            rollout_k=2,
            lambda_output_mse=1.0,
            lambda_dir=0.0,
            lambda_latent=0.0,
            lambda_anchor=0.0,
            cfg_normalize=False,
        ),
        teacher_forward_fn=lambda latent, sample: latent
        if sample.timestep_index == 0
        else torch.zeros_like(latent),
        provenance={"mode": "unit"},
    )

    assert output_json.is_file()
    assert records_jsonl.is_file()
    assert summary["format"] == "qwen_image_mxfp8_qat_rollout_monitor_v1"
    assert summary["metrics_summary"]["rollout_steps"] == pytest.approx(2.0)
    assert "cond_output_mse_sum" in summary["metrics_summary"]
    assert "negative_output_mse_sum" in summary["metrics_summary"]
    assert summary["provenance"] == {"mode": "unit"}
    records = [
        json.loads(line) for line in records_jsonl.read_text(encoding="utf-8").splitlines() if line
    ]
    assert [record["rollout_index"] for record in records] == [0, 1]
    assert records[0]["cfg_branches"] == ["cond", "negative"]
    assert records[0]["guidance_scale"] == pytest.approx(4.0)


def test_rollout_tuple_augmentation_writes_valid_schema(tmp_path: Path) -> None:
    source_index = tmp_path / "source.jsonl"
    output_index = tmp_path / "rollout.jsonl"
    source_entries = []
    metadata_by_tuple_id = {}
    for branch in ("cond", "negative"):
        tuple_path = tmp_path / "source_tuples" / f"tuple_0_{branch}.pt"
        _write_tuple(tuple_path, cfg_branch=branch, timestep_index=0)
        entry = _write_index(
            source_index,
            tuple_path,
            tuple_id=f"qwen_image_smoke_0000_step000_{branch}",
            cfg_branch=branch,
            timestep_index=0,
        )
        source_entries.append(entry)
        metadata_by_tuple_id[entry["tuple_id"]] = _rollout_metadata(timestep_index=0)
    source_index.write_text(
        "\n".join(json.dumps(entry) for entry in source_entries) + "\n",
        encoding="utf-8",
    )

    output_entries = augment_qwen_image_rollout_tuple_dataset(
        QwenImageRolloutTupleAugmentationConfig(
            source_tuple_index_jsonl=source_index,
            output_tuple_index_jsonl=output_index,
            output_tuple_root=tmp_path / "rollout_tuples",
            metadata_by_tuple_id=metadata_by_tuple_id,
        )
    )

    assert output_index.is_file()
    assert len(output_entries) == 2
    for entry in output_entries:
        assert entry["rollout_tuple_schema"] == ROLLOUT_TUPLE_SCHEMA_VERSION
        assert "latent_after_step" in entry["required_fields"]
        sample = load_qwen_image_tuple_sample(tmp_path / entry["tuple_path"], entry=entry)
        validate_qwen_image_rollout_tuple_sample(sample)


def test_rollout_metadata_manifest_loader_reads_tensor_paths(tmp_path: Path) -> None:
    before_path = tmp_path / "metadata" / "before.pt"
    after_path = tmp_path / "metadata" / "after.pt"
    before_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.ones(1, 2, 4, dtype=torch.bfloat16), before_path)
    torch.save(torch.zeros(1, 2, 4, dtype=torch.bfloat16), after_path)
    manifest_path = tmp_path / "metadata.jsonl"
    record = {
        "format": "qwen_image_rollout_metadata_entry_v1",
        "tuple_id": "qwen_image_smoke_0000_step000_cond",
        "latent_before_step_path": str(before_path),
        "latent_after_step_path": str(after_path),
        "scheduler_state": {
            "scheduler_class": "FlowMatchEulerDiscreteScheduler",
            "timestep": 0.0,
            "step_index": 0,
            "num_inference_steps": 50,
            "sigma": 1.0,
            "next_sigma": 0.75,
        },
        "rollout_provenance": _rollout_provenance(),
        "guidance_scale": 4.0,
        "reference_image_path": "/durable/qwen_image_smoke_0000.png",
    }
    manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    loaded = load_qwen_image_rollout_metadata_manifest(manifest_path)

    metadata = loaded["qwen_image_smoke_0000_step000_cond"]
    assert torch.equal(metadata["latent_before_step"], torch.ones(1, 2, 4, dtype=torch.bfloat16))
    assert metadata["scheduler_state"]["next_sigma"] == pytest.approx(0.75)


def test_run_rollout_tuple_augmentation_writes_summary_from_manifest(tmp_path: Path) -> None:
    source_index = tmp_path / "source.jsonl"
    source_entries = []
    metadata_records = []
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for branch in ("cond", "negative"):
        tuple_id = f"qwen_image_smoke_0000_step000_{branch}"
        tuple_path = tmp_path / "source_tuples" / f"tuple_0_{branch}.pt"
        _write_tuple(tuple_path, cfg_branch=branch, timestep_index=0)
        entry = _write_index(
            source_index,
            tuple_path,
            tuple_id=tuple_id,
            cfg_branch=branch,
            timestep_index=0,
        )
        source_entries.append(entry)
        before_path = metadata_dir / f"{tuple_id}_before.pt"
        after_path = metadata_dir / f"{tuple_id}_after.pt"
        torch.save(torch.ones(1, 2, 4, dtype=torch.bfloat16), before_path)
        torch.save(torch.zeros(1, 2, 4, dtype=torch.bfloat16), after_path)
        metadata_records.append(
            {
                "tuple_id": tuple_id,
                "latent_before_step_path": str(before_path),
                "latent_after_step_path": str(after_path),
                "scheduler_state": {
                    "scheduler_class": "FlowMatchEulerDiscreteScheduler",
                    "timestep": 0.0,
                    "step_index": 0,
                    "num_inference_steps": 50,
                    "sigma": 1.0,
                    "next_sigma": 0.75,
                },
                "rollout_provenance": _rollout_provenance(),
                "guidance_scale": 4.0,
                "reference_image_path": "/durable/qwen_image_smoke_0000.png",
            }
        )
    source_index.write_text(
        "\n".join(json.dumps(entry) for entry in source_entries) + "\n",
        encoding="utf-8",
    )
    metadata_manifest = tmp_path / "metadata.jsonl"
    metadata_manifest.write_text(
        "\n".join(json.dumps(record) for record in metadata_records) + "\n",
        encoding="utf-8",
    )
    summary_json = tmp_path / "summary.json"

    summary = run_qwen_image_rollout_tuple_augmentation(
        source_tuple_index_jsonl=source_index,
        rollout_metadata_jsonl=metadata_manifest,
        output_tuple_root=tmp_path / "rollout_tuples",
        output_tuple_index_jsonl=tmp_path / "rollout.jsonl",
        summary_json=summary_json,
        provenance=_rollout_command_provenance(),
    )

    assert summary_json.is_file()
    assert summary["format"] == "qwen_image_rollout_tuple_augmentation_summary_v1"
    assert summary["tuple_count"] == 2
    assert summary["provenance"]["cluster_alias"] == "B300-mars"
    assert summary["provenance"]["closed_set_target_manifest"]


def test_run_rollout_tuple_augmentation_rejects_incomplete_provenance(
    tmp_path: Path,
) -> None:
    metadata_manifest = tmp_path / "metadata.jsonl"
    metadata_manifest.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="closed_set_target_manifest"):
        run_qwen_image_rollout_tuple_augmentation(
            source_tuple_index_jsonl=tmp_path / "source.jsonl",
            rollout_metadata_jsonl=metadata_manifest,
            output_tuple_root=tmp_path / "rollout_tuples",
            output_tuple_index_jsonl=tmp_path / "rollout.jsonl",
            summary_json=tmp_path / "summary.json",
            provenance=_rollout_command_provenance(closed_set_target_manifest=None),
        )


def test_rollout_tuple_augmentation_relative_paths_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source_index = Path("source.jsonl")
    source_entries = []
    metadata_by_tuple_id = {}
    for branch in ("cond", "negative"):
        tuple_path = Path("source_tuples") / f"tuple_0_{branch}.pt"
        _write_tuple(tuple_path, cfg_branch=branch, timestep_index=0)
        entry = _write_index(
            source_index,
            tuple_path,
            tuple_id=f"qwen_image_smoke_0000_step000_{branch}",
            cfg_branch=branch,
            timestep_index=0,
        )
        source_entries.append(entry)
        metadata_by_tuple_id[entry["tuple_id"]] = _rollout_metadata(timestep_index=0)
    source_index.write_text(
        "\n".join(json.dumps(entry) for entry in source_entries) + "\n",
        encoding="utf-8",
    )

    output_entries = augment_qwen_image_rollout_tuple_dataset(
        QwenImageRolloutTupleAugmentationConfig(
            source_tuple_index_jsonl=source_index,
            output_tuple_index_jsonl=Path("runs/foo/rollout.jsonl"),
            output_tuple_root=Path("runs/foo/tuples"),
            metadata_by_tuple_id=metadata_by_tuple_id,
        )
    )

    for entry in output_entries:
        assert not Path(entry["tuple_path"]).is_absolute()
        assert not str(entry["tuple_path"]).startswith("runs/foo/runs/foo")
        sample = load_qwen_image_tuple_sample(
            Path("runs/foo") / str(entry["tuple_path"]),
            entry=entry,
        )
        validate_qwen_image_rollout_tuple_sample(sample)


def test_rollout_window_dataset_builds_consecutive_cfg_windows(tmp_path: Path) -> None:
    index_path = tmp_path / "rollout.jsonl"
    entries = []
    for timestep_index in range(2):
        for branch in ("cond", "negative"):
            tuple_path = tmp_path / "tuples" / f"tuple_{timestep_index}_{branch}.pt"
            _write_tuple(
                tuple_path,
                **_rollout_tuple_fields(branch=branch, timestep_index=timestep_index),
            )
            entries.append(
                {
                    "prompt_id": "qwen_image_smoke_0000",
                    "split": "smoke",
                    "timestep_index": timestep_index,
                    "timestep_bin": "late",
                    "cfg_branch": branch,
                    "trajectory_source": "bf16_teacher",
                    "status": "captured",
                    "tuple_path": str(tuple_path),
                }
            )
    index_path.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    dataset = QwenImageRolloutWindowDataset(index_path, rollout_k=2)

    assert len(dataset) == 1
    assert [sample.timestep_index for sample in dataset[0]] == [0, 0, 1, 1]


def test_build_parser_accepts_rollout_commands(tmp_path: Path) -> None:
    parser = build_parser()

    augment_args = parser.parse_args(
        [
            "augment-rollout-tuples",
            "--source-tuple-index-jsonl",
            str(tmp_path / "source.jsonl"),
            "--rollout-metadata-jsonl",
            str(tmp_path / "metadata.jsonl"),
            "--output-tuple-root",
            str(tmp_path / "tuples"),
            "--output-tuple-index-jsonl",
            str(tmp_path / "rollout.jsonl"),
            "--summary-json",
            str(tmp_path / "summary.json"),
        ]
    )
    rollout_args = parser.parse_args(
        [
            "run-rollout-qat",
            "--config",
            str(tmp_path / "rollout_qat.json"),
            "--model",
            "Qwen/Qwen-Image",
            "--visual-gen-args",
            str(tmp_path / "bf16.yaml"),
        ]
    )

    assert augment_args.subcommand == "augment-rollout-tuples"
    assert augment_args.func.__name__ == "_run_augment_rollout_tuples_command"
    assert rollout_args.subcommand == "run-rollout-qat"
    assert rollout_args.func.__name__ == "_run_rollout_qat_command"


def test_load_rollout_qat_config_defaults_closed_set_recipe(tmp_path: Path) -> None:
    config_path = tmp_path / "rollout_qat.json"
    config_path.write_text(
        json.dumps(
            {
                "tuple_index_jsonl": str(tmp_path / "rollout.jsonl"),
                "output_dir": str(tmp_path / "out"),
                "max_steps": 500,
                "checkpoint_interval_steps": 250,
                "loss": {"rollout_k": 4},
            }
        ),
        encoding="utf-8",
    )

    config = load_qwen_image_rollout_qat_config(config_path)

    assert config.max_steps == 500
    assert config.lora_rank == 64
    assert config.lora_alpha == pytest.approx(128.0)
    assert config.learning_rate == pytest.approx(2.0e-5)
    assert config.expected_num_layers == 60
    assert config.expected_target_count == 840
    assert config.diagnostic_only is False
    assert config.teacher_target_mode == "on_policy"
    assert config.loss.rollout_k == 4
    assert config.loss.timestep_weights["late"] == pytest.approx(8.0)
    assert config.loss.student_activation_checkpoint is False
    assert config.checkpoint_interval_steps == 250
    assert config.resume_checkpoint_path is None


def test_load_rollout_qat_config_requires_diagnostic_for_non_840(tmp_path: Path) -> None:
    config_path = tmp_path / "rollout_qat.json"
    base_config = {
        "tuple_index_jsonl": str(tmp_path / "rollout.jsonl"),
        "output_dir": str(tmp_path / "out"),
        "max_steps": 2,
        "expected_num_layers": 1,
        "expected_target_count": 1,
    }
    config_path.write_text(json.dumps(base_config), encoding="utf-8")

    with pytest.raises(ValueError, match="diagnostic_only=true"):
        load_qwen_image_rollout_qat_config(config_path)

    config_path.write_text(
        json.dumps({**base_config, "diagnostic_only": True}),
        encoding="utf-8",
    )

    config = load_qwen_image_rollout_qat_config(config_path)

    assert config.expected_num_layers == 1
    assert config.expected_target_count == 1
    assert config.diagnostic_only is True


def test_formal_rollout_defaults_match_prepare_target_count(tmp_path: Path) -> None:
    config_path = tmp_path / "rollout_qat.json"
    config_path.write_text(
        json.dumps(
            {
                "tuple_index_jsonl": str(tmp_path / "rollout.jsonl"),
                "output_dir": str(tmp_path / "out"),
                "max_steps": 1,
            }
        ),
        encoding="utf-8",
    )
    config = load_qwen_image_rollout_qat_config(config_path)
    model = _TinyQwenTransformer(num_layers=60)

    injections = prepare_qwen_image_qat_model(
        model,
        target_layers=config.target_layers,
        lora_rank=1,
        lora_alpha=1.0,
        lora_dropout=0.0,
        expected_num_layers=config.expected_num_layers,
        expected_target_count=config.expected_target_count,
        linear_cls=nn.Linear,
    )

    assert config.expected_num_layers == 60
    assert config.expected_target_count == 840
    assert len(injections) == 840


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
def test_train_qwen_image_rollout_qat_updates_lora_from_rollout_loss(tmp_path: Path) -> None:
    index_path = tmp_path / "rollout.jsonl"
    entries = []
    for branch in ("cond", "negative"):
        tuple_path = tmp_path / "tuples" / f"tuple_0_{branch}.pt"
        _write_tuple(
            tuple_path,
            **_rollout_tuple_fields(
                branch=branch,
                timestep_index=0,
                hidden_size=128,
                image_seq_len=2,
                text_seq_len=3,
            ),
        )
        entries.append(
            {
                "prompt_id": "qwen_image_smoke_0000",
                "split": "smoke",
                "timestep_index": 0,
                "timestep_bin": "late",
                "cfg_branch": branch,
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

    result = train_qwen_image_rollout_qat(
        model,
        QwenImageRolloutQatTrainingConfig(
            tuple_index_jsonl=index_path,
            output_dir=tmp_path / "out",
            max_steps=2,
            learning_rate=1.0e-3,
            device="cpu",
            lora_rank=4,
            lora_alpha=8.0,
            expected_num_layers=1,
            expected_target_count=14,
            diagnostic_only=True,
            teacher_target_mode="captured_tuple",
            loss=QwenImageRolloutLossConfig(
                rollout_k=1,
                cfg_normalize=False,
                student_activation_checkpoint=True,
            ),
            checkpoint_name="rollout_last.pt",
            checkpoint_interval_steps=1,
            compute_lora_delta_norm=True,
        ),
        scheduler_step_fn=_rollout_scheduler_step,
        teacher_forward_fn=lambda latent, _sample: torch.zeros_like(latent),
        linear_cls=nn.Linear,
    )

    records = [
        json.loads(line)
        for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert result.rollout_window_count == 1
    assert records[-1]["rollout_steps"] == pytest.approx(1.0)
    assert records[-1]["first_timestep_weight"] == pytest.approx(
        CLOSED_SET_ROLLOUT_TIMESTEP_WEIGHTS["late"]
    )
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["config"]["recipe"] == "closed_set_rollout_qat_v1"
    assert checkpoint["checkpoint_kind"] == "rollout_training_state"
    assert checkpoint["config"]["lora_rank"] == 4
    assert checkpoint["config"]["diagnostic_only"] is True
    assert checkpoint["config"]["teacher_target_mode"] == "captured_tuple"
    assert checkpoint["config"]["checkpoint_interval_steps"] == 1
    assert checkpoint["config"]["loss"]["timestep_weights"]["late"] == pytest.approx(8.0)
    assert checkpoint["config"]["loss"]["student_activation_checkpoint"] is True
    assert checkpoint["completed_steps"] == 2
    assert checkpoint["optimizer_state_dict"]
    assert checkpoint["sampling_state"]["dataset_size"] == 1
    step_checkpoint = result.output_dir / "rollout_last_step0001.pt"
    assert step_checkpoint.exists()
    step_payload = torch.load(step_checkpoint, map_location="cpu", weights_only=True)
    assert step_payload["train_steps"] == 1
    assert step_payload["checkpoint_kind"] == "rollout_training_state"
    assert (result.output_dir / "rollout_last_step0002.pt").exists()
    trainable_state = checkpoint["trainable_state_dict"]
    lora_up_tensors = [
        tensor for name, tensor in trainable_state.items() if str(name).endswith("lora_up.weight")
    ]
    assert lora_up_tensors
    assert any(tensor.abs().sum().item() > 0.0 for tensor in lora_up_tensors)


@requires_float8
def test_train_qwen_image_rollout_qat_resume_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "rollout.jsonl"
    entries = []
    for branch in ("cond", "negative"):
        tuple_path = tmp_path / "tuples" / f"tuple_0_{branch}.pt"
        _write_tuple(
            tuple_path,
            **_rollout_tuple_fields(
                branch=branch,
                timestep_index=0,
                hidden_size=128,
                image_seq_len=2,
                text_seq_len=3,
            ),
        )
        entries.append(
            {
                "prompt_id": "qwen_image_smoke_0000",
                "split": "smoke",
                "timestep_index": 0,
                "timestep_bin": "late",
                "cfg_branch": branch,
                "trajectory_source": "bf16_teacher",
                "status": "captured",
                "tuple_path": str(tuple_path),
            }
        )
    index_path.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    base_model = _TinyQwenTransformer().to(dtype=torch.bfloat16)
    base_state = {name: tensor.detach().clone() for name, tensor in base_model.state_dict().items()}

    def new_model() -> _TinyQwenTransformer:
        model = _TinyQwenTransformer().to(dtype=torch.bfloat16)
        model.load_state_dict(base_state)
        return model

    def rollout_config(
        output_dir: Path,
        *,
        max_steps: int,
        resume_checkpoint_path: Path | None = None,
    ) -> QwenImageRolloutQatTrainingConfig:
        return QwenImageRolloutQatTrainingConfig(
            tuple_index_jsonl=index_path,
            output_dir=output_dir,
            max_steps=max_steps,
            learning_rate=1.0e-3,
            device="cpu",
            lora_rank=4,
            lora_alpha=8.0,
            expected_num_layers=1,
            expected_target_count=14,
            diagnostic_only=True,
            teacher_target_mode="captured_tuple",
            loss=QwenImageRolloutLossConfig(rollout_k=1, cfg_normalize=False),
            checkpoint_name="rollout_last.pt",
            metrics_name="rollout_metrics.jsonl",
            resume_checkpoint_path=resume_checkpoint_path,
        )

    torch.manual_seed(1234)
    uninterrupted = train_qwen_image_rollout_qat(
        new_model(),
        rollout_config(tmp_path / "full", max_steps=2),
        scheduler_step_fn=_rollout_scheduler_step,
        teacher_forward_fn=lambda latent, _sample: torch.zeros_like(latent),
        linear_cls=nn.Linear,
    )

    torch.manual_seed(1234)
    first_leg = train_qwen_image_rollout_qat(
        new_model(),
        rollout_config(tmp_path / "resume", max_steps=1),
        scheduler_step_fn=_rollout_scheduler_step,
        teacher_forward_fn=lambda latent, _sample: torch.zeros_like(latent),
        linear_cls=nn.Linear,
    )
    resumed = train_qwen_image_rollout_qat(
        new_model(),
        rollout_config(
            tmp_path / "resume",
            max_steps=2,
            resume_checkpoint_path=first_leg.checkpoint_path,
        ),
        scheduler_step_fn=_rollout_scheduler_step,
        teacher_forward_fn=lambda latent, _sample: torch.zeros_like(latent),
        linear_cls=nn.Linear,
    )

    uninterrupted_state = load_qwen_image_qat_checkpoint(uninterrupted.checkpoint_path)[
        "trainable_state_dict"
    ]
    resumed_state = load_qwen_image_qat_checkpoint(resumed.checkpoint_path)["trainable_state_dict"]
    assert set(resumed_state) == set(uninterrupted_state)
    for name, tensor in uninterrupted_state.items():
        assert torch.equal(resumed_state[name], tensor), name
    records = [
        json.loads(line)
        for line in resumed.metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [record["step"] for record in records] == [1, 2]


@requires_float8
def test_rollout_qat_captured_tuple_requires_diagnostic_only(tmp_path: Path) -> None:
    index_path = tmp_path / "rollout.jsonl"
    index_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostic-only"):
        train_qwen_image_rollout_qat(
            _TinyQwenTransformer().to(dtype=torch.bfloat16),
            QwenImageRolloutQatTrainingConfig(
                tuple_index_jsonl=index_path,
                output_dir=tmp_path / "out",
                max_steps=1,
                teacher_target_mode="captured_tuple",
                expected_num_layers=1,
                expected_target_count=14,
                diagnostic_only=False,
            ),
            scheduler_step_fn=_rollout_scheduler_step,
            teacher_forward_fn=lambda latent, _sample: torch.zeros_like(latent),
            linear_cls=nn.Linear,
        )


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
