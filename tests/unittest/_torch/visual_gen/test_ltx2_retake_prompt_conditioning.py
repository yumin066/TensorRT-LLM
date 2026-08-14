# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for persistent LTX-2 retake prompt conditioning."""

from __future__ import annotations

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_retake import LTX2RetakePipeline
from tensorrt_llm._torch.visual_gen.models.ltx2.retake_prompt_conditioning import (
    DEFAULT_RETAKE_MAX_SEQUENCE_LENGTH,
    DEFAULT_RETAKE_PROMPT,
    RetakePromptConditioning,
    checkpoint_fingerprint,
    is_default_retake_prompt,
    load_retake_prompt_conditioning,
    save_retake_prompt_conditioning,
    sha256_path,
)

_CHECKPOINT_SHA256 = "a" * 64
_CHECKPOINT_FINGERPRINT = "f" * 64


def _conditioning() -> RetakePromptConditioning:
    return RetakePromptConditioning(
        prompt=DEFAULT_RETAKE_PROMPT,
        max_sequence_length=DEFAULT_RETAKE_MAX_SEQUENCE_LENGTH,
        checkpoint_sha256=_CHECKPOINT_SHA256,
        checkpoint_fingerprint=_CHECKPOINT_FINGERPRINT,
        text_encoder_id="gemma-test",
        video_embeds=torch.arange(32, dtype=torch.bfloat16).reshape(1, 4, 8),
        audio_embeds=torch.arange(16, dtype=torch.bfloat16).reshape(1, 4, 4),
        connector_mask=torch.zeros(1, 1, 1, 4, dtype=torch.bfloat16),
    )


def test_default_prompt_match_only_normalizes_outer_whitespace():
    assert is_default_retake_prompt(f"  {DEFAULT_RETAKE_PROMPT}\n")
    assert not is_default_retake_prompt(DEFAULT_RETAKE_PROMPT.upper())
    assert not is_default_retake_prompt(DEFAULT_RETAKE_PROMPT + ".")


def test_prompt_conditioning_round_trip_is_exact(tmp_path):
    path = tmp_path / "conditioning.safetensors"
    expected = _conditioning()
    save_retake_prompt_conditioning(path, expected)

    actual = load_retake_prompt_conditioning(
        path,
        expected_prompt=DEFAULT_RETAKE_PROMPT,
        expected_max_sequence_length=DEFAULT_RETAKE_MAX_SEQUENCE_LENGTH,
        expected_checkpoint_fingerprint=_CHECKPOINT_FINGERPRINT,
    )

    assert actual.prompt == expected.prompt
    assert actual.text_encoder_id == expected.text_encoder_id
    assert torch.equal(actual.video_embeds, expected.video_embeds)
    assert torch.equal(actual.audio_embeds, expected.audio_embeds)
    assert torch.equal(actual.connector_mask, expected.connector_mask)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"expected_prompt": "different"}, "prompt mismatch"),
        ({"expected_max_sequence_length": 256}, "max sequence length mismatch"),
        ({"expected_checkpoint_fingerprint": "b" * 64}, "checkpoint mismatch"),
    ],
)
def test_prompt_conditioning_rejects_incompatible_metadata(tmp_path, kwargs, match):
    path = tmp_path / "conditioning.safetensors"
    save_retake_prompt_conditioning(path, _conditioning())
    expected = {
        "expected_prompt": DEFAULT_RETAKE_PROMPT,
        "expected_max_sequence_length": DEFAULT_RETAKE_MAX_SEQUENCE_LENGTH,
        "expected_checkpoint_fingerprint": _CHECKPOINT_FINGERPRINT,
    }
    expected.update(kwargs)

    with pytest.raises(ValueError, match=match):
        load_retake_prompt_conditioning(path, **expected)


def test_sha256_path_uses_file_bytes(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    assert (
        sha256_path(checkpoint)
        == "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef"
    )


def test_checkpoint_fingerprint_is_fast_metadata_identity(tmp_path):
    import safetensors.torch

    first = tmp_path / "first.safetensors"
    renamed = tmp_path / "renamed.safetensors"
    safetensors.torch.save_file(
        {"weight": torch.ones(2, dtype=torch.bfloat16)},
        str(first),
        metadata={"config": '{"model":"ltx2"}'},
    )
    renamed.write_bytes(first.read_bytes())
    assert checkpoint_fingerprint(first) == checkpoint_fingerprint(renamed)


class _NativeMustNotEncode:
    def _encode_prompt(self, *_args, **_kwargs):
        raise AssertionError("Gemma must not run for cached prompt conditioning")


class _NativeRecorder:
    def __init__(self):
        self.encode_calls = []
        self.connector_calls = []

    def _encode_prompt(self, prompt, **kwargs):
        self.encode_calls.append((prompt, kwargs))
        return torch.ones(1, 2, 3), torch.ones(1, 2)

    def _process_connectors(self, embeds, mask):
        self.connector_calls.append((embeds, mask))
        return torch.ones(1, 2, 4), torch.ones(1, 2, 2), torch.zeros(1, 1, 1, 2)


def _pipeline_stub(conditioning):
    pipeline = LTX2RetakePipeline.__new__(LTX2RetakePipeline)
    object.__setattr__(pipeline, "_prompt_conditioning", conditioning)
    return pipeline


def test_cached_prompt_conditioning_skips_gemma_and_connectors():
    outputs = _pipeline_stub(_conditioning())._prepare_prompt_conditioning(
        _NativeMustNotEncode(),
        DEFAULT_RETAKE_PROMPT,
        DEFAULT_RETAKE_MAX_SEQUENCE_LENGTH,
        torch.device("cpu"),
        torch.bfloat16,
    )
    expected = (
        _conditioning().video_embeds,
        _conditioning().audio_embeds,
        _conditioning().connector_mask,
    )
    assert all(torch.equal(actual, reference) for actual, reference in zip(outputs, expected))


def test_custom_prompt_runs_gemma_and_connectors():
    native = _NativeRecorder()
    outputs = _pipeline_stub(None)._prepare_prompt_conditioning(
        native,
        "a custom prompt",
        256,
        torch.device("cpu"),
        torch.bfloat16,
    )
    assert native.encode_calls == [
        ("a custom prompt", {"num_videos_per_prompt": 1, "max_sequence_length": 256})
    ]
    assert len(native.connector_calls) == 1
    assert len(outputs) == 3


def test_cached_prompt_conditioning_rejects_a_different_request_prompt():
    with pytest.raises(ValueError, match="does not match"):
        _pipeline_stub(_conditioning())._prepare_prompt_conditioning(
            _NativeMustNotEncode(),
            "a custom prompt",
            DEFAULT_RETAKE_MAX_SEQUENCE_LENGTH,
            torch.device("cpu"),
            torch.bfloat16,
        )
