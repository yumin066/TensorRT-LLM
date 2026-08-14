# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent prompt conditioning for the native LTX-2 retake workflow."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import safetensors.torch
import torch

DEFAULT_RETAKE_PROMPT = "a person talking to the camera, natural head motion, clear speech"
DEFAULT_RETAKE_MAX_SEQUENCE_LENGTH = 1024
RETAKE_PROMPT_CONDITIONING_SCHEMA_VERSION = "1"

_VIDEO_EMBEDS_KEY = "video_embeds"
_AUDIO_EMBEDS_KEY = "audio_embeds"
_CONNECTOR_MASK_KEY = "connector_mask"
_REQUIRED_TENSOR_KEYS = {
    _VIDEO_EMBEDS_KEY,
    _AUDIO_EMBEDS_KEY,
    _CONNECTOR_MASK_KEY,
}


def normalize_retake_prompt(prompt: str) -> str:
    """Apply the same outer-whitespace normalization as Gemma prompt encoding."""
    return prompt.strip()


def is_default_retake_prompt(prompt: str) -> bool:
    """Return whether *prompt* can use the bundled default conditioning."""
    return normalize_retake_prompt(prompt) == DEFAULT_RETAKE_PROMPT


def sha256_path(path: str | Path) -> str:
    """Return a deterministic SHA-256 for a checkpoint file or shard directory."""
    checkpoint_path = Path(path)
    if checkpoint_path.is_file():
        files = [checkpoint_path]
        include_names = False
    elif checkpoint_path.is_dir():
        files = sorted(checkpoint_path.glob("*.safetensors"))
        include_names = True
        if not files:
            raise FileNotFoundError(f"No .safetensors checkpoint shards found in {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    digest = hashlib.sha256()
    for file_path in files:
        if include_names:
            digest.update(file_path.name.encode("utf-8"))
            digest.update(b"\0")
        with file_path.open("rb") as checkpoint_file:
            while chunk := checkpoint_file.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint(path: str | Path) -> str:
    """Return a fast checkpoint identity from shard layout, sizes, and config metadata."""
    checkpoint_path = Path(path)
    if checkpoint_path.is_file():
        files = [checkpoint_path]
        include_names = False
    elif checkpoint_path.is_dir():
        files = sorted(checkpoint_path.glob("*.safetensors"))
        include_names = True
        if not files:
            raise FileNotFoundError(f"No .safetensors checkpoint shards found in {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    digest = hashlib.sha256(b"ltx2-retake-checkpoint-fingerprint-v1\0")
    for file_path in files:
        if include_names:
            digest.update(file_path.name.encode("utf-8"))
            digest.update(b"\0")
        digest.update(str(file_path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with safetensors.torch.safe_open(str(file_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        digest.update(metadata.get("config", "").encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class RetakePromptConditioning:
    """Post-connector text tensors reusable across retake inputs and recipes."""

    prompt: str
    max_sequence_length: int
    checkpoint_sha256: str
    checkpoint_fingerprint: str
    text_encoder_id: str
    video_embeds: torch.Tensor
    audio_embeds: torch.Tensor
    connector_mask: torch.Tensor

    def tensors(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move the cached conditioning tensors to the pipeline device and dtype."""
        return (
            self.video_embeds.to(device=device, dtype=dtype),
            self.audio_embeds.to(device=device, dtype=dtype),
            self.connector_mask.to(device=device, dtype=dtype),
        )


def _validate_tensors(tensors: Mapping[str, torch.Tensor]) -> None:
    keys = set(tensors)
    if keys != _REQUIRED_TENSOR_KEYS:
        missing = sorted(_REQUIRED_TENSOR_KEYS - keys)
        unexpected = sorted(keys - _REQUIRED_TENSOR_KEYS)
        raise ValueError(
            "Retake prompt-conditioning tensor keys do not match the schema: "
            f"missing={missing}, unexpected={unexpected}"
        )

    video_embeds = tensors[_VIDEO_EMBEDS_KEY]
    audio_embeds = tensors[_AUDIO_EMBEDS_KEY]
    connector_mask = tensors[_CONNECTOR_MASK_KEY]
    if video_embeds.ndim != 3 or audio_embeds.ndim != 3:
        raise ValueError(
            "Retake prompt embeddings must be rank 3 [batch, sequence, channels]; "
            f"got video={tuple(video_embeds.shape)}, audio={tuple(audio_embeds.shape)}"
        )
    if connector_mask.ndim != 4:
        raise ValueError(
            "Retake connector mask must be rank 4 [batch, 1, 1, sequence]; "
            f"got {tuple(connector_mask.shape)}"
        )
    if video_embeds.shape[:2] != audio_embeds.shape[:2]:
        raise ValueError(
            "Video and audio prompt embeddings must have the same batch/sequence dimensions; "
            f"got video={tuple(video_embeds.shape)}, audio={tuple(audio_embeds.shape)}"
        )
    if (
        connector_mask.shape[0] != video_embeds.shape[0]
        or connector_mask.shape[-1] != video_embeds.shape[1]
    ):
        raise ValueError(
            "Connector mask must align with the prompt embeddings; "
            f"got mask={tuple(connector_mask.shape)}, video={tuple(video_embeds.shape)}"
        )
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors.values()):
        dtypes = {name: str(tensor.dtype) for name, tensor in tensors.items()}
        raise ValueError(f"Retake prompt conditioning must be BF16; got {dtypes}")
    if any(tensor.device.type != "cpu" for tensor in tensors.values()):
        raise ValueError("Retake prompt conditioning must be saved and loaded on CPU")


def save_retake_prompt_conditioning(
    path: str | Path,
    conditioning: RetakePromptConditioning,
) -> None:
    """Save post-connector retake prompt conditioning as a safe tensor payload."""
    tensors = {
        _VIDEO_EMBEDS_KEY: conditioning.video_embeds.detach()
        .to(device="cpu", dtype=torch.bfloat16)
        .contiguous(),
        _AUDIO_EMBEDS_KEY: conditioning.audio_embeds.detach()
        .to(device="cpu", dtype=torch.bfloat16)
        .contiguous(),
        _CONNECTOR_MASK_KEY: conditioning.connector_mask.detach()
        .to(device="cpu", dtype=torch.bfloat16)
        .contiguous(),
    }
    _validate_tensors(tensors)
    metadata = {
        "schema_version": RETAKE_PROMPT_CONDITIONING_SCHEMA_VERSION,
        "prompt": normalize_retake_prompt(conditioning.prompt),
        "max_sequence_length": str(conditioning.max_sequence_length),
        "dtype": "bfloat16",
        "checkpoint_sha256": conditioning.checkpoint_sha256,
        "checkpoint_fingerprint": conditioning.checkpoint_fingerprint,
        "text_encoder_id": conditioning.text_encoder_id,
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(tensors, str(output_path), metadata=metadata)


def load_retake_prompt_conditioning(
    path: str | Path,
    *,
    expected_prompt: str,
    expected_max_sequence_length: int,
    expected_checkpoint_fingerprint: str,
) -> RetakePromptConditioning:
    """Load and strictly validate a persistent retake prompt-conditioning cache."""
    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Retake prompt-conditioning cache does not exist: {cache_path}")

    with safetensors.torch.safe_open(str(cache_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}

    schema_version = metadata.get("schema_version")
    if schema_version != RETAKE_PROMPT_CONDITIONING_SCHEMA_VERSION:
        raise ValueError(
            "Retake prompt-conditioning schema mismatch: "
            f"cache={schema_version!r}, expected={RETAKE_PROMPT_CONDITIONING_SCHEMA_VERSION!r}"
        )
    cached_prompt = metadata.get("prompt", "")
    normalized_prompt = normalize_retake_prompt(expected_prompt)
    if cached_prompt != normalized_prompt:
        raise ValueError(
            "Retake prompt-conditioning prompt mismatch: "
            f"cache={cached_prompt!r}, request={normalized_prompt!r}"
        )
    try:
        cached_max_sequence_length = int(metadata["max_sequence_length"])
    except (KeyError, ValueError) as error:
        raise ValueError(
            "Retake prompt-conditioning metadata has an invalid max_sequence_length"
        ) from error
    if cached_max_sequence_length != expected_max_sequence_length:
        raise ValueError(
            "Retake prompt-conditioning max sequence length mismatch: "
            f"cache={cached_max_sequence_length}, request={expected_max_sequence_length}"
        )
    if metadata.get("dtype") != "bfloat16":
        raise ValueError(
            "Retake prompt-conditioning dtype mismatch: "
            f"cache={metadata.get('dtype')!r}, expected='bfloat16'"
        )
    cached_checkpoint_fingerprint = metadata.get("checkpoint_fingerprint", "")
    if cached_checkpoint_fingerprint != expected_checkpoint_fingerprint:
        raise ValueError(
            "Retake prompt-conditioning checkpoint mismatch: "
            f"cache={cached_checkpoint_fingerprint!r}, "
            f"request={expected_checkpoint_fingerprint!r}"
        )

    _validate_tensors(tensors)
    return RetakePromptConditioning(
        prompt=cached_prompt,
        max_sequence_length=cached_max_sequence_length,
        checkpoint_sha256=metadata.get("checkpoint_sha256", ""),
        checkpoint_fingerprint=cached_checkpoint_fingerprint,
        text_encoder_id=metadata.get("text_encoder_id", ""),
        video_embeds=tensors[_VIDEO_EMBEDS_KEY],
        audio_embeds=tensors[_AUDIO_EMBEDS_KEY],
        connector_mask=tensors[_CONNECTOR_MASK_KEY],
    )
