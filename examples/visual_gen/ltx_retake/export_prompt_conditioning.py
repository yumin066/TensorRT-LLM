#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the default LTX-2 retake prompt conditioning to safetensors."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch


class _ConnectorOnlyTransformer(torch.nn.Module):
    """Minimal transformer contract needed while loading text connectors."""

    def __init__(self, transformer_config: dict, device: torch.device):
        super().__init__()
        self.register_buffer("_device_anchor", torch.empty(0, device=device), persistent=False)
        self._transformer_config = transformer_config

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device


def _sha256_digest(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise argparse.ArgumentTypeError("expected a 64-character SHA-256 hex digest")
    return normalized


def _parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parent / "default_prompt_conditioning.safetensors"
    parser = argparse.ArgumentParser(
        description="Export native LTX-2 post-connector conditioning for a fixed prompt."
    )
    parser.add_argument("--checkpoint", required=True, help="LTX-2 checkpoint file or directory.")
    parser.add_argument("--text-encoder", required=True, help="Gemma3 model directory.")
    parser.add_argument("--output", default=str(default_output), help="Output safetensors path.")
    parser.add_argument(
        "--text-encoder-id",
        default=None,
        help="Stable Gemma model/revision identifier stored in cache metadata.",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        default=None,
        type=_sha256_digest,
        help=(
            "Known full checkpoint SHA-256 to record without re-reading a large shared "
            "checkpoint. Omit to calculate it during export."
        ),
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, DiffusionPipelineConfig
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import (
        _find_safetensors_files,
        _read_safetensors_config,
    )
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_retake import _NativeLTX2Companion
    from tensorrt_llm._torch.visual_gen.models.ltx2.retake_prompt_conditioning import (
        DEFAULT_RETAKE_MAX_SEQUENCE_LENGTH,
        DEFAULT_RETAKE_PROMPT,
        RetakePromptConditioning,
        checkpoint_fingerprint,
        load_retake_prompt_conditioning,
        save_retake_prompt_conditioning,
        sha256_path,
    )

    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Exporting LTX-2 retake prompt conditioning requires a CUDA GPU.")

    checkpoint_files = _find_safetensors_files(args.checkpoint)
    if not checkpoint_files:
        raise FileNotFoundError(f"No LTX-2 safetensors checkpoint found at {args.checkpoint}")
    config = _read_safetensors_config(checkpoint_files[0])
    if config is None:
        raise ValueError(f"LTX-2 checkpoint config metadata is missing: {checkpoint_files[0]}")

    transformer_config = config.get("transformer", config)
    pipeline_config = DiffusionPipelineConfig(
        model_configs={
            "transformer": DiffusionModelConfig(
                pretrained_config=SimpleNamespace(**transformer_config)
            )
        }
    )
    pipeline_config.cuda_graph.enable = False
    device = torch.device("cuda")
    dummy_transformer = _ConnectorOnlyTransformer(transformer_config, device)
    native = _NativeLTX2Companion(pipeline_config, dummy_transformer)
    native.load_standard_components(
        args.checkpoint,
        device,
        text_encoder_path=args.text_encoder,
        skip_components=["vae", "audio_vae", "vocoder", "video_encoder", "scheduler"],
        prefetch_safetensors=False,
    )

    prompt_embeds, prompt_attention_mask = native._encode_prompt(
        DEFAULT_RETAKE_PROMPT,
        num_videos_per_prompt=1,
        max_sequence_length=DEFAULT_RETAKE_MAX_SEQUENCE_LENGTH,
    )
    video_embeds, audio_embeds, connector_mask = native._process_connectors(
        prompt_embeds, prompt_attention_mask
    )
    checkpoint_sha256 = args.checkpoint_sha256 or sha256_path(args.checkpoint)
    source_checkpoint_fingerprint = checkpoint_fingerprint(args.checkpoint)
    conditioning = RetakePromptConditioning(
        prompt=DEFAULT_RETAKE_PROMPT,
        max_sequence_length=DEFAULT_RETAKE_MAX_SEQUENCE_LENGTH,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_fingerprint=source_checkpoint_fingerprint,
        text_encoder_id=args.text_encoder_id or Path(args.text_encoder).resolve().name,
        video_embeds=video_embeds.cpu(),
        audio_embeds=audio_embeds.cpu(),
        connector_mask=connector_mask.cpu(),
    )
    save_retake_prompt_conditioning(args.output, conditioning)

    reloaded = load_retake_prompt_conditioning(
        args.output,
        expected_prompt=DEFAULT_RETAKE_PROMPT,
        expected_max_sequence_length=DEFAULT_RETAKE_MAX_SEQUENCE_LENGTH,
        expected_checkpoint_fingerprint=source_checkpoint_fingerprint,
    )
    for name in ("video_embeds", "audio_embeds", "connector_mask"):
        if not torch.equal(getattr(conditioning, name), getattr(reloaded, name)):
            raise RuntimeError(f"Prompt-conditioning serialization changed tensor {name}")

    print(
        f"saved {args.output} (checkpoint_sha256={checkpoint_sha256}, "
        f"video={tuple(video_embeds.shape)}, audio={tuple(audio_embeds.shape)}, "
        f"mask={tuple(connector_mask.shape)})",
        flush=True,
    )


if __name__ == "__main__":
    main()
