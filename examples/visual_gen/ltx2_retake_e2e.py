#!/usr/bin/env python3
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

r"""End-to-end example for the native LTX-2 retake workflow.

Regenerates an internal time-window ``[start, end)`` of a source video and emits
the whole VAE-decoded retake clip. It drives the native
:class:`LTX2RetakePipeline` through the importable ``run_ltx2_retake`` API.

Example::

    : "${RETAKE_START:?set RETAKE_START to the retake-window start in seconds}"
    : "${RETAKE_END:?set RETAKE_END to the retake-window end in seconds}"

    python examples/visual_gen/ltx2_retake_e2e.py \
        --checkpoint /path/to/ltx-2.3-22b-distilled.safetensors \
        --source retake_input.mp4 --output retake_output.mp4 \
        --start "$RETAKE_START" --end "$RETAKE_END" \
        --prompt "a person talking to the camera, natural head motion, clear speech"

The bundled default-prompt conditioning skips Gemma loading and encoding. Pass
``--text-encoder`` when using a custom prompt or as a fallback when the cache is
unavailable or incompatible with the checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorrt_llm._torch.visual_gen.models.ltx2.retake import run_ltx2_retake
from tensorrt_llm._torch.visual_gen.models.ltx2.retake_prompt_conditioning import (
    DEFAULT_RETAKE_PROMPT,
)

_DEFAULT_PROMPT_CONDITIONING = (
    Path(__file__).resolve().parent / "ltx_retake" / "default_prompt_conditioning.safetensors"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the native LTX-2 retake workflow end-to-end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True, help="LTX-2 checkpoint (.safetensors or dir)."
    )
    parser.add_argument(
        "--text-encoder",
        default=None,
        help=(
            "Gemma3 text encoder directory. Required for a custom prompt and used "
            "as fallback if the default-prompt conditioning cache is unavailable."
        ),
    )
    parser.add_argument(
        "--prompt-conditioning-cache",
        default=str(_DEFAULT_PROMPT_CONDITIONING),
        help=(
            "Safetensors cache for the default prompt's post-connector conditioning. "
            "Ignored for a custom prompt."
        ),
    )
    parser.add_argument("--source", required=True, help="Source video to retake (.mp4).")
    parser.add_argument("--output", required=True, help="Output decoded retake video (.mp4).")
    parser.add_argument(
        "--dump-frames",
        default=None,
        help=(
            "Optional path to save the decoded uint8 frames (1, T, H, W, C) as a "
            ".pt tensor, before H.264 encoding."
        ),
    )
    parser.add_argument(
        "--start",
        type=float,
        required=True,
        metavar="SECONDS",
        help="Required start time of the regenerated window.",
    )
    parser.add_argument(
        "--end",
        type=float,
        required=True,
        metavar="SECONDS",
        help="Required end time of the regenerated window.",
    )
    parser.add_argument("--prompt", default=DEFAULT_RETAKE_PROMPT, help="Text prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=8,
        choices=(8,),
        help="Denoise steps; native retake uses the distilled 8-step schedule.",
    )
    parser.add_argument(
        "--quant-algo",
        default=None,
        choices=("FP8", "NVFP4"),
        help=(
            "Optional runtime weight-quantization recipe for the transformer linears. "
            "Omit for bf16."
        ),
    )
    parser.add_argument(
        "--fp8-linear-step",
        type=int,
        action="append",
        default=None,
        metavar="STEP",
        help=(
            "Run the transformer in FP8 at this diffusion step index instead of the "
            "primary quant. Repeatable; e.g. --fp8-linear-step 4 --fp8-linear-step 7."
        ),
    )
    parser.add_argument(
        "--nvfp4-attn",
        action="store_true",
        help="Use the SM120 FlashInfer NVFP4 video-self attention path.",
    )
    parser.add_argument(
        "--lora",
        default=None,
        help="Optional identity/style LoRA fused into the base transformer weights.",
    )
    parser.add_argument("--lora-strength", type=float, default=1.0, help="LoRA scale.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_ltx2_retake(
        checkpoint=args.checkpoint,
        source=args.source,
        output=args.output,
        start=args.start,
        end=args.end,
        prompt=args.prompt,
        seed=args.seed,
        text_encoder=args.text_encoder,
        prompt_conditioning_cache=args.prompt_conditioning_cache,
        lora=args.lora,
        lora_strength=args.lora_strength,
        quant_algo=args.quant_algo,
        nvfp4_attn=args.nvfp4_attn,
        fp8_linear_steps=args.fp8_linear_step,
        num_inference_steps=args.num_inference_steps,
        dump_frames=args.dump_frames,
    )


if __name__ == "__main__":
    main()
