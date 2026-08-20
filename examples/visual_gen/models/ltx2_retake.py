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
"""Run a one-GPU native LTX-2 retake smoke case with the public VisualGen API."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorrt_llm import VisualGen, VisualGenArgs

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "ltx2-retake-1gpu.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="LTX-2.3 checkpoint file or directory.")
    parser.add_argument("--source", required=True, help="Source MP4 to retake.")
    parser.add_argument("--output_path", default="retake_output.mp4", help="Output MP4 path.")
    parser.add_argument("--start", type=float, required=True, help="Retake start time in seconds.")
    parser.add_argument("--end", type=float, required=True, help="Retake end time in seconds.")
    parser.add_argument("--prompt", default="", help="Retake text prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--visual_gen_args",
        default=str(_DEFAULT_CONFIG),
        help="VisualGen YAML configuration.",
    )
    parser.add_argument(
        "--text_encoder_path",
        default=None,
        help="Gemma text encoder path, required without precomputed prompt conditioning.",
    )
    parser.add_argument(
        "--prompt_conditioning_path",
        default=None,
        help="Optional precomputed prompt-conditioning safetensors file.",
    )
    parser.add_argument("--lora_path", default=None, help="Optional TalkVid/style LoRA path.")
    parser.add_argument("--lora_strength", type=float, default=1.0, help="LoRA scale.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prompt_conditioning_path is not None:
        if args.prompt:
            raise ValueError("--prompt and --prompt_conditioning_path are mutually exclusive.")
        if args.text_encoder_path is not None:
            raise ValueError(
                "--text_encoder_path and --prompt_conditioning_path are mutually exclusive."
            )
    elif args.text_encoder_path is None:
        raise ValueError(
            "Set --prompt_conditioning_path, or set --text_encoder_path for Gemma encoding."
        )

    visual_gen_args = VisualGenArgs.from_yaml(args.visual_gen_args)
    pipeline_config = dict(visual_gen_args.pipeline_config)
    if args.text_encoder_path is not None:
        pipeline_config["text_encoder_path"] = args.text_encoder_path
    if args.prompt_conditioning_path is not None:
        pipeline_config["retake_prompt_conditioning_path"] = args.prompt_conditioning_path
    if args.lora_path is not None:
        pipeline_config["retake_lora_path"] = args.lora_path
        pipeline_config["retake_lora_strength"] = args.lora_strength
    visual_gen_args = visual_gen_args.model_copy(update={"pipeline_config": pipeline_config})

    with VisualGen(model=args.model, args=visual_gen_args) as visual_gen:
        params = visual_gen.default_params
        params.seed = args.seed
        params.num_inference_steps = 8
        params.extra_params.update(
            {
                "retake_video_path": args.source,
                "retake_start_time": args.start,
                "retake_end_time": args.end,
            }
        )
        output = visual_gen.generate(inputs=args.prompt, params=params)
        output.save(args.output_path)

    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
