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
"""Run one-GPU native LTX-2 retake with the public VisualGen API.

The YAML recipe configures the retake window, seed, prompt conditioning, and
LoRA. Use ``--text_encoder_path`` and ``--prompt`` instead to encode text with
Gemma. Source decoding requires PyAV, and audio conditioning requires a
``torchaudio`` build compatible with the installed PyTorch.
"""

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
    parser.add_argument("--prompt", default="", help="Retake text prompt.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visual_gen_args = VisualGenArgs.from_yaml(args.visual_gen_args)
    pipeline_config = dict(visual_gen_args.pipeline_config)
    prompt_conditioning_path = pipeline_config.get("retake_prompt_conditioning_path")
    if prompt_conditioning_path is not None:
        if args.prompt:
            raise ValueError(
                "--prompt and pipeline_config.retake_prompt_conditioning_path "
                "are mutually exclusive."
            )
        if args.text_encoder_path is not None:
            raise ValueError(
                "--text_encoder_path and pipeline_config.retake_prompt_conditioning_path "
                "are mutually exclusive."
            )
    elif args.text_encoder_path is None and not pipeline_config.get("text_encoder_path"):
        raise ValueError(
            "Set pipeline_config.retake_prompt_conditioning_path, or set "
            "pipeline_config.text_encoder_path/--text_encoder_path for Gemma encoding."
        )

    if args.text_encoder_path is not None:
        pipeline_config["text_encoder_path"] = args.text_encoder_path
    visual_gen_args = visual_gen_args.model_copy(update={"pipeline_config": pipeline_config})

    with VisualGen(model=args.model, args=visual_gen_args) as visual_gen:
        params = visual_gen.default_params
        params.extra_params["retake_video_path"] = args.source
        output = visual_gen.generate(inputs=args.prompt, params=params)
        output.save(args.output_path)

    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
