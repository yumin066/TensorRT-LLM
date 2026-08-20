#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score an LTX-2 retake bridge window against a reference video with LPIPS."""

from __future__ import annotations

import argparse
import json
import os
import pathlib

import torch
from visual_gen_lpips_score_eval import (
    _decode_video_to_lpips_batch,
    _make_lpips_model,
    _validate_paired_video_shapes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-video", type=pathlib.Path, required=True)
    parser.add_argument("--generated-video", type=pathlib.Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-stop", type=int, required=True)
    parser.add_argument("--output-json", type=pathlib.Path, required=True)
    parser.add_argument("--lpips-net", default="alex")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_start < 0 or args.frame_start >= args.frame_stop:
        raise ValueError(
            "Retake LPIPS requires a non-empty half-open frame window: "
            f"got [{args.frame_start}, {args.frame_stop})."
        )

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generated = _decode_video_to_lpips_batch(args.generated_video, device)
    reference = _decode_video_to_lpips_batch(args.reference_video, device)
    available_frames = min(generated.shape[0], reference.shape[0])
    if args.frame_stop > available_frames:
        raise ValueError(
            f"Retake frame window [{args.frame_start}, {args.frame_stop}) exceeds "
            f"the {available_frames} paired decoded frames."
        )

    generated = generated[args.frame_start : args.frame_stop]
    reference = reference[args.frame_start : args.frame_stop]
    _validate_paired_video_shapes(generated, reference)
    model = _make_lpips_model(args.lpips_net, device)
    with torch.no_grad():
        score = float(model(generated, reference).flatten().mean().item())

    result = {
        "mean_lpips_score": score,
        "lpips_net": args.lpips_net,
        "lpips_device": device,
        "frame_start": args.frame_start,
        "frame_stop": args.frame_stop,
        "num_paired_frames": args.frame_stop - args.frame_start,
        "reference_video": str(args.reference_video),
        "generated_video": str(args.generated_video),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result) if args.json else f"LPIPS: {score:.6f}")


if __name__ == "__main__":
    main()
