#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline global-scale calibration for the LTX-2.3 Retake video VAE.

This tool deliberately lives outside the runtime model implementation.  It
prepares every ``delete_disfluency`` case with the script-editing evaluation
harness, runs a BF16 VAE round trip, and records the maximum input magnitude at
each Conv3d selected by the stable Retake NVFP4 recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import OrderedDict
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import (
    _find_safetensors_files,
    _load_component_weights,
    _read_safetensors_config,
)
from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.video_vae_ltx23 import (
    LTX23VideoDecoderConfigurator,
    LTX23VideoEncoderConfigurator,
)
from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.video_vae_ltx23_nvfp4 import (
    NVFP4_GLOBAL_QUANT_MAX,
    decoder_resnet_sites,
    decoder_upsample_sites,
    encoder_resnet_sites,
)
from tensorrt_llm._torch.visual_gen.models.ltx23.media_io import (
    decode_video_by_frame,
    get_videostream_metadata,
)
from tensorrt_llm._torch.visual_gen.models.ltx23.pipeline_ltx23_retake import _RETAKE_TILING_CONFIG


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-case", type=int, default=0)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument(
        "--preprocess-accelerator",
        choices=("auto", "cpu", "nvcodec"),
        default="auto",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_metadata(paths: list[str]) -> OrderedDict:
    files = []
    for value in paths:
        path = Path(value)
        files.append(
            OrderedDict(
                filename=path.name,
                size_bytes=path.stat().st_size,
                sha256=_sha256(path),
            )
        )
    if len(files) == 1:
        return files[0]
    return OrderedDict(files=files)


class _MaxCollector:
    def __init__(self) -> None:
        self.maxima: OrderedDict[str, float] = OrderedDict()
        self.calls: OrderedDict[str, int] = OrderedDict()
        self._handles = []

    def watch(self, name: str, module: torch.nn.Module) -> None:
        if name in self.maxima:
            raise ValueError(f"duplicate calibration site: {name}")
        self.maxima[name] = 0.0
        self.calls[name] = 0

        def capture(_module, args) -> None:
            observed = float(args[0].detach().float().abs().amax())
            self.maxima[name] = max(self.maxima[name], observed)
            self.calls[name] += 1

        self._handles.append(module.register_forward_pre_hook(capture))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _load_vae(checkpoint: str, device: torch.device):
    paths = _find_safetensors_files(checkpoint)
    if not paths:
        raise FileNotFoundError(f"no safetensors checkpoint found at {checkpoint}")
    config = _read_safetensors_config(paths[0])
    if config is None:
        raise ValueError("checkpoint has no embedded LTX-2.3 config")

    encoder = LTX23VideoEncoderConfigurator.from_config(config)
    _load_component_weights(paths, encoder, "vae.encoder.")
    _load_component_weights(paths, encoder.per_channel_statistics, "vae.per_channel_statistics.")
    encoder = encoder.eval().to(device=device, dtype=torch.bfloat16)

    decoder = LTX23VideoDecoderConfigurator.from_config(config)
    _load_component_weights(paths, decoder, ["vae.decoder.", "vae."])
    decoder = decoder.eval().to(device=device, dtype=torch.bfloat16)
    return encoder, decoder


def _register_sites(encoder, decoder):
    encoder_collector = _MaxCollector()
    for site in encoder_resnet_sites(encoder):
        encoder_collector.watch(f"{site.name}.conv1", site.module.conv1)
        encoder_collector.watch(f"{site.name}.conv2", site.module.conv2)

    decoder_collector = _MaxCollector()
    for site in decoder_resnet_sites(decoder):
        decoder_collector.watch(f"{site.name}.conv1", site.module.conv1)
        decoder_collector.watch(f"{site.name}.conv2", site.module.conv2)
    for site in decoder_upsample_sites(decoder):
        decoder_collector.watch(site.name, site.module.conv)
    return encoder_collector, decoder_collector


def _read_video(path: str, device: torch.device) -> torch.Tensor:
    shape = get_videostream_metadata(path)
    frames = list(decode_video_by_frame(path))
    if len(frames) != shape.frames:
        raise ValueError(f"decoded {len(frames)} frames, expected {shape.frames}: {path}")
    uint8 = torch.cat(frames, dim=0)
    del frames
    normalized = uint8.to(torch.float32).div_(127.5).sub_(1.0)
    del uint8
    return normalized.permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=torch.bfloat16)


def _write_result(
    output: Path,
    *,
    args: argparse.Namespace,
    total_cases: int,
    checkpoint: Mapping[str, object],
    dataset_sha256: str,
    hardware: str,
    completed_cases: list[str],
    case_geometry: Mapping[str, dict[str, int]],
    encoder_collector: _MaxCollector,
    decoder_collector: _MaxCollector,
) -> None:
    def sites(collector: _MaxCollector) -> OrderedDict[str, dict[str, float | int]]:
        result = OrderedDict()
        for name, maximum in collector.maxima.items():
            result[name] = {
                "observed_amax": maximum,
                "activation_quant_multiplier": (
                    NVFP4_GLOBAL_QUANT_MAX / maximum if maximum > 0.0 else 0.0
                ),
                "forward_calls": collector.calls[name],
            }
        return result

    payload = OrderedDict(
        schema_version=1,
        model="LTX-2.3",
        pipeline="retake",
        recipe="resnet_conv3d_plus_decoder_upsample_conv3d",
        checkpoint=checkpoint,
        quantization=OrderedDict(
            activation_format="NVFP4 E2M1",
            activation_block_size=16,
            dynamic_block_scale_format="E4M3",
            static_global_scale_format="FP32",
            global_quant_max=NVFP4_GLOBAL_QUANT_MAX,
        ),
        calibration=OrderedDict(
            dataset=(f"LTX2.3-eval/{args.cases.relative_to(args.eval_root)}#delete_disfluency"),
            dataset_sha256=dataset_sha256,
            total_case_count=total_cases,
            completed_case_count=len(completed_cases),
            completed_cases=completed_cases,
            case_geometry=case_geometry,
            method="max",
            global_quant_max=NVFP4_GLOBAL_QUANT_MAX,
            headroom=1.0,
            activation_dtype="bfloat16",
            note=(
                "Each case uses its script-editing Retake input window. Short clips that cannot "
                "fit the default 209 frames use their largest valid 8k+1 window. Decoder "
                "calibration runs on the BF16 encoder latent for the same window."
            ),
            calibrated_on=date.today().isoformat(),
            calibration_tool="examples/visual_gen/calibrate_ltx23_retake_vae_nvfp4.py",
            hardware=hardware,
            activation_quant_multiplier_formula=(f"{NVFP4_GLOBAL_QUANT_MAX:g} / observed_amax"),
        ),
        encoder=sites(encoder_collector),
        decoder=sites(decoder_collector),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    args.eval_root = args.eval_root.resolve()
    args.cases = args.cases.resolve()
    args.work_dir = args.work_dir.resolve()
    args.output = args.output.resolve()
    sys.path.insert(0, str(args.eval_root))

    from nvidia_acceleration.nvcodec.media_backend import ScriptEditingMediaEngine
    from script_editing.benchmark.delete_disfluency_benchmark import prepare_case, video_frame_count

    all_cases = json.loads(args.cases.read_text())["delete_disfluency"]
    stop = (
        len(all_cases)
        if args.stop_after is None
        else min(len(all_cases), args.start_case + args.stop_after)
    )
    selected = all_cases[args.start_case : stop]
    if not selected:
        raise ValueError("selected calibration case range is empty")

    device = torch.device("cuda")
    checkpoint_paths = _find_safetensors_files(args.checkpoint)
    if not checkpoint_paths:
        raise FileNotFoundError(f"no safetensors checkpoint found at {args.checkpoint}")
    checkpoint = _checkpoint_metadata(checkpoint_paths)
    dataset_sha256 = _sha256(args.cases)
    capability = torch.cuda.get_device_capability(device)
    hardware = f"{torch.cuda.get_device_name(device)} (SM{capability[0]}{capability[1]})"
    encoder, decoder = _load_vae(args.checkpoint, device)
    encoder_collector, decoder_collector = _register_sites(encoder, decoder)
    media_engine = ScriptEditingMediaEngine(
        preprocess_accelerator=args.preprocess_accelerator,
        postprocess_accelerator="cpu",
    )
    completed: list[str] = []
    case_geometry: OrderedDict[str, dict[str, int]] = OrderedDict()
    if args.output.is_file():
        previous = json.loads(args.output.read_text())
        calibration = previous.get("calibration", {})
        if calibration.get("dataset_sha256") != dataset_sha256:
            raise ValueError("existing output was calibrated with a different dataset")
        completed = list(calibration.get("completed_cases", []))
        case_geometry.update(calibration.get("case_geometry", {}))
        for name, record in previous.get("encoder", {}).items():
            encoder_collector.maxima[name] = float(record["observed_amax"])
            encoder_collector.calls[name] = int(record["forward_calls"])
        for name, record in previous.get("decoder", {}).items():
            decoder_collector.maxima[name] = float(record["observed_amax"])
            decoder_collector.calls[name] = int(record["forward_calls"])
        print(f"[calibration] resuming after {len(completed)} completed case(s)", flush=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index, case in enumerate(selected, start=args.start_case):
            case_id = f"{case['clip_id']}_which{case['which']}"
            if case_id in completed:
                continue
            print(f"[calibration] [{index + 1}/{len(all_cases)}] {case_id}", flush=True)
            prep_args = SimpleNamespace(
                video=str(args.eval_root / case["video"]),
                transcript=str(args.eval_root / case["transcript"]),
                which=int(case["which"]),
                pad_ms=20.0,
                cond_frames=90,
                retake_frames=25,
                width=int(case["width"]),
                height=int(case["height"]),
            )
            case_dir = args.work_dir / case_id
            try:
                prepared = prepare_case(prep_args, media_engine, case_dir)
            except SystemExit as error:
                edited = case_dir / "edited_full.mp4"
                if not edited.is_file() or "too short" not in str(error):
                    raise
                edited_frames = video_frame_count(str(edited))
                largest_valid = ((edited_frames - 1) // 8) * 8 + 1
                fallback_cond_frames = max(0, (largest_valid - prep_args.retake_frames) // 2)
                while fallback_cond_frames > 0:
                    requested = 2 * fallback_cond_frames + prep_args.retake_frames
                    total = ((requested - 1 + 7) // 8) * 8 + 1
                    if total <= edited_frames:
                        break
                    fallback_cond_frames -= 1
                if fallback_cond_frames <= 0:
                    raise ValueError(f"{case_id} is too short for NVFP4 calibration") from error
                prep_args.cond_frames = fallback_cond_frames
                print(
                    f"[calibration] short clip: using cond_frames={fallback_cond_frames}",
                    flush=True,
                )
                prepared = prepare_case(prep_args, media_engine, case_dir)
            video = _read_video(prepared["retake_input"], device)
            latent = encoder.tiled_encode(video, _RETAKE_TILING_CONFIG)
            del video
            for chunk in decoder.tiled_decode(latent, _RETAKE_TILING_CONFIG):
                del chunk
            del latent
            torch.cuda.synchronize()
            completed.append(case_id)
            case_geometry[case_id] = {
                "cond_frames_per_side": prep_args.cond_frames,
                "retake_frames_requested": prep_args.retake_frames,
                "retake_input_frames": int(prepared["total_frames"]),
                "width": prep_args.width,
                "height": prep_args.height,
            }
            _write_result(
                args.output,
                args=args,
                total_cases=len(all_cases),
                checkpoint=checkpoint,
                dataset_sha256=dataset_sha256,
                hardware=hardware,
                completed_cases=completed,
                case_geometry=case_geometry,
                encoder_collector=encoder_collector,
                decoder_collector=decoder_collector,
            )
            shutil.rmtree(case_dir)
            torch.cuda.empty_cache()
    finally:
        encoder_collector.close()
        decoder_collector.close()

    print(f"[calibration] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
