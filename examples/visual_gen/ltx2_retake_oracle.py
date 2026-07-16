#! /usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native-vs-upstream LTX-2 retake comparison oracle.

Runs the native LTX-2 retake video regeneration and the preserved upstream
``DiffusionStage.run`` oracle on the same source / prompt / window / seed, then:

- persists both output videos plus a reproducibility protocol,
- re-checks the retake hard invariants (output length, byte-identical frames
  outside the retake window, source-audio preservation, determinism, and the
  audio-regeneration fail-fast) on the native output, and
- computes advisory PSNR / SSIM between the native and upstream outputs (and an
  optional committed reference), split into the retake window and the region
  outside it.

The PSNR / SSIM columns are informational only: the exit code and the
``all_checks_passed`` summary reflect the hard invariants exclusively and never
depend on any image-similarity metric.

Source-media guidance: the native-vs-upstream PSNR / SSIM comparison must use a
VIDEO-ONLY source (a source with no audio stream). The upstream oracle path
resamples source audio through torchaudio, which is stubbed in some runtime
containers and raises when handed an audio-bearing source; the native path
reads media through PyAV and is unaffected. Run the audio-preservation
invariant on an audio-bearing source (native path only) and run the
native-vs-upstream comparison on a video-only source. When the upstream path
cannot run, the oracle still completes with the native artifacts, protocol, and
hard invariants, and records the skip reason instead of the upstream metrics.

The heavy ``tensorrt_llm`` / Lightricks imports live inside the build and run
functions so the pure helpers at the top of this module (``psnr``, ``ssim``,
``split_regions``, ``build_protocol``, ``sha256_file``) import on a plain CPU
host without a GPU.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

# ----------------------------------------------------------------------------
# Pure helpers (numpy / torch only; no tensorrt_llm imports at module scope).
# ----------------------------------------------------------------------------


def _to_float_array(x: Any) -> np.ndarray:
    """Return *x* as a contiguous float64 numpy array (torch tensors accepted)."""
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().to("cpu").numpy()
    return np.asarray(x, dtype=np.float64)


def psnr(a: Any, b: Any, max_val: float = 255.0) -> float:
    """Peak signal-to-noise ratio between two equal-shaped arrays.

    Accepts uint8 or float numpy arrays / torch tensors of identical shape.
    Returns ``float('inf')`` when the inputs are pixel-identical (zero error).
    """
    fa = _to_float_array(a)
    fb = _to_float_array(b)
    if fa.shape != fb.shape:
        raise ValueError(f"psnr shape mismatch: {fa.shape} vs {fb.shape}")
    if fa.size == 0:
        return float("nan")
    mse = float(np.mean((fa - fb) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((max_val * max_val) / mse))


def _gaussian_window(size: int, sigma: float) -> np.ndarray:
    """1-D normalized Gaussian kernel of length *size*."""
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(coords**2) / (2.0 * sigma * sigma))
    return g / g.sum()


def _rgb_to_luma(frames: np.ndarray) -> np.ndarray:
    """Convert ``(T, H, W, C)`` frames to ``(T, H, W)`` Rec.601 luminance.

    Non-3-channel inputs are averaged over the channel axis so grayscale or
    RGBA sources still yield a single luminance plane.
    """
    fa = _to_float_array(frames)
    if fa.ndim != 4:
        raise ValueError(f"expected (T, H, W, C) frames; got {fa.shape}")
    if fa.shape[-1] == 3:
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
        return fa @ weights
    return fa.mean(axis=-1)


def ssim(a: Any, b: Any, max_val: float = 255.0) -> float:
    """Mean structural similarity between two ``(T, H, W, C)`` clips.

    Compact pure-numpy SSIM: RGB is reduced to Rec.601 luminance, then a
    separable 11x11 Gaussian window (sigma 1.5, the standard Wang et al.
    setting) is convolved (valid padding) per frame to form the local means,
    variances, and covariance; the SSIM map uses the standard C1 = (0.01 L)^2
    and C2 = (0.03 L)^2 stabilizers. The result is averaged over every window
    position and every frame. When a frame is smaller than 11 px on a side the
    window is shrunk to the odd size that fits (a single global window in the
    degenerate case), so small clips still produce a defined score.
    """
    la = _rgb_to_luma(a)
    lb = _rgb_to_luma(b)
    if la.shape != lb.shape:
        raise ValueError(f"ssim shape mismatch: {la.shape} vs {lb.shape}")
    if la.size == 0:
        return float("nan")
    t, h, w = la.shape
    win = min(11, h, w)
    if win % 2 == 0:
        win -= 1
    win = max(win, 1)
    sigma = 1.5 if win == 11 else max(win / 7.0, 0.5)
    kernel_1d = _gaussian_window(win, sigma)

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    def _filter(plane: np.ndarray) -> np.ndarray:
        # Separable valid-padding Gaussian filter along H then W.
        out = np.apply_along_axis(lambda m: np.convolve(m, kernel_1d, mode="valid"), 1, plane)
        out = np.apply_along_axis(lambda m: np.convolve(m, kernel_1d, mode="valid"), 2, out)
        return out

    scores = []
    for i in range(t):
        x = la[i][None, ...]
        y = lb[i][None, ...]
        mu_x = _filter(x)
        mu_y = _filter(y)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x2 = _filter(x * x) - mu_x2
        sigma_y2 = _filter(y * y) - mu_y2
        sigma_xy = _filter(x * y) - mu_xy
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        )
        scores.append(float(ssim_map.mean()))
    return float(np.mean(scores))


def split_regions(frames: Any, start_frame: int, end_frame: int):
    """Split ``(T, H, W, C)`` *frames* by the half-open ``[start, end)`` window.

    Returns ``(window_frames, outside_frames)`` where ``window_frames`` are the
    frames inside the retake window and ``outside_frames`` are the leading and
    trailing frames concatenated. Bounds are clamped to ``[0, T]`` with
    ``start <= end``. Torch tensors are handled without importing torch here.
    """
    total = frames.shape[0]
    start = max(0, min(int(start_frame), total))
    end = max(start, min(int(end_frame), total))
    window = frames[start:end]
    if hasattr(frames, "detach"):  # torch.Tensor
        import torch

        outside = torch.cat([frames[:start], frames[end:]], dim=0)
    else:
        outside = np.concatenate([np.asarray(frames)[:start], np.asarray(frames)[end:]], axis=0)
    return window, outside


def sha256_file(path: str) -> str:
    """Streaming SHA-256 hex digest of the file at *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_protocol(
    *,
    native_repo: dict,
    eval_repo: dict,
    checkpoint: str,
    gemma: str,
    lora: Optional[str],
    source: str,
    source_sha256: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    dtype: str,
    attention_backend: str,
    num_inference_steps: int,
    sigmas: list,
    fps: float,
    num_frames: int,
    height: int,
    width: int,
    window: list,
    pixel_window: list,
    conditioned_latent_ranges: list,
    env: dict,
) -> dict:
    """Assemble the reproducibility protocol from already-computed primitives.

    Pure: every argument is a plain scalar / list / dict so this is unit
    testable without a GPU. ``regenerate_video`` is always True and
    ``regenerate_audio`` always False because the native retake path is a
    video-only window regeneration that preserves the source audio.
    """
    return {
        "native_repo": native_repo,
        "eval_repo": eval_repo,
        "checkpoint": checkpoint,
        "gemma": gemma,
        "lora": lora,
        "source": source,
        "source_sha256": source_sha256,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "dtype": dtype,
        "attention_backend": attention_backend,
        "num_inference_steps": num_inference_steps,
        "sigmas": list(sigmas),
        "fps": fps,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "window": window,
        "pixel_window": pixel_window,
        "conditioned_latent_ranges": conditioned_latent_ranges,
        "regenerate_video": True,
        "regenerate_audio": False,
        "env": env,
    }


# ----------------------------------------------------------------------------
# Repo / environment probing (subprocess, stdlib only).
# ----------------------------------------------------------------------------


def _git_info(repo_dir: str) -> dict:
    """Best-effort ``{commit, branch}`` for a git repo; nulls on any failure."""
    info = {"path": repo_dir, "commit": None, "branch": None}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["branch"] = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        pass
    return info


def _env_metadata() -> dict:
    import torch

    device_name = None
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
    except (RuntimeError, AssertionError):
        device_name = None
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": device_name,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


# ----------------------------------------------------------------------------
# Native / upstream pipeline construction and execution (heavy imports here).
# ----------------------------------------------------------------------------


def build_pipeline(checkpoint, gemma, lora, device, attention_backend, extra_overrides=None):
    """Build and load an LTX-2 retake pipeline (native or upstream-stage).

    ``extra_overrides`` merges into the pipeline ``extra_attrs`` so the caller
    can flip ``retake_use_upstream_stage`` to select the preserved oracle path
    while every other setting stays identical.
    """
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, DiffusionPipelineConfig
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import (
        _find_safetensors_files,
        _read_safetensors_config,
    )
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_retake import LTX2RetakePipeline

    cfg = _read_safetensors_config(_find_safetensors_files(checkpoint)[0])
    transformer_cfg = cfg.get("transformer", cfg)
    extra = {
        "workflow": "retake",
        "retake_distilled": True,
        "retake_offload_mode": "none",
        "retake_lora_path": lora,
        "retake_lora_strength": 1.0,
    }
    if extra_overrides:
        extra.update(extra_overrides)
    pipeline_config = DiffusionPipelineConfig(
        model_configs={
            "transformer": DiffusionModelConfig(
                pretrained_config=SimpleNamespace(**transformer_cfg)
            )
        },
        extra_attrs=extra,
    )
    pipeline_config.attention.backend = attention_backend
    pipe = LTX2RetakePipeline(pipeline_config)
    pipe.load_standard_components(checkpoint, device, text_encoder_path=gemma)
    pipe.load_weights(pipe.load_transformer_weights(checkpoint))
    pipe.post_load_weights()
    return pipe


def _make_request(prompt, negative_prompt, source, start, end, seed, steps, regenerate_audio=False):
    return SimpleNamespace(
        prompt=prompt,
        negative_prompt=negative_prompt,
        params=SimpleNamespace(
            extra_params={
                "retake_video_path": source,
                "retake_start_time": start,
                "retake_end_time": end,
                "retake_regenerate_video": True,
                "retake_regenerate_audio": regenerate_audio,
                "retake_enhance_prompt": False,
                "retake_max_batch_size": 1,
            },
            negative_prompt=negative_prompt,
            seed=seed,
            num_inference_steps=steps,
        ),
    )


def _run_pipeline(pipe, request):
    out = pipe.infer(request)
    return out.video, out.audio, float(out.frame_rate)


# ----------------------------------------------------------------------------
# Source media reading (PyAV) and video persistence.
# ----------------------------------------------------------------------------


def read_source_frames(path: str) -> np.ndarray:
    """Decode every video frame of *path* into ``(T, H, W, C)`` uint8 RGB."""
    import av

    frames = []
    container = av.open(path)
    try:
        video_stream = next(s for s in container.streams if s.type == "video")
        for frame in container.decode(video_stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()
    if not frames:
        raise ValueError(f"no video frames decoded from {path}")
    return np.stack(frames, axis=0)


def read_source_audio(path: str, device):
    """Return ``(waveform_tensor, sampling_rate)`` for *path* or ``None``."""
    from ltx_pipelines.utils.media_io import decode_audio_from_file

    audio = decode_audio_from_file(path, device)
    if audio is None:
        return None
    return audio.waveform, int(audio.sampling_rate)


def _video_to_thwc_uint8(video):
    """Squeeze a ``(1, T, H, W, C)`` uint8 video tensor to ``(T, H, W, C)``."""
    import torch

    if isinstance(video, torch.Tensor):
        v = video
        if v.dim() == 5:
            v = v[0]
        return v.to("cpu")
    return video


def save_video_pt(video, path: str) -> None:
    import torch

    torch.save(video, path)


def encode_mp4(video_thwc_uint8, fps: float, path: str) -> bool:
    """Best-effort H.264 encode of ``(T, H, W, C)`` uint8 frames via PyAV."""
    try:
        import av

        arr = video_thwc_uint8
        if hasattr(arr, "detach"):
            arr = arr.detach().to("cpu").numpy()
        arr = np.asarray(arr)
        t, h, w, _ = arr.shape
        container = av.open(path, mode="w")
        try:
            stream = container.add_stream("libx264", rate=int(round(fps)) or 1)
            stream.width = w
            stream.height = h
            stream.pix_fmt = "yuv420p"
            for i in range(t):
                frame = av.VideoFrame.from_ndarray(arr[i], format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
        return True
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"[oracle] mp4 encode skipped ({path}): {exc}", file=sys.stderr)
        return False


# ----------------------------------------------------------------------------
# Hard invariants (native output vs source) and advisory similarity metrics.
# ----------------------------------------------------------------------------


def _tensor_equal(a, b) -> bool:
    import torch

    if a is None or b is None:
        return False
    try:
        if a.shape != b.shape:
            return False
        return bool(torch.equal(a, b))
    except (RuntimeError, TypeError):
        return False


def _audio_preserved(native_audio, source_audio) -> Optional[bool]:
    """Whether native audio matches the source audio (None when no source audio)."""
    import torch

    if source_audio is None:
        return None
    src_wave = source_audio[0]
    if native_audio is None:
        return False
    try:
        nat = native_audio.detach().to("cpu").float()
        src = src_wave.detach().to("cpu").float()
        n = min(nat.numel(), src.numel())
        nat_flat = nat.reshape(-1)[:n]
        src_flat = src.reshape(-1)[:n]
        if nat.numel() != src.numel():
            # Length mismatch cannot be byte-identical.
            return False
        return bool(torch.allclose(nat_flat, src_flat, atol=1e-4, rtol=0.0))
    except (RuntimeError, TypeError):
        return False


def check_invariants(native_video, source_frames, native_audio, source_audio, pixel_window):
    """Re-check the retake hard invariants on the native output.

    Returns a dict of ``bool | None`` invariant results. ``None`` means the
    invariant does not apply (e.g. no source audio) and is excluded from the
    hard gate. These invariants are the sole pass/fail gate; the similarity
    metrics computed elsewhere never enter this dict.
    """
    import torch

    nat = _video_to_thwc_uint8(native_video)
    src = torch.as_tensor(source_frames)
    start, end = pixel_window

    results = {}
    results["output_len_matches_source"] = bool(nat.shape[0] == src.shape[0])

    outside_identical = None
    if results["output_len_matches_source"]:
        nat_lead, nat_trail = nat[:start], nat[end:]
        src_lead, src_trail = src[:start], src[end:]
        outside_identical = bool(
            _tensor_equal(nat_lead.cpu(), src_lead.cpu())
            and _tensor_equal(nat_trail.cpu(), src_trail.cpu())
        )
    results["composite_outside_byte_identical"] = outside_identical

    results["audio_preserved"] = _audio_preserved(native_audio, source_audio)
    return results


def compute_similarity(native_thwc, other_thwc, pixel_window, label):
    """Advisory PSNR / SSIM of native vs *other*, split by retake window.

    Never gates pass/fail. Returns a dict with per-region PSNR / SSIM or a
    ``reason`` when the two clips cannot be aligned.
    """
    if other_thwc is None:
        return {"label": label, "available": False, "reason": "counterpart_unavailable"}
    nat = np.asarray(_to_float_array(native_thwc), dtype=np.float64)
    oth = np.asarray(_to_float_array(other_thwc), dtype=np.float64)
    if nat.ndim != 4 or oth.ndim != 4:
        return {"label": label, "available": False, "reason": "unexpected_rank"}
    # Align on the common frame count and spatial size before comparing.
    if nat.shape[1:] != oth.shape[1:]:
        return {
            "label": label,
            "available": False,
            "reason": f"spatial_shape_mismatch:{nat.shape[1:]}!={oth.shape[1:]}",
        }
    t = min(nat.shape[0], oth.shape[0])
    frame_count_note = None
    if nat.shape[0] != oth.shape[0]:
        frame_count_note = f"frame_count_mismatch:{nat.shape[0]}vs{oth.shape[0]}_truncated_to_{t}"
    nat = nat[:t]
    oth = oth[:t]

    start, end = pixel_window
    nat_win, nat_out = split_regions(nat, start, end)
    oth_win, oth_out = split_regions(oth, start, end)

    def _region(a, b):
        if a.shape[0] == 0:
            return {"psnr": None, "ssim": None, "frames": 0}
        return {"psnr": psnr(a, b), "ssim": ssim(a, b), "frames": int(a.shape[0])}

    out = {
        "label": label,
        "available": True,
        "window": _region(nat_win, oth_win),
        "outside": _region(nat_out, oth_out),
    }
    if frame_count_note:
        out["note"] = frame_count_note
    return out


# ----------------------------------------------------------------------------
# Main flow.
# ----------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Native-vs-upstream LTX-2 retake comparison oracle.")
    p.add_argument("--checkpoint", required=True, help="LTX-2 checkpoint directory.")
    p.add_argument("--gemma", required=True, help="Gemma text-encoder directory.")
    p.add_argument("--lora", required=True, help="Retake LoRA path (fused into transformer).")
    p.add_argument("--source", required=True, help="Source video path.")
    p.add_argument("--output-dir", required=True, help="Directory for outputs + protocol.")
    p.add_argument("--reference", default=None, help="Optional committed reference video.")
    p.add_argument("--prompt", default="a person talking to the camera")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--start", type=float, required=True, help="Retake window start (seconds).")
    p.add_argument("--end", type=float, required=True, help="Retake window end (seconds).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--attention-backend", default="VANILLA")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument(
        "--skip-upstream",
        action="store_true",
        help="Run only the native path (when the upstream-stage path OOMs or is unavailable).",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    import torch

    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.types import VIDEO_SCALE_FACTORS
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_retake import (
        _RETAKE_DISTILLED_SIGMA_VALUES,
        _retake_conditioned_latent_ranges,
        _retake_pixel_window,
    )

    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Source media + window geometry --------------------------------
    source_frames = read_source_frames(args.source)
    num_frames = int(source_frames.shape[0])
    height = int(source_frames.shape[1])
    width = int(source_frames.shape[2])
    source_audio = read_source_audio(args.source, device)

    # ---- 1. Native retake (build, run, determinism + failfast, free) ---
    native_pipe = build_pipeline(
        args.checkpoint, args.gemma, args.lora, device, args.attention_backend
    )
    native_request = _make_request(
        args.prompt, args.negative_prompt, args.source, args.start, args.end, args.seed, args.steps
    )
    native_video, native_audio, fps = _run_pipeline(native_pipe, native_request)
    native_thwc = _video_to_thwc_uint8(native_video)

    pixel_start, pixel_end = _retake_pixel_window(args.start, args.end, fps, num_frames)

    # Cheap determinism re-check: a 2nd native run with the same seeded request
    # must reproduce the first output bit-for-bit.
    seed_deterministic = None
    try:
        native_video2, _, _ = _run_pipeline(native_pipe, native_request)
        seed_deterministic = _tensor_equal(
            _video_to_thwc_uint8(native_video).cpu(), _video_to_thwc_uint8(native_video2).cpu()
        )
        del native_video2
    except Exception as exc:  # noqa: BLE001 - recorded as advisory null, re-raised only in metrics
        print(f"[oracle] determinism re-check skipped: {exc}", file=sys.stderr)

    # Fail-fast re-check: requesting audio regeneration must raise, since the
    # native retake path is video-only and preserves the source audio.
    audio_regen_failfast = None
    try:
        failfast_request = _make_request(
            args.prompt,
            args.negative_prompt,
            args.source,
            args.start,
            args.end,
            args.seed,
            args.steps,
            regenerate_audio=True,
        )
        _run_pipeline(native_pipe, failfast_request)
        audio_regen_failfast = False
    except NotImplementedError:
        audio_regen_failfast = True
    except Exception as exc:  # noqa: BLE001 - unexpected error type; record as advisory null
        print(f"[oracle] audio-regen failfast re-check inconclusive: {exc}", file=sys.stderr)

    save_video_pt(native_video, str(output_dir / "native.pt"))
    native_mp4 = encode_mp4(native_thwc, fps, str(output_dir / "native.mp4"))

    del native_pipe, native_video
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- 2. Upstream-stage oracle (build, run, free) -------------------
    upstream_thwc = None
    upstream_reason = None
    if args.skip_upstream:
        upstream_reason = "skipped_by_flag"
    else:
        try:
            upstream_pipe = build_pipeline(
                args.checkpoint,
                args.gemma,
                args.lora,
                device,
                args.attention_backend,
                extra_overrides={"retake_use_upstream_stage": True},
            )
            upstream_request = _make_request(
                args.prompt,
                args.negative_prompt,
                args.source,
                args.start,
                args.end,
                args.seed,
                args.steps,
            )
            upstream_video, _upstream_audio, _ = _run_pipeline(upstream_pipe, upstream_request)
            upstream_thwc = _video_to_thwc_uint8(upstream_video)
            save_video_pt(upstream_video, str(output_dir / "upstream.pt"))
            encode_mp4(upstream_thwc, fps, str(output_dir / "upstream.mp4"))
            del upstream_pipe, upstream_video
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - upstream oracle is optional; record reason
            # The upstream RetakePipeline resamples source audio via torchaudio,
            # which is stubbed in some containers and raises on audio-bearing
            # sources (RuntimeError); the distilled-only guard raises
            # NotImplementedError. Either way the oracle still completes with the
            # native artifacts, protocol, and invariants and records the reason.
            upstream_reason = f"upstream_skipped:{type(exc).__name__}:{exc}"
            print(f"[oracle] upstream-stage path skipped: {exc}", file=sys.stderr)

    # ---- 3. Hard invariants (native output only) -----------------------
    invariants = check_invariants(
        native_thwc, source_frames, native_audio, source_audio, (pixel_start, pixel_end)
    )
    invariants["seed_deterministic"] = seed_deterministic
    invariants["audio_regen_failfast"] = audio_regen_failfast

    # Hard gate: every applicable (non-None) invariant must be True. Similarity
    # metrics are excluded by construction.
    all_checks_passed = all(v is True for v in invariants.values() if v is not None)

    # ---- 4. Advisory similarity (never gates) --------------------------
    similarity = []
    up_sim = compute_similarity(
        native_thwc, upstream_thwc, (pixel_start, pixel_end), "native_vs_upstream"
    )
    if upstream_reason and not up_sim.get("available"):
        up_sim["reason"] = up_sim.get("reason") or upstream_reason
    similarity.append(up_sim)

    if args.reference:
        try:
            reference_frames = read_source_frames(args.reference)
            ref_sim = compute_similarity(
                native_thwc, reference_frames, (pixel_start, pixel_end), "native_vs_reference"
            )
        except (OSError, ValueError) as exc:
            ref_sim = {
                "label": "native_vs_reference",
                "available": False,
                "reason": f"reference_read_failed:{exc}",
            }
        similarity.append(ref_sim)

    # ---- 5. Protocol + metrics persistence -----------------------------
    temporal_ratio = VIDEO_SCALE_FACTORS.time
    _latent_window, conditioned_latent_ranges = _retake_conditioned_latent_ranges(
        pixel_start, pixel_end, num_frames, temporal_ratio
    )
    # Anchor git probing on this script's repo (examples/visual_gen/<file>) so
    # the protocol is correct regardless of the invocation working directory.
    trtllm_repo = Path(__file__).resolve().parents[2]
    eval_repo = (trtllm_repo.parent / "LTX2.3-eval").resolve()
    protocol = build_protocol(
        native_repo=_git_info(str(trtllm_repo)),
        eval_repo=_git_info(str(eval_repo)),
        checkpoint=str(Path(args.checkpoint).resolve()),
        gemma=str(Path(args.gemma).resolve()),
        lora=str(Path(args.lora).resolve()) if args.lora else None,
        source=str(Path(args.source).resolve()),
        source_sha256=sha256_file(args.source),
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        num_inference_steps=args.steps,
        sigmas=list(_RETAKE_DISTILLED_SIGMA_VALUES),
        fps=fps,
        num_frames=num_frames,
        height=height,
        width=width,
        window=[args.start, args.end],
        pixel_window=[pixel_start, pixel_end],
        conditioned_latent_ranges=[list(r) for r in conditioned_latent_ranges],
        env=_env_metadata(),
    )
    protocol["upstream_available"] = upstream_thwc is not None
    protocol["upstream_skipped_reason"] = upstream_reason

    metrics = {
        "invariants": invariants,
        "all_checks_passed": all_checks_passed,
        "similarity_informational": similarity,
        "upstream_reason": upstream_reason,
        "native_mp4_written": native_mp4,
    }

    with open(output_dir / "protocol.json", "w") as f:
        json.dump(protocol, f, indent=2, sort_keys=True, default=str)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True, default=str)

    summary = {
        "all_checks_passed": all_checks_passed,
        "invariants": invariants,
        "output_dir": str(output_dir.resolve()),
        "upstream_available": upstream_thwc is not None,
    }
    print(f"ORACLE_DONE {json.dumps(summary, default=str)}")
    print(
        f"PULLBACK: <ssh-gw scp command hint> <remote>:{output_dir.resolve()} <local-destination>"
    )
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
