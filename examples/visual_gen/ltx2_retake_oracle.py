#! /usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native-vs-upstream LTX-2 retake comparison oracle.

Runs the native LTX-2 retake video regeneration and the preserved upstream
``DiffusionStage.run`` oracle on the same source / prompt / window / seed, then:

- persists both output videos plus a reproducibility protocol,
- re-checks the retake hard invariants (output length, output shape, output
  fps, byte-identical frames outside the retake window, source-audio
  preservation, determinism, and the audio-regeneration fail-fast) on the
  native output, and
- computes advisory PSNR / SSIM between the native and upstream outputs (and an
  optional committed reference), split into the retake window and the region
  outside it.

The PSNR / SSIM columns are informational only: the exit code and the
``all_checks_passed`` summary reflect the hard invariants exclusively and never
depend on any image-similarity metric.

The native-vs-upstream comparison is the point of this oracle, so the upstream
run is REQUIRED by default: if it is skipped, raises, or produces no output the
gate fails and the process exits nonzero. Pass ``--native-only`` for a
diagnostic run that exercises the native hard invariants alone (upstream not
run); that mode reports ``task6_satisfied = False`` and its gate reflects the
native invariants only.

Source-media guidance: the native-vs-upstream comparison must use a VIDEO-ONLY
source (a source with no audio stream). The upstream oracle path resamples
source audio through torchaudio, which is stubbed in some runtime containers and
raises when handed an audio-bearing source; the native path reads media through
PyAV and is unaffected. In the default (native-vs-upstream) mode an audio-
bearing source fails fast with a message directing the operator to
``--native-only``. Run the audio-preservation invariant on an audio-bearing
source via ``--native-only``.

The heavy ``tensorrt_llm`` / Lightricks imports live inside the build and run
functions so the pure helpers at the top of this module (``psnr``, ``ssim``,
``split_regions``, ``build_protocol``, ``sha256_file``, ``assemble_gate``,
``build_manifest_entries``) import on a plain CPU host without a GPU.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
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
    scheduler: str = "",
    source_fps: Optional[float] = None,
    code_commit: Optional[str] = None,
) -> dict:
    """Assemble the reproducibility protocol from already-computed primitives.

    Pure: every argument is a plain scalar / list / dict so this is unit
    testable without a GPU. ``regenerate_video`` is always True and
    ``regenerate_audio`` always False because the native retake path is a
    video-only window regeneration that preserves the source audio.

    ``code_commit`` is the authoritative source-of-truth for the running code
    (the operator-supplied reviewed local commit), distinct from the possibly
    stale git HEAD ``commit`` recorded inside ``native_repo`` / ``eval_repo``.
    ``scheduler`` is a stable identifier for the native denoise schedule and
    ``source_fps`` is the source frame rate read via PyAV.
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
        "scheduler": scheduler,
        "num_inference_steps": num_inference_steps,
        "sigmas": list(sigmas),
        "fps": fps,
        "source_fps": source_fps,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "window": window,
        "pixel_window": pixel_window,
        "conditioned_latent_ranges": conditioned_latent_ranges,
        "regenerate_video": True,
        "regenerate_audio": False,
        "code_commit": code_commit,
        "env": env,
    }


# Only these invariants may legitimately be ``None`` (Not-Applicable). Every
# other hard invariant must be exactly ``True``; an unexpected ``None`` (e.g. a
# helper that silently returned nothing) fails the native gate rather than being
# ignored.
_NONE_OK_INVARIANTS = frozenset({"audio_preserved"})


def native_invariants_ok(invariants: dict, invariant_errors: dict) -> bool:
    """Strict per-key native-invariant check (pure).

    ``invariant_errors`` must be empty, and every invariant must be exactly
    ``True`` — except the keys in ``_NONE_OK_INVARIANTS``, which may be ``None``
    (N/A). A ``None`` for any other key, or any non-``True`` value, fails.
    """
    if invariant_errors:
        return False
    for key, value in invariants.items():
        if value is None:
            if key not in _NONE_OK_INVARIANTS:
                return False
        elif value is not True:
            return False
    return True


def assemble_gate(
    invariants: dict,
    invariant_errors: dict,
    upstream_available: bool,
    native_only: bool,
    provenance_ok: bool = True,
) -> tuple:
    """Compute ``(all_checks_passed, task6_satisfied)`` from the invariant state.

    Pure and side-effect free so it is unit testable without a GPU.

    - Native hard invariants pass only when :func:`native_invariants_ok` holds
      (strict per-key: only ``audio_preserved`` may be ``None``).
    - Default (native-vs-upstream) mode: ``all_checks_passed`` / ``task6_satisfied``
      require the upstream oracle to have produced output, the native invariants
      to pass, AND authoritative provenance (``provenance_ok``).
    - ``--native-only`` diagnostic mode: ``task6_satisfied`` is always ``False``
      (this is not the native-vs-upstream oracle) and ``all_checks_passed``
      reflects the native hard invariants alone (provenance not required).
    """
    native_ok = native_invariants_ok(invariants, invariant_errors)
    if native_only:
        return native_ok, False
    if not upstream_available:
        return False, False
    passed = native_ok and provenance_ok
    return passed, passed


def validate_provenance(protocol: dict, native_only: bool) -> tuple:
    """Whether the protocol carries authoritative, reproducible provenance (pure).

    In default (native-vs-upstream) mode a satisfied oracle result must be
    reproducible, so it requires an authoritative ``code_commit`` and non-null
    ``commit``/``dirty`` for both repos. In ``--native-only`` diagnostic mode
    provenance is not required. Returns ``(ok, reasons)``.
    """
    if native_only:
        return True, []
    reasons = []
    if not protocol.get("code_commit"):
        reasons.append("missing code_commit (pass --code-commit or set LTX2_ORACLE_CODE_COMMIT)")
    for repo in ("native_repo", "eval_repo"):
        info = protocol.get(repo) or {}
        if not info.get("commit"):
            reasons.append(f"{repo}.commit is null")
        if info.get("dirty") is None:
            reasons.append(f"{repo}.dirty is null (git status unavailable)")
    return (not reasons), reasons


# ----------------------------------------------------------------------------
# Local artifact manifest (filesystem, stdlib only).
# ----------------------------------------------------------------------------

MANIFEST_ARTIFACTS = (
    "protocol.json",
    "metrics.json",
    "native.mp4",
    "upstream.mp4",
    "native.pt",
    "upstream.pt",
)


def build_manifest_entries(paths_by_name: dict) -> dict:
    """Map ``name -> {path, exists, size_bytes, sha256}`` for each artifact.

    Pure aside from reading the referenced files: missing files yield
    ``exists=False`` with ``None`` size / digest, present files carry their byte
    size and streaming SHA-256 so the copied package is self-describing.
    """
    entries = {}
    for name, path in paths_by_name.items():
        p = Path(path)
        if p.exists() and p.is_file():
            entries[name] = {
                "path": str(p),
                "exists": True,
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(str(p)),
            }
        else:
            entries[name] = {
                "path": str(p),
                "exists": False,
                "size_bytes": None,
                "sha256": None,
            }
    return entries


def build_manifest(
    output_dir: str,
    remote_output_dir: str,
    code_commit: Optional[str],
    artifact_names=MANIFEST_ARTIFACTS,
) -> dict:
    """Assemble a self-describing manifest for the artifacts in *output_dir*."""
    out = Path(output_dir)
    paths_by_name = {name: out / name for name in artifact_names}
    return {
        "remote_output_dir": remote_output_dir,
        "code_commit": code_commit,
        "artifacts": build_manifest_entries(paths_by_name),
    }


def _json_parses(path: str) -> bool:
    """Whether the file at *path* exists and parses as JSON."""
    try:
        with open(path) as f:
            json.load(f)
        return True
    except (OSError, ValueError):
        return False


# Artifacts that must be present in a pulled-back local package vs optional heavy
# tensors that may be intentionally trimmed locally (their digest stays in the
# manifest for cluster-side verification).
_REQUIRED_LOCAL_ARTIFACTS = ("protocol.json", "metrics.json", "native.mp4", "upstream.mp4")
_OPTIONAL_LOCAL_ARTIFACTS = ("native.pt", "upstream.pt")


def verify_local(local_dir: str) -> dict:
    """Validate a pulled-back local artifact package for self-consistency.

    Checks that ``protocol.json`` / ``metrics.json`` / ``manifest.json`` parse as
    JSON; that every required artifact is present locally; and that each
    locally-present artifact's SHA-256 matches the generation manifest. Heavy
    ``*.pt`` tensors may be intentionally absent locally (trimmed) — their
    absence is recorded, not failed, but a present tensor whose digest mismatches
    fails. Returns a report dict whose ``ok`` field is the overall pass/fail.
    """
    d = Path(local_dir)
    problems = []
    json_valid = {
        name: _json_parses(str(d / name))
        for name in ("protocol.json", "metrics.json", "manifest.json")
    }
    for name, ok in json_valid.items():
        if not ok:
            problems.append(f"{name} missing or not valid JSON")

    manifest_artifacts = {}
    if json_valid["manifest.json"]:
        with open(d / "manifest.json") as f:
            manifest_artifacts = json.load(f).get("artifacts") or {}

    artifacts = {}
    for name in MANIFEST_ARTIFACTS:
        p = d / name
        present = p.exists() and p.is_file()
        entry = {
            "present_local": present,
            "size_bytes": p.stat().st_size if present else None,
            "sha256": sha256_file(str(p)) if present else None,
            "sha256_matches_manifest": None,
        }
        if present:
            man_sha = (manifest_artifacts.get(name) or {}).get("sha256")
            if man_sha is not None:
                entry["sha256_matches_manifest"] = entry["sha256"] == man_sha
                if not entry["sha256_matches_manifest"]:
                    problems.append(f"{name} sha256 does not match the manifest")
        elif name in _REQUIRED_LOCAL_ARTIFACTS:
            problems.append(f"required artifact {name} is missing locally")
        artifacts[name] = entry

    return {
        "local_dir": str(d),
        "json_valid": json_valid,
        "artifacts": artifacts,
        "problems": problems,
        "ok": not problems,
    }


# ----------------------------------------------------------------------------
# Repo / environment probing (subprocess, stdlib only).
# ----------------------------------------------------------------------------


def _git_info(repo_dir: str) -> dict:
    """Best-effort ``{commit, branch, dirty}`` for a git repo; nulls on failure.

    ``dirty`` is ``True`` when ``git status --porcelain`` reports any uncommitted
    change in the working tree, so a synced-but-uncommitted tree is never hidden.
    """
    info = {"path": repo_dir, "commit": None, "branch": None, "dirty": None}
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
        status = subprocess.check_output(
            ["git", "-C", repo_dir, "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        info["dirty"] = bool(status.strip())
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


def read_source_fps(path: str) -> Optional[float]:
    """Read the source video frame rate via PyAV, or ``None`` if unavailable.

    Uses the stream's ``average_rate`` (a ``Fraction`` such as ``30000/1001``)
    so fractional rates are preserved before conversion to float.
    """
    import av

    container = av.open(path)
    try:
        video_stream = next(s for s in container.streams if s.type == "video")
        rate = video_stream.average_rate
        if rate is None:
            rate = video_stream.base_rate
        return float(rate) if rate is not None else None
    finally:
        container.close()


def has_audio_stream(path: str) -> bool:
    """Whether *path* carries at least one audio stream (PyAV probe)."""
    import av

    container = av.open(path)
    try:
        return any(s.type == "audio" for s in container.streams)
    finally:
        container.close()


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


def _safe_invariant(name: str, fn, invariant_errors: dict):
    """Evaluate a hard invariant, recording exceptions as a strict ``False``.

    A broad ``except Exception`` is intended here: any failure to evaluate a
    hard invariant must fail the gate (never become an ignorable ``None``), and
    the exception type + message is recorded in ``invariant_errors[name]`` for
    triage. The callable's own return value (including a legitimate ``None`` for
    an N/A invariant) is passed through unchanged when no exception is raised.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - any failure fails the gate; see docstring
        invariant_errors[name] = f"{type(exc).__name__}: {exc}"
        return False


def check_invariants(
    native_video,
    source_frames,
    native_audio,
    source_audio,
    pixel_window,
    source_fps,
    native_frame_rate,
    invariant_errors: dict,
):
    """Re-check the retake hard invariants on the native output.

    Returns a dict of ``bool | None`` invariant results and mutates
    ``invariant_errors`` (``name -> "ExceptionType: message"``) for any
    invariant that raised. Only ``audio_preserved`` may be ``None`` (no source
    audio, N/A) and is excluded from the hard gate; every other invariant is a
    strict bool. These invariants are the sole pass/fail gate; the similarity
    metrics computed elsewhere never enter this dict.
    """
    import torch

    nat = _video_to_thwc_uint8(native_video)
    src = torch.as_tensor(source_frames)
    start, end = pixel_window

    results = {}

    def _len():
        return bool(nat.shape[0] == src.shape[0])

    results["output_len_matches_source"] = _safe_invariant(
        "output_len_matches_source", _len, invariant_errors
    )

    def _shape():
        # Full (T, H, W, C) equality, not just the frame count.
        return bool(tuple(nat.shape) == tuple(src.shape))

    results["output_shape_matches"] = _safe_invariant(
        "output_shape_matches", _shape, invariant_errors
    )

    def _fps():
        if source_fps is None:
            raise ValueError("source_fps unavailable")
        if native_frame_rate is None:
            raise ValueError("native frame_rate unavailable")
        # fps can be a fraction (e.g. 30000/1001), so compare with a small
        # relative tolerance rather than exact equality.
        return bool(np.allclose(float(source_fps), float(native_frame_rate), rtol=1e-3, atol=0.0))

    results["output_fps_matches"] = _safe_invariant("output_fps_matches", _fps, invariant_errors)

    def _outside():
        # _tensor_equal returns False on shape mismatch, so a length or spatial
        # mismatch already fails this without a separate length guard.
        nat_lead, nat_trail = nat[:start], nat[end:]
        src_lead, src_trail = src[:start], src[end:]
        return bool(
            _tensor_equal(nat_lead.cpu(), src_lead.cpu())
            and _tensor_equal(nat_trail.cpu(), src_trail.cpu())
        )

    results["composite_outside_byte_identical"] = _safe_invariant(
        "composite_outside_byte_identical", _outside, invariant_errors
    )

    def _audio():
        # Passes through a legitimate None (no source audio -> N/A).
        return _audio_preserved(native_audio, source_audio)

    results["audio_preserved"] = _safe_invariant("audio_preserved", _audio, invariant_errors)
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
    # Pipeline args are required for a run, but not for ``--verify-local`` (which
    # only validates an already-produced local artifact package); enforced below.
    p.add_argument("--checkpoint", help="LTX-2 checkpoint directory.")
    p.add_argument("--gemma", help="Gemma text-encoder directory.")
    p.add_argument("--lora", help="Retake LoRA path (fused into transformer).")
    p.add_argument("--source", help="Source video path.")
    p.add_argument("--output-dir", help="Directory for outputs + protocol.")
    p.add_argument("--reference", default=None, help="Optional committed reference video.")
    p.add_argument("--prompt", default="a person talking to the camera")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--start", type=float, help="Retake window start (seconds).")
    p.add_argument("--end", type=float, help="Retake window end (seconds).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--eval-repo",
        default=None,
        help=(
            "Path to the sibling LTX2.3-eval repo (upstream oracle source). "
            "Defaults to <TensorRT-LLM>/../LTX2.3-eval. In default mode its git "
            "commit/dirty must be probeable (provenance requirement)."
        ),
    )
    p.add_argument(
        "--verify-local",
        default=None,
        metavar="DIR",
        help=(
            "Verify an already-pulled local artifact package (JSON validity + "
            "presence + SHA-256 vs manifest) and exit; does not run the pipeline."
        ),
    )
    p.add_argument(
        "--dtype",
        default="bf16",
        help="Native path dtype. Only 'bf16' is supported today; anything else errors.",
    )
    p.add_argument("--attention-backend", default="VANILLA")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument(
        "--native-only",
        action="store_true",
        help=(
            "Diagnostic run of the native path alone (upstream not run). "
            "task6_satisfied is reported False and the gate reflects only the "
            "native hard invariants. Without this flag the upstream oracle is "
            "REQUIRED and its absence fails the gate."
        ),
    )
    p.add_argument(
        "--code-commit",
        default=None,
        help=(
            "Authoritative commit for the running code (the reviewed local "
            "commit). Set this on clusters whose git HEAD is a stale base "
            "commit while the working tree is synced from the reviewed commit. "
            "Falls back to the LTX2_ORACLE_CODE_COMMIT environment variable."
        ),
    )
    args = p.parse_args(argv)
    if args.verify_local:
        return args
    missing = [
        name
        for name in ("checkpoint", "gemma", "lora", "source", "output_dir", "start", "end")
        if getattr(args, name) is None
    ]
    if missing:
        p.error(
            "the following arguments are required for a run: "
            + ", ".join("--" + m.replace("_", "-") for m in missing)
        )
    if args.dtype != "bf16":
        p.error(
            f"--dtype only supports 'bf16' on the native path; got {args.dtype!r}. "
            "Re-run with --dtype bf16."
        )
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    # ``--verify-local`` validates an already-pulled local artifact package and
    # exits without importing torch / tensorrt_llm or touching a GPU.
    if args.verify_local:
        report = verify_local(args.verify_local)
        report_path = Path(args.verify_local) / "local_verify.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print("LOCAL_VERIFY", json.dumps({"ok": report["ok"], "problems": report["problems"]}))
        return 0 if report["ok"] else 1

    import torch

    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.types import VIDEO_SCALE_FACTORS
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_retake import (
        _RETAKE_DISTILLED_SIGMA_VALUES,
        _retake_conditioned_latent_ranges,
        _retake_pixel_window,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    code_commit = args.code_commit or os.environ.get("LTX2_ORACLE_CODE_COMMIT")
    invariant_errors: dict = {}

    # ---- Source media + window geometry --------------------------------
    source_frames = read_source_frames(args.source)
    num_frames = int(source_frames.shape[0])
    height = int(source_frames.shape[1])
    width = int(source_frames.shape[2])
    source_audio = read_source_audio(args.source, device)
    source_fps = None
    try:
        source_fps = read_source_fps(args.source)
    except (OSError, ValueError, StopIteration) as exc:
        print(f"[oracle] source fps read failed: {exc}", file=sys.stderr)

    # Audio-bearing sources cannot run through the upstream oracle in this
    # container (upstream resamples audio via the stubbed torchaudio), so the
    # default native-vs-upstream mode fails fast before building any pipeline.
    source_has_audio = source_audio is not None
    if not source_has_audio:
        try:
            source_has_audio = has_audio_stream(args.source)
        except (OSError, ValueError, StopIteration) as exc:
            print(f"[oracle] audio stream probe failed: {exc}", file=sys.stderr)
    if source_has_audio and not args.native_only:
        print(
            "[oracle] audio-bearing source detected: the upstream oracle path cannot "
            "decode audio-bearing sources in this container (upstream resamples audio "
            "via the stubbed torchaudio). Re-run with --native-only for the "
            "audio-preservation check.",
            file=sys.stderr,
        )
        return 2

    # ---- 1. Native retake (build, run, determinism + failfast, free) ---
    native_pipe = build_pipeline(
        args.checkpoint, args.gemma, args.lora, device, args.attention_backend
    )
    scheduler = f"NativeSchedulerAdapter/distilled-euler-{args.steps}"
    native_request = _make_request(
        args.prompt, args.negative_prompt, args.source, args.start, args.end, args.seed, args.steps
    )
    native_video, native_audio, fps = _run_pipeline(native_pipe, native_request)
    native_thwc = _video_to_thwc_uint8(native_video)

    pixel_start, pixel_end = _retake_pixel_window(args.start, args.end, fps, num_frames)

    # Cheap determinism re-check: a 2nd native run with the same seeded request
    # must reproduce the first output bit-for-bit. Any failure to evaluate this
    # hard invariant fails the gate and is recorded in invariant_errors.
    seed_deterministic = None
    try:
        native_video2, _, _ = _run_pipeline(native_pipe, native_request)
        seed_deterministic = _tensor_equal(
            _video_to_thwc_uint8(native_video).cpu(), _video_to_thwc_uint8(native_video2).cpu()
        )
        del native_video2
    except Exception as exc:  # noqa: BLE001 - any failure fails this hard invariant
        seed_deterministic = False
        invariant_errors["seed_deterministic"] = repr(exc)
        print(f"[oracle] determinism re-check failed: {exc}", file=sys.stderr)

    # Fail-fast re-check: requesting audio regeneration must raise
    # NotImplementedError, since the native retake path is video-only and
    # preserves the source audio. Any other outcome fails the gate.
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
    except Exception as exc:  # noqa: BLE001 - unexpected error type; fail the hard invariant
        audio_regen_failfast = False
        invariant_errors["audio_regen_failfast"] = repr(exc)
        print(f"[oracle] audio-regen failfast re-check errored: {exc}", file=sys.stderr)

    save_video_pt(native_video, str(output_dir / "native.pt"))
    native_mp4 = encode_mp4(native_thwc, fps, str(output_dir / "native.mp4"))

    del native_pipe, native_video
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- 2. Upstream-stage oracle (build, run, free) -------------------
    # Required by default: the native-vs-upstream comparison is the point of the
    # oracle. Only --native-only skips it, as an explicit diagnostic run.
    upstream_thwc = None
    upstream_reason = None
    if args.native_only:
        upstream_reason = "native_only_diagnostic_run"
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
        native_thwc,
        source_frames,
        native_audio,
        source_audio,
        (pixel_start, pixel_end),
        source_fps,
        fps,
        invariant_errors,
    )
    invariants["seed_deterministic"] = seed_deterministic
    invariants["audio_regen_failfast"] = audio_regen_failfast

    # ---- Provenance (required in default native-vs-upstream mode) ------
    # Anchor git probing on this script's repo (examples/visual_gen/<file>) so
    # the protocol is correct regardless of the invocation working directory.
    trtllm_repo = Path(__file__).resolve().parents[2]
    eval_repo_dir = (
        Path(args.eval_repo).resolve()
        if args.eval_repo
        else (trtllm_repo.parent / "LTX2.3-eval").resolve()
    )
    native_repo_info = _git_info(str(trtllm_repo))
    eval_repo_info = _git_info(str(eval_repo_dir))
    provenance_ok, provenance_reasons = validate_provenance(
        {"code_commit": code_commit, "native_repo": native_repo_info, "eval_repo": eval_repo_info},
        args.native_only,
    )

    # Hard gate: strict per-key native invariants (only audio_preserved may be
    # None) + no invariant_errors; in default mode the upstream oracle must also
    # have produced output AND authoritative provenance must be present.
    # Similarity metrics are excluded by construction.
    upstream_available = upstream_thwc is not None
    all_checks_passed, task6_satisfied = assemble_gate(
        invariants, invariant_errors, upstream_available, args.native_only, provenance_ok
    )

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
    protocol = build_protocol(
        native_repo=native_repo_info,
        eval_repo=eval_repo_info,
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
        scheduler=scheduler,
        source_fps=source_fps,
        code_commit=code_commit,
    )
    protocol["upstream_available"] = upstream_available
    protocol["upstream_skipped_reason"] = upstream_reason

    metrics = {
        "invariants": invariants,
        "invariant_errors": invariant_errors,
        "all_checks_passed": all_checks_passed,
        "task6_satisfied": task6_satisfied,
        "native_only": args.native_only,
        "provenance_ok": provenance_ok,
        "provenance_reasons": provenance_reasons,
        "similarity_informational": similarity,
        "upstream_reason": upstream_reason,
        "native_mp4_written": native_mp4,
    }
    if args.native_only:
        mode_note = (
            "diagnostic native-only run: this is NOT the native-vs-upstream oracle. "
            "task6_satisfied is False and the gate reflects the native hard invariants only."
        )
    elif not upstream_available:
        mode_note = (
            f"native-vs-upstream oracle: upstream run was required but unavailable "
            f"({upstream_reason}); gate failed."
        )
    elif not provenance_ok:
        mode_note = (
            "native-vs-upstream oracle: upstream produced output but provenance is "
            f"incomplete ({'; '.join(provenance_reasons)}); gate failed."
        )
    else:
        mode_note = (
            "native-vs-upstream oracle: upstream produced output with authoritative provenance."
        )
    metrics["mode_note"] = mode_note

    with open(output_dir / "protocol.json", "w") as f:
        json.dump(protocol, f, indent=2, sort_keys=True, default=str)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True, default=str)

    # ---- 6. Self-describing local artifact manifest --------------------
    # Validate that the just-written JSON artifacts parse before advertising
    # them in the manifest, so the copied package is self-consistent.
    protocol_valid = _json_parses(str(output_dir / "protocol.json"))
    metrics_valid = _json_parses(str(output_dir / "metrics.json"))
    manifest = build_manifest(str(output_dir), str(output_dir.resolve()), code_commit)
    manifest["protocol_json_valid"] = protocol_valid
    manifest["metrics_json_valid"] = metrics_valid
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)

    print(f"[oracle] {mode_note}")
    summary = {
        "all_checks_passed": all_checks_passed,
        "task6_satisfied": task6_satisfied,
        "native_only": args.native_only,
        "invariants": invariants,
        "invariant_errors": invariant_errors,
        "output_dir": str(output_dir.resolve()),
        "upstream_available": upstream_available,
    }
    print(f"ORACLE_DONE {json.dumps(summary, default=str)}")
    print(
        f"PULLBACK: <ssh-gw scp command hint> <remote>:{output_dir.resolve()} <local-destination>"
    )
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
