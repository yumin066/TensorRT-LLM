# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side unit tests for the pure helpers in ``ltx2_retake_oracle``.

Only the pure numpy helpers (``psnr``, ``ssim``, ``split_regions``,
``build_protocol``, ``sha256_file``) are exercised here; they must import
without a GPU or the heavy ``tensorrt_llm`` stack (those imports live inside the
CLI's build/run functions). The module lives under ``examples/`` so it is loaded
by path via ``importlib``.
"""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_ORACLE_PATH = (
    Path(__file__).resolve().parents[4] / "examples" / "visual_gen" / "ltx2_retake_oracle.py"
)


def _load_oracle():
    spec = importlib.util.spec_from_file_location("ltx2_retake_oracle", _ORACLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = _load_oracle()


# --------------------------------------------------------------------------- #
# psnr
# --------------------------------------------------------------------------- #


def test_psnr_identical_is_inf():
    a = np.full((3, 8, 8, 3), 127, dtype=np.uint8)
    assert oracle.psnr(a, a.copy()) == float("inf")


def test_psnr_known_constant_offset():
    # A uniform offset of d gives mse == d**2 and a deterministic PSNR value.
    a = np.zeros((2, 4, 4, 3), dtype=np.float64)
    b = np.full((2, 4, 4, 3), 10.0)
    expected = 10.0 * np.log10((255.0**2) / 100.0)
    assert oracle.psnr(a, b) == pytest.approx(expected)


def test_psnr_shape_mismatch_raises():
    with pytest.raises(ValueError):
        oracle.psnr(np.zeros((2, 4, 4, 3)), np.zeros((2, 4, 5, 3)))


# --------------------------------------------------------------------------- #
# ssim
# --------------------------------------------------------------------------- #


def test_ssim_identical_is_one():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, size=(2, 16, 16, 3), dtype=np.uint8)
    assert oracle.ssim(a, a.copy()) == pytest.approx(1.0, abs=1e-6)


def test_ssim_degrades_with_noise():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, size=(2, 16, 16, 3), dtype=np.uint8).astype(np.float64)
    noisy = np.clip(a + rng.normal(0, 40, size=a.shape), 0, 255)
    assert oracle.ssim(a, noisy) < 0.999


def test_ssim_small_frame_uses_shrunk_window():
    # 5x5 frames are smaller than the 11px window; identical inputs still score 1.
    a = np.full((1, 5, 5, 3), 50, dtype=np.uint8)
    assert oracle.ssim(a, a.copy()) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# split_regions
# --------------------------------------------------------------------------- #


def test_split_regions_basic():
    frames = np.arange(10 * 2 * 2 * 3).reshape(10, 2, 2, 3)
    window, outside = oracle.split_regions(frames, 3, 7)
    assert window.shape[0] == 4
    assert outside.shape[0] == 6
    np.testing.assert_array_equal(window, frames[3:7])
    np.testing.assert_array_equal(outside, np.concatenate([frames[:3], frames[7:]], axis=0))


def test_split_regions_clamps_bounds():
    frames = np.zeros((5, 2, 2, 3))
    window, outside = oracle.split_regions(frames, -3, 99)
    assert window.shape[0] == 5
    assert outside.shape[0] == 0


def test_split_regions_empty_window():
    frames = np.zeros((5, 2, 2, 3))
    window, outside = oracle.split_regions(frames, 2, 2)
    assert window.shape[0] == 0
    assert outside.shape[0] == 5


# --------------------------------------------------------------------------- #
# sha256_file
# --------------------------------------------------------------------------- #


def test_sha256_file(tmp_path):
    payload = b"retake-oracle-bytes"
    f = tmp_path / "blob.bin"
    f.write_bytes(payload)
    assert oracle.sha256_file(str(f)) == hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- #
# build_protocol
# --------------------------------------------------------------------------- #


def _protocol_kwargs():
    return dict(
        native_repo={"path": "/repo", "commit": "abc", "branch": "main"},
        eval_repo={"path": "/eval", "commit": None, "branch": None},
        checkpoint="/ckpt",
        gemma="/gemma",
        lora="/lora",
        source="/src.mp4",
        source_sha256="deadbeef",
        prompt="a person talking to the camera",
        negative_prompt="",
        seed=42,
        dtype="bf16",
        attention_backend="VANILLA",
        num_inference_steps=8,
        sigmas=[1.0, 0.5, 0.0],
        fps=25.0,
        num_frames=100,
        height=512,
        width=768,
        window=[0.5, 1.5],
        pixel_window=[12, 38],
        conditioned_latent_ranges=[[0, 2], [5, 13]],
        env={
            "torch_version": "x",
            "cuda_version": "12",
            "device_name": "H100",
            "platform": "linux",
            "python": "3.12",
        },
    )


def test_build_protocol_fields_and_fixed_flags():
    proto = oracle.build_protocol(**_protocol_kwargs())
    # Retake is a video-only window regeneration that preserves source audio.
    assert proto["regenerate_video"] is True
    assert proto["regenerate_audio"] is False
    assert proto["seed"] == 42
    assert proto["pixel_window"] == [12, 38]
    assert proto["conditioned_latent_ranges"] == [[0, 2], [5, 13]]
    assert proto["source_sha256"] == "deadbeef"
    assert proto["sigmas"] == [1.0, 0.5, 0.0]
    assert proto["env"]["device_name"] == "H100"


def test_build_protocol_is_pure_copy_of_sigmas():
    kwargs = _protocol_kwargs()
    proto = oracle.build_protocol(**kwargs)
    proto["sigmas"].append(999.0)
    # Mutating the returned list must not corrupt the caller's input.
    assert kwargs["sigmas"] == [1.0, 0.5, 0.0]
