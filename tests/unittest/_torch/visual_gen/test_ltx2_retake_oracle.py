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


def test_build_protocol_records_new_fields():
    proto = oracle.build_protocol(
        **_protocol_kwargs(),
        scheduler="NativeSchedulerAdapter/distilled-euler-8",
        source_fps=29.97,
        code_commit="cafef00d",
    )
    assert proto["scheduler"] == "NativeSchedulerAdapter/distilled-euler-8"
    assert proto["source_fps"] == 29.97
    assert proto["code_commit"] == "cafef00d"


# --------------------------------------------------------------------------- #
# assemble_gate
# --------------------------------------------------------------------------- #


def _passing_invariants():
    # A no-audio source: audio_preserved is legitimately None (N/A) and excluded.
    return {
        "output_len_matches_source": True,
        "output_shape_matches": True,
        "output_fps_matches": True,
        "composite_outside_byte_identical": True,
        "audio_preserved": None,
        "seed_deterministic": True,
        "audio_regen_failfast": True,
    }


def test_gate_upstream_unavailable_not_native_only_fails():
    all_passed, task6 = oracle.assemble_gate(
        _passing_invariants(), {}, upstream_available=False, native_only=False
    )
    assert all_passed is False
    assert task6 is False


def test_gate_upstream_unavailable_native_only_reflects_native():
    # Diagnostic mode: task6 always False; gate reflects native invariants only.
    all_passed, task6 = oracle.assemble_gate(
        _passing_invariants(), {}, upstream_available=False, native_only=True
    )
    assert all_passed is True
    assert task6 is False


def test_gate_upstream_available_and_native_pass_satisfies_task6():
    all_passed, task6 = oracle.assemble_gate(
        _passing_invariants(), {}, upstream_available=True, native_only=False
    )
    assert all_passed is True
    assert task6 is True


def test_gate_invariant_error_fails_even_with_upstream():
    all_passed, task6 = oracle.assemble_gate(
        _passing_invariants(),
        {"seed_deterministic": "RuntimeError: boom"},
        upstream_available=True,
        native_only=False,
    )
    assert all_passed is False
    assert task6 is False


def test_gate_false_invariant_fails():
    invs = _passing_invariants()
    invs["output_fps_matches"] = False
    all_passed, task6 = oracle.assemble_gate(invs, {}, upstream_available=True, native_only=False)
    assert all_passed is False
    assert task6 is False


def test_gate_shape_mismatch_fails():
    invs = _passing_invariants()
    invs["output_shape_matches"] = False
    all_passed, _ = oracle.assemble_gate(invs, {}, upstream_available=True, native_only=False)
    assert all_passed is False


def test_gate_none_audio_excluded_when_others_pass():
    # audio_preserved=None with a no-audio source does not fail the gate.
    invs = _passing_invariants()
    assert invs["audio_preserved"] is None
    all_passed, task6 = oracle.assemble_gate(invs, {}, upstream_available=True, native_only=False)
    assert all_passed is True
    assert task6 is True


# --------------------------------------------------------------------------- #
# manifest assembly
# --------------------------------------------------------------------------- #


def test_build_manifest_entries_size_and_sha256(tmp_path):
    present = tmp_path / "protocol.json"
    payload = b'{"ok": true}'
    present.write_bytes(payload)
    missing = tmp_path / "upstream.mp4"

    entries = oracle.build_manifest_entries({"protocol.json": present, "upstream.mp4": missing})

    p = entries["protocol.json"]
    assert p["exists"] is True
    assert p["size_bytes"] == len(payload)
    assert p["sha256"] == hashlib.sha256(payload).hexdigest()

    m = entries["upstream.mp4"]
    assert m["exists"] is False
    assert m["size_bytes"] is None
    assert m["sha256"] is None


def test_build_manifest_top_level_fields(tmp_path):
    (tmp_path / "metrics.json").write_bytes(b"{}")
    manifest = oracle.build_manifest(
        str(tmp_path), remote_output_dir="/remote/out", code_commit="abc123"
    )
    assert manifest["remote_output_dir"] == "/remote/out"
    assert manifest["code_commit"] == "abc123"
    # Every known artifact name is represented, present or not.
    assert set(manifest["artifacts"]) == set(oracle.MANIFEST_ARTIFACTS)
    assert manifest["artifacts"]["metrics.json"]["exists"] is True


# --------------------------------------------------------------------------- #
# native_invariants_ok / strict-None gate (Round 23)
# --------------------------------------------------------------------------- #


def test_native_invariants_ok_only_audio_may_be_none():
    assert oracle.native_invariants_ok(
        {"output_len_matches_source": True, "audio_preserved": None, "seed_deterministic": True}, {}
    )


def test_native_invariants_ok_non_audio_none_fails():
    assert not oracle.native_invariants_ok({"seed_deterministic": None}, {})
    assert not oracle.native_invariants_ok({"output_fps_matches": None}, {})


def test_native_invariants_ok_invariant_errors_fail():
    assert not oracle.native_invariants_ok({"x": True}, {"x": "Boom: kaput"})


def test_gate_non_audio_none_fails_default_mode():
    all_passed, task6 = oracle.assemble_gate(
        {"seed_deterministic": None},
        {},
        upstream_available=True,
        native_only=False,
        provenance_ok=True,
    )
    assert all_passed is False and task6 is False


def test_gate_default_mode_requires_provenance():
    inv = {"output_len_matches_source": True, "audio_preserved": None}
    ok_all, ok_t6 = oracle.assemble_gate(inv, {}, True, False, provenance_ok=True)
    assert ok_all is True and ok_t6 is True
    bad_all, bad_t6 = oracle.assemble_gate(inv, {}, True, False, provenance_ok=False)
    assert bad_all is False and bad_t6 is False


def test_gate_native_only_ignores_provenance():
    inv = {"output_len_matches_source": True, "audio_preserved": None}
    ok_all, ok_t6 = oracle.assemble_gate(inv, {}, False, native_only=True, provenance_ok=False)
    assert ok_all is True and ok_t6 is False


# --------------------------------------------------------------------------- #
# validate_provenance (Round 23)
# --------------------------------------------------------------------------- #


def test_validate_provenance_full_is_ok():
    proto = {
        "code_commit": "abc",
        "native_repo": {"commit": "x", "dirty": False},
        "eval_repo": {"commit": "y", "dirty": True},
    }
    ok, reasons = oracle.validate_provenance(proto, native_only=False)
    assert ok and reasons == []


def test_validate_provenance_missing_code_commit_fails():
    proto = {
        "code_commit": None,
        "native_repo": {"commit": "x", "dirty": False},
        "eval_repo": {"commit": "y", "dirty": False},
    }
    ok, reasons = oracle.validate_provenance(proto, native_only=False)
    assert not ok and any("code_commit" in r for r in reasons)


def test_validate_provenance_null_repo_metadata_fails():
    proto = {"code_commit": "abc", "native_repo": {"commit": None, "dirty": None}, "eval_repo": {}}
    ok, reasons = oracle.validate_provenance(proto, native_only=False)
    assert not ok and len(reasons) >= 2


def test_validate_provenance_native_only_not_required():
    ok, reasons = oracle.validate_provenance({"code_commit": None}, native_only=True)
    assert ok and reasons == []


# --------------------------------------------------------------------------- #
# verify_local (Round 23)
# --------------------------------------------------------------------------- #


def _make_local_package(tmp_path):
    (tmp_path / "native.mp4").write_bytes(b"vid1")
    (tmp_path / "upstream.mp4").write_bytes(b"vid2")
    (tmp_path / "protocol.json").write_text('{"a": 1}')
    (tmp_path / "metrics.json").write_text('{"b": 2}')
    manifest = oracle.build_manifest(str(tmp_path), "remote:/x", "commitZ")
    import json

    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_verify_local_ok_with_trimmed_tensors(tmp_path):
    _make_local_package(tmp_path)
    report = oracle.verify_local(str(tmp_path))
    assert report["ok"] is True, report["problems"]
    assert report["artifacts"]["native.mp4"]["present_local"] is True
    # Heavy tensors are absent locally (optional) and must not fail the package.
    assert report["artifacts"]["native.pt"]["present_local"] is False


def test_verify_local_sha256_mismatch_fails(tmp_path):
    _make_local_package(tmp_path)
    (tmp_path / "native.mp4").write_bytes(b"TAMPERED")
    report = oracle.verify_local(str(tmp_path))
    assert report["ok"] is False
    assert any("native.mp4 sha256" in p for p in report["problems"])


def test_verify_local_missing_required_fails(tmp_path):
    _make_local_package(tmp_path)
    (tmp_path / "upstream.mp4").unlink()
    report = oracle.verify_local(str(tmp_path))
    assert report["ok"] is False
    assert any("upstream.mp4 is missing" in p for p in report["problems"])


def test_verify_local_invalid_json_fails(tmp_path):
    _make_local_package(tmp_path)
    (tmp_path / "protocol.json").write_text("{not valid json")
    report = oracle.verify_local(str(tmp_path))
    assert report["ok"] is False
    assert any("protocol.json" in p for p in report["problems"])


# --------------------------------------------------------------------------- #
# _audio_preserved sample-rate comparison (Round 24)
# --------------------------------------------------------------------------- #


def test_audio_preserved_none_when_no_source_audio():
    # No source audio is genuinely N/A (excluded from the hard gate as None).
    assert oracle._audio_preserved(object(), None, 44100) is None


def test_audio_preserved_missing_native_sample_rate_fails():
    # Audio-bearing source but the native output reported no sample rate.
    src = (object(), 44100)
    assert oracle._audio_preserved(object(), src, None) is False


def test_audio_preserved_sample_rate_mismatch_fails():
    # Even a sample-for-sample identical waveform at a different sampling rate is
    # a different signal; the mismatch fails before any waveform comparison
    # (so this path needs neither torch nor a real waveform).
    src = (object(), 44100)
    assert oracle._audio_preserved(object(), src, 48000) is False


def test_audio_preserved_missing_native_audio_fails():
    src = (object(), 44100)
    assert oracle._audio_preserved(None, src, 44100) is False


def _require_real_torch():
    # Some host envs ship a stub ``torch`` (importable but missing ops), so
    # ``importorskip`` alone is not enough to gate the waveform-comparison path.
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "zeros") or not hasattr(torch, "allclose"):
        pytest.skip("host environment has a stub torch; real torch required")
    return torch


def test_audio_preserved_matching_rate_and_waveform_true():
    torch = _require_real_torch()
    wave = torch.zeros(2, 128)
    # Same sampling rate + identical waveform -> preserved.
    assert oracle._audio_preserved(wave, (wave.clone(), 44100), 44100) is True


def test_audio_preserved_matching_waveform_wrong_rate_false():
    torch = _require_real_torch()
    wave = torch.zeros(2, 128)
    # A real identical waveform but a different native sample rate must still
    # fail -- the rate gate applies even when the samples would compare equal.
    assert oracle._audio_preserved(wave, (wave.clone(), 44100), 22050) is False


# --------------------------------------------------------------------------- #
# --verify-local runs stdlib-only, without numpy (Round 24)
# --------------------------------------------------------------------------- #


def test_verify_local_cli_runs_without_numpy(tmp_path):
    """The checked-in ``--verify-local`` mode must run on a machine without numpy.

    The verifier uses only ``json`` / ``hashlib`` / ``pathlib``; numpy is used
    solely on the GPU/similarity paths and is now imported lazily. This spawns a
    subprocess whose import system raises on any ``numpy`` import, then runs the
    real script with ``--verify-local`` and asserts a clean exit -- proving that
    neither module import nor the verifier path touches numpy.
    """
    import json
    import subprocess
    import sys

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _make_local_package(pkg)

    bootstrap = (
        "import sys, runpy\n"
        "class _BlockNumpy:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'numpy' or name.startswith('numpy.'):\n"
        "            raise ImportError('numpy blocked for test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _BlockNumpy())\n"
        f"sys.argv = ['ltx2_retake_oracle', '--verify-local', {str(pkg)!r}]\n"
        f"runpy.run_path({str(_ORACLE_PATH)!r}, run_name='__main__')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", bootstrap],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "LOCAL_VERIFY" in proc.stdout
    payload = json.loads(proc.stdout.split("LOCAL_VERIFY", 1)[1].strip())
    assert payload["ok"] is True, payload
    # The verifier wrote its report and never imported numpy.
    assert (pkg / "local_verify.json").exists()
    assert "No module named 'numpy'" not in proc.stderr
