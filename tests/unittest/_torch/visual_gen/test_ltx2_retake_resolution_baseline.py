# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side unit tests for the pure helpers in ``ltx2_retake_resolution_baseline``.

Only stdlib-only helpers (resolution validity, percentile/summary, GiB, token
ratio, status classify, OOM heuristic, summary, RES_RESULT parse, JSON sanitize)
are exercised; the GPU build/measure + cv2 source-gen live in the heavy path.
"""

import importlib.util
from pathlib import Path

import pytest

_RES_PATH = (
    Path(__file__).resolve().parents[4]
    / "examples"
    / "visual_gen"
    / "ltx2_retake_resolution_baseline.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("ltx2_retake_resolution_baseline", _RES_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


res = _load()
_GIB = 1024**3


# --------------------------------------------------------------------------- #
# valid_resolution
# --------------------------------------------------------------------------- #


def test_valid_resolution_accepts_32_multiple_and_8kp1():
    assert res.valid_resolution(1280, 704, 89) == (True, None)
    assert res.valid_resolution(1920, 1088, 97) == (True, None)


def test_valid_resolution_rejects_non_32_spatial():
    ok, reason = res.valid_resolution(1280, 720, 89)
    assert ok is False and "not_multiple_of_32" in reason


def test_valid_resolution_rejects_bad_frame_count():
    ok, reason = res.valid_resolution(1280, 704, 90)
    assert ok is False and "8k+1" in reason


# --------------------------------------------------------------------------- #
# token_ratio
# --------------------------------------------------------------------------- #


def test_token_ratio_vs_baseline():
    # 512x320 baseline -> (16*10)=160 tokens. 1280x704 -> (40*22)=880 -> 5.5x.
    assert res.token_ratio(1280, 704, 512, 320) == pytest.approx(880 / 160)
    assert res.token_ratio(512, 320, 512, 320) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# classify_res_status + is_oom
# --------------------------------------------------------------------------- #


def test_classify_ok():
    assert res.classify_res_status(True, None, True, None) == ("ok", None)


def test_classify_unsupported():
    assert res.classify_res_status(False, "x_not_multiple_of_32", False, None) == (
        "unsupported",
        "x_not_multiple_of_32",
    )


def test_classify_error():
    assert res.classify_res_status(True, None, False, "OutOfMemoryError: ...") == (
        "error",
        "OutOfMemoryError: ...",
    )


def test_is_oom_detects_cuda_oom():
    assert res.is_oom("OutOfMemoryError: CUDA out of memory. Tried to allocate ...") is True
    assert res.is_oom("ValueError: bad shape") is False
    assert res.is_oom(None) is False


# --------------------------------------------------------------------------- #
# summary / gib / parse / json_safe
# --------------------------------------------------------------------------- #


def test_gib():
    assert res.gib(_GIB) == 1.0
    assert res.gib(None) is None


def test_summarize_resolutions():
    records = [
        {"label": "512x320", "status": "ok"},
        {"label": "720p_1280x704", "status": "ok"},
        {"label": "1080p_1920x1088", "status": "error"},
    ]
    s = res.summarize_resolutions(records)
    assert s["resolutions_ok"] == ["512x320", "720p_1280x704"]
    assert s["by_status"] == {"ok": 2, "error": 1}


def test_parse_res_result_roundtrip():
    line = 'RES_RESULT {"label": "720p_1280x704", "ran_ok": true}'
    assert res.parse_res_result("x\n" + line + "\ny")["label"] == "720p_1280x704"


def test_parse_res_result_absent():
    assert res.parse_res_result("nothing") is None


def test_json_safe():
    out = res._json_safe({"a": float("inf"), "b": [float("nan")]})
    assert out["a"] == "inf" and out["b"][0] == "nan"


# --------------------------------------------------------------------------- #
# no plan-process labels
# --------------------------------------------------------------------------- #


def test_resolution_baseline_has_no_plan_process_labels():
    text = _RES_PATH.read_text()
    assert "AC-" not in text
    for label in ("Milestone", "Phase "):
        assert label not in text


# --------------------------------------------------------------------------- #
# stdlib-only import (helpers load with no numpy / torch / tensorrt_llm / cv2)
# --------------------------------------------------------------------------- #


def test_resolution_baseline_pure_helpers_run_without_numpy():
    import subprocess
    import sys

    bootstrap = (
        "import sys, importlib.util\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        top = name.split('.')[0]\n"
        "        if top in ('numpy', 'torch', 'tensorrt_llm', 'cv2'):\n"
        "            raise ImportError(f'{top} blocked for test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        f"spec = importlib.util.spec_from_file_location('r', {str(_RES_PATH)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "assert m.valid_resolution(1280, 704, 89) == (True, None)\n"
        "assert m.token_ratio(1280, 704, 512, 320) == 880/160\n"
        "assert m.is_oom('CUDA out of memory') is True\n"
        "print('RES_PURE_OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", bootstrap], capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "RES_PURE_OK" in proc.stdout
