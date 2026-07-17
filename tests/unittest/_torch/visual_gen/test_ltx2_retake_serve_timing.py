# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side unit tests for the pure helpers in ``ltx2_retake_serve_timing``.

Only the stdlib-only helpers (payload assembly, Server-Timing parsing, response
classification, first/steady split, summary) are exercised; the server lifecycle
+ HTTP live in ``main()``. The runner lives under ``examples/`` so it is loaded by
path via ``importlib``.
"""

import importlib.util
from pathlib import Path

import pytest

_SERVE_PATH = (
    Path(__file__).resolve().parents[4] / "examples" / "visual_gen" / "ltx2_retake_serve_timing.py"
)


def _load_serve():
    spec = importlib.util.spec_from_file_location("ltx2_retake_serve_timing", _SERVE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


serve = _load_serve()


def _fake_summarize(samples):
    s = sorted(samples)
    return {"p50": s[len(s) // 2], "min": s[0], "count": len(s)}


# --------------------------------------------------------------------------- #
# build_retake_payload
# --------------------------------------------------------------------------- #


def test_payload_has_retake_extra_params_and_format():
    p = serve.build_retake_payload("hi", "", "/tmp/src.mp4", 1.0, 2.0, 42, 8)
    assert p["format"] == "pt"
    assert p["seed"] == 42
    assert p["num_inference_steps"] == 8
    ep = p["extra_params"]
    assert ep["retake_video_path"] == "/tmp/src.mp4"
    assert ep["retake_start_time"] == 1.0
    assert ep["retake_end_time"] == 2.0
    assert ep["retake_regenerate_video"] is True
    assert ep["retake_regenerate_audio"] is False


# --------------------------------------------------------------------------- #
# parse_server_timing
# --------------------------------------------------------------------------- #


def test_parse_server_timing_ms_to_seconds():
    t = serve.parse_server_timing("generation;dur=1826.5, denoise;dur=1225.9")
    assert t["generation"] == pytest.approx(1.8265)
    assert t["denoise"] == pytest.approx(1.2259)


def test_parse_server_timing_empty_and_malformed():
    assert serve.parse_server_timing(None) == {}
    assert serve.parse_server_timing("") == {}
    # A metric without a numeric dur is skipped, not crashed on.
    assert serve.parse_server_timing("generation;dur=abc") == {}


# --------------------------------------------------------------------------- #
# classify_response
# --------------------------------------------------------------------------- #


def test_classify_ok_on_200_with_positive_generation():
    ok, reason = serve.classify_response(200, {"generation": 1.8})
    assert ok is True and reason is None


def test_classify_error_on_non_200():
    ok, reason = serve.classify_response(500, {"generation": 1.8})
    assert ok is False and "500" in reason


def test_classify_error_on_missing_generation():
    ok, reason = serve.classify_response(200, {"denoise": 1.2})
    assert ok is False and "generation" in reason


# --------------------------------------------------------------------------- #
# split_first_steady / summarize_serve
# --------------------------------------------------------------------------- #


def test_split_first_steady_uses_only_ok_records():
    records = [
        {"index": 0, "ok": False, "wall": 9.0},
        {"index": 1, "ok": True, "wall": 2.0},
        {"index": 2, "ok": True, "wall": 2.1},
        {"index": 3, "ok": True, "wall": 2.2},
    ]
    first, steady = serve.split_first_steady(records)
    assert first["index"] == 1  # first SUCCESSFUL request
    assert [r["index"] for r in steady] == [2, 3]


def test_split_first_steady_none_when_all_failed():
    first, steady = serve.split_first_steady([{"ok": False}])
    assert first is None and steady == []


def test_summarize_serve_per_key():
    records = [
        {"wall": 2.0, "generation": 1.8, "denoise": 1.2},
        {"wall": 2.2, "generation": 1.9, "denoise": 1.3},
    ]
    s = serve.summarize_serve(records, _fake_summarize)
    assert s["wall"]["min"] == 2.0
    assert s["generation"]["count"] == 2
    assert s["denoise"]["min"] == 1.2


# --------------------------------------------------------------------------- #
# stdlib-only import (Round 31)
# --------------------------------------------------------------------------- #


def test_serve_pure_helpers_run_without_numpy():
    import subprocess
    import sys

    bootstrap = (
        "import sys, importlib.util\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        top = name.split('.')[0]\n"
        "        if top in ('numpy', 'torch', 'tensorrt_llm'):\n"
        "            raise ImportError(f'{top} blocked for test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        f"spec = importlib.util.spec_from_file_location('s', {str(_SERVE_PATH)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "assert m.build_retake_payload('p','','x',1.0,2.0,42,8)['format']=='pt'\n"
        "assert m.parse_server_timing('generation;dur=1000')['generation']==1.0\n"
        "assert m.classify_response(200, {'generation': 1.0})[0] is True\n"
        "print('SERVE_PURE_OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", bootstrap], capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "SERVE_PURE_OK" in proc.stdout
