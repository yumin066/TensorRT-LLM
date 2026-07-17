# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side unit tests for the pure helpers in ``ltx2_retake_timing``.

Only the stdlib-only helpers (percentiles, sample summary, measured split, stage
aggregation, timeline assembly, JSON sanitization) are exercised; the heavy
load-once/serve-many run lives inside ``main()`` and reuses the oracle module.
The harness lives under ``examples/`` so it is loaded by path via ``importlib``.
"""

import importlib.util
from pathlib import Path

import pytest

_TIMING_PATH = (
    Path(__file__).resolve().parents[4] / "examples" / "visual_gen" / "ltx2_retake_timing.py"
)


def _load_timing():
    spec = importlib.util.spec_from_file_location("ltx2_retake_timing", _TIMING_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timing = _load_timing()


# --------------------------------------------------------------------------- #
# percentile / summarize_samples
# --------------------------------------------------------------------------- #


def test_percentile_linear_interpolation():
    s = [1.0, 2.0, 3.0, 4.0]
    assert timing.percentile(s, 0) == 1.0
    assert timing.percentile(s, 100) == 4.0
    assert timing.percentile(s, 50) == pytest.approx(2.5)
    assert timing.percentile(s, 90) == pytest.approx(3.7)


def test_percentile_edge_cases():
    assert timing.percentile([], 50) is None
    assert timing.percentile([5.0], 50) == 5.0
    assert timing.percentile([5.0], 90) == 5.0


def test_summarize_samples_is_order_independent():
    a = timing.summarize_samples([3.0, 1.0, 2.0, 4.0])
    b = timing.summarize_samples([4.0, 3.0, 2.0, 1.0])
    assert a == b
    assert a["min"] == 1.0
    assert a["count"] == 4
    assert a["p50"] == pytest.approx(2.5)


# --------------------------------------------------------------------------- #
# split_measured
# --------------------------------------------------------------------------- #


def test_split_measured_separates_first_from_steady():
    records = [{"index": i, "wall": float(i)} for i in range(4)]
    first, steady = timing.split_measured(records)
    assert first["index"] == 0
    assert [r["index"] for r in steady] == [1, 2, 3]


def test_split_measured_empty():
    first, steady = timing.split_measured([])
    assert first is None and steady == []


# --------------------------------------------------------------------------- #
# aggregate_stages
# --------------------------------------------------------------------------- #


def test_aggregate_stages_per_key_summary():
    records = [
        {"wall": 2.0, "denoise": 1.0},
        {"wall": 4.0, "denoise": 3.0},
    ]
    agg = timing.aggregate_stages(records, stage_keys=("wall", "denoise"))
    assert agg["wall"]["min"] == 2.0
    assert agg["denoise"]["min"] == 1.0
    assert agg["wall"]["count"] == 2


def test_aggregate_stages_skips_missing_keys():
    records = [{"wall": 2.0}, {"wall": 4.0}]
    agg = timing.aggregate_stages(records, stage_keys=("wall", "denoise"))
    assert "wall" in agg
    assert "denoise" not in agg


# --------------------------------------------------------------------------- #
# cpu_io_seconds / aggregate_fine_stages
# --------------------------------------------------------------------------- #


def test_cpu_io_is_wall_minus_gpu_stages():
    rec = {"wall": 2.0, "pre_denoise": 0.5, "denoise": 1.0, "post_denoise": 0.2}
    # host-side remainder = 2.0 - (0.5 + 1.0 + 0.2) = 0.3
    assert timing.cpu_io_seconds(rec) == pytest.approx(0.3)


def test_cpu_io_clamps_negative_to_zero():
    # If GPU-stage sum slightly exceeds wall (timer/sync skew), clamp at 0.
    rec = {"wall": 1.0, "pre_denoise": 0.6, "denoise": 0.6, "post_denoise": 0.0}
    assert timing.cpu_io_seconds(rec) == 0.0


def test_cpu_io_treats_missing_stages_as_zero():
    rec = {"wall": 1.5}
    assert timing.cpu_io_seconds(rec) == 1.5


def test_aggregate_fine_stages_summarizes_and_flattens_per_step():
    records = [
        {
            "stage_timings": {
                "vae_encode": 0.20,
                "denoise_total": 1.0,
                "denoise_per_step": [0.1, 0.12],
            }
        },
        {
            "stage_timings": {
                "vae_encode": 0.24,
                "denoise_total": 1.2,
                "denoise_per_step": [0.11, 0.13],
            }
        },
    ]
    agg = timing.aggregate_fine_stages(records)
    assert agg["vae_encode"]["min"] == 0.20
    assert agg["denoise_total"]["count"] == 2
    # per-step values are flattened across records (4 total).
    assert agg["denoise_per_step"]["count"] == 4
    assert agg["denoise_per_step"]["min"] == pytest.approx(0.1)


def test_aggregate_fine_stages_handles_missing_stage_timings():
    records = [{"wall": 1.0}, {"stage_timings": None}]
    assert timing.aggregate_fine_stages(records) == {}


# --------------------------------------------------------------------------- #
# build_timeline
# --------------------------------------------------------------------------- #


def test_build_timeline_separates_cold_first_and_steady():
    records = [
        {"index": 0, "wall": 10.0, "denoise": 8.0},  # first_served (slow)
        {"index": 1, "wall": 2.0, "denoise": 1.0},
        {"index": 2, "wall": 2.2, "denoise": 1.1},
    ]
    tl = timing.build_timeline(5.0, records, stage_keys=("wall", "denoise"))
    assert tl["cold_model_build_load_seconds"] == 5.0
    # first_served is the first measured request, kept out of steady-warm.
    assert tl["first_served"]["index"] == 0
    assert tl["steady_warm_count"] == 2
    # steady-warm min excludes the slow first_served request.
    assert tl["steady_warm"]["wall"]["min"] == 2.0


def test_build_timeline_single_measured_has_no_steady():
    records = [{"index": 0, "wall": 3.0}]
    tl = timing.build_timeline(5.0, records)
    assert tl["first_served"]["index"] == 0
    assert tl["steady_warm_count"] == 0


# --------------------------------------------------------------------------- #
# _json_safe
# --------------------------------------------------------------------------- #


def test_json_safe_sanitizes_non_finite():
    import json
    import math

    safe = timing._json_safe({"psnr": math.inf, "ok": 1.5, "nested": [{"x": math.nan}]})
    assert safe["psnr"] == "inf"
    assert safe["ok"] == 1.5
    assert safe["nested"][0]["x"] == "nan"
    json.dumps(safe, allow_nan=False)  # must not raise


# --------------------------------------------------------------------------- #
# stdlib-only: helpers import + run without numpy/torch (Round 27)
# --------------------------------------------------------------------------- #


def test_timing_pure_helpers_run_without_numpy():
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
        f"spec = importlib.util.spec_from_file_location('ltx2_retake_timing', {str(_TIMING_PATH)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "recs = [{'index': i, 'wall': float(i + 1), 'denoise': float(i)} for i in range(3)]\n"
        "tl = m.build_timeline(4.0, recs, stage_keys=('wall', 'denoise'))\n"
        "assert tl['steady_warm_count'] == 2\n"
        "assert m.percentile([1.0, 2.0, 3.0], 50) == 2.0\n"
        "print('TIMING_PURE_OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", bootstrap], capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "TIMING_PURE_OK" in proc.stdout
