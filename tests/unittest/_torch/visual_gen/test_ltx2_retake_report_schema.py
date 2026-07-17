# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side unit tests for the pure aggregator in ``ltx2_retake_report_schema``.

The report schema runner is stdlib-only, so every helper and adapter is exercised
directly on small synthetic artifact dicts written to ``tmp_path`` -- the real
measurement artifacts are never a dependency. The runner lives under
``examples/`` so it is loaded by path via ``importlib``.
"""

import importlib.util
import json
import math
from pathlib import Path

_SCHEMA_PATH = (
    Path(__file__).resolve().parents[4] / "examples" / "visual_gen" / "ltx2_retake_report_schema.py"
)


def _load_schema():
    spec = importlib.util.spec_from_file_location("ltx2_retake_report_schema", _SCHEMA_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


schema = _load_schema()


# --------------------------------------------------------------------------- #
# Synthetic artifact builders (tiny, self-contained)
# --------------------------------------------------------------------------- #


_PROVENANCE = {
    "code_commit": "abc123",
    "native_repo": {
        "branch": "script-editing-ltx-retake",
        "commit": "deadbeef",
        "dirty": True,
        "path": "/scratch/TensorRT-LLM",
    },
    "env": {
        "cuda_version": "13.1",
        "device_name": "NVIDIA RTX PRO 6000",
        "platform": "Linux",
        "python": "3.12.3",
        "torch_version": "2.11.0",
    },
    "source": "/assets/vonly_512x320_89f.mp4",
    "source_sha256": "sha-src",
    "window": [1.0, 2.0],
    "seed": 42,
    "steps": 8,
}


def _native_timing_artifact():
    art = dict(_PROVENANCE)
    art.update(
        {
            "mode": "resident_warm_native_retake",
            "config": {
                "attention_backend": "VANILLA",
                "cuda_graph": False,
                "dtype": "bf16",
                "quant_algo": None,
                "retake_offload_mode": "none",
            },
            "quality_informational": {"psnr": "inf", "ssim": 1.0},
            "records": [{"wall": 1.8}, {"wall": 1.82}],
            "timeline": {
                "cold_model_build_load_seconds": 87.6,
                "compile_graph_capture_note": "not applicable for bf16/VANILLA",
                "first_served": {"wall": 1.80, "index": 0},
                "steady_warm": {"wall": {"p50": 1.82, "p90": 1.83, "min": 1.81, "count": 1}},
                "steady_warm_count": 1,
            },
        }
    )
    return art


def _upstream_baseline_artifact():
    art = _native_timing_artifact()
    art["mode"] = "resident_warm_upstream_stage_baseline"
    art["config"]["retake_use_upstream_stage"] = True
    return art


def _mode_a_artifact():
    art = dict(_PROVENANCE)
    art.update(
        {
            "mode": "mode_a_every_rebuild_upstream_retake",
            "quality_informational": {"all_identical": True},
            "records": [{"index": 0, "model_build_load": 70.0, "run_total": 15.6, "total": 85.6}],
            "summary": {
                "model_build_load": {"p50": 68.0, "p90": 69.6, "min": 66.0, "count": 3},
                "run_total": {"p50": 15.4, "p90": 15.5, "min": 15.2, "count": 3},
                "total": {"p50": 83.3, "p90": 85.1, "min": 81.5, "count": 3},
            },
        }
    )
    return art


def _serve_artifact():
    art = dict(_PROVENANCE)
    art.update(
        {
            "mode": "mode_b_trtllm_serve_http_resident_warm",
            "config": {
                "attention_backend": "VANILLA",
                "checkpoint": "/ckpt.safetensors",
                "dtype": "bf16",
                "lora": "/lora.safetensors",
                "lora_strength": 1.0,
                "retake_offload_mode": "none",
            },
            "first_served": {"wall": 2.10, "index": 0},
            "records": [{"wall": 2.10}, {"wall": 2.12}],
            "run_ok": True,
            "run_reason": None,
            "server_log": "/server.log",
            "steady_warm": {
                "denoise": {"p50": 1.14, "p90": 1.15, "min": 1.13, "count": 7},
                "generation": {"p50": 1.74, "p90": 1.75, "min": 1.73, "count": 7},
                "wall": {"p50": 2.12, "p90": 2.13, "min": 2.10, "count": 7},
            },
            "steady_warm_count": 7,
        }
    )
    return art


def _gating_record(label, axis, kind, status, **overrides):
    record = {
        "attention_backend": "VANILLA",
        "axis": axis,
        "baseline": kind == "baseline",
        "cold_model_build_load": 56.3,
        "cuda_graph": False,
        "dtype": "bf16",
        "first_served": 1.73,
        "kind": kind,
        "label": label,
        "peak_memory": {"allocated": 72578496512, "reserved": 75111596032},
        "per_stage": {"denoise_total": {"p50": 1.14, "count": 7}},
        "quality_informational": {"available": True},
        "quant_algo": None,
        "raw_samples": [1.73, 1.74],
        "reason": None,
        "status": status,
        "steady_warm": {"p50": 1.74, "p90": 1.75, "min": 1.73, "count": 7},
        "torch_compile": False,
    }
    record.update(overrides)
    return record


def _gating_artifact():
    art = dict(_PROVENANCE)
    art.update(
        {
            "mode": "accel_gating",
            "num_frames": 89,
            "records": [
                _gating_record("bf16/VANILLA", "baseline", "baseline", "ok"),
                _gating_record(
                    "bf16/torch_compile",
                    "torch_compile",
                    "single-axis",
                    "ok",
                    torch_compile=True,
                ),
                _gating_record(
                    "bf16/FA4",
                    "attn",
                    "single-axis",
                    "regresses",
                    attention_backend="FA4",
                    reason="steady p50 slower than baseline",
                ),
                _gating_record(
                    "stack:torch_compile",
                    "stack",
                    "stack",
                    "ok",
                    torch_compile=True,
                ),
            ],
        }
    )
    return art


def _compile_cost_artifact():
    art = dict(_PROVENANCE)
    art.update(
        {
            "mode": "compile_cost",
            "config": {"attention_backend": "VANILLA", "dtype": "bf16", "torch_compile": True},
            "derived": {
                "cache_saved_seconds": 5.13,
                "compile_cost_seconds": 6.97,
                "steady_p50": 1.739,
                "warm_disk_first_seconds": 3.57,
            },
            "records": [
                {"mode": "empty", "first_call": 8.71, "steady": {"p50": 1.739}},
                {"mode": "warm", "first_call": 3.57, "steady": {"p50": 1.738}},
            ],
        }
    )
    return art


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


# --------------------------------------------------------------------------- #
# shape manifest / hash
# --------------------------------------------------------------------------- #


def test_shape_hash_is_stable_for_equal_manifests():
    a = schema.canonical_shape_manifest(resolution="512x320", num_frames=89, seed=42, steps=8)
    b = schema.canonical_shape_manifest(resolution="512x320", num_frames=89, seed=42, steps=8)
    assert schema.shape_hash(a) == schema.shape_hash(b)


def test_shape_hash_diverges_on_changed_field():
    base = schema.canonical_shape_manifest(resolution="512x320", num_frames=89, seed=42)
    changed = schema.canonical_shape_manifest(resolution="512x320", num_frames=89, seed=43)
    assert schema.shape_hash(base) != schema.shape_hash(changed)


def test_shape_manifest_defaults_edit_type_and_keeps_all_keys():
    manifest = schema.canonical_shape_manifest(resolution="512x320")
    assert manifest["edit_type"] == "retake"
    for key in schema._SHAPE_KEYS:
        assert key in manifest
    # Unknown fields are explicit null rather than dropped.
    assert manifest["latent_shape"] is None
    assert manifest["prompt_len"] is None


# --------------------------------------------------------------------------- #
# normalize_memory
# --------------------------------------------------------------------------- #


def test_normalize_memory_explicit_null_when_absent():
    mem = schema.normalize_memory(None)
    assert mem["peak_allocated"] is None
    assert mem["peak_reserved"] is None
    assert mem["stage_attribution"] is None
    assert mem["has_stage_attribution"] is False


def test_normalize_memory_carries_peak_when_present():
    mem = schema.normalize_memory({"allocated": 100, "reserved": 200})
    assert mem["peak_allocated"] == 100
    assert mem["peak_reserved"] == 200
    assert mem["has_stage_attribution"] is False


# --------------------------------------------------------------------------- #
# normalize_provenance / normalize_env
# --------------------------------------------------------------------------- #


def test_normalize_provenance_preserves_dirty_and_adds_note():
    prov = schema.normalize_provenance(_PROVENANCE)
    assert prov["native_repo"]["dirty"] is True
    assert prov["native_repo"]["commit"] == "deadbeef"
    assert prov["code_commit"] == "abc123"
    assert prov["patch_hash"] is None
    assert "provenance_note" in prov
    assert "rsync" in prov["provenance_note"]


def test_normalize_provenance_no_note_when_clean():
    art = {"code_commit": "x", "native_repo": {"dirty": False, "commit": "c"}}
    prov = schema.normalize_provenance(art)
    assert "provenance_note" not in prov


def test_normalize_env_adds_explicit_missing_keys():
    env = schema.normalize_env(_PROVENANCE["env"])
    assert env["torch_version"] == "2.11.0"
    assert env["gpu_total_memory"] is None
    assert env["flash_attn_version"] is None
    assert env["cutedsl_version"] is None


# --------------------------------------------------------------------------- #
# adapters
# --------------------------------------------------------------------------- #


def _ref():
    return schema._source_ref("timing.json", "round-29-timing", "/tmp/timing.json")


def test_adapt_native_timing_maps_cold_and_warm():
    rows = schema.adapt_native_timing(_native_timing_artifact(), _ref())
    assert len(rows) == 1
    row = rows[0]
    assert row["mode"] == "native_resident_warm"
    assert row["config"]["attention_backend"] == "VANILLA"
    assert row["cold"]["model_build_load"] == 87.6
    assert row["warm"]["first_served"] == 1.80
    assert row["warm"]["steady"]["p50"] == 1.82
    assert row["memory"]["peak_allocated"] is None
    assert row["shape_manifest"]["resolution"] == "512x320"


def test_adapt_upstream_baseline_mode_and_flag():
    rows = schema.adapt_upstream_baseline(_upstream_baseline_artifact(), _ref())
    row = rows[0]
    assert row["mode"] == "upstream_stage_persistent"
    assert row["config"]["retake_use_upstream_stage"] is True


def test_adapt_mode_a_has_no_warm_steady_and_carries_totals():
    rows = schema.adapt_mode_a(_mode_a_artifact(), _ref())
    row = rows[0]
    assert row["mode"] == "mode_a_every_rebuild"
    assert row["warm"]["steady"] is None
    assert row["cold"]["model_build_load"] == 68.0
    assert row["cold"]["first_call_or_compile"] == 15.4
    assert row["cold"]["every_rebuild_total_p50"] == 83.3


def test_adapt_serve_maps_steady_wall_and_serve_metrics():
    rows = schema.adapt_serve(_serve_artifact(), _ref())
    row = rows[0]
    assert row["mode"] == "serve_http"
    assert row["status"] == "ok"
    assert row["warm"]["first_served"] == 2.10
    assert row["warm"]["steady"]["p50"] == 2.12
    assert row["serve_metrics"]["generation"]["p50"] == 1.74
    assert row["serve_metrics"]["denoise"]["min"] == 1.13


def test_adapt_serve_failure_carries_status_and_reason():
    art = _serve_artifact()
    art["run_ok"] = False
    art["run_reason"] = "a request returned 500"
    rows = schema.adapt_serve(art, _ref())
    row = rows[0]
    assert row["status"] == "error"
    assert row["failure"] == "a request returned 500"


def test_adapt_gating_one_row_per_record_with_modes():
    rows = schema.adapt_gating(_gating_artifact(), _ref())
    assert len(rows) == 4
    by_label = {r["config"]["label"]: r for r in rows}
    assert by_label["bf16/VANILLA"]["mode"] == "accel_gating_single_axis"
    assert by_label["stack:torch_compile"]["mode"] == "accel_gating_stack"
    tc = by_label["bf16/torch_compile"]
    assert tc["config"]["torch_compile"] is True
    assert tc["warm"]["steady"]["p50"] == 1.74
    assert tc["warm"]["raw_samples"] == [1.73, 1.74]
    assert tc["memory"]["peak_allocated"] == 72578496512
    assert tc["per_stage"]["denoise_total"]["count"] == 7


def test_adapt_gating_error_record_carries_failure():
    art = _gating_artifact()
    art["records"].append(
        _gating_record(
            "bf16/broken",
            "attn",
            "single-axis",
            "error",
            attention_backend="FA4",
            reason="RuntimeError: kernel launch failed",
        )
    )
    rows = schema.adapt_gating(art, _ref())
    broken = [r for r in rows if r["config"]["label"] == "bf16/broken"][0]
    assert broken["status"] == "error"
    assert "kernel launch failed" in broken["failure"]


def test_gating_regresses_status_carries_reason_as_failure():
    rows = schema.adapt_gating(_gating_artifact(), _ref())
    fa4 = [r for r in rows if r["config"]["label"] == "bf16/FA4"][0]
    assert fa4["status"] == "regresses"
    assert "slower than baseline" in fa4["failure"]


# --------------------------------------------------------------------------- #
# compile-cost block attach
# --------------------------------------------------------------------------- #


def test_compile_cost_block_attaches_to_torch_compile_row():
    rows = schema.adapt_gating(_gating_artifact(), _ref())
    block = schema.build_compile_cost_block(_compile_cost_artifact(), _ref())
    leftover = schema.attach_compile_cost(rows, block)
    assert leftover is None
    tc = [r for r in rows if r["config"]["label"] == "bf16/torch_compile"][0]
    assert tc["compile_cost"]["compile_cost_seconds"] == 6.97
    assert tc["compile_cost"]["empty_first"] == 8.71
    assert tc["compile_cost"]["same_process_steady_p50"] == 1.739
    # No other row got the block.
    others = [r for r in rows if r["config"]["label"] != "bf16/torch_compile"]
    assert all(r["compile_cost"] is None for r in others)


def test_compile_cost_block_survives_when_no_torch_compile_row():
    # Only a baseline row exists (no torch_compile single-axis row).
    rows = schema.adapt_gating(
        {**_PROVENANCE, "records": [_gating_record("bf16/VANILLA", "baseline", "baseline", "ok")]},
        _ref(),
    )
    block = schema.build_compile_cost_block(_compile_cost_artifact(), _ref())
    leftover = schema.attach_compile_cost(rows, block)
    assert leftover is block
    assert rows[0]["compile_cost"] is None


# --------------------------------------------------------------------------- #
# build_report: partial-artifact handling + serialization
# --------------------------------------------------------------------------- #


def test_build_report_records_missing_sources_without_crashing(tmp_path):
    # Only the native timing artifact is present; everything else is missing.
    paths = {
        "native_timing": _write(tmp_path, "timing.json", _native_timing_artifact()),
        "mode_a": tmp_path / "absent_mode_a.json",
        "serve": tmp_path / "absent_serve.json",
        "upstream_baseline": tmp_path / "absent_upstream.json",
        "gating": tmp_path / "absent_gating.json",
        "compile_cost": tmp_path / "absent_compile.json",
    }
    report = schema.build_report(paths)
    assert report["row_count"] == 1
    assert report["rows"][0]["mode"] == "native_resident_warm"
    missing = {m["source"] for m in report["missing_sources"]}
    assert missing == {"mode_a", "serve", "upstream_baseline", "gating", "compile_cost"}
    assert len(report["generated_from"]) == 1


def test_build_report_full_set_serializes_strictly(tmp_path):
    paths = {
        "native_timing": _write(tmp_path, "timing.json", _native_timing_artifact()),
        "mode_a": _write(tmp_path, "mode_a.json", _mode_a_artifact()),
        "serve": _write(tmp_path, "serve.json", _serve_artifact()),
        "upstream_baseline": _write(tmp_path, "upstream.json", _upstream_baseline_artifact()),
        "gating": _write(tmp_path, "gating.json", _gating_artifact()),
        "compile_cost": _write(tmp_path, "compile.json", _compile_cost_artifact()),
    }
    report = schema.build_report(paths)
    modes = set(report["modes"])
    assert "native_resident_warm" in modes
    assert "upstream_stage_persistent" in modes
    assert "mode_a_every_rebuild" in modes
    assert "serve_http" in modes
    assert "accel_gating_single_axis" in modes
    assert "accel_gating_stack" in modes
    # compile-cost was attached to a torch_compile row, so the top-level block is null.
    assert report["compile_cost"] is None
    # Strict JSON (the inf-valued psnr becomes a sentinel string).
    text = json.dumps(schema._json_safe(report), allow_nan=False, sort_keys=True)
    assert json.loads(text)["row_count"] == report["row_count"]


def test_main_writes_report_and_returns_zero(tmp_path):
    paths_dir = tmp_path / "artifacts"
    (paths_dir / "round-29-timing").mkdir(parents=True)
    (paths_dir / "round-29-timing" / "timing.json").write_text(
        json.dumps(_native_timing_artifact())
    )
    output = tmp_path / "report.json"
    rc = schema.main(["--artifacts-dir", str(paths_dir), "--output", str(output)])
    assert rc == 0
    loaded = json.loads(output.read_text())
    assert loaded["row_count"] == 1
    assert loaded["schema_version"] == "1"


# --------------------------------------------------------------------------- #
# _json_safe
# --------------------------------------------------------------------------- #


def test_json_safe_sanitizes_non_finite():
    safe = schema._json_safe({"psnr": math.inf, "neg": -math.inf, "n": math.nan, "ok": 1.5})
    assert safe["psnr"] == "inf"
    assert safe["neg"] == "-inf"
    assert safe["n"] == "nan"
    assert safe["ok"] == 1.5
    json.dumps(safe, allow_nan=False)  # must not raise


# --------------------------------------------------------------------------- #
# hygiene: no plan-process labels leaked into the runner
# --------------------------------------------------------------------------- #


def test_report_schema_runner_has_no_plan_process_labels():
    # Plan-process terminology is banned in checked-in implementation text. The
    # banned tokens are assembled from fragments so this guard file does not
    # itself contain them literally (its own check would otherwise flag them).
    text = _SCHEMA_PATH.read_text()
    banned = [
        "A" + "C-",
        "Stage" + " 1",
        "Stage" + " 2",
        "Stage" + "-2",
        "Mile" + "stone",
        "Ph" + "ase ",
    ]
    for label in banned:
        assert label not in text, f"plan-process label {label!r} leaked into the report runner"
