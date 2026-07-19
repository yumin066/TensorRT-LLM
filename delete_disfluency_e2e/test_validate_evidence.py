#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Negative-case regression for the delete_disfluency evidence validator.

Copies the pulled host ``artifacts/`` bundle into a temporary directory, mutates one
piece of evidence, and asserts the validator rejects it via the targeted per-criterion
boolean. Skips when the (gitignored) host bundle is not present, e.g. in CI.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"


def _load():
    spec = importlib.util.spec_from_file_location(
        "validate_artifacts", HERE / "validate_artifacts.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["validate_artifacts"] = m
    spec.loader.exec_module(m)
    return m


pytestmark = pytest.mark.skipif(
    not (ART / "720p" / "manifest.json").exists(),
    reason="host artifacts/ evidence bundle not present (gitignored)",
)


@pytest.fixture
def bundle(tmp_path):
    """Repo-internal (gitignored) copy so the mutation is the only failing signal."""
    dst = ART / ".mut" / tmp_path.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ART, dst, ignore=shutil.ignore_patterns(".mut"))
    yield dst
    shutil.rmtree(dst, ignore_errors=True)


def _ev(root: Path):
    return _load().validate_evidence(root, ["720p", "1080p"])


def _patch(p: Path, fn):
    d = json.loads(p.read_text())
    fn(d)
    p.write_text(json.dumps(d, indent=2))


def test_real_bundle_passes():
    ev = _ev(ART)
    assert ev["ok"] is True, ev["problems"]
    assert all(ev["ac"].values()), ev["ac"]


def test_missing_required_variant_row_fails(bundle):
    _patch(bundle / "720p" / "manifest.json", lambda d: d["variant_status"].pop("serve_bf16"))
    ev = _ev(bundle)
    assert ev["ac"]["AC-4"] is False
    assert ev["ok"] is False


def test_native_missing_ltx_seconds_fails(bundle):
    _patch(
        bundle / "720p" / "phase_timing.json",
        lambda d: d["variants"]["native_fp8"].pop("ltx_seconds"),
    )
    ev = _ev(bundle)
    assert ev["ac"]["AC-5"] is False


def test_quality_status_only_fails(bundle):
    _patch(
        bundle / "720p" / "quality_metrics.json",
        lambda d: d["variants"].__setitem__("native_fp8", {"status": "ok"}),
    )
    ev = _ev(bundle)
    assert ev["ac"]["AC-6"] is False


def test_bad_ffprobe_geometry_fails(bundle):
    def bad(d):
        d["variants"]["native_fp8"]["ffprobe"]["retake_output"]["video"]["nb_frames"] = "7"

    _patch(bundle / "720p" / "quality_metrics.json", bad)
    ev = _ev(bundle)
    assert ev["ac"]["AC-6"] is False


def test_contradictory_non_ok_status_fails(bundle):
    _patch(bundle / "720p" / "serve_fp8" / "status.json", lambda d: d.__setitem__("status", "ok"))
    ev = _ev(bundle)
    assert ev["ac"]["AC-4"] is False


@pytest.mark.parametrize("field", ["status", "mode", "quant"])
def test_phase_identity_missing_fails(bundle, field):
    _patch(
        bundle / "720p" / "phase_timing.json",
        lambda d: d["variants"]["native_fp8"].pop(field),
    )
    ev = _ev(bundle)
    assert ev["ac"]["AC-5"] is False


def test_non_ok_phase_identity_missing_fails(bundle):
    _patch(
        bundle / "720p" / "phase_timing.json",
        lambda d: d["variants"]["serve_fp8"].pop("status"),
    )
    ev = _ev(bundle)
    assert ev["ac"]["AC-5"] is False


def test_weak_oom_diagnostic_fails(bundle):
    def weak(d):
        d["status"] = "oom"
        d["reason"] = "oom"
        d.pop("oom_excerpt", None)

    _patch(bundle / "1080p" / "native_bf16" / "status.json", weak)
    (bundle / "1080p" / "native_bf16" / "run.log").write_text("startup ok\ninference done\n")
    ev = _ev(bundle)
    assert ev["ac"]["AC-4"] is False


def test_weak_unsupported_diagnostic_fails(bundle):
    def weak(d):
        d["status"] = "unsupported"
        d["reason"] = "quant"
        d.pop("error_excerpt", None)

    _patch(bundle / "720p" / "serve_fp8" / "status.json", weak)
    (bundle / "720p" / "serve_fp8" / "run.log").write_text("startup ok\nserved request\n")
    ev = _ev(bundle)
    assert ev["ac"]["AC-4"] is False


def test_non_ignored_artifact_path_fails(tmp_path):
    dst = tmp_path / "artifacts"
    shutil.copytree(ART, dst, ignore=shutil.ignore_patterns(".mut"))
    ev = _ev(dst)
    assert ev["ac"]["AC-9"] is False
    assert ev["ok"] is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
