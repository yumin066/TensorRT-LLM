# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merged customer matrix for the native LTX-2 retake acceleration study.

Aggregates the per-round measurement artifacts (Mode A every-rebuild reference,
Mode B resident-warm serve, the acceleration-axis smoke, the attention profiler,
the quantization latency/memory sweep, and the resolution/device baseline) into
one customer-facing matrix with latency / memory / status / informational
quality, plus the Mode B amortization (the break-even call count where a resident
worker overtakes every-rebuild), and a production configuration recommendation.

Pure aggregation only — this reads already-pulled artifact JSONs from
``--artifacts-dir`` and emits ``final_matrix.json`` (strict) + ``final_matrix.md``.
A missing artifact is recorded as an explicit gap, never a crash. Everything here
is stdlib-only so it runs on any host.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Optional

# Artifact locations relative to --artifacts-dir (round dir / filename).
ARTIFACTS = {
    "mode_a": ("round-30-mode-a", "mode_a_timing.json"),
    "serve": ("round-32-serve-timing", "serve_timing.json"),
    "timing": ("round-29-timing", "timing.json"),
    "smoke": ("round-35-smoke", "smoke_matrix.json"),
    "attn": ("round-43-attn-profile", "attn_profile.json"),
    "quant": ("round-44-quant-mem", "quant_mem.json"),
    "resolution": ("round-45-resolution", "resolution_baseline.json"),
}


# --------------------------------------------------------------------------- #
# Pure helpers (stdlib-only)
# --------------------------------------------------------------------------- #


def load_json(path: str) -> Optional[dict]:
    """Load a JSON artifact, or ``None`` if absent / unparsable (recorded as a gap)."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def speedup(slow: Optional[float], fast: Optional[float]) -> Optional[float]:
    """``slow / fast`` (how many times faster ``fast`` is), or ``None`` if unusable."""
    if not slow or not fast or slow <= 0 or fast <= 0:
        return None
    return slow / fast


def amortization(
    cold_load_s: Optional[float], warm_s: Optional[float], every_rebuild_s: Optional[float]
) -> dict:
    """Mode B (resident) vs Mode A (every-rebuild) amortization.

    Mode B cost for N calls = ``cold_load + N*warm``; Mode A = ``N*every_rebuild``.
    Returns the break-even N where Mode B first wins (``ceil(cold_load /
    (every_rebuild - warm))``) and the per-call and 100-call ratios.
    """
    if not cold_load_s or not warm_s or not every_rebuild_s:
        return {"break_even_calls": None, "per_call_speedup": None, "ratio_100_calls": None}
    if every_rebuild_s <= warm_s:
        return {
            "break_even_calls": None,
            "per_call_speedup": speedup(every_rebuild_s, warm_s),
            "ratio_100_calls": None,
        }
    break_even = math.ceil(cold_load_s / (every_rebuild_s - warm_s))
    mode_b_100 = cold_load_s + 100 * warm_s
    mode_a_100 = 100 * every_rebuild_s
    return {
        "break_even_calls": break_even,
        "per_call_speedup": speedup(every_rebuild_s, warm_s),
        "ratio_100_calls": speedup(mode_a_100, mode_b_100),
    }


def _p50(block: Optional[dict], *keys) -> Optional[float]:
    cur = block
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def build_latency_section(
    mode_a: Optional[dict], serve: Optional[dict], timing: Optional[dict]
) -> dict:
    """Mode A every-rebuild vs Mode B resident-warm (serve HTTP + pipeline-direct) + amortization."""
    mode_a_total = _p50(mode_a, "summary", "total", "p50") if mode_a else None
    mode_a_load = _p50(mode_a, "summary", "model_build_load", "p50") if mode_a else None
    serve_wall = _p50(serve, "steady_warm", "wall", "p50") if serve else None
    serve_gen = _p50(serve, "steady_warm", "generation", "p50") if serve else None
    warm_wall = _p50(timing, "timeline", "steady_warm", "wall", "p50") if timing else None
    cold_load = _p50(timing, "timeline", "cold_model_build_load_seconds") if timing else None
    if cold_load is None and mode_a_load is not None:
        cold_load = mode_a_load
    amort = amortization(cold_load, warm_wall or serve_gen, mode_a_total)
    return {
        "mode_a_every_rebuild_total_p50_s": mode_a_total,
        "mode_b_serve_http_wall_p50_s": serve_wall,
        "mode_b_serve_generation_p50_s": serve_gen,
        "mode_b_pipeline_warm_wall_p50_s": warm_wall,
        "mode_b_cold_load_s": cold_load,
        "amortization": amort,
        "gaps": [
            k for k, v in {"mode_a": mode_a, "serve": serve, "timing": timing}.items() if v is None
        ],
    }


def build_accel_section(smoke: Optional[dict], attn: Optional[dict]) -> dict:
    """Acceleration axes: per-config smoke status + the attention-share finding."""
    configs = []
    if smoke:
        for r in smoke.get("records", smoke.get("matrix", [])):
            configs.append(
                {
                    "config": r.get("config", r.get("label")),
                    "status": r.get("status"),
                    "reason": r.get("reason"),
                }
            )
    attn_rows = []
    if attn:
        for r in attn.get("records", []):
            attn_rows.append(
                {
                    "backend": r.get("backend"),
                    "status": r.get("status"),
                    "denoise_p50_s": _p50(r, "denoise", "p50"),
                    "attention_share": _p50(r, "attention_share", "share"),
                    "reason": r.get("reason"),
                }
            )
    return {
        "smoke_configs": configs,
        "attention_backends": attn_rows,
        "gaps": [k for k, v in {"smoke": smoke, "attn": attn}.items() if v is None],
    }


def build_quant_section(quant: Optional[dict]) -> dict:
    """bf16 vs FP8 vs NVFP4: latency + resident memory + informational window quality."""
    rows = []
    if quant:
        for r in quant.get("records", []):
            rows.append(
                {
                    "mode": r.get("mode"),
                    "status": r.get("status"),
                    "denoise_p50_s": _p50(r, "denoise", "p50"),
                    "speedup_vs_bf16": _p50(r, "latency_delta", "speedup_vs_bf16"),
                    "resident_gib": r.get("resident_allocated_gib"),
                    "saved_gib": _p50(r, "memory_delta", "saved_gib"),
                    "window_psnr": _p50(r, "quality_informational", "window", "psnr"),
                    "window_ssim": _p50(r, "quality_informational", "window", "ssim"),
                }
            )
    return {"modes": rows, "gaps": [] if quant else ["quant"]}


def build_device_section(resolution: Optional[dict]) -> dict:
    """Device × resolution: cold + warm latency + peak reserved memory on the 6000 PRO."""
    rows = []
    device = None
    if resolution:
        device = resolution.get("device_query", {}).get("name")
        for r in resolution.get("records", []):
            rows.append(
                {
                    "resolution": r.get("label"),
                    "status": r.get("status"),
                    "token_ratio": r.get("token_ratio_vs_baseline"),
                    "cold_load_s": _p50(r, "cold", "model_build_load"),
                    "cold_first_infer_s": _p50(r, "cold", "first_inference"),
                    "warm_denoise_p50_s": _p50(r, "warm_denoise", "p50"),
                    "infer_reserved_gib": _gib(_p50(r, "inference_peak", "reserved_bytes")),
                }
            )
    return {"device": device, "resolutions": rows, "gaps": [] if resolution else ["resolution"]}


def _gib(nbytes) -> Optional[float]:
    if not isinstance(nbytes, (int, float)) or nbytes <= 0:
        return None
    return round(nbytes / (1024**3), 1)


def _json_safe(obj):
    """Recursively replace non-finite floats with string sentinels (strict JSON)."""
    if isinstance(obj, float):
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        if math.isnan(obj):
            return "nan"
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def render_markdown(matrix: dict) -> str:
    """Render the merged matrix as a compact human-readable markdown report."""
    lat = matrix["latency"]
    amo = lat["amortization"]
    lines = ["# LTX-2 retake acceleration — customer matrix", ""]
    lines.append("## Customer value #2 — production warm latency (Mode A vs Mode B)")
    lines.append("")
    lines.append("| platform | latency/call |")
    lines.append("|----------|--------------|")
    lines.append(
        f"| Mode A (LTX2.3-eval every-rebuild) | {lat['mode_a_every_rebuild_total_p50_s']} s |"
    )
    lines.append(
        f"| Mode B (TRT-LLM serve HTTP, resident warm) | {lat['mode_b_serve_http_wall_p50_s']} s "
        f"wall / {lat['mode_b_serve_generation_p50_s']} s generation |"
    )
    lines.append(f"| Mode B (pipeline-direct warm) | {lat['mode_b_pipeline_warm_wall_p50_s']} s |")
    lines.append("")
    lines.append(
        f"One-time cold load ~{lat['mode_b_cold_load_s']} s. **Amortization: Mode B overtakes "
        f"every-rebuild after ~{amo['break_even_calls']} call(s)**; per-call {amo['per_call_speedup']}× "
        f"faster; over 100 calls {amo['ratio_100_calls']}× faster."
    )
    lines.append("")
    lines.append("## Acceleration axes (native retake)")
    for c in matrix["acceleration"]["smoke_configs"]:
        lines.append(
            f"- {c['config']}: {c['status']}" + (f" ({c['reason']})" if c.get("reason") else "")
        )
    lines.append("")
    lines.append("Attention backends (attention is a small share of denoise):")
    for a in matrix["acceleration"]["attention_backends"]:
        sh = a.get("attention_share")
        lines.append(
            f"- {a['backend']}: {a['status']}"
            + (f", share={round(sh, 4)}" if isinstance(sh, (int, float)) else "")
            + (f" ({a['reason']})" if a.get("reason") else "")
        )
    lines.append("")
    lines.append("## Quantization (latency / memory / informational quality)")
    lines.append("")
    lines.append("| mode | denoise p50 | speedup | resident | saved | window PSNR / SSIM |")
    lines.append("|------|-------------|---------|----------|-------|--------------------|")
    for m in matrix["quantization"]["modes"]:
        lines.append(
            f"| {m['mode']} | {m['denoise_p50_s']} s | {m['speedup_vs_bf16']} | "
            f"{m['resident_gib']} GiB | {m['saved_gib']} GiB | "
            f"{m['window_psnr']} / {m['window_ssim']} |"
        )
    lines.append("")
    lines.append("## Customer value #3 — device × resolution (RTX PRO 6000 Blackwell, 96GB)")
    lines.append("")
    lines.append("| resolution | tokens× | cold load | warm denoise p50 | infer reserved |")
    lines.append("|------------|---------|-----------|------------------|----------------|")
    for r in matrix["device"]["resolutions"]:
        lines.append(
            f"| {r['resolution']} | {r['token_ratio']} | {r['cold_load_s']} s | "
            f"{r['warm_denoise_p50_s']} s | {r['infer_reserved_gib']} GiB |"
        )
    lines.append("")
    if matrix.get("recommendation"):
        lines.append("## Production recommendation")
        lines.append("")
        lines.append(matrix["recommendation"])
        lines.append("")
    gaps = matrix.get("gaps", [])
    if gaps:
        lines.append(f"_Gaps (absent artifacts): {', '.join(gaps)}_")
    return "\n".join(lines) + "\n"


def build_matrix(artifacts_dir: str, recommendation: Optional[str] = None) -> dict:
    """Load all artifacts and assemble the merged matrix (missing → recorded gap)."""
    loaded = {
        key: load_json(os.path.join(artifacts_dir, rnd, fn)) for key, (rnd, fn) in ARTIFACTS.items()
    }
    latency = build_latency_section(loaded["mode_a"], loaded["serve"], loaded["timing"])
    accel = build_accel_section(loaded["smoke"], loaded["attn"])
    quant = build_quant_section(loaded["quant"])
    device = build_device_section(loaded["resolution"])
    gaps = sorted(set(latency["gaps"] + accel["gaps"] + quant["gaps"] + device["gaps"]))
    return {
        "latency": latency,
        "acceleration": accel,
        "quantization": quant,
        "device": device,
        "recommendation": recommendation,
        "gaps": gaps,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Merged LTX-2 retake customer matrix.")
    p.add_argument(
        "--artifacts-dir", required=True, help="dir holding the round-*/*.json artifacts"
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--recommendation-file",
        default=None,
        help="optional path to a production-recommendation markdown/text to embed",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    recommendation = None
    if args.recommendation_file and os.path.exists(args.recommendation_file):
        with open(args.recommendation_file) as f:
            recommendation = f.read().strip()
    matrix = build_matrix(args.artifacts_dir, recommendation)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "final_matrix.json"), "w") as f:
        json.dump(_json_safe(matrix), f, indent=2, allow_nan=False)
    with open(os.path.join(args.output_dir, "final_matrix.md"), "w") as f:
        f.write(render_markdown(matrix))
    print(f"FINAL_MATRIX_DONE gaps={matrix['gaps']}")
    print(f"wrote {args.output_dir}/final_matrix.json + final_matrix.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
