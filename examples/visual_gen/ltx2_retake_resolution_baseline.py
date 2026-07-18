# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolution / device baseline for the native LTX-2 retake on the 6000 PRO.

Measures the native retake denoise on an RTX PRO 6000 Blackwell (sm_120) across
source resolutions — 512x320, 1280x704 (~720p), 1920x1088 (~1080p) — reporting,
per resolution:

- cold model build/load + first-inference latency,
- warm ``denoise`` latency (``p50`` / ``p90`` / ``min``) over M measured retakes,
- peak GPU memory (``allocated`` + ``reserved``) for build/load and inference,

so the device-leverage question — does the 96GB 6000 PRO enable a 720p/1080p
retake, and at what latency/memory — is answered directly. The retake VAE
requires ``8k+1`` frames and ``/32`` spatial dims, so resolutions are validated
before use; a resolution that OOMs under bf16 is retried under FP8 (weights are
~half the size). Content is irrelevant to a latency/memory baseline, so higher-
resolution sources are synthesized (cv2). Each resolution runs in its own
subprocess so an OOM never poisons the others. Pure helpers are stdlib-only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# (label, width, height). 512x320 reuses the existing known-good source; the two
# larger ones are synthesized. All are /32 spatial with an 8k+1 frame count.
RESOLUTIONS = (
    ("512x320", 512, 320),
    ("720p_1280x704", 1280, 704),
    ("1080p_1920x1088", 1920, 1088),
)
BASELINE_LABEL = "512x320"
_BYTES_PER_GIB = 1024**3
_FRAME_TIME_RATIO = 8  # LTX causal VAE: source frames must be 8k+1


# --------------------------------------------------------------------------- #
# Pure helpers (stdlib-only; no torch / numpy / tensorrt_llm / cv2)
# --------------------------------------------------------------------------- #


def valid_resolution(width: int, height: int, frames: int) -> tuple:
    """Return ``(ok, reason)`` for the retake VAE shape constraints.

    ``8k+1`` frames, and width/height multiples of 32. ``reason`` is ``None`` when
    valid, else a concrete string.
    """
    if frames <= 0 or (frames - 1) % _FRAME_TIME_RATIO != 0:
        return False, f"frames={frames}_not_8k+1"
    if width % 32 != 0 or height % 32 != 0:
        return False, f"{width}x{height}_not_multiple_of_32"
    return True, None


def percentile(sorted_samples: list, p: float) -> Optional[float]:
    """Linear-interpolation percentile of an ascending-sorted list (p in [0,100])."""
    if not sorted_samples:
        return None
    if len(sorted_samples) == 1:
        return float(sorted_samples[0])
    rank = (p / 100.0) * (len(sorted_samples) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_samples[lo])
    frac = rank - lo
    return float(sorted_samples[lo] * (1.0 - frac) + sorted_samples[hi] * frac)


def summarize_samples(samples: list) -> dict:
    """``p50`` / ``p90`` / ``min`` / ``count`` over a sample list (order-independent)."""
    if not samples:
        return {"p50": None, "p90": None, "min": None, "count": 0}
    s = sorted(float(x) for x in samples)
    return {"p50": percentile(s, 50.0), "p90": percentile(s, 90.0), "min": s[0], "count": len(s)}


def gib(nbytes: Optional[float]) -> Optional[float]:
    """Bytes → GiB (rounded to 4 dp)."""
    if nbytes is None:
        return None
    return round(float(nbytes) / _BYTES_PER_GIB, 4)


def token_ratio(width: int, height: int, base_w: int, base_h: int) -> Optional[float]:
    """Spatial-latent token count ratio vs the baseline (``(w/32)*(h/32)``)."""
    base = (base_w // 32) * (base_h // 32)
    if base <= 0:
        return None
    return ((width // 32) * (height // 32)) / base


def classify_res_status(supported: bool, precheck_reason, ran_ok: bool, error_reason) -> tuple:
    """Map a resolution's precheck + run outcome to ``(status, reason)``."""
    if not supported:
        return "unsupported", precheck_reason
    if not ran_ok:
        return "error", error_reason
    return "ok", None


def is_oom(reason: Optional[str]) -> bool:
    """Heuristic: does an error reason look like a CUDA out-of-memory?"""
    if not reason:
        return False
    low = reason.lower()
    return "out of memory" in low or "outofmemory" in low or "cuda oom" in low


def summarize_resolutions(records: list) -> dict:
    """Roll up statuses + note which resolutions ran on the device."""
    return {
        "baseline_label": BASELINE_LABEL,
        "resolutions_ok": [r["label"] for r in records if r.get("status") == "ok"],
        "by_status": _count_status(records),
    }


def _count_status(records: list) -> dict:
    out = {}
    for r in records:
        out[r.get("status")] = out.get(r.get("status"), 0) + 1
    return out


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


# --------------------------------------------------------------------------- #
# Heavy path (imports inside functions so pure helpers load on a plain host).
# --------------------------------------------------------------------------- #


def _load_oracle():
    oracle_path = Path(__file__).resolve().with_name("ltx2_retake_oracle.py")
    spec = importlib.util.spec_from_file_location("ltx2_retake_oracle", oracle_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_source(path: str, width: int, height: int, frames: int, fps: int) -> int:
    """Synthesize a deterministic gradient .mp4 at ``width x height x frames``.

    Content is irrelevant to a latency/memory baseline; a moving gradient gives a
    real (non-constant) source the VAE encoder must process. Returns the frame
    count the file reads back as (for validation).
    """
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cv2 could not open VideoWriter for {path}")
    xr = np.linspace(0, 255, width, dtype=np.float32)
    yr = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    for i in range(frames):
        r = ((xr + i * 3) % 256).astype(np.uint8)[None, :].repeat(height, axis=0)
        g = ((yr + i * 5) % 256).astype(np.uint8).repeat(width, axis=1)
        b = np.full((height, width), (i * 7) % 256, dtype=np.uint8)
        writer.write(np.stack([b, g, r], axis=-1))
    writer.release()
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _run_single_res(args) -> int:
    """Build the retake pipeline, measure cold + warm + memory at one resolution."""
    oracle = _load_oracle()
    import time

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quant_algo = args.quant_algo or None

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    ran_ok, error_reason = False, None
    denoise_samples: list = []
    model_load_peak = inference_peak = None
    cold = None

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _sync()
        t0 = time.perf_counter()
        pipe = oracle.build_pipeline(
            args.checkpoint, args.gemma, args.lora, device, "VANILLA", quant_algo=quant_algo
        )
        _sync()
        build_load = time.perf_counter() - t0
        model_load_peak = _mem(torch)

        request = oracle._make_request(
            args.prompt,
            args.negative_prompt,
            args.source,
            args.start,
            args.end,
            args.seed,
            args.steps,
        )
        _sync()
        c0 = time.perf_counter()
        pipe.infer(request)  # cold first inference (lazy init + first-run cost)
        _sync()
        first_infer = time.perf_counter() - c0
        cold = {"model_build_load": build_load, "first_inference": first_infer}

        for _ in range(max(0, args.warmup - 1)):
            pipe.infer(request)
            _sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for _ in range(max(1, args.measured)):
            _sync()
            out = pipe.infer(request)
            _sync()
            denoise_samples.append(float(out.denoise))
        inference_peak = _mem(torch)
        ran_ok = True
    except Exception as exc:  # noqa: BLE001 - per-resolution isolation: record + continue
        error_reason = f"{type(exc).__name__}: {exc}"

    result = {
        "label": args.label,
        "width": args.width,
        "height": args.height,
        "quant_algo": quant_algo,
        "ran_ok": ran_ok,
        "error_reason": error_reason,
        "cold": cold,
        "warm_denoise": summarize_samples(denoise_samples) if denoise_samples else None,
        "model_load_peak": model_load_peak,
        "inference_peak": inference_peak,
    }
    print("RES_RESULT " + json.dumps(_json_safe(result)))
    return 0


def _mem(torch) -> dict:
    if not torch.cuda.is_available():
        return {"allocated_bytes": 0, "reserved_bytes": 0}
    return {
        "allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def parse_res_result(stdout: str) -> Optional[dict]:
    for line in stdout.splitlines():
        if line.startswith("RES_RESULT "):
            try:
                return json.loads(line[len("RES_RESULT ") :])
            except json.JSONDecodeError:
                return None
    return None


def _spawn_single_res(args, label, width, height, source, quant_algo) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-res",
        "--label",
        label,
        "--width",
        str(width),
        "--height",
        str(height),
        "--checkpoint",
        args.checkpoint,
        "--gemma",
        args.gemma,
        "--source",
        source,
        "--output-dir",
        args.output_dir,
        "--start",
        str(args.start),
        "--end",
        str(args.end),
        "--seed",
        str(args.seed),
        "--steps",
        str(args.steps),
        "--warmup",
        str(args.warmup),
        "--measured",
        str(args.measured),
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
    ]
    if quant_algo:
        cmd += ["--quant-algo", quant_algo]
    if args.lora:
        cmd += ["--lora", args.lora]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    rec = parse_res_result(proc.stdout)
    if rec is None:
        rec = {
            "label": label,
            "width": width,
            "height": height,
            "quant_algo": quant_algo,
            "ran_ok": False,
            "error_reason": f"no RES_RESULT (exit {proc.returncode}); stderr tail: "
            + "".join(proc.stderr.splitlines()[-3:]),
            "cold": None,
            "warm_denoise": None,
            "model_load_peak": None,
            "inference_peak": None,
        }
    return rec


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Native LTX-2 retake resolution/device baseline.")
    p.add_argument("--checkpoint")
    p.add_argument("--gemma")
    p.add_argument("--lora", default=None)
    p.add_argument(
        "--base-source", default=None, help="existing 512x320 source for the baseline row"
    )
    p.add_argument("--output-dir")
    p.add_argument("--start", type=float, default=1.0)
    p.add_argument("--end", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--frames", type=int, default=89)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--measured", type=int, default=6)
    p.add_argument("--prompt", default="a person talking to the camera")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--code-commit", default=None)
    # --single-res mode args
    p.add_argument("--single-res", action="store_true", help="internal: measure ONE resolution")
    p.add_argument("--label", default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--quant-algo", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.single_res:
        return _run_single_res(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_commit = args.code_commit or os.environ.get("LTX2_ORACLE_CODE_COMMIT")

    records = []
    for label, width, height in RESOLUTIONS:
        supported, precheck_reason = valid_resolution(width, height, args.frames)
        if not supported:
            print(f"[res] {label}: unsupported ({precheck_reason})", flush=True)
            records.append(_unsupported_record(label, width, height, precheck_reason))
            continue

        # Baseline reuses the existing known-good source; others are synthesized.
        if label == BASELINE_LABEL and args.base_source:
            source = args.base_source
        else:
            source = str(output_dir / f"src_{label}.mp4")
            if not os.path.exists(source):
                n = generate_source(source, width, height, args.frames, args.fps)
                print(f"[res] {label}: generated source ({n} frames)", flush=True)

        print(f"[res] {label}: bf16 build + measure ...", flush=True)
        rec = _spawn_single_res(args, label, width, height, source, None)
        # If bf16 OOMs, retry the SAME resolution under FP8 (half-size weights).
        if not rec.get("ran_ok") and is_oom(rec.get("error_reason")):
            print(f"[res] {label}: bf16 OOM -> retry FP8 ...", flush=True)
            fp8 = _spawn_single_res(args, label, width, height, source, "FP8")
            fp8["bf16_oom"] = True
            rec = fp8
        status, reason = classify_res_status(True, None, rec.get("ran_ok"), rec.get("error_reason"))
        rec["status"] = status
        rec["reason"] = reason
        rec["token_ratio_vs_baseline"] = token_ratio(width, height, 512, 320)
        print(f"[res] {label}: {status} ({reason}) quant={rec.get('quant_algo')}", flush=True)
        records.append(rec)

    report = {
        "mode": "native_retake_resolution_device_baseline",
        "device_query": _device_query(),
        "config": {
            "checkpoint": args.checkpoint,
            "gemma": args.gemma,
            "lora": args.lora,
            "frames": args.frames,
            "fps": args.fps,
            "window": [args.start, args.end],
            "seed": args.seed,
            "steps": args.steps,
            "warmup": args.warmup,
            "measured": args.measured,
            "code_commit": code_commit,
        },
        "records": records,
        "summary": summarize_resolutions(records),
    }
    out_path = output_dir / "resolution_baseline.json"
    with open(out_path, "w") as f:
        json.dump(_json_safe(report), f, indent=2, allow_nan=False)
    print(f"RESOLUTION_BASELINE_DONE {json.dumps(_json_safe(report['summary']))}")
    print(f"wrote {out_path}")
    return 0


def _unsupported_record(label, width, height, reason) -> dict:
    return {
        "label": label,
        "width": width,
        "height": height,
        "quant_algo": None,
        "status": "unsupported",
        "reason": reason,
        "ran_ok": False,
        "error_reason": reason,
        "cold": None,
        "warm_denoise": None,
        "model_load_peak": None,
        "inference_peak": None,
        "token_ratio_vs_baseline": token_ratio(width, height, 512, 320),
    }


def _device_query() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "name": torch.cuda.get_device_name(0),
            "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
            "total_gib": gib(torch.cuda.get_device_properties(0).total_memory),
        }
    except Exception:  # noqa: BLE001 - provenance only
        return {}


if __name__ == "__main__":
    raise SystemExit(main())
