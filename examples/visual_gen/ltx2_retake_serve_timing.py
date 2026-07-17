#! /usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm-serve HTTP resident-warm timing for the native LTX-2 retake (Mode B).

Launches ONE ``trtllm-serve`` server for the retake 22b (``retake_offload_mode:
none``), waits for ``GET /health`` readiness, then sends ``N`` warmup + ``M``
measured ``POST /v1/videos/generations`` requests reusing the same source /
prompt / window / seed as the Mode A and native-harness artifacts. Each response
carries a ``Server-Timing`` header (engine-side ``generation`` + ``denoise``
seconds); this tool records that plus per-request wall latency and HTTP status,
reports the **first-served request separately from steady-warm p50/p90/min** (raw
samples included), and always tears the server down.

This is the AC-2 production serve surface (HTTP, resident worker) as opposed to
the pipeline-direct probe in ``ltx2_retake_timing.py``. It records the server
command + captured log path + env + provenance + the Mode A every-rebuild
comparison, as strict JSON.

The percentile / JSON helpers are reused from ``ltx2_retake_timing`` (loaded by
path). The pure helpers here (payload assembly, ``Server-Timing`` parsing,
first/steady split, response classification) import on a plain host with no
numpy / torch / tensorrt_llm; all HTTP uses the stdlib (``urllib``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

_STEADY_KEYS = ("wall", "generation", "denoise")


# ----------------------------------------------------------------------------
# Pure helpers (stdlib only; host-testable without numpy / torch / tensorrt_llm).
# ----------------------------------------------------------------------------


def build_retake_payload(
    prompt: str,
    negative_prompt: str,
    source: str,
    start: float,
    end: float,
    seed: int,
    steps: int,
    model: str = "ltx2-retake",
    fmt: str = "pt",
) -> dict:
    """A ``/v1/videos/generations`` retake payload (retake knobs in extra_params).

    ``format='pt'`` returns the raw tensor and avoids video-encoder noise in the
    measured latency; retake-specific fields go under ``extra_params`` (1:1 with
    ``VisualGenParams``). ``response_format='url'`` selects the ``FileResponse``
    branch, which is the ONLY one the route attaches the ``Server-Timing`` header
    to (the ``b64_json`` JSONResponse omits it), so ``generation``/``denoise``
    engine metrics are available.
    """
    return {
        "model": model,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "num_inference_steps": steps,
        "format": fmt,
        "response_format": "url",
        "extra_params": {
            "retake_video_path": source,
            "retake_start_time": start,
            "retake_end_time": end,
            "retake_regenerate_video": True,
            "retake_regenerate_audio": False,
            "retake_enhance_prompt": False,
            "retake_max_batch_size": 1,
        },
    }


def parse_server_timing(header_value: Optional[str]) -> dict:
    """Parse a ``Server-Timing`` header into ``{metric: seconds}`` (``dur`` is ms)."""
    out: dict = {}
    if not header_value:
        return out
    for entry in header_value.split(","):
        parts = [p.strip() for p in entry.split(";")]
        name = parts[0]
        if not name:
            continue
        for param in parts[1:]:
            key, _, val = param.partition("=")
            if key.strip() == "dur":
                try:
                    out[name] = float(val) / 1000.0
                except ValueError:
                    pass
                break
    return out


def classify_response(status: int, server_timing: dict) -> tuple:
    """Return ``(ok, reason)``: 200 with a positive ``generation`` metric is ok."""
    if status != 200:
        return False, f"http_status_{status}"
    gen = server_timing.get("generation")
    if gen is None or gen <= 0:
        return False, "missing_or_nonpositive_generation_timing"
    return True, None


def split_first_steady(records: list) -> tuple:
    """Split successful measured records into ``(first, steady_list)``."""
    oks = [r for r in records if r.get("ok")]
    if not oks:
        return None, []
    return oks[0], oks[1:]


def summarize_serve(records: list, summarize_fn, keys=_STEADY_KEYS) -> dict:
    """Per-key ``summarize_fn`` (p50/p90/min/count) across steady-warm records."""
    out = {}
    for key in keys:
        samples = [r[key] for r in records if key in r and r[key] is not None]
        if samples:
            out[key] = summarize_fn(samples)
    return out


# ----------------------------------------------------------------------------
# Heavy path (server lifecycle + HTTP; stdlib only, no torch/tensorrt_llm here).
# ----------------------------------------------------------------------------


def _load_sibling(name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().with_name(f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _write_retake_serve_yaml(path: Path, gemma: str, lora: Optional[str]) -> None:
    """Write a retake serve config with resident-warm (offload none) defaults.

    The serve pipeline_config validator for LTX2Pipeline uses ``distilled_lora_path``
    for the (distilled) retake LoRA — NOT ``retake_lora_path`` (that key is only
    accepted on the oracle's direct-``build_pipeline`` path, which bypasses this
    validator).
    """
    lines = [
        "pipeline_config:",
        "  workflow: retake",
        "  retake_distilled: true",
        "  retake_offload_mode: none",
        f"  text_encoder_path: {gemma}",
    ]
    if lora:
        lines += [f"  distilled_lora_path: {lora}"]
    lines += ["attention_config:", "  backend: VANILLA", ""]
    path.write_text("\n".join(lines))


def _wait_health(base_url: str, timeout_s: float, poll_s: float = 3.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(poll_s)
    return False


def _post_generation(base_url: str, payload: dict, timeout_s: float) -> dict:
    """POST one request; return status/latency/server_timing/size/error."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/videos/generations",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
            latency = time.perf_counter() - t0
            timing = parse_server_timing(resp.headers.get("Server-Timing"))
            return {
                "status": resp.status,
                "wall": latency,
                "server_timing": timing,
                "body_size": len(body),
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        return {
            "status": exc.code,
            "wall": time.perf_counter() - t0,
            "server_timing": {},
            "body_size": 0,
            "error": (exc.read()[:500].decode("utf-8", "replace") if exc.fp else str(exc)),
        }
    except (urllib.error.URLError, OSError) as exc:
        return {
            "status": -1,
            "wall": time.perf_counter() - t0,
            "server_timing": {},
            "body_size": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _record_from_response(index: int, resp: dict) -> dict:
    ok, reason = classify_response(resp["status"], resp["server_timing"])
    return {
        "index": index,
        "ok": ok,
        "reason": reason,
        "status": resp["status"],
        "wall": resp["wall"],
        "generation": resp["server_timing"].get("generation"),
        "denoise": resp["server_timing"].get("denoise"),
        "body_size": resp["body_size"],
        "error": resp["error"],
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="trtllm-serve HTTP resident-warm retake timing.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--gemma", required=True)
    p.add_argument("--lora", default=None)
    p.add_argument("--source", required=True, help="video-only source clip")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--measured", type=int, default=8)
    p.add_argument("--prompt", default="a person talking to the camera")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--health-timeout", type=float, default=900.0)
    p.add_argument(
        "--serve-cmd",
        default="trtllm-serve",
        help="serve launcher (shlex-split); e.g. 'python -m tensorrt_llm.commands.serve'",
    )
    p.add_argument("--mode-b-warm-p50-seconds", type=float, default=None)
    p.add_argument("--mode-a-total-p50-seconds", type=float, default=None)
    p.add_argument("--code-commit", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    timing = _load_sibling("ltx2_retake_timing")
    oracle = _load_sibling("ltx2_retake_oracle")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_commit = args.code_commit or os.environ.get("LTX2_ORACLE_CODE_COMMIT")

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    yaml_path = output_dir / "retake_serve.yml"
    _write_retake_serve_yaml(yaml_path, args.gemma, args.lora)
    server_log = output_dir / "server.log"
    import shlex

    server_cmd = shlex.split(args.serve_cmd) + [
        args.checkpoint,
        "--visual_gen_args",
        str(yaml_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]

    payload = build_retake_payload(
        args.prompt, args.negative_prompt, args.source, args.start, args.end, args.seed, args.steps
    )

    records = []
    ready = False
    server_error = None
    log_fh = open(server_log, "w")
    proc = subprocess.Popen(server_cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    try:
        print(f"[serve] launching (port {port}); waiting for /health...", flush=True)
        ready = _wait_health(base_url, args.health_timeout)
        if not ready:
            server_error = "server did not become healthy within --health-timeout"
            print(f"[serve] {server_error}", file=sys.stderr, flush=True)
        else:
            print("[serve] healthy; warmup...", flush=True)
            for _ in range(max(0, args.warmup)):
                _post_generation(base_url, payload, args.request_timeout)
            for i in range(args.measured):
                resp = _post_generation(base_url, payload, args.request_timeout)
                rec = _record_from_response(i, resp)
                records.append(rec)
                print(
                    f"[serve] measured {i}: ok={rec['ok']} status={rec['status']} "
                    f"wall={rec['wall']:.3f}s generation={rec['generation']}",
                    flush=True,
                )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_fh.close()

    first, steady = split_first_steady(records)
    steady_summary = summarize_serve(steady, timing.summarize_samples)
    comparison = {
        "mode_b_serve_first_served_wall": first["wall"] if first else None,
        "mode_b_serve_steady_wall_p50": steady_summary.get("wall", {}).get("p50"),
        "mode_b_pipeline_direct_warm_p50_seconds": args.mode_b_warm_p50_seconds,
        "mode_a_every_rebuild_total_p50_seconds": args.mode_a_total_p50_seconds,
    }

    native_repo = oracle._git_info(str(Path(__file__).resolve().parents[2]))
    result = {
        "mode": "mode_b_trtllm_serve_http_resident_warm",
        "server_ready": ready,
        "server_error": server_error,
        "server_cmd": server_cmd,
        "server_log": str(server_log),
        "first_served": first,
        "steady_warm": steady_summary,
        "steady_warm_count": len(steady),
        "records": records,
        "comparison": comparison,
        "warmup_iterations": max(0, args.warmup),
        "measured_iterations": args.measured,
        "config": {
            "attention_backend": "VANILLA",
            "dtype": "bf16",
            "retake_offload_mode": "none",
            "format": payload["format"],
            "response_format": payload["response_format"],
        },
        "env": oracle._env_metadata(),
        "code_commit": code_commit,
        "native_repo": native_repo,
        "source": str(Path(args.source).resolve()),
        "source_sha256": oracle.sha256_file(args.source),
        "window": [args.start, args.end],
        "seed": args.seed,
        "steps": args.steps,
        "note": (
            "AC-2 production serve surface: one trtllm-serve resident worker "
            "(retake_offload_mode: none) over HTTP. first_served is separated from "
            "steady-warm p50/p90/min; generation/denoise come from the response "
            "Server-Timing header. Compare with the pipeline-direct native harness "
            "and the Mode A every-rebuild reference."
        ),
    }
    (output_dir / "serve_timing.json").write_text(
        json.dumps(timing._json_safe(result), indent=2, sort_keys=True, allow_nan=False)
    )
    print(
        "SERVE_TIMING_DONE",
        json.dumps(
            {
                "ready": ready,
                "steady_count": len(steady),
                "steady_wall_p50": steady_summary.get("wall", {}).get("p50"),
            }
        ),
    )
    # Success needs a healthy server and at least one steady-warm sample.
    ok = ready and len(steady) >= 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
