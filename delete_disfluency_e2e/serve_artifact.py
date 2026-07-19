#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch a trtllm-serve retake worker and save ONE mp4 artifact via HTTP.

Timing (cold/first/warm, Server-Timing) is produced by ltx2_retake_serve_timing.py
(response_format='url'); this script instead requests response_format='b64_json'
with format='mp4' to pull the actual generated clip bytes, so the serve variant
gets a real deliverable that can be composited through --external-retake.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path


class _GpuPeak:
    """Poll nvidia-smi in a thread to capture the serve process's peak GPU memory."""

    def __init__(self):
        self.max_mib = 0
        self._stop = False
        self._t = None

    def _loop(self):
        while not self._stop:
            try:
                out = (
                    subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits",
                            "-i",
                            "0",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    .stdout.strip()
                    .splitlines()
                )
                if out:
                    self.max_mib = max(self.max_mib, int(out[0]))
            except Exception:
                pass
            time.sleep(1.5)

    def start(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        self._stop = True
        if self._t:
            self._t.join(timeout=3)
        return round(self.max_mib / 1024, 2)


def _load_mod(repo: str, name: str):
    p = f"{repo}/examples/visual_gen/{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_serve_mod(repo: str):
    return _load_mod(repo, "ltx2_retake_serve_timing")


def _load_oracle(repo: str):
    return _load_mod(repo, "ltx2_retake_oracle")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_health(base: str, timeout: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--yaml", required=True, help="existing retake_serve.yml")
    ap.add_argument("--serve-cmd", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out-mp4", required=True)
    ap.add_argument("--fps", type=float, default=29.97002997002997)
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--startup-timeout", type=float, default=900)
    ap.add_argument("--request-timeout", type=float, default=900)
    args = ap.parse_args()

    serve = load_serve_mod(args.repo)
    payload = serve.build_retake_payload(
        "a person talking to the camera",
        "",
        args.source,
        args.start,
        args.end,
        args.seed,
        args.steps,
    )
    # format='pt' returns the raw tensor and avoids the server-side GPU mp4 encode,
    # which OOMs against the resident model; we encode the mp4 offline on CPU.
    payload["format"] = "pt"
    payload["response_format"] = "b64_json"  # returns {id, format, b64_json}

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    cmd = shlex.split(args.serve_cmd) + [
        args.checkpoint,
        "--visual_gen_args",
        args.yaml,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    t_launch = time.time()
    with open(args.server_log, "w") as lg:
        srv = subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT)
    try:
        if not wait_health(base, args.startup_timeout):
            print("SERVE_ARTIFACT_FAIL health timeout")
            sys.exit(2)
        cold_start_s = round(time.time() - t_launch, 2)
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            base + "/v1/videos/generations",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        gpu = _GpuPeak()
        gpu.start()
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=args.request_timeout) as r:
            body = json.loads(r.read())
        wall = time.time() - t0
        peak_gib = gpu.stop()
        b64 = body.get("b64_json")
        if not b64:
            print("SERVE_ARTIFACT_FAIL no b64_json: " + json.dumps(body)[:300])
            sys.exit(3)
        pt_path = str(Path(args.out_mp4).with_suffix(".pt"))
        Path(pt_path).write_bytes(base64.b64decode(b64))
        # offline CPU encode: the served tensor -> THWC uint8 -> mp4 via the oracle helpers
        import io

        import torch

        oracle = _load_oracle(args.repo)
        obj = torch.load(
            io.BytesIO(Path(pt_path).read_bytes()), map_location="cpu", weights_only=False
        )
        vid = obj["video"] if isinstance(obj, dict) and "video" in obj else obj
        if isinstance(vid, (list, tuple)):
            vid = vid[0]
        thwc = oracle._video_to_thwc_uint8(vid)
        ok = oracle.encode_mp4(thwc, args.fps, args.out_mp4)
        n = int(getattr(thwc, "shape", [None])[0]) if hasattr(thwc, "shape") else None
        meta_path = str(Path(args.out_mp4).with_name("serve_cold_start.json"))
        Path(meta_path).write_text(
            json.dumps(
                {
                    "cold_start_seconds": cold_start_s,
                    "request_wall_seconds": round(wall, 2),
                    "single_shot_seconds": cold_start_s + round(wall, 2),
                    "peak_reserved_gib": peak_gib,
                    "frames": n,
                    "served_format": body.get("format"),
                },
                indent=2,
            )
        )
        print(
            "SERVE_ARTIFACT_DONE "
            + json.dumps(
                {
                    "out_mp4": args.out_mp4,
                    "pt": pt_path,
                    "frames": n,
                    "mp4_ok": bool(ok),
                    "bytes": Path(args.out_mp4).stat().st_size
                    if Path(args.out_mp4).exists()
                    else 0,
                    "wall_seconds": round(wall, 2),
                    "served_format": body.get("format"),
                }
            )
        )
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except Exception:
            srv.kill()


if __name__ == "__main__":
    main()
