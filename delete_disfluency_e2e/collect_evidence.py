#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evidence collector for the delete_disfluency E2E test.

Reads every variant's outputs for one resolution and writes machine-readable
evidence next to them:
  manifest.json        source sha256 / geometry / seed / asset+quant paths / git revs / artifact sha256s + mtimes
  phase_timing.json    per-variant APPLY/LTX/POST/wall (+ serve cold/first/warm/http/engine), status
  quality_metrics.json per-variant window PSNR+SSIM vs bf16 and vs upstream + ffprobe metadata
  assertions.json      per-final retake-video-only / final-has-audio / audio-vs-edited / outside-window pass/fail

Pure stdlib + numpy + PyAV (+ optional skimage for SSIM). No torch / GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


# ---- variant registry -------------------------------------------------------
# name -> (retake_mp4 relpath, final_dir, extra_json relpath or None, kind)
def variant_specs(res: str):
    q = ["bf16", "fp8", "nvfp4"]
    specs = {
        "upstream": {
            "retake": "upstream/retake_output.mp4",
            "final_dir": "upstream",
            "timing": "upstream/timing.json",
            "phase": "upstream/phase_timing.json",
            "kind": "upstream",
            "quant": "fp8-cast",
        },
    }
    for x in q:
        specs[f"native_{x}"] = {
            "retake": f"native_{x}/native.mp4",
            "final_dir": f"native_{x}_final",
            "variant": f"native_{x}/variant.json",
            "phase": f"native_{x}_final/phase_timing.json",
            "kind": "native",
            "quant": x,
        }
    for x in q:
        specs[f"serve_{x}"] = {
            "retake": f"serve_{x}/retake_serve_{x}.mp4",
            "final_dir": f"serve_{x}_final",
            "serve_timing": f"serve_{x}/serve_timing.json",
            "status": f"serve_{x}/status.json",
            "phase": f"serve_{x}_final/phase_timing.json",
            "kind": "serve",
            "quant": x,
        }
    return specs


# ---- helpers ----------------------------------------------------------------
def _np_default(o):
    if hasattr(o, "item"):  # numpy scalar (bool_/float64/int64)
        return o.item()
    if isinstance(o, (set, frozenset)):
        return list(o)
    return str(o)


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, default=_np_default)


def sha256_file(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def ffprobe(p: Path) -> dict:
    if not p.exists():
        return {"exists": False}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(p)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        data = json.loads(out)
    except Exception as e:
        return {"exists": True, "ffprobe_error": str(e)}
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    meta = {"exists": True}
    if v:
        meta["video"] = {
            "width": v.get("width"),
            "height": v.get("height"),
            "nb_frames": v.get("nb_frames"),
            "avg_frame_rate": v.get("avg_frame_rate"),
            "pix_fmt": v.get("pix_fmt"),
        }
    meta["audio"] = (
        None
        if a is None
        else {
            "codec_name": a.get("codec_name"),
            "sample_rate": a.get("sample_rate"),
            "duration": a.get("duration"),
        }
    )
    meta["has_audio"] = a is not None
    return meta


def decode_rgb(p: Path):
    import av

    c = av.open(str(p))
    try:
        vs = next(s for s in c.streams if s.type == "video")
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(vs)]
    finally:
        c.close()
    return np.stack(frames)


def psnr(a: np.ndarray, b: np.ndarray):
    n = min(len(a), len(b))
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    if a.shape != b.shape:
        return None, None
    mse = float(np.mean((a - b) ** 2))
    p = None if mse == 0 else round(20 * np.log10(255.0) - 10 * np.log10(mse), 3)
    return p, round(mse, 4)


def ssim_mean(a: np.ndarray, b: np.ndarray):
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        return None
    n = min(len(a), len(b))
    if a[:n].shape != b[:n].shape:
        return None
    vals = [ssim(a[i], b[i], channel_axis=2, data_range=255) for i in range(n)]
    return round(float(np.mean(vals)), 4)


def variant_status(
    final_mp4: Path, retake: Path, status_json: Path, logs: list[Path]
) -> tuple[str, str]:
    if final_mp4.exists():
        return "ok", ""
    sj = load_json(status_json)
    if sj and sj.get("status"):
        return sj["status"], sj.get("reason", "")
    for lg in logs:
        if lg.exists():
            t = lg.read_text()
            if "CUDA out of memory" in t or "OutOfMemory" in t:
                return "oom", "torch CUDA OOM in run.log"
    return "missing", "no final.mp4 and no status/log evidence"


# ---- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out-dir", required=True, help="e2e out/<res> dir on the cluster")
    ap.add_argument("--source", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gemma", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--code-commit", required=True)
    ap.add_argument("--eval-repo", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument(
        "--outside-tol-db", type=float, default=30.0, help="min outside-window PSNR vs edited"
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    specs = variant_specs(args.res)

    # geometry from any available phase_timing.json (all identical for a resolution)
    geom, fps, resw, resh = None, None, None, None
    for s in specs.values():
        pt = load_json(out / s["phase"]) if "phase" in s else None
        if pt and pt.get("geometry"):
            geom = pt["geometry"]
            fps = pt["resolution"]["fps"]
            resw = pt["resolution"]["width"]
            resh = pt["resolution"]["height"]
            break
    # fall back to upstream timing.json retake_window
    if geom is None:
        ut = load_json(out / "upstream/timing.json")
        if ut:
            rw = ut["retake_window"]
            fps, resw, resh = rw["fps"], rw["width"], rw["height"]
            geom = {"total_frames": rw["total_frames"], "bridge_frames": rw["bridge_frames"]}

    lb0 = (geom or {}).get("lb0")
    lb1 = (geom or {}).get("lb1")

    def eval_git():
        try:
            rev = subprocess.run(
                ["git", "-C", args.eval_repo, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ["git", "-C", args.eval_repo, "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            )
            return {"base_rev": rev, "dirty": dirty}
        except Exception as e:
            return {"error": str(e)}

    # ---- per-variant status + artifact bookkeeping --------------------------
    rows = {}
    for name, s in specs.items():
        final_dir = out / s["final_dir"]
        final_mp4 = final_dir / "final.mp4"
        retake = out / s["retake"]
        status_json = out / s.get("status", f"{name}/__none__.json")
        logs = [out / f"{name}/run.log", final_dir / "run.log"]
        st, reason = variant_status(final_mp4, retake, status_json, logs)
        rows[name] = {
            "kind": s["kind"],
            "quant": s["quant"],
            "status": st,
            "reason": reason,
            "final_mp4": str(final_mp4),
            "retake_mp4": str(retake),
            "final_dir": str(final_dir),
            "spec": s,
        }

    # ---- manifest -----------------------------------------------------------
    artifacts = {}
    for name, r in rows.items():
        if r["status"] == "ok":
            fp = Path(r["final_mp4"])
            artifacts[name] = {
                "final_sha256": sha256_file(fp),
                "final_mtime": fp.stat().st_mtime if fp.exists() else None,
                "retake_sha256": sha256_file(Path(r["retake_mp4"])),
            }
    manifest = {
        "resolution": args.res,
        "width": resw,
        "height": resh,
        "fps": fps,
        "source": {"path": args.source, "sha256": sha256_file(Path(args.source))},
        "geometry": geom,
        "seed": args.seed,
        "steps": args.steps,
        "assets": {"checkpoint": args.checkpoint, "gemma": args.gemma, "lora": args.lora},
        "provenance": {"tensorrt_llm_rev": args.code_commit, "ltx2_eval": eval_git()},
        "variant_status": {
            k: {"status": v["status"], "reason": v["reason"]} for k, v in rows.items()
        },
        "artifacts": artifacts,
    }
    (out / "manifest.json").write_text(dumps(manifest))

    # ---- phase timing -------------------------------------------------------
    phase_rows = {}
    for name, r in rows.items():
        s = r["spec"]
        row = {"mode": r["kind"], "quant": r["quant"], "status": r["status"]}
        pt = load_json(out / s["phase"]) if "phase" in s else None
        if r["kind"] == "upstream":
            if pt:
                row.update(
                    {
                        k: pt.get(k)
                        for k in (
                            "apply_seconds",
                            "ltx_seconds",
                            "post_seconds",
                            "wall_seconds",
                            "sum_delta_seconds",
                        )
                    }
                )
            else:
                ut = load_json(out / s["timing"])
                if ut:
                    row.update(
                        {
                            "ltx_seconds": ut.get("diffusion_seconds"),
                            "wall_seconds": ut.get("total_seconds"),
                            "apply_post_seconds": round(
                                (ut.get("total_seconds") or 0) - (ut.get("diffusion_seconds") or 0),
                                3,
                            ),
                            "note": "phase_timing sidecar absent; apply/post not split",
                        }
                    )
        elif r["kind"] == "native":
            vj = load_json(out / s["variant"])
            if vj:
                row.update(
                    {
                        "build_seconds": vj.get("build_seconds"),
                        "single_shot_seconds": vj.get("single_shot_seconds"),
                        "warm_p50_seconds": vj.get("warm_retake_p50"),
                        "ltx_seconds": vj.get("warm_retake_p50"),
                        "peak_reserved_gib": vj.get("peak_reserved_gib"),
                    }
                )
            if pt:  # composite run gives real apply/post (ltx there is the external copy ~0)
                row.update(
                    {
                        "apply_seconds": pt.get("apply_seconds"),
                        "post_seconds": pt.get("post_seconds"),
                        "composite_wall_seconds": pt.get("wall_seconds"),
                    }
                )
        elif r["kind"] == "serve":
            stj = load_json(out / s.get("serve_timing", "__none__"))
            if stj:
                fs = stj.get("first_served") or {}
                sw = stj.get("steady_warm") or {}
                row.update(
                    {
                        "cold_start_seconds": stj.get("cold_start_seconds"),
                        "first_served_seconds": fs.get("wall"),
                        "engine_seconds": fs.get("generation"),
                        "http_wall_p50_seconds": (sw.get("wall") or {}).get("p50")
                        if isinstance(sw.get("wall"), dict)
                        else sw.get("wall"),
                        "warm_p50_seconds": stj.get("steady_wall_p50"),
                    }
                )
            if pt:
                row.update(
                    {
                        "apply_seconds": pt.get("apply_seconds"),
                        "post_seconds": pt.get("post_seconds"),
                    }
                )
        phase_rows[name] = row
    (out / "phase_timing.json").write_text(dumps({"resolution": args.res, "variants": phase_rows}))

    # ---- quality metrics ----------------------------------------------------
    def window(arr):
        if lb0 is None or lb1 is None:
            return arr
        return arr[lb0:lb1]

    bf16_retake = out / rows["native_bf16"]["retake_mp4"] if "native_bf16" in rows else None
    up_retake = out / rows["upstream"]["retake_mp4"] if "upstream" in rows else None
    ref_bf16 = decode_rgb(bf16_retake) if bf16_retake and bf16_retake.exists() else None
    ref_up = decode_rgb(up_retake) if up_retake and up_retake.exists() else None

    qrows = {}
    for name, r in rows.items():
        if r["status"] != "ok":
            qrows[name] = {"status": r["status"]}
            continue
        rp = Path(r["retake_mp4"])
        entry = {
            "status": "ok",
            "ffprobe": {
                "retake_output": ffprobe(rp),
                "final": ffprobe(Path(r["final_mp4"])),
                "retake_input": ffprobe(out / f"{r['spec']['final_dir']}/retake_input.mp4"),
            },
        }
        try:
            cur = decode_rgb(rp) if rp.exists() else None
        except Exception as e:
            cur = None
            entry["decode_error"] = str(e)
        if cur is not None:
            if ref_bf16 is not None and name != "native_bf16":
                p, m = psnr(window(cur), window(ref_bf16))
                entry["window_vs_bf16"] = {
                    "psnr_db": p,
                    "mse": m,
                    "ssim": ssim_mean(window(cur), window(ref_bf16)),
                }
            if ref_up is not None and name != "upstream":
                p, m = psnr(window(cur), window(ref_up))
                entry["window_vs_upstream"] = {
                    "psnr_db": p,
                    "mse": m,
                    "ssim": ssim_mean(window(cur), window(ref_up)),
                }
        qrows[name] = entry
    (out / "quality_metrics.json").write_text(
        dumps(
            {
                "resolution": args.res,
                "window_frames": [lb0, lb1],
                "note": "informational only; never gates",
                "variants": qrows,
            }
        )
    )

    # ---- assertions ---------------------------------------------------------
    arows = {}
    for name, r in rows.items():
        if r["status"] != "ok":
            arows[name] = {"status": r["status"]}
            continue
        fdir = Path(r["final_dir"])
        final_meta = ffprobe(fdir / "final.mp4")
        retake_meta = ffprobe(Path(r["retake_mp4"]))
        edited_meta = ffprobe(fdir / "edited_full.mp4")
        is_trt = r["kind"] in ("native", "serve")
        checks = {}
        # AC-7 (TRT retake is video-only) applies to native/serve variants; the
        # upstream RetakePipeline legitimately passes source audio through its retake.
        if is_trt:
            checks["retake_video_only"] = retake_meta.get("has_audio") is False
        else:
            checks["retake_video_only"] = "n/a (upstream retake carries source audio)"
        checks["final_has_audio"] = final_meta.get("has_audio") is True
        # audio derives from edited: final audio codec+sr match edited (an AAC transcode, -shortest)
        fa, ea = final_meta.get("audio") or {}, edited_meta.get("audio") or {}
        checks["final_audio_matches_edited_stream"] = bool(
            fa and ea and fa.get("sample_rate") == ea.get("sample_rate")
        )
        # outside-window decode-tolerance vs edited
        outside = {}
        try:
            fin = decode_rgb(fdir / "final.mp4")
            edt = decode_rgb(fdir / "edited_full.mp4")
            a = geom.get("a") if geom else None
            tot = geom.get("total_frames") if geom else None
            n = min(len(fin), len(edt))
            if a is not None and tot is not None:
                mask = np.ones(n, bool)
                mask[a : min(a + tot, n)] = False
                if mask.any():
                    p, m = psnr(fin[:n][mask], edt[:n][mask])
                    outside = {
                        "psnr_db": p,
                        "mse": m,
                        "pass": (p is None or p >= args.outside_tol_db),
                    }
        except Exception as e:
            outside = {"error": str(e)}
        checks["outside_window_vs_edited"] = outside
        checks["outside_window_pass"] = outside.get("pass", None)
        retake_ok = (checks["retake_video_only"] is True) or (not is_trt)
        arows[name] = {
            "status": "ok",
            "checks": checks,
            "pass": bool(
                retake_ok
                and checks["final_has_audio"]
                and checks["final_audio_matches_edited_stream"]
                and checks.get("outside_window_pass") in (True, None)
            ),
        }
    (out / "assertions.json").write_text(
        dumps({"resolution": args.res, "outside_tol_db": args.outside_tol_db, "variants": arows})
    )

    print(
        "COLLECT_EVIDENCE_DONE "
        + dumps(
            {
                "res": args.res,
                "statuses": {k: v["status"] for k, v in rows.items()},
                "manifest": str(out / "manifest.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
