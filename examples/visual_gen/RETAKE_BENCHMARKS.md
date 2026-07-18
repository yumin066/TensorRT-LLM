<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LTX-2 retake — benchmark & measurement harness index

The native LTX-2.3 **retake** workload regenerates an arbitrary internal
time-window of a source video (the non-window frames stay byte-identical) on
TensorRT-LLM's native `LTXModel` path, inheriting the config-driven accel stack
(FP8/NVFP4 dynamic quant, FA4/CuTeDSL attention, CUDA graphs, torch.compile).

These `examples/visual_gen/ltx2_retake_*.py` scripts are the checked-in
measurement harnesses used to characterize that workload. They share a house
design: **stdlib-only pure helpers** (percentile/summary, delta math, status
classification, JSON sanitize — unit-tested on any host with no torch/GPU) plus a
**lazily-imported heavy path** that builds the 22b and runs on GPU. Near-capacity
runs use **subprocess-per-item isolation** so one OOM never poisons the sweep, and
every artifact is emitted as strict RFC-8259 JSON (`allow_nan=False`).

> See the polished results write-up in
> [`RETAKE_ACCELERATION_REPORT.md`](RETAKE_ACCELERATION_REPORT.md).

## Common arguments

Most harnesses share the same asset/window flags:

```bash
--checkpoint <ltx-2.3-22b-distilled.safetensors>   # native LTX-2 checkpoint
--gemma      <gemma-qat>                            # text encoder
--lora       <talkvid-id-lora.safetensors>          # retake identity/style LoRA
--source     <source.mp4>                           # source video (8k+1 frames, /32 spatial)
--output-dir <dir>                                  # artifact + JSON output
--start 1.0 --end 2.0                               # retake window (seconds)
--seed 42 --steps 8                                 # determinism + denoise steps
```

Each writes `<output-dir>/<name>.json` and prints a `..._DONE {...}` line on
success. The pure helpers are covered by
`tests/unittest/_torch/visual_gen/test_ltx2_retake_*.py` (run on any host).

---

## 1. Correctness & quality oracle

| Harness | What it does |
|---------|--------------|
| **`ltx2_retake_oracle.py`** | Native-vs-upstream comparison oracle. Runs the native retake and the upstream `DiffusionStage.run` on the same source/window/seed, re-checks the functional hard invariants (frame/window index, two-sided mask, byte-identical composite, duration/FPS/shape, source-audio unchanged, seed-42 determinism, `regenerate_audio=True` fail-fast), and computes **informational** PSNR/SSIM (never gates). Emits `protocol.json` + `metrics.json` + `manifest.json` + `.mp4`/`.pt`; `--verify-local <dir>` re-validates a pulled package stdlib-only. Reused as the shared `build_pipeline` / `_run_pipeline` by every other harness. |

## 2. Latency — Mode A (every-rebuild) vs Mode B (resident)

| Harness | What it does |
|---------|--------------|
| **`ltx2_retake_mode_a_timing.py`** | **Mode A** reference: the upstream every-rebuild path — each call rebuilds the pipeline in its own subprocess, runs, and exits (full GPU release). Reports `model_build_load` + `run_total` + `total` p50/p90/min and a `../LTX2.3-eval/packages/` pristine check. |
| **`ltx2_retake_timing.py`** | **Mode B** pipeline-direct: load-once resident worker, N warmup + M measured retakes; staged CUDA-event timing (source_read / vae_encode / conditioning / denoise_total / denoise_per_step / vae_decode / composite) + a cold→first→warm timeline + explicit `cpu_io`. |
| **`ltx2_retake_serve_timing.py`** | **Mode B** production surface: launches one `trtllm-serve` retake worker, sends warmup+measured HTTP `POST /v1/videos/generations`, parses `Server-Timing` (`generation`/`denoise`), first-served vs steady p50/p90/min under a strict success gate (both metrics positive; whole-run gate). |

## 3. Acceleration axes

| Harness | What it does |
|---------|--------------|
| **`ltx2_retake_smoke.py`** | Early per-axis feasibility matrix {bf16/VANILLA, bf16/FA4, bf16/CUDAGraph, NVFP4/VANILLA} with capability precheck, per-config status/skip-reason, and informational quality vs the bf16/VANILLA baseline. Success is anchored on the baseline running `ok`. |
| **`ltx2_retake_accel_gating.py`** | Authoritative AC-4 gating: bf16/VANILLA/no-graph baseline → single-axis increments (torch_compile, cuda_graph, attention backend, NVFP4) → the fixed stack order ① compile → ② cuda_graph → ③ fastest attn → ④ NVFP4, each row carrying status + skip reason. |
| **`ltx2_retake_compile_cost.py`** | torch.compile / autotune / graph-capture cost split across three cache modes: empty-cache first process, same-process later request, warm-disk-cache new process — keeping one-time costs out of steady warm. |
| **`ltx2_retake_attn_profile.py`** | Attention-backend profiler: VANILLA vs FA4 vs CUTEDSL per-backend denoise timing + the **profiler-confirmed attention share** of denoise GPU time (from an `nsys cuda_gpu_kern_sum` CSV, capture-range = denoise only), CUTEDSL head_dim/dtype gate, quant-attention-NotImplemented note. |

## 4. Quantization

| Harness | What it does |
|---------|--------------|
| **`ltx2_retake_quant_mem.py`** | bf16 vs FP8(dynamic) vs NVFP4(dynamic): per-mode warm denoise p50/p90/min + peak GPU memory (allocated+reserved) for model-load and inference + steady resident + informational PSNR/SSIM of the regenerated window vs bf16. |

## 5. Device × resolution

| Harness | What it does |
|---------|--------------|
| **`ltx2_retake_resolution_baseline.py`** | RTX PRO 6000 device baseline across 512×320 / 1280×704 (~720p) / 1920×1088 (~1080p): per-resolution cold (build/load + first-inference) + warm denoise + peak memory; validates the VAE shape constraints (`8k+1` frames, `/32` spatial), synthesizes higher-res sources (cv2), and retries under FP8 if bf16 OOMs. |

## 6. Reporting & aggregation

| Harness | What it does |
|---------|--------------|
| **`ltx2_retake_report_schema.py`** | Unified per-row report schema: aggregates the raw artifacts into one `report.json` with cold/warm blocks, shape-manifest hash, normalized config/memory/quality/status/failure/env + two-repo provenance, and a compile-cost block on torch_compile rows. |
| **`ltx2_retake_final_matrix.py`** | Merged **customer matrix**: reads the pulled artifacts from `--artifacts-dir` → `final_matrix.json` (strict) + `final_matrix.md` covering Mode A/B latency + amortization break-even, acceleration axes, quant trade-off, and device×resolution, plus a production recommendation. A missing artifact is a recorded gap, not a crash. |

---

## Running order (typical study)

1. `ltx2_retake_oracle.py` — establish correctness + the informational quality anchor.
2. `ltx2_retake_mode_a_timing.py` + `ltx2_retake_timing.py` + `ltx2_retake_serve_timing.py` — Mode A vs Mode B latency.
3. `ltx2_retake_smoke.py` → `ltx2_retake_accel_gating.py` + `ltx2_retake_compile_cost.py` + `ltx2_retake_attn_profile.py` — acceleration axes.
4. `ltx2_retake_quant_mem.py` — quantization latency/memory/quality.
5. `ltx2_retake_resolution_baseline.py` — device × resolution.
6. `ltx2_retake_final_matrix.py --artifacts-dir <pulled artifacts>` — merge everything into the customer matrix.
