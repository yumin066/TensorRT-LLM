<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LTX-2.3 retake on TensorRT-LLM — acceleration report

**Workload.** "Retake" regenerates an arbitrary internal time-window of a source
video (e.g. re-say a line in a talking-head clip) while every frame outside the
window stays byte-identical to the source. This report characterizes the workload
after migrating it onto TensorRT-LLM's **native `LTXModel`** path, which inherits
the config-driven acceleration stack (FP8/NVFP4 dynamic quantization, FA4/CuTeDSL
attention, CUDA graphs, torch.compile).

**Model / device.** LTX-2.3 22b distilled, bf16, 8-step distilled denoise, with
the TalkVid identity LoRA fused. All numbers measured on a single **NVIDIA RTX PRO
6000 Blackwell Server Edition (sm_120, 96 GB)**. Source clip 512×320, 89 frames,
retake window [1.0 s, 2.0 s], seed 42.

---

## TL;DR

- **Resident serving is ~46× faster per retake** than the every-rebuild baseline
  (**1.82 s** warm vs **83.3 s**), and pays back its one-time load after **~2
  calls**.
- **FP8 dynamic quantization is a free win**: 1.28× faster, 17.7 GiB less memory,
  visually near-lossless.
- **The 96 GB device is what unlocks native 1080p retake** (it reserves 91.5 GiB —
  right up against the ceiling).
- The attention backend is **not** a useful tuning knob for this workload
  (attention is ~7% of compute).

---

## 1. Production warm latency — resident serving vs every-rebuild

| Serving mode | Latency per retake |
|--------------|--------------------|
| Every-rebuild (rebuild the pipeline for each call) | **83.3 s** |
| Resident worker — pipeline-direct, warm | **1.82 s** |
| Resident worker — `trtllm-serve` over HTTP, warm | **2.12 s** wall (1.74 s engine) |

A resident worker pays a **one-time ~88 s cold load**, then serves each retake in
~1.8 s. Because every-rebuild pays the full ~83 s on *every* call, the resident
worker overtakes it almost immediately:

| Calls served | Every-rebuild | Resident (Mode B) | Speedup |
|-------------:|--------------:|------------------:|--------:|
| 1 | 83 s | ~90 s | 0.9× |
| **2 (break-even)** | 167 s | 92 s | **1.8×** |
| 10 | 833 s | 106 s | 7.8× |
| 100 | 8,330 s | 270 s | **30.9×** |

**Per-call steady-state speedup: ~45.8×.** The HTTP surface adds ~0.3 s of
transport/serialization over the engine time — report both depending on whether
the client measures end-to-end wall or engine latency.

## 2. Device leverage — resolution scaling (customer value: RTX PRO 6000)

| Resolution | Latent tokens | Warm denoise | Peak reserved memory |
|------------|--------------:|-------------:|---------------------:|
| 512×320 | 1.0× | **1.23 s** | 67.5 GiB |
| 1280×704 (~720p) | 5.5× | **9.6 s** | 77.0 GiB |
| 1920×1088 (~1080p) | 12.75× | **29.5 s** | **91.5 GiB** |

All three run natively in bf16. At 1080p the retake reserves **91.5 GiB of the
95 GiB** available — so the **96 GB RTX PRO 6000 is precisely what makes native
1080p retake feasible without offload**. FP8 (below) frees ~18 GiB, giving
comfortable 1080p headroom or room for larger batches/longer windows.

## 3. Quantization trade-off

| Mode | Denoise speedup | Resident memory | Memory saved | Regenerated-window fidelity vs bf16 |
|------|----------------:|----------------:|-------------:|-------------------------------------|
| bf16 | 1.00× | 65.4 GiB | — | reference |
| **FP8** (dynamic) | **1.28×** | 47.7 GiB | **−17.7 GiB** | **near-lossless** (PSNR 36.5, SSIM 0.984) |
| **NVFP4** (dynamic) | **1.99×** | 40.0 GiB | **−25.4 GiB** | visibly changed (PSNR 10.1, SSIM 0.228) |

- **FP8** is the recommended default: substantial latency + memory wins with
  essentially no visible quality change on the regenerated window.
- **NVFP4** gives the largest latency/memory reduction, but the 4-bit weight cast
  materially changes the regenerated content — use it only where visual drift is
  acceptable (exploratory / latency-or-memory-critical serving). Non-window frames
  remain byte-identical to the source regardless (they are composited, not
  regenerated).

## 4. Acceleration axes

- **torch.compile** and **CUDA graphs** both apply to the native retake path and
  are recommended for production (CUDA graphs required a per-step conditioning fix
  to compose with the model's prompt-adaptive normalization).
- **Attention backend is not a lever.** Profiling shows attention is only **~7.4 %**
  of denoise GPU time — an Amdahl ceiling of ~1.08× from any attention swap. In
  addition, on this audio-bearing model FA4 is not yet wired for the retake
  forward and CuTeDSL is ineligible (its cubins need head_dim 128; the audio
  attention is head_dim 64). **Keep the default SDPA attention** and spend tuning
  effort on the ~93 % that is linear/projection compute (quantization, compile,
  graph capture).

## 5. Correctness

The migrated native retake passes the functional hard gate on the real 22b:
frame/window indexing, two-sided conditioning mask, **byte-identical composite** of
the non-window frames, duration/FPS/shape preservation, **source audio unchanged**,
seed-42 determinism, and fail-fast on unsupported audio regeneration. PSNR/SSIM vs
the upstream reference are reported as *informational* signals only and never gate
correctness.

---

## Production recommendation

Deploy the native retake as a **resident TensorRT-LLM worker** (`trtllm-serve`) —
it amortizes its ~88 s load after ~2 calls and then serves each retake ~45× faster
than every-rebuild. Use **FP8 dynamic quantization** as the default (near-lossless,
1.28× faster, 18 GiB lighter — and the headroom that makes 1080p robust); reserve
**NVFP4** for latency/memory-critical or exploratory use where regenerated-window
drift is acceptable. Enable **torch.compile** and **CUDA graphs**; keep the default
**SDPA attention**. The **96 GB RTX PRO 6000 Blackwell** is the enabling device —
it covers 512p/720p comfortably and is what makes native 1080p retake possible.

---

## Methodology & reproducibility

Every number above is produced by a checked-in, unit-tested harness and validated
by a real 22b GPU run; see [`RETAKE_BENCHMARKS.md`](RETAKE_BENCHMARKS.md) for the
harness index and how to reproduce each row. Latency is warm-steady p50 over
2 warmup + 6–8 measured retakes (load-once); memory is `torch.cuda` peak
allocated/reserved; the attention share is from an `nsys` kernel-time summary
captured over the denoise only; quality columns are informational PSNR/SSIM of the
regenerated window vs the bf16 baseline. Results reflect the LTX-2.3 22b distilled
model at the stated window/seed on one RTX PRO 6000 Blackwell (96 GB); absolute
latencies scale with resolution, window length, and step count.
