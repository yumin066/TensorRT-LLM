<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# delete_disfluency 端到端测试 — upstream vs TensorRT-LLM retake

> 由 `render_report.py` 从证据 JSON 自动生成(无手填);每个 AC 行状态由校验推导,任一缺口显示 `INCOMPLETE` + 产物路径。

**用例:** clip `--Y9imYnfBw-f0-484.mp4`,filler `[UH]`,窗口 total=209f,bridge=29,lb0/lb1=90/119,a=52,splice_frame=156。
**设备:** RTX PRO 6000 (sm_120, 96GB)。seed 42 · TRT-LLM `bcc0920033` · LTX2.3-eval `73c9d47b8b4f660dc88912c54ca474826d4120e1`(dirty=True, +`--external-retake` patch)。源 sha256 `dc6ab372e4ef5d3e…`

## 720p (1280×704, 209f)

| variant | status | APPLY | LTX | POST | single-shot | warm p50 | peak GiB | win PSNR/SSIM vs bf16 | assert |
|---|---|---|---|---|---|---|---|---|---|
| upstream | ok | 2.2 | 63.72 | 4.56 | — | — | — | 27.51 / 0.92 | True |
| native_bf16 | ok | 2.22 | 34.5 | 4.32 | 131.2 | 34.5 | 92.82 | — | True |
| native_fp8 | ok | 2.2 | 30.89 | 4.33 | 118.47 | 30.89 | 76.97 | 35.01 / 0.97 | True |
| native_nvfp4 | ok | 2.25 | 25.32 | 4.41 | 105.61 | 25.32 | 69.29 | 9.11 / 0.14 | True |
| serve_bf16 | ok | 2.29 | 31.37 | 4.38 | — | 37.13 | — | 32.64 / 0.97 | True |
| serve_fp8 | unsupported | — | — | — | — | — | — | — | — |
| serve_nvfp4 | unsupported | — | — | — | — | — | — | — | — |

serve bf16 明细: cold-start 138.03s · first-served 36.87s · steady wall p50 37.13s / engine p50 31.37s。

## 1080p (1920×1088, 209f)

| variant | status | APPLY | LTX | POST | single-shot | warm p50 | peak GiB | win PSNR/SSIM vs bf16 | assert |
|---|---|---|---|---|---|---|---|---|---|
| upstream | ok | 3.35 | 156.14 | 7.96 | — | — | — | — | True |
| native_bf16 | oom | — | — | — | — | — | — | — | — |
| native_fp8 | oom | — | — | — | — | — | — | — | — |
| native_nvfp4 | ok | 3.4 | 95.78 | 7.2 | 166.99 | 95.78 | 88.06 | — | True |

*serve 仅 720p 表征。serve FP8/NVFP4 = `unsupported`,证据 `artifacts/720p/serve_{fp8,nvfp4}/status.json`+`run.log`:`Unknown pipeline_config keys for LTX2Pipeline (/home/scratch.minyu_gpu/ltx-retake-assets/ltx2-22b-distilled): …`*

## fast-init 安全性(每交付 quant)

- **bf16**: touched 1061 nn.init 张量, uncovered_nan=0 → `fast_init_safe=True` (`artifacts/init_coverage/init_coverage_bf16.json`)
- **fp8**: touched 1061 nn.init 张量, uncovered_nan=0 → `fast_init_safe=True` (`artifacts/init_coverage/init_coverage_fp8.json`)
- **nvfp4**: touched 1060 nn.init 张量, uncovered_nan=0 → `fast_init_safe=True` (`artifacts/init_coverage/init_coverage_nvfp4.json`)

## 校验

- **host sha256 校验** (`validate_artifacts.py`): checked 14, ok=True
- serve 计时字段齐全: True · 音频断言全过: True · 三 quant 覆盖: True · 1080p OOM 日志: True

## AC 覆盖(状态由校验推导 → 产物)

| AC | 状态 | 证据产物 |
|---|---|---|
| AC-1 upstream 基线+阶段图+manifest | ok | `{720p,1080p}/upstream/final.mp4`+`manifest.json`+`phase_timing.json` |
| AC-2 --external-retake,默认路径不变 | ok | patch + 默认 timing.json 无 retake_source(GPU 验证) |
| AC-3 接缝预检 | ok | composite run.log `external retake accepted` |
| AC-4 每变体 final 或 oom/unsupported(带日志) | ok | 上表 status;1080p oom `native_{bf16,fp8}/status.json`+`run.log`;serve fp8/nvfp4 status.json |
| AC-5 阶段图+单发+warm+显存+serve 拆分 | ok | `phase_timing.json`(apply/ltx/post/single_shot + serve cold/first/warm/engine) |
| AC-6 窗口 PSNR/SSIM + frame grid + 窗外一致 | ok | `quality_metrics.json`(PSNR+SSIM vs bf16 & upstream)+`assertions.json` outside + `frame_grid_720p_t5.0.png` |
| AC-7 final 音频源自 edited(内容比对) | ok | `assertions.json` audio_similarity(dur_delta + correlation) |
| AC-8 --external-retake GPU 跑一次 | ok | composite run.log(smc521ge-0038 / ltx_r35) |
| AC-9 产物拉回 host + gitignore + sha 一致 | ok | host `artifacts/` 树;`validate_artifacts.py` ok=True;`/artifacts/` gitignored |

## 公平性(task12)

- upstream 720p warm LTX 63.72s(冷 NFS 首跑曾 750s)。合法对比:native single_shot vs upstream(含加载);native warm vs serve engine。不得下 46× 结论。
- FP8 近无损、NVFP4 明显改变(见质量列),速度不得脱离质量呈现。
