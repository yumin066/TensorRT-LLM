<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# delete_disfluency 端到端测试 — upstream vs TensorRT-LLM retake

> 由 `render_report.py` 从证据 JSON 自动生成(无手填);每个准则行状态由校验推导,任一缺口显示 `INCOMPLETE` + 产物路径。

**用例:** clip `--Y9imYnfBw-f0-484.mp4`,filler `[UH]`,窗口 total=209f,bridge=29,lb0/lb1=90/119,a=52,splice_frame=156。
**设备:** RTX PRO 6000 (sm_120, 96GB)。seed 42 · TRT-LLM `b38c3d1a69` · LTX2.3-eval `73c9d47b8b4f660dc88912c54ca474826d4120e1`(dirty=True, +`--external-retake` patch)。源 sha256 `dc6ab372e4ef5d3e…`

## 720p (1280×704, 209f)

| variant | status | APPLY | LTX | POST | single-shot | warm p50 | peak GiB | win PSNR/SSIM vs upstream | assert |
|---|---|---|---|---|---|---|---|---|---|
| upstream | ok | 2.23 | 61.92 | 4.52 | 68.66 | — | 94.13 | 基线 | True |
| native_bf16 | ok | 2.22 | 34.5 | 4.32 | 131.2 | 34.5 | 92.82 | 27.51 / 0.92 | True |
| native_fp8 | ok | 2.2 | 30.89 | 4.33 | 118.47 | 30.89 | 76.97 | 27.56 / 0.93 | True |
| native_nvfp4 | ok | 2.25 | 25.32 | 4.41 | 105.61 | 25.32 | 69.29 | 9.09 / 0.14 | True |
| serve_bf16 | ok | 2.27 | 31.37 | 4.31 | 182.12 | 37.13 | 94.89 | 28.67 / 0.94 | True |
| serve_fp8 | unsupported | — | — | — | — | — | — | — | — |
| serve_nvfp4 | unsupported | — | — | — | — | — | — | — | — |

serve bf16 明细: cold-start 138.03s · first-served 36.87s · steady wall p50 37.13s / engine p50 31.37s。

## 1080p (1920×1088, 209f)

| variant | status | APPLY | LTX | POST | single-shot | warm p50 | peak GiB | win PSNR/SSIM vs upstream | assert |
|---|---|---|---|---|---|---|---|---|---|
| upstream | ok | 3.3 | 155.68 | 8.23 | 167.21 | — | 94.22 | 基线 | True |
| native_bf16 | oom | — | — | — | — | — | — | — | — |
| native_fp8 | oom | — | — | — | — | — | — | — | — |
| native_nvfp4 | ok | 3.4 | 95.78 | 7.2 | 166.99 | 95.78 | 88.06 | 8.23 / 0.35 | True |

*serve 仅 720p 表征。serve FP8/NVFP4 = `unsupported`,证据 `artifacts/720p/serve_{fp8,nvfp4}/status.json`+`run.log`:`Unknown pipeline_config keys for LTX2Pipeline (/home/scratch.minyu_gpu/ltx-retake-assets/ltx2-22b-distilled): …`*

## LTX 阶段图细分(§2 对齐 · 720p · 均为常驻/warm 纯计算,model_load 单列)

| LTX 子阶段 | upstream(fp8-cast,常驻) | native FP8(常驻 warm) | 说明 |
|---|---|---|---|
| subprocess_start | 18.23 | 3.49 | python + torch/CUDA import(一次性) |
| model_load | 3.82 | ≈88(一次性¹) | 22B transformer + VAE + Gemma 载入显存(H2D) |
| video_vae_encode | 6.04 | 1.88 | 条件视频 → latent |
| audio_vae_encode | 0.13 | N/A | 编辑音频 → latent(native 纯视频 → N/A) |
| text_encode | 7.27 | 0.18 | Gemma,近固定 prompt |
| diffusion | 39.8 | 24.22 | **瓶颈**:8 步 Euler 去噪 |
| video_vae_decode | 0.18 | 3.7 | latent → 像素(upstream 流式,折入 mp4) |
| audio_vae_decode | 0.7 | N/A | latent → 波形(native 纯视频 → N/A) |
| mp4_encode | 5.43 | 1.47 | libx264 编码 |
| **LTX wall(warm)** | **59.55** | **30.94** | 常驻纯计算,不含 model_load |

*单位秒,均为常驻/warm 纯计算(通过 warmup 触发惰性权重加载后再测量,`model_load` 已单列、不计入 LTX wall)。**diffusion(8 步去噪)是主瓶颈**:upstream fp8-cast 39.8s vs native **真 FP8** 24.2s —— 真 FP8 量化比 fp8-cast 约 1.6× 快(用硬件 FP8 张量核)。¹native model_load 仅 server 启动时一次(此次 NFS 冷启容器测得 760s,页缓存热约 88s);upstream 若每次重建,则每次额外付 model_load 3.8s + subprocess。upstream video/audio VAE 解码流式化,真实开销折入 `mp4_encode`;native 纯视频路径无 audio VAE。设备 RTX PRO 6000(非 §2 的 H100)。*

## fast-init 安全性(每交付 quant)

- **bf16**: touched 1061 nn.init 张量, uncovered_nan=0 → `fast_init_safe=True` (`artifacts/init_coverage/init_coverage_bf16.json`)
- **fp8**: touched 1061 nn.init 张量, uncovered_nan=0 → `fast_init_safe=True` (`artifacts/init_coverage/init_coverage_fp8.json`)
- **nvfp4**: touched 1060 nn.init 张量, uncovered_nan=0 → `fast_init_safe=True` (`artifacts/init_coverage/init_coverage_nvfp4.json`)

## 校验(单一 `validate_evidence` 汇总,报告与 CLI 共用)

- 全量校验 ok=True · host sha256=True · frame grid=True · 三 quant 覆盖=True · gitignore=True

## 验收准则覆盖(每行状态均由 `validate_evidence` 推导 → 产物)

| 准则 | 状态 | 证据产物 |
|---|---|---|
| AC-1 upstream 基线+阶段图+manifest | ok | `{720p,1080p}/upstream/final.mp4`+`manifest.json`+`phase_timing.json`(hash+timing 校验) |
| AC-2 --external-retake,默认路径不变 | ok | patch 存在 + 默认 upstream `timing.json` 无 retake_source |
| AC-3 接缝预检 | ok | composite `run.log` `external retake accepted` |
| AC-4 每变体 final 或 oom/unsupported(带日志) | ok | 上表 status;1080p oom `native_{bf16,fp8}/status.json`+`run.log`;serve fp8/nvfp4 status.json |
| AC-5 阶段图+单发+warm+显存+serve 拆分 | ok | `phase_timing.json` 每 ok 行 apply/ltx/post/single_shot/peak 齐全 + sum≈wall;serve cold/first/warm/engine |
| AC-6 窗口 PSNR/SSIM + frame grid + 窗外一致 | ok | `quality_metrics.json`+`assertions.json` outside + `frame_grid_720p_t5.0.png`(存在校验) |
| AC-7 final 音频源自 edited(内容比对) | ok | `assertions.json` audio_similarity(dur_delta + correlation) |
| AC-8 --external-retake GPU 跑一次 | ok | composite `run.log`(smc521ge-0038 / ltx_r35) |
| AC-9 产物拉回 host + gitignore + sha 一致 | ok | host `artifacts/` 树;`validate_evidence` hash_ok+gitignore |

## 公平性

- upstream 720p warm LTX 61.92s(冷 NFS 首跑曾 750s)。合法对比:native single_shot vs upstream(含加载);native warm vs serve engine。不得下 46× 结论。
- 质量列以 upstream retake 为共同基线(vs upstream)。native bf16 与 FP8 与 upstream 的差距基本持平(~27.5 dB)——FP8 相对 bf16 无额外量化损失;NVFP4 窗口塌(~9 dB)。速度不得脱离质量呈现。
