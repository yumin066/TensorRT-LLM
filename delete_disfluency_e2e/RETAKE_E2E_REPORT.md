<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# delete_disfluency 端到端测试 — upstream vs TensorRT-LLM retake

> 本报告由 `render_report.py` 从证据 JSON(manifest / phase_timing / quality_metrics /
> assertions / init_coverage)自动生成,无手填数字。每个 AC 行都指向 host `artifacts/` 下的产物。

**用例:** clip `--Y9imYnfBw-f0-484.mp4`,filler `[UH]`,retake 窗口 total=209 帧,bridge=29,lb0/lb1=90/119,a=52,splice_frame=156。
**设备:** RTX PRO 6000 Blackwell (sm_120, 96 GB)。**seed** 42 · **TRT-LLM rev** `92bd8613cb` · **LTX2.3-eval** `73c9d47b8b4f660dc88912c54ca474826d4120e1` (dirty=True, 加 `--external-retake` patch)。
**源 sha256** `dc6ab372e4ef5d3e…`

## 720p (1280×704, 209f)

| variant | status | APPLY | LTX | POST | warm p50 | peak infer GiB | win PSNR vs bf16 | SSIM | assert |
|---|---|---|---|---|---|---|---|---|---|
| upstream | ok | 2.2 | 63.72 | 4.56 | — | — | 27.51 | 0.92 | True |
| native_bf16 | ok | 2.22 | 34.5 | 4.32 | 34.5 | 92.82 | — | — | True |
| native_fp8 | ok | 2.2 | 30.89 | 4.33 | 30.89 | 76.97 | 35.01 | 0.97 | True |
| native_nvfp4 | ok | 2.25 | 25.32 | 4.41 | 25.32 | 69.29 | 9.11 | 0.14 | True |
| serve_bf16 | ok | 2.21 | 31.0 (engine) | 4.31 | — | — | 32.64 | 0.97 | True |
| serve_fp8 | unsupported | — | — | — | — | — | — | — | — |
| serve_nvfp4 | unsupported | — | — | — | — | — | — | — | — |

## 1080p (1920×1088, 209f)

| variant | status | APPLY | LTX | POST | warm p50 | peak infer GiB | win PSNR vs bf16 | SSIM | assert |
|---|---|---|---|---|---|---|---|---|---|
| upstream | ok | 3.35 | 156.14 | 7.96 | — | — | — | — | True |
| native_bf16 | oom | — | — | — | — | — | — | — | — |
| native_fp8 | oom | — | — | — | — | — | — | — | — |
| native_nvfp4 | ok | 3.4 | 95.78 | 7.2 | 95.78 | 88.06 | — | — | True |

*serve(HTTP Mode B)只在 720p 表征;serve FP8/NVFP4 = `unsupported`,证据:`artifacts/720p/serve_{fp8,nvfp4}/status.json` + `run.log`,配置报错 `Unknown pipeline_config keys for LTX2Pipeline (/home/scratch.minyu_gpu/ltx-retake-assets/ltx2-22b-distilled): …`*

## fast-init 安全性证明

`artifacts/init_coverage/init_coverage.json`:fast-init 跳过(NaN-poison)**1061** 个 nn.init 张量,加载 checkpoint 后 **uncovered_nan_count=0** → `fast_init_safe=True`。即每个被 fast-init 跳过的张量都被 checkpoint 覆盖;交付物用 fast-init(fp8/nvfp4)因此被证明安全。

## AC 覆盖(每行 → 状态 → 产物)

| AC | 状态 | 证据产物 |
|---|---|---|
| AC-1 upstream 基线 + 阶段图 + manifest | ok(720p+1080p) | `artifacts/{720p,1080p}/upstream/final.mp4` + `manifest.json` + `phase_timing.json` |
| AC-2 `--external-retake` flag,默认路径不变 | ok | patch `delete_disfluency_external_retake.patch`;默认 `timing.json` 无 `retake_source`(GPU 验证) |
| AC-3 接缝预检(恰好 total 帧/分辨率/fps) | ok | 各 composite `run.log` 的 `external retake accepted` |
| AC-4 每变体产 final 或记 oom/unsupported | ok | 见上表 status 列;1080p bf16/fp8=oom;serve fp8/nvfp4=unsupported(带 status.json) |
| AC-5 §2 阶段图 + 单发/warm + 显存 + 状态 | ok | `phase_timing.json`(apply/ltx/post + serve engine/first-served)+ `quality_metrics.json` ffprobe |
| AC-6 窗口 PSNR/SSIM(信息性)+ frame grid + 窗外一致 | ok | `quality_metrics.json`(PSNR+SSIM vs bf16 & upstream)+ `assertions.json` 窗外 + `frame_grid_720p_t5.0.png` |
| AC-7 final 音频源自 edited,retake 视频-only | ok | `assertions.json`(retake_video_only / final_has_audio / audio-vs-edited) |
| AC-8 `--external-retake` GPU 跑一次 | ok | composite `run.log`(节点 smc521ge-0038 / 容器 ltx_r35) |
| AC-9 产物拉回 host artifacts/ + gitignore | ok | 本 host `artifacts/` 树;`/artifacts/` 已 gitignore |

## 公平性说明(task12,Codex needs-caveats)

- upstream 720p LTX(warm cache 重跑)= 63.717s(冷 NFS 首跑曾 750s,已用暖 cache 重测)。
- 合法对比:①含加载 native single_shot vs upstream(warm);②常驻 native warm vs serve engine。**不得**把 upstream 与 native warm 直接下 46× 结论。
- FP8 近无损(PSNR/SSIM 见表),NVFP4 明显改变(质量警示),二者不得脱离质量呈现速度。
