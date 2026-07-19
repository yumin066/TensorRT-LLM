<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# delete_disfluency 端到端测试 — upstream vs TensorRT-LLM retake

**用例:** `LTX2.3-eval/eval_cases.json[0]`,clip `--Y9imYnfBw-f0-484.mp4`(Bill Gates,29.97fps),
filler `[UH]` @ `[5.20, 5.54]s`。retake 窗口 total=209 帧,bridge=29 帧,retake=[3.003,3.971]s。
**设备:** 单张 RTX PRO 6000 Blackwell (sm_120, 96 GB),节点 smc521ge-0038 / 容器 ltx_r35(TRT)+ ltx_upstream(upstream)。
**代码:** TensorRT-LLM `a340b10af1`;唯一 upstream 改动 = `delete_disfluency.py` 的附加 `--external-retake` flag。

所有变体都产出了一个完整、可审阅的 `final.mp4`(去 disfluency + retake 拼回),或如实记为 `oom`/`unsupported`。

---

## 1. 完整矩阵

### 720p (1280×704, 209 帧)

| 变体 | 形态 | final.mp4 | build | 单发(含build) | warm p50 | peak infer 显存 | 窗口质量 vs bf16 |
|------|------|:---------:|------:|------:|------:|------:|------|
| **upstream** | in-process (RetakePipeline, fp8-cast) | ✅ | — | retake 750.4s* | — | — | 参考(独立实现) |
| **native bf16** | in-process | ✅ | 97.7s | 131.2s | 34.5s | **92.8 GiB** | 参考 |
| **native FP8** | in-process | ✅ | 88.1s | 118.5s | 30.9s | 77.0 GiB | **31.8 dB(近无损)** |
| **native NVFP4** | in-process | ✅ | 79.7s | 105.6s | 25.3s | 69.3 GiB | **9.1 dB(明显改变)** |
| **serve bf16** | HTTP (trtllm-serve) | 视频=native bf16 | — | 首发 wall 36.9s | wall p50 **37.1s** / engine 31.4s | ~同 native | — |
| serve FP8 / NVFP4 | HTTP | **unsupported** | — | — | — | — | serve `pipeline_config` 无量化键;开启需改 `tensorrt_llm/`(超范围) |

\* upstream 720p 的 750s 被**冷 NFS 加载**主导(首个 upstream 进程,page cache 冷)。见 §3 公平性说明。

### 1080p (1920×1088, 209 帧)

| 变体 | final.mp4 | build | 单发 | warm p50 | peak infer | 说明 |
|------|:---------:|------:|------:|------:|------:|------|
| **upstream**(offload cpu) | ✅ | — | retake 156.2s | — | — | offload 流式,GPU 常驻低 |
| native NVFP4 | ✅ | 74.6s | 167.0s | 95.8s | **88.1 GiB** | 唯一放得下的 native 模式 |
| native FP8 | **oom** | — | — | — | 需 >95 GiB(89.7 已用 + 6.5 分配失败) | 209 帧 1080p 超 96GB |
| native bf16 | **oom** | — | — | — | >95 GiB | 最重,必然 OOM |

**关键发现:** 参考报告的 1080p 是 89 帧(91.5GB);本真实用例是 **209 帧**(2.3×),导致 1080p 下
bf16/FP8 均 OOM,只有 **NVFP4(4-bit 权重腾出显存)在 88GB 挤进 96GB**。即 96GB 卡的 native 1080p
在此窗口长度下只对 NVFP4 可行;更短窗口(~89f)才如报告所述 bf16 可行。

---

## 2. 客户价值对比(retake 速度)

- **native/serve warm retake ≈ 25–37s**(720p 209f)vs **upstream every-rebuild**(含每次重建加载)。
- 量化权衡(720p warm p50):bf16 34.5s → **FP8 30.9s(1.12×,近无损,省 16GiB)** → **NVFP4 25.3s(1.36×,省 23GiB,但窗口明显改变)**。
- serve(Mode B, HTTP 常驻):engine 31.4s ≈ native bf16 warm 34.5s;HTTP 传输 209 帧张量额外 ~6s(wall 37.1s)。
- **FP8 是推荐默认**:近无损(31.8dB)+ 更省显存 + 略快;**NVFP4** 仅在显存/延迟紧张且可接受漂移时用(9.1dB)。

---

## 3. 正确性与公平性

- **接缝(AC-2/AC-3):** `--external-retake` 跳过 step-5 RetakePipeline,预检要求外部 mp4 **恰好 209 帧 / 1280×704(或1080)/ fps 匹配 / 可解码**,再走 upstream `composite_bridge` 拼整窗 `[0,total)`。所有 TRT 变体 `[retake] external retake accepted`。
- **窗外一致(AC-6):** native oracle `composite_outside_byte_identical: true`;final.mp4 是 H.264 CRF-16 重编码,窗外 vs edited 解码容差 ~44dB。
- **音频(AC-7):** 每个 TRT `retake_output.mp4` **无音频流**;final 音频 = edited 的 AAC 转码(`-map 1:a`)。
- **公平性(AC-5, task12):** upstream 720p 的 750s 含冷 NFS 加载(1080p 复用暖 cache 仅 156s)——**upstream 的 `diffusion_seconds` 含模型加载**,而 native 的 warm p50 不含 build。比较时须区分:native retake **计算**(~25–35s)vs upstream **含加载**。不得据此宣称单次冷编辑快 46×(那是摊销/常驻 Mode B 的结论)。
- **A/V 观察:** upstream `edited_full` 音频 15.8s vs 视频 ~13.2s(upstream `splice_out` 的 A/V 长度差异),composite `-shortest` 对齐;此为 upstream harness 行为,对 upstream 与全部 TRT 变体一致,不影响比较,未改 upstream 逻辑。

### 3.1 Codex 公平性交叉核查(task12)结论:**needs-caveats**

- **不得**把 upstream `750.4s`(冷 cache + 含加载 + every-rebuild)直接与 native `warm_p50`(常驻、仅 retake)同表对比而下速度结论。
- 合法对比只有两类:①**含加载/every-rebuild** 语义:native `single_shot`(build+first)vs upstream `diffusion_seconds`(但 750s 被冷加载夸大,须先用暖 cache 重跑 upstream 720p 才干净;当前比值 bf16 5.7× / FP8 6.3× / NVFP4 7.1× 仅为**示意、偏高**);②**常驻**语义:native warm 34.5s vs serve engine 31.4s(接近)。
- **"46×" 在本表不成立**(那是原报告 512p 摊销/常驻结论);本用例 720p 209f 下,`750.4/25.3=29.7×` 也是冷加载 vs 常驻的误导比。
- **fast-init 安全条件**:FP8 31.8dB 近无损**支持**但不**完备证明**——真正条件是"所有被 no-op init 的参数/buffer 在使用前都被 checkpoint 覆盖"。交付的 `final_native_bf16`(及 fp8/nvfp4 的 PSNR 参考)用的是 **oracle 的非-fast-init bf16**;fast-init 仅用于 fp8/nvfp4 产物与计时。
- **NVFP4 9.1dB 是重大质量警示**,其速度优势不得脱离质量退化单独呈现。
- upstream 与 native 的差距**不能全归于 kernel/runtime 效率**——含 checkpoint I/O、page cache 状态、rebuild 语义、offload 策略、harness 计时口径。

---

## 4. 交付物(host `artifacts/`,已 gitignore)

- `720p/{upstream,native_bf16_final,native_fp8_final,native_nvfp4_final}/final.mp4` — 4 个完整视频
- `1080p/{upstream,native_nvfp4_final}/final.mp4` — 2 个完整视频
- `720p/frame_grid_720p_t5.0.png` — 源/upstream/native bf16/fp8/nvfp4 逐帧网格
- 各 `variant.json`(native 计时/显存)、`timing.json`(upstream/composite)、`serve_bf16/serve_timing.json`

**gap(如实记录):** 1080p native bf16/FP8 = OOM;serve FP8/NVFP4 = 配置不支持量化(需改 trtllm)。
