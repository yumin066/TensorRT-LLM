# draft_plan_test.md — 端到端 delete-disfluency:upstream vs TRT-LLM retake

> 状态:**待评审草稿。** 尚未启动任何 GPU 运行。§8 的开放问题需要先决策再执行。

## 0. 目标

把**第一个 `delete_disfluency` 评测用例**端到端跑一遍,为每个引擎变体产出一个
**最终合成视频**(去掉 disfluency、把 bridge 拼回原视频),外加一份与
`latency_report.pdf` §2 流水线阶段图对齐的**逐阶段延迟拆解**。

必须产出两样东西:
1. retake 这一步的**性能 + 质量对比**:upstream vs TRT-LLM(bf16 / FP8 / NVFP4,
   各含 **native** 与 **serve** 两种形态)。
2. **证明 TRT-LLM 的 retake 真能拼回**源视频、达成 delete-disfluency 效果 ——
   即每个变体都有一个真实、可审阅的 `final.mp4`,而不是孤立的 retake 窗口。

合成/拼接工作仍留在 **upstream** 流水线里;TRT-LLM 代码与 LTX `packages/`
**不改动**。

### 用例(来自 `eval_cases.json[0]`)
```
clip_id        --Y9imYnfBw-f0-484   (Bill Gates "world's richest man")
task           delete_disfluency
which=0        word_index=16  filler_text=[UH]  span_s=[5.20, 5.54]
video          LTX2.3-eval/clips/--Y9imYnfBw-f0-484.mp4   (1920x1080, 29.97 fps, 16.15 s, 484 f)
transcript     LTX2.3-eval/script_editing/eval/transcripts/--Y9imYnfBw-f0-484.json
width=1920  height=1088  fps=29.97
```
Retake 几何(upstream 默认):`cond-frames=90`(每侧) + `retake-frames=25` bridge,
按 `8k+1` 向上取整 → **retake 窗口 ≈ 205 帧**,跨在 5.2 s 的切点上。
(整段 484 帧的视频只在 APPLY/POST 阶段被 CPU ffmpeg 触碰。)

---

## 1. 关键事实与约束

- **硬件与 PDF 不同。** `latency_report.pdf` 是在 **H100 80GB ×8, torch 2.9.1**
  上测的。我们跑在 **1× RTX PRO 6000 Blackwell 96GB (sm_120)**。
  → 绝对秒数不会与 PDF 对得上;我们只对齐**阶段图结构**,报本机数字。
  每张表都要标注这一点。
- **retake 窗口在 1080p 下约 205 帧。** 我们实测 1080p retake 在 **89 帧**时
  已经 reserve **91.5 GiB**(offload=none)。205 帧 ≈ 2.3× 的 latent tokens →
  **bf16 native offload=none 在 1080p 下极可能 OOM**。见 §5 的分辨率/offload 决策。
- **upstream 靠 `--offload-mode cpu` 能放下 1080p**(其默认;720p 与 1080p 参考
  输出都存在)。为了做到 apples-to-apples,我们的 TRT-LLM retake 在每次对比里
  必须跑在**与 upstream 相同的分辨率**。
- **serve 的细粒度阶段更粗。** 通过 HTTP,引擎在 `Server-Timing` 里只报
  `generation` + `denoise`;细的 LTX 子阶段(vae_encode、text_encode、
  vae_decode)只在 **native/in-process** 路径上才有。因此 serve 变体拿到的
  LTX-block 拆解更粗 —— 属预期,不是缺陷。
- **TRT-LLM retake 是纯视频**(`regenerate_audio=False`);它通过 PyAV 保留源
  音频,**不**跑 audio_vae_encode/decode/vocoder。这些行对 TRT-LLM 的 LTX block
  为 N/A(upstream 有小的 ~0.5 s + 0.3 s 条目)。

---

## 2. 测试矩阵(7 个变体)

每个变体产出一个 `final.mp4` + 一个 `timing.json`。

| # | 引擎 | 量化 | 形态 | Retake 由谁产出 | 合成由谁做 |
|---|--------|-------|------|--------------------|--------------|
| 1 | **upstream**(参考) | fp8-cast(其默认) | in-process | upstream `RetakePipeline`(完整 `delete_disfluency.py`) | upstream |
| 2 | TRT-LLM native | bf16 | in-process | 我们的 `ltx2_retake` native 路径 | upstream step 6 |
| 3 | TRT-LLM native | FP8 | in-process | 我们的 native 路径 | upstream step 6 |
| 4 | TRT-LLM native | NVFP4 | in-process | 我们的 native 路径 | upstream step 6 |
| 5 | TRT-LLM serve | bf16 | HTTP | `trtllm-serve` retake | upstream step 6 |
| 6 | TRT-LLM serve | FP8 | HTTP | `trtllm-serve` retake | upstream step 6 |
| 7 | TRT-LLM serve | NVFP4 | HTTP | `trtllm-serve` retake | upstream step 6 |

全部 7 个共享**完全相同**的 APPLY(①)与 POST(③)块(同样的 upstream ffmpeg
胶水、同样的 conditioned input、同样的 splice),因此只有 block ②(模型 retake)
不同。

---

## 3. 流水线集成设计(不改 trtllm / 不改 packages)

`delete_disfluency.py` 干净地把模型调用与胶水分开:
- **steps 1–4** 构建*已编辑片段* + *retake 输入窗口*(`crop_window`),
- **step 5** 是单次 `RetakePipeline` 调用(唯一的模型推理),
- **step 6** `composite_bridge(edited, retake_output, …)` 把 retake 窗口拼回,
  写出 `final.mp4` + `timing.json`。

**注入点 = step 4 与 step 6 之间。** 为了在复用 upstream 胶水的同时换入 TRT-LLM
retake,在 `script_editing/` 里加一个薄的**两阶段接缝**(这是评测 harness ——
**不是** `packages/`,也**不是** trtllm 仓库):

- **Phase A(upstream,一次):** 跑 steps 1–4,转储
  (a) `retake_input.mp4`(模型必须重生成的窗口)以及
  (b) `geometry.json`(`a`、`lb0`、`lb1`、`fps`、`width`、`height`、`splice_frame`,
  以及 `apply`+几何计时)。
- **Phase B(每个引擎变体):** 在 `retake_input.mp4` 上跑 retake →
  `retake_output_<variant>.mp4`。
  - upstream 变体:直接让 `delete_disfluency.py` 正常跑 step 5。
  - trtllm 变体:我们的 runner(native 或 serve)读 `retake_input.mp4`,
    重生成窗口,写出 `retake_output_<variant>.mp4`。
- **Phase C(upstream,每个变体):** 只跑 **step 6**(`composite_bridge`),用选定的
  `retake_output_<variant>.mp4` → `final_<variant>.mp4` + `timing.json`。

接缝的实现选项(在 §8 里选):
- **(优先)在 `delete_disfluency.py` 上加小的附加 flag**:
  `--dump-retake-input <path>`(step 4 后停)与
  `--external-retake <path>`(跳过 step 5,把这个当作 step 6 的 retake 输出)。
  ~30 行,附加式,保持 LTX `packages/` 原封不动。
- **(备选)workstream 里一个薄编排器**,`import` 那些纯 helper
  (`find_disfluency`、`splice_out`、`crop_window`、`composite_bridge`、`read_frames`)
  并驱动 A/B/C —— 对 LTX2.3-eval 零改动。

无论哪种:**trtllm 代码不动;LTX `packages/` 不动。** `--start 5.20 --end 5.54
--which 0` override 使用评测用例的精确 span(跳过重新检测)。

---

## 4. 阶段图与指标(对齐 `latency_report.pdf` §2)

每个变体都对着同一张三块图报;只有 block ② 是每引擎重测,block ① 和 ③ 测一次
(共享,来自 Phase A/C)。

| Block | 阶段 | 在哪测 | upstream | trtllm native | trtllm serve |
|-------|-------|----------------|----------|---------------|--------------|
| ① APPLY | whisper ASR / propose edits / prepare_base_video / make_conditioned_input | Phase A(共享) | ✅ | ✅(复用) | ✅(复用) |
| ② LTX | subprocess_start / model_load | 每变体 | ✅ | ✅(`model_build_load`) | 常驻(一次) |
| ② LTX | video_vae_encode | 每变体 | ✅ | ✅(`vae_encode`) | —(HTTP) |
| ② LTX | text_encode (Gemma) | 每变体 | ✅ | ✅(`conditioning`) | —(HTTP) |
| ② LTX | **diffusion(8 steps)** | 每变体 | ✅ | ✅(`denoise_total` + `denoise_per_step`) | ✅(`denoise`) |
| ② LTX | video_vae_decode | 每变体 | ✅ | ✅(`vae_decode`) | —(HTTP) |
| ② LTX | audio_vae_encode/decode | 每变体 | ✅(小) | N/A(纯视频) | N/A |
| ② LTX | mp4_encode(窗口) | 每变体 | ✅ | ✅(encode_mp4) | ✅ |
| ② LTX | (仅 serve) generation / wall | 每变体 | — | — | ✅(`Server-Timing` + 客户端 wall) |
| ③ POST | ffmpeg_splice_chunks / post_process_match_source_dims | Phase C(共享) | ✅ | ✅(复用) | ✅(复用) |
| — | **端到端 wall** | apply + LTX + post | ✅ | ✅ | ✅ |

测量规则(与我们的 AC-1 harness 一致):GPU 阶段用 CUDA events +
`torch.cuda.synchronize()`-括起的 wall;retake 做 **N warmup + M measured**
(warm p50/p90/min);冷 `model_load` 报一次、且**不**折进常驻(serve)形态的每次
调用 warm 合计;upstream/native 每调用形态**要**含 `model_load`(它每次重载)。

---

## 5. 分辨率与显存计划

进入完整矩阵前的决策门:

1. **容量预检**(便宜,1 次 build):在真实 ~205 帧窗口上把 TRT-LLM native retake
   在 **1080p (1920×1088), offload=none, bf16** 跑一次。记录 OK / OOM + peak reserved。
2. **分支:**
   - **如果 1080p bf16 native 放得下** → 整个矩阵在 1080p 跑。
   - **如果 OOM**(预期):从下列里按优先级选**一个** ——
     - **(a) 整个矩阵在 720p (1280×704)** —— 报告的另一个 regime,全部舒适放下,
       完全 apples-to-apples。*(推荐默认)*
     - (b) 1080p,native/serve retake 也用 `offload-mode cpu`(匹配 upstream 的
       1080p 路径)—— 放得下但加了 offload 时间;标注。
     - (c) 1080p 但**缩小窗口**(如 `cond-frames 30`)使其在 offload=none 下放下 ——
       相对评测默认改了几何;标注。

FP8 / NVFP4 各释放 ~18 / ~25 GiB,所以即便 bf16 放不下,它们也可能在 1080p
offload=none 下放下 —— 矩阵会**逐模式记录状态**(`ok` / `oom`)而非假设。

**推荐:** 主矩阵在 **720p** 跑,做干净的 7 路对比;并额外**尝试 1080p**、只跑
放得下的模式(至少 upstream,以及 FP8/NVFP4 native)作为拉伸,OOM 处标注。

---

## 6. 交付物(供评审)

在 `Script_Editing_Workstream/test_outputs/<res>/` 下:
- `final_upstream.mp4`、`final_native_bf16.mp4`、`final_native_fp8.mp4`、
  `final_native_nvfp4.mp4`、`final_serve_bf16.mp4`、`final_serve_fp8.mp4`、
  `final_serve_nvfp4.mp4` —— **7 个完整的 delete-disfluency 视频**。
- `retake_output_<variant>.mp4` —— 每变体的原始重生成窗口。
- `timing_<variant>.json` —— 逐阶段拆解(§4 schema)。
- `latency_comparison.md` + 一张阶段图表(本机,upstream vs 6 个 trtllm)。
- `quality_comparison`:(a) 每个 `final_*` 供肉眼看;(b) 每个 trtllm `retake_output`
  vs upstream `retake_output` 的信息性 PSNR/SSIM(仅窗口),以及每个 `final_*`
  vs `final_upstream`(整帧);(c) 切点接缝处的 frame-grid,显示 bridge 无缝。
- 可选:把结果并入更新版的一页报告 / 客户 deck。

---

## 7. 执行步骤(有序)

1. **环境检查(集群):** 确认 upstream `delete_disfluency.py` 能跑
   (LTX2.3-eval `.venv` 或我们容器里有 `ltx_pipelines` + `ffmpeg` 在 PATH +
   `models/` 接到我们的 `ltx-retake-assets`),且 `--Y9imYnfBw-f0-484` 的
   CrisperWhisper transcript 在场(在,`eval/transcripts/` 里)。
2. **加两阶段接缝**(§3)—— flag 或编排器;CPU 胶水上做 unit-smoke。
3. **容量预检**(§5)→ 选分辨率。
4. **变体 1(upstream):** 完整 `delete_disfluency.py` → `final_upstream.mp4` +
   计时(这一步也产出共享的 `retake_input.mp4` + APPLY/POST 计时)。
5. **变体 2–4(native bf16/fp8/nvfp4):** 在 `retake_input.mp4` 上 retake → 合成
   (step 6)→ `final_native_*` + retake 计时(复用 step 4 的 APPLY/POST)。
6. **变体 5–7(serve bf16/fp8/nvfp4):** 每量化起一个 `trtllm-serve`,在同一窗口
   上 HTTP retake → 合成 → `final_serve_*` + serve 计时。
7. **组装** latency + quality 对比,把所有视频拉到本地,评审。
8. 按既定规则:每条代码路径在 GPU 上跑一次才算 "done"。

---

## 8. 决策

1. **接缝机制 —— 已定:选项 A(A-lite)。** 给
   `LTX2.3-eval/script_editing/delete_disfluency.py`(不是 `packages/`,不是 trtllm)
   加一个附加 flag `--external-retake <mp4>`(+ 可选的 `diffusion_seconds` sidecar):
   设了它就跳过 step-5 的 `RetakePipeline` 调用,把提供的 mp4 当作 retake 输出,
   正常跑 step 6 `composite_bridge` + 计时。Steps 1–4(便宜:读 transcript json +
   2 个 ffmpeg 操作;whisper ASR 在 `transcribe.py` 里单独做)重算出完全相同的
   `a/total/bridge` 几何,所以胶水与 upstream 路径逐字节一致。Diff 作为一个
   `.patch` 隔离保存。
   - upstream 变体:正常跑(无 flag)→ 产出共享的 `retake_input.mp4` +
     `edited_full.mp4` + `final_upstream.mp4` + 计时。
   - trtllm 变体:在 upstream 的 `retake_input.mp4` 上 retake → `my_out.mp4` →
     `delete_disfluency.py --external-retake my_out.mp4` → `final_<variant>.mp4`。

2. **分辨率 —— 已定:** 完整 **7 路矩阵在 720p**(1280×704;全放得下)+
   **1080p (1920×1088) 只跑放得下的模式**(upstream 经 offload cpu;trtllm 模式按
   §5 预检记 `ok`/`oom` —— FP8/NVFP4 比 bf16 更可能)。质量**在各分辨率内**比
   (720p 组;1080p 组)。
3. **serve 范围 —— 已定:** 三个 serve 变体**全跑**(bf16 / FP8 / NVFP4)。
4. **Warm vs 单发 —— 已定:两者都报。** 单发(每变体一次真实 retake → 实际的
   `final.mp4`,直接可比 upstream 固有的单发)为主;warm p50(trtllm native/serve,
   load-once 摊销)为辅助列。分析必须区分**retake 计算**(native 比 upstream 的
   retake ~8.5× 快)与**含模型加载**(单次冷编辑大致相当 —— trtllm 冷 build 更重;
   ~46× 的优势只在摊销/常驻下成立)。**不要**暗示单次冷编辑快 46×。
5. **最终分辨率 —— 已定(代码里已核实):** `delete_disfluency.py` **不**做 resample
   —— `splice_out` 把整段片段缩放到 `args.width×args.height`,最终就停在那儿
   (签入脚本里没有 `match_source_dims`)。每个变体的 `final.mp4` 处在其处理分辨率
   (720p → 1280×704,1080p → 1920×1088);无人为 resample;分辨率内比较。

## 9. 风险

- 1080p offload=none OOM(预期)—— 由 §5 分支缓解。
- 集群上 upstream 环境 / `models/` 接线可能需要一次性设置(§7.1)。
- serve 细阶段粒度更粗(HTTP)—— 已记录,无服务端改动无法修(超范围)。
- 绝对延迟是设备相关的(6000 PRO,不是 PDF 的 H100)—— 始终标注。
