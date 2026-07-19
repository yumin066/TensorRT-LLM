# LTX-2.3 delete_disfluency 端到端测试 — upstream vs TensorRT-LLM retake(一个真实用例)

## Goal Description(目标描述)

把一个真实的 `delete_disfluency` 编辑用例端到端跑一遍,为每个 retake 引擎产出一个
**完整、可审阅的最终视频**,从而对比 upstream LTX2.3-eval retake 与 TensorRT-LLM
native retake(bf16 / FP8 / NVFP4,含 in-process 的 *native* 与 *trtllm-serve* HTTP
两种形态)在**延迟**(对齐 `latency_report.pdf` §2 的 APPLY/LTX/POST 阶段图)与
**重生成窗口质量**两方面的表现。

用例为 `LTX2.3-eval/script_editing/eval_cases.json[0]`(`delete_disfluency`):片段
`--Y9imYnfBw-f0-484.mp4`(talking-head "Bill Gates",1920×1080,29.97 fps,~16.15 s,
484 帧),filler `[UH]` 位于 span `[5.20, 5.54] s`。

upstream 流水线除模型调用外仍是唯一权威。它的 steps 1–4(transcript → `splice_out`
→ `crop_window` → `retake_input.mp4` + `a`/`total`/`bridge`/`lb0`/`lb1` 几何)与
step 6(`composite_bridge`)原样复用;只有 step 5(`RetakePipeline` 模型调用)换成
TensorRT-LLM retake。**upstream 的确切行为:** 模型只重生成 `total` 帧
`retake_input.mp4` 里中间的 bridge 窗口 `[bridge_start_s, bridge_end_s] = [lb0/fps,
lb1/fps]`,返回一个完整的 `total` 帧视频(conditioning 帧 ≈ 输入,中间重生成);
step 6 随后把**整个** `total` 帧窗口 `[0, total)` 在偏移 `a` 处拼回,即
`composite_bridge(edited, retake_out, a, 0, total, fps, final)`。TensorRT-LLM native
retake 与此完全一致:它的 `infer()` 内部就做合成,返回一个 `frame_rate = 源 fps`
的完整同长视频。

TensorRT-LLM 代码(`tensorrt_llm/`)与 upstream packages(`../LTX2.3-eval/packages/`)
**不**改动。对 upstream/产品代码唯一的改动,是在评测 harness
`LTX2.3-eval/script_editing/delete_disfluency.py` 上加一个附加的 `--external-retake`
flag;所有测量/汇总脚手架放在独立的测试驱动脚本里。

所有数字都在一台 RTX PRO 6000 Blackwell(sm_120,96 GB)上测量 —— 绝对延迟与 H100
参考报告不同;只继承 §2 阶段图的*结构*。

## Acceptance Criteria(验收标准)

遵循 TDD 理念,每条标准都含正/负测试以便确定性验证。

- AC-1:upstream 基线把用例端到端跑通,产出完整 `final.mp4` 加 APPLY/LTX/POST 阶段图计时和一份 manifest。
  - 正测试(应 PASS):
    - `delete_disfluency.py` 用显式横版 `--width 1280 --height 704`(~720p)、无 `--external-retake` 跑该用例,产出的 `final.mp4` 在 `[a, a+total)` 之外的帧与拼接后的 `edited_full.mp4` 解码一致,且 disfluency 窗口被重生成。
    - 一份 `timing.json` 把实测 wall 拆成 APPLY(transcript + splice + crop)+ LTX(那唯一的模型调用)+ POST(composite/mux);一份 `manifest.json` 记录源 sha256、span `[5.20, 5.54]`、`a`/`total`/`bridge`/`lb0`/`lb1`、分辨率、fps、seed、model/LoRA/quant 路径,以及两个仓库的 git revision。
    - 同用例用 `--width 1920 --height 1088`(~1080p)在 upstream `--offload-mode cpu` 下跑通。
  - 负测试(应 FAIL):
    - 一次静默复用了过期 `final.mp4`(mtime 未变 / manifest sha256 不匹配)的运行被检查驱动拒绝。
    - 一份 APPLY+LTX+POST 之和与实测 wall 在容差外不相等的 `timing.json` 被拒绝。

- AC-2:在 `delete_disfluency.py`(且仅此)上加的附加 `--external-retake <mp4>` flag 使 step 5 用提供的视频替代调用 `RetakePipeline`,而其余每条路径与未改脚本逐字节一致。
  - 正测试:
    - 带 `--external-retake X.mp4` 时,`RetakePipeline` 从不被构造或调用;`X.mp4` 被用作 step-6 `composite_bridge` 的 `retake_output`;steps 1–4 与 6 原样跑。
    - 不带该 flag 时,脚本在固定 seed 上的行为与输出与当前签入脚本一致(基线 diff 无变化)。
  - 负测试:
    - 设了 `--external-retake` 却仍构造/调用 `RetakePipeline`(未短路 step 5)的运行被拒绝。
    - 为此特性对 `../LTX2.3-eval/packages/` 或 `tensorrt_llm/` 的任何改动都在评审中失败(原封树检查)。

- AC-3:在 `composite_bridge` 之前有一个接缝一致性预检,拒绝会破坏拼接的产物。
  - 正测试:
    - 一个分辨率 == 处理尺寸、帧数**恰好** == `retake_input.mp4` 帧数(`total`)、fps/timebase 在容差内匹配 `retake_input.mp4`、像素格式可解码的 mp4 —— 通过并干净地合成。
  - 负测试:
    - 一个分辨率错、帧数错(≠ `total`)、或 fps 不匹配的 mp4 被拒绝,并给出点名不匹配字段的具体诊断。不做静默 scale/pad/resample。

- AC-4:7 个变体(upstream;native bf16/FP8/NVFP4;serve bf16/FP8/NVFP4)中每一个,要么经同一条 step-6 路径产出完整合成的 `final_<variant>.mp4`,要么被记为 `oom` / `unsupported` / `timeout` 并带日志 —— 绝不静默缺失。
  - 正测试:
    - 每个 TRT 变体在转储的 `retake_input.mp4` 上以模型窗口 `[lb0/fps, lb1/fps]` 跑 retake,返回 `total` 帧视频,产出 `final_<variant>.mp4`。
    - 一个在 1080p 下 OOM 的变体被记为 `oom`,且不中止其余变体(子进程隔离)。
  - 负测试:
    - 一个标为 `ok` 却没有预期时长/分辨率的可解码 `final_<variant>.mp4` 的变体被拒绝。

- AC-5:每变体计时遵循 §2 阶段图,同时报单发与 warm p50,外加峰值显存与状态。
  - 正测试:
    - 每行带 APPLY / LTX / POST、一个冷单发延迟、一个 warm p50(常驻 native + serve)、峰值 reserved 显存、量化模式、native-vs-serve 模式与状态;serve 行额外拆 cold-start / first-served / warm 以及 HTTP-wall vs engine 时间。
    - TRT 行的 APPLY 与 POST 是共享的 Phase-A/Phase-C 成本,明确标为复用;upstream 带自己的每次运行 APPLY/POST。
  - 负测试:
    - 把 warm p50 样本混进单发数字,或把复用的 APPLY/POST 当作某 TRT 行独立重测来呈现,被汇总 schema 拒绝。

- AC-6:重生成窗口质量仅作为信息性信号报告,绝不作为 pass/fail 门;为人工审阅产出一张 源-vs-upstream-vs-变体 的 frame grid。
  - 正测试:
    - 每变体输出重生成**窗口**(仅编辑区,而非整段)相对 bf16/upstream 参考的 PSNR/SSIM;在切点接缝处写一张并排 frame grid 供肉眼看 FP8/NVFP4 漂移。
    - 每个 `final_<variant>.mp4` 在 `[a, a+total)` 之外的帧与 `edited_full.mp4` **解码容差相等**(两者都是 H.264 CRF-16 重编码,所以这是解码容差比对,不是字节一致)。字节级合成一致仅在 TRT tensor/PT 层经现有 `ltx2_retake_oracle.py` harness 断言,不在交付 mp4 上。
  - 负测试:
    - 把正确性门挂在 PSNR/SSIM 阈值上的运行被拒绝(按项目规则质量为信息性)。
    - 一个 `final_<variant>.mp4` 窗外帧偏离 `edited_full.mp4` 超出解码容差,被拒绝。

- AC-7:每个 TRT `final_<variant>.mp4` 里的音频源自 upstream 拼接后的音频(retake 是纯视频),需断言 —— 不是 TRT 生成的、也不来自外部 retake mp4。
  - 正测试:
    - step 6 从 `retake_out` 取视频(`-map 0:v`)、从 `edited_full.mp4` 取音频(`-map 1:a`, `-c:a aac`);断言确认 `final` 音频在时长/内容上匹配 `edited` 音频(一次 AAC 转码,不是流字节相等),外部 retake mp4 的音频(若有)被忽略。
  - 负测试:
    - 一个音频来自外部 retake mp4、或音频缺失的 final,被拒绝。

- AC-8:在特性算 done 之前,`--external-retake` 代码路径在 GPU 上端到端跑一次(项目硬规则)。
  - 正测试:
    - 至少一次在该用例上真实的 GPU 运行 `delete_disfluency.py --external-retake ...` 产出一个 `final.mp4`,并记录节点/容器。
  - 负测试:
    - 只在 host 上验证 flag 逻辑、无 GPU 运行,不满足 AC-8。

- AC-9:每个变体产出的 `final_<variant>.mp4` 与 retake 输出 mp4(native/serve 的 `retake_<variant>.mp4`、upstream 的 `final_upstream.mp4` / `retake_output.mp4`)跑完后都从 GPU 节点拉回 host 机器,收进 host 的 `artifacts/<res>/` 目录;该 `artifacts/` 目录加入 `.gitignore`,产物不进版本管理。
  - 正测试(应 PASS):
    - 一次完整跑后,host 的 `artifacts/<res>/` 下列得出全部 final + retake 输出 mp4(每个成功变体各一份);`git check-ignore artifacts/` 返回该路径,且 `git status` 不显示 `artifacts/` 下任何文件。
  - 负测试(应 FAIL):
    - 某个 final/retake mp4 只留在 GPU 节点、未拉回 host `artifacts/` 的运行,被交付检查拒绝。
    - `artifacts/` 或其下 mp4 出现在 `git status` 的待提交/已跟踪列表里(漏加 `.gitignore`),被拒绝。

## Path Boundaries(路径边界)

### Upper Bound(最大可接受范围)
完整 7 变体矩阵在 720p,加上放得下的变体在 1080p;每个变体产出可审阅的
`final_<variant>.mp4`;一个接缝一致性预检(精确帧数 / 分辨率 / fps / pix_fmt);
一张 §2 对齐的计时表,含单发 + warm p50 + 峰值显存 + 状态;信息性 PSNR/SSIM +
源/upstream/变体 frame grid;音频来源与窗外解码一致性断言;一份两仓库 provenance
manifest(两个 git revision 都钉住)。所有 final 与 retake 输出 mp4 跑完后拉回 host
`artifacts/<res>/`(该目录 gitignored,产物不进仓库)。评测 harness 上一个附加
`--external-retake` flag 是唯一的 upstream/产品代码改动;所有测量/汇总都在独立测试
驱动脚本里。

### Lower Bound(最小可接受范围)
upstream 基线 + 至少 native bf16、native FP8 与一个 serve 变体,各经 `--external-retake`
接缝在 720p 产出 `final.mp4`,带 APPLY/LTX/POST 单发计时以及音频来源 +
窗外解码一致性断言,且 final/retake 输出 mp4 拉回 host `artifacts/`(`.gitignore`)。
NVFP4、1080p、warm-p50 与 frame grid 可降级为 `unsupported`/`skipped`,**但仅当**被记为
显式 gap。

### Allowed Choices(允许的选择)
- 可用:现有 `examples/visual_gen/ltx2_retake_*.py` harness(oracle / timing /
  serve_timing / quant_mem)驱动 TRT retake;一个独立的 workstream 测试驱动做
  编排/计时/汇总;`ffprobe`/PyAV/OpenCV 做预检 + 指标;近容量行用子进程逐项隔离;
  upstream 的 1080p 基线用 `--offload-mode cpu`;每次调用都显式传横版 `--width/--height`
  (评测默认是竖版 704×1280,必须覆盖)。
- 不可用:对 `../LTX2.3-eval/packages/` 或 `tensorrt_llm/` 的任何改动;在比较前把变体
  输出 resample 到别的分辨率(只做分辨率内比较);把正确性门挂在 PSNR/SSIM 上;
  任何 mp4 字节一致的说法(final 是 CRF-16 重编码)。

## Feasibility Hints and Suggestions(可行性提示与建议)

> **注**:本节仅供参考理解,是概念性建议、非硬性要求。

### 概念方法
三阶段:
- **Phase A(转储,一次):** 跑 upstream steps 1–4,产出 `retake_input.mp4`(一个
  `total` 帧窗口,横版处理尺寸)+ `geometry.json`(`a`、`total`、`bridge`、`lb0`、
  `lb1`、`fps`、`w×h`、`splice_frame`)与 APPLY-block 计时。所有 TRT 变体的共享输入。
- **Phase B(每引擎 retake):** 把 `retake_input.mp4` 喂给 TRT retake,模型窗口
  `[lb0/fps, lb1/fps]`(中间 bridge —— TRT 的 `_retake_pixel_window` 用 `round(t*fps)`,
  故 `lb0/fps, lb1/fps` 精确还原 `[lb0, lb1]`)。native 经 oracle/timing harness
  in-process;serve 经 `POST /v1/videos/generations`,带
  `retake_video_path/retake_start_time/retake_end_time`。TRT 内部合成,返回 `total`
  帧视频 → `retake_<variant>.mp4`。serve 需要量化(FP8/NVFP4)server 配置和一个
  **产物**调用(非 `format='pt'` 纯测时路径),以便落一个可解码 mp4。
- **Phase C(合成,每变体):** 调 `delete_disfluency.py --external-retake
  retake_<variant>.mp4`,让 upstream step-6 `composite_bridge(edited, retake_out, a,
  0, total, fps, final)` 把整窗拼回 → `final_<variant>.mp4`,并触发预检。

### 相关参考
- `LTX2.3-eval/script_editing/delete_disfluency.py` — `crop_window`(第 139 行,
  retake_input)、`composite_bridge`(第 160 行,拼回 `[0,total)`)、`main()` 几何
  (第 262–336 行);在此加 `--external-retake`。
- `tensorrt_llm/_torch/visual_gen/models/ltx2/pipeline_ltx2_retake.py` —
  `_retake_pixel_window`、`_composite_retake_window`、`infer()`(返回完整同长视频,
  `frame_rate=fps`)。
- `examples/visual_gen/ltx2_retake_oracle.py` / `ltx2_retake_serve_timing.py` —
  native + serve 驱动;`build_retake_payload`(serve payload;注意 `format='pt'` 是
  纯测时,产物需要一个视频格式的变体)。
- `examples/visual_gen/RETAKE_ACCELERATION_REPORT.md` §2/§3 — 分辨率/显存 + 量化
  保真锚点。

## Dependencies and Sequence(依赖与顺序)

### Milestones(里程碑)
1. harness + 接缝:给 `delete_disfluency.py` 加 `--external-retake`;加帧数/分辨率/fps
   预检;GPU 验证 flag 路径一次。
   - Phase A:转储 `retake_input.mp4` + 几何。
2. upstream 基线(AC-1):在 720p(以及 1080p 经 cpu-offload)完整跑 upstream,带
   §2 计时 + manifest。
3. TRT 变体扫描(AC-4/5):native{bf16,fp8,nvfp4} + serve{bf16,fp8,nvfp4} 在转储窗口
   上 retake(子进程隔离),再经接缝做 Phase-C 合成;逐行计时/显存/状态。
4. 质量 + 音频(AC-6/7):信息性 PSNR/SSIM(仅窗口)、源/upstream/变体 frame grid、
   音频来源 + 窗外解码一致性断言。
5. 汇总(AC-5):把所有行汇成一张报告表(720p 全 + 1080p 放得下的);gap 显式记录;
   两仓库 provenance。

<依赖是相对的、非时间线:task3(转储)门控整个扫描;task1/task2(flag+预检)门控每次合成。>

## Task Breakdown(任务拆解)

每个任务必须恰好带一个路由标签:`coding`(Claude)或 `analyze`(Codex)。

| Task ID | 描述 | 目标 AC | 标签 | 依赖 |
|---------|-------------|-----------|-----|------------|
| task1 | 给 `delete_disfluency.py` 加附加 `--external-retake` flag(跳过 step-5 `RetakePipeline` 构造+调用,用 mp4 做 step-6);保持默认路径不变 | AC-2 | coding | - |
| task2 | 在 `composite_bridge` 前加接缝一致性预检:精确 `total` 帧数、分辨率 == 处理尺寸、fps/timebase、可解码 pix_fmt | AC-3 | coding | task1 |
| task3 | Phase A:从 upstream steps 1–4 转储 `retake_input.mp4` + `geometry.json`(a/total/bridge/lb0/lb1/fps/w×h/splice_frame) | AC-1 | coding | - |
| task4 | upstream 基线在 720p + 1080p(cpu-offload)完整跑,带 §2 APPLY/LTX/POST 计时 + manifest(两仓库 rev) | AC-1,AC-5 | coding | task3 |
| task5 | Phase B:在转储窗口上驱动 native bf16/fp8/nvfp4 retake(oracle/timing harness),子进程隔离,窗口 `[lb0/fps,lb1/fps]` | AC-4,AC-5 | coding | task3 |
| task6 | Phase B:驱动 serve bf16/fp8/nvfp4 retake —— 量化 server 配置 + 产物(非 `pt`)调用落可解码 mp4;cold/first/warm 拆分 | AC-4,AC-5 | coding | task3 |
| task7 | Phase C:每变体经 `--external-retake` 合成 → `final_<variant>.mp4`(触发预检) | AC-4 | coding | task1,task2,task5,task6 |
| task8 | 质量:信息性 PSNR/SSIM(仅窗口)+ 源/upstream/变体 frame grid;记录 retake_input / retake_<variant> / final_<variant> 的 ffprobe 元数据 | AC-6 | coding | task7 |
| task9 | 断言:final 音频源自 `edited`(AAC 转码,非外部 retake 音频);窗外相对 `edited` 解码容差一致 | AC-7,AC-6 | coding | task7 |
| task10 | 把所有行汇成一张报告(720p 全 + 1080p 放得下的);记录 gap;两仓库 provenance | AC-5 | coding | task4,task7,task8,task9 |
| task11 | 在节点/容器上 GPU 验证 `--external-retake` 路径一次 | AC-8 | coding | task7 |
| task12 | 交叉核查计时归因 + 公平性(upstream 与 TRT 之间锁定 prompt/seed/steps/LoRA/LoRA-strength;APPLY/POST 复用如实标注) | AC-5 | analyze | task4,task5,task6 |
| task13 | 把每个变体的 `final_<variant>.mp4` + retake 输出 mp4(含 upstream)从 GPU 节点拉回 host `artifacts/<res>/`;把 `artifacts/` 加入 `.gitignore` | AC-9 | coding | task7 |

## Claude-Codex Deliberation(Claude-Codex 商讨)

### Agreements(共识)
- 原样复用 upstream steps 1–4 + 6;只换 step 5;评测 harness 上单个附加 flag;测量驱动是独立测试脚本。
- 接缝一致性(精确 `total` 帧数 / 分辨率 / fps / pix_fmt)必须预检验证,不能假设。
- 质量为信息性;正确性门挂在 结构 + 音频来源 + 窗外解码一致性 上。
- 近容量 native 行子进程隔离,单次 OOM 不污染整个扫描。
- 所有 final / retake 输出 mp4 跑完拉回 host `artifacts/<res>/`;`artifacts/` 加入 `.gitignore`,产物不进仓库。

### Resolved Disagreements(已解决的分歧)
- "TRT 产出完整合成视频还是只有窗口?":由代码判定 —— native `infer()` 内部合成,返回完整同长视频,故可直接作为 step-5 替代物落入 `composite_bridge`。
- "合成范围":工作树里的 `delete_disfluency.py` 调用 `composite_bridge(edited, retake_out, a, 0, total, fps, final)` —— 拼接的是**整个** `total` 帧窗口(模型只重生成中间 bridge)。因此预检要求**恰好 `total`** 帧,窗外区域是 `[a, a+total)` 之外的一切。
- "MP4 字节一致":`composite_bridge` 把全部视频 H.264 CRF-16、音频转 AAC 重编码,故交付 mp4 **非**字节一致。窗外一致性变为相对 `edited` 的解码容差比对;字节一致仅在 TRT tensor/PT 层(oracle harness)断言。
- "音频验证":final 音频是从 `edited` 映射来的 AAC 转码(`-map 1:a`);断言它源自 `edited`(外部 retake 音频忽略),不是流字节相等。
- "serve FP8/NVFP4 + 产物":现有 serve-timing harness 只有 bf16 且仅 `format='pt'`;task6 加量化 server 配置和一个落可解码 mp4 的产物调用。
- "分辨率默认":评测默认竖版 704×1280;每次调用显式传横版 `--width/--height`(1280×704 / 1920×1088)。

### Convergence Status(收敛状态)
- Final Status:`converged`

## Pending User Decisions(待用户决策)

五项设计决策均在规划前与用户敲定;此处记为已定(Decision Status 是用户最终决定,而非 `PENDING`)。

- DEC-1:接缝机制。Claude 立场 / Codex 立场:评测 harness 上的附加 `--external-retake` flag(不是 `packages/`,不是 trtllm)。取舍:最小、附加、保持 upstream 胶水逐字节一致;备选是一个 import 纯 helper 的外部编排器。Decision Status:**已定 —— 选项 A(`--external-retake`)。**
- DEC-2:分辨率。取舍:1080p offload=none 在 ~`total` 帧下大概率 OOM;720p 完全 apples-to-apples。Decision Status:**已定 —— 完整 7 变体矩阵在 720p + 1080p 只跑放得下的变体;分辨率内比较。**
- DEC-3:serve 范围。Decision Status:**已定 —— 三个 serve 变体全跑(bf16/FP8/NVFP4)。**
- DEC-4:Warm vs 单发。取舍:单发直接可比 upstream 固有单发;warm p50 展示摊销/常驻优势;不得暗示单次冷编辑快 ~46×。Decision Status:**已定 —— 两者都报(单发为主,warm p50 为辅)。**
- DEC-5:最终分辨率 / resample。取舍:`splice_out` 把整段片段缩放到 `w×h`,最终停在那儿(签入脚本无 `match_source_dims`)。Decision Status:**已定 —— 不 resample;分辨率内比较。**
- DEC-6:FP8/NVFP4 的通过判据。Claude 立场:结构有效 + 音频来源 + 产出最终视频;PSNR/SSIM 供人工肉眼看的信息性。Codex 立场:同意(与既定项目方法一致)。Decision Status:**已定 —— 结构/音频门;质量信息性。**

## Implementation Notes(实现注记)

### Code Style Requirements(代码风格要求)
- 实现代码与注释**不得**含计划专用术语,如 "AC-"、"Milestone"、"Step"、"Phase" 等工作流标记。
- 这些术语仅用于计划文档,不进入产物代码。
- 代码里用描述性、贴合领域的命名。

### Scope & Verification(范围与验证)
- 对 upstream/产品代码唯一的改动,是 `LTX2.3-eval/script_editing/delete_disfluency.py` 上的附加 `--external-retake` flag(+ 其预检)。不改 `tensorrt_llm/` 或 `../LTX2.3-eval/packages/`。所有计时/manifest/过期检查/汇总/serve 产物采集逻辑放在独立的 workstream 测试驱动脚本里。
- 每处代码改动在节点/容器上 GPU 验证一次(项目硬规则)—— host 侧验证 flag 逻辑必要但不充分。
- 始终显式传横版 `--width/--height`;评测默认是竖版(704×1280),会把宽高比转向。
- manifest 钉住两个仓库的 git revision(TensorRT-LLM 与工作树版 `LTX2.3-eval`,其 `composite_bridge(… a, 0, total …)` 对 step 6 为权威)。
- 所有变体的 `final_<variant>.mp4` 与 retake 输出 mp4(含 upstream)跑完后从 GPU 节点拉回 host 的 `artifacts/<res>/` 目录供审阅;`artifacts/` 加入 `.gitignore`,产物(mp4/timing/report)不进版本管理。

## Output File Convention(输出文件约定)

本模板用于产出主输出文件(如 `plan.md`)。

### 翻译语言变体

当 `alternative_plan_language` 经合并配置加载解析为某个受支持语言名时,主文件之后还会写一份翻译变体。Humanize 按如下顺序从合并层加载配置:默认配置、可选用户配置、可选项目配置;`alternative_plan_language` 可在任一层设置。变体文件名通过在扩展名前插入 `_<code>`(内置映射表里的 ISO 639-1 代码)构造:

- `plan.md` → `plan_<code>.md`(如中文 `plan_zh.md`、韩文 `plan_ko.md`)
- `docs/my-plan.md` → `docs/my-plan_<code>.md`
- `output`(无扩展名) → `output_<code>`

翻译变体文件包含主计划文件当前内容在配置语言下的完整翻译。所有标识符(`AC-*`、task ID、文件路径、API 名、命令 flag)保持不变,因其与语言无关。

当 `alternative_plan_language` 为空、缺失、设为 `"English"`、或设为不受支持的语言时,不写翻译变体。当无项目配置文件时,Humanize 不会自动创建 `.humanize/config.json`。

--- Original Design Draft Start ---

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

--- Original Design Draft End ---
