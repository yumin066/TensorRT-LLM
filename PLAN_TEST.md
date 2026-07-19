# LTX-2.3 delete_disfluency E2E test — upstream vs TensorRT-LLM retake (one real case)

## Goal Description

Run ONE real `delete_disfluency` edit case end-to-end and produce a **complete, reviewable final video** for every retake engine, so we can compare the upstream LTX2.3-eval retake against the TensorRT-LLM native retake (bf16 / FP8 / NVFP4, in both in-process *native* and *trtllm-serve* HTTP forms) on both **latency** (aligned to the `latency_report.pdf` §2 APPLY/LTX/POST stage map) and **regenerated-window quality**.

The case is `LTX2.3-eval/script_editing/eval_cases.json[0]` (`delete_disfluency`): clip `--Y9imYnfBw-f0-484.mp4` (talking-head "Bill Gates", 1920×1080, 29.97 fps, ~16.15 s, 484 frames), filler `[UH]` at span `[5.20, 5.54] s`.

The upstream pipeline stays the single source of truth for everything except the model call. Its steps 1–4 (transcript → `splice_out` → `crop_window` → `retake_input.mp4` + the `a`/`total`/`bridge`/`lb0`/`lb1` geometry) and step 6 (`composite_bridge`) are reused unchanged; only step 5 (the `RetakePipeline` model call) is swapped for a TensorRT-LLM retake. **Exactly what upstream does:** the model regenerates only the middle bridge window `[bridge_start_s, bridge_end_s] = [lb0/fps, lb1/fps]` inside the `total`-frame `retake_input.mp4` and returns a full `total`-frame video (conditioning frames ≈ input, middle regenerated); step 6 then splices the **whole** `total`-frame window `[0, total)` back at offset `a` via `composite_bridge(edited, retake_out, a, 0, total, fps, final)`. The TensorRT-LLM native retake matches this exactly: its `infer()` composites internally and returns a full same-length video with `frame_rate = source fps`.

TensorRT-LLM code (`tensorrt_llm/`) and upstream packages (`../LTX2.3-eval/packages/`) are **not** modified. The only change to upstream/product code is an additive `--external-retake` flag on the eval harness `LTX2.3-eval/script_editing/delete_disfluency.py`; all measurement/aggregation scaffolding lives in separate test-driver scripts.

All numbers are measured on one RTX PRO 6000 Blackwell (sm_120, 96 GB) — absolute latencies differ from the H100 reference report; only the §2 stage-map *structure* is inherited.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Upstream baseline runs the case end-to-end and emits a full `final.mp4` plus APPLY/LTX/POST stage-map timing and a manifest.
  - Positive Tests (expected to PASS):
    - `delete_disfluency.py` on the case with explicit landscape `--width 1280 --height 704` (~720p) and no `--external-retake` produces `final.mp4` whose frames outside `[a, a+total)` decode-match the spliced `edited_full.mp4` and whose disfluency window is regenerated.
    - A `timing.json` splits measured wall into APPLY (transcript + splice + crop) + LTX (the one model call) + POST (composite/mux); a `manifest.json` records source sha256, span `[5.20, 5.54]`, `a`/`total`/`bridge`/`lb0`/`lb1`, resolution, fps, seed, model/LoRA/quant paths, and both repos' git revisions.
    - The same case with `--width 1920 --height 1088` (~1080p) succeeds under upstream `--offload-mode cpu`.
  - Negative Tests (expected to FAIL):
    - A run that silently reuses a stale `final.mp4` (unchanged mtime / manifest sha256 mismatch) is rejected by the check driver.
    - A `timing.json` whose APPLY+LTX+POST do not sum to within tolerance of measured wall is rejected.

- AC-2: An additive `--external-retake <mp4>` flag on `delete_disfluency.py` (and nowhere else) makes step 5 use the provided video instead of calling `RetakePipeline`, while every other path stays identical to the unmodified script.
  - Positive Tests:
    - With `--external-retake X.mp4`, `RetakePipeline` is never constructed nor called; `X.mp4` is used as `retake_output` for step-6 `composite_bridge`; steps 1–4 and 6 run unchanged.
    - Without the flag, the script's behavior and outputs are identical to the current committed script on a fixed seed (baseline diff shows no change).
  - Negative Tests:
    - A run with `--external-retake` set that still constructs/calls `RetakePipeline` (rather than short-circuiting step 5) is rejected.
    - Any edit under `../LTX2.3-eval/packages/` or under `tensorrt_llm/` for this feature fails review (pristine-tree check).

- AC-3: A seam-parity preflight validates the external retake mp4 before `composite_bridge`, rejecting artifacts that would corrupt the splice.
  - Positive Tests:
    - An mp4 whose resolution == the processing size, whose frame count is **exactly** the `retake_input.mp4` frame count (`total`), whose fps/timebase matches `retake_input.mp4` within tolerance, and whose pixel format is decodable — passes and composites cleanly.
  - Negative Tests:
    - An mp4 with wrong resolution, wrong frame count (≠ `total`), or mismatched fps is rejected with a specific diagnostic naming the mismatched field. No silent scale/pad/resample.

- AC-4: Every one of the 7 variants (upstream; native bf16/FP8/NVFP4; serve bf16/FP8/NVFP4) either produces a full composited `final_<variant>.mp4` through the same step-6 path, or is recorded as `oom` / `unsupported` / `timeout` with logs — never a silent gap.
  - Positive Tests:
    - Each TRT variant runs its retake on the dumped `retake_input.mp4` with model window `[lb0/fps, lb1/fps]`, returns a `total`-frame video, and yields `final_<variant>.mp4`.
    - A variant that OOMs at 1080p is recorded as `oom` and does not abort the remaining variants (subprocess-per-item isolation).
  - Negative Tests:
    - A variant marked `ok` without a decodable `final_<variant>.mp4` of the expected duration/resolution is rejected.

- AC-5: Per-variant timing follows the §2 stage map and reports both single-shot and warm p50, plus peak memory and status.
  - Positive Tests:
    - Each row carries APPLY / LTX / POST, one cold single-shot latency, a warm p50 (resident native + serve), peak reserved memory, quant mode, native-vs-serve mode, and status; serve rows additionally split cold-start / first-served / warm and HTTP-wall vs engine time.
    - APPLY and POST for TRT rows are the shared Phase-A/Phase-C costs, clearly labeled as reused; upstream carries its own per-run APPLY/POST.
  - Negative Tests:
    - Mixing a warm p50 sample into the single-shot number, or presenting reused APPLY/POST as if independently re-measured for a TRT row, is rejected by the aggregation schema.

- AC-6: Regenerated-window quality is reported as informational signal only and never gates pass/fail; a source-vs-upstream-vs-variant frame grid is produced for human review.
  - Positive Tests:
    - PSNR/SSIM of the regenerated **window** (edited region only, not the whole clip) vs the bf16/upstream reference is emitted per variant; a side-by-side frame grid at the cut seam is written for eyeball review of FP8/NVFP4 drift.
    - Frames of each `final_<variant>.mp4` outside `[a, a+total)` are **decode-tolerant-equal** to `edited_full.mp4` (both are H.264 CRF-16 re-encodes, so this is a tolerant decoded comparison, not byte identity). Byte-level composite identity is asserted only at the TRT tensor/PT level via the existing `ltx2_retake_oracle.py` harness, not on the deliverable mp4.
  - Negative Tests:
    - A run that gates correctness on a PSNR/SSIM threshold is rejected (quality is informational by project rule).
    - A `final_<variant>.mp4` whose outside-window frames diverge from `edited_full.mp4` beyond decode tolerance is rejected.

- AC-7: Audio in every TRT `final_<variant>.mp4` derives from the upstream spliced audio (the retake is video-only), asserted — not TRT-generated and not from the external retake mp4.
  - Positive Tests:
    - Step 6 muxes video from `retake_out` (`-map 0:v`) and audio from `edited_full.mp4` (`-map 1:a`, `-c:a aac`); the assertion confirms `final` audio matches the `edited` audio in duration/content (an AAC transcode, not stream byte-equality), and the external retake mp4's audio (if any) is ignored.
  - Negative Tests:
    - A final whose audio came from the external retake mp4, or is missing, is rejected.

- AC-8: The `--external-retake` code path is exercised once end-to-end on the GPU before the feature is considered done (project hard rule).
  - Positive Tests:
    - At least one real GPU run of `delete_disfluency.py --external-retake ...` on the case produces a `final.mp4`, logged with the node/container.
  - Negative Tests:
    - Host-only validation of the flag logic, with no GPU run, does not satisfy AC-8.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
The full 7-variant matrix at 720p plus 1080p for the variants that fit; each variant produces a reviewable `final_<variant>.mp4`; a seam-parity preflight (exact frame count / resolution / fps / pix_fmt); a §2-aligned timing table with single-shot + warm p50 + peak memory + status; informational PSNR/SSIM + a source/upstream/variant frame grid; audio-derivation and outside-window decode-identity assertions; a two-repo provenance manifest (both git revisions pinned). One additive `--external-retake` flag on the eval harness is the only upstream/product code change; all measurement/aggregation is in separate test-driver scripts.

### Lower Bound (Minimum Acceptable Scope)
Upstream baseline + at least native bf16, native FP8, and one serve variant, each producing a `final.mp4` at 720p through the `--external-retake` seam, with APPLY/LTX/POST single-shot timing and the audio-derivation + outside-window-decode-identity assertions. NVFP4, 1080p, warm-p50, and the frame grid may degrade to `unsupported`/`skipped` **only** if recorded as an explicit gap.

### Allowed Choices
- Can use: the existing `examples/visual_gen/ltx2_retake_*.py` harnesses (oracle / timing / serve_timing / quant_mem) to drive the TRT retakes; a separate workstream test-driver for orchestration/timing/aggregation; `ffprobe`/PyAV/OpenCV for preflight + metrics; subprocess-per-item isolation for near-capacity rows; upstream `--offload-mode cpu` for the upstream 1080p baseline; explicit landscape `--width/--height` on every invocation (eval defaults are portrait 704×1280 and must be overridden).
- Cannot use: any modification to `../LTX2.3-eval/packages/` or to `tensorrt_llm/`; any resample of a variant's output to a different resolution before compare (compare within-resolution only); any gating of correctness on PSNR/SSIM; any claim of mp4 byte-identity (the final is CRF-16 re-encoded).

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
Three phases:
- **Phase A (dump, once):** run upstream steps 1–4 to materialize `retake_input.mp4` (a `total`-frame window, landscape processing size) + `geometry.json` (`a`, `total`, `bridge`, `lb0`, `lb1`, `fps`, `w×h`, `splice_frame`) and the APPLY-block timing. Shared input for all TRT variants.
- **Phase B (per-engine retake):** feed `retake_input.mp4` to the TRT retake with model window `[lb0/fps, lb1/fps]` (the middle bridge — TRT's `_retake_pixel_window` uses `round(t*fps)`, so `lb0/fps, lb1/fps` recovers exactly `[lb0, lb1]`). Native via the oracle/timing harness in-process; serve via `POST /v1/videos/generations` with `retake_video_path/retake_start_time/retake_end_time`. TRT composites internally and returns a `total`-frame video → `retake_<variant>.mp4`. Serve needs a quantized (FP8/NVFP4) server config and an **artifact-producing** call (not the `format='pt'` pure-latency path) so a decodable mp4 is saved.
- **Phase C (composite, per variant):** call `delete_disfluency.py --external-retake retake_<variant>.mp4` so upstream step-6 `composite_bridge(edited, retake_out, a, 0, total, fps, final)` splices the whole window back → `final_<variant>.mp4`, exercising the preflight.

### Relevant References
- `LTX2.3-eval/script_editing/delete_disfluency.py` — `crop_window` (line 139, retake_input), `composite_bridge` (line 160, splice-back `[0,total)`), `main()` geometry (lines 262–336); add `--external-retake` here.
- `tensorrt_llm/_torch/visual_gen/models/ltx2/pipeline_ltx2_retake.py` — `_retake_pixel_window`, `_composite_retake_window`, `infer()` (returns full same-length video, `frame_rate=fps`).
- `examples/visual_gen/ltx2_retake_oracle.py` / `ltx2_retake_serve_timing.py` — native + serve drivers; `build_retake_payload` (serve payload; note `format='pt'` is pure-latency, needs a video-format variant for artifacts).
- `examples/visual_gen/RETAKE_ACCELERATION_REPORT.md` §2/§3 — resolution/memory + quant fidelity anchors.

## Dependencies and Sequence

### Milestones
1. Harness + seam: add `--external-retake` to `delete_disfluency.py`; add the frame-count/resolution/fps preflight; GPU-verify the flag path once.
   - Phase A: dump `retake_input.mp4` + geometry.
2. Upstream baseline (AC-1): full upstream run at 720p (and 1080p under cpu-offload) with §2 timing + manifest.
3. TRT variant sweep (AC-4/5): native{bf16,fp8,nvfp4} + serve{bf16,fp8,nvfp4} retakes on the dumped window (subprocess-isolated), then Phase-C composite via the seam; per-row timing/memory/status.
4. Quality + audio (AC-6/7): informational PSNR/SSIM (window-only), source/upstream/variant frame grid, audio-derivation + outside-window decode-identity assertions.
5. Aggregate (AC-5): one report table over all variants (720p full + 1080p-that-fit); gaps recorded explicitly; two-repo provenance.

<Dependencies are relative, not time-based: task3 (dump) gates the whole sweep; task1/task2 (flag+preflight) gate every composite.>

## Task Breakdown

Each task must include exactly one routing tag: `coding` (Claude) or `analyze` (Codex).

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Add additive `--external-retake` flag to `delete_disfluency.py` (skip step-5 `RetakePipeline` construction+call, use mp4 for step-6); keep default path identical | AC-2 | coding | - |
| task2 | Add seam-parity preflight before `composite_bridge`: exact `total` frame count, resolution == processing size, fps/timebase, decodable pix_fmt | AC-3 | coding | task1 |
| task3 | Phase A: dump `retake_input.mp4` + `geometry.json` (a/total/bridge/lb0/lb1/fps/w×h/splice_frame) from upstream steps 1–4 | AC-1 | coding | - |
| task4 | Upstream baseline full run at 720p + 1080p(cpu-offload) with §2 APPLY/LTX/POST timing + manifest (both repo revs) | AC-1,AC-5 | coding | task3 |
| task5 | Phase B: drive native bf16/fp8/nvfp4 retakes on the dumped window (oracle/timing harness), subprocess-isolated, window `[lb0/fps,lb1/fps]` | AC-4,AC-5 | coding | task3 |
| task6 | Phase B: drive serve bf16/fp8/nvfp4 retakes — quantized server config + artifact-producing (non-`pt`) call saving a decodable mp4; cold/first/warm split | AC-4,AC-5 | coding | task3 |
| task7 | Phase C: composite every variant via `--external-retake` → `final_<variant>.mp4` (exercises preflight) | AC-4 | coding | task1,task2,task5,task6 |
| task8 | Quality: informational PSNR/SSIM (window-only) + source/upstream/variant frame grid; record ffprobe metadata for retake_input / retake_<variant> / final_<variant> | AC-6 | coding | task7 |
| task9 | Assertions: final audio derives from `edited` (AAC transcode, not external-retake audio); outside-window decode-tolerant identity vs `edited` | AC-7,AC-6 | coding | task7 |
| task10 | Aggregate all rows into one report (720p full + 1080p-that-fit); record gaps; two-repo provenance | AC-5 | coding | task4,task7,task8,task9 |
| task11 | GPU-verify the `--external-retake` path once on the node/container | AC-8 | coding | task7 |
| task12 | Cross-check timing-attribution + fairness (locked prompt/seed/steps/LoRA/LoRA-strength across upstream and TRT; APPLY/POST reuse honesty) | AC-5 | analyze | task4,task5,task6 |

## Claude-Codex Deliberation

### Agreements
- Reuse upstream steps 1–4 + 6 unchanged; swap only step 5; single additive flag on the eval harness; measurement drivers are separate test scripts.
- Seam parity (exact `total` frame count / resolution / fps / pix_fmt) must be preflight-validated, not assumed.
- Quality is informational; correctness gates on structure + audio-derivation + outside-window decode-identity.
- Near-capacity native rows run subprocess-isolated so one OOM never poisons the sweep.

### Resolved Disagreements
- "Does TRT emit a full composited video or just the window?": resolved by code — native `infer()` composites internally and returns a full same-length video, so it drops directly into `composite_bridge` as the step-5 replacement.
- "Composite bounds": the working `delete_disfluency.py` calls `composite_bridge(edited, retake_out, a, 0, total, fps, final)` — the **whole** `total`-frame window is spliced (the model regenerates only the middle bridge). Preflight therefore requires **exactly `total`** frames, and the outside-window region is everything outside `[a, a+total)`.
- "MP4 byte-identity": `composite_bridge` re-encodes all video H.264 CRF-16 and audio to AAC, so the deliverable mp4 is **not** byte-identical. Outside-window identity becomes a tolerant decoded comparison vs `edited`; byte-identity is asserted only at the TRT tensor/PT level (oracle harness).
- "Audio validation": final audio is an AAC transcode mapped from `edited` (`-map 1:a`); assert it derives from `edited` (external-retake audio ignored), not stream byte-equality.
- "Serve FP8/NVFP4 + artifacts": the existing serve-timing harness is bf16 and `format='pt'` only; task6 adds a quantized server config and an artifact-producing call that saves a decodable mp4.
- "Resolution defaults": eval defaults are portrait 704×1280; pass explicit landscape `--width/--height` (1280×704 / 1920×1088) on every invocation.

### Convergence Status
- Final Status: `converged`

## Pending User Decisions

All five design decisions were resolved with the user before planning; recorded here as settled (Decision Status is the user's final decision, not `PENDING`).

- DEC-1: Seam mechanism. Claude Position / Codex Position: additive `--external-retake` flag on the eval harness (not `packages/`, not trtllm). Tradeoff Summary: minimal, additive, keeps upstream glue byte-for-byte; alternative was an external orchestrator importing the pure helpers. Decision Status: **DECIDED — Option A (`--external-retake`).**
- DEC-2: Resolution. Tradeoff Summary: 1080p offload=none likely OOMs at ~`total` frames; 720p is fully apples-to-apples. Decision Status: **DECIDED — full 7-variant matrix at 720p + 1080p only for variants that fit; compare within-resolution.**
- DEC-3: Serve scope. Decision Status: **DECIDED — run all three serve variants (bf16/FP8/NVFP4).**
- DEC-4: Warm vs single-shot. Tradeoff Summary: single-shot is directly comparable to upstream's inherent single-shot; warm p50 shows amortized/resident wins; must not imply a single cold edit is ~46× faster. Decision Status: **DECIDED — report both (single-shot primary, warm p50 supplementary).**
- DEC-5: Final resolution / resample. Tradeoff Summary: `splice_out` scales the whole clip to `w×h` and the final stays there (no `match_source_dims` in the checked-in script). Decision Status: **DECIDED — no resample; compare within-resolution.**
- DEC-6: Pass criterion for FP8/NVFP4. Claude Position: structural validity + audio-derivation + produces final video; PSNR/SSIM informational for human eyeball. Codex Position: agreed (aligns with established project methodology). Decision Status: **DECIDED — structure/audio gate; quality informational.**

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead.

### Scope & Verification
- The only change to upstream/product code is the additive `--external-retake` flag (+ its preflight) on `LTX2.3-eval/script_editing/delete_disfluency.py`. No `tensorrt_llm/` or `../LTX2.3-eval/packages/` edits. All timing/manifest/stale-check/aggregation/serve-artifact-capture logic lives in separate workstream test-driver scripts.
- Every code change is GPU-verified once on the node/container (project hard rule) — host validation of the flag logic is necessary but not sufficient.
- Always pass explicit landscape `--width/--height`; the eval defaults are portrait (704×1280) and would rotate the aspect.
- The manifest pins both repositories' git revisions (TensorRT-LLM and the working-tree `LTX2.3-eval`, whose `composite_bridge(… a, 0, total …)` is authoritative for step 6).

## Output File Convention

This template is used to produce the main output file (e.g., `plan.md`).

### Translated Language Variant

When `alternative_plan_language` resolves to a supported language name through merged config loading, a translated variant of the output file is also written after the main file. Humanize loads config from merged layers in this order: default config, optional user config, then optional project config; `alternative_plan_language` may be set at any of those layers. The variant filename is constructed by inserting `_<code>` (the ISO 639-1 code from the built-in mapping table) immediately before the file extension:

- `plan.md` becomes `plan_<code>.md` (e.g. `plan_zh.md` for Chinese, `plan_ko.md` for Korean)
- `docs/my-plan.md` becomes `docs/my-plan_<code>.md`
- `output` (no extension) becomes `output_<code>`

The translated variant file contains a full translation of the main plan file's current content in the configured language. All identifiers (`AC-*`, task IDs, file paths, API names, command flags) remain unchanged, as they are language-neutral.

When `alternative_plan_language` is empty, absent, set to `"English"`, or set to an unsupported language, no translated variant is written. Humanize does not auto-create `.humanize/config.json` when no project config file is present.

--- Original Design Draft Start ---

# draft_plan_test.md — end-to-end delete-disfluency: upstream vs TRT-LLM retake

> Status: **DRAFT for review.** No GPU runs started yet. Open questions in §8 need
> a decision before execution.

## 0. Goal

Run the **first `delete_disfluency` eval case** end-to-end and produce, for each
engine variant, a **final composited video** (disfluency removed, bridge stitched
back into the original) plus a **per-stage latency breakdown** aligned to the
`latency_report.pdf` §2 pipeline stage map.

Two things must come out of this:
1. **Performance + quality comparison** of the *retake* step: upstream vs TRT-LLM
   (bf16 / FP8 / NVFP4, each in **native** and **serve** form).
2. **Proof the TRT-LLM retake really composites back** into the source to achieve
   the delete-disfluency effect — i.e. a real, reviewable `final.mp4` per variant,
   not a retake window in isolation.

The compositing/splice work stays in the **upstream** pipeline; TRT-LLM code and
the LTX `packages/` are **not modified**.

### The case (from `eval_cases.json[0]`)
```
clip_id        --Y9imYnfBw-f0-484   (Bill Gates "world's richest man")
task           delete_disfluency
which=0        word_index=16  filler_text=[UH]  span_s=[5.20, 5.54]
video          LTX2.3-eval/clips/--Y9imYnfBw-f0-484.mp4   (1920x1080, 29.97 fps, 16.15 s, 484 f)
transcript     LTX2.3-eval/script_editing/eval/transcripts/--Y9imYnfBw-f0-484.json
width=1920  height=1088  fps=29.97
```
Retake geometry (upstream defaults): `cond-frames=90` each side + `retake-frames=25`
bridge, snapped to `8k+1` → **retake window ≈ 205 frames** straddling the 5.2 s cut.
(The full 484-frame video is only touched by CPU ffmpeg in APPLY/POST.)

---

## 1. Key facts & constraints

- **Hardware differs from the PDF.** `latency_report.pdf` was measured on **H100
  80GB ×8, torch 2.9.1**. Our runs are on **1× RTX PRO 6000 Blackwell 96GB (sm_120)**.
  → Absolute seconds will NOT match the PDF; we align to the **stage-map structure**
  and report our own device's numbers. State this on every table.
- **The retake window is ~205 frames at 1080p.** Our measured 1080p retake at
  **89 frames** already reserves **91.5 GiB** (offload=none). 205 frames ≈ 2.3× the
  latent tokens → **bf16 native offload=none will very likely OOM at 1080p**. See §5
  for the resolution/offload decision.
- **Upstream fits 1080p via `--offload-mode cpu`** (its default; reference outputs
  exist at 720p and 1080p). To compare apples-to-apples, our TRT-LLM retake must run
  at the **same resolution** as upstream in each comparison.
- **serve fine-stages are coarser.** Over HTTP the engine only reports `generation`
  + `denoise` in `Server-Timing`; the fine LTX sub-stages (vae_encode, text_encode,
  vae_decode) are only available on the **native/in-process** path. serve variants
  therefore get a coarser LTX-block breakdown — expected, not a defect.
- **TRT-LLM retake is video-only** (`regenerate_audio=False`); it preserves source
  audio via PyAV and does **not** run audio_vae_encode/decode/vocoder. Those rows are
  N/A for the TRT-LLM LTX block (upstream has small ~0.5 s + 0.3 s entries).

---

## 2. Test matrix (7 variants)

Each variant produces one `final.mp4` + one `timing.json`.

| # | Engine | Quant | Form | Retake produced by | Composite by |
|---|--------|-------|------|--------------------|--------------|
| 1 | **upstream** (reference) | fp8-cast (its default) | in-process | upstream `RetakePipeline` (full `delete_disfluency.py`) | upstream |
| 2 | TRT-LLM native | bf16 | in-process | our `ltx2_retake` native path | upstream step 6 |
| 3 | TRT-LLM native | FP8 | in-process | our native path | upstream step 6 |
| 4 | TRT-LLM native | NVFP4 | in-process | our native path | upstream step 6 |
| 5 | TRT-LLM serve | bf16 | HTTP | `trtllm-serve` retake | upstream step 6 |
| 6 | TRT-LLM serve | FP8 | HTTP | `trtllm-serve` retake | upstream step 6 |
| 7 | TRT-LLM serve | NVFP4 | HTTP | `trtllm-serve` retake | upstream step 6 |

All 7 share the **identical** APPLY (①) and POST (③) blocks (same upstream ffmpeg
glue, same conditioned input, same splice) so only block ② (the model retake)
differs.

---

## 3. Pipeline integration design (no trtllm / no packages changes)

`delete_disfluency.py` cleanly separates the model call from the glue:
- **steps 1–4** build the *edited clip* + the *retake input window* (`crop_window`),
- **step 5** is the single `RetakePipeline` call (the only model inference),
- **step 6** `composite_bridge(edited, retake_output, …)` splices the retake window
  back and writes `final.mp4` + `timing.json`.

**Injection point = between step 4 and step 6.** To swap in the TRT-LLM retake while
reusing upstream's glue, add a thin **two-phase seam** to `script_editing/`
(this is the eval harness — **not** `packages/`, and **not** the trtllm repo):

- **Phase A (upstream, once):** run steps 1–4 and dump
  (a) `retake_input.mp4` (the window the model must regenerate) and
  (b) `geometry.json` (`a`, `lb0`, `lb1`, `fps`, `width`, `height`, `splice_frame`,
  the `apply`+geometry timings).
- **Phase B (per engine variant):** run the retake on `retake_input.mp4` →
  `retake_output_<variant>.mp4`.
  - upstream variant: just let `delete_disfluency.py` run step 5 normally.
  - trtllm variants: our runner (native or serve) reads `retake_input.mp4`,
    regenerates the window, writes `retake_output_<variant>.mp4`.
- **Phase C (upstream, per variant):** run **step 6 only** (`composite_bridge`) with
  the chosen `retake_output_<variant>.mp4` → `final_<variant>.mp4` + `timing.json`.

Implementation options for the seam (pick in §8):
- **(pref) small additive flags** on `delete_disfluency.py`:
  `--dump-retake-input <path>` (stop after step 4) and
  `--external-retake <path>` (skip step 5, use this as the retake output for step 6).
  ~30 lines, additive, keeps LTX `packages/` pristine.
- **(alt) thin orchestrator** in the workstream that `import`s the pure helpers
  (`find_disfluency`, `splice_out`, `crop_window`, `composite_bridge`, `read_frames`)
  and drives A/B/C — zero edits to LTX2.3-eval.

Either way: **trtllm code unchanged; LTX `packages/` unchanged.** The `--start 5.20
--end 5.54 --which 0` overrides use the eval case's exact span (skip re-detection).

---

## 4. Stage map & metrics (aligned to `latency_report.pdf` §2)

Report every variant against the same three-block map; only block ② is re-measured
per engine, blocks ① and ③ are measured once (shared, from Phase A/C).

| Block | Stage | Measured where | upstream | trtllm native | trtllm serve |
|-------|-------|----------------|----------|---------------|--------------|
| ① APPLY | whisper ASR / propose edits / prepare_base_video / make_conditioned_input | Phase A (shared) | ✅ | ✅ (reused) | ✅ (reused) |
| ② LTX | subprocess_start / model_load | per variant | ✅ | ✅ (`model_build_load`) | resident (once) |
| ② LTX | video_vae_encode | per variant | ✅ | ✅ (`vae_encode`) | — (HTTP) |
| ② LTX | text_encode (Gemma) | per variant | ✅ | ✅ (`conditioning`) | — (HTTP) |
| ② LTX | **diffusion (8 steps)** | per variant | ✅ | ✅ (`denoise_total` + `denoise_per_step`) | ✅ (`denoise`) |
| ② LTX | video_vae_decode | per variant | ✅ | ✅ (`vae_decode`) | — (HTTP) |
| ② LTX | audio_vae_encode/decode | per variant | ✅ (small) | N/A (video-only) | N/A |
| ② LTX | mp4_encode (window) | per variant | ✅ | ✅ (encode_mp4) | ✅ |
| ② LTX | (serve only) generation / wall | per variant | — | — | ✅ (`Server-Timing` + client wall) |
| ③ POST | ffmpeg_splice_chunks / post_process_match_source_dims | Phase C (shared) | ✅ | ✅ (reused) | ✅ (reused) |
| — | **end-to-end wall** | apply + LTX + post | ✅ | ✅ | ✅ |

Measurement rules (same as our AC-1 harness): CUDA events for GPU stages +
`torch.cuda.synchronize()`-bracketed wall; **N warmup + M measured** for the retake
(warm p50/p90/min); cold `model_load` reported once and NOT folded into per-call
warm totals for the resident (serve) form; upstream/native every-call form DOES
include `model_load` (it reloads per call).

---

## 5. Resolution & memory plan

Decision gate before the full matrix:

1. **Capacity pre-check** (cheap, 1 build): run the TRT-LLM native retake once on the
   real ~205-frame window at **1080p (1920×1088), offload=none, bf16**. Record
   OK / OOM + peak reserved.
2. **Branch:**
   - **If 1080p bf16 native fits** → run the whole matrix at 1080p.
   - **If it OOMs** (expected): choose ONE of, in order of preference —
     - **(a) 720p (1280×704) for the whole matrix** — the report's other regime,
       everything fits comfortably, fully apples-to-apples. *(recommended default)*
     - (b) 1080p with `offload-mode cpu` for the native/serve retake too (matches
       upstream's 1080p path) — fits but adds offload time; note it.
     - (c) 1080p with a **reduced window** (e.g. `cond-frames 30`) so it fits
       offload=none — changes geometry vs the eval default; note it.

FP8 / NVFP4 free ~18 / ~25 GiB, so they may fit 1080p offload=none even when bf16
does not — the matrix will record per-mode status (`ok` / `oom`) rather than assume.

**Recommendation:** run the primary matrix at **720p** for a clean 7-way comparison,
and additionally attempt **1080p** for whichever modes fit (at least upstream, and
FP8/NVFP4 native) as a stretch, marking OOM where it happens.

---

## 6. Deliverables (for review)

Under `Script_Editing_Workstream/test_outputs/<res>/`:
- `final_upstream.mp4`, `final_native_bf16.mp4`, `final_native_fp8.mp4`,
  `final_native_nvfp4.mp4`, `final_serve_bf16.mp4`, `final_serve_fp8.mp4`,
  `final_serve_nvfp4.mp4` — **7 complete delete-disfluency videos**.
- `retake_output_<variant>.mp4` — the raw regenerated window per variant.
- `timing_<variant>.json` — per-stage breakdown (§4 schema).
- `latency_comparison.md` + a stage-map table (our device, upstream vs 6 trtllm).
- `quality_comparison`: (a) each `final_*` for eyeball; (b) informational PSNR/SSIM of
  each trtllm `retake_output` vs the upstream `retake_output` (window only), and of
  each `final_*` vs `final_upstream` (whole frame); (c) a frame-grid at the cut seam
  showing the bridge is seamless.
- Optional: fold results into an updated one-page report / the customer deck.

---

## 7. Execution steps (ordered)

1. **Setup check (cluster):** confirm the upstream `delete_disfluency.py` runs
   (LTX2.3-eval `.venv` or our container has `ltx_pipelines` + `ffmpeg` on PATH +
   `models/` wired to our `ltx-retake-assets`), and CrisperWhisper transcript for
   `--Y9imYnfBw-f0-484` is present (it is, in `eval/transcripts/`).
2. **Add the two-phase seam** (§3) — flags or orchestrator; unit-smoke on CPU glue.
3. **Capacity pre-check** (§5) → pick resolution(s).
4. **Variant 1 (upstream):** full `delete_disfluency.py` → `final_upstream.mp4` +
   timing (this also produces the shared `retake_input.mp4` + APPLY/POST timings).
5. **Variants 2–4 (native bf16/fp8/nvfp4):** retake on `retake_input.mp4` → composite
   (step 6) → `final_native_*` + retake timing (reuse APPLY/POST from step 4).
6. **Variants 5–7 (serve bf16/fp8/nvfp4):** launch `trtllm-serve` per quant, HTTP
   retake on the same window → composite → `final_serve_*` + serve timing.
7. **Assemble** latency + quality comparison, pull all videos locally, review.
8. Per the standing rule: every code path exercised once on GPU before it's "done".

---

## 8. Decisions

1. **Seam mechanism — DECIDED: Option A (A-lite).** Add one additive flag
   `--external-retake <mp4>` (+ optional sidecar for `diffusion_seconds`) to
   `LTX2.3-eval/script_editing/delete_disfluency.py` (not `packages/`, not trtllm):
   when set, skip the step-5 `RetakePipeline` call, use the provided mp4 as the
   retake output, and run step 6 `composite_bridge` + timing as normal. Steps 1–4
   (cheap: read transcript json + 2 ffmpeg ops; whisper ASR is done separately in
   `transcribe.py`) recompute the identical `a/total/bridge` geometry, so the glue
   is byte-for-byte the upstream path. Diff kept isolated as a `.patch`.
   - Upstream variant: run normally (no flag) → produces the shared
     `retake_input.mp4` + `edited_full.mp4` + `final_upstream.mp4` + timing.
   - trtllm variants: retake on upstream's `retake_input.mp4` → `my_out.mp4` →
     `delete_disfluency.py --external-retake my_out.mp4` → `final_<variant>.mp4`.

2. **Resolution — DECIDED:** full **7-way matrix at 720p** (1280×704; all fit) +
   **1080p (1920×1088) only for modes that fit** (upstream via offload cpu; trtllm
   modes recorded `ok`/`oom` per the §5 pre-check — FP8/NVFP4 more likely than bf16).
   Quality compared **within each resolution** (720p group; 1080p group).
3. **serve scope — DECIDED:** run **all three** serve variants (bf16 / FP8 / NVFP4).
4. **Warm vs single-shot — DECIDED: report both.** Single-shot (one real retake per
   variant → the actual `final.mp4`, directly comparable to upstream's inherent
   single-shot) as primary; warm p50 (trtllm native/serve, load-once amortized) as a
   supplementary column. Analysis must separate **retake compute** (native ~8.5×
   faster than upstream's retake) from **incl. model load** (a single cold edit is
   ~comparable — trtllm's cold build is heavier; the ~46× win is amortized/resident
   only). Do NOT imply single cold edits are 46× faster.
5. **Final resolution — DECIDED (verified in code):** `delete_disfluency.py` does
   **NOT** resample — `splice_out` scales the whole clip to `args.width×args.height`
   and the final stays there (no `match_source_dims` in the checked-in script). Each
   variant's `final.mp4` is at its processing resolution (720p → 1280×704, 1080p →
   1920×1088); no artificial resample; compare within-resolution.

## 9. Risks

- 1080p offload=none OOM (expected) — mitigated by §5 branch.
- Upstream env / `models/` wiring on the cluster may need a one-time setup (§7.1).
- serve fine-stage granularity is coarser (HTTP) — documented, not fixable without a
  server-side change (out of scope).
- Absolute latencies are device-specific (6000 PRO, not the PDF's H100) — always
  labeled.

--- Original Design Draft End ---
