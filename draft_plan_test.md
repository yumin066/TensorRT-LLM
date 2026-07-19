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
