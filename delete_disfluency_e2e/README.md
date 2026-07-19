<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# delete_disfluency end-to-end test harness

Reproducibility bundle for the one-case `delete_disfluency` test that compares the upstream
LTX2.3-eval retake against the TensorRT-LLM native/serve retake, and proves the TRT retake
composites back into a full delete-disfluency video. See `RETAKE_E2E_REPORT.md` for results.

## Two runtimes (this is the key)

- **TRT-LLM path** (native retake, serve, and the `--external-retake` composite): nvcr
  `tensorrt-llm/release:1.3.0rc20` (container `ltx_r35`). torch 2.11nv here has a **stubbed
  torchaudio**, which is fine because the native retake reads audio via PyAV and the composite
  skips the model call.
- **upstream path** (`RetakePipeline`, which encodes source audio for conditioning): its own
  `torchtrt-dev:cu13.2` container (`ltx_upstream`) with the faithful stack built by upstream's
  own `uv sync` → **torch 2.9.1 + torchaudio 2.9.1**. Running upstream in the trtllm image fails
  on the torchaudio stub — upstream must use its own env.

## The seam (only upstream/product code change)

`delete_disfluency_external_retake.patch` adds one additive flag `--external-retake <mp4>` to
`LTX2.3-eval/script_editing/delete_disfluency.py`: it skips the step-5 `RetakePipeline` call and
uses the supplied clip as the regenerated window, after a preflight (`preflight_external_retake`)
that requires the clip to match the retake-input geometry exactly (frame count == `total`,
resolution, fps, decodable pix_fmt). `tensorrt_llm/` and `../LTX2.3-eval/packages/` are unchanged.

## Flow

1. **Phase A** — run upstream `delete_disfluency.py` (no flag) once per resolution → `retake_input.mp4`
   (209 frames for this case) + geometry (window `[3.003,3.971]s`). Also yields the upstream baseline.
2. **Phase B** — TRT native retake on `retake_input.mp4`:
   - bf16: `examples/visual_gen/ltx2_retake_oracle.py --native-only`.
   - fp8 / nvfp4: `e2e_native_variant.py --quant fp8|nvfp4 --fast-init` (this dir). `--fast-init`
     no-ops the CPU weight init (overwritten by the checkpoint anyway), cutting the cold build
     from ~10 min to ~80–90 s; validated safe because fp8 stays near-lossless (31.8 dB vs bf16).
   - serve bf16: `examples/visual_gen/ltx2_retake_serve_timing.py` (HTTP Mode B). serve fp8/nvfp4
     are unsupported — the serve `pipeline_config` allowlist has no quantization key.
3. **Phase C** — `delete_disfluency.py --external-retake <trt_retake.mp4>` → `final_<variant>.mp4`.

## e2e_native_variant.py

Produces the deliverable clip + single-shot/warm timing + peak memory for one quant, reusing the
oracle's module-level helpers (`build_pipeline(quant_algo=…)`, `_make_request`, `_run_pipeline`,
`_video_to_thwc_uint8`, `encode_mp4`). One quant per process for OOM isolation.

```
python e2e_native_variant.py --repo <TensorRT-LLM> \
  --checkpoint <ltx2-22b-distilled dir> --gemma <gemma-qat> --lora <talkvid lora> \
  --source <retake_input.mp4> --start 3.003 --end 3.971 --quant fp8 --fast-init \
  --output-dir <out> --code-commit <rev>
```

## Artifacts

Final videos + timing JSON + frame grid are pulled to the repo-root `artifacts/` directory
(gitignored). This bundle tracks only the reproducible code + the report.
