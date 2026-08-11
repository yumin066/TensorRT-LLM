# LTX-2.3 Retake — SM120 (RTX PRO 6000 Blackwell) reproduction

Reusable environment build + run scripts for the native LTX-2.3 22B retake
workflow (`../ltx2_retake_e2e.py`) on SM120 (Blackwell) GPUs. Encodes jaywan's
frozen SM120 contract: a pinned rc22 base image + the LTX python overlay from
this checkout + FlashInfer 0.6.17 (PR #4272, SM120 nvfp4 attention).

## Why the pins matter

- **Base image is a fixed rc22 digest**, not a tag. A mismatched image on
  Blackwell faults advanced CUDA ops (`unknown NVRM ioctl`) and crashes the
  container.
- **FlashInfer 0.6.17 from PR #4272** carries the SM120 nvfp4 attention kernels
  that stock PyPI flashinfer lacks. `setup_sm120_env.sh` clones it straight from
  GitHub (`refs/pull/4272/head` @ the pinned commit) so the build is
  reproducible for customers with no access to our cluster.
- **Overlay = this checkout's `_torch/visual_gen/**.py` + `_torch/modules/linear.py`**
  only. The rest of the repo python is newer than rc22 and would fail against
  rc22's pre-compiled `.so`.

## Files

| File | Role |
|------|------|
| `start_sm120_container.sh` | Build the delivery image on the pinned digest + start a detached container (as the host user, via gosu). |
| `setup_sm120_env.sh` | Inside the container: overlay LTX python, build+install flashinfer 0.6.17, add PyAV + import stubs, preflight self-check. |
| `run_recipes.sh` | Run the retake e2e for bf16 / fp8 / nvfp4, outputs to `/tmp`. |
| `lpips_eval.sh` | LPIPS-score the outputs on the bridge window [90,119). |
| `Dockerfile`, `entrypoint.sh` | Thin delivery layer (ffmpeg + drop-to-user entrypoint). |

## Runbook

```bash
# 1. Start the container (mounts this checkout + model/customer/flashinfer scratch).
bash examples/visual_gen/ltx2_retake_sm120/start_sm120_container.sh

# 2. Build the SM120 env (overlay + flashinfer 0.6.17 + stubs + preflight).
#    Watch that the container survives the flashinfer build on shared "gen"
#    nodes that reap non-allocation docker containers (~4 min).
docker exec -u "$(id -un)" ltx_sm120 bash \
  examples/visual_gen/ltx2_retake_sm120/setup_sm120_env.sh

# 3. Run the recipes (source = a 704x1280, 209-frame clip; window [90,119)).
SRC=/home/scratch.minyu_gpu/project/shopee/LTX2.3-script-editing/script_editing/eval/reference_outputs/720p/retake_output.mp4
docker exec -u "$(id -un)" ltx_sm120 bash \
  examples/visual_gen/ltx2_retake_sm120/run_recipes.sh "$SRC" bf16 fp8 nvfp4

# 4. Pull outputs to scratch + score LPIPS.
for r in bf16 fp8 nvfp4; do docker cp ltx_sm120:/tmp/retake_$r.mp4 ./rtx_retake_$r.mp4; done
docker exec -u "$(id -un)" ltx_sm120 bash \
  examples/visual_gen/ltx2_retake_sm120/lpips_eval.sh
```

## Cluster-specific defaults

Paths for the checkpoint, Gemma text encoder, and customer `ltx-pipelines`
default to the reproduction cluster's scratch layout. Override via env vars
(`LTX_MOUNTS`, `LTX_CHECKPOINT`, `LTX_TEXT_ENCODER`, `LTX_PIPELINES_ROOT`) for a
different host. FlashInfer PR #4272 is fetched from GitHub by default;
`FLASHINFER_SHARED_ARCHIVE` is an optional internal override for air-gapped
hosts (extracted-source tarball) and is not needed by customers.
