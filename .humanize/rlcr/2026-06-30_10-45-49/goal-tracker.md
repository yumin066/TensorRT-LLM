<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->

# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal

Implement an offline post-training QAT flow for Qwen-Image-Layered so every transformer Linear layer can use static MXFP8 (`FP8_BLOCK_SCALES`) while preserving the existing TRT-LLM VisualGen runtime path, CUDA Graph / `torch.compile` compatibility, and stable layered-output PSNR of at least 38 dB versus the BF16 baseline.

### Acceptance Criteria
<!-- Each criterion must be independently verifiable -->

1. Reproducible quality loop: fixed layered generation manifest, BF16 reference artifacts, transformer-level cache schema, and PSNR/SSIM metric JSON can be regenerated from recorded provenance.
2. Layer sensitivity evidence: `K=12`, `K=8`, `K=4`, and `K=0` MXFP8 sweeps identify which transformer blocks and Linear groups drive the PSNR drop.
3. Fake MXFP8 parity: training-time fake MXFP8 Linear matches TRT-LLM runtime `FP8_BLOCK_SCALES` semantics on synthetic tensors, boundary tensors, and captured real activations.
4. Offline QAT training: a native PyTorch loop reads cached transformer tuples, injects Linear-only fake MXFP8, supports LoRA/adapter-first training, and uses step-based smoke, pilot, formal, and fallback schedules.
5. Static checkpoint export: trained deltas are merged and exported as all-layer MXFP8 weights with block scales and ModelOpt-style metadata that the existing VisualGen loader can consume without public API changes.
6. Final quality and performance: the QAT all-layer MXFP8 checkpoint reaches the agreed PSNR gate, improves over untrained all-layer MXFP8, avoids BF16 fallback, and keeps the current MXFP8 CUDA Graph / `torch.compile` performance path.

---

## MUTABLE SECTION
<!-- Update each round with justification for changes -->

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | - | - |

#### Active Tasks
<!-- Mainline tasks only: each task must directly advance the current round objective and carry routing metadata -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| task2: Implement or organize fixed eval script and manifest/cache builder with PSNR/SSIM JSON and provenance | AC-1 | pending | coding | claude | Depends on task1 |
| task3: Run `K=12/K=8/K=4/K=0` sweep and generate sensitivity inputs | AC-2 | pending | analyze | codex | Depends on task2 |
| task4: Write layer sensitivity summary script and target layer set recommendation | AC-2 | pending | coding | claude | Depends on task3 |
| task5: Design fake MXFP8 Linear injection and activation snapshot collection without touching attention quant paths | AC-3, AC-4 | pending | analyze | codex | Depends on task4 |
| task6: Implement fake MXFP8 Linear helper and synthetic/real activation parity tests | AC-3 | pending | coding | claude | Depends on task5 |
| task7: Implement native PyTorch QAT loop with cached tuples, step schedules, and LoRA/adapter-first recipe | AC-4 | pending | coding | claude | Depends on task6 |
| task8: Choose LoRA-only formal QAT or sensitivity-guided partial-unfreeze fallback from pilot results | AC-4, AC-6 | pending | analyze | codex | Depends on task7 |
| task9: Implement static MXFP8 checkpoint export and ModelOpt-style metadata writing | AC-5 | pending | coding | claude | Depends on task7 |
| task10: Add VisualGen loader layer-coverage test for all-layer MXFP8 without BF16 fallback | AC-5, AC-7 | pending | coding | claude | Depends on task9 |
| task11: Run quality eval comparing BF16, `K=12`, all-layer dynamic MXFP8, and QAT all-layer MXFP8 | AC-6 | pending | analyze | codex | Depends on task9 |
| task12: Run B300 latency/profile validation for CUDA Graph, `torch.compile`, and MXFP8 runtime path | AC-7 | pending | analyze | codex | Depends on task10 |
| task13: Clean scope, docs, and tests for reviewable PR stack | AC-8 | pending | coding | claude | Depends on task11 and task12 |

### Blocking Side Issues
<!-- Only issues that directly block current mainline progress belong here -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|

### Queued Side Issues
<!-- Non-blocking issues stay queued and must NOT replace the round objective -->
| Issue | Discovered Round | Why Not Blocking | Revisit Trigger |
|-------|-----------------|------------------|-----------------|
| Exact PSNR statistic for the 38 dB gate | 0 | Plan defaults to per-required-sample minimum, which is strict enough to proceed with tooling | User or validation evidence chooses average, p10, or dual gate |
| QAT dataset scale and train/validation split | 0 | Data type and schema are defined; implementation can begin with a small fixed manifest | Before formal pilot training |
| First fallback recipe details for partial unfreeze | 0 | LoRA/adapter-first path is accepted; fallback depends on sensitivity and pilot results | After task8 analysis |
| Concrete GPU/model/data artifact paths | 0 | Discovery and tooling can proceed without running full generation yet | Before executing BF16 reference generation or sweeps |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|
| AC-1 | task1: Qwen-Image-Layered discovery | 0 | 0 | `round-0-discovery.md`; ask-codex output `.humanize/skill/2026-06-30_10-48-29-30813-3e22c3d8/output.md`; local `rg`/`sed` survey |

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|
