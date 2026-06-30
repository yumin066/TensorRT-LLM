<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->

# Round 0 Contract

## Mainline Objective

Establish the AC-1 discovery base for Qwen-Image-Layered MXFP8 QAT by identifying the existing generation entry points, VisualGen benchmark path, current `K=12` configuration, BF16/reference flow candidates, and transformer inputs that can be cached for offline QAT.

## Target Acceptance Criteria

- AC-1: Reproducible Qwen-Image-Layered quality evaluation loop.

## Required Work

- Keep the immutable goal and acceptance criteria anchored in `goal-tracker.md`.
- Run BitLesson selection for each round task before acting on it.
- Execute the analyze-scoped survey via the Humanize ask-codex workflow and integrate the returned findings.
- Record local verification and repository survey findings in `round-0-discovery.md`.
- Leave implementation of eval/cache tooling for task2; this round should not modify VisualGen runtime or public APIs.

## Success Conditions

- `goal-tracker.md` contains the ultimate goal, independently verifiable acceptance criteria, active task map, and queued non-blocking decisions.
- `round-0-discovery.md` lists concrete repository paths and recommended next implementation targets for task2.
- `round-0-summary.md` records files changed, commands run, BitLesson results, and remaining risks.
- The round artifacts are committed with a DCO sign-off.

## Out Of Scope For This Round

- Implementing QAT, fake quant modules, checkpoint export, or benchmark execution.
- Running GPU-heavy generation, B300 profiling, or K-sweeps.
- Modifying `tensorrt_llm/visual_gen/` public API.

## Side Issues

No blocking side issues are active for this round. Metric statistic, dataset scale, partial-unfreeze fallback details, and concrete artifact paths remain queued until the relevant implementation or execution task needs them.
