<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->

# Round 0 Discovery

## Scope

This round was a read-only AC-1 survey for the Qwen-Image-Layered all-layer MXFP8 QAT plan. No GPU-heavy generation, K-sweep, or runtime code modification was attempted.

BitLesson selector result for this task: `LESSON_IDS: NONE`; the BitLesson file has no recorded lessons yet.

Humanize ask-codex output: `.humanize/skill/2026-06-30_10-48-29-30813-3e22c3d8/output.md`.

## Entry Points

- `examples/visual_gen/models/qwen_image.py` is the current offline example for Qwen-Image and Qwen-Image-Layered. Layered usage is documented in the module docstring, requires `--image` for layered models, accepts `--layers`, `--resolution`, `--cfg_normalize`, `--use_en_prompt`, and saves each layer plus an RGB composite through `_save_layered_output`.
- `tensorrt_llm/_torch/visual_gen/models/qwen_image/pipeline_qwen_image_layered.py` contains `QwenImageLayeredPipeline.forward`, the actual denoising loop for layered generation.
- `tensorrt_llm/_torch/visual_gen/models/qwen_image/transformer_qwen_image.py` contains `QwenImageTransformer2DModel.forward`, which already accepts the tensor tuple needed for transformer-level QAT data.
- `tests/unittest/_torch/visual_gen/test_qwen_image_layered_registry.py` and `tests/unittest/_torch/visual_gen/test_qwen_image_layered_pipeline_config.py` cover registration, default params, layered config fields, and tiny transformer sanity.

## Current K=12 Preset

- `examples/visual_gen/configs/qwen-image-layered-fp8-blockscale-edge-bf16-sage-fp8-1gpu.yaml` is the current quality preset.
- It sets `quant_config.quant_algo: FP8_BLOCK_SCALES` and `dynamic: true`.
- Its ignore list keeps `img_in`, `txt_in`, `norm_out`, `proj_out`, transformer blocks `0-11`, and transformer blocks `48-59` in BF16. That matches the preset where only the middle 36 transformer blocks use MXFP8 Linear.
- It sets `attention_config.backend: TRTLLM` with FP8 attention quant fields. QAT scope should keep attention fixed and only target Linear MXFP8.
- `cuda_graph_config.enable` is false in this YAML. The performance baseline from the previous session used additional CUDA Graph / compile settings, so the final benchmark config must preserve those settings explicitly rather than treating this YAML as the whole performance recipe.

## Cache Tuple Capture

The best transformer-level cache point is inside `QwenImageLayeredPipeline.forward` around the transformer call:

- `prompt_embeds`, `prompt_embeds_mask`, and `prompt_embeds_lens` are created before denoising.
- `latents` and `image_latents` are produced by `_prepare_layered_latents`.
- `img_shapes` is constructed from `layers`, output size, and the conditioning image size.
- `additional_t_cond` is currently a zero tensor for layered generation.
- Each denoise step builds `latent_model_input = torch.cat([latents, image_latents], dim=1)`.
- The transformer call passes `hidden_states=latent_model_input`, `timestep=timestep / 1000`, `encoder_hidden_states_mask`, `encoder_hidden_states`, `img_shapes`, `txt_seq_lens`, `additional_t_cond`, and `return_dict=False`.
- The BF16 teacher target can be the pre-slice transformer output from the BF16 model. For student training on generated layers only, the target should be sliced with `[:, :latents.size(1)]`, matching the runtime `noise_pred` path.

The top-level transformer signature already matches the planned cache schema: `hidden_states`, `encoder_hidden_states`, `encoder_hidden_states_mask`, `timestep`, `img_shapes`, `txt_seq_lens`, and `additional_t_cond`.

Implementation note for task2: a low-risk first version can wrap `pipeline.transformer.forward` from the offline eval/cache script to capture kwargs and output, avoiding a new public API. If true CFG is enabled, the cache schema must label conditional and negative-prompt transformer calls separately.

## Reference Artifacts

- The existing example writes layer PNGs and a composite PNG, which is useful for human inspection but not ideal as the only metric source because PNG conversion loses raw tensor structure.
- `VisualGenOutput.save` supports tensor formats such as `.safetensors` and `.pt`; for layered output the tensor is exposed through the `video` slot as `(B, layers, H, W, C)`.
- Task2 should save both raw tensor artifacts for PSNR/SSIM and optional PNG/composite artifacts for visual audit.
- The manifest should record prompt, input image path or latent input descriptor, seed, scheduler/timestep settings, output resolution, layer count, `cfg_normalize`, `use_en_prompt`, VisualGen YAML path, checkpoint path, and git commit.

## Benchmark And Eval Reuse

- `tensorrt_llm/bench/benchmark/visual_gen.py` is useful for offline latency JSON and supports `--visual_gen_args`, seed, prompt or prompt file, steps, size, concurrency, and result JSON.
- The current offline benchmark builds `VisualGenParams` from prompt/size/video-style fields and does not expose layered `image` or `extra_params`, so it is not enough for Qwen-Image-Layered quality evaluation as-is.
- The benchmark saves media only to a temporary directory for timing and deletes it afterward, so it should not be used as the persistent BF16 reference artifact path.
- `tensorrt_llm/serve/scripts/benchmark_visual_gen.py` is useful later for online serving performance, but quality/QAT data capture should start offline.
- `scripts/visualgen_eval/visual_gen_lpips_score_eval.py` and existing golden-media tests show a pattern for media-reference evaluation, but there is no checked-in Qwen-Image-Layered PSNR/SSIM comparator yet.

## Quantization Loading Context

- `tensorrt_llm/_torch/visual_gen/quantization/loader.py` performs dynamic load-time BF16/FP16/FP32 to FP8/NVFP4 quantization for `Linear`.
- `tensorrt_llm/_torch/visual_gen/quantization/ops.py` implements `quantize_fp8_blockwise` with 128x128 blockwise E4M3 weights and block scales.
- `tensorrt_llm/_torch/visual_gen/config.py` parses ModelOpt-style `quantization_config`, `ignore`, and dynamic flags. It also supports checkpoint-embedded `quantization_config` for safetensors.
- Existing tests in `tests/unittest/_torch/visual_gen/test_model_loader.py` verify VisualGenArgs quant config parsing and dynamic FP8 blockwise loading. Future static-QAT loader tests should build from this area.

## Task2 Recommendation

For the next coding task, implement a small offline eval/cache tool rather than changing public VisualGen APIs:

1. Add a manifest-driven Qwen-Image-Layered quality script under `scripts/visualgen_eval/` or `examples/visual_gen/`.
2. Load BF16, current K=12, all-layer MXFP8, and future QAT configs through existing `VisualGenArgs`.
3. Generate raw `.safetensors` or `.pt` layer-stack artifacts plus optional PNG/composite outputs.
4. Compute PSNR/SSIM against the BF16 raw tensor reference and write metrics/provenance JSON.
5. Add an opt-in transformer capture mode in the script, using a wrapper around `pipeline.transformer.forward`, to save cached tuples and BF16 target outputs.
6. Add unit tests for manifest validation, missing artifact failures, tensor shape checks, and cache tuple schema using tiny tensors or mocks, without requiring full Qwen model weights.

## Open Risks

- The exact PSNR gate statistic is still defaulting to per-required-sample minimum until the user says otherwise.
- The representative dataset size and train/validation split are still pending.
- Full generation and K-sweeps require model weights and GPUs; this round only identified the code paths.
