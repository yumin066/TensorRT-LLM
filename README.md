<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# VisualGen quantization-ignore integration images

Image artifacts for NVIDIA/TensorRT-LLM PR #22120, regenerated after the checkpoint-alias loading fix.

- Model: black-forest-labs/FLUX.2-klein-4B, revision e7b7dc27f91deacad38e78976d1f2b499d76a294.
- Prompt: A small red sailboat on a calm lake, mountains in the distance, morning light.
- Parameters: 512 x 512, four complete diffusion steps, seed 42, guidance_scale 1.0.
- Environment: SC B200, nvcr.io/nvidia/tensorrt-llm/devel:1.4.0rc3.
- Block and step indices start at zero.
- PNG files are byte-identical to the original generation artifacts. Two independent complete seven-case runs produced identical image hashes and LPIPS values.
- LPIPS is measured relative to this run's BF16 Linear + VANILLA BF16 attention reference. Limits apply only to this fixed B200 fixture.
- Initial-noise SHA256 (all cases): 61747b8d76d6b16fdd1febf0e53cb081f9591657c062462a98ec94c2ff82d6cf.

| Case | LPIPS vs BF16 | Limit | Original PNG SHA256 |
|---|---:|---:|---|
| [bf16_reference](bf16_reference.png) | Reference | - | `0d163d669e4b6ca1600d8ca0f3e19126a91067d798d3726f3cdf4db854ca22e5` |
| [linear_fp8_block_layer](linear_fp8_block_layer.png) | 0.0014603407 | 0.01 | `25df72dcf7a578969f278feb2b8f57db593e62aa09e45457d54b6ecbed462749` |
| [linear_nvfp4_step](linear_nvfp4_step.png) | 0.0109277479 | 0.03 | `6d65c556ced2ce56b4f727844144479918797a62ae3e914581437d0c4b4933b7` |
| [attention_flashinfer_step](attention_flashinfer_step.png) | 0.0093087973 | 0.03 | `0279354793cae3dc212a8c6c082f28252e376b7eb1e55bccd06012d7bb88d0d1` |
| [attention_trtllm_sage_layer](attention_trtllm_sage_layer.png) | 0.0003980861 | 0.005 | `5a98ed346c4416180f0b0b934d03a04d4ef2b07828533bdc609ae42e84b55647` |
| [attention_cutedsl_step](attention_cutedsl_step.png) | 0.0058070812 | 0.02 | `75105e01f50bb6b31ff77be9a26c8f0fc932bd394a5a4fd8022e89a22e930791` |
| [attention_cudnn_layer](attention_cudnn_layer.png) | 0.0003195395 | 0.005 | `70728fe8ab4aa9695f1b4e0102664d79cfdebd9e5155e623a8249e31f1a0fbc0` |
