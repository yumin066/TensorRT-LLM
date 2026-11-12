<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# VisualGen quantization-ignore integration images

Image artifacts for NVIDIA/TensorRT-LLM PR #22120.

- Model: black-forest-labs/FLUX.2-klein-4B, revision e7b7dc27f91deacad38e78976d1f2b499d76a294.
- Prompt: A small red sailboat on a calm lake, mountains in the distance, morning light.
- Parameters: 512 x 512, four complete diffusion steps, seed 42, guidance_scale 1.0.
- Environment: SC B200, nvcr.io/nvidia/tensorrt-llm/devel:1.4.0rc3.
- Block and step indices start at zero.
- PNG files are byte-identical to the original generation artifacts.
- LPIPS is measured relative to BF16 Linear + VANILLA BF16 attention; no quality threshold was calibrated.
- Initial-noise SHA256 (all cases): 61747b8d76d6b16fdd1febf0e53cb081f9591657c062462a98ec94c2ff82d6cf.

| Case | LPIPS vs BF16 | Original PNG SHA256 |
|---|---:|---|
| [bf16_reference](bf16_reference.png) | Reference | `5fbe54093721afa2f625204669c2280fa25a791cce9eaba613736688e209e3b3` |
| [linear_fp8_block_layer](linear_fp8_block_layer.png) | 0.1414295733 | `c3f942c7b809a4c2c89ed6ca81bb82c993693d6c2437dec1efae03d54c6b7609` |
| [linear_nvfp4_step](linear_nvfp4_step.png) | 0.0100274542 | `c257a8ae834a0f740efad43264790104d3f165e2f0f7d6b6b3e6f02cf0b3d36e` |
| [attention_flashinfer_step](attention_flashinfer_step.png) | 0.0130511532 | `a7a03a665f4a429f5610ce3623e87ecc121f2c59ed172e719fe2550eb6b12fff` |
| [attention_trtllm_sage_layer](attention_trtllm_sage_layer.png) | 0.0003775273 | `87d7347b002c773352595f0581f31d16be88d2070979835c917ac4da540b5a58` |
| [attention_cutedsl_step](attention_cutedsl_step.png) | 0.0054410491 | `937d04916ccd354e248fb4bf730fb1dba19b60ecd16ac77a6de8d5c3b05b6741` |
| [attention_cudnn_layer](attention_cudnn_layer.png) | 0.0004633406 | `7330c391f30d96dc13367c77b624a25ffeaa80a7613185d66570a1a8a91db7c7` |
