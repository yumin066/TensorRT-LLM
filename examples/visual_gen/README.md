# Visual Generation Examples

See [the VisualGen doc](https://nvidia.github.io/TensorRT-LLM/models/visual-generation.html)
for feature details.

## Layout

| Path | Purpose |
|---|---|
| [`quickstart_example.py`](quickstart_example.py) | Minimal VisualGen API example |
| [`models/`](models/) | Per-model example scripts |
| [`configs/`](configs/) | Shared `VisualGenArgs` YAMLs (used by `--visual_gen_args` and `trtllm-serve`) |
| [`serve/`](serve/) | `trtllm-serve` usage, benchmarking, and clients |

## Usage

```bash
# Defaults
python quickstart_example.py
python models/wan_t2v.py
python models/ltx2.py
python models/flux1.py
python models/flux2.py
python models/cosmos3_ti2v.py --prompt "A robot arm picks fruit in a grocery store"
python models/qwen_image.py
python models/qwen_image_layered.py --image /path/to/image.png
python models/qwen_image_edit.py --image /path/to/source.png --prompt "Make the image look like a watercolor painting"
python models/glm_image.py
python models/hunyuan_t2v.py

# With engine config (quant, parallelism, etc.)
python models/wan_t2v.py --visual_gen_args configs/wan2.2-t2v-fp4-1gpu.yaml
python models/wan_i2v.py --visual_gen_args configs/wan2.2-i2v-fp4-1gpu.yaml --image /path/to/image.png
python models/ltx2.py --visual_gen_args configs/ltx2-1gpu.yaml
python models/flux1.py --visual_gen_args configs/flux1-dev-fp4-1gpu.yaml
python models/flux2.py --visual_gen_args configs/flux2-dev-fp4-1gpu.yaml
python models/cosmos3_ti2v.py --visual_gen_args configs/cosmos3-nano-1gpu.yaml --prompt "A robot arm picks fruit in a grocery store"
python models/qwen_image.py --visual_gen_args configs/qwen-image-fp8-1gpu.yaml
python models/qwen_image_layered.py --visual_gen_args configs/qwen-image-layered-1gpu.yaml --image /path/to/image.png
python models/qwen_image_edit.py --visual_gen_args configs/qwen-image-edit-2511-fp4-1gpu.yaml --image /path/to/source.png --prompt "Make the image look like a watercolor painting"
python models/hunyuan_t2v.py --visual_gen_args configs/hunyuan-t2v-fp8-1gpu.yaml
```

LTX-2.3 retake takes an edited source video and regenerates the requested time
window while conditioning on the surrounding frames:

```bash
python models/ltx2_retake.py \
  --model /path/to/ltx-2.3-22b-distilled.safetensors \
  --visual_gen_args configs/ltx2-retake-1gpu.yaml \
  --source /path/to/retake_input.mp4 \
  --text_encoder_path /path/to/gemma-3-12b-it \
  --prompt "a person talking to the camera"
```

The selected YAML configures the start/end time, seed, prompt-conditioning
path, LoRA path, and LoRA strength. To use precomputed text conditioning, set
`retake_prompt_conditioning_path` in `pipeline_config` and omit
`--text_encoder_path` and `--prompt`. The FP8 and NVFP4 recipes are in
`configs/ltx2-retake-*-1gpu.yaml`.

Retake source-media decoding and audio conditioning need two optional packages:

```bash
pip install av
pip install --no-deps --index-url https://download.pytorch.org/whl/cpu \
  torchaudio==2.11.0+cpu
```

Install deps from the repo root: `pip install -r requirements-dev.txt`.

Output: `.png` for image models; `.mp4` for video models when FFmpeg is installed (otherwise `.avi`).
