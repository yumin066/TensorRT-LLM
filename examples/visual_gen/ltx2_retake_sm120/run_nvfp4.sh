CUST=/home/scratch.minyu_gpu/project/shopee/LTX2.3-script-editing
REPO=/home/scratch.minyu_gpu/project/shopee/TensorRT-LLM
OUT=/home/scratch.minyu_gpu/project/shopee/host_push/manual_m034
mkdir -p $OUT

# OpenImageIO 只是 import-time stub；torchaudio 必须是真的，别让 stub 盖住
STUB=/tmp/native_import_stubs
mkdir -p $STUB && rm -f $STUB/torchaudio.py
printf 'def __getattr__(n): raise NotImplementedError(n)\n' > $STUB/OpenImageIO.py
export PYTHONPATH=$STUB:$CUST/packages/ltx-pipelines/src:$CUST/packages/ltx-core/src

# 中性 CWD：import tensorrt_llm 必须命中已安装的 overlay 包，不能命中未编译的源码树
cd /tmp

python3 $REPO/examples/visual_gen/ltx2_retake_e2e.py \
  --checkpoint /home/scratch.ylichen_sw/LTX2.3-script-editing/models/ltx-2.3-22b-distilled.safetensors \
  --text-encoder /home/scratch.ylichen_sw/LTX2.3-script-editing/models/gemma \
  --source $CUST/script_editing/eval/reference_outputs/720p/retake_output.mp4 \
  --output $OUT/retake_nvfp4.mp4 \
  --start 3.0 --end 3.9667 \
  --quant-algo NVFP4 --nvfp4-attn \
  --fp8-linear-step 4 --fp8-linear-step 7 \
  2>&1 | tee $OUT/m034.log
