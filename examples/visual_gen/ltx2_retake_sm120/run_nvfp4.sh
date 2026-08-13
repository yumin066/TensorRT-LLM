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
  --source $CUST/artifacts/bf16/retake_input.mp4 \
  --output $OUT/retake_nvfp4.mp4 \
  --dump-frames $OUT/native_nvfp4_frames.pt \
  --start 2.9667 --end 3.9333 \
  --quant-algo NVFP4 --nvfp4-attn \
  --fp8-linear-step 4 --fp8-linear-step 7 \
  2>&1 | tee $OUT/m034.log

# 窗口必须与 bf16 golden 一致：round(2.9667*30)=89、round(3.9333*30)=118 -> 重生成 [89,118)。
# 原来的 3.0/3.9667 会得到 [90,119) —— 同样 29 帧但整体平移一帧，于是帧 89 和 118 会拿
# "重生成帧"去比"源片帧"。质量门槛按逐帧最大值判定，这两帧会直接主导结果，让 NVFP4 因为
# 与量化无关的原因判失败。
#
# --dump-frames 是必需的：质量门槛在 H.264 编码之前的帧张量上打分。Round 12 实测编解码
# 自身的重编码损失就有 0.041-0.049 LPIPS，比要测的差异还大。
#
# 跑完后与 bf16 golden 比对：
#   python3 quality_gate.py \
#     --native-frames $OUT/native_nvfp4_frames.pt \
#     --source $CUST/artifacts/bf16/retake_input.mp4 \
#     --golden-frames $CUST/artifacts/native_bf16_golden/native_bf16_frames.pt \
#     --expect-window 89:118 --window-threshold 0.05
