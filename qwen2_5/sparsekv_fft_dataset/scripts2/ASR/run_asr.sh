#!/bin/bash
set -e

# 参数（带默认值）
KV_METHOD="${1:-pyramidkv}"
BUDGET="${2:-0.8}"
MODEL_SIZE="${3:-3B}"
CUDA_ID="${4:-3}"
NAME=${5:-PyramidKV} 

SCRIPT_PATH="/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_dataset/scripts2/ASR/run_multilingual_librispeech_spanish.sh"

bash "$SCRIPT_PATH" \
  "$KV_METHOD" \
  "$BUDGET" \
  "$MODEL_SIZE" \
  "$CUDA_ID" \
  "$NAME"