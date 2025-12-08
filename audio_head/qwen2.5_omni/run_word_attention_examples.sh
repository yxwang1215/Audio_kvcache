#!/bin/bash

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate Qwen3_2

# 切换到脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=5 python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/word_attention_analysis.py \
    --num_samples 0 \
    --topk 16 \
    --skip_detailed_analysis \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/results/topk16/ 

conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=6 python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/word_attention_analysis.py \
    --num_samples 0 \
    --topk 24 \
    --skip_detailed_analysis \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/results/topk24/

conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=7 python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/word_attention_analysis.py \
    --num_samples 0 \
    --topk 48 \
    --skip_detailed_analysis \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/results/topk48/
