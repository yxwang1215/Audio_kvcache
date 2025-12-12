#!/bin/bash

# Word-Level Attention Analysis Script
# 分析在生成高质量单词的 token 时，注意力是否集中在对应的 prefill 音频 token 位置

source $(conda info --base)/etc/profile.d/conda.sh
conda activate Qwen3_2

cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen3_omni

CUDA_VISIBLE_DEVICES=4 python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen3_omni/word_attention_analysis.py \
    --model_path /mnt/public/models/Qwen3-Omni-30B-A3B-Instruct \
    --audio_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/output_audio \
    --whisperx_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/whisperx_parquet_output \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen3_omni/results/topk32 \
    --skip_detailed_analysis \
    --topk 32 \
    --num_samples 0 \
    --score_threshold 0.95 \