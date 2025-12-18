#!/bin/bash

# Word-Level Attention Analysis Script for Gemma E4B (35 layers)
# 分析在生成高质量单词的 token 时，注意力是否集中在对应的 prefill 音频 token 位置

cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/gemma

CUDA_VISIBLE_DEVICES=5 python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/gemma/word_attention_analysis.py \
    --model_type E4B \
    --model_path /mnt/public/models/gemma-3n-E4B-it \
    --audio_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/output_audio \
    --whisperx_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/whisperx_parquet_output \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/gemma/results/Gemma_3n_E4B \
    --num_samples 0 \
    --score_threshold 0.95 \
    --topk 16 

echo "E4B analysis complete! Check results/Gemma_3n_E4B/ for results."

