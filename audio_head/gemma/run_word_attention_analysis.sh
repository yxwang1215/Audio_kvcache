#!/bin/bash

# Word-Level Attention Analysis Script for Gemma LibriSpeech
# 分析在生成高质量单词的 token 时，注意力是否集中在对应的 prefill 音频 token 位置

cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/gemma

CUDA_VISIBLE_DEVICES=5 python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/gemma/word_attention_analysis.py \
    --model_path /mnt/petrelfs/wenzichen/hf_models/google/gemma-3n-E2B-it \
    --audio_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/output_audio \
    --whisperx_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/whisperx_parquet_output \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/gemma/word_attention_results \
    --num_samples 10 \
    --score_threshold 0.95 \
    --device cuda \
    --topk 32

echo "Analysis complete! Check word_attention_results/ for results."

