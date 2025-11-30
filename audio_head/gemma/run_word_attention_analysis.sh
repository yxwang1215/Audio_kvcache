#!/bin/bash

# Word-Level Attention Analysis Script
# 分析在生成高质量单词的 token 时，注意力是否集中在对应的 prefill 音频 token 位置

cd /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/audio_head

CUDA_VISIBLE_DEVICES=5 python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/audio_head/word_attention_analysis.py \
    --model_path /mnt/users/wangyuxuan-20250915/Models/Qwen2.5-Omni-3B \
    --audio_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/output_audio \
    --whisperx_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/whisperx_parquet_output \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/audio_head/word_attention_results \
    --num_samples 10 \
    --score_threshold 0.95 \
    --device cuda \
    --gpu_id 7

echo "Analysis complete! Check word_attention_results/ for results."

