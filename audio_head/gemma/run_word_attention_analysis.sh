#!/bin/bash

# Word-Level Attention Analysis Script for Gemma LibriSpeech
# 分析在生成高质量单词的 token 时，注意力是否集中在对应的 prefill 音频 token 位置
# 支持 Gemma E2B (30 layers) 和 E4B (35 layers) 两个模型

cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/gemma

# ========== Gemma E2B 模型示例 ==========
echo "Running analysis with Gemma E2B model..."
CUDA_VISIBLE_DEVICES=5 python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/gemma/word_attention_analysis.py \
    --model_type E2B \
    --model_path /mnt/petrelfs/wenzichen/hf_models/google/gemma-3n-E2B-it \
    --audio_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/output_audio \
    --whisperx_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/whisperx_parquet_output \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/gemma/word_attention_results_E2B \
    --num_samples 10 \
    --score_threshold 0.95 \
    --topk 32 \
    --alpha 1.0 --beta 1.0 --gamma 1.0

echo "E2B analysis complete! Check word_attention_results_E2B/ for results."

# ========== Gemma E4B 模型示例 ==========
# 取消下面的注释以运行 E4B 模型分析
# echo "Running analysis with Gemma E4B model..."
# CUDA_VISIBLE_DEVICES=5 python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/gemma/word_attention_analysis.py \
#     --model_type E4B \
#     --model_path /mnt/petrelfs/wenzichen/hf_models/google/gemma-3n-E4B-it \
#     --audio_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/output_audio \
#     --whisperx_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/whisperx_parquet_output \
#     --output_dir /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/gemma/word_attention_results_E4B \
#     --num_samples 10 \
#     --score_threshold 0.95 \
#     --topk 32 \
#     --alpha 1.0 --beta 1.0 --gamma 1.0
# 
# echo "E4B analysis complete! Check word_attention_results_E4B/ for results."

# ========== 其他使用示例 ==========
# 
# 1. 使用自动检测模型类型（从路径或配置自动检测）
# CUDA_VISIBLE_DEVICES=5 python word_attention_analysis.py \
#     --model_type auto \
#     --model_path /mnt/petrelfs/wenzichen/hf_models/google/gemma-3n-E2B-it \
#     --num_samples 10
# 
# 2. 处理所有样本（E2B）
# CUDA_VISIBLE_DEVICES=7 python word_attention_analysis.py \
#     --model_type E2B \
#     --num_samples 0 \
#     --alpha 1.0 --beta 1.0 --gamma 1.0
# 
# 3. 处理所有样本（E4B）
# CUDA_VISIBLE_DEVICES=7 python word_attention_analysis.py \
#     --model_type E4B \
#     --num_samples 0 \
#     --alpha 1.0 --beta 1.0 --gamma 1.0
# 
# 4. 处理指定范围的样本（E2B）
# CUDA_VISIBLE_DEVICES=5 python word_attention_analysis.py \
#     --model_type E2B \
#     --start_idx 100 --end_idx 200 \
#     --topk 32
# 
# 5. 处理指定范围的样本（E4B）
# CUDA_VISIBLE_DEVICES=5 python word_attention_analysis.py \
#     --model_type E4B \
#     --start_idx 100 --end_idx 200 \
#     --topk 32

