#!/bin/bash
# Multilingual LibriSpeech ASR Evaluation with Qwen2.5-Omni - German
# 德语 ASR 评估脚本（已优化）
# 
# 德语优化措施：
# 1. 增加 max_new_tokens 到 1500（德语单词更长，需要更多tokens）
# 2. 语言特定的文本标准化（保留德语特殊字符）
# 3. 可选的 FFT 滤波和 Audio-aware TopK 优化
export CUDA_VISIBLE_DEVICES=3
# 模型路径配置
export MODEL_PATH="/mnt/public/models/Qwen3-Omni-30B-A3B-Instruct"
export SAMPLE_LIMIT=0
export USE_SPARSEKV="false"
# export BUDGET=0.7
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export HEAD_SCORE_FILE="aggregated"
# export HEAD_SCORE_BATCH=None  

# German-specific optimization: increase max_new_tokens (German words are longer)
# Default: 1500 (vs 1100 for other languages)
export MAX_NEW_TOKENS_GERMAN=1500

# Optional: Enable FFT filter for better audio quality
export USE_FFT_FILTER="false"
export FFT_FILTER_TYPE="lowpass"
export FFT_CUTOFF_RATIO="0.3"
export FFT_ALPHA="0.5"
export FFT_ADAPTIVE_CUTOFF="true"
export FFT_USE_SALIENCY="false"

# Optional: Enable Audio-aware TopK for better attention selection
# export USE_AUDIO_AWARE_TOPK="true"
# export AUDIO_BOUNDARY_BOOST="1.3"
# export AUDIO_CONTINUITY_WEIGHT="0.1"
# export AUDIO_SMOOTH_WINDOW="5"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh

conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/vanilla_dataset && \
python code/qwen3_multilingual_librispeech.py \
    --model-path "${MODEL_PATH}" \
    --languages german \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen3-Omni-30B-A3B-Instruct/datasets/MultilingualLibriSpeech/German/vanilla/full_cache \
    --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen3-Omni-30B-A3B-Instruct/datasets/MultilingualLibriSpeech/German/vanilla/full_cache/timing_stats.json

# ============================================
# 常用配置示例
# ============================================

# 示例1: 基础运行（使用默认配置）
# conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=0 python code/qwen2.5_multilingual_librispeech.py --languages german

# 示例2: 限制样本数量（测试用）
# conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=0 SAMPLE_LIMIT=10 python code/qwen2.5_multilingual_librispeech.py --languages german

# 示例3: 使用比例预算模式
# conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=0 BUDGET=0.2 python code/qwen2.5_multilingual_librispeech.py --languages german

# 示例4: 启用 FFT 滤波
# conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=0 USE_FFT_FILTER="true" python code/qwen2.5_multilingual_librispeech.py --languages german

# 示例5: 启用 Audio-aware TopK
# conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=0 USE_AUDIO_AWARE_TOPK="true" python code/qwen2.5_multilingual_librispeech.py --languages german

# 示例6: 完整配置示例
# conda activate Qwen3_2 && \
# CUDA_VISIBLE_DEVICES=0 \
# BUDGET=2048 \
# RATIO=0.2 \
# WINDOW_SIZE=64 \
# SAMPLE_LIMIT=100 \
# python code/qwen2.5_multilingual_librispeech.py \
#     --languages german \
#     --output-dir ./MultilingualLibriSpeech_Results/German \
#     --timing-output-file ./MultilingualLibriSpeech_Results/German/timing_stats.json

