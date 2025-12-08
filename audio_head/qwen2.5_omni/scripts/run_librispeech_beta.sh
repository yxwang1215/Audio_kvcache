#!/bin/bash
# LibriSpeech ASR Evaluation with Qwen2.5-Omni=================================

export CUDA_VISIBLE_DEVICES=2
export MODEL_PATH="/mnt/public/models/Qwen2.5-Omni-3B"
export SAMPLE_LIMIT=0
export USE_SPARSEKV="true"
export BUDGET=0.3
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"

# export HEAD_SCORE_PATH="/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/head_score/audio_head_scores_topk32_alpha1_beta1_gamma1.json"
# export HEAD_SCORE_PATH="/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/head_score/audio_head_scores_topk32_alpha0_beta0_gamma1.json"
export HEAD_SCORE_PATH="/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/head_score/audio_head_scores_topk64_alpha0_beta1_gamma0.json"
# export HEAD_SCORE_PATH="/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/head_score/audio_head_scores_topk32_alpha1_beta0_gamma0.json"

# Head Score 聚合方式配置（当 JSON 中有多个分数时如何聚合）
# 可选值: mean (平均值，默认), first (第一个值), second (第二个值), max (最大值), min (最小值), sum (求和)
export HEAD_SCORE_AGGREGATION="first"
# export HEAD_SCORE_AGGREGATION="mean"

# FFT 滤波配置（可选）
export USE_FFT_FILTER="true"
export FFT_FILTER_TYPE="lowpass"
export FFT_CUTOFF_RATIO="0.7"
export FFT_ALPHA="0.5"
export FFT_ADAPTIVE_CUTOFF="true"
export FFT_USE_SALIENCY="false"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
# 激活环境并运行
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_dataset && \
python code/qwen2.5_librispeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_3B/datasets/sparsekv_fft/ASR_Alignment/budget0.3_topk64_alpha0_beta1_gamma0_first \
    --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_3B/datasets/sparsekv_fft/ASR_Alignment/budget0.3_topk64_alpha0_beta1_gamma0_first/timing_stats.json

# ============================================
# 常用配置示例
# ============================================

# 示例1: 基础运行（使用默认配置）
# conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=0 python code/qwen2.5_librispeech.py

# 示例2: 限制样本数量（测试用）
# conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=0 SAMPLE_LIMIT=10 python code/qwen2.5_librispeech.py

# 示例3: 使用比例预算模式
# conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=0 BUDGET=0.2 python code/qwen2.5_librispeech.py

# 示例4: 启用 FFT 滤波
# conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=0 USE_FFT_FILTER="true" python code/qwen2.5_librispeech.py

# 示例5: 直接指定 head score JSON 文件路径（推荐方式）
# conda activate Qwen3_2 && \
# CUDA_VISIBLE_DEVICES=0 \
# HEAD_SCORE_PATH="/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/head_score/audio_head_scores_topk32_alpha1_beta1_gamma1.json" \
# python code/qwen2.5_librispeech.py

# 示例5b: 使用 alpha, beta, gamma 组合（备选方式）
# conda activate Qwen3_2 && \
# CUDA_VISIBLE_DEVICES=0 \
# HEAD_SCORE_FILE="topk32" \
# HEAD_SCORE_ALPHA="1" \
# HEAD_SCORE_BETA="1" \
# HEAD_SCORE_GAMMA="1" \
# python code/qwen2.5_librispeech.py

# 示例6: 完整配置示例
# conda activate Qwen3_2 && \
# CUDA_VISIBLE_DEVICES=0 \
# BUDGET=2048 \
# RATIO=0.2 \
# WINDOW_SIZE=64 \
# SAMPLE_LIMIT=100 \
# python code/qwen2.5_librispeech.py \
#     --output-dir ./LibriSpeech_Results \
#     --timing-output-file ./LibriSpeech_Results/timing_stats.json
