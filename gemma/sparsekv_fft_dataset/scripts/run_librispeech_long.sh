#!/bin/bash
# LibriSpeech-long ASR Evaluation with Gemma-3n-E2B-it
# 运行脚本示例

# ============================================
# 基础配置
# ============================================
# GPU 设置
export CUDA_VISIBLE_DEVICES=6

# 模型路径配置
export MODEL_PATH="/mnt/public/models/gemma-3n-E2B-it"

# 数据集配置（已有默认值，可选）
# 默认路径: /mnt/users/wangyuxuan-20250915/Datasets/AudioMarathon/librispeech-long

# 样本限制（0表示处理全部样本）
export SAMPLE_LIMIT=0

# ============================================
# SparseKV 配置
# ============================================
# 是否使用 SparseKV（默认 true）
# export USE_SPARSEKV="true"
export USE_SPARSEKV="false"

# 预算配置（可以是绝对数值或比例，如 0.2 表示 20%）
# export BUDGET=1.0
# export BUDGET=0.2  # 使用比例模式

# SparseMM 参数
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"

# Head Score 配置
export HEAD_SCORE_FILE="aggregated"
# export HEAD_SCORE_BATCH=None  # 可选

# FFT 滤波配置（可选）
# export USE_FFT_FILTER="true"
# export FFT_FILTER_TYPE="lowpass"
# export FFT_CUTOFF_RATIO="0.3"
# export FFT_ALPHA="0.5"
# export FFT_ADAPTIVE_CUTOFF="true"
# export FFT_USE_SALIENCY="false"

# Audio-aware TopK 配置（可选）
# export USE_AUDIO_AWARE_TOPK="true"
# export AUDIO_BOUNDARY_BOOST=1.3
# export AUDIO_CONTINUITY_WEIGHT=0.1
# export AUDIO_SMOOTH_WINDOW=5

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh

# 激活环境并运行
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft_dataset && \
python code/gemma_librispeech_long.py \
    --model-path "${MODEL_PATH}" \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E2B_it/datasets/LibriSpeech_Results \
    --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E2B_it/datasets/LibriSpeech_Results/timing_stats.json

