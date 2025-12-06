#!/bin/bash
# CoVoST2 Speech Translation Evaluation with Gemma-3n-E2B-it
# Chinese to English Translation
# 运行脚本示例

# ============================================
# 基础配置
# ============================================
# GPU 设置
export CUDA_VISIBLE_DEVICES=6

# 模型路径配置
export MODEL_PATH="/mnt/public/models/gemma-3n-E2B-it"

# 数据集配置（已有默认值，可选）
# 默认路径: /mnt/users/wangyuxuan-20250915/Audio/UltraEval-Audio/raw/TwinkStart/covost2_zh_en_test

# 样本限制（0表示处理全部样本）
export SAMPLE_LIMIT=0

# 翻译方向
export COVOST2_DIRECTION="zh_en"

# ============================================
# SparseKV 配置
# ============================================
# 是否使用 SparseKV（默认 true）
# export USE_SPARSEKV="true"
export USE_SPARSEKV="false"

# 预算配置（可以是绝对数值或比例，如 0.2 表示 20%）
# export BUDGET=0.7
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

# 重复检测配置（防止模型陷入重复循环）
export NO_REPEAT_NGRAM_SIZE=3
export REPETITION_PENALTY=1.1

# ============================================
# 运行命令
# ============================================
# 初始化 conda（在 bash 脚本中必须初始化才能使用 conda activate）
source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh

# 激活环境并运行
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft_dataset && \
python code/gemma_covost2.py \
    --direction "${COVOST2_DIRECTION}" \
    --model-path "${MODEL_PATH}" \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E2B_it/datasets/CoVoST2_Results/ZH_EN \
    --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E2B_it/datasets/CoVoST2_Results/ZH_EN/timing_stats.json

