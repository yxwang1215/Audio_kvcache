#!/bin/bash
# KeSpeech ASR Evaluation with Qwen2.5-Omni
# 运行脚本示例

# ============================================
# 基础配置
# ============================================
# GPU 设置
export CUDA_VISIBLE_DEVICES=2

# 模型路径配置
export MODEL_PATH="/mnt/public/models/Qwen2.5-Omni-7B"

# 数据集配置（已有默认值，可选）
# 默认路径: /mnt/users/wangyuxuan-20250915/Audio/UltraEval-Audio/raw/TwinkStart/kespeech

# 样本限制（0表示处理全部样本）
export SAMPLE_LIMIT=0

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

# ============================================
# 剪枝配置（如果使用）
# ============================================
# export PRUNE_RATIO=0.5
# export PRUNE_METHOD="fast_v"  # 可选: "fast_v", "random", "frame"

# ============================================
# 输出配置
# ============================================
# 结果目录（可选，默认 ./KeSpeech_Results）
# export RESULTS_DIR="./KeSpeech_Results"

# ============================================
# 运行命令
# ============================================
# 初始化 conda（在 bash 脚本中必须初始化才能使用 conda activate）
source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh

# 激活环境并运行
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_dataset && \
python code/qwen2.5_kespeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_7B/datasets/KeSpeech_Results \
    --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_7B/datasets/KeSpeech_Results/timing_stats.json

# 或者指定具体的输出文件
# python code/qwen2.5_kespeech.py \
#     --output-file ./KeSpeech_Results/kespeech_results_qwen25.json \
#     --timing-output-file ./KeSpeech_Results/timing_stats.json

# ============================================
# 常用配置示例
# ============================================

# 示例1: 基础运行（使用默认配置）
# conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=0 python code/qwen2.5_kespeech.py

# 示例2: 限制样本数量（测试用）
# conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=0 SAMPLE_LIMIT=10 python code/qwen2.5_kespeech.py

# 示例3: 使用比例预算模式
# conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=0 BUDGET=0.2 python code/qwen2.5_kespeech.py

# 示例4: 启用 FFT 滤波
# conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=0 USE_FFT_FILTER="true" python code/qwen2.5_kespeech.py

# 示例5: 启用 Audio-aware TopK
# conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=0 USE_AUDIO_AWARE_TOPK="true" python code/qwen2.5_kespeech.py

# 示例6: 完整配置示例
# conda activate Qwen3_2 && \
# CUDA_VISIBLE_DEVICES=0 \
# BUDGET=2048 \
# RATIO=0.2 \
# WINDOW_SIZE=64 \
# SAMPLE_LIMIT=100 \
# python code/qwen2.5_kespeech.py \
#     --output-dir ./KeSpeech_Results \
#     --timing-output-file ./KeSpeech_Results/timing_stats.json
