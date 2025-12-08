#!/bin/bash
# CoVoST2 Chinese-to-English Speech Translation Evaluation with Qwen2.5-Omni
# 中文到英语语音翻译评估脚本

export CUDA_VISIBLE_DEVICES=2
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

export USE_FFT_FILTER="false"
export FFT_FILTER_TYPE="lowpass"
export FFT_CUTOFF_RATIO="0.3"
export FFT_ALPHA="0.5"
export FFT_ADAPTIVE_CUTOFF="true"
export FFT_USE_SALIENCY="false"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh

# 激活环境并运行（中文到英语）
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/vanilla_dataset && \
python code/qwen3_covost2.py \
    --direction zh_en \
    --model-path "${MODEL_PATH}" \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen3-Omni-30B-A3B-Instruct/datasets/CoVoST2/ZH_EN/vanilla/full_cache \
    --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen3-Omni-30B-A3B-Instruct/datasets/CoVoST2/ZH_EN/vanilla/full_cache/timing_stats.json

# ============================================
# 常用配置示例
# ============================================

# 示例1: 基础运行（使用默认配置）
# conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=0 python code/qwen2.5_covost2.py --direction zh_en

# 示例2: 限制样本数量（测试用）
# conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=0 SAMPLE_LIMIT=10 python code/qwen2.5_covost2.py --direction zh_en

# 示例3: 使用比例预算模式
# conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=0 BUDGET=0.2 python code/qwen2.5_covost2.py --direction zh_en

# 示例4: 启用 FFT 滤波
# conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=0 USE_FFT_FILTER="true" python code/qwen2.5_covost2.py --direction zh_en

# 示例5: 启用 Audio-aware TopK
# conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=0 USE_AUDIO_AWARE_TOPK="true" python code/qwen2.5_covost2.py --direction zh_en

# 示例6: 完整配置示例
# conda activate Qwen3_2 && \
# CUDA_VISIBLE_DEVICES=0 \
# BUDGET=2048 \
# RATIO=0.2 \
# WINDOW_SIZE=64 \
# SAMPLE_LIMIT=100 \
# python code/qwen2.5_covost2.py \
#     --direction zh_en \
#     --output-dir ./CoVoST2_ZH_EN_Results \
#     --timing-output-file ./CoVoST2_ZH_EN_Results/timing_stats.json

