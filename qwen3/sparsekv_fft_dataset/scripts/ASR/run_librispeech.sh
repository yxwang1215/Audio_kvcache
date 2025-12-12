#!/bin/bash
# LibriSpeech ASR Evaluation with Qwen3-Omni

export CUDA_VISIBLE_DEVICES=5
export MODEL_PATH="/mnt/public/models/Qwen3-Omni-30B-A3B-Instruct"
export SAMPLE_LIMIT=0

export USE_SPARSEKV="true"
export BUDGET=0.8
# export BUDGET=0.2  # 使用比例模式
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export HEAD_SCORE_PATH="/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen3_omni/head_score/audio_head_scores_topk16_alpha1_beta1_gamma1.json"
export HEAD_SCORE_AGGREGATION="first"

# FFT 滤波配置（可选）
# export USE_FFT_FILTER="true"
# export FFT_FILTER_TYPE="lowpass"
# export FFT_CUTOFF_RATIO="0.3"
# export FFT_ALPHA="0.5"
# export FFT_ADAPTIVE_CUTOFF="true"
# export FFT_USE_SALIENCY="false"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh

conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/sparsekv_fft_dataset && \
python code/qwen3_librispeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen3-Omni-30B-A3B-Instruct/datasets/LibriSpeech_Results/SparseKV/budget0.8_win32_ratio0.5_kernel5_topk16_alpha1_beta1_gamma1_first \
    --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen3-Omni-30B-A3B-Instruct/datasets/LibriSpeech_Results/SparseKV/budget0.8_win32_ratio0.5_kernel5_topk16_alpha1_beta1_gamma1_first/timing_stats.json

# 或者指定具体的输出文件
# python code/qwen2.5_librispeech.py \
#     --output-file ./LibriSpeech_Results/librispeech_results_qwen25.json \
#     --timing-output-file ./LibriSpeech_Results/timing_stats.json

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
