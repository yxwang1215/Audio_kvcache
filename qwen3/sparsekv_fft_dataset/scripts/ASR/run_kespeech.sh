#!/bin/bash
# KeSpeech ASR Evaluation with Qwen2.5-Omni

export CUDA_VISIBLE_DEVICES=7
export MODEL_PATH="/mnt/public/models/Qwen3-Omni-30B-A3B-Instruct"
export SAMPLE_LIMIT=0

export USE_SPARSEKV="true"
export BUDGET=0.8
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export HEAD_SCORE_PATH="/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen3_omni/head_score/audio_head_scores_topk16_alpha1_beta1_gamma1.json"
export HEAD_SCORE_AGGREGATION="first"

# FFT 滤波配置（可选）
export USE_FFT_FILTER="true"
export FFT_FILTER_TYPE="lowpass"
export FFT_CUTOFF_RATIO="0.3"
export FFT_ALPHA="0.5"
export FFT_ADAPTIVE_CUTOFF="false"
export FFT_USE_SALIENCY="false"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/sparsekv_fft_dataset && \
python code/qwen3_kespeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen3-Omni-30B-A3B-Instruct/datasets/KeSpeech_Results/SparseKV_fft/budget0.8_topk16_alpha1_beta1_gamma1_first \
    --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen3-Omni-30B-A3B-Instruct/datasets/KeSpeech_Results/SparseKV_fft/budget0.8_topk16_alpha1_beta1_gamma1_first/timing_stats.json