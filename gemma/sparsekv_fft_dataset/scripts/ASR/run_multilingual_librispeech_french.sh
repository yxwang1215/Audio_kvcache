#!/bin/bash
# Multilingual LibriSpeech ASR Evaluation with Gemma-3n-E2B-it - French

export CUDA_VISIBLE_DEVICES=7
export MODEL_PATH="/mnt/public/models/gemma-3n-E4B-it"
export SAMPLE_LIMIT=0

# KV Cache compression method: sparsemm, snapkv, adakv, pyramidkv, mixkv
export KV_METHOD="sparsekv"
export USE_SPARSEKV="true"
export BUDGET=0.8
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export HEAD_SCORE_FILE="uniform"

export USE_FFT_FILTER="false"
export FFT_FILTER_TYPE="lowpass"
export FFT_CUTOFF_RATIO="0.3"
export FFT_ALPHA="0.5"
export FFT_ADAPTIVE_CUTOFF="true"
export FFT_USE_SALIENCY="false"

# 重复检测配置（防止模型陷入重复循环）
export NO_REPEAT_NGRAM_SIZE=3
export REPETITION_PENALTY=1.1

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh

# 激活环境并运行（只处理法语）
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft_dataset && \
python code/gemma_multilingual_librispeech.py \
    --model-path "${MODEL_PATH}" \
    --languages french \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E4B_it/datasets/MultilingualLibriSpeech/French/SnapKV/budget0.8 \
    --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E4B_it/datasets/MultilingualLibriSpeech/French/SnapKV/budget0.8/timing_stats.json

