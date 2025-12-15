#!/bin/bash
# KeSpeech ASR Evaluation with Gemma-3n-E2B-it
# 运行脚本示例
export CUDA_VISIBLE_DEVICES=1
export MODEL_PATH="/mnt/public/models/gemma-3n-E4B-it"
export SAMPLE_LIMIT=0

# KV Cache compression method: sparsemm, snapkv, adakv, pyramidkv, mixkv (disabled when USE_SPARSEKV=false)
export KV_METHOD="sparsekv"
export USE_SPARSEKV="false"
export BUDGET=1.0
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export HEAD_SCORE_FILE="uniform"

# FFT 滤波配置（可选）
# export USE_FFT_FILTER="true"
# export FFT_FILTER_TYPE="lowpass"
# export FFT_CUTOFF_RATIO="0.3"
# export FFT_ALPHA="0.5"
# export FFT_ADAPTIVE_CUTOFF="true"
# export FFT_USE_SALIENCY="false"

export NO_REPEAT_NGRAM_SIZE=3
export REPETITION_PENALTY=1.1

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh

conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft_dataset && \
python code/gemma_kespeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E4B_it/datasets/KeSpeech_Results/vanilla/full_cache \
    --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E4B_it/datasets/KeSpeech_Results/vanilla/full_cache/timing_stats.json

