#!/bin/bash
# LibriSpeech-long ASR Evaluation with Gemma-3n-E2B-it

export CUDA_VISIBLE_DEVICES=6
export MODEL_PATH="/mnt/public/models/gemma-3n-E4B-it"
export SAMPLE_LIMIT=0

# KV Cache compression method: sparsemm, snapkv, adakv, pyramidkv, mixkv
export KV_METHOD="snapkv"
# export USE_SPARSEKV="true"
export USE_SPARSEKV="false"
export BUDGET=0.8
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export HEAD_SCORE_FILE="uniform"
# For MixKV: select method can be 'attn', 'keydiff', 'mixkv', 'headwisemixkv', 'knorm', 'vnorm'
# export SELECT_METHOD="mixkv"

# FFT 滤波配置（可选）
# export USE_FFT_FILTER="true"
# export FFT_FILTER_TYPE="lowpass"
# export FFT_CUTOFF_RATIO="0.3"
# export FFT_ALPHA="0.5"
# export FFT_ADAPTIVE_CUTOFF="true"
# export FFT_USE_SALIENCY="false"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh

# 激活环境并运行
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft_dataset && \
python code/gemma_librispeech_long.py \
    --model-path "${MODEL_PATH}" \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E4B_it/datasets/LibriSpeech_long/SnapKV/budget0.8 \
    --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E4B_it/datasets/LibriSpeech_long/SnapKV/budget0.8/timing_stats.json

