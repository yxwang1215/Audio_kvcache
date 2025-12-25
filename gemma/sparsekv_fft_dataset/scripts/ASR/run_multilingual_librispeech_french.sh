#!/bin/bash
# Multilingual LibriSpeech ASR Evaluation with Gemma-3n-E2B-it - French

export CUDA_VISIBLE_DEVICES=5
export MODEL_PATH="/mnt/public/models/gemma-3n-E4B-it"
export SAMPLE_LIMIT=0

export KV_METHOD="tova"
# export KV_METHOD="mixkv"
# export SELECT_METHOD="headwisemixkv"
# export METHOD="mixsparsemm"
# export USE_SPARSEKV="true"
export BUDGET=0.4
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
# export HEAD_SCORE_FILE="uniform"
# export HEAD_SCORE_PATH="/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/gemma/head_score/Gemma_3n_E2B/audio_head_scores_topk24_alpha1_beta1_gamma1.json"
export HEAD_SCORE_AGGREGATION="first"

export USE_FFT_FILTER="false"
export FFT_FILTER_TYPE="lowpass"
export FFT_CUTOFF_RATIO="0.7"
export FFT_ALPHA="0.5"
export FFT_ADAPTIVE_CUTOFF="false"
export FFT_USE_SALIENCY="false"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft_dataset && \
python code/gemma_multilingual_librispeech.py \
    --model-path "${MODEL_PATH}" \
    --languages french \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E4B_it/datasets/MultilingualLibriSpeech/French/ToVA/budget0.4 \
    --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E4B_it/datasets/MultilingualLibriSpeech/French/ToVA/budget0.4/timing_stats.json


