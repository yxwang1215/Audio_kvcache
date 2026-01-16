#!/bin/bash
# LibriSpeech ASR Evaluation with Qwen2.5-Omni

export CUDA_VISIBLE_DEVICES=7
export MODEL_PATH="/mnt/public/models/Qwen2.5-Omni-3B"
export SAMPLE_LIMIT=0

export KV_METHOD="snapkv"
# export KV_METHOD="pyramidkv"
# export USE_OPTIMIZED_SPARSEMM=true
export BUDGET=0.99
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export HEAD_SCORE_FILE="uniform"
# export HEAD_SCORE_PATH="/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/head_score/qwen2_5_3B_old/audio_head_scores_topk24_alpha1_beta1_gamma1.json"
export HEAD_SCORE_AGGREGATION="first"

export USE_FFT_FILTER="false"
export FFT_FILTER_TYPE="lowpass"
export FFT_CUTOFF_RATIO="0.7"
export FFT_ALPHA="0.5"
export FFT_ADAPTIVE_CUTOFF="false"
export FFT_USE_SALIENCY="false"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_dataset && \
python code/qwen2.5_librispeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_3B/datasets/librispeech_Results/KNorm/budget1.0 \
    --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_3B/datasets/librispeech_Results/KNorm/budget1.0/timing_stats.json

