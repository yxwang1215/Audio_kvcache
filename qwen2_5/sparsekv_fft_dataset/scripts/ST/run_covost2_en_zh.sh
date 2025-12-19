#!/bin/bash
# CoVoST2 English-to-Chinese Speech Translation Evaluation with Qwen2.5-Omni

export CUDA_VISIBLE_DEVICES=3
export MODEL_PATH="/mnt/public/models/Qwen2.5-Omni-3B"
export SAMPLE_LIMIT=4898

export KV_METHOD="adakv"
export BUDGET=0.4
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
# export HEAD_SCORE_FILE="uniform"
export HEAD_SCORE_PATH="/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/head_score/qwen2_5_3B/audio_head_scores_topk24_alpha1_beta1_gamma1.json"

export USE_FFT_FILTER="false"
export FFT_FILTER_TYPE="lowpass"
export FFT_CUTOFF_RATIO="0.7"
export FFT_ALPHA="0.5"
export FFT_ADAPTIVE_CUTOFF="false"
export FFT_USE_SALIENCY="false"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_dataset && \
python code/qwen2.5_covost2.py \
    --direction en_zh \
    --model-path "${MODEL_PATH}" \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_3B/datasets/CoVoST2/EN_ZH/SnapKV/budget0.8_topk24_alpha1_beta1_gamma1_1214 \
    --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_3B/datasets/CoVoST2/EN_ZH/SnapKV/budget0.8_topk24_alpha1_beta1_gamma1_1214/timing_stats.json

    # --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_3B/datasets/CoVoST2/EN_ZH/SparseKV_FFT/lowpass_0.7_alpha0.5/budget0.8_topk24_alpha1_beta1_gamma1_1214 \
    # --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_3B/datasets/CoVoST2/EN_ZH/SparseKV_FFT/lowpass_0.7_alpha0.5/budget0.8_topk24_alpha1_beta1_gamma1_1214/timing_stats.json