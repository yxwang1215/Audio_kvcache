#!/bin/bash
# CoVoST2 Speech Translation Evaluation with Gemma-3n-E2B-it
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/mnt/public/models/gemma-3n-E4B-it"
export SAMPLE_LIMIT=4898
export COVOST2_DIRECTION="en_zh"

# KV Cache compression method: sparsemm, snapkv, adakv, pyramidkv, mixkv
export KV_METHOD="sparsekv"
export USE_SPARSEKV="true"
export BUDGET=0.8  
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
# HEAD_SCORE_FILE: for sparsemm method, options: "uniform", "aggregated", or path to head score file
export HEAD_SCORE_FILE="uniform"
# HEAD_SCORE_PATH: path to head score JSON file (for sparsemm method)
# export HEAD_SCORE_PATH="/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/gemma/head_score/audio_head_scores_aggregated.json"
export HEAD_SCORE_AGGREGATION="first"

# FFT 滤波配置（可选）
# export USE_FFT_FILTER="true"
# export FFT_FILTER_TYPE="lowpass"
# export FFT_CUTOFF_RATIO="0.3"
# export FFT_ALPHA="0.5"
# export FFT_ADAPTIVE_CUTOFF="true"
# export FFT_USE_SALIENCY="false"

# 重复检测配置（防止模型陷入重复循环）
export NO_REPEAT_NGRAM_SIZE=3
export REPETITION_PENALTY=1.1

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft_dataset && \
python code/gemma_covost2.py \
    --direction "${COVOST2_DIRECTION}" \
    --model-path "${MODEL_PATH}" \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E4B_it/datasets/CoVoST2_Results/EN_ZH/SnapKV/budget0.8 \
    --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E4B_it/datasets/CoVoST2_Results/EN_ZH/SnapKV/budget0.8/timing_stats.json

