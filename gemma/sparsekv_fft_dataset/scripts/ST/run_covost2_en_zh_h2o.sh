#!/bin/bash
# CoVoST2 Speech Translation Evaluation with Gemma-3n-E2B-it English to Chinese using H2O

export CUDA_VISIBLE_DEVICES=2
export MODEL_PATH="/mnt/public/models/gemma-3n-E2B-it"
export COVOST2_DIRECTION="en_zh"
export SAMPLE_LIMIT=4898

export KV_METHOD="h2o"
export BUDGET=0.8
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="avgpool"
# H2O specific: ratio of heavy hitters vs recent tokens
export H2O_HEAVY_RATIO=0.5

export NO_REPEAT_NGRAM_SIZE=3
export REPETITION_PENALTY=1.1

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft_dataset && \
python code/gemma_covost2.py \
    --direction "${COVOST2_DIRECTION}" \
    --model-path "${MODEL_PATH}" \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E2B_it/datasets/CoVoST2_Results/EN_ZH/H2O/budget0.8 \
    --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E2B_it/datasets/CoVoST2_Results/EN_ZH/H2O/budget0.8/timing_stats.json

