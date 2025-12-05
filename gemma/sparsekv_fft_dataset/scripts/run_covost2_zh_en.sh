#!/bin/bash
# CoVoST2 Chinese-to-English Speech Translation Evaluation with Gemma-3n-E2B-it
# 中文到英语语音翻译评估脚本

export CUDA_VISIBLE_DEVICES=6
# 模型路径配置
export MODEL_PATH="/mnt/public/models/gemma-3n-E2B-it"
export SAMPLE_LIMIT=0

export USE_SPARSEKV="false"
# export BUDGET=1.0
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export HEAD_SCORE_FILE="aggregated"

export USE_FFT_FILTER="false"
export FFT_FILTER_TYPE="lowpass"
export FFT_CUTOFF_RATIO="0.3"
export FFT_ALPHA="0.5"
export FFT_ADAPTIVE_CUTOFF="true"
export FFT_USE_SALIENCY="false"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh

# 激活环境并运行（中文到英语）
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft_dataset && \
python code/gemma_covost2.py \
    --direction zh_en \
    --model-path "${MODEL_PATH}" \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E2B_it/datasets/CoVoST2/ZH_EN/vanilla/full_cache \
    --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E2B_it/datasets/CoVoST2/ZH_EN/vanilla/full_cache/timing_stats.json

