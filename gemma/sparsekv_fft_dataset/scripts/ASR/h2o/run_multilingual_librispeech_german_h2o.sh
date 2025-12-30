#!/bin/bash
# Multilingual LibriSpeech ASR Evaluation with Gemma-3n-E2B-it - German using H2O

export CUDA_VISIBLE_DEVICES=6
export MODEL_PATH="/mnt/public/models/gemma-3n-E2B-it"
export SAMPLE_LIMIT=0

export KV_METHOD="h2o"
export BUDGET=1.0
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="avgpool"
# H2O specific: ratio of heavy hitters vs recent tokens
export H2O_HEAVY_RATIO=0.5

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft_dataset && \
python code/gemma_multilingual_librispeech.py \
    --model-path "${MODEL_PATH}" \
    --languages german \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E2B_it/datasets/MultilingualLibriSpeech/German/vanilla/full_cache \
    --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E2B_it/datasets/MultilingualLibriSpeech/German/vanilla/full_cache/timing_stats.json

