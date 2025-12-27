#!/bin/bash
# CoVoST2 English-to-Chinese Speech Translation Evaluation with Qwen3-Omni using H2O

export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/mnt/public/models/Qwen3-Omni-30B-A3B-Instruct"
export SAMPLE_LIMIT=0

export KV_METHOD="h2o"
export BUDGET=0.4
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="avgpool"
# H2O specific: ratio of heavy hitters vs recent tokens
export H2O_HEAVY_RATIO=0.5

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/sparsekv_fft_dataset && \
python code/qwen3_covost2.py \
    --direction en_zh \
    --model-path "${MODEL_PATH}" \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen3-Omni-30B-A3B-Instruct/datasets/CoVoST2/EN_ZH/H2O/budget0.4 \
    --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen3-Omni-30B-A3B-Instruct/datasets/CoVoST2/EN_ZH/H2O/budget0.4/timing_stats.json

