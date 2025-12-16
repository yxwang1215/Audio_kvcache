#!/bin/bash
# LibriSpeech-long ASR Evaluation with Gemma-3n-E2B-it

export CUDA_VISIBLE_DEVICES=1
export MODEL_PATH="/mnt/public/models/gemma-3n-E2B-it"
export SAMPLE_LIMIT=0

# export KV_METHOD="adakv"
export KV_METHOD="mixkv"
export SELECT_METHOD="headwisemixkv"
export METHOD="mixsparsemm"
export USE_SPARSEKV="true"
export BUDGET=0.99
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
# HEAD_SCORE_FILE: for sparsemm method, options: "uniform", "aggregated", or path to head score file
export HEAD_SCORE_FILE="uniform"

# export USE_FFT_FILTER="true"
# export FFT_FILTER_TYPE="lowpass"
# export FFT_CUTOFF_RATIO="0.3"
# export FFT_ALPHA="0.5"
# export FFT_ADAPTIVE_CUTOFF="true"
# export FFT_USE_SALIENCY="false"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft_dataset && \
python code/gemma_librispeech_long.py \
    --model-path "${MODEL_PATH}" \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E2B_it/datasets/LibriSpeech_long/PyramidKV/budget0.8 \
    --timing-output-file /mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E2B_it/datasets/LibriSpeech_long/PyramidKV/budget0.8/timing_stats.json

