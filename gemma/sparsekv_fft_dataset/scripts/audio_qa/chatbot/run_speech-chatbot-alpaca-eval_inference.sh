#!/bin/bash
# speech-chatbot-alpaca-eval 推理脚本
# 用于生成模型答案

set -e
export GPU_ID="1"
export MODEL_PATH="${MODEL_PATH:-/mnt/public/models/gemma-3n-E2B-it}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E2B_it/datasets/speech_chatbot_alpaca_eval_Results/SnapKV/budget0.2}"

# 推理配置
export SAMPLE_LIMIT=0
# KV Cache compression method: sparsemm, snapkv, adakv, pyramidkv, mixkv
export KV_METHOD="sparsekv"
export USE_SPARSEKV="true"
export BUDGET=0.2
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
# =============================================================

export CUDA_VISIBLE_DEVICES=$GPU_ID
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== 启动 speech-chatbot-alpaca-eval 推理 ===${NC}"
echo "Config: GPU=${GPU_ID}, Model=${MODEL_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")/code"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh

# 推理
echo -e "${GREEN}>>> 推理生成答案...${NC}"
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft_dataset && \
python code/gemma_speech-chatbot-alpaca-eval.py \
    --mode inference \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --timing-output-file "${OUTPUT_DIR}/timing_stats.json"

# 确定推理结果文件路径
INFERENCE_OUTPUT_FILE="${OUTPUT_DIR}/speech_chatbot_alpaca_eval_results_gemma.json"

echo -e "${GREEN}=== 推理完成 ===${NC}"
echo -e "${BLUE}推理结果保存在: ${INFERENCE_OUTPUT_FILE}${NC}"
echo -e "${BLUE}可以使用以下命令进行打分:${NC}"
echo -e "${BLUE}  bash run_speech-chatbot-alpaca-eval_judge.sh${NC}"

