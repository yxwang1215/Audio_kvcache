#!/bin/bash
# speech-chatbot-alpaca-eval 完整流程脚本
# 依次执行推理和打分（可选：使用独立脚本分别执行）

set -e

# =================== 用户配置区 ===================
export GPU_ID="0"
export MODEL_PATH="${MODEL_PATH:-/mnt/public/models/gemma-3n-E4B-it}"
# export JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/mnt/users/hepeize-20251016/Model/Qwen3-8B}"  # 本地 judge 模型路径（可选）
export JUDGE_API_MODEL="${JUDGE_API_MODEL:-gpt4o-mini}" 
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E4B_it/datasets/speech_chatbot_alpaca_eval_Results}"

# 推理配置
export SAMPLE_LIMIT=0
# KV Cache compression method: sparsemm, snapkv, adakv, pyramidkv, mixkv (disabled when USE_SPARSEKV=false)
export KV_METHOD="sparsekv"
export USE_SPARSEKV="false"
export BUDGET=0.7
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export HEAD_SCORE_FILE="aggregated"
# export HEAD_SCORE_BATCH=None  # 可选

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
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== 启动 speech-chatbot-alpaca-eval 完整流程 ===${NC}"
echo -e "${YELLOW}提示: 也可以分别运行以下脚本:${NC}"
echo -e "${YELLOW}  1. bash run_speech-chatbot-alpaca-eval_inference.sh  (推理)${NC}"
echo -e "${YELLOW}  2. bash run_speech-chatbot-alpaca-eval_judge.sh       (打分)${NC}"
echo ""
echo "Config: GPU=${GPU_ID}, Model=${MODEL_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")/code"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh

# 1. 推理
echo -e "${GREEN}>>> [1/2] 推理...${NC}"
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft_dataset && \
python code/gemma_speech-chatbot-alpaca-eval.py \
    --mode inference \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --timing-output-file "${OUTPUT_DIR}/timing_stats.json"

# 确定推理结果文件路径
INFERENCE_OUTPUT_FILE="${OUTPUT_DIR}/speech_chatbot_alpaca_eval_results_gemma.json"

# 2. 打分（使用 LLM as Judge）
echo -e "${GREEN}>>> [2/2] 裁判打分...${NC}"
JUDGE_ARGS=""
if [ -n "${JUDGE_MODEL_PATH}" ]; then
    JUDGE_ARGS="--judge-model-path ${JUDGE_MODEL_PATH}"
    echo "Using local judge model: ${JUDGE_MODEL_PATH}"
elif [ -n "${JUDGE_API_MODEL}" ]; then
    JUDGE_ARGS="--judge-api-model ${JUDGE_API_MODEL}"
    echo "Using API judge model: ${JUDGE_API_MODEL}"
else
    echo -e "${YELLOW}Warning: No judge model specified, skipping judge step${NC}"
    echo -e "${YELLOW}可以稍后运行: bash run_speech-chatbot-alpaca-eval_judge.sh${NC}"
    exit 0
fi

conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft_dataset && \
python code/gemma_speech-chatbot-alpaca-eval.py \
    --mode judge \
    --input-file "${INFERENCE_OUTPUT_FILE}" \
    --output-file "${OUTPUT_DIR}/judge_results.jsonl" \
    ${JUDGE_ARGS}

echo -e "${GREEN}=== 流程结束 ===${NC}"

