#!/bin/bash
# speech-triviaqa Judge 打分脚本

set -e
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  
export JUDGE_API_MODEL="${JUDGE_API_MODEL:-gemini-2.5-flash}"  
export JUDGE_API_KEY="${JUDGE_API_KEY:-sk-nSeZoIkSf8DGjeO5IvHnZpEc9O0QqeJrUyiJ5Oij55LW4QjF}"  # API Key
export JUDGE_API_BASE_URL="${JUDGE_API_BASE_URL:-http://123.129.219.111:3000/v1}"  # 
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_3B/datasets/speech_triviaqa_Results/SparseKV/budget0.2_topk24_alpha1_beta1_gamma1_1222}"
export INFERENCE_OUTPUT_FILE="${INFERENCE_OUTPUT_FILE:-${OUTPUT_DIR}/speech_triviaqa_results_qwen25.json}"
export JUDGE_OUTPUT_FILE="${JUDGE_OUTPUT_FILE:-${OUTPUT_DIR}/speech_triviaqa_results_qwen25_judge.json}"
export FORCE_REJUDGE="${FORCE_REJUDGE:-true}"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== 启动 speech-triviaqa Judge 打分 ===${NC}"
echo "Config: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# 检查输入文件是否存在
if [ ! -f "${INFERENCE_OUTPUT_FILE}" ]; then
    echo -e "${RED}Error: 推理结果文件不存在: ${INFERENCE_OUTPUT_FILE}${NC}"
    echo -e "${RED}请先运行推理脚本: bash run_speech-triviaqa_inference.sh${NC}"
    exit 1
fi

echo "Input file: ${INFERENCE_OUTPUT_FILE}"
echo "Output file: ${JUDGE_OUTPUT_FILE}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")/code"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh

# 判断使用哪种 judge 模型
JUDGE_ARGS=""
if [ -n "${JUDGE_MODEL_PATH}" ]; then
    JUDGE_ARGS="--judge-model-path ${JUDGE_MODEL_PATH}"
    echo -e "${BLUE}Using local judge model: ${JUDGE_MODEL_PATH}${NC}"
elif [ -n "${JUDGE_API_MODEL}" ]; then
    JUDGE_ARGS="--judge-api-model ${JUDGE_API_MODEL}"
    echo -e "${BLUE}Using API judge model: ${JUDGE_API_MODEL}${NC}"
    
    # 添加 API Key 和 Base URL 参数
    if [ -n "${JUDGE_API_KEY}" ]; then
        JUDGE_ARGS="${JUDGE_ARGS} --judge-api-key \"${JUDGE_API_KEY}\""
        echo -e "${BLUE}API Key: ${JUDGE_API_KEY:0:10}...${NC}"
    fi
    if [ -n "${JUDGE_API_BASE_URL}" ]; then
        JUDGE_ARGS="${JUDGE_ARGS} --judge-api-base-url \"${JUDGE_API_BASE_URL}\""
        echo -e "${BLUE}API Base URL: ${JUDGE_API_BASE_URL}${NC}"
    fi
else
    echo -e "${RED}Error: 必须指定 JUDGE_MODEL_PATH 或 JUDGE_API_MODEL${NC}"
    echo -e "${RED}例如: export JUDGE_API_MODEL=\"gemini-2.5-flash\"${NC}"
    exit 1
fi

# 添加强制重新打分参数
if [ "${FORCE_REJUDGE}" = "true" ]; then
    JUDGE_ARGS="${JUDGE_ARGS} --force-rejudge"
    echo -e "${BLUE}Force rejudge enabled: will re-process all samples${NC}"
fi

# 打分（使用 LLM as Judge）
echo -e "${GREEN}>>> 裁判打分...${NC}"
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_dataset && \
eval python code/qwen2.5_speech-triviaqa.py \
    --mode judge \
    --input-file "${INFERENCE_OUTPUT_FILE}" \
    --output-file "${JUDGE_OUTPUT_FILE}" \
    ${JUDGE_ARGS}

echo -e "${GREEN}=== Judge 打分完成 ===${NC}"
echo -e "${BLUE}打分结果保存在: ${JUDGE_OUTPUT_FILE}${NC}"