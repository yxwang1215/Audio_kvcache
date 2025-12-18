#!/bin/bash
# speech-web-questions 完整流程脚本
# 依次执行推理和打分（可选：使用独立脚本分别执行）

set -e

# =================== 用户配置区 ===================
KV_METHOD=${1:-sparsekv}           # 第1个参数
BUDGET=${2:-0.8}                   # 第2个参数
MODEL_SIZE=${3:-3B}                # 第3个参数
CUDA_VISIBLE_DEVICES=${4:-3}       # 第4个参数
NAME=${5:-SnapKV}                  # 第5个参数

export CUDA_VISIBLE_DEVICES
export MODEL_PATH="/mnt/public/models/Qwen2.5-Omni-${MODEL_SIZE}"
# export JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/mnt/users/hepeize-20251016/Model/Qwen3-8B}"  # 本地 judge 模型路径（可选）
# export JUDGE_API_MODEL="${JUDGE_API_MODEL:-gpt4o-mini}" 
export OUTPUT_DIR="/mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_${MODEL_SIZE}/datasets/speech_web_questions_Results/${NAME}/budget${BUDGET}_win32_ratio0.5_kernel5_poolmax_ketopk24_alpha1_beta1_gamma1_1215"

# 推理配置
export SAMPLE_LIMIT=0
export KV_METHOD
export BUDGET
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export HEAD_SCORE_PATH="/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/head_score/qwen2_5_${MODEL_SIZE}/audio_head_scores_topk24_alpha1_beta1_gamma1.json"
export HEAD_SCORE_AGGREGATION="first"

# FFT 滤波配置（可选）
# export USE_FFT_FILTER="true"
# export FFT_FILTER_TYPE="lowpass"
# export FFT_CUTOFF_RATIO="0.3"
# export FFT_ALPHA="0.5"
# export FFT_ADAPTIVE_CUTOFF="true"
# export FFT_USE_SALIENCY="false"
# =============================================================


export JUDGE_API_MODEL="${JUDGE_API_MODEL:-gemini-2.5-flash}" 
export JUDGE_API_KEY="${JUDGE_API_KEY:-sk-nSeZoIkSf8DGjeO5IvHnZpEc9O0QqeJrUyiJ5Oij55LW4QjF}"  
export JUDGE_API_BASE_URL="${JUDGE_API_BASE_URL:-http://123.129.219.111:3000/v1}"  
# export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_7B/datasets/speech_chatbot_alpaca_eval_Results/SparseKV/budget0.2_topk24_alpha1_beta1_gamma1_1215}"
export INFERENCE_OUTPUT_FILE="${INFERENCE_OUTPUT_FILE:-${OUTPUT_DIR}/speech_chatbot_alpaca_eval_results_qwen25.json}"
export JUDGE_OUTPUT_FILE="${JUDGE_OUTPUT_FILE:-${OUTPUT_DIR}/judge_results.jsonl}"
export FORCE_REJUDGE="${FORCE_REJUDGE:-true}"


GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== 启动 speech-web-questions 完整流程 ===${NC}"
echo -e "${YELLOW}提示: 也可以分别运行以下脚本:${NC}"
echo -e "${YELLOW}  1. bash run_speech-web-questions_inference.sh  (推理)${NC}"
echo -e "${YELLOW}  2. bash run_speech-web-questions_judge.sh       (打分)${NC}"
echo ""
echo "Config: GPU=${CUDA_VISIBLE_DEVICES}, Model=${MODEL_PATH}"
echo ">>> KV_METHOD=${KV_METHOD}, BUDGET=${BUDGET}"
echo ">>> 清空旧结果目录"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")/code"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh

# 1. 推理
echo -e "${GREEN}>>> [1/2] 推理...${NC}"
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_dataset && \
python code/qwen2.5_speech-web-questions.py \
    --mode inference \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --timing-output-file "${OUTPUT_DIR}/timing_stats.json"

# 确定推理结果文件路径
INFERENCE_OUTPUT_FILE="${OUTPUT_DIR}/speech_web_questions_results_qwen25.json"

# 2. 打分（使用 LLM as Judge）
echo -e "${GREEN}>>> [2/2] 裁判打分...${NC}"
JUDGE_ARGS=""
if [ -n "${JUDGE_MODEL_PATH}" ]; then
    JUDGE_ARGS="--judge-model-path ${JUDGE_MODEL_PATH}"
    echo "Using local judge model: ${JUDGE_MODEL_PATH}"
elif [ -n "${JUDGE_API_MODEL}" ]; then
    JUDGE_ARGS="--judge-api-model ${JUDGE_API_MODEL}"
    echo "Using API judge model: ${JUDGE_API_MODEL}"
    # ======【最小新增】======
    JUDGE_ARGS="${JUDGE_ARGS} --judge-api-key ${JUDGE_API_KEY}"
    JUDGE_ARGS="${JUDGE_ARGS} --judge-api-base-url ${JUDGE_API_BASE_URL}"
    # =======================
else
    echo -e "${YELLOW}Warning: No judge model specified, skipping judge step${NC}"
    echo -e "${YELLOW}可以稍后运行: bash run_speech-web-questions_judge.sh${NC}"
    exit 0
fi

conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_dataset && \
python code/qwen2.5_speech-web-questions.py \
    --mode judge \
    --input-file "${INFERENCE_OUTPUT_FILE}" \
    --output-file "${OUTPUT_DIR}/judge_results.jsonl" \
    ${JUDGE_ARGS}

echo -e "${GREEN}=== 流程结束 ===${NC}"

