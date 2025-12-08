#!/bin/bash
# speech-triviaqa Judge 打分脚本
# 使用 LLM as Judge 对推理结果进行打分

set -e

# =================== 用户配置区 ===================
# Judge 模型配置（二选一）
# export JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/mnt/users/hepeize-20251016/Model/Qwen3-8B}"  # 本地 judge 模型路径
export JUDGE_API_MODEL="${JUDGE_API_MODEL:-gpt4o-mini}"  # API judge 模型名称（如 gpt4o-mini, qwen-audio）

# 输入输出路径
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_3B/datasets/speech_triviaqa_Results}"
export INFERENCE_OUTPUT_FILE="${INFERENCE_OUTPUT_FILE:-${OUTPUT_DIR}/speech_triviaqa_results_qwen25.json}"
export JUDGE_OUTPUT_FILE="${JUDGE_OUTPUT_FILE:-${OUTPUT_DIR}/judge_results.jsonl}"
# =============================================================

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== 启动 speech-triviaqa Judge 打分 ===${NC}"

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
else
    echo -e "${RED}Error: 必须指定 JUDGE_MODEL_PATH 或 JUDGE_API_MODEL${NC}"
    echo -e "${RED}例如: export JUDGE_API_MODEL=\"gpt4o-mini\"${NC}"
    exit 1
fi

# 打分（使用 LLM as Judge）
echo -e "${GREEN}>>> 裁判打分...${NC}"
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_dataset && \
python code/qwen2.5_speech-triviaqa.py \
    --mode judge \
    --input-file "${INFERENCE_OUTPUT_FILE}" \
    --output-file "${JUDGE_OUTPUT_FILE}" \
    ${JUDGE_ARGS}

echo -e "${GREEN}=== Judge 打分完成 ===${NC}"
echo -e "${BLUE}打分结果保存在: ${JUDGE_OUTPUT_FILE}${NC}"

