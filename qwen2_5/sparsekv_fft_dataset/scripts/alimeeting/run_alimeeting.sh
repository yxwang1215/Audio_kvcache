#!/bin/bash
# AliMeeting 基础评估脚本
# 使用单次推理模式处理整个音频

set -e

# =================== 用户配置区 ===================
export GPU_ID="6"

# 剪枝设置
export PRUNE_RATIO=0.0
export PRUNE_METHOD="random"
export PRUNE_LAYER_IDX=1

# 显存碎片整理
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# 模型和数据路径（请根据实际情况修改）
export MODEL_PATH="${MODEL_PATH:-/mnt/users/hepeize-20251016/Model/Qwen2.5-Omni-7B}"
export JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/mnt/users/hepeize-20251016/Model/Qwen3-8B}"
export ROOT_DIR="${ROOT_DIR:-/mnt/users/hepeize-20251016}"
export CSV_PATH="${CSV_PATH:-${ROOT_DIR}/AliMeeting/Test_Ali.csv}"
export INPUT_DATA="${INPUT_DATA:-${ROOT_DIR}/Dataset/AudioMarathon_long/alimeeting_test.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/src/Qwen_2.5_Omni/AliMeeting/results}"
# =============================================================

export CUDA_VISIBLE_DEVICES=$GPU_ID

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== 启动 AliMeeting 评估 ===${NC}"
echo "Config: GPU=${GPU_ID}, Method=${PRUNE_METHOD}, Ratio=${PRUNE_RATIO}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")/code"

# 1. 构建数据
echo -e "${GREEN}>>> [1/4] 构建数据集 (解析 TextGrid)...${NC}"
python "${CODE_DIR}/qwen2.5_alimeeting.py" \
    --mode build \
    --csv-path "${CSV_PATH}" \
    --output-file "${INPUT_DATA}" \
    --root-dir "${ROOT_DIR}"

# 2. 推理
echo -e "${GREEN}>>> [2/4] 推理...${NC}"
python "${CODE_DIR}/qwen2.5_alimeeting.py" \
    --mode inference \
    --model-path "${MODEL_PATH}" \
    --input-file "${INPUT_DATA}" \
    --output-file "${OUTPUT_DIR}/alimeeting_prediction.jsonl"

# 3. 打分
echo -e "${GREEN}>>> [3/4] 裁判打分...${NC}"
python "${CODE_DIR}/qwen2.5_alimeeting.py" \
    --mode judge \
    --judge-model-path "${JUDGE_MODEL_PATH}" \
    --input-file "${OUTPUT_DIR}/alimeeting_prediction.jsonl" \
    --output-file "${OUTPUT_DIR}/alimeeting_scores.jsonl"

echo -e "${BLUE}=== 流程结束 ===${NC}"

