#!/bin/bash
# AliMeeting 多轮对话评估脚本
# 使用多轮对话模式分段处理长音频

set -e

# =================== 用户配置区 ===================
export GPU_ID="7"

# 剪枝设置
export PRUNE_RATIO=0.0
export PRUNE_METHOD="random"
export PRUNE_LAYER_IDX=1

# 音频切片配置
export CHUNK_DURATION_SEC=1200  # 20分钟
export OVERLAP_SEC=30            # 30秒重叠

# 显存碎片整理
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# 模型和数据路径（请根据实际情况修改）
export MODEL_PATH="${MODEL_PATH:-/mnt/users/hepeize-20251016/Model/Qwen2.5_Omni}"
export JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/mnt/users/hepeize-20251016/Model/Qwen3-8B}"
export ROOT_DIR="${ROOT_DIR:-/mnt/users/hepeize-20251016}"
export CSV_PATH="${CSV_PATH:-${ROOT_DIR}/AliMeeting/Test_Ali.csv}"
export INPUT_DATA="${INPUT_DATA:-${ROOT_DIR}/Dataset/AudioMarathon_long/alimeeting_test.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/users/hepeize-20251016/src/Qwen_2.5_Omni/AliMeeting/results}"
# =============================================================

export CUDA_VISIBLE_DEVICES=$GPU_ID
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== 启动 AliMeeting 多轮对话评测 ===${NC}"
echo "Config: GPU=${GPU_ID}, Method=${PRUNE_METHOD}, Ratio=${PRUNE_RATIO}"
echo "Chunk=${CHUNK_DURATION_SEC}s, Overlap=${OVERLAP_SEC}s"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")/code"

# 1. 构建数据
echo -e "${GREEN}>>> [1/4] 构建数据集...${NC}"
python "${CODE_DIR}/qwen2.5_alimeeting.py" \
    --mode build \
    --csv-path "${CSV_PATH}" \
    --output-file "${INPUT_DATA}" \
    --root-dir "${ROOT_DIR}"

# 2. 推理（多轮对话）
echo -e "${GREEN}>>> [2/4] 多轮对话推理...${NC}"
python -u "${CODE_DIR}/qwen2.5_alimeeting.py" \
    --mode chat \
    --model-path "${MODEL_PATH}" \
    --input-file "${INPUT_DATA}" \
    --output-file "${OUTPUT_DIR}/alimeeting_prediction_chat.jsonl"

# 3. 打分
echo -e "${GREEN}>>> [3/4] 裁判打分...${NC}"
python "${CODE_DIR}/qwen2.5_alimeeting.py" \
    --mode judge \
    --judge-model-path "${JUDGE_MODEL_PATH}" \
    --input-file "${OUTPUT_DIR}/alimeeting_prediction_chat.jsonl" \
    --output-file "${OUTPUT_DIR}/alimeeting_scores_chat.jsonl"

echo -e "${GREEN}=== 流程结束 ===${NC}"

