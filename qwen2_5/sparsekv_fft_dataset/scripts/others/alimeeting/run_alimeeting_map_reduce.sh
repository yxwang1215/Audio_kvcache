#!/bin/bash
# AliMeeting Map-Reduce 评估脚本
# 使用 Map-Reduce 模式处理长音频：先分段提取关键信息，再汇总

set -e

# =================== 用户配置区 ===================
export GPU_ID="6"

# 剪枝设置（Map阶段开启）
export PRUNE_RATIO=0.0
export PRUNE_METHOD="frame"
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

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== 启动 AliMeeting Map-Reduce 评估 ===${NC}"
echo "Config: GPU=${GPU_ID}, Method=${PRUNE_METHOD}, Ratio=${PRUNE_RATIO}"
echo "Chunk=${CHUNK_DURATION_SEC}s, Overlap=${OVERLAP_SEC}s"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")/code"

# 中间文件路径
MAP_OUTPUT="${OUTPUT_DIR}/intermediate_map.jsonl"
FINAL_OUTPUT="${OUTPUT_DIR}/alimeeting_prediction_map_reduce.jsonl"

# 1. 构建数据
echo -e "${GREEN}>>> [1/5] 构建数据集...${NC}"
python "${CODE_DIR}/qwen2.5_alimeeting.py" \
    --mode build \
    --csv-path "${CSV_PATH}" \
    --output-file "${INPUT_DATA}" \
    --root-dir "${ROOT_DIR}"

# 2. Map 阶段：分段提取关键信息
echo -e "${GREEN}>>> [2/5] Map 阶段（提取关键信息）...${NC}"
python "${CODE_DIR}/qwen2.5_alimeeting.py" \
    --mode map \
    --model-path "${MODEL_PATH}" \
    --input-file "${INPUT_DATA}" \
    --output-file "${MAP_OUTPUT}"

# 3. Reduce 阶段：汇总所有片段
echo -e "${GREEN}>>> [3/5] Reduce 阶段（汇总）...${NC}"
python "${CODE_DIR}/qwen2.5_alimeeting.py" \
    --mode reduce \
    --model-path "${MODEL_PATH}" \
    --input-file "${MAP_OUTPUT}" \
    --output-file "${FINAL_OUTPUT}"

# 4. 打分
echo -e "${GREEN}>>> [4/5] 裁判打分...${NC}"
python "${CODE_DIR}/qwen2.5_alimeeting.py" \
    --mode judge \
    --judge-model-path "${JUDGE_MODEL_PATH}" \
    --input-file "${FINAL_OUTPUT}" \
    --output-file "${OUTPUT_DIR}/alimeeting_scores_map_reduce.jsonl"

echo -e "${BLUE}=== 流程结束 ===${NC}"

