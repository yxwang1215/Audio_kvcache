#!/bin/bash
set -e

# =================== 用户配置区 ===================
export GPU_ID="6"

# 1. 剪枝设置: Random + 0.0
export PRUNE_RATIO=0.0
export PRUNE_METHOD="random"
export PRUNE_LAYER_IDX=1

# 2. 显存碎片整理
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
# =============================================================

export CUDA_VISIBLE_DEVICES=$GPU_ID

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== 启动 AliMeeting 评估 ===${NC}"
echo "Config: GPU=${GPU_ID}, Method=${PRUNE_METHOD}, Ratio=${PRUNE_RATIO}"

# 1. 构建数据
echo -e "${GREEN}>>> [1/3] 构建数据集 (解析 TextGrid)...${NC}"
python /mnt/users/hepeize-20251016/src/Qwen_2.5_Omni/AliMeeting/1_build_dataset.py

# 2. 推理
echo -e "${GREEN}>>> [2/3] 推理...${NC}"
python /mnt/users/hepeize-20251016/src/Qwen_2.5_Omni/AliMeeting/2_run_inference.py

# 3. 打分
echo -e "${GREEN}>>> [3/3] 裁判打分...${NC}"
python /mnt/users/hepeize-20251016/src/Qwen_2.5_Omni/AliMeeting/3_llm_judge.py

echo -e "${BLUE}=== 流程结束 ===${NC}"