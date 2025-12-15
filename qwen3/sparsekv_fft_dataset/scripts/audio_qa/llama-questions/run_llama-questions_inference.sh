#!/bin/bash
# llama-questions 推理脚本
# 用于生成模型答案

set -e

# =================== 用户配置区 ===================
export GPU_ID="5"
export MODEL_PATH="${MODEL_PATH:-/mnt/public/models/Qwen3-Omni-30B-A3B-Instruct}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/users/wangyuxuan-20250915/Audio/Results/Qwen3-Omni-30B-A3B-Instruct/datasets/llama_questions_Results/SparseKV/budget0.8}"

export SAMPLE_LIMIT=0
# KV Cache compression method: sparsekv, snapkv, adakv, pyramidkv, mixkv, vanilla
export KV_METHOD="sparsekv"
export BUDGET=0.8
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export HEAD_SCORE_PATH="/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen3_omni/head_score/audio_head_scores_topk16_alpha1_beta1_gamma1.json"
# export HEAD_SCORE_FILE="uniform"
export HEAD_SCORE_AGGREGATION="first"
# export HEAD_SCORE_BATCH=None  # 可选

# FFT 滤波配置（可选）
# export USE_FFT_FILTER="true"
# export FFT_FILTER_TYPE="lowpass"
# export FFT_CUTOFF_RATIO="0.3"
# export FFT_ALPHA="0.5"
# export FFT_ADAPTIVE_CUTOFF="false"
# export FFT_USE_SALIENCY="false"
# =============================================================

export CUDA_VISIBLE_DEVICES=$GPU_ID
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== 启动 llama-questions 推理 ===${NC}"
echo "Config: GPU=${GPU_ID}, Model=${MODEL_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")/code"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh

# 推理
echo -e "${GREEN}>>> 推理生成答案...${NC}"
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/sparsekv_fft_dataset && \
python code/qwen3_llama-questions.py \
    --mode inference \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --timing-output-file "${OUTPUT_DIR}/timing_stats.json"

# 确定推理结果文件路径
INFERENCE_OUTPUT_FILE="${OUTPUT_DIR}/llama_questions_results_qwen3.json"

echo -e "${GREEN}=== 推理完成 ===${NC}"
echo -e "${BLUE}推理结果保存在: ${INFERENCE_OUTPUT_FILE}${NC}"
echo -e "${BLUE}可以使用以下命令进行打分:${NC}"
echo -e "${BLUE}  bash run_llama-questions_judge.sh${NC}"

