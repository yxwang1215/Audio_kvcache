#!/bin/bash
# llama-questions 推理脚本

set -e
export GPU_ID="1"
export MODEL_PATH="${MODEL_PATH:-/mnt/public/models/Qwen2.5-Omni-7B}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_7B/datasets/llama_questions_Results/SparseKV/budget0.2}"

export SAMPLE_LIMIT=0
export KV_METHOD="sparsekv"
export BUDGET=0.2
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export HEAD_SCORE_PATH="/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/head_score/qwen2_5_7B/audio_head_scores_topk24_alpha1_beta1_gamma1.json"
# export HEAD_SCORE_FILE="uniform"
export HEAD_SCORE_AGGREGATION="first"

export USE_FFT_FILTER="false"
export FFT_FILTER_TYPE="lowpass"
export FFT_CUTOFF_RATIO="0.7"
export FFT_ALPHA="0.5"
export FFT_ADAPTIVE_CUTOFF="false"
export FFT_USE_SALIENCY="false"

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
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_dataset && \
python code/qwen2.5_llama-questions.py \
    --mode inference \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --timing-output-file "${OUTPUT_DIR}/timing_stats.json"

# 确定推理结果文件路径
INFERENCE_OUTPUT_FILE="${OUTPUT_DIR}/llama_questions_results_qwen25.json"

echo -e "${GREEN}=== 推理完成 ===${NC}"
echo -e "${BLUE}推理结果保存在: ${INFERENCE_OUTPUT_FILE}${NC}"
echo -e "${BLUE}可以使用以下命令进行打分:${NC}"
echo -e "${BLUE}  bash run_llama-questions_judge.sh${NC}"

