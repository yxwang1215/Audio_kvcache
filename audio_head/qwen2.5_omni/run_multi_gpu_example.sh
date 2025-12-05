#!/bin/bash
# 多 GPU 运行示例脚本

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置 GPU 列表（可以根据实际情况修改）
GPUS="0,5,6,7"

# 基本参数
MODEL_PATH="/mnt/users/wangyuxuan-20250915/Models/Qwen2.5-Omni-3B"
AUDIO_DIR="/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/output_audio"
WHISPERX_DIR="/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/whisperx_parquet_output"
OUTPUT_DIR="/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni"

# 运行参数
NUM_SAMPLES=0  # 要分析的样本数量（设置为 None 或 0 表示全部）
SCORE_THRESHOLD=0.95
TOPK=32

# 参数搜索设置
ENABLE_SEARCH=true
ALPHA_RANGE="0.0,1.0,2.0"
BETA_RANGE="0.0,1.0,2.0"
GAMMA_RANGE="0.0,1.0,2.0"
# ALPHA_RANGE="0.0,0.5,1.0,1.5,2.0"
# BETA_RANGE="0.0,0.5,1.0,1.5,2.0"
# GAMMA_RANGE="0.0,0.5,1.0,1.5,2.0"

# 运行脚本
python run_head_scores_search_multi_gpu.py \
    --model_path "$MODEL_PATH" \
    --audio_dir "$AUDIO_DIR" \
    --whisperx_dir "$WHISPERX_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES \
    --score_threshold $SCORE_THRESHOLD \
    --topk $TOPK \
    --gpus "$GPUS" \
    --enable_search \
    --alpha_range "$ALPHA_RANGE" \
    --beta_range "$BETA_RANGE" \
    --gamma_range "$GAMMA_RANGE" \
    --num_layers 36 \
    --num_heads 16

echo "Multi-GPU analysis completed!"

