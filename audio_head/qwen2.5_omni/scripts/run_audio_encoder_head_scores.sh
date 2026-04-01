#!/bin/bash
# Generate Audio Encoder Head Importance Scores
# ==============================================================================

# 配置
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/mnt/public/models/Qwen2.5-Omni-7B"

# 音频文件路径（参考 word_attention_analysis.py 的方式）
export AUDIO_DIR="/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/output_audio"
export WHISPERX_DIR="/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/whisperx_parquet_output"

# 方式2: 指定具体的音频文件列表（如果设置了，会优先使用）
# export AUDIO_FILES=(
#     "/path/to/audio1.wav"
#     "/path/to/audio2.wav"
# )

# 输出文件路径
export OUTPUT_FILE="/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/head_score/qwen2_5_7B/audio_encoder_head_scores.json"

# 样本选择（参考 word_attention_analysis.py）
# 方式1: 使用范围索引
export START_IDX=0
export END_IDX=10

# 方式2: 使用数量限制（如果设置了，会优先使用）
# export NUM_SAMPLES=10

# 可选：限制处理的文件数量（已废弃，建议使用 START_IDX/END_IDX 或 NUM_SAMPLES）
# export MAX_FILES=10

# 切换到脚本目录
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni || exit 1

# 激活环境
source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 || exit 1

# 构建命令
CMD="python generate_audio_encoder_head_scores.py"
CMD="$CMD --model_path \"${MODEL_PATH}\""
CMD="$CMD --output_file \"${OUTPUT_FILE}\""
CMD="$CMD --device cuda"

# 添加音频文件参数
if [ -n "${AUDIO_FILES[*]}" ]; then
    # 使用指定的音频文件列表
    CMD="$CMD --audio_files ${AUDIO_FILES[*]}"
elif [ -n "${AUDIO_DIR}" ]; then
    # 使用音频目录
    CMD="$CMD --audio_dir \"${AUDIO_DIR}\""
    
    # 添加 WhisperX 目录（如果指定）
    if [ -n "${WHISPERX_DIR}" ]; then
        CMD="$CMD --whisperx_dir \"${WHISPERX_DIR}\""
    fi
else
    echo "Error: Must specify either AUDIO_DIR or AUDIO_FILES"
    exit 1
fi

# 添加样本选择参数（优先级：start_idx/end_idx > num_samples > max_files）
if [ -n "${START_IDX}" ] && [ -n "${END_IDX}" ]; then
    CMD="$CMD --start_idx ${START_IDX} --end_idx ${END_IDX}"
elif [ -n "${NUM_SAMPLES}" ] && [ "${NUM_SAMPLES}" -gt 0 ]; then
    CMD="$CMD --num_samples ${NUM_SAMPLES}"
elif [ -n "${MAX_FILES}" ] && [ "${MAX_FILES}" -gt 0 ]; then
    # 兼容旧的 max_files 参数
    CMD="$CMD --max_files ${MAX_FILES}"
fi

# 打印配置
echo "=============================================================================="
echo "Audio Encoder Head Importance Score Generation"
echo "=============================================================================="
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_FILE}"
if [ -n "${AUDIO_FILES[*]}" ]; then
    echo "Audio files: ${#AUDIO_FILES[@]} files"
    for f in "${AUDIO_FILES[@]}"; do
        echo "  - $f"
    done
elif [ -n "${AUDIO_DIR}" ]; then
    echo "Audio directory: ${AUDIO_DIR}"
    if [ -n "${WHISPERX_DIR}" ]; then
        echo "WhisperX directory: ${WHISPERX_DIR} (for reference)"
    fi
fi
if [ -n "${START_IDX}" ] && [ -n "${END_IDX}" ]; then
    echo "Range: [${START_IDX}:${END_IDX}]"
elif [ -n "${NUM_SAMPLES}" ] && [ "${NUM_SAMPLES}" -gt 0 ]; then
    echo "Count: first ${NUM_SAMPLES} files"
elif [ -n "${MAX_FILES}" ] && [ "${MAX_FILES}" -gt 0 ]; then
    echo "Max files: ${MAX_FILES}"
fi
echo "=============================================================================="
echo ""

# 运行命令
eval $CMD

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================================================="
    echo "✓ Head scores generated successfully!"
    echo "Output file: ${OUTPUT_FILE}"
    echo "=============================================================================="
else
    echo ""
    echo "=============================================================================="
    echo "✗ Failed to generate head scores"
    echo "=============================================================================="
    exit 1
fi

