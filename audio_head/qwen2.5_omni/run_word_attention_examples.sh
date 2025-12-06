#!/bin/bash

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate Qwen3_2

# 切换到脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 6. 指定所有importance score权重参数（alpha, beta, gamma）
echo "执行命令 6: 指定所有importance score权重参数（处理全部样本）..."
conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=0 python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/word_attention_analysis.py \
    --num_samples 0 \
    --topk 32 \
    --alpha 1.0 \
    --beta 1.0 \
    --gamma 1.0 \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/results/results_custom_weights

# 8. 只关注hit ratio（设置alpha=1, beta=0, gamma=0）
echo "执行命令 8: 只关注hit ratio（处理全部样本）..."
conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=1 python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/word_attention_analysis.py \
    --num_samples 0 \
    --topk 32 \
    --alpha 1.0 \
    --beta 0.0 \
    --gamma 0.0 \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/results/results_hit_only

# 9. 只关注attention ratio（设置alpha=0, beta=1, gamma=0）
echo "执行命令 9: 只关注attention ratio（处理全部样本）..."
conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=2 python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/word_attention_analysis.py \
    --num_samples 0 \
    --topk 32 \
    --alpha 0.0 \
    --beta 1.0 \
    --gamma 0.0 \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/results/results_attn_only

# 10. 只关注coverage ratio（设置alpha=0, beta=0, gamma=1）
echo "执行命令 10: 只关注coverage ratio（处理全部样本）..."
conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES=3 python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/word_attention_analysis.py \
    --num_samples 0 \
    --topk 32 \
    --alpha 0.0 \
    --beta 0.0 \
    --gamma 1.0 \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/results/results_cov_only

echo "所有命令执行完成！"
