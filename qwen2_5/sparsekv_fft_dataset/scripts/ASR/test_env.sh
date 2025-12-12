#!/bin/bash
# 测试环境变量是否被正确传递

export HEAD_SCORE_PATH="/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/head_score/audio_head_scores_topk32_alpha0_beta0_gamma1.json"
export HEAD_SCORE_AGGREGATION="first"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
python3 -c "import os; path = os.environ.get('HEAD_SCORE_PATH', None); print(f'HEAD_SCORE_PATH = {path}'); print('Environment variable is', 'SET' if path else 'NOT SET')"

