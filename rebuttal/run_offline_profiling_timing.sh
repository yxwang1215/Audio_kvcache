#!/bin/bash
# Measure one-time offline head-profiling runtime for rebuttal jV7P-Q4.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=3 bash run_offline_profiling_timing.sh
#   SAMPLE_LIMIT=16 TOPK=24 CUDA_VISIBLE_DEVICES=3 bash run_offline_profiling_timing.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export CONDA_ENV="${CONDA_ENV:-Qwen3_2}"
export MODEL_PATH="${MODEL_PATH:-/mnt/public/models/Qwen2.5-Omni-3B}"
export AUDIO_DIR="${AUDIO_DIR:-/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/output_audio}"
export WHISPERX_DIR="${WHISPERX_DIR:-/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/whisperx_parquet_output}"
export SAMPLE_LIMIT="${SAMPLE_LIMIT:-8}"
export TOPK="${TOPK:-24}"
export OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/profiling_timing_benchmark/topk${TOPK}}"

mkdir -p "${OUT_DIR}"

echo "========== Offline head-profiling timing =========="
echo "GPU=${CUDA_VISIBLE_DEVICES}  samples=${SAMPLE_LIMIT}  topk=${TOPK}"
echo "Output=${OUT_DIR}"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
set +u
conda activate "${CONDA_ENV}"
set -u

/usr/bin/time -v python "${REPO_DIR}/audio_head/qwen2.5_omni/word_attention_analysis.py" \
  --model_path "${MODEL_PATH}" \
  --audio_dir "${AUDIO_DIR}" \
  --whisperx_dir "${WHISPERX_DIR}" \
  --output_dir "${OUT_DIR}" \
  --num_samples "${SAMPLE_LIMIT}" \
  --topk "${TOPK}" \
  --skip_detailed_analysis 2>&1 | tee "${SCRIPT_DIR}/profiling_timing_benchmark/run_topk${TOPK}_n${SAMPLE_LIMIT}.log"
