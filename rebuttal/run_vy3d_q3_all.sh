#!/usr/bin/env bash
# VY3D-Q3 (ASR-only): budget=0.8, head-score JSON ablation.
#
# Compares (SSS fixed ON, same SparseKV budget):
#   Audio-aware JSON  vs  inverse JSON  vs  random-init JSON  (seed 42)
# on LibriSpeech-long ASR by default.
#
#   bash rebuttal/run_vy3d_q3_all.sh
#   GPU_ID=5 TASKS=multilingual-french bash rebuttal/run_vy3d_q3_all.sh
#   MODEL_FAMILY=gemma TASKS=multilingual-spanish,multilingual-french,multilingual-german bash rebuttal/run_vy3d_q3_all.sh
#   TASKS=multilingual-spanish,multilingual-french,multilingual-german MLS_SAMPLE_FRACTION=0.2 bash rebuttal/run_vy3d_q3_all.sh
#   LIBRISPEECH_SAMPLE_FRACTION=0.2 bash rebuttal/run_vy3d_q3_all.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_SH="${SCRIPT_DIR}/run_audio_supervision_ablation.sh"

export MODEL_FAMILY="${MODEL_FAMILY:-qwen2_5}"
export GPU_ID="${GPU_ID:-5}"
export BUDGET="${BUDGET:-0.8}"
export SEEDS="${SEEDS:-42}"
export RUN_INVERSE="${RUN_INVERSE:-true}"
export RUN_UNIFORM="${RUN_UNIFORM:-false}"
export LIBRISPEECH_SAMPLE_FRACTION="${LIBRISPEECH_SAMPLE_FRACTION:-0.2}"
export MLS_SAMPLE_FRACTION="${MLS_SAMPLE_FRACTION:-0.2}"
export KESPEECH_SAMPLE_FRACTION="${KESPEECH_SAMPLE_FRACTION:-0.2}"
export LIBRISPEECH_LONG_DIR="${LIBRISPEECH_LONG_DIR:-/mnt/public/datasets}"
# Comma-separated ASR tasks. Override TASKS to run MLS/KeSpeech variants.
export TASKS="${TASKS:-librispeech}"

IFS=',' read -r -a TASK_LIST <<< "${TASKS}"

for task in "${TASK_LIST[@]}"; do
  task="$(echo "${task}" | xargs)"
  [ -n "${task}" ] || continue
  echo "========== ${MODEL_FAMILY}  task=${task}  budget=${BUDGET} =========="
  bash "${ABLATION_SH}" "${MODEL_FAMILY}" "${task}"
done

if [ "${MODEL_FAMILY}" = "gemma" ]; then
  echo "Done. Results: Results/Gemma_3n_E4B_it/ablation_vy3d_q3/<task>/{AudioAware_SSS,InverseAudioAware_SSS,RandomInit_SSS_seed42}/budget${BUDGET}/"
else
  echo "Done. Results: Results/Qwen2_5_Omni_3B/ablation_vy3d_q3/<task>/{AudioAware_SSS,InverseAudioAware_SSS,RandomInit_SSS_seed42}/budget${BUDGET}/"
fi
