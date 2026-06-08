#!/usr/bin/env bash
# ovZo-Q2: SSS token-scope ablation.
#
# Compares:
#   FullHistory_SSS: RFFT smoothing over the full historical importance sequence
#   AudioOnly_SSS:   RFFT smoothing only over detected audio-token span
#
# Example:
#   GPU_ID=5 TASK=multilingual-spanish BUDGET=0.8 bash rebuttal/run_sss_scope_ablation.sh
#   TASK=covost2-zh-en COVOST2_SAMPLE_FRACTION=0.2 bash rebuttal/run_sss_scope_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TASK="${TASK:-multilingual-spanish}"
GPU_ID="${GPU_ID:-5}"
BUDGET="${BUDGET:-0.8}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
FORCE_RERUN="${FORCE_RERUN:-false}"

CODE_ROOT="${ROOT_DIR}/qwen2_5/sparsekv_fft_dataset"
MODEL_PATH="${MODEL_PATH:-/mnt/public/models/Qwen2.5-Omni-3B}"
AUDIO_HEAD_JSON="${AUDIO_HEAD_JSON:-${ROOT_DIR}/audio_head/qwen2.5_omni/head_score/qwen2_5_3B/audio_head_scores_topk24_alpha1_beta1_gamma1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_3B/ablation_ovzo_q2_sss_scope}"
CONDA_ENV="${CONDA_ENV:-Qwen3_2}"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export SAMPLE_LIMIT
export MLS_SAMPLE_FRACTION="${MLS_SAMPLE_FRACTION:-0.2}"
export COVOST2_SAMPLE_FRACTION="${COVOST2_SAMPLE_FRACTION:-0.2}"
export KESPEECH_SAMPLE_FRACTION="${KESPEECH_SAMPLE_FRACTION:-0.2}"
export THINKER_MAX_NEW_TOKENS="${THINKER_MAX_NEW_TOKENS:-64}"
export BUDGET
export RATIO="${RATIO:-0.5}"
export WINDOW_SIZE="${WINDOW_SIZE:-32}"
export KERNEL_SIZE="${KERNEL_SIZE:-5}"
export POOLING="${POOLING:-maxpool}"
export HEAD_SCORE_AGGREGATION="${HEAD_SCORE_AGGREGATION:-first}"
export KV_METHOD="sparsekv"
export USE_FFT_FILTER="true"
export SCORE_SMOOTH_METHOD="sss"
export FFT_FILTER_TYPE="${FFT_FILTER_TYPE:-lowpass}"
export FFT_CUTOFF_RATIO="${FFT_CUTOFF_RATIO:-0.7}"
export FFT_ALPHA="${FFT_ALPHA:-0.5}"
export FFT_ADAPTIVE_CUTOFF="${FFT_ADAPTIVE_CUTOFF:-false}"
export FFT_USE_SALIENCY="${FFT_USE_SALIENCY:-false}"

resolve_task_python() {
  EXTRA_PY_ARGS=()
  case "${TASK}" in
    librispeech)
      PY_SCRIPT="${CODE_ROOT}/code/qwen2.5_librispeech.py"
      RESULT_BASENAME="librispeech_results_qwen25.json"
      ;;
    kespeech)
      PY_SCRIPT="${CODE_ROOT}/code/qwen2.5_kespeech.py"
      RESULT_BASENAME="kespeech_results_qwen25.json"
      ;;
    multilingual-spanish|mls-spanish|spanish)
      PY_SCRIPT="${CODE_ROOT}/code/qwen2.5_multilingual_librispeech.py"
      RESULT_BASENAME="multilingual_librispeech_results_qwen25.json"
      EXTRA_PY_ARGS=(--languages spanish)
      ;;
    multilingual-french|mls-french|french)
      PY_SCRIPT="${CODE_ROOT}/code/qwen2.5_multilingual_librispeech.py"
      RESULT_BASENAME="multilingual_librispeech_results_qwen25.json"
      EXTRA_PY_ARGS=(--languages french)
      ;;
    multilingual-german|mls-german|german)
      PY_SCRIPT="${CODE_ROOT}/code/qwen2.5_multilingual_librispeech.py"
      RESULT_BASENAME="multilingual_librispeech_results_qwen25.json"
      EXTRA_PY_ARGS=(--languages german)
      ;;
    covost2-zh-en|covost2_zh_en)
      PY_SCRIPT="${CODE_ROOT}/code/qwen2.5_covost2.py"
      RESULT_BASENAME="covost2_zh_en_results_qwen25.json"
      EXTRA_PY_ARGS=(--direction zh_en)
      ;;
    covost2-en-zh|covost2_en_zh)
      PY_SCRIPT="${CODE_ROOT}/code/qwen2.5_covost2.py"
      RESULT_BASENAME="covost2_en_zh_results_qwen25.json"
      EXTRA_PY_ARGS=(--direction en_zh)
      ;;
    *)
      echo "Unsupported TASK=${TASK}" >&2
      echo "Use librispeech|kespeech|multilingual-spanish|multilingual-french|multilingual-german|covost2-zh-en|covost2-en-zh" >&2
      exit 1
      ;;
  esac
}

is_done() {
  local out_file="$1"
  [ "${FORCE_RERUN}" = "true" ] && return 1
  [ -f "${out_file}" ] || return 1
  python3 - "${out_file}" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
    sys.exit(0 if isinstance(data, dict) and data.get("status") == "completed" else 1)
except Exception:
    sys.exit(1)
PY
}

run_scope_condition() {
  local label="$1"
  local scope="$2"
  local output_dir="${OUTPUT_ROOT}/${TASK}/${label}/budget${BUDGET}"
  local timing_file="${output_dir}/timing_stats_sparsekv_${label}.json"
  local result_file="${output_dir}/${RESULT_BASENAME}"

  if is_done "${result_file}"; then
    echo "[skip] ${label} already completed: ${result_file}"
    return 0
  fi

  export HEAD_SCORE_PATH="${AUDIO_HEAD_JSON}"
  export OUTPUT_DIR="${output_dir}"
  export SSS_TOKEN_SCOPE="${scope}"

  mkdir -p "${output_dir}"
  echo ""
  echo ">>> Running ${label}"
  echo "    SSS_TOKEN_SCOPE=${SSS_TOKEN_SCOPE}"
  echo "    OUTPUT_DIR=${output_dir}"

  conda activate "${CONDA_ENV}"
  cd "${CODE_ROOT}"

  unset SAMPLE_FRACTION
  if [ "${SAMPLE_LIMIT}" = "0" ]; then
    if [[ "${TASK}" == covost2* ]]; then
      export SAMPLE_FRACTION="${COVOST2_SAMPLE_FRACTION}"
    elif [[ "${TASK}" == kespeech ]]; then
      export SAMPLE_FRACTION="${KESPEECH_SAMPLE_FRACTION}"
    elif [[ "${TASK}" == multilingual-* || "${TASK}" == mls-* || "${TASK}" == spanish || "${TASK}" == french || "${TASK}" == german ]]; then
      export SAMPLE_FRACTION="${MLS_SAMPLE_FRACTION}"
    fi
    [ -z "${SAMPLE_FRACTION:-}" ] || echo "    SAMPLE_FRACTION=${SAMPLE_FRACTION}"
  fi

  python "${PY_SCRIPT}" \
    "${EXTRA_PY_ARGS[@]}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${output_dir}" \
    --timing-output-file "${timing_file}"
}

resolve_task_python

echo "=== ovZo-Q2 SSS token-scope ablation ==="
echo "Task: ${TASK}  GPU: ${GPU_ID}  Budget: ${BUDGET}"
echo "Audio-aware head scores: ${AUDIO_HEAD_JSON}"

run_scope_condition "FullHistory_SSS" "full"
run_scope_condition "AudioOnly_SSS" "audio_only"

echo ""
echo "Done. Results under: ${OUTPUT_ROOT}/${TASK}"
