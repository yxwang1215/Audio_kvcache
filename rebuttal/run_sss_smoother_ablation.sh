#!/usr/bin/env bash
# jV7P-Q2: Compare SSS vs. simpler score smoothers (same SparseKV + head scores + budget).
#
# Methods (SCORE_SMOOTH_METHOD):
#   none | sss | maxpool | gaussian | uniform_stride | uniform_prior
#
# Usage:
#   GPU_ID=5 BUDGET=0.4 bash rebuttal/run_sss_smoother_ablation.sh
#   GPU_ID=5 TASKS="librispeech kespeech multilingual-spanish multilingual-french multilingual-german" bash rebuttal/run_sss_smoother_ablation.sh
#   GPU_ID=5 TASK=librispeech BUDGET=0.4 bash rebuttal/run_sss_smoother_ablation.sh
#   GPU_ID=5 TASK=multilingual-spanish MLS_SAMPLE_FRACTION=0.2 bash rebuttal/run_sss_smoother_ablation.sh
#   METHODS="sss maxpool gaussian" GPU_ID=5 bash rebuttal/run_sss_smoother_ablation.sh
#   SAMPLE_LIMIT=50 bash rebuttal/run_sss_smoother_ablation.sh   # quick subset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-1}"
TASK="${TASK:-}"
TASKS="${TASKS:-${TASK}}"
if [ -z "${TASKS}" ]; then
  TASKS="librispeech kespeech multilingual-spanish multilingual-french multilingual-german"
fi
BUDGET="${BUDGET:-0.4}"
MODEL_FAMILY="${MODEL_FAMILY:-qwen2_5}"
METHODS="${METHODS:-none sss maxpool gaussian uniform_stride uniform_prior}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
FORCE_RERUN="${FORCE_RERUN:-false}"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export SAMPLE_LIMIT
export MLS_SAMPLE_FRACTION="${MLS_SAMPLE_FRACTION:-0.2}"
export KESPEECH_SAMPLE_FRACTION="${KESPEECH_SAMPLE_FRACTION:-0.2}"
export COVOST2_SAMPLE_FRACTION="${COVOST2_SAMPLE_FRACTION:-0.2}"
export BUDGET
export RATIO="${RATIO:-0.5}"
export WINDOW_SIZE="${WINDOW_SIZE:-32}"
export KERNEL_SIZE="${KERNEL_SIZE:-5}"
export POOLING="${POOLING:-maxpool}"
export HEAD_SCORE_AGGREGATION="${HEAD_SCORE_AGGREGATION:-first}"
export KV_METHOD="sparsekv"
export FFT_FILTER_TYPE="${FFT_FILTER_TYPE:-lowpass}"
export FFT_CUTOFF_RATIO="${FFT_CUTOFF_RATIO:-0.7}"
export FFT_ALPHA="${FFT_ALPHA:-0.5}"
export FFT_ADAPTIVE_CUTOFF="${FFT_ADAPTIVE_CUTOFF:-false}"
export FFT_USE_SALIENCY="${FFT_USE_SALIENCY:-false}"
export GAUSSIAN_SIGMA="${GAUSSIAN_SIGMA:-0}"
export STRIDE_BLEND="${STRIDE_BLEND:-${KERNEL_SIZE}}"

CODE_ROOT="${ROOT_DIR}/qwen2_5/sparsekv_fft_dataset"
MODEL_PATH="${MODEL_PATH:-/mnt/public/models/Qwen2.5-Omni-3B}"
AUDIO_HEAD_JSON="${AUDIO_HEAD_JSON:-${ROOT_DIR}/audio_head/qwen2.5_omni/head_score/qwen2_5_3B/audio_head_scores_topk24_alpha1_beta1_gamma1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_3B/ablation_jv7p_q2_smoothers}"
CONDA_ENV="${CONDA_ENV:-Qwen3_2}"

resolve_task_python() {
  EXTRA_PY_ARGS=()
  USE_RESULTS_DIR=false
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
      echo "Unsupported TASK=${TASK} (use librispeech|kespeech|multilingual-spanish|multilingual-french|multilingual-german|covost2-zh-en|covost2-en-zh)"
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
path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    sys.exit(0 if isinstance(d, dict) and d.get("status") == "completed" else 1)
except Exception:
    sys.exit(1)
PY
}

run_task() {
  TASK="$1"
  resolve_task_python

  echo ""
  echo "=== jV7P-Q2 smoother ablation ==="
  echo "Task=${TASK}  Budget=${BUDGET}  GPU=${GPU_ID}"
  echo "Methods: ${METHODS}"
  echo "Head JSON: ${AUDIO_HEAD_JSON}"

  for method in ${METHODS}; do
    label="Smooth_${method}"
    output_dir="${OUTPUT_ROOT}/${TASK}/${label}/budget${BUDGET}"
    result_file="${output_dir}/${RESULT_BASENAME}"
    timing_file="${output_dir}/timing_stats_sparsekv_${label}.json"

    if is_done "${result_file}"; then
      echo "[skip] ${label} — ${result_file}"
      continue
    fi

    export SCORE_SMOOTH_METHOD="${method}"
    if [ "${method}" = "sss" ]; then
      export USE_FFT_FILTER="true"
    else
      export USE_FFT_FILTER="false"
    fi

    export OUTPUT_DIR="${output_dir}"
    export HEAD_SCORE_PATH="${AUDIO_HEAD_JSON}"
    unset HEAD_SCORE_FILE

    mkdir -p "${output_dir}"
    echo ""
    echo ">>> ${label}  SCORE_SMOOTH_METHOD=${method}  USE_FFT_FILTER=${USE_FFT_FILTER}"

    set +u
    conda activate "${CONDA_ENV}"
    set -u
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
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

    echo "    Done: ${result_file}"
  done

  echo ""
  echo "=== Finished ${TASK}. Results: ${OUTPUT_ROOT}/${TASK}/ ==="
}

echo "Planned TASKS: ${TASKS}"
for task in ${TASKS}; do
  run_task "${task}"
done

echo ""
echo "=== Finished all smoother ablations. Results: ${OUTPUT_ROOT}/ ==="
for task in ${TASKS}; do
  echo "Aggregate ${task}: python3 ${SCRIPT_DIR}/aggregate_smoother_ablation.py --root ${OUTPUT_ROOT} --task ${task}"
done
