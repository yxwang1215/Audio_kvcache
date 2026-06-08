#!/usr/bin/env bash
# VY3D-Q3: head-score JSON ablation only (SSS ON, SparseKV, fixed budget).
#   (A) Audio-aware JSON  vs  (B) inverse JSON  vs  (C) random-init JSON
#
#   GPU_ID=5 bash rebuttal/run_audio_supervision_ablation.sh qwen2_5 multilingual-french
#   GPU_ID=5 bash rebuttal/run_audio_supervision_ablation.sh gemma multilingual-french
#   MODEL_FAMILY=gemma TASKS=multilingual-spanish,multilingual-french,multilingual-german bash rebuttal/run_vy3d_q3_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_FAMILY="${1:-qwen2_5}"
TASK="${2:-librispeech}"
GPU_ID="${GPU_ID:-5}"
BUDGET="${BUDGET:-0.8}"
SEEDS="${SEEDS:-42}"
RUN_INVERSE="${RUN_INVERSE:-true}"
RUN_UNIFORM="${RUN_UNIFORM:-false}"
GENERATE_ONLY="${GENERATE_ONLY:-false}"
FORCE_RERUN="${FORCE_RERUN:-false}"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
export LIBRISPEECH_SAMPLE_FRACTION="${LIBRISPEECH_SAMPLE_FRACTION:-0.2}"
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
export FFT_FILTER_TYPE="${FFT_FILTER_TYPE:-lowpass}"
export FFT_CUTOFF_RATIO="${FFT_CUTOFF_RATIO:-0.7}"
export FFT_ALPHA="${FFT_ALPHA:-0.5}"
export FFT_ADAPTIVE_CUTOFF="${FFT_ADAPTIVE_CUTOFF:-false}"
export FFT_USE_SALIENCY="${FFT_USE_SALIENCY:-false}"

case "${MODEL_FAMILY}" in
  qwen2_5)
    CODE_ROOT="${ROOT_DIR}/qwen2_5/sparsekv_fft_dataset"
    MODEL_PATH="${MODEL_PATH:-/mnt/public/models/Qwen2.5-Omni-3B}"
    AUDIO_HEAD_JSON="${AUDIO_HEAD_JSON:-${ROOT_DIR}/audio_head/qwen2.5_omni/head_score/qwen2_5_3B/audio_head_scores_topk24_alpha1_beta1_gamma1.json}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_3B/ablation_vy3d_q3}"
    CONDA_ENV="Qwen3_2"
    ;;
  gemma)
    CODE_ROOT="${ROOT_DIR}/gemma/sparsekv_fft_dataset"
    MODEL_PATH="${MODEL_PATH:-/mnt/public/models/gemma-3n-E4B-it}"
    AUDIO_HEAD_JSON="${AUDIO_HEAD_JSON:-${ROOT_DIR}/audio_head/gemma/head_score/Gemma_3n_E4B/audio_head_scores_topk16_alpha1_beta1_gamma1.json}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E4B_it/ablation_vy3d_q3}"
    CONDA_ENV="Qwen3_2"
    ;;
  *)
    echo "Unknown MODEL_FAMILY=${MODEL_FAMILY} (use qwen2_5 or gemma)"
    exit 1
    ;;
esac
export MODEL_PATH
if [ "${MODEL_FAMILY}" = "gemma" ]; then
  export LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-${MODEL_PATH}}"
fi

RANDOM_JSON_DIR="${RANDOM_JSON_DIR:-$(dirname "${AUDIO_HEAD_JSON}")/ablation_random}"
mkdir -p "${RANDOM_JSON_DIR}"

echo "=== VY3D-Q3 head-importance ablation ==="
echo "Model: ${MODEL_FAMILY}  Task: ${TASK}  GPU: ${GPU_ID}  Budget: ${BUDGET}"
echo "Audio-aware JSON: ${AUDIO_HEAD_JSON}"
echo "Random JSON dir:  ${RANDOM_JSON_DIR}"

python3 "${SCRIPT_DIR}/generate_random_head_scores.py" \
  --reference "${AUDIO_HEAD_JSON}" \
  --output-dir "${RANDOM_JSON_DIR}" \
  --mode random \
  --seeds ${SEEDS}

if [ "${RUN_INVERSE}" = "true" ]; then
  python3 "${SCRIPT_DIR}/generate_random_head_scores.py" \
    --reference "${AUDIO_HEAD_JSON}" \
    --output-dir "${RANDOM_JSON_DIR}" \
    --mode inverse \
    --inverse-aggregation "${HEAD_SCORE_AGGREGATION}"
fi

if [ "${GENERATE_ONLY}" = "true" ]; then
  echo "GENERATE_ONLY=true — skipping inference."
  exit 0
fi

resolve_task_python() {
  EXTRA_PY_ARGS=()
  USE_RESULTS_DIR=false
  case "${TASK}" in
    librispeech)
      if [ "${MODEL_FAMILY}" = "qwen2_5" ]; then
        PY_SCRIPT="${CODE_ROOT}/code/qwen2.5_librispeech.py"
        RESULT_BASENAME="librispeech_results_qwen25.json"
      else
        PY_SCRIPT="${CODE_ROOT}/code/gemma_librispeech.py"
        RESULT_BASENAME="LibriSpeech_ASR_results_${LIBRISPEECH_SPLIT:-test-clean}.json"
        USE_RESULTS_DIR=true
      fi
      ;;
    kespeech)
      if [ "${MODEL_FAMILY}" != "qwen2_5" ]; then
        echo "TASK=kespeech is currently wired only for qwen2_5 in this rebuttal launcher."
        exit 1
      fi
      PY_SCRIPT="${CODE_ROOT}/code/qwen2.5_kespeech.py"
      RESULT_BASENAME="kespeech_results_qwen25.json"
      ;;
    multilingual-spanish|mls-spanish|spanish)
      if [ "${MODEL_FAMILY}" = "qwen2_5" ]; then
        PY_SCRIPT="${CODE_ROOT}/code/qwen2.5_multilingual_librispeech.py"
        RESULT_BASENAME="multilingual_librispeech_results_qwen25.json"
      else
        PY_SCRIPT="${CODE_ROOT}/code/gemma_multilingual_librispeech.py"
        RESULT_BASENAME="multilingual_librispeech_results_gemma.json"
      fi
      EXTRA_PY_ARGS=(--languages spanish)
      ;;
    multilingual-french|mls-french|french)
      if [ "${MODEL_FAMILY}" = "qwen2_5" ]; then
        PY_SCRIPT="${CODE_ROOT}/code/qwen2.5_multilingual_librispeech.py"
        RESULT_BASENAME="multilingual_librispeech_results_qwen25.json"
      else
        PY_SCRIPT="${CODE_ROOT}/code/gemma_multilingual_librispeech.py"
        RESULT_BASENAME="multilingual_librispeech_results_gemma.json"
      fi
      EXTRA_PY_ARGS=(--languages french)
      ;;
    multilingual-german|mls-german|german)
      if [ "${MODEL_FAMILY}" = "qwen2_5" ]; then
        PY_SCRIPT="${CODE_ROOT}/code/qwen2.5_multilingual_librispeech.py"
        RESULT_BASENAME="multilingual_librispeech_results_qwen25.json"
      else
        PY_SCRIPT="${CODE_ROOT}/code/gemma_multilingual_librispeech.py"
        RESULT_BASENAME="multilingual_librispeech_results_gemma.json"
      fi
      EXTRA_PY_ARGS=(--languages german)
      ;;
    multilingual-all|mls-all)
      if [ "${MODEL_FAMILY}" = "qwen2_5" ]; then
        PY_SCRIPT="${CODE_ROOT}/code/qwen2.5_multilingual_librispeech.py"
        RESULT_BASENAME="multilingual_librispeech_results_qwen25.json"
      else
        PY_SCRIPT="${CODE_ROOT}/code/gemma_multilingual_librispeech.py"
        RESULT_BASENAME="multilingual_librispeech_results_gemma.json"
      fi
      EXTRA_PY_ARGS=(--languages spanish french german)
      ;;
    covost2-zh-en|covost2_zh_en)
      COVOST2_DIRECTION="zh_en"
      if [ "${MODEL_FAMILY}" = "qwen2_5" ]; then
        PY_SCRIPT="${CODE_ROOT}/code/qwen2.5_covost2.py"
        RESULT_BASENAME="covost2_zh_en_results_qwen25.json"
        EXTRA_PY_ARGS=(--direction zh_en)
      else
        PY_SCRIPT="${CODE_ROOT}/code/gemma_covost2.py"
        export COVOST2_DIRECTION="zh_en"
        RESULT_BASENAME="CoVoST2_ZH_EN_results_gemma.json"
        USE_RESULTS_DIR=true
      fi
      ;;
    covost2-en-zh|covost2_en_zh)
      COVOST2_DIRECTION="en_zh"
      if [ "${MODEL_FAMILY}" = "qwen2_5" ]; then
        PY_SCRIPT="${CODE_ROOT}/code/qwen2.5_covost2.py"
        RESULT_BASENAME="covost2_en_zh_results_qwen25.json"
        EXTRA_PY_ARGS=(--direction en_zh)
      else
        PY_SCRIPT="${CODE_ROOT}/code/gemma_covost2.py"
        export COVOST2_DIRECTION="en_zh"
        RESULT_BASENAME="CoVoST2_EN_ZH_results_gemma.json"
        USE_RESULTS_DIR=true
      fi
      ;;
    llama-questions)
      echo "[warn] TASK=llama-questions is AQA; prefer librispeech / covost2 for VY3D-Q3."
      if [ "${MODEL_FAMILY}" = "qwen2_5" ]; then
        PY_SCRIPT="${CODE_ROOT}/code/qwen2.5_llama-questions.py"
        RESULT_BASENAME="llama_questions_results_qwen25.json"
      else
        PY_SCRIPT="${CODE_ROOT}/code/gemma_llama-questions.py"
        RESULT_BASENAME="llama_questions_results_gemma.json"
      fi
      ;;
    *)
      echo "Unsupported TASK=${TASK}"
      echo "  ASR: multilingual-spanish | multilingual-french | multilingual-german | multilingual-all | librispeech | kespeech"
      echo "  ST: covost2-zh-en | covost2-en-zh"
      echo "  AQA (manual only): llama-questions"
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

run_sparsekv_condition() {
  local label="$1"
  local head_path="$2"   # file path, or "uniform"
  local condition_output_dir="${OUTPUT_ROOT}/${TASK}/${label}/budget${BUDGET}"
  local timing_file="${condition_output_dir}/timing_stats_sparsekv_${label}_fft_on.json"
  local result_file="${condition_output_dir}/${RESULT_BASENAME}"

  if is_done "${result_file}"; then
    echo "[skip] ${label} — already completed: ${result_file}"
    return 0
  fi

  unset HEAD_SCORE_FILE
  unset HEAD_SCORE_PATH
  export OUTPUT_DIR="${condition_output_dir}"
  if [ "${head_path}" = "uniform" ]; then
    export HEAD_SCORE_FILE="uniform"
  else
    export HEAD_SCORE_PATH="${head_path}"
  fi

  mkdir -p "${condition_output_dir}"
  echo ""
  echo ">>> Running ${label}"
  echo "    HEAD_SCORE_PATH=${HEAD_SCORE_PATH:-<uniform>}"
  echo "    OUTPUT_DIR=${condition_output_dir}"

  set +u
  conda activate "${CONDA_ENV}"
  set -u
  cd "${CODE_ROOT}"

  unset SAMPLE_FRACTION
  if [ "${SAMPLE_LIMIT}" = "0" ]; then
    if [[ "${TASK}" == librispeech ]]; then
      export SAMPLE_FRACTION="${LIBRISPEECH_SAMPLE_FRACTION}"
      echo "    SAMPLE_FRACTION=${SAMPLE_FRACTION} (LibriSpeech subset)"
    elif [[ "${TASK}" == covost2* ]]; then
      export SAMPLE_FRACTION="${COVOST2_SAMPLE_FRACTION}"
      echo "    SAMPLE_FRACTION=${SAMPLE_FRACTION} (CoVoST2 subset)"
    elif [[ "${TASK}" == kespeech ]]; then
      export SAMPLE_FRACTION="${KESPEECH_SAMPLE_FRACTION}"
      echo "    SAMPLE_FRACTION=${SAMPLE_FRACTION} (KeSpeech subset)"
    elif [[ "${TASK}" == multilingual-* || "${TASK}" == mls-* || "${TASK}" == spanish || "${TASK}" == french || "${TASK}" == german ]]; then
      export SAMPLE_FRACTION="${MLS_SAMPLE_FRACTION}"
      echo "    SAMPLE_FRACTION=${SAMPLE_FRACTION} (Multilingual LibriSpeech subset)"
    fi
  fi

  if [ "${USE_RESULTS_DIR}" = "true" ]; then
    export RESULTS_DIR="${condition_output_dir}"
    python "${PY_SCRIPT}" "${EXTRA_PY_ARGS[@]}"
  elif [ "${TASK}" = "llama-questions" ]; then
    python "${PY_SCRIPT}" \
      --mode inference \
      --model-path "${MODEL_PATH}" \
      --output-dir "${condition_output_dir}" \
      --timing-output-file "${timing_file}"
  else
    python "${PY_SCRIPT}" \
      "${EXTRA_PY_ARGS[@]}" \
      --model-path "${MODEL_PATH}" \
      --output-dir "${condition_output_dir}" \
      --timing-output-file "${timing_file}"
  fi

  echo "    Done: ${result_file}"
}

resolve_task_python

# (A) Audio-aware supervision + SSS
run_sparsekv_condition "AudioAware_SSS" "${AUDIO_HEAD_JSON}"

# (B) Inverse audio-aware importance + SSS (same settings otherwise)
ref_stem="$(basename "${AUDIO_HEAD_JSON}" .json)"
if [ "${RUN_INVERSE}" = "true" ]; then
  inverse_json="${RANDOM_JSON_DIR}/${ref_stem}_inverse.json"
  if [ ! -f "${inverse_json}" ]; then
    echo "Missing ${inverse_json}" >&2
    exit 1
  fi
  run_sparsekv_condition "InverseAudioAware_SSS" "${inverse_json}"
fi

# (C) Random-init head importance + SSS (same settings otherwise)
for seed in ${SEEDS}; do
  random_json="${RANDOM_JSON_DIR}/${ref_stem}_random_seed${seed}.json"
  if [ ! -f "${random_json}" ]; then
    echo "Missing ${random_json}" >&2
    exit 1
  fi
  run_sparsekv_condition "RandomInit_SSS_seed${seed}" "${random_json}"
done

# (C) Uniform allocation + SSS (optional control)
if [ "${RUN_UNIFORM}" = "true" ]; then
  run_sparsekv_condition "Uniform_SSS" "uniform"
fi

echo ""
echo "=== Ablation inference finished ==="
echo "Results under: ${OUTPUT_ROOT}"
echo "Next: run task-specific judge scripts on each output subfolder."
