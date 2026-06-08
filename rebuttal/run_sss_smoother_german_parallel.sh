#!/usr/bin/env bash
# jV7P-Q2: Parallel German smoother ablation — one method per GPU.
#
# Writes to a separate OUTPUT_ROOT so ongoing French/Spanish results are not touched.
#
# Usage:
#   bash rebuttal/run_sss_smoother_german_parallel.sh
#   GPUS="0 1 6" bash rebuttal/run_sss_smoother_german_parallel.sh   # 3-way parallel + next wave
#   GPUS="0 1 6 2 4 7" bash rebuttal/run_sss_smoother_german_parallel.sh  # 6-way if all free
#   METHODS="none sss" GPUS="0 1" bash rebuttal/run_sss_smoother_german_parallel.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_SH="${SCRIPT_DIR}/run_sss_smoother_ablation.sh"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/german_smoother_parallel}"
mkdir -p "${LOG_DIR}"

export BUDGET="${BUDGET:-0.4}"
export MLS_SAMPLE_FRACTION="${MLS_SAMPLE_FRACTION:-0.2}"
export SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
export FORCE_RERUN="${FORCE_RERUN:-false}"
export CONDA_ENV="${CONDA_ENV:-Qwen3_2}"

# Separate root — do NOT write into the main ablation tree used by tmux French job.
export OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_3B/ablation_jv7p_q2_smoothers_german_parallel}"

# Use ALL_METHODS (array) in this launcher; pass a single scalar METHODS to each child.
ALL_METHODS=(${METHODS:-none sss maxpool gaussian uniform_stride uniform_prior})
read -r -a GPUS <<< "${GPUS:-0 1 6}"

if [ "${#GPUS[@]}" -lt 1 ]; then
  echo "Need at least 1 GPU in GPUS, got: ${GPUS[*]:-<empty>}" >&2
  exit 1
fi

# Optional backup of stale partial in the old tree (best-effort; skip if not writable).
OLD_ROOT="/mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_3B/ablation_jv7p_q2_smoothers"
STALE="${OLD_ROOT}/multilingual-german/Smooth_none/budget${BUDGET}/multilingual_librispeech_results_qwen25.json"
if [ -f "${STALE}" ]; then
  BACKUP_DIR="${LOG_DIR}/stale_german_none_backup_$(date +%Y%m%d_%H%M%S)"
  if cp -a "${STALE}" "${BACKUP_DIR}.json" 2>/dev/null; then
    echo "Backed up stale German none JSON -> ${BACKUP_DIR}.json"
  else
    echo "[warn] Could not backup stale JSON (continuing; new results go to ${OUTPUT_ROOT})"
  fi
fi

echo "=== Parallel German smoother ablation ==="
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "Methods: ${ALL_METHODS[*]}"
echo "GPUs:    ${GPUS[*]} (${#GPUS[@]} per wave)"
echo "Logs:    ${LOG_DIR}/"
echo ""

launch_method() {
  local method="$1"
  local gpu="$2"
  local log_file="${LOG_DIR}/german_${method}_gpu${gpu}.log"

  echo "Launch method=${method} on GPU ${gpu} -> ${log_file}"
  (
    export GPU_ID="${gpu}"
    export TASKS="multilingual-german"
    export OUTPUT_ROOT
    # Bash arrays are not exported; unset inherited array before scalar export.
    unset METHODS
    export METHODS="${method}"
    bash "${ABLATION_SH}"
  ) > "${log_file}" 2>&1 &
}

FAIL=0
NUM_GPUS="${#GPUS[@]}"
for ((start=0; start<${#ALL_METHODS[@]}; start+=NUM_GPUS)); do
  end=$((start + NUM_GPUS))
  if [ "${end}" -gt "${#ALL_METHODS[@]}" ]; then
    end="${#ALL_METHODS[@]}"
  fi
  batch_methods=("${ALL_METHODS[@]:start:end-start}")
  echo "--- Wave $((start / NUM_GPUS + 1)): ${batch_methods[*]} ---"

  PIDS=()
  for idx in "${!batch_methods[@]}"; do
    method="${batch_methods[$idx]}"
    gpu="${GPUS[$idx]}"
    launch_method "${method}" "${gpu}"
    PIDS+=("$!")
  done

  echo "Waiting for wave ($((start / NUM_GPUS + 1))) — ${#PIDS[@]} jobs..."
  for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
      FAIL=1
    fi
  done
  echo ""
done
if [ "${FAIL}" -eq 0 ]; then
  echo "All German parallel jobs finished."
else
  echo "One or more jobs failed — check ${LOG_DIR}/" >&2
  exit 1
fi

echo "Aggregate:"
echo "  python3 ${SCRIPT_DIR}/aggregate_smoother_ablation.py --root ${OUTPUT_ROOT} --task multilingual-german"
