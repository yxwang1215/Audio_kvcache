#!/bin/bash
# Compare real end-to-end runtime with SSS disabled/enabled.
#
# Outputs:
#   sss_timing_e2e/sss_e2e_timing_delta.json  (end-to-end SSS on/off delta)
#   sss_timing_e2e/timing_without_sss.json     (FFT off)
#   sss_timing_e2e/timing_with_sss.json        (FFT on)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash run_sss_timing_comparison.sh
#   SAMPLE_LIMIT=3 E2E_BACKEND=librispeech bash run_sss_timing_comparison.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export MODEL_PATH="${MODEL_PATH:-/mnt/public/models/Qwen2.5-Omni-3B}"
export SAMPLE_LIMIT="${SAMPLE_LIMIT:-10}"
export BUDGET="${BUDGET:-0.4}"
export CONDA_ENV="${CONDA_ENV:-Qwen3_2}"
export E2E_BACKEND="${E2E_BACKEND:-librispeech}"
export LIBRISPEECH_LONG_DIR="${LIBRISPEECH_LONG_DIR:-/mnt/public/datasets}"

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/sss_timing_e2e}"
mkdir -p "${OUT_DIR}"

echo "========== SSS end-to-end timing delta =========="
echo "GPU=${CUDA_VISIBLE_DEVICES}  samples=${SAMPLE_LIMIT}  budget=${BUDGET}  backend=${E2E_BACKEND}"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
set +u
conda activate "${CONDA_ENV}"
set -u

python3 "${SCRIPT_DIR}/benchmark_sss_overhead.py" \
  --mode e2e \
  --sample-limit "${SAMPLE_LIMIT}" \
  --gpu "${CUDA_VISIBLE_DEVICES}" \
  --model-path "${MODEL_PATH}" \
  --budget "${BUDGET}" \
  --e2e-backend "${E2E_BACKEND}" \
  --e2e-output-dir "${OUT_DIR}" \
  --conda-env "${CONDA_ENV}" \
  --output-json "${OUT_DIR}/sss_e2e_timing_delta.json"

echo "Done. See end-to-end timing delta in ${OUT_DIR}/sss_e2e_timing_delta.json"
