#!/bin/bash
# speech-web-questions 推理脚本

set -e
export GPU_ID="${GPU_ID:-4}"
export MODEL_PATH="${MODEL_PATH:-/mnt/public/models/gemma-3n-E4B-it}"
export SAMPLE_LIMIT=512
export THINKER_MAX_NEW_TOKENS="${THINKER_MAX_NEW_TOKENS:-64}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E4B_it/datasets/speech_web_questions_Results}"
export BUDGET="${BUDGET:-0.4}"
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export SPARSEKV_HEAD_SCORE_PATH="${SPARSEKV_HEAD_SCORE_PATH:-/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/gemma/head_score/Gemma_3n_E4B/audio_head_scores_topk16_alpha1_beta1_gamma1.json}"

export USE_FFT_FILTER="true"
export FFT_FILTER_TYPE="lowpass"
export FFT_CUTOFF_RATIO="0.7"
export FFT_ALPHA="0.5"
export FFT_ADAPTIVE_CUTOFF="false"
export FFT_USE_SALIENCY="false"

# KV Cache compression method: sparsekv, snapkv, adakv, pyramidkv, mixkv, vanilla
METHOD_MODE="${1:-${METHOD_MODE:-remaining}}"
METHOD_MODE="$(printf '%s' "${METHOD_MODE}" | tr '[:upper:]' '[:lower:]')"
export FORCE_RERUN="${FORCE_RERUN:-true}"

export CUDA_VISIBLE_DEVICES=$GPU_ID
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh

is_completed_json_file() {
    local json_file="$1"
    [ -f "${json_file}" ] || return 1
    python - "${json_file}" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and data.get("status") == "completed":
        sys.exit(0)
except Exception:
    pass
sys.exit(1)
PY
}

run_one_method() {
    local method_name="$1"
    local method_display="$2"
    local fft_state="${3:-true}"
    local sparsekv_head_score_mode="${4:-none}"
    local fft_tag
    local fft_log_tag
    local head_score_tag
    local output_label
    local output_dir
    local timing_output_file
    local inference_output_file

    fft_state="$(printf '%s' "${fft_state}" | tr '[:upper:]' '[:lower:]')"
    sparsekv_head_score_mode="$(printf '%s' "${sparsekv_head_score_mode}" | tr '[:upper:]' '[:lower:]')"
    if [ "${fft_state}" = "true" ]; then
        fft_tag="FFTOn"
        fft_log_tag="fft_on"
    else
        fft_tag="FFTOff"
        fft_log_tag="fft_off"
    fi

    unset HEAD_SCORE_FILE
    unset HEAD_SCORE_PATH

    export KV_METHOD="${method_name}"
    export USE_FFT_FILTER="${fft_state}"

    if [ "${method_name}" = "sparsekv" ]; then
        case "${sparsekv_head_score_mode}" in
            uniform)
                export HEAD_SCORE_FILE="uniform"
                output_label="${method_display}_Uniform_${fft_tag}"
                head_score_tag="uniform"
                ;;
            head_path)
                export HEAD_SCORE_PATH="${SPARSEKV_HEAD_SCORE_PATH}"
                output_label="${method_display}_HeadPath_${fft_tag}"
                head_score_tag="head_path"
                ;;
            *)
                export HEAD_SCORE_PATH="${SPARSEKV_HEAD_SCORE_PATH}"
                output_label="${method_display}_HeadPath_${fft_tag}"
                head_score_tag="head_path"
                ;;
        esac
    else
        output_label="${method_display}"
        head_score_tag="default"
    fi

    output_dir="${OUTPUT_ROOT}/${output_label}/budget${BUDGET}"
    timing_output_file="${output_dir}/timing_stats_${method_name}_${head_score_tag}_${fft_log_tag}.json"
    inference_output_file="${output_dir}/speech_web_questions_results_gemma.json"
    export OUTPUT_DIR="${output_dir}"

    if [ "${FORCE_RERUN}" != "true" ] && is_completed_json_file "${inference_output_file}"; then
        echo -e "${BLUE}Skip completed inference: ${inference_output_file}${NC}"
        return 0
    fi

    echo -e "${GREEN}=== 启动 speech-web-questions 推理 ===${NC}"
    echo "Baseline: ${output_label} (KV_METHOD=${KV_METHOD}, USE_FFT_FILTER=${USE_FFT_FILTER}, HEAD_SCORE_TAG=${head_score_tag})"
    echo "Config: GPU=${GPU_ID}, Model=${MODEL_PATH}"
    echo "Output Dir: ${OUTPUT_DIR}"

    echo -e "${GREEN}>>> 推理生成答案...${NC}"
    conda activate Qwen3_2 && \
    cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft_dataset && \
    python code/gemma_speech-web-questions.py \
        --mode inference \
        --model-path "${MODEL_PATH}" \
        --output-dir "${OUTPUT_DIR}" \
        --timing-output-file "${timing_output_file}"

    INFERENCE_OUTPUT_FILE="${OUTPUT_DIR}/speech_web_questions_results_gemma.json"

    echo -e "${GREEN}=== 推理完成 ===${NC}"
    echo -e "${BLUE}Baseline: ${output_label} (${KV_METHOD}, FFT=${USE_FFT_FILTER})${NC}"
    echo -e "${BLUE}推理结果保存在: ${INFERENCE_OUTPUT_FILE}${NC}"
}

run_sparsekv_fft_sweep() {
    run_one_method "sparsekv" "SparseKV" "false" "head_path"
    echo
    run_one_method "sparsekv" "SparseKV" "true" "head_path"
    echo
    run_one_method "sparsekv" "SparseKV" "true" "uniform"
}

case "${METHOD_MODE}" in
    all)
        run_sparsekv_fft_sweep
        echo
        run_one_method "snapkv" "SnapKV"
        echo
        run_one_method "adakv" "AdaKV"
        echo
        run_one_method "pyramidkv" "PyramidKV"
        echo -e "${BLUE}四个 baseline 已全部跑完。${NC}"
        ;;
    remaining)
        run_sparsekv_fft_sweep
        echo
        run_one_method "snapkv" "SnapKV"
        echo
        run_one_method "adakv" "AdaKV"
        echo
        run_one_method "pyramidkv" "PyramidKV"
        echo -e "${BLUE}四个 baseline 已全部跑完。${NC}"
        ;;
    sparsekv)
        run_sparsekv_fft_sweep
        ;;
    snapkv)
        run_one_method "snapkv" "SnapKV"
        ;;
    adakv)
        run_one_method "adakv" "AdaKV"
        ;;
    pyramidkv)
        run_one_method "pyramidkv" "PyramidKV"
        ;;
    *)
        echo "Usage: bash $0 [remaining|all|sparsekv|snapkv|adakv|pyramidkv]"
        exit 1
        ;;
esac

echo -e "${BLUE}可以使用以下命令进行打分:${NC}"
echo -e "${BLUE}  bash run_speech-web-questions_judge.sh${NC}"

