#!/bin/bash
# speech-web-questions Judge 打分脚本

set -e
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"  
export JUDGE_API_MODEL="${JUDGE_API_MODEL:-gpt-4o-mini}" 
export JUDGE_API_KEY="${JUDGE_API_KEY:-sk-BQv7PBMJZ9Iaff5EpguCrUNHTuP6QobfUma3cKguHeUH1GLc}"  # API Key
export JUDGE_API_BASE_URL="${JUDGE_API_BASE_URL:-https://yunwu.ai/v1}"  
export OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/users/wangyuxuan-20250915/Audio/Results/Qwen3-Omni-30B-A3B-Instruct/datasets/speech_web_questions_Results}"
export BUDGET="${BUDGET:-0.4}"
METHOD_MODE="${1:-${METHOD_MODE:-remaining}}"
METHOD_MODE="$(printf '%s' "${METHOD_MODE}" | tr '[:upper:]' '[:lower:]')"
export FORCE_REJUDGE="${FORCE_REJUDGE:-false}"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
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
    local head_score_tag
    local output_label
    local output_dir
    local inference_output_file
    local judge_output_file

    fft_state="$(printf '%s' "${fft_state}" | tr '[:upper:]' '[:lower:]')"
    sparsekv_head_score_mode="$(printf '%s' "${sparsekv_head_score_mode}" | tr '[:upper:]' '[:lower:]')"
    if [ "${fft_state}" = "true" ]; then
        fft_tag="FFTOn"
    else
        fft_tag="FFTOff"
    fi

    if [ "${method_name}" = "sparsekv" ]; then
        case "${sparsekv_head_score_mode}" in
            uniform)
                output_label="${method_display}_Uniform_${fft_tag}"
                head_score_tag="uniform"
                ;;
            head_path)
                output_label="${method_display}_HeadPath_${fft_tag}"
                head_score_tag="head_path"
                ;;
            *)
                output_label="${method_display}_HeadPath_${fft_tag}"
                head_score_tag="head_path"
                ;;
        esac
    else
        output_label="${method_display}"
        head_score_tag="default"
    fi

    output_dir="${OUTPUT_ROOT}/${output_label}/budget${BUDGET}"
    inference_output_file="${output_dir}/speech_web_questions_results_qwen3.json"
    judge_output_file="${JUDGE_OUTPUT_FILE:-${output_dir}/speech_web_questions_results_qwen3_judge.json}"

    echo -e "${GREEN}=== 启动 speech-web-questions Judge 打分 ===${NC}"
    echo "Config: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "Baseline: ${output_label} (FFT=${fft_state}, HEAD_SCORE_TAG=${head_score_tag})"

    # 检查输入文件是否存在
    if [ ! -f "${inference_output_file}" ]; then
        echo -e "${RED}Error: 推理结果文件不存在: ${inference_output_file}${NC}"
        echo -e "${RED}请先运行推理脚本: bash run_speech-web-questions_inference.sh${NC}"
        exit 1
    fi

    echo "Input file: ${inference_output_file}"
    echo "Output file: ${judge_output_file}"

    if [ "${FORCE_REJUDGE}" != "true" ] && is_completed_json_file "${judge_output_file}"; then
        echo -e "${BLUE}Skip completed judge output: ${judge_output_file}${NC}"
        return 0
    fi

    # 获取脚本所在目录
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CODE_DIR="$(dirname "$SCRIPT_DIR")/code"

    # 判断使用哪种 judge 模型
    JUDGE_ARGS=""
    if [ -n "${JUDGE_MODEL_PATH}" ]; then
        JUDGE_ARGS="--judge-model-path ${JUDGE_MODEL_PATH}"
        echo -e "${BLUE}Using local judge model: ${JUDGE_MODEL_PATH}${NC}"
    elif [ -n "${JUDGE_API_MODEL}" ]; then
        JUDGE_ARGS="--judge-api-model ${JUDGE_API_MODEL}"
        echo -e "${BLUE}Using API judge model: ${JUDGE_API_MODEL}${NC}"

        # 添加 API Key 和 Base URL 参数
        if [ -n "${JUDGE_API_KEY}" ]; then
            JUDGE_ARGS="${JUDGE_ARGS} --judge-api-key \"${JUDGE_API_KEY}\""
            echo -e "${BLUE}API Key: ${JUDGE_API_KEY:0:10}...${NC}"
        fi
        if [ -n "${JUDGE_API_BASE_URL}" ]; then
            JUDGE_ARGS="${JUDGE_ARGS} --judge-api-base-url \"${JUDGE_API_BASE_URL}\""
            echo -e "${BLUE}API Base URL: ${JUDGE_API_BASE_URL}${NC}"
        fi
    else
        echo -e "${RED}Error: 必须指定 JUDGE_MODEL_PATH 或 JUDGE_API_MODEL${NC}"
        echo -e "${RED}例如: export JUDGE_API_MODEL=\"gemini-2.5-flash\"${NC}"
        exit 1
    fi

    # 添加强制重新打分参数
    if [ "${FORCE_REJUDGE}" = "true" ]; then
        JUDGE_ARGS="${JUDGE_ARGS} --force-rejudge"
        echo -e "${BLUE}Force rejudge enabled: will re-process all samples${NC}"
    fi

    # 打分（使用 LLM as Judge）
    echo -e "${GREEN}>>> 裁判打分...${NC}"
    conda activate Qwen3_2 && \
    cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/sparsekv_fft_dataset && \
    eval python code/qwen3_speech-web-questions.py \
        --mode judge \
        --input-file "${inference_output_file}" \
        --output-file "${judge_output_file}" \
        ${JUDGE_ARGS}

    echo -e "${GREEN}=== Judge 打分完成 ===${NC}"
    echo -e "${BLUE}打分结果保存在: ${judge_output_file}${NC}"
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
        ;;
    remaining)
        run_sparsekv_fft_sweep
        echo
        run_one_method "snapkv" "SnapKV"
        echo
        run_one_method "adakv" "AdaKV"
        echo
        run_one_method "pyramidkv" "PyramidKV"
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

