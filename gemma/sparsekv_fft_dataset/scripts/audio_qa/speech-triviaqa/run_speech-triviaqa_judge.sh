#!/bin/bash
# speech-triviaqa Judge 打分脚本

set -e
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export JUDGE_API_MODEL="${JUDGE_API_MODEL:-gemini-2.5-flash}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-${OPENAI_API_KEY:-sk-BQv7PBMJZ9Iaff5EpguCrUNHTuP6QobfUma3cKguHeUH1GLc}}"  # 建议通过环境变量注入
export OPENAI_API_KEY="${OPENAI_API_KEY:-${JUDGE_API_KEY}}"
export JUDGE_API_BASE_URL="${JUDGE_API_BASE_URL:-https://yunwu.ai/v1}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E4B_it/datasets/speech_triviaqa_Results}"
export BUDGET="${BUDGET:-0.4}"
export SPARSEKV_HEAD_SCORE_PATH="${SPARSEKV_HEAD_SCORE_PATH:-/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/gemma/head_score/audio_head_scores_aggregated.json}"

# KV Cache compression method: sparsekv, snapkv, adakv, pyramidkv
METHOD_MODE="${1:-${METHOD_MODE:-all}}"
METHOD_MODE="$(printf '%s' "${METHOD_MODE}" | tr '[:upper:]' '[:lower:]')"

export JUDGE_OUTPUT_FILE="${JUDGE_OUTPUT_FILE:-}"
export FORCE_REJUDGE="${FORCE_REJUDGE:-true}"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== 启动 speech-triviaqa Judge 打分 ===${NC}"
echo "Config: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh

JUDGE_ARGS=""
if [ -n "${JUDGE_MODEL_PATH}" ]; then
    JUDGE_ARGS="--judge-model-path ${JUDGE_MODEL_PATH}"
    echo -e "${BLUE}Using local judge model: ${JUDGE_MODEL_PATH}${NC}"
elif [ -n "${JUDGE_API_MODEL}" ]; then
    JUDGE_ARGS="--judge-api-model ${JUDGE_API_MODEL}"
    echo -e "${BLUE}Using API judge model: ${JUDGE_API_MODEL}${NC}"
    echo -e "${BLUE}API Provider: yunwu.ai (OpenAI-compatible)${NC}"

    if [ -n "${JUDGE_API_KEY}" ]; then
        JUDGE_ARGS="${JUDGE_ARGS} --judge-api-key \"${JUDGE_API_KEY}\""
        echo -e "${BLUE}API Key: ${JUDGE_API_KEY:0:10}...${NC}"
    fi
    if [ -n "${JUDGE_API_BASE_URL}" ]; then
        JUDGE_ARGS="${JUDGE_ARGS} --judge-api-base-url \"${JUDGE_API_BASE_URL}\""
        echo -e "${BLUE}API Base URL: ${JUDGE_API_BASE_URL}${NC}"
    fi
    if [ -z "${JUDGE_API_KEY}" ] && [ -z "${OPENAI_API_KEY}" ]; then
        echo -e "${RED}Error: 未检测到 API Key。请先设置 JUDGE_API_KEY 或 OPENAI_API_KEY${NC}"
        exit 1
    fi
else
    echo -e "${RED}Error: 必须指定 JUDGE_MODEL_PATH 或 JUDGE_API_MODEL${NC}"
    echo -e "${RED}例如: export JUDGE_API_MODEL=\"gemini-2.5-flash\"${NC}"
    exit 1
fi

if [ "${FORCE_REJUDGE}" = "true" ]; then
    JUDGE_ARGS="${JUDGE_ARGS} --force-rejudge"
    echo -e "${BLUE}Force rejudge enabled: will re-process all samples${NC}"
fi

run_one_method() {
    local method_name="$1"
    local method_display="$2"
    local fft_state="${3:-true}"
    local sparsekv_head_score_mode="${4:-none}"
    local fft_tag
    local output_label
    local output_dir
    local inference_output_file
    local judge_output_file
    local score_tag

    fft_state="$(printf '%s' "${fft_state}" | tr '[:upper:]' '[:lower:]')"
    if [ "${fft_state}" = "true" ]; then
        fft_tag="FFTOn"
    else
        fft_tag="FFTOff"
    fi

    if [ "${method_name}" = "sparsekv" ]; then
        case "${sparsekv_head_score_mode}" in
            uniform)
                output_label="${method_display}_Uniform_${fft_tag}"
                score_tag="uniform"
                ;;
            head_path)
                output_label="${method_display}_HeadPath_${fft_tag}"
                score_tag="head_path"
                ;;
            *)
                output_label="${method_display}_${fft_tag}"
                score_tag="default"
                ;;
        esac
    else
        output_label="${method_display}"
        score_tag="default"
    fi

    output_dir="${OUTPUT_ROOT}/${output_label}/budget${BUDGET}"
    inference_output_file="${output_dir}/speech_triviaqa_results_gemma.json"
    judge_output_file="${JUDGE_OUTPUT_FILE:-${output_dir}/speech_triviaqa_results_gemma_judge.json}"

    if [ ! -f "${inference_output_file}" ]; then
        echo -e "${RED}Error: 推理结果文件不存在: ${inference_output_file}${NC}"
        echo -e "${RED}请先运行第一阶段推理脚本，或检查 OUTPUT_ROOT/BUDGET/METHOD_MODE 是否一致${NC}"
        exit 1
    fi

    echo "Input file: ${inference_output_file}"
    echo "Output file: ${judge_output_file}"
    echo "Case: ${output_label} (${method_name}, ${score_tag}, FFT=${fft_state})"

    echo -e "${GREEN}>>> 裁判打分...${NC}"
    conda activate Qwen3_2 && \
    cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft_dataset && \
    eval python code/gemma_speech-triviaqa.py \
        --mode judge \
        --input-file "${inference_output_file}" \
        --output-file "${judge_output_file}" \
        ${JUDGE_ARGS}

    echo -e "${GREEN}=== Judge 打分完成 ===${NC}"
    echo -e "${BLUE}打分结果保存在: ${judge_output_file}${NC}"
}

case "${METHOD_MODE}" in
    all)
        run_one_method "sparsekv" "SparseKV" "false" "head_path"
        echo
        run_one_method "sparsekv" "SparseKV" "true" "head_path"
        echo
        run_one_method "sparsekv" "SparseKV" "true" "uniform"
        echo
        run_one_method "snapkv" "SnapKV"
        echo
        run_one_method "adakv" "AdaKV"
        echo
        run_one_method "pyramidkv" "PyramidKV"
        ;;
    sparsekv)
        run_one_method "sparsekv" "SparseKV" "false" "head_path"
        echo
        run_one_method "sparsekv" "SparseKV" "true" "head_path"
        echo
        run_one_method "sparsekv" "SparseKV" "true" "uniform"
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
        echo -e "${RED}Usage: bash $0 [all|sparsekv|snapkv|adakv|pyramidkv]${NC}"
        exit 1
        ;;
esac

