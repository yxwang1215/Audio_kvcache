#!/bin/bash
# ASR Budget 参数搜索总脚本
# 对 ASR 目录下的所有脚本进行 budget 参数搜索
#
# 使用方法:
#   ./run_budget_search.sh                    # 使用默认配置
#   ./run_budget_search.sh --budgets "0.1 0.2 0.3"  # 指定 budget 值
#   ./run_budget_search.sh --scripts "run_kespeech.sh"  # 指定脚本
#   ./run_budget_search.sh --model-name "Qwen2_5_Omni_3B"  # 指定模型名称
#   ./run_budget_search.sh --help             # 显示帮助信息
#
# 功能:
#   - 自动发现 ASR 目录下的所有 run_*.sh 脚本
#   - 对每个脚本使用不同的 budget 值运行
#   - 自动修改输出路径，添加 budget 和模型名称信息
#   - 记录详细的运行日志
#   - 支持自定义 budget 值、脚本列表和模型名称
#
# 输出路径格式:
#   原路径: /mnt/users/.../Results/Qwen2_5_Omni_3B/datasets/KeSpeech_Results
#   新路径: /mnt/users/.../Results/{MODEL_NAME}/datasets/{DATASET_PATH}/budget_{BUDGET}
#
# 注意:
#   - 数据集路径（datasets/ 之后的部分）会自动从原脚本中提取
#   - 如果指定了 --model-name，会替换路径中的模型名称
#   - Budget 值会转换为文件名格式（0.5 -> 0_5）并添加到路径末尾

# ============================================
# 配置参数
# ============================================

# Budget 参数搜索范围（可以是比例或绝对数值）
# 例如: 0.1 表示 10%, 0.2 表示 20%, 等等
BUDGET_VALUES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# 或者使用绝对数值
# BUDGET_VALUES=(512 1024 2048 4096 8192)

# 要搜索的脚本列表（如果为空，则自动搜索所有 run_*.sh 脚本）
# 可以手动指定，例如: SCRIPTS=("run_kespeech.sh" "run_librispeech.sh")
SCRIPTS=()

# 是否在每次运行前显示确认（设置为 true 会在每次运行前暂停）
CONFIRM_EACH_RUN=false

# 要跳过的脚本列表（例如: SKIP_SCRIPTS=("run_test.sh")）
SKIP_SCRIPTS=()

# 模型名称（用于修改输出路径，例如: "Qwen2_5_Omni_3B" 或 "Qwen2_5_Omni_7B"）
# 如果为空，将从脚本中的 MODEL_PATH 自动提取，或使用原路径中的模型名称
MODEL_NAME=""

# ============================================
# 命令行参数解析
# ============================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --budgets)
                IFS=' ' read -ra BUDGET_VALUES <<< "$2"
                shift 2
                ;;
            --scripts)
                IFS=' ' read -ra SCRIPTS <<< "$2"
                shift 2
                ;;
            --skip)
                IFS=' ' read -ra SKIP_SCRIPTS <<< "$2"
                shift 2
                ;;
            --confirm)
                CONFIRM_EACH_RUN=true
                shift
                ;;
            --model-name)
                MODEL_NAME="$2"
                shift 2
                ;;
            --help|-h)
                echo "ASR Budget 参数搜索脚本"
                echo ""
                echo "用法: $0 [选项]"
                echo ""
                echo "选项:"
                echo "  --budgets \"值1 值2 ...\"  指定要搜索的 budget 值"
                echo "  --scripts \"脚本1 脚本2\"  指定要运行的脚本（默认：所有 run_*.sh）"
                echo "  --skip \"脚本1 脚本2\"     指定要跳过的脚本"
                echo "  --model-name \"名称\"      指定模型名称（用于输出路径，例如: Qwen2_5_Omni_3B）"
                echo "  --confirm                 每次运行前确认"
                echo "  --help, -h                显示此帮助信息"
                echo ""
                echo "示例:"
                echo "  $0 --budgets \"0.1 0.2 0.3\""
                echo "  $0 --scripts \"run_kespeech.sh\" --budgets \"0.5 0.7\" --model-name \"Qwen2_5_Omni_3B\""
                exit 0
                ;;
            *)
                echo "未知参数: $1"
                echo "使用 --help 查看帮助信息"
                exit 1
                ;;
        esac
    done
}

# ============================================
# 初始化（在解析参数后执行）
# ============================================

init_scripts() {
    # 如果 SCRIPTS 为空，自动查找所有 run_*.sh 脚本（排除当前脚本和跳过的脚本）
    if [ ${#SCRIPTS[@]} -eq 0 ]; then
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        for script in "$SCRIPT_DIR"/run_*.sh; do
            script_name=$(basename "$script")
            # 排除当前脚本
            if [ "$script_name" = "run_budget_search.sh" ]; then
                continue
            fi
            # 检查是否在跳过列表中
            skip=false
            for skip_script in "${SKIP_SCRIPTS[@]}"; do
                if [ "$script_name" = "$skip_script" ]; then
                    skip=true
                    break
                fi
            done
            if [ "$skip" = false ]; then
                SCRIPTS+=("$script_name")
            fi
        done
    fi
}

# ============================================
# 辅助函数
# ============================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 从路径中提取模型名称
extract_model_name_from_path() {
    local path="$1"
    # 尝试从路径中提取模型名称，例如: /mnt/users/.../Results/Qwen2_5_Omni_3B/...
    if [[ "$path" =~ /Results/([^/]+)/ ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo ""
    fi
}

# 从路径中提取数据集部分（datasets/ 之后的内容）
extract_dataset_path() {
    local path="$1"
    # 提取 datasets/ 之后的所有内容
    if [[ "$path" =~ /datasets/(.+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo ""
    fi
}

# 修改输出路径，添加 budget 和模型名称信息，保留数据集路径
modify_output_path() {
    local original_path="$1"
    local budget="$2"
    local model_name="$3"
    
    # 将 budget 值转换为文件名友好的格式（将 . 替换为 _）
    local budget_str=$(echo "$budget" | sed 's/\./_/g')
    
    # 提取数据集路径（datasets/ 之后的部分）
    local dataset_path=$(extract_dataset_path "$original_path")
    
    # 提取基础路径（到 Results/ 之前的部分）
    local base_path=""
    if [[ "$original_path" =~ ^(.+)/Results/ ]]; then
        base_path="${BASH_REMATCH[1]}"
    else
        # 如果没有找到 Results/，尝试提取到 datasets/ 之前的部分
        if [[ "$original_path" =~ ^(.+)/datasets/ ]]; then
            base_path="${BASH_REMATCH[1]}"
            # 从 base_path 中提取模型名称部分
            if [[ "$base_path" =~ /Results/([^/]+)$ ]]; then
                base_path=$(echo "$base_path" | sed "s|/Results/${BASH_REMATCH[1]}$||")
            fi
        else
            base_path=$(dirname "$original_path")
        fi
    fi
    
    # 确定模型名称
    local final_model_name="$model_name"
    if [ -z "$final_model_name" ]; then
        # 如果没有指定模型名称，尝试从原路径中提取
        if [[ "$original_path" =~ /Results/([^/]+)/ ]]; then
            final_model_name="${BASH_REMATCH[1]}"
        else
            final_model_name="Qwen2_5_Omni_3B"  # 默认值
        fi
    fi
    
    # 构建新路径
    # 如果路径以 / 结尾，去掉它
    dataset_path=$(echo "$dataset_path" | sed 's|/$||')
    echo "${base_path}/Results/${final_model_name}/datasets/${dataset_path}/budget_${budget_str}"
}

run_script_with_budget() {
    local script_path="$1"
    local budget="$2"
    local script_name=$(basename "$script_path")
    
    log "============================================"
    log "运行脚本: $script_name, Budget: $budget"
    if [ -n "$MODEL_NAME" ]; then
        log "模型名称: $MODEL_NAME"
    fi
    log "============================================"
    
    # 如果设置了确认，则暂停
    if [ "$CONFIRM_EACH_RUN" = true ]; then
        read -p "按 Enter 继续运行，或 Ctrl+C 取消..."
    fi
    
    # 创建临时脚本，修改原脚本中的 BUDGET 值和输出路径
    local temp_script=$(mktemp)
    
    # 将 budget 值转换为文件名友好的格式（将 . 替换为 _）
    local budget_str=$(echo "$budget" | sed 's/\./_/g')
    
    # 使用 awk 处理脚本内容，逐行修改
    awk -v budget="$budget" \
        -v budget_str="$budget_str" \
        -v model_name="$MODEL_NAME" \
    '
    function extract_dataset_path(path) {
        # 提取 datasets/ 之后的所有内容
        if (match(path, /\/datasets\/(.+)$/, arr)) {
            return arr[1]
        }
        return ""
    }
    
    function modify_path(path, budget_str, model_name) {
        # 提取数据集路径
        dataset_path = extract_dataset_path(path)
        
        # 提取基础路径（到 Results/ 之前的部分）
        base_path = ""
        if (match(path, /^(.+)\/Results\//, arr)) {
            base_path = arr[1]
        } else if (match(path, /^(.+)\/datasets\//, arr)) {
            base_path = arr[1]
            # 从 base_path 中移除模型名称部分
            if (match(base_path, /\/Results\/[^\/]+$/)) {
                sub(/\/Results\/[^\/]+$/, "", base_path)
            }
        } else {
            # 如果都不匹配，使用 dirname
            n = split(path, parts, "/")
            for (i = 1; i < n; i++) {
                if (i > 1) base_path = base_path "/"
                base_path = base_path parts[i]
            }
        }
        
        # 确定模型名称
        final_model_name = model_name
        if (final_model_name == "") {
            # 如果没有指定模型名称，尝试从原路径中提取
            if (match(path, /\/Results\/([^\/]+)\//, arr)) {
                final_model_name = arr[1]
            } else {
                final_model_name = "Qwen2_5_Omni_3B"  # 默认值
            }
        }
        
        # 清理数据集路径（去掉末尾的 /）
        gsub(/\/$/, "", dataset_path)
        
        # 构建新路径
        if (dataset_path != "") {
            return base_path "/Results/" final_model_name "/datasets/" dataset_path "/budget_" budget_str
        } else {
            # 如果没有找到 datasets/，使用原来的逻辑
            gsub(/\/$/, "", path)
            if (final_model_name != "") {
                gsub(/\/Results\/[^\/]+\//, "/Results/" final_model_name "/", path)
            }
            return path "/budget_" budget_str
        }
    }
    
    {
        line = $0
        # 修改 BUDGET 值
        if (match(line, /^[[:space:]]*export BUDGET=/)) {
            sub(/export BUDGET=.*/, "export BUDGET=" budget, line)
        }
        # 修改 --output-dir 路径
        if (match(line, /--output-dir[[:space:]]+([^[:space:]]+)/, arr)) {
            old_path = arr[1]
            new_path = modify_path(old_path, budget_str, model_name)
            gsub(/--output-dir[[:space:]]+[^[:space:]]+/, "--output-dir " new_path, line)
        }
        # 修改 --timing-output-file 路径
        if (match(line, /--timing-output-file[[:space:]]+([^[:space:]]+)/, arr)) {
            old_path = arr[1]
            new_path = modify_path(old_path, budget_str, model_name)
            gsub(/--timing-output-file[[:space:]]+[^[:space:]]+/, "--timing-output-file " new_path, line)
        }
        print line
    }
    ' "$script_path" > "$temp_script"
    
    chmod +x "$temp_script"
    
    # 运行临时脚本
    local start_time=$(date +%s)
    if bash "$temp_script" >> "$LOG_FILE" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log "✓ 成功完成: $script_name (Budget: $budget, 耗时: ${duration}秒)"
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log "✗ 运行失败: $script_name (Budget: $budget, 耗时: ${duration}秒)"
    fi
    
    # 清理临时脚本
    rm -f "$temp_script"
    
    log ""
}

# ============================================
# 主程序
# ============================================

main() {
    log "开始 Budget 参数搜索"
    log "脚本目录: $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    log "Budget 值: ${BUDGET_VALUES[*]}"
    log "脚本列表: ${SCRIPTS[*]}"
    if [ -n "$MODEL_NAME" ]; then
        log "模型名称: $MODEL_NAME"
    fi
    if [ ${#SKIP_SCRIPTS[@]} -gt 0 ]; then
        log "跳过的脚本: ${SKIP_SCRIPTS[*]}"
    fi
    log "日志文件: $LOG_FILE"
    log ""
    
    local total_runs=$((${#SCRIPTS[@]} * ${#BUDGET_VALUES[@]}))
    local current_run=0
    
    # 遍历每个脚本
    for script_name in "${SCRIPTS[@]}"; do
        script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$script_name"
        
        if [ ! -f "$script_path" ]; then
            log "警告: 脚本不存在: $script_path，跳过"
            continue
        fi
        
        log "处理脚本: $script_name"
        
        # 遍历每个 budget 值
        for budget in "${BUDGET_VALUES[@]}"; do
            current_run=$((current_run + 1))
            log "进度: [$current_run/$total_runs]"
            
            run_script_with_budget "$script_path" "$budget"
        done
        
        log "完成脚本: $script_name"
        log ""
    done
    
    log "============================================"
    log "所有 Budget 搜索完成！"
    log "总运行次数: $current_run"
    log "日志文件: $LOG_FILE"
    log "============================================"
}

# ============================================
# 运行主程序
# ============================================

# 解析命令行参数
parse_args "$@"

# 初始化脚本列表
init_scripts

# 设置日志文件
LOG_DIR="./budget_search_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/budget_search_${TIMESTAMP}.log"

# 运行主程序
main

