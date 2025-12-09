#!/usr/bin/env python3
"""
批量生成音频头的 JSON 分数文件
遍历 results 目录下的所有子文件夹，为每个 word_attention_analysis_*.json 生成对应的 score JSON 文件
输出到 head_score 目录
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import sys
import os

# 添加脚本所在目录到 Python 路径，以便导入 generate_head_scores
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# 导入 generate_head_scores.py 中的函数
from generate_head_scores import extract_per_sample_scores, detect_num_layers_and_heads
import re

def parse_params_from_filename(filename):
    """
    从文件名中解析参数：topk, alpha, beta, gamma
    例如：layer_head_attention_stats_topk64_alpha0_beta0_gamma1.json
    返回: dict with keys: topk, alpha, beta, gamma (values as strings, or None if not found)
    """
    params = {'topk': None, 'alpha': None, 'beta': None, 'gamma': None}
    
    # 提取 topk
    topk_match = re.search(r'_topk(\d+)', filename)
    if topk_match:
        params['topk'] = topk_match.group(1)
    
    # 提取 alpha
    alpha_match = re.search(r'_alpha([\d.]+)', filename)
    if alpha_match:
        params['alpha'] = alpha_match.group(1)
    
    # 提取 beta
    beta_match = re.search(r'_beta([\d.]+)', filename)
    if beta_match:
        params['beta'] = beta_match.group(1)
    
    # 提取 gamma
    gamma_match = re.search(r'_gamma([\d.]+)', filename)
    if gamma_match:
        params['gamma'] = gamma_match.group(1)
    
    return params

def generate_output_filename(input_filename, alpha=None, beta=None, gamma=None):
    """
    生成输出文件名
    - topk 从输入文件名中提取
    - alpha, beta, gamma 优先使用命令行参数，如果没有则从文件名中提取
    
    返回: 输出文件名（不含路径）
    """
    # 从文件名中解析参数
    file_params = parse_params_from_filename(input_filename)
    
    # 确定 topk（必须从文件名提取）
    topk = file_params['topk']
    if topk is None:
        raise ValueError(f"Cannot extract topk from filename: {input_filename}")
    
    # 确定 alpha, beta, gamma（优先使用命令行参数）
    # 如果命令行参数是 None，则从文件名提取；如果都没有，使用默认值 1.0
    alpha_val = alpha if alpha is not None else (float(file_params['alpha']) if file_params['alpha'] is not None else 1.0)
    beta_val = beta if beta is not None else (float(file_params['beta']) if file_params['beta'] is not None else 1.0)
    gamma_val = gamma if gamma is not None else (float(file_params['gamma']) if file_params['gamma'] is not None else 1.0)
    
    # 格式化参数值（去掉小数点后的0）
    alpha_str = f"{alpha_val:.1f}".rstrip('0').rstrip('.')
    beta_str = f"{beta_val:.1f}".rstrip('0').rstrip('.')
    gamma_str = f"{gamma_val:.1f}".rstrip('0').rstrip('.')
    
    # 生成文件名
    output_filename = f"audio_head_scores_topk{topk}_alpha{alpha_str}_beta{beta_str}_gamma{gamma_str}.json"
    
    return output_filename

def generate_aggregated_head_scores(input_file, output_file, num_layers=None, num_heads=None, 
                                     alpha=None, beta=None, gamma=None):
    """
    从 word_attention_analysis.json 或 layer_head_attention_stats.json 生成聚合的 head scores
    格式: {"layer-head": [importance_score, alpha_score, beta_score, gamma_score]}
    
    优先从 layer_head_attention_stats_*.json 读取 importance_score（已包含 alpha/beta/gamma 信息），
    如果没有该文件，则从 word_attention_analysis.json 计算。
    
    如果提供了 alpha, beta, gamma 参数，将使用这些参数重新计算 importance_score：
    importance_score = alpha * avg_hit_ratio + beta * avg_attn_ratio + gamma * avg_coverage_ratio
    
    输出格式说明：
    - importance_score: 加权总分 = alpha * avg_hit_ratio + beta * avg_attn_ratio + gamma * avg_coverage_ratio
    - alpha_score: 只算 alpha 参数的分数 = alpha * avg_hit_ratio
    - beta_score: 只算 beta 参数的分数 = beta * avg_attn_ratio
    - gamma_score: 只算 gamma 参数的分数 = gamma * avg_coverage_ratio
    
    Args:
        input_file: 输入文件路径（word_attention_analysis.json 或 layer_head_attention_stats.json）
        output_file: 输出文件路径
        num_layers: 层数（如果为 None，则自动检测）
        num_heads: 头数（如果为 None，则自动检测）
        alpha: hit ratio 权重（如果为 None，使用文件中的值）
        beta: attention ratio 权重（如果为 None，使用文件中的值）
        gamma: coverage ratio 权重（如果为 None，使用文件中的值）
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return False
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating aggregated head scores")
    print(f"{'='*60}")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    
    # 检查是否是 layer_head_attention_stats 文件
    is_layer_stats_file = "layer_head_attention_stats" in input_path.name
    
    if is_layer_stats_file:
        # 从 layer_head_attention_stats_*.json 读取 importance_score
        print(f"\nLoading layer_head_attention_stats from {input_path.name}...")
        with open(input_path, 'r') as f:
            layer_head_stats = json.load(f)
        
        # 自动检测层数和头数（如果需要）
        if num_layers is None or num_heads is None:
            all_layers = set()
            all_heads = set()
            for key in layer_head_stats.keys():
                if key.startswith("L") and "_H" in key:
                    parts = key[1:].split("_H")
                    if len(parts) == 2:
                        layer = int(parts[0])
                        head = int(parts[1])
                        all_layers.add(layer)
                        all_heads.add(head)
            if num_layers is None:
                num_layers = max(all_layers) + 1 if all_layers else 36
            if num_heads is None:
                num_heads = max(all_heads) + 1 if all_heads else 16
            print(f"Detected: {num_layers} layers, {num_heads} heads")
        else:
            print(f"\nUsing specified: {num_layers} layers, {num_heads} heads")
        
        # 提取 importance_score 和 avg_attn_ratio
        print(f"Extracting importance_score and avg_attn_ratio...")
        
        # 检查是否需要重新计算 importance_score
        recalculate_score = (alpha is not None) or (beta is not None) or (gamma is not None)
        if recalculate_score:
            # 使用默认值 1.0 如果未指定
            alpha_val = alpha if alpha is not None else 1.0
            beta_val = beta if beta is not None else 1.0
            gamma_val = gamma if gamma is not None else 1.0
            print(f"  Recalculating importance_score with alpha={alpha_val}, beta={beta_val}, gamma={gamma_val}")
        else:
            # 如果没有提供参数，尝试从文件名中提取，否则使用默认值 1.0
            file_params = parse_params_from_filename(input_path.name)
            alpha_val = float(file_params['alpha']) if file_params['alpha'] is not None else 1.0
            beta_val = float(file_params['beta']) if file_params['beta'] is not None else 1.0
            gamma_val = float(file_params['gamma']) if file_params['gamma'] is not None else 1.0
            print(f"  Using importance_score from file, with alpha={alpha_val}, beta={beta_val}, gamma={gamma_val} from filename")
        
        aggregated_format = {}
        
        for layer in range(num_layers):
            for head in range(num_heads):
                key = f"{layer}-{head}"
                layer_head_key = f"L{layer}_H{head}"
                
                if layer_head_key in layer_head_stats:
                    stats = layer_head_stats[layer_head_key]
                    avg_attn_ratio = stats.get("avg_attn_ratio", 0.0)
                    avg_hit_ratio = stats.get("avg_hit_ratio", 0.0)
                    avg_coverage_ratio = stats.get("avg_coverage_ratio", 0.0)
                    
                    if recalculate_score:
                        # 重新计算 importance_score
                        importance_score = (
                            alpha_val * avg_hit_ratio +
                            beta_val * avg_attn_ratio +
                            gamma_val * avg_coverage_ratio
                        )
                    else:
                        # 使用文件中的 importance_score
                        importance_score = stats.get("importance_score", 0.0)
                    
                    # 计算单独的 alpha, beta, gamma 分数
                    alpha_score = alpha_val * avg_hit_ratio
                    beta_score = beta_val * avg_attn_ratio
                    gamma_score = gamma_val * avg_coverage_ratio
                    
                    # 输出格式: [importance_score, alpha_score, beta_score, gamma_score]
                    aggregated_format[key] = [importance_score, alpha_score, beta_score, gamma_score]
                else:
                    # 如果该 head 没有数据，使用 [0.0, 0.0, 0.0, 0.0]
                    aggregated_format[key] = [0.0, 0.0, 0.0, 0.0]
    else:
        # 从 word_attention_analysis.json 计算（原有逻辑）
        # 自动检测层数和头数（如果需要）
        if num_layers is None or num_heads is None:
            print(f"\nAuto-detecting layers and heads...")
            detected_layers, detected_heads = detect_num_layers_and_heads(input_path)
            if num_layers is None:
                num_layers = detected_layers
            if num_heads is None:
                num_heads = detected_heads
            print(f"Detected: {num_layers} layers, {num_heads} heads")
        else:
            print(f"\nUsing specified: {num_layers} layers, {num_heads} heads")
        
        # 提取每个样本的原始分数
        print(f"\nLoading {input_path.name}...")
        print(f"Extracting per-sample scores...")
        per_sample_scores = extract_per_sample_scores(input_path)
        
        # 计算聚合分数（平均值和中位数）
        print(f"Computing aggregated scores (mean and median)...")
        # 注意：从 word_attention_analysis.json 读取时，没有 alpha/beta/gamma 的概念
        # 为了保持格式一致性，输出 4 个值，但后三个值使用相同的 avg_ratio
        aggregated_format = {}
        
        for layer in range(num_layers):
            for head in range(num_heads):
                key = f"{layer}-{head}"
                if key in per_sample_scores and per_sample_scores[key]:
                    scores = per_sample_scores[key]
                    # 计算平均值
                    avg_ratio = sum(scores) / len(scores) if scores else 0.0
                    # 计算中位数
                    sorted_scores = sorted(scores)
                    n = len(sorted_scores)
                    if n == 0:
                        median_ratio = 0.0
                    elif n % 2 == 0:
                        median_ratio = (sorted_scores[n//2 - 1] + sorted_scores[n//2]) / 2.0
                    else:
                        median_ratio = sorted_scores[n//2]
                    
                    # 注意：从 word_attention_analysis.json 计算时，没有 importance_score 和单独的 alpha/beta/gamma 分数
                    # 为了保持格式一致性，输出 4 个值：
                    # [avg_ratio, avg_ratio, avg_ratio, avg_ratio]
                    # 第一个是加权分数（这里等于 avg_ratio），后三个分别对应 alpha/beta/gamma（这里都等于 avg_ratio）
                    aggregated_format[key] = [avg_ratio, avg_ratio, avg_ratio, avg_ratio]
                else:
                    # 如果该 head 没有数据，使用 [0.0, 0.0, 0.0, 0.0]
                    aggregated_format[key] = [0.0, 0.0, 0.0, 0.0]
    
    # 保存为 JSON 文件
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(aggregated_format, f, indent=4)
    
    print(f"✓ Saved successfully!")
    print(f"\nStatistics:")
    print(f"  Total heads: {len(aggregated_format)}")
    if aggregated_format:
        non_zero_heads = sum(1 for v in aggregated_format.values() if any(x > 0 for x in v))
        print(f"  Non-zero heads: {non_zero_heads}")
        print(f"  Zero heads: {len(aggregated_format) - non_zero_heads}")
        # 计算所有加权分数的统计
        weighted_scores = [v[0] for v in aggregated_format.values()]
        if weighted_scores:
            print(f"  Mean weighted score (across all heads): {sum(weighted_scores) / len(weighted_scores):.6f}")
            print(f"  Max weighted score: {max(weighted_scores):.6f}")
            print(f"  Min weighted score: {min(weighted_scores):.6f}")
    
    return True

def batch_process_results(results_dir, head_score_dir, num_layers=None, num_heads=None,
                          alpha=None, beta=None, gamma=None, specific_folder=None):
    """
    批量处理 results 目录下的所有子文件夹，或处理指定的单个文件夹
    
    Args:
        results_dir: results 目录路径
        head_score_dir: head_score 输出目录路径
        num_layers: 层数（如果为 None，则自动检测）
        num_heads: 头数（如果为 None，则自动检测）
        alpha: hit ratio 权重（如果为 None，使用文件中的值）
        beta: attention ratio 权重（如果为 None，使用文件中的值）
        gamma: coverage ratio 权重（如果为 None，使用文件中的值）
        specific_folder: 如果指定，只处理该文件夹（可以是绝对路径或相对于 results_dir 的路径）
    """
    results_path = Path(results_dir)
    head_score_path = Path(head_score_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_path}")
        return
    
    # 确保输出目录存在
    head_score_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Batch Processing Head Scores")
    print(f"{'='*80}")
    print(f"Results directory: {results_path}")
    print(f"Output directory: {head_score_path}")
    if specific_folder:
        print(f"Specific folder: {specific_folder}")
    if alpha is not None or beta is not None or gamma is not None:
        print(f"Weights: alpha={alpha if alpha is not None else 'default'}, "
              f"beta={beta if beta is not None else 'default'}, "
              f"gamma={gamma if gamma is not None else 'default'}")
    print(f"{'='*80}\n")
    
    # 确定要处理的文件夹列表
    if specific_folder:
        # 处理指定的文件夹
        specific_path = Path(specific_folder)
        if not specific_path.is_absolute():
            # 相对路径，相对于 results_dir
            specific_path = results_path / specific_path
        elif not specific_path.exists():
            # 绝对路径，检查是否存在
            print(f"Error: Specific folder not found: {specific_path}")
            return
        
        if not specific_path.is_dir():
            print(f"Error: Path is not a directory: {specific_path}")
            return
        
        subdirs_to_process = [specific_path]
    else:
        # 遍历 results 目录下的所有子文件夹
        subdirs_to_process = [subdir for subdir in sorted(results_path.iterdir()) if subdir.is_dir()]
    
    # 遍历要处理的文件夹
    processed_count = 0
    failed_count = 0
    
    for subdir in subdirs_to_process:
        if not subdir.is_dir():
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing: {subdir.name}")
        print(f"{'='*80}")
        
        # 优先查找 layer_head_attention_stats_*.json 文件（包含 importance_score）
        layer_stats_files = list(subdir.glob("layer_head_attention_stats_*.json"))
        word_attention_files = list(subdir.glob("word_attention_analysis_*.json"))
        
        # 优先使用 layer_head_attention_stats 文件
        if layer_stats_files:
            input_file = layer_stats_files[0]
            if len(layer_stats_files) > 1:
                print(f"  ⚠ Multiple layer_head_attention_stats files found, using: {input_file.name}")
            
            # 生成输出文件名：topk 从文件名提取，alpha/beta/gamma 从命令行参数提取
            try:
                output_filename = generate_output_filename(
                    input_file.name, 
                    alpha=alpha, 
                    beta=beta, 
                    gamma=gamma
                )
                # 显示文件名生成信息
                file_params = parse_params_from_filename(input_file.name)
                print(f"  Extracted from filename: topk={file_params['topk']}, "
                      f"alpha={file_params['alpha']}, beta={file_params['beta']}, gamma={file_params['gamma']}")
                if alpha is not None or beta is not None or gamma is not None:
                    print(f"  Using command-line params: alpha={alpha}, beta={beta}, gamma={gamma}")
                print(f"  Output filename: {output_filename}")
            except ValueError as e:
                print(f"  ⚠ Error generating output filename: {e}")
                # 回退到使用子文件夹名
                output_filename = f"audio_head_scores_{subdir.name}.json"
        elif word_attention_files:
            # 如果没有 layer_head_attention_stats 文件，使用 word_attention_analysis 文件
            input_file = word_attention_files[0]
            if len(word_attention_files) > 1:
                print(f"  ⚠ Multiple word_attention_analysis files found, using: {word_attention_files[0].name}")
            
            # 生成输出文件名：topk 从文件名提取，alpha/beta/gamma 从命令行参数提取
            try:
                output_filename = generate_output_filename(
                    input_file.name, 
                    alpha=alpha, 
                    beta=beta, 
                    gamma=gamma
                )
                # 显示文件名生成信息
                file_params = parse_params_from_filename(input_file.name)
                print(f"  Extracted from filename: topk={file_params['topk']}, "
                      f"alpha={file_params['alpha']}, beta={file_params['beta']}, gamma={file_params['gamma']}")
                if alpha is not None or beta is not None or gamma is not None:
                    print(f"  Using command-line params: alpha={alpha}, beta={beta}, gamma={gamma}")
                print(f"  Output filename: {output_filename}")
            except ValueError as e:
                print(f"  ⚠ Error generating output filename: {e}")
                # 回退到使用子文件夹名
                output_filename = f"audio_head_scores_{subdir.name}.json"
        else:
            print(f"  ⚠ No layer_head_attention_stats_*.json or word_attention_analysis_*.json file found in {subdir.name}")
            continue
        
        output_file = head_score_path / output_filename
        
        # 生成聚合的 head scores
        success = generate_aggregated_head_scores(
            input_file=str(input_file),
            output_file=str(output_file),
            num_layers=num_layers,
            num_heads=num_heads,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )
        
        if success:
            processed_count += 1
            print(f"  ✓ Successfully processed: {subdir.name}")
        else:
            failed_count += 1
            print(f"  ✗ Failed to process: {subdir.name}")
    
    # 总结
    print(f"\n{'='*80}")
    print(f"Batch Processing Summary")
    print(f"{'='*80}")
    print(f"  Total processed: {processed_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Output directory: {head_score_path}")
    print(f"{'='*80}\n")

def main():
    # 获取脚本所在目录作为基准路径
    script_dir = Path(__file__).parent.absolute()
    
    parser = argparse.ArgumentParser(
        description='Batch generate audio head scores from word_attention_analysis.json files in results subdirectories',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default=None,
        help='Path to results directory containing subdirectories with word_attention_analysis_*.json files (default: script_dir/results)'
    )
    parser.add_argument(
        '--head_score_dir',
        type=str,
        default=None,
        help='Path to head_score output directory (default: script_dir/head_score)'
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=None,
        help='Number of layers (auto-detect if not specified)'
    )
    parser.add_argument(
        '--num_heads',
        type=int,
        default=None,
        help='Number of attention heads (auto-detect if not specified)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=None,
        help='Weight for hit ratio (S_hit) in importance_score calculation. If not specified, uses value from file.'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=None,
        help='Weight for attention ratio (S_attn) in importance_score calculation. If not specified, uses value from file.'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=None,
        help='Weight for coverage ratio (S_cov) in importance_score calculation. If not specified, uses value from file.'
    )
    parser.add_argument(
        '--specific_folder',
        type=str,
        default=None,
        help='Process only a specific folder (can be absolute path or relative to results_dir). If not specified, processes all subdirectories.'
    )
    
    args = parser.parse_args()
    
    # 如果没有指定路径，使用脚本所在目录的相对路径
    if args.results_dir is None:
        results_dir = script_dir / 'results'
    else:
        results_dir = Path(args.results_dir)
        if not results_dir.is_absolute():
            results_dir = script_dir / results_dir
    
    if args.head_score_dir is None:
        head_score_dir = script_dir / 'head_score'
    else:
        head_score_dir = Path(args.head_score_dir)
        if not head_score_dir.is_absolute():
            head_score_dir = script_dir / head_score_dir
    
    batch_process_results(
        results_dir=str(results_dir),
        head_score_dir=str(head_score_dir),
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        specific_folder=args.specific_folder
    )

if __name__ == "__main__":
    main()

'''
python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/batch_generate_head_scores.py \
    --specific_folder /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/results/topk48 \
    --alpha 1.0 \
    --beta 1.0 \
    --gamma 1.0 &
    
python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/batch_generate_head_scores.py \
    --specific_folder /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/results/topk48 \
    --alpha 1.0 \
    --beta 0 \
    --gamma 0 &
    
python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/batch_generate_head_scores.py \
    --specific_folder /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/results/topk48 \
    --alpha 0 \
    --beta 1.0 \
    --gamma 0 &
    
python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/batch_generate_head_scores.py \
    --specific_folder /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/results/topk48 \
    --alpha 0 \
    --beta 0 \
    --gamma 1.0

'''