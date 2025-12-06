#!/usr/bin/env python3
"""
生成音频头的 JSON 分数文件，格式与 visual_head 保持一致
从 word_attention_analysis.json 提取每个样本的原始分数
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

def extract_per_sample_scores(word_attention_file):
    """
    从 word_attention_analysis.json 提取每个样本的原始分数
    格式: {"layer-head": [score1, score2, score3, ...], ...}
    每个分数对应一个样本
    """
    with open(word_attention_file, 'r') as f:
        sample_results = json.load(f)
    
    # 初始化：为每个 layer-head 创建一个列表
    head_scores = defaultdict(list)
    
    # 遍历每个样本
    for sample in sample_results:
        tokens = sample.get('tokens', [])
        
        # 收集该样本所有 token 的 layer-head 分数
        sample_head_scores = defaultdict(list)
        
        for token in tokens:
            layer_head_stats = token.get('layer_head_stats', {})

            # 为了避免把 per-head 的 attn_weight_on_word 除以错误的分母（例如某些实现中
            # attn_weight_on_audio 可能表示每个 head 的值或已被平均），这里按 token 层面
            # 对所有 heads 的 attn_weight_on_audio 求和作为归一化分母。这样得到的 attn_ratio
            # 表示该 head 在该 token 上相对于所有 heads 的比例，避免被乘上 head 数量。
            total_audio_weight = sum(s.get('attn_weight_on_audio', 0.0) for s in layer_head_stats.values())

            for key, stats in layer_head_stats.items():
                # 解析 "L{layer}_H{head}" 格式
                if key.startswith("L") and "_H" in key:
                    parts = key[1:].split("_H")
                    if len(parts) == 2:
                        layer = int(parts[0])
                        head = int(parts[1])
                        visual_key = f"{layer}-{head}"

                        attn_weight_on_word = stats.get('attn_weight_on_word', 0.0)

                        if total_audio_weight > 0:
                            attn_ratio = attn_weight_on_word / total_audio_weight
                        else:
                            attn_ratio = 0.0

                        # 保证数值在合理范围
                        if attn_ratio < 0:
                            attn_ratio = 0.0
                        if attn_ratio > 1:
                            attn_ratio = 1.0

                        sample_head_scores[visual_key].append(attn_ratio)
        
        # 对每个 layer-head，计算该样本的平均分数
        for visual_key, ratios in sample_head_scores.items():
            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                head_scores[visual_key].append(avg_ratio)
            else:
                head_scores[visual_key].append(0.0)
    
    return dict(head_scores)

def detect_num_layers_and_heads(word_attention_file):
    """
    自动检测实际的层数和头数
    从 word_attention_analysis.json 中检测
    """
    num_layers = None
    num_heads = None
    
    try:
        with open(word_attention_file, 'r') as f:
            data = json.load(f)
        
        all_layers = set()
        all_heads = set()
        
        if isinstance(data, list) and len(data) > 0:
            for sample in data:
                tokens = sample.get('tokens', [])
                for token in tokens:
                    layer_head_stats = token.get('layer_head_stats', {})
                    for key in layer_head_stats.keys():
                        if key.startswith('L') and '_H' in key:
                            parts = key[1:].split('_H')
                            if len(parts) == 2:
                                layer = int(parts[0])
                                head = int(parts[1])
                                all_layers.add(layer)
                                all_heads.add(head)
        
        if all_layers:
            num_layers = max(all_layers) + 1
        if all_heads:
            num_heads = max(all_heads) + 1
    except Exception as e:
        print(f"Warning: Failed to detect from word_attention_analysis.json: {e}")
    
    # 默认值（如果检测失败）
    if num_layers is None:
        num_layers = 36  # Qwen2.5-Omni-3B 默认 36 层
    if num_heads is None:
        num_heads = 16  # Qwen2.5-Omni-3B 默认 16 个 attention heads
    
    return num_layers, num_heads

def generate_head_scores(input_file, output_file, num_layers=None, num_heads=None):
    """
    从 word_attention_analysis.json 生成每个 head 的分数文件
    
    Args:
        input_file: 输入文件路径（word_attention_analysis.json）
        output_file: 输出文件路径
        num_layers: 层数（如果为 None，则自动检测）
        num_heads: 头数（如果为 None，则自动检测）
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating head scores")
    print(f"{'='*60}")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    
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
    
    # 确保所有头都存在（缺失的用零列表填充）
    complete_format = {}
    sample_count = len(next(iter(per_sample_scores.values()))) if per_sample_scores else 0
    
    for layer in range(num_layers):
        for head in range(num_heads):
            key = f"{layer}-{head}"
            if key in per_sample_scores:
                complete_format[key] = per_sample_scores[key]
            else:
                # 如果该 head 没有数据，创建一个与样本数相同的零列表
                complete_format[key] = [0.0] * sample_count
    
    # 保存为 JSON 文件
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(complete_format, f, indent=4)
    
    print(f"✓ Saved successfully!")
    print(f"\nStatistics:")
    print(f"  Total heads: {len(complete_format)}")
    if complete_format:
        sample_count = len(next(iter(complete_format.values())))
        print(f"  Samples per head: {sample_count}")
        non_zero_heads = sum(1 for v in complete_format.values() if any(x > 0 for x in v))
        print(f"  Non-zero heads: {non_zero_heads}")
        print(f"  Zero heads: {len(complete_format) - non_zero_heads}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate audio head scores from word_attention_analysis.json',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to word_attention_analysis.json'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to output JSON file (e.g., audio_head_scores.json)'
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
    
    args = parser.parse_args()
    
    generate_head_scores(
        input_file=args.input_file,
        output_file=args.output_file,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    )
    
    print("\n" + "="*60)
    print("Head score file generated successfully!")
    print("="*60)

if __name__ == "__main__":
    main()

