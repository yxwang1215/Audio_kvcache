#!/usr/bin/env python3
"""
生成音频头的 JSON 分数文件，格式与 visual_head 保持一致
从现有的分析结果中提取数据并转换为 visual_head 格式
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def load_layer_head_stats(stats_file):
    """加载 layer_head_attention_stats.json"""
    with open(stats_file, 'r') as f:
        return json.load(f)

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
    
    # 转换为 visual_head 格式（但每个 head 对应一个列表，而不是两个值）
    # 注意：visual_head 格式是 [score1, score2]，但实际应该是 [score1, score2, ..., scoreN]
    # 让我检查一下 visual_head 的实际格式...
    return dict(head_scores)

def convert_to_visual_head_format(layer_head_stats):
    """
    将 layer_head_stats 转换为 visual_head 格式（聚合版本）
    格式: {"layer-head": [score1, score2], ...}
    
    使用:
    - score1: avg_attn_ratio (平均注意力比例)
    - score2: median_attn_ratio (中位数注意力比例)
    
    注意：这是聚合版本，不是原始格式
    """
    visual_format = {}
    
    for key, stats in layer_head_stats.items():
        # 解析 "L{layer}_H{head}" 格式
        if key.startswith("L") and "_H" in key:
            parts = key[1:].split("_H")
            if len(parts) == 2:
                layer = int(parts[0])
                head = int(parts[1])
                
                # 转换为 "layer-head" 格式
                visual_key = f"{layer}-{head}"
                
                # 提取两个分数
                avg_ratio = stats.get("avg_attn_ratio", 0.0)
                median_ratio = stats.get("median_attn_ratio", 0.0)
                
                visual_format[visual_key] = [avg_ratio, median_ratio]
    
    return visual_format

def generate_head_scores_from_ranking(ranking_file):
    """
    从 head_ranking_analysis.json 生成分数文件
    优先使用 significant_heads，如果不存在则使用 top_20_by_avg_ratio
    """
    with open(ranking_file, 'r') as f:
        ranking_data = json.load(f)
    
    # 创建完整的 layer-head 映射
    visual_format = {}
    
    # 优先从 significant_heads 中提取所有头的信息
    heads_list = None
    if 'significant_heads' in ranking_data:
        heads_list = ranking_data['significant_heads']
    elif 'top_20_by_avg_ratio' in ranking_data:
        heads_list = ranking_data['top_20_by_avg_ratio']
    
    if heads_list:
        for head_info in heads_list:
            layer_head = head_info.get('layer_head', '')
            if layer_head.startswith("L") and "_H" in layer_head:
                parts = layer_head[1:].split("_H")
                if len(parts) == 2:
                    layer = int(parts[0])
                    head = int(parts[1])
                    visual_key = f"{layer}-{head}"
                    
                    avg_ratio = head_info.get('avg_attn_ratio', 0.0)
                    median_ratio = head_info.get('median_attn_ratio', 0.0)
                    
                    visual_format[visual_key] = [avg_ratio, median_ratio]
    
    return visual_format

def detect_num_layers_and_heads(word_attention_file=None, stats_file=None):
    """
    自动检测实际的层数和头数
    从 word_attention_analysis.json 或 layer_head_attention_stats.json 中检测
    """
    num_layers = None
    num_heads = None
    
    # 方法1: 从 word_attention_analysis.json 检测
    if word_attention_file and os.path.exists(word_attention_file):
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
    
    # 方法2: 从 layer_head_attention_stats.json 检测
    if (num_layers is None or num_heads is None) and stats_file and os.path.exists(stats_file):
        try:
            with open(stats_file, 'r') as f:
                data = json.load(f)
            
            all_layers = set()
            all_heads = set()
            
            for key in data.keys():
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
            print(f"Warning: Failed to detect from layer_head_attention_stats.json: {e}")
    
    # 默认值（如果检测失败）
    if num_layers is None:
        num_layers = 30  # Gemma3n-E2B 实际有 30 层
    if num_heads is None:
        num_heads = 8  # Gemma3n-E2B 有 8 个 attention heads
    
    return num_layers, num_heads

def ensure_all_heads_present(visual_format, num_layers=30, num_heads=8):
    """
    确保所有 layer-head 组合都存在，缺失的用 [0.0, 0.0] 填充
    注意: Gemma3n-E2B 实际有 30 层，8 个 attention heads
    """
    complete_format = {}
    
    for layer in range(num_layers):
        for head in range(num_heads):
            key = f"{layer}-{head}"
            if key in visual_format:
                complete_format[key] = visual_format[key]
            else:
                complete_format[key] = [0.0, 0.0]
    
    return complete_format

def process_batch(audio_head_dir, batch_name, batch_dir, head_score_dir, num_layers=None, num_heads=None):
    """处理单个批次的结果
    如果 num_layers 或 num_heads 为 None，会自动从数据中检测
    """
    print(f"\n{'='*60}")
    print(f"Processing {batch_name}")
    print(f"{'='*60}")
    
    # 自动检测层数和头数
    word_attention_file = batch_dir / "word_attention_analysis.json"
    stats_file = batch_dir / "layer_head_attention_stats.json"
    
    if num_layers is None or num_heads is None:
        detected_layers, detected_heads = detect_num_layers_and_heads(
            word_attention_file if word_attention_file.exists() else None,
            stats_file if stats_file.exists() else None
        )
        if num_layers is None:
            num_layers = detected_layers
        if num_heads is None:
            num_heads = detected_heads
        print(f"Auto-detected: {num_layers} layers, {num_heads} heads")
    else:
        print(f"Using specified: {num_layers} layers, {num_heads} heads")
    
    # 方法1: 从 word_attention_analysis.json 提取每个样本的原始分数（与 visual_head 格式一致）
    per_sample_scores = None
    if word_attention_file.exists():
        print(f"\nLoading word_attention_analysis.json...")
        print(f"Extracting per-sample scores (visual_head format)...")
        per_sample_scores = extract_per_sample_scores(word_attention_file)
        
        # 确保所有头都存在（缺失的用空列表填充）
        complete_format = {}
        for layer in range(num_layers):
            for head in range(num_heads):
                key = f"{layer}-{head}"
                if key in per_sample_scores:
                    complete_format[key] = per_sample_scores[key]
                else:
                    # 如果该 head 没有数据，创建一个与样本数相同的零列表
                    # 但我们需要知道样本数，所以先检查第一个有数据的 head
                    sample_count = len(next(iter(per_sample_scores.values()))) if per_sample_scores else 0
                    complete_format[key] = [0.0] * sample_count
        
        # 保存为 JSON 文件（与 visual_head 格式一致）
        if batch_name == "main":
            output_file = head_score_dir / "audio_head_scores.json"
        else:
            output_file = head_score_dir / f"audio_head_scores_{batch_name}.json"
        
        with open(output_file, 'w') as f:
            json.dump(complete_format, f, indent=4)
        
        print(f"Saved to: {output_file}")
        print(f"Total heads: {len(complete_format)}")
        if complete_format:
            sample_count = len(next(iter(complete_format.values())))
            print(f"Samples per head: {sample_count}")
            print(f"Non-zero heads: {sum(1 for v in complete_format.values() if any(x > 0 for x in v))}")
    else:
        print(f"Warning: {word_attention_file} not found")
    
    # 方法2: 从 layer_head_attention_stats.json 生成聚合版本（备用）
    stats_file = batch_dir / "layer_head_attention_stats.json"
    if stats_file.exists():
        print(f"\nLoading layer_head_attention_stats.json (aggregated version)...")
        layer_head_stats = load_layer_head_stats(stats_file)
        
        print(f"Converting to aggregated format...")
        visual_format = convert_to_visual_head_format(layer_head_stats)
        
        # 确保所有头都存在
        complete_format = ensure_all_heads_present(visual_format, num_layers=num_layers, num_heads=num_heads)
        
        # 保存为 JSON 文件（聚合版本）
        if batch_name == "main":
            output_file = head_score_dir / "audio_head_scores_aggregated.json"
        else:
            output_file = head_score_dir / f"audio_head_scores_aggregated_{batch_name}.json"
        
        with open(output_file, 'w') as f:
            json.dump(complete_format, f, indent=4)
        
        print(f"Saved aggregated version to: {output_file}")
        print(f"Total heads: {len(complete_format)}")
        print(f"Non-zero heads: {sum(1 for v in complete_format.values() if v[0] > 0 or v[1] > 0)}")
    else:
        print(f"Warning: {stats_file} not found")
    
    # 方法3: 生成聚合版本（每个 head 一个平均值，便于直接使用）
    if per_sample_scores:
        print(f"\nGenerating aggregated version (mean scores)...")
        aggregated_format = {}
        for key, scores in per_sample_scores.items():
            if scores:
                mean_score = sum(scores) / len(scores)
                aggregated_format[key] = [mean_score]  # 保持列表格式以兼容
            else:
                aggregated_format[key] = [0.0]
        
        # 确保所有头都存在
        complete_aggregated = {}
        sample_mean = next(iter(aggregated_format.values()))[0] if aggregated_format else 0.0
        for layer in range(num_layers):
            for head in range(num_heads):
                key = f"{layer}-{head}"
                if key in aggregated_format:
                    complete_aggregated[key] = aggregated_format[key]
                else:
                    complete_aggregated[key] = [0.0]
        
        # 保存聚合版本
        if batch_name == "main":
            output_file_agg = head_score_dir / "audio_head_scores_mean.json"
        else:
            output_file_agg = head_score_dir / f"audio_head_scores_mean_{batch_name}.json"
        
        with open(output_file_agg, 'w') as f:
            json.dump(complete_aggregated, f, indent=4)
        
        print(f"Saved aggregated (mean) version to: {output_file_agg}")
        print(f"Total heads: {len(complete_aggregated)}")
        print(f"Non-zero heads: {sum(1 for v in complete_aggregated.values() if v[0] > 0)}")
    
    # 方法4: 从 head_ranking_analysis.json 生成（如果存在）
    ranking_file = batch_dir / "head_ranking_analysis.json"
    if ranking_file.exists():
        print(f"\nLoading head_ranking_analysis.json...")
        visual_format_ranking = generate_head_scores_from_ranking(ranking_file)
        
        # 确保所有头都存在
        complete_format_ranking = ensure_all_heads_present(visual_format_ranking, num_layers=num_layers, num_heads=num_heads)
        
        # 保存为 JSON 文件
        if batch_name == "main":
            output_file_ranking = head_score_dir / "audio_head_scores_from_ranking.json"
        else:
            output_file_ranking = head_score_dir / f"audio_head_scores_from_ranking_{batch_name}.json"
        
        with open(output_file_ranking, 'w') as f:
            json.dump(complete_format_ranking, f, indent=4)
        
        print(f"Saved to: {output_file_ranking}")
        print(f"Total heads: {len(complete_format_ranking)}")
        print(f"Non-zero heads: {sum(1 for v in complete_format_ranking.values() if v[0] > 0 or v[1] > 0)}")
    else:
        print(f"Warning: {ranking_file} not found")

def main():
    # 设置路径
    audio_head_dir = Path(__file__).parent
    head_score_dir = audio_head_dir / "head_score"
    
    # 创建 head_score 目录
    head_score_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created head_score directory: {head_score_dir}")
    
    # 处理主结果目录（自动检测层数和头数）
    results_dir = audio_head_dir / "word_attention_results"
    if results_dir.exists():
        process_batch(audio_head_dir, "main", results_dir, head_score_dir, num_layers=None, num_heads=None)
    
    # 处理 batch1 结果目录（自动检测层数和头数）
    batch1_dir = audio_head_dir / "word_attention_results_batch1"
    if batch1_dir.exists():
        process_batch(audio_head_dir, "batch1", batch1_dir, head_score_dir, num_layers=None, num_heads=None)
    
    print("\n" + "="*60)
    print("Head score files generated successfully!")
    print("="*60)
    print(f"Output directory: {head_score_dir}")
    print("\nFiles created:")
    for file in sorted(head_score_dir.glob("*.json")):
        file_size = file.stat().st_size / 1024  # KB
        print(f"  - {file.name} ({file_size:.1f} KB)")

if __name__ == "__main__":
    main()

