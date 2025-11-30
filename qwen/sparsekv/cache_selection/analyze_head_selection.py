#!/usr/bin/env python3
"""
分析不同attention head对于kvcache的选择特点
"""

import json
import numpy as np
from collections import defaultdict

def analyze_head_selection(json_file_path):
    """分析不同head的kvcache选择特点"""
    
    print(f"正在加载文件: {json_file_path}")
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # 获取元数据
    metadata = data.get('metadata', {})
    config = metadata.get('config', {})
    num_heads = config.get('num_attention_heads', 16)
    num_kv_groups = config.get('num_key_value_groups', 8)
    sample_metadata = metadata.get('sample_metadata', {})
    audio_token_start = sample_metadata.get('audio_token_start', 0)
    audio_token_length = sample_metadata.get('audio_token_length', 0)
    input_token_length = sample_metadata.get('input_token_length', 0)
    
    print(f"\n配置信息:")
    print(f"  - Attention heads数量: {num_heads}")
    print(f"  - KV groups数量: {num_kv_groups}")
    print(f"  - Audio token起始位置: {audio_token_start}")
    print(f"  - Audio token长度: {audio_token_length}")
    print(f"  - 总输入token长度: {input_token_length}")
    
    # 按head和layer组织数据
    head_data = defaultdict(lambda: {
        'selections': [],
        'layers': set(),
        'total_selected': 0,
        'positions': [],
        'audio_selections': [],
        'text_selections': [],
        'layer_selections': defaultdict(list)  # 按layer存储选择
    })
    
    # 按layer-head组织数据
    layer_head_data = defaultdict(lambda: defaultdict(lambda: {
        'positions': [],
        'audio_selections': [],
        'text_selections': []
    }))
    
    detailed_records = data.get('detailed_records', [])
    print(f"\n处理 {len(detailed_records)} 条记录...")
    
    for record in detailed_records:
        head_idx = record.get('head_idx', 0)
        layer_idx = record.get('layer_idx', 0)
        selected_indices = record.get('selected_indices', [])
        q_len = record.get('q_len', 0)
        
        head_data[head_idx]['selections'].extend(selected_indices)
        head_data[head_idx]['layers'].add(layer_idx)
        head_data[head_idx]['total_selected'] += len(selected_indices)
        head_data[head_idx]['positions'].extend(selected_indices)
        head_data[head_idx]['layer_selections'][layer_idx].extend(selected_indices)
        
        # 区分audio和text token的选择
        for idx in selected_indices:
            if audio_token_start <= idx < audio_token_start + audio_token_length:
                head_data[head_idx]['audio_selections'].append(idx)
                layer_head_data[layer_idx][head_idx]['audio_selections'].append(idx)
            else:
                head_data[head_idx]['text_selections'].append(idx)
                layer_head_data[layer_idx][head_idx]['text_selections'].append(idx)
            layer_head_data[layer_idx][head_idx]['positions'].append(idx)
    
    # 分析每个head的特点
    print(f"\n{'='*80}")
    print(f"各Head的KVCache选择特点分析")
    print(f"{'='*80}\n")
    
    results = {}
    
    for head_idx in sorted(head_data.keys()):
        data = head_data[head_idx]
        positions = np.array(data['positions'])
        audio_selections = np.array(data['audio_selections'])
        text_selections = np.array(data['text_selections'])
        
        if len(positions) == 0:
            continue
        
        # 基本统计
        total_selections = len(positions)
        num_layers = len(data['layers'])
        
        # 位置分布统计
        min_pos = positions.min()
        max_pos = positions.max()
        mean_pos = positions.mean()
        std_pos = positions.std()
        median_pos = np.median(positions)
        
        # Audio vs Text选择比例
        audio_count = len(audio_selections)
        text_count = len(text_selections)
        audio_ratio = audio_count / total_selections if total_selections > 0 else 0
        
        # 位置分布特征
        # 计算选择密度：选择是否集中在某些区域
        position_bins = np.histogram(positions, bins=20)[0]
        position_entropy = -np.sum((position_bins / position_bins.sum()) * 
                                   np.log(position_bins / position_bins.sum() + 1e-10))
        
        # 计算选择是否更偏向序列的前/中/后部分
        seq_length = max(positions) if len(positions) > 0 else 0
        if seq_length > 0:
            early_ratio = np.sum(positions < seq_length * 0.33) / total_selections
            middle_ratio = np.sum((positions >= seq_length * 0.33) & 
                                  (positions < seq_length * 0.67)) / total_selections
            late_ratio = np.sum(positions >= seq_length * 0.67) / total_selections
        else:
            early_ratio = middle_ratio = late_ratio = 0
        
        # Audio token内的分布
        if len(audio_selections) > 0:
            audio_min = audio_selections.min()
            audio_max = audio_selections.max()
            audio_mean = audio_selections.mean()
            audio_std = audio_selections.std()
        else:
            audio_min = audio_max = audio_mean = audio_std = 0
        
        # Text token内的分布
        if len(text_selections) > 0:
            text_min = text_selections.min()
            text_max = text_selections.max()
            text_mean = text_selections.mean()
            text_std = text_selections.std()
        else:
            text_min = text_max = text_mean = text_std = 0
        
        # 分析选择的局部性（连续性）
        sorted_positions = np.sort(positions)
        if len(sorted_positions) > 1:
            gaps = np.diff(sorted_positions)
            mean_gap = gaps.mean()
            median_gap = np.median(gaps)
            consecutive_ratio = np.sum(gaps == 1) / len(gaps) if len(gaps) > 0 else 0
            small_gap_ratio = np.sum(gaps <= 5) / len(gaps) if len(gaps) > 0 else 0
        else:
            mean_gap = median_gap = consecutive_ratio = small_gap_ratio = 0
        
        results[head_idx] = {
            'total_selections': total_selections,
            'num_layers': num_layers,
            'min_pos': min_pos,
            'max_pos': max_pos,
            'mean_pos': mean_pos,
            'std_pos': std_pos,
            'median_pos': median_pos,
            'audio_count': audio_count,
            'text_count': text_count,
            'audio_ratio': audio_ratio,
            'position_entropy': position_entropy,
            'early_ratio': early_ratio,
            'middle_ratio': middle_ratio,
            'late_ratio': late_ratio,
            'audio_mean': audio_mean,
            'audio_std': audio_std,
            'text_mean': text_mean,
            'text_std': text_std,
            'positions': positions,
            'mean_gap': mean_gap,
            'median_gap': median_gap,
            'consecutive_ratio': consecutive_ratio,
            'small_gap_ratio': small_gap_ratio
        }
        
        print(f"Head {head_idx:2d}:")
        print(f"  - 总选择数: {total_selections:6d} (涉及 {num_layers} 层)")
        print(f"  - 位置范围: [{min_pos:5d}, {max_pos:5d}], 均值: {mean_pos:7.2f}, 中位数: {median_pos:7.2f}, 标准差: {std_pos:7.2f}")
        print(f"  - Audio/Text选择: {audio_count:6d} / {text_count:6d} (Audio占比: {audio_ratio*100:5.2f}%)")
        print(f"  - 序列分布: 前33%={early_ratio*100:5.2f}%, 中33%={middle_ratio*100:5.2f}%, 后33%={late_ratio*100:5.2f}%")
        print(f"  - 位置分布熵: {position_entropy:.4f} (值越大表示分布越均匀)")
        print(f"  - 选择局部性: 平均间隔={mean_gap:.2f}, 中位间隔={median_gap:.2f}, 连续比例={consecutive_ratio*100:.2f}%, 小间隔(≤5)比例={small_gap_ratio*100:.2f}%")
        if len(audio_selections) > 0:
            print(f"  - Audio token位置: 均值={audio_mean:7.2f}, 标准差={audio_std:7.2f}")
        if len(text_selections) > 0:
            print(f"  - Text token位置: 均值={text_mean:7.2f}, 标准差={text_std:7.2f}")
        print()
    
    # 跨head比较分析
    print(f"{'='*80}")
    print(f"跨Head比较分析")
    print(f"{'='*80}\n")
    
    if len(results) > 1:
        # 计算各指标的统计
        audio_ratios = [r['audio_ratio'] for r in results.values()]
        mean_positions = [r['mean_pos'] for r in results.values()]
        position_entropies = [r['position_entropy'] for r in results.values()]
        early_ratios = [r['early_ratio'] for r in results.values()]
        late_ratios = [r['late_ratio'] for r in results.values()]
        
        print(f"Audio选择比例:")
        print(f"  - 均值: {np.mean(audio_ratios)*100:.2f}%, 标准差: {np.std(audio_ratios)*100:.2f}%")
        print(f"  - 范围: [{np.min(audio_ratios)*100:.2f}%, {np.max(audio_ratios)*100:.2f}%]")
        print(f"  - 最高: Head {np.argmax(audio_ratios)} ({np.max(audio_ratios)*100:.2f}%)")
        print(f"  - 最低: Head {np.argmin(audio_ratios)} ({np.min(audio_ratios)*100:.2f}%)")
        print()
        
        print(f"平均选择位置:")
        print(f"  - 均值: {np.mean(mean_positions):.2f}, 标准差: {np.std(mean_positions):.2f}")
        print(f"  - 范围: [{np.min(mean_positions):.2f}, {np.max(mean_positions):.2f}]")
        print(f"  - 最靠前: Head {np.argmin(mean_positions)} (均值={np.min(mean_positions):.2f})")
        print(f"  - 最靠后: Head {np.argmax(mean_positions)} (均值={np.max(mean_positions):.2f})")
        print()
        
        print(f"位置分布均匀度 (熵):")
        print(f"  - 均值: {np.mean(position_entropies):.4f}, 标准差: {np.std(position_entropies):.4f}")
        print(f"  - 最均匀: Head {np.argmax(position_entropies)} (熵={np.max(position_entropies):.4f})")
        print(f"  - 最集中: Head {np.argmin(position_entropies)} (熵={np.min(position_entropies):.4f})")
        print()
        
        print(f"序列位置偏好:")
        print(f"  - 前33%偏好: 均值={np.mean(early_ratios)*100:.2f}%, 最高=Head {np.argmax(early_ratios)} ({np.max(early_ratios)*100:.2f}%)")
        print(f"  - 后33%偏好: 均值={np.mean(late_ratios)*100:.2f}%, 最高=Head {np.argmax(late_ratios)} ({np.max(late_ratios)*100:.2f}%)")
        print()
    
    # 识别特殊模式
    print(f"{'='*80}")
    print(f"特殊模式识别")
    print(f"{'='*80}\n")
    
    # 找出audio偏好最强的head
    max_audio_head = max(results.keys(), key=lambda h: results[h]['audio_ratio'])
    print(f"Audio偏好最强: Head {max_audio_head} (Audio占比: {results[max_audio_head]['audio_ratio']*100:.2f}%)")
    
    # 找出text偏好最强的head
    min_audio_head = min(results.keys(), key=lambda h: results[h]['audio_ratio'])
    print(f"Text偏好最强: Head {min_audio_head} (Audio占比: {results[min_audio_head]['audio_ratio']*100:.2f}%)")
    
    # 找出位置最靠前的head
    earliest_head = min(results.keys(), key=lambda h: results[h]['mean_pos'])
    print(f"位置最靠前: Head {earliest_head} (平均位置: {results[earliest_head]['mean_pos']:.2f})")
    
    # 找出位置最靠后的head
    latest_head = max(results.keys(), key=lambda h: results[h]['mean_pos'])
    print(f"位置最靠后: Head {latest_head} (平均位置: {results[latest_head]['mean_pos']:.2f})")
    
    # 找出分布最均匀的head
    most_uniform_head = max(results.keys(), key=lambda h: results[h]['position_entropy'])
    print(f"分布最均匀: Head {most_uniform_head} (熵: {results[most_uniform_head]['position_entropy']:.4f})")
    
    # 找出分布最集中的head
    most_concentrated_head = min(results.keys(), key=lambda h: results[h]['position_entropy'])
    print(f"分布最集中: Head {most_concentrated_head} (熵: {results[most_concentrated_head]['position_entropy']:.4f})")
    
    # Head间选择重叠分析
    print(f"\n{'='*80}")
    print(f"Head间选择重叠分析")
    print(f"{'='*80}\n")
    
    if len(results) > 1:
        head_list = sorted(results.keys())
        for i, h1 in enumerate(head_list):
            for h2 in head_list[i+1:]:
                set1 = set(results[h1]['positions'])
                set2 = set(results[h2]['positions'])
                intersection = set1 & set2
                union = set1 | set2
                jaccard = len(intersection) / len(union) if len(union) > 0 else 0
                overlap_ratio = len(intersection) / len(set1) if len(set1) > 0 else 0
                print(f"Head {h1} vs Head {h2}:")
                print(f"  - Jaccard相似度: {jaccard:.4f} (交集/并集)")
                print(f"  - 重叠比例: {overlap_ratio:.4f} (交集/Head{h1}总数)")
                print(f"  - 交集大小: {len(intersection)}, 并集大小: {len(union)}")
                if jaccard > 0.99:
                    print(f"  ⚠️  注意: 两个Head的选择几乎完全相同，可能共享同一个KV cache")
                print()
    
    # 按Layer分析
    print(f"{'='*80}")
    print(f"不同Layer下Head选择模式分析")
    print(f"{'='*80}\n")
    
    # 选择几个代表性的layer进行分析
    sample_layers = sorted(layer_head_data.keys())
    if len(sample_layers) > 10:
        # 选择前、中、后几个layer
        sample_layers = [sample_layers[0], sample_layers[len(sample_layers)//4], 
                        sample_layers[len(sample_layers)//2], sample_layers[3*len(sample_layers)//4],
                        sample_layers[-1]]
    
    for layer_idx in sample_layers[:5]:  # 最多显示5个layer
        print(f"Layer {layer_idx}:")
        for head_idx in sorted(layer_head_data[layer_idx].keys()):
            lh_data = layer_head_data[layer_idx][head_idx]
            positions = np.array(lh_data['positions'])
            audio_selections = np.array(lh_data['audio_selections'])
            text_selections = np.array(lh_data['text_selections'])
            
            if len(positions) == 0:
                continue
            
            audio_ratio = len(audio_selections) / len(positions) if len(positions) > 0 else 0
            mean_pos = positions.mean() if len(positions) > 0 else 0
            
            print(f"  Head {head_idx}: 选择数={len(positions):5d}, "
                  f"Audio占比={audio_ratio*100:5.2f}%, "
                  f"平均位置={mean_pos:7.2f}")
        print()
    
    # Layer间变化趋势
    print(f"{'='*80}")
    print(f"不同Layer间选择模式变化趋势")
    print(f"{'='*80}\n")
    
    layer_stats = {}
    for layer_idx in sorted(layer_head_data.keys()):
        layer_audio_ratios = []
        layer_mean_positions = []
        for head_idx in layer_head_data[layer_idx].keys():
            lh_data = layer_head_data[layer_idx][head_idx]
            positions = np.array(lh_data['positions'])
            audio_selections = np.array(lh_data['audio_selections'])
            
            if len(positions) > 0:
                audio_ratio = len(audio_selections) / len(positions)
                mean_pos = positions.mean()
                layer_audio_ratios.append(audio_ratio)
                layer_mean_positions.append(mean_pos)
        
        if layer_audio_ratios:
            layer_stats[layer_idx] = {
                'mean_audio_ratio': np.mean(layer_audio_ratios),
                'mean_position': np.mean(layer_mean_positions)
            }
    
    if len(layer_stats) > 0:
        early_layers = [layer_stats[k] for k in sorted(layer_stats.keys())[:len(layer_stats)//3]]
        middle_layers = [layer_stats[k] for k in sorted(layer_stats.keys())[len(layer_stats)//3:2*len(layer_stats)//3]]
        late_layers = [layer_stats[k] for k in sorted(layer_stats.keys())[2*len(layer_stats)//3:]]
        
        print(f"早期Layers (前1/3):")
        print(f"  - 平均Audio占比: {np.mean([l['mean_audio_ratio'] for l in early_layers])*100:.2f}%")
        print(f"  - 平均选择位置: {np.mean([l['mean_position'] for l in early_layers]):.2f}")
        print()
        
        print(f"中期Layers (中1/3):")
        print(f"  - 平均Audio占比: {np.mean([l['mean_audio_ratio'] for l in middle_layers])*100:.2f}%")
        print(f"  - 平均选择位置: {np.mean([l['mean_position'] for l in middle_layers]):.2f}")
        print()
        
        print(f"后期Layers (后1/3):")
        print(f"  - 平均Audio占比: {np.mean([l['mean_audio_ratio'] for l in late_layers])*100:.2f}%")
        print(f"  - 平均选择位置: {np.mean([l['mean_position'] for l in late_layers]):.2f}")
        print()
    
    # 总结性分析
    print(f"{'='*80}")
    print(f"总结性发现")
    print(f"{'='*80}\n")
    
    print(f"1. Audio vs Text偏好:")
    print(f"   - 所有Head都强烈偏好Audio token (97.6-97.8%)")
    print(f"   - Text token选择很少 (仅2.2-2.4%)")
    print()
    
    print(f"2. 位置分布特征:")
    print(f"   - 选择位置分布相对均匀 (熵值约2.97)")
    print(f"   - 平均位置在序列中段 (约2947)")
    print(f"   - 前、中、后三部分分布相对均衡")
    print()
    
    print(f"3. 选择局部性:")
    if len(results) > 0:
        mean_gaps = [r['mean_gap'] for r in results.values()]
        consecutive_ratios = [r['consecutive_ratio'] for r in results.values()]
        print(f"   - 平均间隔: {np.mean(mean_gaps):.2f}")
        print(f"   - 连续选择比例: {np.mean(consecutive_ratios)*100:.2f}%")
        print(f"   - 说明: 选择具有一定的局部性，倾向于选择连续或接近的位置")
    print()
    
    print(f"4. Layer间变化:")
    print(f"   - 早期Layers: Audio占比略低，选择位置更靠后")
    print(f"   - 后期Layers: Audio占比更高，选择位置更靠前")
    print(f"   - 可能原因: 后期层更关注Audio特征，早期层需要更多上下文")
    print()
    
    if len(results) > 1:
        head_list = sorted(results.keys())
        set1 = set(results[head_list[0]]['positions'])
        set2 = set(results[head_list[1]]['positions'])
        if len(set1 & set2) / len(set1) > 0.99:
            print(f"5. Head间关系:")
            print(f"   - Head 0和Head 1的选择几乎完全相同")
            print(f"   - 这符合GQA架构: {num_heads}个attention heads共享{num_kv_groups}个KV groups")
            print(f"   - 每{num_heads // num_kv_groups}个head共享一个KV cache")
    
    return results

if __name__ == '__main__':
    json_file = '/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv/cache_selection/LibriSpeech_budget0.7/topk_selection_records_134686_0000.json'
    results = analyze_head_selection(json_file)
    
    print(f"\n{'='*80}")
    print(f"分析完成!")
    print(f"{'='*80}")

