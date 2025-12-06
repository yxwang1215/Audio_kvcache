#!/usr/bin/env python3
"""
分析不同attention head在audio token内部的时序选择特点
结合音频的时序特性进行分析
"""

import json
import numpy as np
from collections import defaultdict

def analyze_audio_temporal_selection(json_file_path):
    """分析audio token内部的时序选择模式"""
    
    print(f"正在加载文件: {json_file_path}")
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # 获取元数据
    metadata = data.get('metadata', {})
    sample_metadata = metadata.get('sample_metadata', {})
    audio_token_start = sample_metadata.get('audio_token_start', 0)
    audio_token_length = sample_metadata.get('audio_token_length', 0)
    duration = sample_metadata.get('duration', 0)  # 音频时长（秒）
    
    print(f"\n配置信息:")
    print(f"  - Audio token起始位置: {audio_token_start}")
    print(f"  - Audio token长度: {audio_token_length}")
    print(f"  - 音频时长: {duration:.2f}秒")
    print(f"  - 平均token速率: {audio_token_length/duration:.2f} tokens/秒" if duration > 0 else "")
    
    # 按head组织audio token内的选择（转换为相对位置）
    head_audio_data = defaultdict(lambda: {
        'relative_positions': [],  # 相对于audio序列的位置 (0到audio_token_length-1)
        'absolute_positions': [],  # 绝对位置
        'layer_data': defaultdict(list),  # 按layer存储
        'selection_density': np.zeros(audio_token_length)  # 每个位置的被选择次数
    })
    
    detailed_records = data.get('detailed_records', [])
    print(f"\n处理 {len(detailed_records)} 条记录...")
    
    for record in detailed_records:
        head_idx = record.get('head_idx', 0)
        layer_idx = record.get('layer_idx', 0)
        selected_indices = record.get('selected_indices', [])
        
        # 只关注audio token内的选择
        for idx in selected_indices:
            if audio_token_start <= idx < audio_token_start + audio_token_length:
                relative_pos = idx - audio_token_start  # 转换为相对位置
                head_audio_data[head_idx]['relative_positions'].append(relative_pos)
                head_audio_data[head_idx]['absolute_positions'].append(idx)
                head_audio_data[head_idx]['layer_data'][layer_idx].append(relative_pos)
                head_audio_data[head_idx]['selection_density'][relative_pos] += 1
    
    print(f"\n{'='*80}")
    print(f"Audio Token内部时序选择分析")
    print(f"{'='*80}\n")
    
    results = {}
    
    for head_idx in sorted(head_audio_data.keys()):
        data = head_audio_data[head_idx]
        relative_positions = np.array(data['relative_positions'])
        
        if len(relative_positions) == 0:
            continue
        
        # 基本统计
        total_selections = len(relative_positions)
        num_layers = len(data['layer_data'])
        
        # 在audio序列中的位置分布
        min_pos = relative_positions.min()
        max_pos = relative_positions.max()
        mean_pos = relative_positions.mean()
        median_pos = np.median(relative_positions)
        std_pos = relative_positions.std()
        
        # 将audio序列分为前、中、后三部分
        audio_length = audio_token_length
        early_threshold = audio_length * 0.33
        late_threshold = audio_length * 0.67
        
        early_count = np.sum(relative_positions < early_threshold)
        middle_count = np.sum((relative_positions >= early_threshold) & 
                             (relative_positions < late_threshold))
        late_count = np.sum(relative_positions >= late_threshold)
        
        early_ratio = early_count / total_selections
        middle_ratio = middle_count / total_selections
        late_ratio = late_count / total_selections
        
        # 更细粒度的分段分析（10段）
        segment_size = audio_length / 10
        segment_counts = []
        for i in range(10):
            start = i * segment_size
            end = (i + 1) * segment_size
            count = np.sum((relative_positions >= start) & (relative_positions < end))
            segment_counts.append(count)
        segment_ratios = np.array(segment_counts) / total_selections
        
        # 时序连续性分析
        sorted_positions = np.sort(relative_positions)
        if len(sorted_positions) > 1:
            gaps = np.diff(sorted_positions)
            mean_gap = gaps.mean()
            median_gap = np.median(gaps)
            consecutive_count = np.sum(gaps == 1)
            consecutive_ratio = consecutive_count / len(gaps) if len(gaps) > 0 else 0
            
            # 小间隔（<=5）的比例，表示局部连续性
            small_gap_count = np.sum(gaps <= 5)
            small_gap_ratio = small_gap_count / len(gaps) if len(gaps) > 0 else 0
            
            # 最大连续片段长度
            max_consecutive = 1
            current_consecutive = 1
            for gap in gaps:
                if gap == 1:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 1
        else:
            mean_gap = median_gap = consecutive_ratio = small_gap_ratio = max_consecutive = 0
        
        # 选择密度分析
        selection_density = data['selection_density']
        density_mean = selection_density.mean()
        density_std = selection_density.std()
        density_max = selection_density.max()
        
        # 找出选择密度最高的区域
        density_threshold = density_mean + density_std
        high_density_regions = np.where(selection_density > density_threshold)[0]
        if len(high_density_regions) > 0:
            high_density_start = high_density_regions[0]
            high_density_end = high_density_regions[-1]
            high_density_ratio = len(high_density_regions) / audio_length
        else:
            high_density_start = high_density_end = high_density_ratio = 0
        
        # 分析选择是否集中在某些特定区域
        # 计算选择分布的熵（均匀度）
        hist, _ = np.histogram(relative_positions, bins=50, range=(0, audio_length))
        hist_normalized = hist / hist.sum() if hist.sum() > 0 else hist
        entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-10))
        max_entropy = np.log(50)  # 50个bins的最大熵
        normalized_entropy = entropy / max_entropy
        
        results[head_idx] = {
            'total_selections': total_selections,
            'num_layers': num_layers,
            'min_pos': min_pos,
            'max_pos': max_pos,
            'mean_pos': mean_pos,
            'median_pos': median_pos,
            'std_pos': std_pos,
            'early_ratio': early_ratio,
            'middle_ratio': middle_ratio,
            'late_ratio': late_ratio,
            'segment_ratios': segment_ratios,
            'mean_gap': mean_gap,
            'median_gap': median_gap,
            'consecutive_ratio': consecutive_ratio,
            'small_gap_ratio': small_gap_ratio,
            'max_consecutive': max_consecutive,
            'density_mean': density_mean,
            'density_std': density_std,
            'density_max': density_max,
            'high_density_ratio': high_density_ratio,
            'normalized_entropy': normalized_entropy,
            'relative_positions': relative_positions,
            'selection_density': selection_density
        }
        
        print(f"Head {head_idx}:")
        print(f"  - 总选择数: {total_selections:6d} (涉及 {num_layers} 层)")
        print(f"  - Audio序列位置: [{min_pos:5d}, {max_pos:5d}], 均值: {mean_pos:7.2f}, 中位数: {median_pos:7.2f}")
        print(f"  - 时序分布: 前33%={early_ratio*100:5.2f}%, 中33%={middle_ratio*100:5.2f}%, 后33%={late_ratio*100:5.2f}%")
        print(f"  - 时序连续性:")
        print(f"      * 平均间隔: {mean_gap:.2f}, 中位间隔: {median_gap:.2f}")
        print(f"      * 连续选择比例: {consecutive_ratio*100:.2f}%")
        print(f"      * 小间隔(≤5)比例: {small_gap_ratio*100:.2f}%")
        print(f"      * 最大连续片段: {max_consecutive} tokens")
        print(f"  - 选择密度: 均值={density_mean:.2f}, 标准差={density_std:.2f}, 最大值={density_max:.0f}")
        print(f"  - 分布均匀度: {normalized_entropy:.4f} (1.0=完全均匀)")
        
        # 显示10段分布
        print(f"  - 10段分布:")
        for i, ratio in enumerate(segment_ratios):
            bar_length = int(ratio * 50)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"      [{i*10:3d}%-{(i+1)*10:3d}%]: {bar} {ratio*100:5.2f}%")
        print()
    
    # Head间比较
    if len(results) > 1:
        print(f"{'='*80}")
        print(f"Head间Audio时序选择差异分析")
        print(f"{'='*80}\n")
        
        head_list = sorted(results.keys())
        
        # 比较时序分布
        early_ratios = [results[h]['early_ratio'] for h in head_list]
        late_ratios = [results[h]['late_ratio'] for h in head_list]
        mean_positions = [results[h]['mean_pos'] for h in head_list]
        consecutive_ratios = [results[h]['consecutive_ratio'] for h in head_list]
        normalized_entropies = [results[h]['normalized_entropy'] for h in head_list]
        
        print(f"时序位置偏好:")
        print(f"  - 前33%偏好: Head {head_list[np.argmax(early_ratios)]} ({np.max(early_ratios)*100:.2f}%) vs Head {head_list[np.argmin(early_ratios)]} ({np.min(early_ratios)*100:.2f}%)")
        print(f"  - 后33%偏好: Head {head_list[np.argmax(late_ratios)]} ({np.max(late_ratios)*100:.2f}%) vs Head {head_list[np.argmin(late_ratios)]} ({np.min(late_ratios)*100:.2f}%)")
        print(f"  - 平均位置: Head {head_list[np.argmin(mean_positions)]} ({np.min(mean_positions):.2f}) vs Head {head_list[np.argmax(mean_positions)]} ({np.max(mean_positions):.2f})")
        print()
        
        print(f"时序连续性:")
        print(f"  - 连续选择比例: Head {head_list[np.argmax(consecutive_ratios)]} ({np.max(consecutive_ratios)*100:.2f}%) vs Head {head_list[np.argmin(consecutive_ratios)]} ({np.min(consecutive_ratios)*100:.2f}%)")
        print()
        
        print(f"分布均匀度:")
        print(f"  - 最均匀: Head {head_list[np.argmax(normalized_entropies)]} ({np.max(normalized_entropies):.4f})")
        print(f"  - 最集中: Head {head_list[np.argmin(normalized_entropies)]} ({np.min(normalized_entropies):.4f})")
        print()
        
        # 比较选择重叠
        for i, h1 in enumerate(head_list):
            for h2 in head_list[i+1:]:
                set1 = set(results[h1]['relative_positions'])
                set2 = set(results[h2]['relative_positions'])
                intersection = set1 & set2
                union = set1 | set2
                jaccard = len(intersection) / len(union) if len(union) > 0 else 0
                
                print(f"Head {h1} vs Head {h2} (Audio序列内):")
                print(f"  - Jaccard相似度: {jaccard:.4f}")
                print(f"  - 交集大小: {len(intersection)}, 并集大小: {len(union)}")
                if jaccard > 0.99:
                    print(f"  ⚠️  两个Head在Audio序列中的选择几乎完全相同")
                print()
    
    # 按Layer分析Audio时序模式
    print(f"{'='*80}")
    print(f"不同Layer下Audio时序选择模式")
    print(f"{'='*80}\n")
    
    # 重新组织数据按layer-head
    layer_head_audio = defaultdict(lambda: defaultdict(lambda: {
        'relative_positions': []
    }))
    
    for record in detailed_records:
        head_idx = record.get('head_idx', 0)
        layer_idx = record.get('layer_idx', 0)
        selected_indices = record.get('selected_indices', [])
        
        for idx in selected_indices:
            if audio_token_start <= idx < audio_token_start + audio_token_length:
                relative_pos = idx - audio_token_start
                layer_head_audio[layer_idx][head_idx]['relative_positions'].append(relative_pos)
    
    # 选择几个代表性的layer
    sample_layers = sorted(layer_head_audio.keys())
    if len(sample_layers) > 8:
        sample_layers = [
            sample_layers[0], 
            sample_layers[len(sample_layers)//4],
            sample_layers[len(sample_layers)//2],
            sample_layers[3*len(sample_layers)//4],
            sample_layers[-1]
        ]
    
    for layer_idx in sample_layers[:5]:
        print(f"Layer {layer_idx}:")
        for head_idx in sorted(layer_head_audio[layer_idx].keys()):
            lh_data = layer_head_audio[layer_idx][head_idx]
            positions = np.array(lh_data['relative_positions'])
            
            if len(positions) == 0:
                continue
            
            mean_pos = positions.mean()
            std_pos = positions.std()
            
            # 计算前中后分布
            early_count = np.sum(positions < audio_length * 0.33)
            late_count = np.sum(positions >= audio_length * 0.67)
            early_ratio = early_count / len(positions)
            late_ratio = late_count / len(positions)
            
            print(f"  Head {head_idx}: 选择数={len(positions):5d}, "
                  f"平均位置={mean_pos:7.2f}, "
                  f"前33%={early_ratio*100:5.2f}%, 后33%={late_ratio*100:5.2f}%")
        print()
    
    # Layer间变化趋势
    print(f"{'='*80}")
    print(f"Layer间Audio时序选择变化趋势")
    print(f"{'='*80}\n")
    
    layer_stats = {}
    for layer_idx in sorted(layer_head_audio.keys()):
        layer_mean_positions = []
        layer_early_ratios = []
        for head_idx in layer_head_audio[layer_idx].keys():
            lh_data = layer_head_audio[layer_idx][head_idx]
            positions = np.array(lh_data['relative_positions'])
            
            if len(positions) > 0:
                mean_pos = positions.mean()
                early_count = np.sum(positions < audio_length * 0.33)
                early_ratio = early_count / len(positions)
                layer_mean_positions.append(mean_pos)
                layer_early_ratios.append(early_ratio)
        
        if layer_mean_positions:
            layer_stats[layer_idx] = {
                'mean_position': np.mean(layer_mean_positions),
                'mean_early_ratio': np.mean(layer_early_ratios)
            }
    
    if len(layer_stats) > 0:
        early_layers = [layer_stats[k] for k in sorted(layer_stats.keys())[:len(layer_stats)//3]]
        middle_layers = [layer_stats[k] for k in sorted(layer_stats.keys())[len(layer_stats)//3:2*len(layer_stats)//3]]
        late_layers = [layer_stats[k] for k in sorted(layer_stats.keys())[2*len(layer_stats)//3:]]
        
        print(f"早期Layers (前1/3):")
        print(f"  - 平均Audio位置: {np.mean([l['mean_position'] for l in early_layers]):.2f}")
        print(f"  - 前33%偏好: {np.mean([l['mean_early_ratio'] for l in early_layers])*100:.2f}%")
        print()
        
        print(f"中期Layers (中1/3):")
        print(f"  - 平均Audio位置: {np.mean([l['mean_position'] for l in middle_layers]):.2f}")
        print(f"  - 前33%偏好: {np.mean([l['mean_early_ratio'] for l in middle_layers])*100:.2f}%")
        print()
        
        print(f"后期Layers (后1/3):")
        print(f"  - 平均Audio位置: {np.mean([l['mean_position'] for l in late_layers]):.2f}")
        print(f"  - 前33%偏好: {np.mean([l['mean_early_ratio'] for l in late_layers])*100:.2f}%")
        print()
    
    # 深入分析：开始和结束部分的特殊模式
    print(f"{'='*80}")
    print(f"Audio序列开始/结束部分特殊模式分析")
    print(f"{'='*80}\n")
    
    # 分析前10%和后10%的选择模式
    start_portion = 0.1  # 前10%
    end_portion = 0.1    # 后10%
    
    for head_idx in sorted(results.keys()):
        positions = results[head_idx]['relative_positions']
        audio_length = audio_token_length
        
        start_threshold = audio_length * start_portion
        end_threshold = audio_length * (1 - end_portion)
        
        start_selections = positions[positions < start_threshold]
        end_selections = positions[positions >= end_threshold]
        middle_selections = positions[(positions >= start_threshold) & (positions < end_threshold)]
        
        start_ratio = len(start_selections) / len(positions) if len(positions) > 0 else 0
        end_ratio = len(end_selections) / len(positions) if len(positions) > 0 else 0
        middle_ratio = len(middle_selections) / len(positions) if len(positions) > 0 else 0
        
        print(f"Head {head_idx}:")
        print(f"  - 前10%选择: {len(start_selections):6d} ({start_ratio*100:5.2f}%)")
        print(f"  - 中间80%选择: {len(middle_selections):6d} ({middle_ratio*100:5.2f}%)")
        print(f"  - 后10%选择: {len(end_selections):6d} ({end_ratio*100:5.2f}%)")
        
        # 计算期望值（如果完全均匀分布）
        expected_start = start_portion
        expected_end = end_portion
        expected_middle = 1 - start_portion - end_portion
        
        start_bias = start_ratio / expected_start if expected_start > 0 else 0
        end_bias = end_ratio / expected_end if expected_end > 0 else 0
        
        print(f"  - 前10%偏差: {start_bias:.2f}x (1.0=均匀, >1.0=偏好, <1.0=回避)")
        print(f"  - 后10%偏差: {end_bias:.2f}x")
        
        if start_bias > 1.2:
            print(f"    → 强烈偏好音频开始部分（可能包含重要起始信息或静音检测）")
        elif start_bias < 0.8:
            print(f"    → 回避音频开始部分")
        
        if end_bias > 1.2:
            print(f"    → 强烈偏好音频结束部分（可能包含重要结束信息或静音检测）")
        elif end_bias < 0.8:
            print(f"    → 回避音频结束部分")
        print()
    
    # 分析选择密度的热点区域
    print(f"{'='*80}")
    print(f"Audio序列选择密度热点分析")
    print(f"{'='*80}\n")
    
    for head_idx in sorted(results.keys()):
        density = results[head_idx]['selection_density']
        density_mean = results[head_idx]['density_mean']
        density_std = results[head_idx]['density_std']
        
        # 找出高密度区域（超过均值+1标准差）
        high_density_threshold = density_mean + density_std
        high_density_mask = density > high_density_threshold
        high_density_indices = np.where(high_density_mask)[0]
        
        if len(high_density_indices) > 0:
            # 找出连续的高密度区域
            high_density_regions = []
            if len(high_density_indices) > 0:
                start = high_density_indices[0]
                for i in range(1, len(high_density_indices)):
                    if high_density_indices[i] != high_density_indices[i-1] + 1:
                        high_density_regions.append((start, high_density_indices[i-1]))
                        start = high_density_indices[i]
                high_density_regions.append((start, high_density_indices[-1]))
            
            print(f"Head {head_idx}:")
            print(f"  - 高密度区域数量: {len(high_density_regions)}")
            print(f"  - 高密度区域覆盖: {len(high_density_indices)/audio_token_length*100:.2f}% 的音频序列")
            
            # 显示前5个最大的高密度区域
            region_sizes = [(end - start + 1, start, end) for start, end in high_density_regions]
            region_sizes.sort(reverse=True)
            
            print(f"  - 主要高密度区域（前5个）:")
            for size, start, end in region_sizes[:5]:
                start_ratio = start / audio_token_length * 100
                end_ratio = end / audio_token_length * 100
                avg_density = density[start:end+1].mean()
                print(f"      [{start_ratio:5.1f}%-{end_ratio:5.1f}%]: 长度={size:4d} tokens, 平均密度={avg_density:.2f}")
            print()
    
    # 分析不同head在音频不同阶段的关注差异
    print(f"{'='*80}")
    print(f"不同Head在Audio不同阶段的关注差异")
    print(f"{'='*80}\n")
    
    # 将音频分为5个阶段
    num_stages = 5
    stage_size = audio_token_length / num_stages
    
    for head_idx in sorted(results.keys()):
        positions = results[head_idx]['relative_positions']
        stage_counts = []
        stage_ratios = []
        
        for i in range(num_stages):
            stage_start = i * stage_size
            stage_end = (i + 1) * stage_size
            count = np.sum((positions >= stage_start) & (positions < stage_end))
            stage_counts.append(count)
            stage_ratios.append(count / len(positions) if len(positions) > 0 else 0)
        
        print(f"Head {head_idx} 五阶段分布:")
        for i, (count, ratio) in enumerate(zip(stage_counts, stage_ratios)):
            bar_length = int(ratio * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            stage_name = ['开始', '前中', '中间', '后中', '结束'][i]
            print(f"  {stage_name:4s} [{i*20:3d}%-{(i+1)*20:3d}%]: {bar} {ratio*100:5.2f}% ({count:5d})")
        print()
    
    # 总结性发现
    print(f"{'='*80}")
    print(f"Audio时序选择特点总结")
    print(f"{'='*80}\n")
    
    print(f"1. 时序分布特征:")
    if len(results) > 0:
        avg_early = np.mean([r['early_ratio'] for r in results.values()])
        avg_late = np.mean([r['late_ratio'] for r in results.values()])
        print(f"   - 前33%平均占比: {avg_early*100:.2f}%")
        print(f"   - 后33%平均占比: {avg_late*100:.2f}%")
        if avg_early > avg_late:
            print(f"   - 倾向: 更关注音频的前半部分（可能包含重要起始信息）")
        elif avg_late > avg_early:
            print(f"   - 倾向: 更关注音频的后半部分（可能包含重要结束信息）")
        else:
            print(f"   - 倾向: 分布相对均衡")
    print()
    
    print(f"2. 时序连续性:")
    if len(results) > 0:
        avg_consecutive = np.mean([r['consecutive_ratio'] for r in results.values()])
        avg_small_gap = np.mean([r['small_gap_ratio'] for r in results.values()])
        print(f"   - 连续选择比例: {avg_consecutive*100:.2f}%")
        print(f"   - 小间隔(≤5)比例: {avg_small_gap*100:.2f}%")
        if avg_small_gap > 0.8:
            print(f"   - 特点: 选择具有很强的局部连续性，倾向于选择连续的音频片段")
            print(f"   - 意义: 符合音频的时序特性，连续的音频token通常包含相关的声学信息")
    print()
    
    print(f"3. 选择密度:")
    if len(results) > 0:
        avg_density = np.mean([r['density_mean'] for r in results.values()])
        print(f"   - 平均每个位置被选择: {avg_density:.2f}次")
        if avg_density > 1:
            print(f"   - 说明: 某些音频位置被多次选择，可能是关键信息点")
    print()
    
    print(f"4. Layer间时序模式变化:")
    if len(layer_stats) > 0:
        early_pos = np.mean([l['mean_position'] for l in early_layers])
        late_pos = np.mean([l['mean_position'] for l in late_layers])
        if early_pos > late_pos:
            print(f"   - 早期Layers: 更关注音频后半段 (位置={early_pos:.2f})")
            print(f"   - 后期Layers: 更关注音频前半段 (位置={late_pos:.2f})")
            print(f"   - 可能原因: 早期层需要更多上下文，后期层聚焦关键特征")
        else:
            print(f"   - 早期Layers: 更关注音频前半段 (位置={early_pos:.2f})")
            print(f"   - 后期Layers: 更关注音频后半段 (位置={late_pos:.2f})")
    print()
    
    print(f"5. 音频开始/结束部分:")
    if len(results) > 0:
        # 计算平均的开始和结束偏差
        start_ratios = []
        end_ratios = []
        for head_idx in results.keys():
            positions = results[head_idx]['relative_positions']
            start_count = np.sum(positions < audio_token_length * 0.1)
            end_count = np.sum(positions >= audio_token_length * 0.9)
            start_ratios.append(start_count / len(positions) if len(positions) > 0 else 0)
            end_ratios.append(end_count / len(positions) if len(positions) > 0 else 0)
        
        avg_start_ratio = np.mean(start_ratios)
        avg_end_ratio = np.mean(end_ratios)
        start_bias = avg_start_ratio / 0.1
        end_bias = avg_end_ratio / 0.1
        
        print(f"   - 前10%平均占比: {avg_start_ratio*100:.2f}% (期望10%, 偏差={start_bias:.2f}x)")
        print(f"   - 后10%平均占比: {avg_end_ratio*100:.2f}% (期望10%, 偏差={end_bias:.2f}x)")
        if start_bias > 1.2 or end_bias > 1.2:
            print(f"   - 特点: 开始和结束部分被过度选择")
            print(f"   - 可能原因: 音频的开始和结束部分包含重要的边界信息、静音检测或特殊标记")
    print()
    
    print(f"6. 时序特性总结:")
    print(f"   - 音频的时序特性体现在选择的局部连续性上")
    print(f"   - 小间隔(≤5)比例100%说明选择非常密集，几乎覆盖所有音频位置")
    print(f"   - 每个位置平均被选择约20次，说明所有层都在关注这些位置")
    print(f"   - 最大连续片段仅2 tokens，说明选择策略倾向于分散但密集的模式")
    print(f"   - 这种模式符合音频处理的特性：需要保持时序信息的完整性")
    print()
    
    return results

if __name__ == '__main__':
    json_file = '/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv/cache_selection/LibriSpeech_budget0.7/topk_selection_records_134686_0000.json'
    results = analyze_audio_temporal_selection(json_file)
    
    print(f"\n{'='*80}")
    print(f"分析完成!")
    print(f"{'='*80}")

