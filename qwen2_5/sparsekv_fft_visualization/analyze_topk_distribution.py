"""
分析 SAS 前后 topk 选择的分布变化情况
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from collections import defaultdict, Counter
from pathlib import Path
import argparse

# Set ACL-style plotting parameters
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 11
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.titlesize'] = 12
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.1

# Use matplotlib style if seaborn is not available
if not HAS_SEABORN:
    plt.style.use('default')
    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.3

def load_stats(filepath):
    """Load statistics from JSON file"""
    print(f"Loading statistics from {filepath}...")
    with open(filepath, 'r') as f:
        stats = json.load(f)
    print(f"Loaded statistics for {len(stats)} layer-head combinations")
    return stats

def analyze_topk_distribution(stats):
    """
    分析 SAS 前后 topk 选择的分布变化
    
    返回:
        - position_distribution: 每个位置被选中的频率分布
        - selection_frequency: 每个token位置被选中的次数统计
        - distribution_shift: 分布的变化情况
    """
    print("\n=== Analyzing TopK Distribution Changes ===")
    
    # 统计每个位置被选中的频率（在所有样本中）
    before_position_freq = defaultdict(int)  # {position: count}
    after_position_freq = defaultdict(int)
    
    # 统计每个样本的选择情况
    all_before_selections = []  # 所有样本的before选择
    all_after_selections = []   # 所有样本的after选择
    
    total_samples = 0
    
    for key, data in stats.items():
        parts = key.split('_')
        layer_idx = int(parts[1])
        head_idx = int(parts[3])
        
        before_indices_list = data['before_fft_topk_indices']
        after_indices_list = data['after_fft_topk_indices']
        
        for before_indices, after_indices in zip(before_indices_list, after_indices_list):
            total_samples += 1
            
            # 统计位置频率（每个位置被选中的次数）
            for pos in before_indices:
                before_position_freq[int(pos)] += 1
            for pos in after_indices:
                after_position_freq[int(pos)] += 1
            
            # 保存选择情况
            all_before_selections.append(set(before_indices))
            all_after_selections.append(set(after_indices))
    
    print(f"Total samples analyzed: {total_samples}")
    print(f"Unique positions in before: {len(before_position_freq)}")
    print(f"Unique positions in after: {len(after_position_freq)}")
    print(f"Note: Y-axis will show average selection probability (0-1) instead of cumulative frequency")
    
    return {
        'before_position_freq': dict(before_position_freq),
        'after_position_freq': dict(after_position_freq),
        'all_before_selections': all_before_selections,
        'all_after_selections': all_after_selections,
        'total_samples': total_samples
    }

def plot_position_distribution(analysis, output_dir):
    """绘制位置分布图 - 显示每个位置被选中的平均概率（而不是累积频率）"""
    before_freq = analysis['before_position_freq']
    after_freq = analysis['after_position_freq']
    total_samples = analysis['total_samples']
    
    # 获取所有位置
    all_positions = set(before_freq.keys()) | set(after_freq.keys())
    max_pos = max(all_positions) if all_positions else 0
    
    # 创建位置频率数组
    before_counts = np.zeros(max_pos + 1)
    after_counts = np.zeros(max_pos + 1)
    
    for pos, count in before_freq.items():
        before_counts[int(pos)] = count
    for pos, count in after_freq.items():
        after_counts[int(pos)] = count
    
    # 计算每个位置被选中的平均概率（在所有样本中）
    # 这样y轴就是0-1之间的值，而不是累积频率范围
    before_avg_prob = before_counts / total_samples if total_samples > 0 else before_counts
    after_avg_prob = after_counts / total_samples if total_samples > 0 else after_counts
    
    # 归一化为相对概率分布（所有位置的概率和为1）
    before_total = before_counts.sum()
    after_total = after_counts.sum()
    before_rel_prob = before_counts / before_total if before_total > 0 else before_counts
    after_rel_prob = after_counts / after_total if after_total > 0 else after_counts
    
    # 只保留非零值的位置（被选中的位置）
    before_nonzero_mask = before_avg_prob > 0
    after_nonzero_mask = after_avg_prob > 0
    
    before_positions = np.arange(max_pos + 1)[before_nonzero_mask]
    before_values = before_avg_prob[before_nonzero_mask]
    after_positions = np.arange(max_pos + 1)[after_nonzero_mask]
    after_values = after_avg_prob[after_nonzero_mask]
    
    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: 平均选择概率 - 使用散点图只显示被选中的位置
    if len(before_positions) > 0:
        axes[0].scatter(before_positions, before_values, s=20, c='blue', 
                       label='Before FFT', alpha=0.6, edgecolors='blue', linewidths=0.5)
    if len(after_positions) > 0:
        axes[0].scatter(after_positions, after_values, s=20, c='red', 
                       label='After FFT', alpha=0.6, edgecolors='red', linewidths=0.5, marker='s')
    axes[0].set_xlabel('Token Position', fontsize=10)
    axes[0].set_ylabel('Average Selection Probability', fontsize=10)
    axes[0].set_title('TopK Selection: Average Probability by Position (only selected positions shown)', fontsize=11)
    max_val = max(before_values.max() if len(before_values) > 0 else 0, 
                  after_values.max() if len(after_values) > 0 else 0)
    axes[0].set_ylim([0, max(1.0, max_val * 1.1)])
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: 相对概率分布 - 只显示非零值
    before_rel_nonzero_mask = before_rel_prob > 0
    after_rel_nonzero_mask = after_rel_prob > 0
    before_rel_positions = np.arange(max_pos + 1)[before_rel_nonzero_mask]
    before_rel_values = before_rel_prob[before_rel_nonzero_mask]
    after_rel_positions = np.arange(max_pos + 1)[after_rel_nonzero_mask]
    after_rel_values = after_rel_prob[after_rel_nonzero_mask]
    
    if len(before_rel_positions) > 0:
        axes[1].scatter(before_rel_positions, before_rel_values, s=20, c='blue', 
                       label='Before FFT', alpha=0.6, edgecolors='blue', linewidths=0.5)
    if len(after_rel_positions) > 0:
        axes[1].scatter(after_rel_positions, after_rel_values, s=20, c='red', 
                       label='After FFT', alpha=0.6, edgecolors='red', linewidths=0.5, marker='s')
    axes[1].set_xlabel('Token Position', fontsize=10)
    axes[1].set_ylabel('Relative Selection Probability', fontsize=10)
    axes[1].set_title('TopK Selection: Relative Probability Distribution (only selected positions shown)', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: 差异（After - Before）- 只显示有差异的位置
    diff = after_avg_prob - before_avg_prob
    diff_nonzero_mask = np.abs(diff) > 1e-6  # 只显示有显著差异的位置
    diff_positions = np.arange(max_pos + 1)[diff_nonzero_mask]
    diff_values = diff[diff_nonzero_mask]
    
    # 分别显示增加和减少的位置
    if len(diff_values) > 0:
        increase_mask = diff_values > 0
        decrease_mask = diff_values < 0
        
        if np.any(increase_mask):
            axes[2].scatter(diff_positions[increase_mask], diff_values[increase_mask], 
                           s=20, c='red', alpha=0.6, edgecolors='red', linewidths=0.5, label='Increased')
        if np.any(decrease_mask):
            axes[2].scatter(diff_positions[decrease_mask], diff_values[decrease_mask], 
                           s=20, c='blue', alpha=0.6, edgecolors='blue', linewidths=0.5, label='Decreased')
    
    axes[2].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    axes[2].set_xlabel('Token Position', fontsize=10)
    axes[2].set_ylabel('Probability Difference (After - Before)', fontsize=10)
    axes[2].set_title('Distribution Shift: Average Selection Probability Difference (only changed positions shown)', fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'topk_position_distribution.png'
    plt.savefig(output_path)
    print(f"Saved position distribution plot to {output_path}")
    print(f"  - Now showing only selected positions as points (not continuous lines)")
    print(f"  - Y-axis shows probability (0-1) instead of cumulative frequency")
    plt.close()

def plot_selection_heatmap(analysis, output_dir, max_samples=10):
    """绘制选择热力图"""
    before_selections = analysis['all_before_selections']
    after_selections = analysis['all_after_selections']
    
    # 限制样本数量
    num_samples = min(max_samples, len(before_selections))
    before_selections = before_selections[:num_samples]
    after_selections = after_selections[:num_samples]
    
    # 获取所有位置
    all_positions = set()
    for sel in before_selections + after_selections:
        all_positions.update(sel)
    all_positions = sorted(all_positions)
    max_pos = max(all_positions) if all_positions else 0
    
    # 创建热力图数据
    before_matrix = np.zeros((num_samples, max_pos + 1))
    after_matrix = np.zeros((num_samples, max_pos + 1))
    
    for i, (before_sel, after_sel) in enumerate(zip(before_selections, after_selections)):
        for pos in before_sel:
            before_matrix[i, int(pos)] = 1
        for pos in after_sel:
            after_matrix[i, int(pos)] = 1
    
    # 创建图形 - 保持原尺寸
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    
    # Before SAS
    im1 = axes[0].imshow(before_matrix, aspect='auto', cmap='Blues', interpolation='nearest')
    axes[0].set_xlabel('Token Position', fontsize=28)  # 进一步增大
    axes[0].set_ylabel('Sample Index', fontsize=28)  # 进一步增大
    axes[0].set_title('TopK Selection Before SAS (Blue = Selected)', fontsize=24)
    # 增大刻度标签字体
    axes[0].tick_params(axis='x', labelsize=24)  # Token Position的数字 (1000, 3000等)
    axes[0].tick_params(axis='y', labelsize=24)  # Sample Index的数字
    # 增大colorbar字体
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('', fontsize=24)  # colorbar标签
    cbar1.ax.tick_params(labelsize=24)  # colorbar刻度 (0.0 - 1.0)
    
    # After SAS
    im2 = axes[1].imshow(after_matrix, aspect='auto', cmap='Reds', interpolation='nearest')
    axes[1].set_xlabel('Token Position', fontsize=28)  # 进一步增大
    axes[1].set_ylabel('Sample Index', fontsize=28)  # 进一步增大
    axes[1].set_title('TopK Selection After SAS (Red = Selected)', fontsize=24)
    # 增大刻度标签字体
    axes[1].tick_params(axis='x', labelsize=24)  # Token Position的数字 (1000, 3000等)
    axes[1].tick_params(axis='y', labelsize=24)  # Sample Index的数字
    # 增大colorbar字体
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('', fontsize=24)  # colorbar标签
    cbar2.ax.tick_params(labelsize=24)  # colorbar刻度 (0.0 - 1.0)
    
    plt.tight_layout()
    
    # 保存为PNG, PDF, SVG
    base_path = output_dir / 'topk_selection_heatmap'
    
    # PNG
    output_path_png = base_path.with_suffix('.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"Saved selection heatmap to {output_path_png}")
    
    # PDF
    output_path_pdf = base_path.with_suffix('.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Saved selection heatmap to {output_path_pdf}")
    
    # SVG
    output_path_svg = base_path.with_suffix('.svg')
    plt.savefig(output_path_svg, bbox_inches='tight')
    print(f"Saved selection heatmap to {output_path_svg}")
    
    plt.close()

def plot_distribution_statistics(analysis, output_dir):
    """绘制分布统计信息"""
    before_freq = analysis['before_position_freq']
    after_freq = analysis['after_position_freq']
    
    before_counts = np.array(list(before_freq.values()))
    after_counts = np.array(list(after_freq.values()))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: 频率分布直方图
    axes[0, 0].hist(before_counts, bins=30, color='blue', alpha=0.6, label='Before SAS', edgecolor='black')
    axes[0, 0].hist(after_counts, bins=30, color='red', alpha=0.6, label='After SAS', edgecolor='black')
    axes[0, 0].set_xlabel('Selection Frequency', fontsize=10)
    axes[0, 0].set_ylabel('Number of Positions', fontsize=10)
    axes[0, 0].set_title('Distribution of Selection Frequencies', fontsize=11)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: 累积分布
    sorted_before = np.sort(before_counts)[::-1]
    sorted_after = np.sort(after_counts)[::-1]
    axes[0, 1].plot(np.arange(len(sorted_before)), np.cumsum(sorted_before), 'b-', 
                   label='Before SAS', linewidth=2)
    axes[0, 1].plot(np.arange(len(sorted_after)), np.cumsum(sorted_after), 'r-', 
                   label='After SAS', linewidth=2)
    axes[0, 1].set_xlabel('Position Rank (by frequency)', fontsize=10)
    axes[0, 1].set_ylabel('Cumulative Frequency', fontsize=10)
    axes[0, 1].set_title('Cumulative Distribution of Selection Frequencies', fontsize=11)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: 位置范围分析
    before_positions = np.array(list(before_freq.keys()))
    after_positions = np.array(list(after_freq.keys()))
    
    # 将位置分成区间
    if len(before_positions) > 0 and len(after_positions) > 0:
        max_pos = max(before_positions.max(), after_positions.max())
        num_bins = 20
        bin_edges = np.linspace(0, max_pos, num_bins + 1)
        
        before_hist, _ = np.histogram(before_positions, bins=bin_edges, 
                                      weights=[before_freq[p] for p in before_positions])
        after_hist, _ = np.histogram(after_positions, bins=bin_edges,
                                     weights=[after_freq[p] for p in after_positions])
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = bin_edges[1] - bin_edges[0]
        
        axes[1, 0].bar(bin_centers - width/4, before_hist, width/2, 
                      label='Before SAS', color='blue', alpha=0.7)
        axes[1, 0].bar(bin_centers + width/4, after_hist, width/2,
                      label='After SAS', color='red', alpha=0.7)
        axes[1, 0].set_xlabel('Token Position Range', fontsize=10)
        axes[1, 0].set_ylabel('Total Selection Frequency', fontsize=10)
        axes[1, 0].set_title('Selection Frequency by Position Range', fontsize=11)
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: 统计摘要
    axes[1, 1].axis('off')
    stats_text = f"""
    Distribution Statistics:
    
    Before SAS:
      - Mean frequency: {before_counts.mean():.2f}
      - Std frequency: {before_counts.std():.2f}
      - Max frequency: {before_counts.max():.0f}
      - Unique positions: {len(before_freq)}
    
    After SAS:
      - Mean frequency: {after_counts.mean():.2f}
      - Std frequency: {after_counts.std():.2f}
      - Max frequency: {after_counts.max():.0f}
      - Unique positions: {len(after_freq)}
    
    Change:
      - Mean change: {after_counts.mean() - before_counts.mean():.2f}
      - Std change: {after_counts.std() - before_counts.std():.2f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, 
                    verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    output_path = output_dir / 'topk_distribution_statistics.png'
    plt.savefig(output_path)
    print(f"Saved distribution statistics to {output_path}")
    plt.close()

def print_summary(stats, analysis):
    """打印分析摘要"""
    print("\n" + "="*80)
    print("TopK Distribution Analysis Summary")
    print("="*80)
    
    print(f"\nTotal samples: {analysis['total_samples']}")
    print(f"Unique positions (before): {len(analysis['before_position_freq'])}")
    print(f"Unique positions (after): {len(analysis['after_position_freq'])}")
    
    before_freq = analysis['before_position_freq']
    after_freq = analysis['after_position_freq']
    
    if before_freq and after_freq:
        before_counts = np.array(list(before_freq.values()))
        after_counts = np.array(list(after_freq.values()))
        
        print(f"\nSelection Frequency Statistics:")
        print(f"  Before SAS - Mean: {before_counts.mean():.2f}, Std: {before_counts.std():.2f}")
        print(f"  After SAS - Mean: {after_counts.mean():.2f}, Std: {after_counts.std():.2f}")
        
        # 找出最常被选中的位置
        top_before = sorted(before_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        top_after = sorted(after_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\nTop 10 Most Selected Positions (Before SAS):")
        for pos, count in top_before:
            print(f"  Position {pos}: {count} times")
        
        print(f"\nTop 10 Most Selected Positions (After SAS):")
        for pos, count in top_after:
            print(f"  Position {pos}: {count} times")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Analyze TopK distribution changes')
    parser.add_argument('--stats-file', type=str, required=True,
                       help='Path to JSON file containing SAS statistics')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for plots')
    parser.add_argument('--max-samples', type=int, default=10,
                       help='Maximum number of samples for heatmap')
    
    args = parser.parse_args()
    
    # Load statistics
    stats = load_stats(args.stats_file)
    
    # Analyze distribution
    analysis = analyze_topk_distribution(stats)
    
    # Print summary
    print_summary(stats, analysis)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print(f"\nGenerating plots in {output_dir}...")
    plot_position_distribution(analysis, output_dir)
    plot_selection_heatmap(analysis, output_dir, max_samples=args.max_samples)
    plot_distribution_statistics(analysis, output_dir)
    
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
    
    
'''
python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_visualization/analyze_topk_distribution.py \
    --stats-file /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_visualization/results/LibriSpeech_budget0.5_fft_saliency_cutoff0.15_alpha0.6/fft_topk_stats.json \
    --output-dir /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_visualization/results/LibriSpeech_budget0.5_fft_saliency_cutoff0.15_alpha0.6 \
    --max-samples 5
'''