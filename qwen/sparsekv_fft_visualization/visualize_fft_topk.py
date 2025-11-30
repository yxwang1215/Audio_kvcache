"""
Visualization script for comparing topk selection before and after FFT filtering.
ACL-style plots showing the differences in attention score selection.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import seaborn as sns
from pathlib import Path
from collections import defaultdict
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

# Use seaborn style for cleaner plots
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_stats(filepath):
    """Load statistics from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def analyze_topk_differences(stats):
    """Analyze differences in topk selection before and after FFT"""
    results = defaultdict(lambda: {
        'jaccard_similarities': [],
        'overlap_ratios': [],
        'unique_before': [],
        'unique_after': [],
        'head_budgets': [],
    })
    
    for key, data in stats.items():
        # Parse layer and head from key
        parts = key.split('_')
        layer_idx = int(parts[1])
        head_idx = int(parts[3])
        
        before_indices_list = data['before_fft_topk_indices']
        after_indices_list = data['after_fft_topk_indices']
        head_budgets = data['head_budgets']
        
        for before_indices, after_indices, budget in zip(
            before_indices_list, after_indices_list, head_budgets
        ):
            before_set = set(before_indices)
            after_set = set(after_indices)
            
            # Calculate metrics
            jaccard = calculate_jaccard_similarity(before_set, after_set)
            overlap = len(before_set & after_set)
            overlap_ratio = overlap / len(before_set) if len(before_set) > 0 else 0.0
            unique_before = len(before_set - after_set)
            unique_after = len(after_set - before_set)
            
            results[key]['jaccard_similarities'].append(jaccard)
            results[key]['overlap_ratios'].append(overlap_ratio)
            results[key]['unique_before'].append(unique_before)
            results[key]['unique_after'].append(unique_after)
            results[key]['head_budgets'].append(budget)
    
    return results

def plot_layer_head_comparison(stats, output_dir):
    """Plot comparison of topk selection for each layer and head"""
    analysis = analyze_topk_differences(stats)
    
    # Group by layer
    layers = set()
    for key in analysis.keys():
        parts = key.split('_')
        layers.add(int(parts[1]))
    layers = sorted(layers)
    
    num_layers = len(layers)
    if num_layers == 0:
        print("No data to plot")
        return
    
    # Create subplots: one per layer
    fig, axes = plt.subplots(num_layers, 1, figsize=(12, 4 * num_layers))
    if num_layers == 1:
        axes = [axes]
    
    for layer_idx, ax in zip(layers, axes):
        # Get all heads for this layer
        layer_keys = [k for k in analysis.keys() if f"layer_{layer_idx}_" in k]
        if not layer_keys:
            continue
        
        # Extract head indices
        head_indices = []
        for key in layer_keys:
            parts = key.split('_')
            head_indices.append(int(parts[3]))
        head_indices = sorted(head_indices)
        
        # Calculate average metrics per head
        jaccard_means = []
        overlap_means = []
        unique_before_means = []
        unique_after_means = []
        budget_means = []
        
        for head_idx in head_indices:
            key = f"layer_{layer_idx}_head_{head_idx}"
            if key in analysis:
                data = analysis[key]
                jaccard_means.append(np.mean(data['jaccard_similarities']))
                overlap_means.append(np.mean(data['overlap_ratios']))
                unique_before_means.append(np.mean(data['unique_before']))
                unique_after_means.append(np.mean(data['unique_after']))
                budget_means.append(np.mean(data['head_budgets']))
        
        # Plot
        x = np.arange(len(head_indices))
        width = 0.35
        
        ax2 = ax.twinx()
        
        # Bar plot for Jaccard similarity
        bars1 = ax.bar(x - width/2, jaccard_means, width, label='Jaccard Similarity', 
                       color='steelblue', alpha=0.7)
        bars2 = ax.bar(x + width/2, overlap_means, width, label='Overlap Ratio', 
                       color='coral', alpha=0.7)
        
        # Line plot for unique counts
        line1 = ax2.plot(x, unique_before_means, 'o-', color='green', 
                        label='Unique Before FFT', linewidth=2, markersize=6)
        line2 = ax2.plot(x, unique_after_means, 's-', color='red', 
                        label='Unique After FFT', linewidth=2, markersize=6)
        
        ax.set_xlabel('Head Index', fontsize=10)
        ax.set_ylabel('Similarity / Overlap Ratio', fontsize=10, color='black')
        ax2.set_ylabel('Unique Token Count', fontsize=10, color='black')
        ax.set_title(f'Layer {layer_idx}: TopK Selection Comparison (Before vs After FFT)', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(head_indices)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / 'layer_head_comparison.png'
    plt.savefig(output_path)
    print(f"Saved layer-head comparison plot to {output_path}")
    plt.close()

def plot_attention_score_comparison(stats, output_dir, max_samples=5):
    """Plot attention scores before and after FFT for selected samples"""
    # Select a few representative layer-head combinations
    selected_keys = list(stats.keys())[:max_samples]
    
    for key in selected_keys:
        data = stats[key]
        parts = key.split('_')
        layer_idx = int(parts[1])
        head_idx = int(parts[3])
        
        # Get first sample
        if len(data['before_fft_scores']) == 0:
            continue
        
        before_scores = np.array(data['before_fft_scores'][0])
        after_scores = np.array(data['after_fft_scores'][0])
        before_topk = np.array(data['before_fft_topk_indices'][0])
        after_topk = np.array(data['after_fft_topk_indices'][0])
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Plot 1: Attention scores over sequence
        seq_len = len(before_scores)
        x = np.arange(seq_len)
        
        axes[0].plot(x, before_scores, 'b-', label='Before FFT', linewidth=1.5, alpha=0.7)
        axes[0].plot(x, after_scores, 'r-', label='After FFT', linewidth=1.5, alpha=0.7)
        
        # Mark selected topk positions
        axes[0].scatter(before_topk, before_scores[before_topk], 
                       color='blue', marker='o', s=50, alpha=0.6, 
                       label='TopK Before FFT', zorder=5)
        axes[0].scatter(after_topk, after_scores[after_topk], 
                       color='red', marker='s', s=50, alpha=0.6, 
                       label='TopK After FFT', zorder=5)
        
        axes[0].set_xlabel('Token Position', fontsize=10)
        axes[0].set_ylabel('Attention Score', fontsize=10)
        axes[0].set_title(f'Layer {layer_idx}, Head {head_idx}: Attention Scores Before/After FFT', 
                         fontsize=11)
        axes[0].legend(loc='best', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: TopK selection comparison
        before_set = set(before_topk)
        after_set = set(after_topk)
        common = before_set & after_set
        only_before = before_set - after_set
        only_after = after_set - before_set
        
        # Create visualization
        y_pos = np.arange(seq_len)
        colors = np.zeros(seq_len)
        colors[list(common)] = 1  # Green for common
        colors[list(only_before)] = 2  # Blue for only before
        colors[list(only_after)] = 3  # Red for only after
        
        scatter = axes[1].scatter(x, y_pos, c=colors, cmap='RdYlGn', 
                                 s=30, alpha=0.6, edgecolors='black', linewidths=0.5)
        
        axes[1].set_xlabel('Token Position', fontsize=10)
        axes[1].set_ylabel('Token Index', fontsize=10)
        axes[1].set_title(f'TopK Selection Comparison: Common (Green), Only Before (Blue), Only After (Red)', 
                         fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.6, label=f'Common ({len(common)})'),
            Patch(facecolor='blue', alpha=0.6, label=f'Only Before ({len(only_before)})'),
            Patch(facecolor='red', alpha=0.6, label=f'Only After ({len(only_after)})'),
        ]
        axes[1].legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        output_path = output_dir / f'attention_comparison_layer{layer_idx}_head{head_idx}.png'
        plt.savefig(output_path)
        print(f"Saved attention comparison plot to {output_path}")
        plt.close()

def plot_statistics_summary(stats, output_dir):
    """Plot summary statistics across all layers and heads"""
    analysis = analyze_topk_differences(stats)
    
    # Collect all metrics
    all_jaccards = []
    all_overlaps = []
    all_unique_before = []
    all_unique_after = []
    all_budgets = []
    layer_head_labels = []
    
    for key, data in analysis.items():
        parts = key.split('_')
        layer_idx = int(parts[1])
        head_idx = int(parts[3])
        
        for jaccard, overlap, u_before, u_after, budget in zip(
            data['jaccard_similarities'],
            data['overlap_ratios'],
            data['unique_before'],
            data['unique_after'],
            data['head_budgets']
        ):
            all_jaccards.append(jaccard)
            all_overlaps.append(overlap)
            all_unique_before.append(u_before)
            all_unique_after.append(u_after)
            all_budgets.append(budget)
            layer_head_labels.append(f"L{layer_idx}H{head_idx}")
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Jaccard similarity distribution
    axes[0, 0].hist(all_jaccards, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(all_jaccards), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(all_jaccards):.3f}')
    axes[0, 0].set_xlabel('Jaccard Similarity', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].set_title('Distribution of Jaccard Similarity', fontsize=11)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Overlap ratio distribution
    axes[0, 1].hist(all_overlaps, bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(all_overlaps), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(all_overlaps):.3f}')
    axes[0, 1].set_xlabel('Overlap Ratio', fontsize=10)
    axes[0, 1].set_ylabel('Frequency', fontsize=10)
    axes[0, 1].set_title('Distribution of Overlap Ratio', fontsize=11)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Unique tokens comparison
    axes[1, 0].scatter(all_unique_before, all_unique_after, alpha=0.5, s=30, edgecolors='black', linewidths=0.5)
    axes[1, 0].plot([0, max(max(all_unique_before), max(all_unique_after))], 
                   [0, max(max(all_unique_before), max(all_unique_after))], 
                   'r--', linewidth=2, label='y=x')
    axes[1, 0].set_xlabel('Unique Tokens Before FFT', fontsize=10)
    axes[1, 0].set_ylabel('Unique Tokens After FFT', fontsize=10)
    axes[1, 0].set_title('Unique Token Count Comparison', fontsize=11)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Budget vs Jaccard similarity
    axes[1, 1].scatter(all_budgets, all_jaccards, alpha=0.5, s=30, edgecolors='black', linewidths=0.5)
    axes[1, 1].set_xlabel('Head Budget (K)', fontsize=10)
    axes[1, 1].set_ylabel('Jaccard Similarity', fontsize=10)
    axes[1, 1].set_title('Budget vs TopK Similarity', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'statistics_summary.png'
    plt.savefig(output_path)
    print(f"Saved statistics summary to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize FFT topk selection differences')
    parser.add_argument('--stats-file', type=str, required=True,
                       help='Path to JSON file containing FFT statistics')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for plots')
    parser.add_argument('--max-samples', type=int, default=5,
                       help='Maximum number of samples to plot for attention comparison')
    
    args = parser.parse_args()
    
    # Load statistics
    print(f"Loading statistics from {args.stats_file}...")
    stats = load_stats(args.stats_file)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating plots in {output_dir}...")
    
    # Generate plots
    plot_statistics_summary(stats, output_dir)
    plot_layer_head_comparison(stats, output_dir)
    plot_attention_score_comparison(stats, output_dir, max_samples=args.max_samples)
    
    print("Visualization complete!")

if __name__ == '__main__':
    main()

