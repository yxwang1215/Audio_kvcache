"""
分析超参数搜索结果
生成可视化图表和统计报告
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import List, Dict
import seaborn as sns

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_results(results_file: Path) -> List[Dict]:
    """加载搜索结果"""
    with open(results_file, 'r') as f:
        return json.load(f)


def analyze_results(results: List[Dict], output_dir: Path):
    """分析结果并生成报告"""
    successful = [r for r in results if r.get('success', False)]
    
    if not successful:
        print("No successful results found!")
        return
    
    print(f"Analyzing {len(successful)} successful results out of {len(results)} total...")
    
    # 1. 参数分布分析
    alphas = [r['alpha'] for r in successful]
    betas = [r['beta'] for r in successful]
    gammas = [r['gamma'] for r in successful]
    
    # 2. 指标分析
    hit_ratios = [r.get('avg_hit_ratio', 0) for r in successful]
    attn_ratios = [r.get('avg_attn_ratio', 0) for r in successful]
    coverage_ratios = [r.get('avg_coverage_ratio', 0) for r in successful]
    composite_scores = [r.get('composite_score', 0) for r in successful]
    
    # 3. 找出最优参数
    if not composite_scores or all(s == 0 for s in composite_scores):
        # 如果没有 composite_score，计算一个
        for r in successful:
            r['composite_score'] = (
                r.get('avg_hit_ratio', 0) * 0.4 +
                r.get('avg_attn_ratio', 0) * 0.3 +
                r.get('avg_coverage_ratio', 0) * 0.3
            )
        composite_scores = [r.get('composite_score', 0) for r in successful]
    
    best = max(successful, key=lambda x: x.get('composite_score', 0))
    
    # 4. 生成统计报告
    report = {
        'total_trials': len(results),
        'successful_trials': len(successful),
        'success_rate': len(successful) / len(results) if results else 0,
        'best_parameters': {
            'alpha': best['alpha'],
            'beta': best['beta'],
            'gamma': best['gamma'],
            'composite_score': best.get('composite_score', 0),
            'avg_hit_ratio': best.get('avg_hit_ratio', 0),
            'avg_attn_ratio': best.get('avg_attn_ratio', 0),
            'avg_coverage_ratio': best.get('avg_coverage_ratio', 0),
            'importance_score_mean': best.get('importance_score_mean', 0),
            'importance_score_std': best.get('importance_score_std', 0),
        },
        'parameter_statistics': {
            'alpha': {
                'mean': float(np.mean(alphas)),
                'std': float(np.std(alphas)),
                'min': float(np.min(alphas)),
                'max': float(np.max(alphas)),
                'median': float(np.median(alphas))
            },
            'beta': {
                'mean': float(np.mean(betas)),
                'std': float(np.std(betas)),
                'min': float(np.min(betas)),
                'max': float(np.max(betas)),
                'median': float(np.median(betas))
            },
            'gamma': {
                'mean': float(np.mean(gammas)),
                'std': float(np.std(gammas)),
                'min': float(np.min(gammas)),
                'max': float(np.max(gammas)),
                'median': float(np.median(gammas))
            }
        },
        'metric_statistics': {
            'hit_ratio': {
                'mean': float(np.mean(hit_ratios)),
                'std': float(np.std(hit_ratios)),
                'min': float(np.min(hit_ratios)),
                'max': float(np.max(hit_ratios)),
                'median': float(np.median(hit_ratios))
            },
            'attn_ratio': {
                'mean': float(np.mean(attn_ratios)),
                'std': float(np.std(attn_ratios)),
                'min': float(np.min(attn_ratios)),
                'max': float(np.max(attn_ratios)),
                'median': float(np.median(attn_ratios))
            },
            'coverage_ratio': {
                'mean': float(np.mean(coverage_ratios)),
                'std': float(np.std(coverage_ratios)),
                'min': float(np.min(coverage_ratios)),
                'max': float(np.max(coverage_ratios)),
                'median': float(np.median(coverage_ratios))
            },
            'composite_score': {
                'mean': float(np.mean(composite_scores)),
                'std': float(np.std(composite_scores)),
                'min': float(np.min(composite_scores)),
                'max': float(np.max(composite_scores)),
                'median': float(np.median(composite_scores))
            }
        },
        'top_10_parameters': [
            {
                'alpha': r['alpha'],
                'beta': r['beta'],
                'gamma': r['gamma'],
                'composite_score': r.get('composite_score', 0)
            }
            for r in sorted(successful, key=lambda x: x.get('composite_score', 0), reverse=True)[:10]
        ]
    }
    
    # 保存报告
    report_file = output_dir / "analysis_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Saved analysis report to: {report_file}")
    
    # 打印报告
    print(f"\n{'='*80}")
    print(f"Search Results Analysis")
    print(f"{'='*80}")
    print(f"Total trials: {report['total_trials']}")
    print(f"Successful trials: {report['successful_trials']}")
    print(f"Success rate: {report['success_rate']*100:.1f}%")
    print(f"\nBest Parameters:")
    print(f"  α = {report['best_parameters']['alpha']:.2f}")
    print(f"  β = {report['best_parameters']['beta']:.2f}")
    print(f"  γ = {report['best_parameters']['gamma']:.2f}")
    print(f"  Composite Score = {report['best_parameters']['composite_score']:.4f}")
    print(f"\nBest Metrics:")
    print(f"  Avg Hit Ratio = {report['best_parameters']['avg_hit_ratio']:.4f}")
    print(f"  Avg Attn Ratio = {report['best_parameters']['avg_attn_ratio']:.4f}")
    print(f"  Avg Coverage Ratio = {report['best_parameters']['avg_coverage_ratio']:.4f}")
    print(f"  Importance Score Mean = {report['best_parameters']['importance_score_mean']:.4f}")
    print(f"  Importance Score Std = {report['best_parameters']['importance_score_std']:.4f}")
    print(f"{'='*80}")
    
    # 5. 生成可视化图表
    create_visualizations(successful, output_dir)
    
    return report


def create_visualizations(results: List[Dict], output_dir: Path):
    """创建可视化图表"""
    print("\nGenerating visualizations...")
    
    # 提取数据
    alphas = [r['alpha'] for r in results]
    betas = [r['beta'] for r in results]
    gammas = [r['gamma'] for r in results]
    hit_ratios = [r.get('avg_hit_ratio', 0) for r in results]
    attn_ratios = [r.get('avg_attn_ratio', 0) for r in results]
    coverage_ratios = [r.get('avg_coverage_ratio', 0) for r in results]
    composite_scores = [r.get('composite_score', 0) for r in results]
    
    # 1. 参数分布图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(alphas, bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Alpha')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Alpha Distribution')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(betas, bins=20, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Beta')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Beta Distribution')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(gammas, bins=20, edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Gamma')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Gamma Distribution')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved parameter_distributions.png")
    
    # 2. 指标分布图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(hit_ratios, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('Hit Ratio')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Hit Ratio Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(attn_ratios, bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[0, 1].set_xlabel('Attention Ratio')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Attention Ratio Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].hist(coverage_ratios, bins=20, edgecolor='black', alpha=0.7, color='salmon')
    axes[1, 0].set_xlabel('Coverage Ratio')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Coverage Ratio Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(composite_scores, bins=20, edgecolor='black', alpha=0.7, color='gold')
    axes[1, 1].set_xlabel('Composite Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Composite Score Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "metric_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved metric_distributions.png")
    
    # 3. 参数 vs 综合得分散点图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    scatter1 = axes[0].scatter(alphas, composite_scores, c=composite_scores, 
                              cmap='viridis', alpha=0.6, s=50)
    axes[0].set_xlabel('Alpha')
    axes[0].set_ylabel('Composite Score')
    axes[0].set_title('Alpha vs Composite Score')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0])
    
    scatter2 = axes[1].scatter(betas, composite_scores, c=composite_scores, 
                              cmap='viridis', alpha=0.6, s=50)
    axes[1].set_xlabel('Beta')
    axes[1].set_ylabel('Composite Score')
    axes[1].set_title('Beta vs Composite Score')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1])
    
    scatter3 = axes[2].scatter(gammas, composite_scores, c=composite_scores, 
                              cmap='viridis', alpha=0.6, s=50)
    axes[2].set_xlabel('Gamma')
    axes[2].set_ylabel('Composite Score')
    axes[2].set_title('Gamma vs Composite Score')
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_vs_score.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved parameter_vs_score.png")
    
    # 4. 相关性热力图
    if len(results) > 10:
        data_matrix = np.array([
            alphas, betas, gammas, hit_ratios, attn_ratios, coverage_ratios, composite_scores
        ]).T
        
        corr_matrix = np.corrcoef(data_matrix.T)
        labels = ['Alpha', 'Beta', 'Gamma', 'Hit Ratio', 'Attn Ratio', 'Coverage Ratio', 'Composite Score']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, xticklabels=labels, yticklabels=labels)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved correlation_heatmap.png")
    
    print("✓ All visualizations generated!")


def main():
    parser = argparse.ArgumentParser(description='Analyze Hyperparameter Search Results')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to search_results.json')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis (default: same as results file directory)')
    
    args = parser.parse_args()
    
    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_file.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = load_results(results_file)
    analyze_results(results, output_dir)


if __name__ == "__main__":
    main()

