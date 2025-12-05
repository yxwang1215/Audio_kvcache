"""
超参数搜索脚本
支持网格搜索和随机搜索，用于找到最优的 alpha, beta, gamma 权重组合
"""

import argparse
import json
import itertools
import numpy as np
from pathlib import Path
import subprocess
import time
import sys
from typing import List, Tuple, Dict

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def grid_search(alpha_range: List[float], beta_range: List[float], 
                gamma_range: List[float]) -> List[Tuple[float, float, float]]:
    """生成网格搜索的所有组合"""
    return list(itertools.product(alpha_range, beta_range, gamma_range))


def random_search(n_trials: int, alpha_bounds: Tuple[float, float],
                  beta_bounds: Tuple[float, float], 
                  gamma_bounds: Tuple[float, float]) -> List[Tuple[float, float, float]]:
    """生成随机搜索的组合"""
    combinations = []
    np.random.seed(42)  # 设置随机种子以保证可重复性
    for _ in range(n_trials):
        alpha = np.random.uniform(alpha_bounds[0], alpha_bounds[1])
        beta = np.random.uniform(beta_bounds[0], beta_bounds[1])
        gamma = np.random.uniform(gamma_bounds[0], gamma_bounds[1])
        combinations.append((alpha, beta, gamma))
    return combinations


def evaluate_params(alpha: float, beta: float, gamma: float, 
                   num_samples: int, output_dir: Path,
                   script_path: str = None) -> Dict:
    """评估一组参数"""
    param_dir = output_dir / f"alpha_{alpha:.2f}_beta_{beta:.2f}_gamma_{gamma:.2f}"
    param_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定脚本路径
    if script_path is None:
        script_path = str(Path(__file__).parent / "word_attention_analysis.py")
    
    # 运行分析
    cmd = [
        "python", script_path,
        "--num_samples", str(num_samples),
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--gamma", str(gamma),
        "--output_dir", str(param_dir)
    ]
    
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  Error: {result.stderr[:200]}")
        return {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'success': False,
            'error': result.stderr[:500]
        }
    
    # 读取结果
    summary_file = param_dir / "word_attention_summary.json"
    layer_head_file = param_dir / "layer_head_attention_stats.json"
    
    if summary_file.exists() and layer_head_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        with open(layer_head_file, 'r') as f:
            layer_head_stats = json.load(f)
        
        # 计算一些额外的评估指标
        # 1. Importance score 的分布特征
        importance_scores = [stats.get('importance_score', 0) 
                           for stats in layer_head_stats.values()]
        
        # 2. Top heads 的指标
        sorted_heads = sorted(
            layer_head_stats.items(),
            key=lambda x: x[1].get('importance_score', 0),
            reverse=True
        )
        top_10_scores = [stats.get('importance_score', 0) 
                        for _, stats in sorted_heads[:10]]
        
        return {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'avg_hit_ratio': summary.get('avg_hit_ratio', 0),
            'avg_attn_ratio': summary.get('avg_attn_ratio', 0),
            'avg_coverage_ratio': summary.get('avg_word_audio_coverage_ratio', 0),
            'importance_score_mean': float(np.mean(importance_scores)) if importance_scores else 0,
            'importance_score_std': float(np.std(importance_scores)) if importance_scores else 0,
            'top_10_mean': float(np.mean(top_10_scores)) if top_10_scores else 0,
            'score_range': float(np.max(importance_scores) - np.min(importance_scores)) if importance_scores else 0,
            'success': True
        }
    else:
        return {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'success': False,
            'error': 'Output files not found'
        }


def compute_composite_score(result: Dict, weights: Dict = None) -> float:
    """计算综合得分
    
    Args:
        result: 评估结果字典
        weights: 各指标的权重，默认 {'hit': 0.4, 'attn': 0.3, 'cov': 0.3}
    """
    if weights is None:
        weights = {'hit': 0.4, 'attn': 0.3, 'cov': 0.3}
    
    score = (
        result.get('avg_hit_ratio', 0) * weights['hit'] +
        result.get('avg_attn_ratio', 0) * weights['attn'] +
        result.get('avg_coverage_ratio', 0) * weights['cov']
    )
    
    # 也可以考虑 importance score 的分布特征
    # 例如：希望有更好的分离度（更高的标准差和范围）
    separation_bonus = (
        result.get('importance_score_std', 0) * 0.1 +
        result.get('score_range', 0) * 0.05
    )
    
    return score + separation_bonus


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter Search for alpha, beta, gamma weights',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--search_method', type=str, default='grid',
                       choices=['grid', 'random'],
                       help='Search method: grid or random')
    
    # 网格搜索参数
    parser.add_argument('--alpha_range', type=str, default='0.0,0.5,1.0,1.5,2.0',
                       help='Alpha range: comma-separated values for grid search, or min,max for random')
    parser.add_argument('--beta_range', type=str, default='0.0,0.5,1.0,1.5,2.0',
                       help='Beta range')
    parser.add_argument('--gamma_range', type=str, default='0.0,0.5,1.0,1.5,2.0',
                       help='Gamma range')
    
    # 随机搜索参数
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of trials for random search')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # 实验参数
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to analyze per trial')
    parser.add_argument('--output_dir', type=str, default='./hyperparameter_search',
                       help='Output directory for all results')
    parser.add_argument('--script_path', type=str, default=None,
                       help='Path to word_attention_analysis.py (default: same directory)')
    
    # 评估权重
    parser.add_argument('--score_weight_hit', type=float, default=0.4,
                       help='Weight for hit ratio in composite score')
    parser.add_argument('--score_weight_attn', type=float, default=0.3,
                       help='Weight for attention ratio in composite score')
    parser.add_argument('--score_weight_cov', type=float, default=0.3,
                       help='Weight for coverage ratio in composite score')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 生成参数组合
    print(f"\n{'='*80}")
    print(f"Hyperparameter Search Configuration")
    print(f"{'='*80}")
    print(f"Search method: {args.search_method}")
    
    if args.search_method == 'grid':
        alpha_list = [float(x.strip()) for x in args.alpha_range.split(',')]
        beta_list = [float(x.strip()) for x in args.beta_range.split(',')]
        gamma_list = [float(x.strip()) for x in args.gamma_range.split(',')]
        combinations = grid_search(alpha_list, beta_list, gamma_list)
        print(f"Alpha range: {alpha_list}")
        print(f"Beta range: {beta_list}")
        print(f"Gamma range: {gamma_list}")
    else:  # random
        alpha_bounds = tuple(float(x.strip()) for x in args.alpha_range.split(','))
        beta_bounds = tuple(float(x.strip()) for x in args.beta_range.split(','))
        gamma_bounds = tuple(float(x.strip()) for x in args.gamma_range.split(','))
        combinations = random_search(args.n_trials, alpha_bounds, 
                                     beta_bounds, gamma_bounds)
        print(f"Alpha bounds: {alpha_bounds}")
        print(f"Beta bounds: {beta_bounds}")
        print(f"Gamma bounds: {gamma_bounds}")
        print(f"Number of trials: {args.n_trials}")
    
    print(f"Total combinations: {len(combinations)}")
    print(f"Samples per trial: {args.num_samples}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    # 评估每个组合
    results = []
    start_time = time.time()
    
    for idx, (alpha, beta, gamma) in enumerate(combinations, 1):
        print(f"\n[{idx}/{len(combinations)}] Testing: α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}")
        trial_start = time.time()
        
        result = evaluate_params(alpha, beta, gamma, args.num_samples, 
                                output_dir, args.script_path)
        results.append(result)
        
        trial_time = time.time() - trial_start
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining = avg_time * (len(combinations) - idx)
        
        if result.get('success', False):
            print(f"  ✓ Success in {trial_time:.1f}s")
            print(f"    Hit ratio: {result.get('avg_hit_ratio', 0):.4f}")
            print(f"    Attn ratio: {result.get('avg_attn_ratio', 0):.4f}")
            print(f"    Coverage ratio: {result.get('avg_coverage_ratio', 0):.4f}")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
        
        print(f"  Elapsed: {elapsed/60:.1f}min, Remaining: {remaining/60:.1f}min")
    
    # 保存结果
    results_file = output_dir / "search_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to: {results_file}")
    
    # 找出最优参数
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        print("\n✗ No successful results found!")
        return
    
    # 计算综合得分
    score_weights = {
        'hit': args.score_weight_hit,
        'attn': args.score_weight_attn,
        'cov': args.score_weight_cov
    }
    
    for r in successful_results:
        r['composite_score'] = compute_composite_score(r, score_weights)
    
    # 按不同指标排序
    best_by_composite = max(successful_results, key=lambda x: x['composite_score'])
    best_by_hit = max(successful_results, key=lambda x: x.get('avg_hit_ratio', 0))
    best_by_attn = max(successful_results, key=lambda x: x.get('avg_attn_ratio', 0))
    best_by_cov = max(successful_results, key=lambda x: x.get('avg_coverage_ratio', 0))
    
    # 打印结果
    print(f"\n{'='*80}")
    print(f"Search Results Summary")
    print(f"{'='*80}")
    print(f"Total trials: {len(results)}")
    print(f"Successful trials: {len(successful_results)}")
    print(f"Success rate: {len(successful_results)/len(results)*100:.1f}%")
    
    print(f"\n{'='*80}")
    print(f"Best Parameters (by Composite Score):")
    print(f"{'='*80}")
    print(f"  α = {best_by_composite['alpha']:.2f}")
    print(f"  β = {best_by_composite['beta']:.2f}")
    print(f"  γ = {best_by_composite['gamma']:.2f}")
    print(f"  Composite Score = {best_by_composite['composite_score']:.4f}")
    print(f"  Avg Hit Ratio = {best_by_composite.get('avg_hit_ratio', 0):.4f}")
    print(f"  Avg Attn Ratio = {best_by_composite.get('avg_attn_ratio', 0):.4f}")
    print(f"  Avg Coverage Ratio = {best_by_composite.get('avg_coverage_ratio', 0):.4f}")
    print(f"  Importance Score Mean = {best_by_composite.get('importance_score_mean', 0):.4f}")
    print(f"  Importance Score Std = {best_by_composite.get('importance_score_std', 0):.4f}")
    
    print(f"\nTop 5 by Composite Score:")
    top_5 = sorted(successful_results, key=lambda x: x['composite_score'], reverse=True)[:5]
    for i, r in enumerate(top_5, 1):
        print(f"  {i}. α={r['alpha']:.2f}, β={r['beta']:.2f}, γ={r['gamma']:.2f} "
              f"(score={r['composite_score']:.4f})")
    
    # 保存最优参数
    best_file = output_dir / "best_parameters.json"
    best_params = {
        'best_by_composite': {
            'alpha': best_by_composite['alpha'],
            'beta': best_by_composite['beta'],
            'gamma': best_by_composite['gamma'],
            **{k: v for k, v in best_by_composite.items() 
               if k not in ['alpha', 'beta', 'gamma', 'success']}
        },
        'best_by_hit_ratio': {
            'alpha': best_by_hit['alpha'],
            'beta': best_by_hit['beta'],
            'gamma': best_by_hit['gamma']
        },
        'best_by_attn_ratio': {
            'alpha': best_by_attn['alpha'],
            'beta': best_by_attn['beta'],
            'gamma': best_by_attn['gamma']
        },
        'best_by_coverage_ratio': {
            'alpha': best_by_cov['alpha'],
            'beta': best_by_cov['beta'],
            'gamma': best_by_cov['gamma']
        },
        'score_weights': score_weights
    }
    
    with open(best_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\n✓ Saved best parameters to: {best_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

