# Head Importance Score 超参数搜索实验设计

## 概述

本文档描述了对 `alpha`, `beta`, `gamma` 三个权重参数的系统化搜索方法，用于优化 head importance score 的计算公式：

```
S = α·S_hit + β·S_attn + γ·S_cov
```

其中：
- **S_hit**: Hit ratio - top-K 中命中目标单词音频 token 的比例
- **S_attn**: Attention ratio - 目标单词的注意力权重占所有音频注意力权重的比例  
- **S_cov**: Coverage ratio - top-K 命中数占目标单词音频 token 总数的比例

## 实验目标

1. **找到最优权重组合**：使 importance score 能够最好地区分 audio-critical heads 和普通 heads
2. **验证权重稳定性**：在不同数据集/样本上验证最优权重的泛化能力
3. **分析各 metric 贡献**：理解三个 metric 的相对重要性

## 评估指标

### 1. 主要指标：Head Ranking Quality

- **Top-K Head Precision**: 在 top-K importance score 的 heads 中，有多少是真正的 audio-critical heads
- **Ranking Correlation**: Importance score 排序与 ground truth audio-critical heads 的相关性
- **Separation Score**: Audio-critical heads 与普通 heads 的 importance score 分布分离度

### 2. 辅助指标

- **Score Distribution**: Importance score 的分布特征（均值、方差、分位数）
- **Metric Correlation**: 三个 metric 之间的相关性分析
- **Head Coverage**: 被识别为 audio-critical 的 heads 覆盖了多少层

## 搜索策略

### 策略 1: 网格搜索 (Grid Search)

**适用场景**: 参数空间较小，需要全面覆盖

**搜索范围**:
```python
# 基础网格：每个参数从 0 到 2，步长 0.5
alpha_range = [0.0, 0.5, 1.0, 1.5, 2.0]
beta_range = [0.0, 0.5, 1.0, 1.5, 2.0]
gamma_range = [0.0, 0.5, 1.0, 1.5, 2.0]

# 总组合数: 5^3 = 125
```

**优点**: 全面、可重复
**缺点**: 计算量大

### 策略 2: 随机搜索 (Random Search)

**适用场景**: 参数空间较大，需要快速探索

**搜索范围**:
```python
# 随机采样 N 次
n_trials = 50
alpha_range = np.random.uniform(0, 3, n_trials)
beta_range = np.random.uniform(0, 3, n_trials)
gamma_range = np.random.uniform(0, 3, n_trials)
```

**优点**: 计算效率高，可能发现意外的最优解
**缺点**: 可能遗漏某些区域

### 策略 3: 贝叶斯优化 (Bayesian Optimization)

**适用场景**: 需要高效找到全局最优解

**工具**: Optuna, Hyperopt, Scikit-optimize

**优点**: 智能搜索，收敛快
**缺点**: 需要额外依赖

### 策略 4: 分层搜索 (Hierarchical Search)

**步骤**:
1. **阶段1**: 固定两个参数，搜索第三个（找出各参数的重要性）
2. **阶段2**: 在阶段1的基础上，精细搜索最优组合
3. **阶段3**: 在最优区域附近进行局部搜索

## 实验流程

### 步骤 1: 数据准备

```bash
# 使用较小的验证集进行快速搜索
python word_attention_analysis.py \
    --num_samples 50 \
    --topk 32 \
    --output_dir ./search_validation_set
```

### 步骤 2: 运行参数搜索

使用提供的搜索脚本：

```bash
# 网格搜索
python hyperparameter_search.py \
    --search_method grid \
    --alpha_range 0.0,0.5,1.0,1.5,2.0 \
    --beta_range 0.0,0.5,1.0,1.5,2.0 \
    --gamma_range 0.0,0.5,1.0,1.5,2.0 \
    --num_samples 50 \
    --output_dir ./grid_search_results

# 随机搜索
python hyperparameter_search.py \
    --search_method random \
    --n_trials 50 \
    --alpha_range 0,3 \
    --beta_range 0,3 \
    --gamma_range 0,3 \
    --num_samples 50 \
    --output_dir ./random_search_results
```

### 步骤 3: 结果分析

分析脚本会生成：
- 参数组合与评估指标的对应表
- 最优参数组合
- 参数敏感性分析图
- 各 metric 贡献度分析

### 步骤 4: 验证最优参数

在完整数据集上验证最优参数：

```bash
python word_attention_analysis.py \
    --num_samples 0 \
    --alpha <optimal_alpha> \
    --beta <optimal_beta> \
    --gamma <optimal_gamma> \
    --output_dir ./final_results
```

## 实验脚本

### 1. 基础搜索脚本 (`hyperparameter_search.py`)

```python
"""
超参数搜索脚本
支持网格搜索和随机搜索
"""

import argparse
import json
import itertools
import numpy as np
from pathlib import Path
import subprocess
import time
from typing import List, Tuple, Dict

def grid_search(alpha_range: List[float], beta_range: List[float], 
                gamma_range: List[float]) -> List[Tuple[float, float, float]]:
    """生成网格搜索的所有组合"""
    return list(itertools.product(alpha_range, beta_range, gamma_range))

def random_search(n_trials: int, alpha_bounds: Tuple[float, float],
                  beta_bounds: Tuple[float, float], 
                  gamma_bounds: Tuple[float, float]) -> List[Tuple[float, float, float]]:
    """生成随机搜索的组合"""
    combinations = []
    for _ in range(n_trials):
        alpha = np.random.uniform(alpha_bounds[0], alpha_bounds[1])
        beta = np.random.uniform(beta_bounds[0], beta_bounds[1])
        gamma = np.random.uniform(gamma_bounds[0], gamma_bounds[1])
        combinations.append((alpha, beta, gamma))
    return combinations

def evaluate_params(alpha: float, beta: float, gamma: float, 
                   num_samples: int, output_dir: Path) -> Dict:
    """评估一组参数"""
    param_dir = output_dir / f"alpha_{alpha:.2f}_beta_{beta:.2f}_gamma_{gamma:.2f}"
    param_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行分析
    cmd = [
        "python", "word_attention_analysis.py",
        "--num_samples", str(num_samples),
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--gamma", str(gamma),
        "--output_dir", str(param_dir)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 读取结果
    summary_file = param_dir / "word_attention_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        return {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'avg_hit_ratio': summary.get('avg_hit_ratio', 0),
            'avg_attn_ratio': summary.get('avg_attn_ratio', 0),
            'avg_coverage_ratio': summary.get('avg_word_audio_coverage_ratio', 0),
            'success': True
        }
    else:
        return {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'success': False
        }

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Search')
    parser.add_argument('--search_method', type=str, default='grid',
                       choices=['grid', 'random'],
                       help='Search method')
    parser.add_argument('--alpha_range', type=str, default='0.0,0.5,1.0,1.5,2.0',
                       help='Alpha range (comma-separated for grid, min,max for random)')
    parser.add_argument('--beta_range', type=str, default='0.0,0.5,1.0,1.5,2.0',
                       help='Beta range')
    parser.add_argument('--gamma_range', type=str, default='0.0,0.5,1.0,1.5,2.0',
                       help='Gamma range')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of trials for random search')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to analyze per trial')
    parser.add_argument('--output_dir', type=str, default='./hyperparameter_search',
                       help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成参数组合
    if args.search_method == 'grid':
        alpha_list = [float(x) for x in args.alpha_range.split(',')]
        beta_list = [float(x) for x in args.beta_range.split(',')]
        gamma_list = [float(x) for x in args.gamma_range.split(',')]
        combinations = grid_search(alpha_list, beta_list, gamma_list)
    else:  # random
        alpha_bounds = tuple(float(x) for x in args.alpha_range.split(','))
        beta_bounds = tuple(float(x) for x in args.beta_range.split(','))
        gamma_bounds = tuple(float(x) for x in args.gamma_range.split(','))
        combinations = random_search(args.n_trials, alpha_bounds, 
                                   beta_bounds, gamma_bounds)
    
    print(f"Total combinations: {len(combinations)}")
    
    # 评估每个组合
    results = []
    start_time = time.time()
    
    for idx, (alpha, beta, gamma) in enumerate(combinations, 1):
        print(f"\n[{idx}/{len(combinations)}] Testing: α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}")
        result = evaluate_params(alpha, beta, gamma, args.num_samples, output_dir)
        results.append(result)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining = avg_time * (len(combinations) - idx)
        print(f"  Elapsed: {elapsed/60:.1f}min, Remaining: {remaining/60:.1f}min")
    
    # 保存结果
    results_file = output_dir / "search_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 找出最优参数（基于综合指标）
    successful_results = [r for r in results if r.get('success', False)]
    if successful_results:
        # 计算综合得分（可以根据需要调整）
        for r in successful_results:
            r['composite_score'] = (
                r['avg_hit_ratio'] * 0.4 +
                r['avg_attn_ratio'] * 0.3 +
                r['avg_coverage_ratio'] * 0.3
            )
        
        best = max(successful_results, key=lambda x: x['composite_score'])
        print(f"\n{'='*80}")
        print(f"Best Parameters:")
        print(f"  α = {best['alpha']:.2f}")
        print(f"  β = {best['beta']:.2f}")
        print(f"  γ = {best['gamma']:.2f}")
        print(f"  Composite Score = {best['composite_score']:.4f}")
        print(f"{'='*80}")
        
        best_file = output_dir / "best_parameters.json"
        with open(best_file, 'w') as f:
            json.dump(best, f, indent=2)

if __name__ == "__main__":
    main()
```

### 2. 结果分析脚本 (`analyze_search_results.py`)

```python
"""
分析超参数搜索结果
生成可视化图表和统计报告
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import List, Dict

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
    
    # 1. 参数分布分析
    alphas = [r['alpha'] for r in successful]
    betas = [r['beta'] for r in successful]
    gammas = [r['gamma'] for r in successful]
    
    # 2. 指标分析
    hit_ratios = [r.get('avg_hit_ratio', 0) for r in successful]
    attn_ratios = [r.get('avg_attn_ratio', 0) for r in successful]
    coverage_ratios = [r.get('avg_coverage_ratio', 0) for r in successful]
    
    # 3. 找出最优参数
    for r in successful:
        r['composite_score'] = (
            r.get('avg_hit_ratio', 0) * 0.4 +
            r.get('avg_attn_ratio', 0) * 0.3 +
            r.get('avg_coverage_ratio', 0) * 0.3
        )
    
    best = max(successful, key=lambda x: x['composite_score'])
    
    # 4. 生成报告
    report = {
        'total_trials': len(results),
        'successful_trials': len(successful),
        'best_parameters': {
            'alpha': best['alpha'],
            'beta': best['beta'],
            'gamma': best['gamma'],
            'composite_score': best['composite_score'],
            'avg_hit_ratio': best.get('avg_hit_ratio', 0),
            'avg_attn_ratio': best.get('avg_attn_ratio', 0),
            'avg_coverage_ratio': best.get('avg_coverage_ratio', 0)
        },
        'parameter_statistics': {
            'alpha': {
                'mean': float(np.mean(alphas)),
                'std': float(np.std(alphas)),
                'min': float(np.min(alphas)),
                'max': float(np.max(alphas))
            },
            'beta': {
                'mean': float(np.mean(betas)),
                'std': float(np.std(betas)),
                'min': float(np.min(betas)),
                'max': float(np.max(betas))
            },
            'gamma': {
                'mean': float(np.mean(gammas)),
                'std': float(np.std(gammas)),
                'min': float(np.min(gammas)),
                'max': float(np.max(gammas))
            }
        },
        'metric_statistics': {
            'hit_ratio': {
                'mean': float(np.mean(hit_ratios)),
                'std': float(np.std(hit_ratios)),
                'min': float(np.min(hit_ratios)),
                'max': float(np.max(hit_ratios))
            },
            'attn_ratio': {
                'mean': float(np.mean(attn_ratios)),
                'std': float(np.std(attn_ratios)),
                'min': float(np.min(attn_ratios)),
                'max': float(np.max(attn_ratios))
            },
            'coverage_ratio': {
                'mean': float(np.mean(coverage_ratios)),
                'std': float(np.std(coverage_ratios)),
                'min': float(np.min(coverage_ratios)),
                'max': float(np.max(coverage_ratios))
            }
        }
    }
    
    # 保存报告
    report_file = output_dir / "analysis_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # 打印报告
    print(f"\n{'='*80}")
    print(f"Search Results Analysis")
    print(f"{'='*80}")
    print(f"Total trials: {report['total_trials']}")
    print(f"Successful trials: {report['successful_trials']}")
    print(f"\nBest Parameters:")
    print(f"  α = {report['best_parameters']['alpha']:.2f}")
    print(f"  β = {report['best_parameters']['beta']:.2f}")
    print(f"  γ = {report['best_parameters']['gamma']:.2f}")
    print(f"  Composite Score = {report['best_parameters']['composite_score']:.4f}")
    print(f"\nBest Metrics:")
    print(f"  Avg Hit Ratio = {report['best_parameters']['avg_hit_ratio']:.4f}")
    print(f"  Avg Attn Ratio = {report['best_parameters']['avg_attn_ratio']:.4f}")
    print(f"  Avg Coverage Ratio = {report['best_parameters']['avg_coverage_ratio']:.4f}")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description='Analyze Search Results')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to search_results.json')
    parser.add_argument('--output_dir', type=str, default='./analysis_output',
                       help='Output directory for analysis')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = load_results(Path(args.results_file))
    analyze_results(results, output_dir)

if __name__ == "__main__":
    main()
```

## 推荐实验计划

### 阶段 1: 快速探索 (1-2天)

1. **随机搜索** (50 trials)
   - 参数范围: [0, 3] for all
   - 样本数: 50
   - 目标: 快速了解参数空间

2. **初步分析**
   - 识别有希望的参数区域
   - 分析各 metric 的贡献

### 阶段 2: 精细搜索 (2-3天)

1. **网格搜索** (在阶段1的基础上)
   - 缩小搜索范围
   - 更小的步长 (0.25)
   - 样本数: 100

2. **分层搜索**
   - 固定一个参数，搜索其他两个
   - 找出各参数的最优范围

### 阶段 3: 验证 (1天)

1. **交叉验证**
   - 在不同数据子集上验证最优参数
   - 检查稳定性

2. **最终评估**
   - 在完整数据集上运行
   - 生成最终报告

## 预期结果

1. **最优参数组合**: α, β, γ 的具体数值
2. **参数敏感性分析**: 各参数对最终结果的影响程度
3. **Metric 贡献度**: 三个 metric 的相对重要性
4. **稳定性报告**: 最优参数在不同数据上的表现

## 注意事项

1. **计算资源**: 网格搜索可能需要大量计算时间，建议先用小样本集
2. **评估指标**: 可以根据实际需求调整综合得分的权重
3. **Ground Truth**: 如果有已知的 audio-critical heads，可以用来验证
4. **并行化**: 可以并行运行多个参数组合以加速搜索

## 快速开始

```bash
# 1. 快速随机搜索
python hyperparameter_search.py \
    --search_method random \
    --n_trials 20 \
    --alpha_range 0,3 \
    --beta_range 0,3 \
    --gamma_range 0,3 \
    --num_samples 30 \
    --output_dir ./quick_search

# 2. 分析结果
python analyze_search_results.py \
    --results_file ./quick_search/search_results.json \
    --output_dir ./quick_search/analysis

# 3. 使用最优参数运行完整分析
python word_attention_analysis.py \
    --num_samples 0 \
    --alpha <best_alpha> \
    --beta <best_beta> \
    --gamma <best_gamma> \
    --output_dir ./final_results
```

## 扩展实验

### 实验 A: 单参数敏感性分析

固定两个参数为1.0，变化第三个参数，观察影响：

```bash
# 测试 alpha 的影响
for alpha in 0.0 0.5 1.0 1.5 2.0 2.5 3.0; do
    python word_attention_analysis.py \
        --num_samples 50 \
        --alpha $alpha \
        --beta 1.0 \
        --gamma 1.0 \
        --output_dir ./sensitivity_alpha_${alpha}
done
```

### 实验 B: 归一化 vs 非归一化

比较归一化权重 (权重和为1) 和非归一化权重的效果。

### 实验 C: 不同 Top-K 值的影响

在不同 top-k 值下搜索最优参数，验证参数对 top-k 的鲁棒性。

