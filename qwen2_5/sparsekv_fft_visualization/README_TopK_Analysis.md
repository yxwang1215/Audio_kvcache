# FFT TopK 统计分析说明

## 1. fft_topk_stats.json 中保存的信息

JSON文件按层和头组织，每个 `layer_{layer_idx}_head_{head_idx}` 键包含以下字段：

```json
{
  "layer_0_head_0": {
    "before_fft_scores": [
      [0.1, 0.2, 0.3, ...],  // 每个样本的FFT前的attention scores (序列长度)
      [0.15, 0.25, 0.35, ...],
      ...
    ],
    "after_fft_scores": [
      [0.12, 0.22, 0.32, ...],  // 每个样本的FFT后的attention scores (序列长度)
      [0.17, 0.27, 0.37, ...],
      ...
    ],
    "before_fft_topk_indices": [
      [10, 25, 30, ...],  // 每个样本FFT前选择的topk位置索引
      [15, 28, 35, ...],
      ...
    ],
    "after_fft_topk_indices": [
      [12, 24, 31, ...],  // 每个样本FFT后选择的topk位置索引
      [16, 27, 36, ...],
      ...
    ],
    "head_budgets": [1967, 1931, ...],  // 每个样本该head的budget (k值)
    "sample_ids": [0, 1, 2, ...]  // 样本ID
  },
  "layer_0_head_1": { ... },
  ...
}
```

### 字段说明：

1. **before_fft_scores**: FFT处理前的attention scores，形状为 `[num_samples, seq_len]`
   - 每个元素是一个数组，表示该样本在该head的所有token位置的attention score

2. **after_fft_scores**: FFT处理后的attention scores，形状为 `[num_samples, seq_len]`
   - 每个元素是一个数组，表示FFT滤波后的attention scores

3. **before_fft_topk_indices**: FFT前根据scores选择的topk位置索引
   - 每个元素是一个数组，包含被选中的token位置索引
   - 长度等于该样本该head的budget (k值)

4. **after_fft_topk_indices**: FFT后根据scores选择的topk位置索引
   - 每个元素是一个数组，包含被选中的token位置索引
   - 长度等于该样本该head的budget (k值)

5. **head_budgets**: 每个样本该head的budget值（即topk的k值）
   - 不同head的budget可能不同（根据head score分配）

6. **sample_ids**: 样本ID，用于追踪是哪个样本的数据

## 2. 如何比较 FFT 前后 topk 分布变化

### 方法1: 使用现有的可视化脚本

运行现有的可视化脚本：
```bash
python visualize_fft_topk.py \
    --stats-file /path/to/fft_topk_stats.json \
    --output-dir /path/to/output \
    --max-samples 5
```

这会生成：
- `statistics_summary.png`: 总体统计摘要
- `layer_head_comparison.png`: 每层每头的对比
- `attention_comparison_layer{X}_head{Y}.png`: 详细的attention score对比

### 方法2: 使用新的分布分析脚本

运行新创建的分布分析脚本：
```bash
python analyze_topk_distribution.py \
    --stats-file /path/to/fft_topk_stats.json \
    --output-dir /path/to/output \
    --max-samples 10
```

这会生成：
- `topk_position_distribution.png`: 位置分布对比图
  - 显示每个位置被选中的频率
  - 显示概率分布
  - 显示分布变化（After - Before）
  
- `topk_selection_heatmap.png`: 选择热力图
  - 显示每个样本选择了哪些位置
  - Before和After的对比
  
- `topk_distribution_statistics.png`: 分布统计信息
  - 频率分布直方图
  - 累积分布
  - 位置范围分析
  - 统计摘要

### 方法3: 手动分析（Python代码示例）

```python
import json
import numpy as np
from collections import Counter

# 加载数据
with open('fft_topk_stats.json', 'r') as f:
    stats = json.load(f)

# 分析特定层和头
layer_idx = 0
head_idx = 0
key = f"layer_{layer_idx}_head_{head_idx}"

if key in stats:
    data = stats[key]
    
    # 1. 计算重叠率
    overlaps = []
    for before_indices, after_indices in zip(
        data['before_fft_topk_indices'],
        data['after_fft_topk_indices']
    ):
        before_set = set(before_indices)
        after_set = set(after_indices)
        overlap = len(before_set & after_set) / len(before_set) if len(before_set) > 0 else 0
        overlaps.append(overlap)
    
    print(f"Average overlap ratio: {np.mean(overlaps):.3f}")
    
    # 2. 统计位置频率
    all_before_positions = []
    all_after_positions = []
    for before_indices, after_indices in zip(
        data['before_fft_topk_indices'],
        data['after_fft_topk_indices']
    ):
        all_before_positions.extend(before_indices)
        all_after_positions.extend(after_indices)
    
    before_freq = Counter(all_before_positions)
    after_freq = Counter(all_after_positions)
    
    print(f"Most frequently selected positions (Before): {before_freq.most_common(10)}")
    print(f"Most frequently selected positions (After): {after_freq.most_common(10)}")
    
    # 3. 分析位置分布变化
    all_positions = set(before_freq.keys()) | set(after_freq.keys())
    position_changes = {}
    for pos in all_positions:
        before_count = before_freq.get(pos, 0)
        after_count = after_freq.get(pos, 0)
        position_changes[pos] = after_count - before_count
    
    # 找出变化最大的位置
    sorted_changes = sorted(position_changes.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"Positions with largest changes: {sorted_changes[:10]}")
```

## 3. 关键分析指标

### 3.1 Jaccard相似度
衡量FFT前后选择的topk集合的相似程度：
- 值范围: [0, 1]
- 1表示完全相同，0表示完全不同
- 公式: `J(A, B) = |A ∩ B| / |A ∪ B|`

### 3.2 重叠率 (Overlap Ratio)
FFT前后共同选择的token比例：
- 值范围: [0, 1]
- 公式: `overlap = |A ∩ B| / |A|`

### 3.3 位置分布
- **频率分布**: 每个token位置被选中的次数
- **概率分布**: 每个位置被选中的概率
- **分布偏移**: FFT后分布相对于FFT前的变化

### 3.4 唯一token数量
- **unique_before**: 只在FFT前被选中的token数量
- **unique_after**: 只在FFT后被选中的token数量
- **common**: 在FFT前后都被选中的token数量

## 4. 可视化图表说明

### 4.1 Position Distribution Plot
- **频率对比**: 显示每个位置被选中的总次数
- **概率分布**: 归一化后的选择概率
- **分布变化**: After - Before 的差值，红色表示增加，蓝色表示减少

### 4.2 Selection Heatmap
- 行：不同样本
- 列：token位置
- 颜色：是否被选中（深色=选中）

### 4.3 Distribution Statistics
- **频率分布直方图**: 显示选择频率的分布
- **累积分布**: 显示最常被选中的位置累积频率
- **位置范围分析**: 按位置区间统计选择频率
- **统计摘要**: 均值、标准差等统计信息

## 5. 使用建议

1. **快速概览**: 先运行 `visualize_fft_topk.py` 查看整体情况
2. **深入分析**: 运行 `analyze_topk_distribution.py` 查看分布变化
3. **特定分析**: 使用Python代码针对特定层/头进行自定义分析

## 6. 注意事项

- JSON文件可能很大（包含所有样本的所有层和头的数据）
- 建议使用 `SAMPLE_LIMIT` 限制样本数量进行测试
- 对于大数据集，考虑只分析部分层和头
- 可视化时注意内存使用，可以分批处理

