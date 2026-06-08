#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cache Size 对比柱状图
ACL顶会风格配色
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置ACL顶会风格：使用LaTeX字体，高质量渲染
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 19
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 数据
budgets = [0.4, 0.6, 0.8]
methods = ['Full KV', 'SnapKV', 'AdaKV', 'AudioKV']

# Cache size 数据 (按 budget 分组)
cache_sizes = {
    0.4: {'audiokv': 0.118, 'adakv': 0.118, 'snapkv': 0.118},
    0.6: {'audiokv': 0.1586, 'adakv': 0.1586, 'snapkv': 0.158},
    0.8: {'audiokv': 0.1844, 'adakv': 0.1846, 'snapkv': 0.1846},
}

# Full KV Cache - 在所有 budget 下都是同一个值
fullkv_cache_size = 0.202972412109375

# 更浅的配色方案 + 斜线阴影（hatch）
colors = {
    'fullkv': '#C7D8F0',
    'snapkv': '#F2CCD0',
    'adakv': '#D3EED8',
    'audiokv': '#F8DEC5',
}

bar_alpha = 0.70

edge_colors = {
    'fullkv': '#3B5A88',
    'snapkv': '#9E4D52',
    'adakv': '#3E7D50',
    'audiokv': '#A76637',
}

hatches = {
    'fullkv': '//',
    'snapkv': '//',
    'adakv': '//',
    'audiokv': '//',
}

# 准备绘图数据
x = np.arange(len(budgets))
width = 0.155
inner_gap = 0.02
step = width + inner_gap

# 提取各方法的数据 (顺序: Full KV, SnapKV, AdaKV, AudioKV)
fullkv_sizes = [fullkv_cache_size] * len(budgets)  # Full KV 在所有 budget 下相同
snapkv_sizes = [cache_sizes[b]['snapkv'] for b in budgets]
adakv_sizes = [cache_sizes[b]['adakv'] for b in budgets]
audiokv_sizes = [cache_sizes[b]['audiokv'] for b in budgets]

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制柱状图 - 4个方法 (顺序: Full KV, SnapKV, AdaKV, AudioKV)
bars_fullkv = ax.bar(x - 1.5 * step, fullkv_sizes, width, label='Full KV', 
                     color=colors['fullkv'], hatch=hatches['fullkv'],
                     edgecolor=edge_colors['fullkv'], linewidth=1.2, alpha=bar_alpha)
bars_snapkv = ax.bar(x - 0.5 * step, snapkv_sizes, width, label='SnapKV', 
                     color=colors['snapkv'], hatch=hatches['snapkv'],
                     edgecolor=edge_colors['snapkv'], linewidth=1.2, alpha=bar_alpha)
bars_adakv = ax.bar(x + 0.5 * step, adakv_sizes, width, label='AdaKV', 
                    color=colors['adakv'], hatch=hatches['adakv'],
                    edgecolor=edge_colors['adakv'], linewidth=1.2, alpha=bar_alpha)
bars_audiokv = ax.bar(x + 1.5 * step, audiokv_sizes, width, label='AudioKV', 
                      color=colors['audiokv'], hatch=hatches['audiokv'],
                      edgecolor=edge_colors['audiokv'], linewidth=1.2, alpha=bar_alpha)

# 设置坐标轴
ax.set_ylabel('Memory (GB)', fontsize=21, fontweight='bold')
# 不添加标题
ax.set_xticks(x)
ax.set_xticklabels([f'{int(b * 100)}%' for b in budgets], fontsize=17, fontweight='bold')
# 不从0开始，让差异更明显
ax.set_ylim(0.06, fullkv_cache_size * 1.15)

# 设置y轴刻度，让数字更稀疏
from matplotlib.ticker import MultipleLocator
ax.yaxis.set_major_locator(MultipleLocator(0.04))  # 每0.04显示一个刻度

# 设置y轴刻度字体
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(17)

# 添加网格 - 确保每0.04都有灰色实线，更粗更明显
ax.yaxis.grid(True, which='major', alpha=0.6, linestyle='-', linewidth=1.2, color='gray')
ax.set_axisbelow(True)

# 添加图例（放在底部，稍微向上）
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
                   ncol=4, fontsize=17, framealpha=0.95,
                   edgecolor='gray', fancybox=True)
for text in legend.get_texts():
    text.set_fontweight('bold')

# 优化布局
plt.tight_layout(rect=[0, 0.02, 1, 1])

# 保存图片
output_dir = Path(__file__).parent
plt.savefig(output_dir / 'cache_size_comparison.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_dir / 'cache_size_comparison.pdf', 
            bbox_inches='tight', facecolor='white')

print("图片已保存到:")
print(f"- {output_dir / 'cache_size_comparison.png'}")
print(f"- {output_dir / 'cache_size_comparison.pdf'}")

plt.close()