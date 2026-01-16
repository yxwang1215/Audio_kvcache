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

# 更美观的配色方案 - 使用学术论文常用的专业配色
colors = {
    'fullkv': '#4C72B0',      # 深蓝色 (seaborn blue)
    'snapkv': '#C44E52',      # 砖红色 (seaborn red)
    'adakv': '#55A868',       # 森林绿色 (seaborn green)
    'audiokv': '#DD8452',     # 珊瑚橙色 (seaborn orange)
}

# 准备绘图数据
x = np.arange(len(budgets))
width = 0.18  # 柱子宽度

# 提取各方法的数据 (顺序: Full KV, SnapKV, AdaKV, AudioKV)
fullkv_sizes = [fullkv_cache_size] * len(budgets)  # Full KV 在所有 budget 下相同
snapkv_sizes = [cache_sizes[b]['snapkv'] for b in budgets]
adakv_sizes = [cache_sizes[b]['adakv'] for b in budgets]
audiokv_sizes = [cache_sizes[b]['audiokv'] for b in budgets]

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制柱状图 - 4个方法 (顺序: Full KV, SnapKV, AdaKV, AudioKV)
offset = width * 1.5
bars_fullkv = ax.bar(x - offset, fullkv_sizes, width, label='Full KV', 
                     color=colors['fullkv'], edgecolor='none')
bars_snapkv = ax.bar(x - width/2, snapkv_sizes, width, label='SnapKV', 
                     color=colors['snapkv'], edgecolor='none')
bars_adakv = ax.bar(x + width/2, adakv_sizes, width, label='AdaKV', 
                    color=colors['adakv'], edgecolor='none')
bars_audiokv = ax.bar(x + offset, audiokv_sizes, width, label='AudioKV', 
                      color=colors['audiokv'], edgecolor='none')

# 设置坐标轴
ax.set_ylabel('memory', fontsize=21, fontweight='bold')
# 不添加标题
ax.set_xticks(x)
ax.set_xticklabels([f'{b}' for b in budgets], fontsize=17, fontweight='bold')
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
