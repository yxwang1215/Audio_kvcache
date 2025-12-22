#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 SAH+FFT 精度与 alpha 和 Cutoff Ratio 的相关性3D立体图
ACL顶会风格
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# 设置ACL顶会风格：使用LaTeX字体，高质量渲染
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 18  # 增大基础字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 数据矩阵
data = np.array([
    [91.36, 91.36, 91.36, 91.36, 91.36, 91.36, 91.36, 91.36, 91.36, 91.36, 91.36],
    [92.03, 92.62, 94.1, 94.73, 94.61, 94.39, 94.37, 93.95, 93.64, 95.28, 89.6],
    [92.66, 92.12, 93.71, 94.2, 94.64, 94.04, 94.31, 94.03, 93.63, 94.17, 89.96],
    [91.75, 91.01, 92.56, 93.5, 94.09, 93.1, 94.44, 94.03, 93.59, 93.96, 88.14],
    [92.26, 90.27, 91.66, 93.63, 94.54, 93.04, 93.36, 94.29, 93.56, 93.93, 89.44],
    [91.57, 89.73, 92.47, 93.75, 94.44, 93.1, 93.98, 94.34, 93.04, 93.11, 87.53],
    [90.28, 89.71, 91.96, 93.37, 92.95, 93.12, 93.44, 94.19, 92.08, 94.73, 88.14],
    [88.25, 88.9, 90.85, 93.83, 92.54, 93.53, 93.48, 93.33, 91.86, 93.29, 88.11],
    [87.19, 88.85, 90.17, 93.27, 93.31, 93.92, 93.05, 93.84, 92.34, 93.29, 88.16],
    [83.97, 88.2, 90.34, 92.44, 92.54, 93.13, 93.47, 93.9, 92.36, 93.34, 88.35],
    [36.86, 87.21, 91.43, 92.61, 93.54, 93.27, 92.18, 93.81, 92.17, 93.76, 88.04]
])

# 去掉小于85的离群点（设为NaN，这样不会显示）
data_cleaned = data.copy()
data_cleaned[data_cleaned < 85] = np.nan

# Alpha 和 Cutoff 值
alpha_values = np.arange(0, 1.1, 0.1)
cutoff_values_all = np.arange(0, 1.1, 0.1)

# 移除 cutoff ratio = 0, 0.1, 1.0 的列（索引 0, 1, 10）
# 保留 cutoff ratio = 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
indices_to_keep = [2, 3, 4, 5, 6, 7, 8, 9]  # 对应 0.2 到 0.9
data_cleaned = data_cleaned[:, indices_to_keep]
cutoff_values = cutoff_values_all[indices_to_keep]

# 创建网格
X, Y = np.meshgrid(cutoff_values, alpha_values)

# 创建3D图形
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 使用ACL顶会常用的专业colormap（plasma风格，从深紫到亮黄）
# 这种配色在学术论文中非常常见，既美观又专业
cmap = plt.cm.plasma

# 基准值
baseline_value = 91.36

# 计算合适的z轴范围
min_val = np.nanmin(data_cleaned)
max_val = np.nanmax(data_cleaned)
z_min = max(87, min_val - 1)  # 下限至少87，或比最小值低1
z_max = min(96, max_val + 1)   # 上限最多96，或比最大值高1

# 绘制完整的3D表面图（保留所有数据，不再区分高于和低于基准）
# 注意：plot_surface的颜色是根据Z值（data_cleaned）通过cmap计算的
# 每个面的颜色由该面的Z值决定，使用plasma colormap映射
# 面的颜色计算：对于每个四边形面片，颜色由该面片的Z值（高度）决定
# 如果面片跨越多个Z值，matplotlib会使用插值或平均Z值来确定颜色
surf = ax.plot_surface(X, Y, data_cleaned, cmap=cmap, alpha=0.95, 
                       linewidth=0.05, antialiased=True, 
                       edgecolor='none', shade=True,
                       rstride=1, cstride=1)

# 添加等高线投影到底面（增强立体感，使用清理后的数据）
contour = ax.contour(X, Y, data_cleaned, zdir='z', offset=z_min, 
                     cmap=cmap, alpha=0.4, linewidths=1.0, 
                     levels=15)

# 添加基准平面（纯AudioKV基准值91.36，不使用FFT）
baseline_Z = np.full_like(X, baseline_value)
# 使用半透明的平面
baseline_surf = ax.plot_surface(X, Y, baseline_Z, alpha=0.25, 
                                color='gray', shade=False)

# 手动绘制虚线网格（使用plot函数逐条绘制虚线）
# 绘制X方向的虚线
baseline_lines = []
for i in range(0, len(alpha_values), 1):  # 可以调整步长来控制网格密度
    line = ax.plot(cutoff_values, 
            np.full_like(cutoff_values, alpha_values[i]),
            np.full_like(cutoff_values, baseline_value),
            'k--', alpha=0.6, linewidth=1.0, label='AudioKV (baseline)' if i == 0 else '')
    if i == 0:
        baseline_lines.append(line[0])
# 绘制Y方向的虚线
for j in range(0, len(cutoff_values), 1):  # 可以调整步长来控制网格密度
    ax.plot(np.full_like(alpha_values, cutoff_values[j]),
            alpha_values,
            np.full_like(alpha_values, baseline_value),
            'k--', alpha=0.6, linewidth=1.0)

# 在右下区域（绿色圈位置）添加 AudioKV (baseline) 标注
# 在右下角（高Alpha、高Cutoff Ratio）绘制指示线
max_cutoff = cutoff_values.max()  # 现在是0.9
line_x = np.array([max_cutoff, max_cutoff + 0.15])  # 从右侧边缘延伸
line_y = np.array([1.0, 1.0])   # 在Alpha=1.0位置
line_z = np.array([baseline_value, baseline_value])
ax.plot(line_x, line_y, line_z, 'k-', linewidth=3.0, alpha=1.0, zorder=10)

# 在线的末端添加文字标注
ax.text(max_cutoff + 0.18, 1.0, baseline_value, 'AudioKV\n(baseline)', 
        fontsize=17, fontweight='bold', color='black',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                 alpha=0.95, edgecolor='black', linewidth=2.0),
        ha='left', va='center', zorder=11)

# 设置视角（美观的透视角度，展示最佳视觉效果）
ax.view_init(elev=35, azim=50)

# 设置坐标轴标签（ACL风格：清晰、专业）
ax.set_xlabel('Cutoff Ratio', fontsize=20, fontweight='bold', labelpad=20)
ax.set_ylabel('Alpha', fontsize=20, fontweight='bold', labelpad=20)
# 移除z轴标签，因为colorbar已经显示了Accuracy (%)
# ax.set_zlabel('Accuracy (%)', fontsize=20, fontweight='bold', labelpad=12)

# 设置坐标轴刻度（更精细的刻度）
# 减少cutoff ratio的刻度数量，避免过于密集
# 手动指定要显示的刻度：0.2, 0.4, 0.6, 0.8（不显示0.9）
xticks_to_show = [0.2, 0.4, 0.6, 0.8]
ax.set_xticks(xticks_to_show)
# Alpha刻度：0, 0.2, 0.4, 0.6, 0.8（不显示1.0）
ax.set_yticks(np.arange(0, 1.0, 0.2))
# 使用已计算的z轴范围设置刻度（压缩间距，减少刻度数量）
ax.set_zticks(np.arange(int(z_min), int(z_max) + 1, 3))  # 每3个单位一个刻度

# 设置刻度标签格式（增大字体并加粗）
ax.tick_params(axis='x', labelsize=20, pad=7)
ax.tick_params(axis='y', labelsize=20, pad=7)
ax.tick_params(axis='z', labelsize=20, pad=7)

# 设置坐标轴范围
ax.set_xlim(cutoff_values.min() - 0.05, cutoff_values.max() + 0.05)  # 稍微扩展范围以便显示
ax.set_ylim(0, 1.0)
# 使用计算好的z轴范围
ax.set_zlim(z_min, z_max)

# 为坐标轴刻度数字加粗（在设置范围后执行，确保所有刻度都已生成）
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
for label in ax.get_zticklabels():
    label.set_fontweight('bold')

# 添加颜色条（压缩刻度间距，增大字体）
cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=25, pad=0.05)
cbar.set_label('Accuracy (%)', fontsize=22, fontweight='bold', rotation=270, labelpad=30)
# 减少colorbar刻度数量，压缩间距
cbar.ax.tick_params(labelsize=20)
# 设置colorbar刻度数量（减少刻度，压缩间距）
cbar.locator = plt.MaxNLocator(nbins=5)  # 最多5个刻度
cbar.update_ticks()
# 为colorbar刻度数字加粗
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')

# 不添加图例，改用侧边标注

# 设置标题（简化，去掉描述性文字）
ax.set_title('AudioKV+FFT Accuracy Versus AudioKV Baseline', 
             fontsize=22, fontweight='bold', pad=25)

# 设置网格样式（ACL风格：清晰但不突兀）
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('gray')
ax.yaxis.pane.set_edgecolor('gray')
ax.zaxis.pane.set_edgecolor('gray')
ax.xaxis.pane.set_alpha(0.1)
ax.yaxis.pane.set_alpha(0.1)
ax.zaxis.pane.set_alpha(0.1)

# 设置背景色
ax.xaxis.pane.set_facecolor('white')
ax.yaxis.pane.set_facecolor('white')
ax.zaxis.pane.set_facecolor('white')

# 调整布局
plt.tight_layout()

# 保存图片（多种格式）
output_dir = Path(__file__).parent

# 保存PNG格式
png_path = output_dir / 'sah_fft_accuracy_heatmap.png'
plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"PNG图片已保存至: {png_path}")

# 保存SVG格式（矢量图）
svg_path = output_dir / 'sah_fft_accuracy_heatmap.svg'
plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
print(f"SVG矢量图已保存至: {svg_path}")

# 保存PDF格式
pdf_path = output_dir / 'sah_fft_accuracy_heatmap.pdf'
plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
print(f"PDF文件已保存至: {pdf_path}")

# 也可以显示图片（如果在交互环境中）
# plt.show()

plt.close()

