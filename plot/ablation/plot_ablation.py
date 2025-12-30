#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制消融实验的8个子图
根据table.txt数据和图片中的消融结果
ACL顶会风格
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置ACL顶会风格：使用LaTeX字体，高质量渲染
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 16  # 增大基础字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 解析数据
def parse_table(file_path):
    """解析table.txt文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    data = {}
    # 表格中的顺序是 0.8, 0.6, 0.4（从高到低）
    retain_ratios_table = [0.8, 0.6, 0.4]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        parts = line.split('\t')
        
        # 检查是否是模型行（包含模型名称）
        if 'Gemma-E2B' in line:
            current_model = 'Gemma-E2B'
        elif 'Gemma-E4B' in line:
            current_model = 'Gemma-E4B'
        elif 'Qwen2.5-7B' in line:
            current_model = 'Qwen2.5-7B'
        elif 'Qwen2.5-3B' in line:
            current_model = 'Qwen2.5-3B'
        elif 'Qwen3-30B' in line:
            current_model = 'Qwen3-30B'
        else:
            i += 1
            continue
        
        # 初始化数据结构
        if current_model not in data:
            data[current_model] = {
                'LibriSpeech-long': {},
                'SP_librispeech': {}
            }
        
        # 跳过retain ratio标题行
        i += 1
        if i >= len(lines) or 'retain ratio' in lines[i]:
            i += 1
        
        # 读取vanilla行
        if i < len(lines):
            vanilla_line = lines[i].split('\t')
            if vanilla_line[0] == 'vanilla':
                # LibriSpeech-long vanilla值在索引1，SP_librispeech在索引4
                try:
                    if len(vanilla_line) > 1 and vanilla_line[1]:
                        vanilla_ls = float(vanilla_line[1])
                        data[current_model]['LibriSpeech-long']['vanilla'] = vanilla_ls
                except:
                    pass
                try:
                    if len(vanilla_line) > 4 and vanilla_line[4]:
                        vanilla_sp = float(vanilla_line[4])
                        data[current_model]['SP_librispeech']['vanilla'] = vanilla_sp
                except:
                    pass
                i += 1
        
        # 读取其他方法行
        while i < len(lines):
            method_line = lines[i].split('\t')
            if not method_line[0]:
                i += 1
                continue
            
            # 检查是否是新的模型行
            if method_line[0] in ['Gemma-E2B', 'Gemma-E4B', 'Qwen2.5-7B', 'Qwen2.5-3B', 'Qwen3-30B']:
                break
            
            method = method_line[0]
            if method == 'retain ratio':
                i += 1
                continue
            
            # 初始化方法数据
            if method not in data[current_model]['LibriSpeech-long']:
                data[current_model]['LibriSpeech-long'][method] = {}
            if method not in data[current_model]['SP_librispeech']:
                data[current_model]['SP_librispeech'][method] = {}
            
            # 解析LibriSpeech-long数据 (索引1-3对应0.8, 0.6, 0.4，表格顺序)
            for ratio_idx, ratio_table in enumerate(retain_ratios_table):
                col_idx = 1 + ratio_idx  # 1, 2, 3
                if col_idx < len(method_line) and method_line[col_idx]:
                    try:
                        val = float(method_line[col_idx])
                        # 保留所有值，包括0，使用表格中的ratio作为键
                        data[current_model]['LibriSpeech-long'][method][ratio_table] = val
                    except:
                        data[current_model]['LibriSpeech-long'][method][ratio_table] = None
                else:
                    data[current_model]['LibriSpeech-long'][method][ratio_table] = None
            
            # 解析SP_librispeech数据 (索引4-6对应0.8, 0.6, 0.4，表格顺序)
            for ratio_idx, ratio_table in enumerate(retain_ratios_table):
                col_idx = 4 + ratio_idx  # 4, 5, 6
                if col_idx < len(method_line) and method_line[col_idx]:
                    try:
                        val = float(method_line[col_idx])
                        # 保留所有值，包括0，使用表格中的ratio作为键
                        data[current_model]['SP_librispeech'][method][ratio_table] = val
                    except:
                        data[current_model]['SP_librispeech'][method][ratio_table] = None
                else:
                    data[current_model]['SP_librispeech'][method][ratio_table] = None
            
            i += 1
    
    return data

# 绘制单个子图
def plot_subplot(ax, model, dataset, data, retain_ratios, model_idx=0, dataset_display_names=None):
    """绘制单个子图"""
    if dataset_display_names is None:
        dataset_display_names = {}
    methods = ['vanilla', 'snapkv', 'AudioKV (without SSS)', 'SnapKV', 'AudioKV']
    colors = {
        'vanilla': '#1f77b4',      # 蓝色
        'snapkv': '#ff7f0e',       # 橙色
        'AudioKV (without SSS)': '#2ca02c',     # 绿色
        'SnapKV': '#d62728',   # 红色
        'AudioKV': '#9467bd'  # 紫色
    }
    markers = {
        'vanilla': 'o',
        'snapkv': 's',
        'AudioKV (without SSS)': '^',
        'SnapKV': 'D',
        'AudioKV': 'v'
    }
    linestyles = {
        'vanilla': '--',
        'snapkv': '-',
        'AudioKV (without SSS)': '-',
        'SnapKV': '-',
        'AudioKV': '-'
    }
    linewidths = {
        'vanilla': 2.5,
        'snapkv': 2.0,
        'AudioKV (without SSS)': 2.0,
        'SnapKV': 2.0,
        'AudioKV': 2.5
    }
    
    # 方法显示名称映射
    method_display_names = {
        'vanilla': 'Vanilla',
        'snapkv': 'SnapKV',
        'AudioKV (without SSS)': 'AudioKV (without SSS)',
        'SnapKV': 'SnapKV (with SSS)',
        'AudioKV': 'AudioKV'
    }
    
    for method in methods:
        if method == 'vanilla':
            # vanilla只有一个值，在retain ratio=1.0的位置（或作为基准线）
            vanilla_val = data[model][dataset].get('vanilla')
            if vanilla_val is not None:
                # 绘制水平基准线
                ax.axhline(y=vanilla_val, color=colors[method], 
                          linestyle=linestyles[method], linewidth=linewidths[method],
                          label=method_display_names[method], alpha=0.8, zorder=1)
        else:
            # 其他方法有多个retain ratio的值
            x_vals = []
            y_vals = []
            
            if method in data[model][dataset]:
                for ratio in retain_ratios:
                    val = data[model][dataset][method].get(ratio)
                    # 包含所有点，包括 acc=0 的点
                    if val is not None:
                        x_vals.append(ratio)
                        y_vals.append(val)
            
            if x_vals and y_vals:
                # 按x值降序排序（从高到低），让左边显示高值
                sorted_pairs = sorted(zip(x_vals, y_vals), reverse=True)
                x_vals = [x for x, _ in sorted_pairs]
                y_vals = [y for _, y in sorted_pairs]
                
                ax.plot(x_vals, y_vals, color=colors[method], 
                       marker=markers[method], linestyle=linestyles[method],
                       linewidth=linewidths[method], markersize=8,
                       label=method_display_names[method], alpha=0.9, zorder=2)
    
    # 模型名称映射到规范名称
    model_name_map = {
        'Gemma-E2B': 'Gemma-3n-E2B',
        'Gemma-E4B': 'Gemma-3n-E4B',
        'Qwen2.5-3B': 'Qwen2.5-Omni-3B',
        'Qwen2.5-7B': 'Qwen-2.5-Omni-7B',
        'Qwen3-30B': 'Qwen3-Omni-30B-A3B-Instruct'
    }
    
    # 设置标签和标题
    ax.set_xlabel('Retain Ratio', fontsize=18, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=18, fontweight='bold')
    
    # 使用规范模型名称和数据集显示名称
    display_model = model_name_map.get(model, model)
    if dataset_display_names:
        display_dataset = dataset_display_names.get(dataset, dataset)
    else:
        display_dataset = dataset
    title = f'{display_model}\n{display_dataset}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
    
    # 设置x轴刻度（从高到低）
    ax.set_xticks(retain_ratios)
    ax.set_xticklabels([f'{r:.1f}' for r in retain_ratios], fontsize=15)
    # 反转x轴，让左边显示高值
    ax.invert_xaxis()
    
    # 设置y轴范围（根据数据自动调整，但留出一些边距）
    all_values = []
    for method in methods:
        if method == 'vanilla':
            vanilla_val = data[model][dataset].get('vanilla')
            if vanilla_val is not None:
                all_values.append(vanilla_val)
        else:
            if method in data[model][dataset]:
                for ratio in retain_ratios:
                    val = data[model][dataset][method].get(ratio)
                    # 考虑所有值（包括0）来计算y轴范围
                    if val is not None:
                        all_values.append(val)
    
    if all_values:
        y_min = -3  # 从-3开始，让线条更清晰
        y_max = 103  # 显示到103，上方留出空间
        ax.set_ylim(y_min, y_max)
    
    # 网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 收集图例信息（返回handles和labels）
    handles, labels = ax.get_legend_handles_labels()
    
    # 按照methods的顺序重新排序图例
    method_order = ['vanilla', 'snapkv', 'AudioKV (without SSS)', 'SnapKV', 'AudioKV']
    method_display_names_map = {
        'vanilla': 'Vanilla',
        'snapkv': 'SnapKV',
        'AudioKV (without SSS)': 'AudioKV (without SSS)',
        'SnapKV': 'SnapKV (with SSS)',
        'AudioKV': 'AudioKV'
    }
    ordered_handles = []
    ordered_labels = []
    for method in method_order:
        expected_label = method_display_names_map[method]
        if expected_label in labels:
            idx = labels.index(expected_label)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])
    
    # 刻度标签加粗并增大字体
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(15)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(15)
    
    return ordered_handles, ordered_labels

# 主函数
def main():
    # 读取数据
    table_path = Path(__file__).parent / 'table.txt'
    data = parse_table(table_path)
    
    # 模型和数据集
    models = ['Gemma-E2B', 'Gemma-E4B', 'Qwen2.5-3B', 'Qwen2.5-7B', 'Qwen3-30B']
    datasets = ['LibriSpeech-long', 'SP_librispeech']  # 数据键名
    dataset_display_names = {
        'LibriSpeech-long': 'LibriSpeech-long',
        'SP_librispeech': 'LibriSpeech-Spanish'
    }
    retain_ratios = [0.8, 0.6, 0.4]  # 从高到低，左边显示高值
    
    # 创建2x5的子图布局（2行5列：2个数据集，5个模型）
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
    # 用于收集图例的handles和labels（从所有子图中收集，确保包含所有方法）
    legend_handles_dict = {}
    legend_labels_dict = {}
    
    # 绘制每个子图
    for dataset_idx, dataset in enumerate(datasets):
        for model_idx, model in enumerate(models):
            ax = axes[dataset_idx, model_idx]
            handles, labels = plot_subplot(ax, model, dataset, data, retain_ratios, model_idx, dataset_display_names)
            
            # 收集所有方法的图例信息
            for handle, label in zip(handles, labels):
                if label not in legend_handles_dict:
                    legend_handles_dict[label] = handle
                    legend_labels_dict[label] = label
    
    # 按照methods的顺序整理图例
    method_order = ['vanilla', 'snapkv', 'AudioKV (without SSS)', 'SnapKV', 'AudioKV']
    method_display_names_map = {
        'vanilla': 'Vanilla',
        'snapkv': 'SnapKV',
        'AudioKV (without SSS)': 'AudioKV (without SSS)',
        'SnapKV': 'SnapKV (with SSS)',
        'AudioKV': 'AudioKV'
    }
    legend_handles = []
    legend_labels = []
    for method in method_order:
        expected_label = method_display_names_map[method]
        if expected_label in legend_handles_dict:
            legend_handles.append(legend_handles_dict[expected_label])
            legend_labels.append(legend_labels_dict[expected_label])
    
    # 添加共享图例（在底部居中，增大字体并加粗）
    legend = fig.legend(legend_handles, legend_labels, 
              loc='lower center', bbox_to_anchor=(0.5, 0.02),
              ncol=5, fontsize=18, framealpha=0.95,
              edgecolor='gray', fancybox=True, shadow=False)
    # 设置图例文本加粗
    for text in legend.get_texts():
        text.set_fontweight('bold')
    
    # 调整布局，为底部图例留出空间
    plt.tight_layout(pad=3.0, rect=[0, 0.05, 1, 1])
    
    # 保存图片
    output_dir = Path(__file__).parent
    
    # 保存PNG格式
    png_path = output_dir / 'ablation_plots.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"PNG图片已保存至: {png_path}")
    
    # 保存SVG格式
    svg_path = output_dir / 'ablation_plots.svg'
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f"SVG矢量图已保存至: {svg_path}")
    
    # 保存PDF格式
    pdf_path = output_dir / 'ablation_plots.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"PDF文件已保存至: {pdf_path}")
    
    plt.close()

if __name__ == '__main__':
    main()

