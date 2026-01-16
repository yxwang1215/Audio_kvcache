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
import re

# 设置ACL顶会风格：使用LaTeX字体，高质量渲染
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 20  # 增大基础字体
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
                'SP_librispeech': {},
                'FR_librispeech': {}
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

# 解析ablation.tex文件提取FR数据
def parse_tex_fr(file_path):
    """从ablation.tex文件中提取FR列的数据（索引7-9对应0.8, 0.6, 0.4）"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    fr_data = {}
    retain_ratios = [0.8, 0.6, 0.4]
    
    # 模型名称映射
    model_map = {
        'Qwen-2.5-Omni 3B': 'Qwen2.5-3B',
        'Qwen-2.5-Omni-7B': 'Qwen2.5-7B',
        'Qwen3-Omni-30B-A3B-Instruct': 'Qwen3-30B',
        'Gemma-3n-E2B': 'Gemma-E2B',
        'Gemma-3n-E4B': 'Gemma-E4B'
    }
    
    # 手动提取的数据（从ablation.tex中FR列，索引7-9）
    # 格式：{模型: {方法: {0.8: val, 0.6: val, 0.4: val}}}
    manual_data = {
        'Qwen2.5-3B': {
            'snapkv': {0.8: 54.2, 0.6: 0.0, 0.4: 0.0},
            'SnapKV': {0.8: 88.9, 0.6: 87.6, 0.4: 0.0},
            'AudioKV (without SSS)': {0.8: 90.7, 0.6: 90.0, 0.4: 0.0},
            'AudioKV': {0.8: 89.9, 0.6: 90.5, 0.4: 85.8}
        },
        'Qwen2.5-7B': {
            'snapkv': {0.8: 72.4, 0.6: 0.0, 0.4: 0.0},
            'SnapKV': {0.8: 83.0, 0.6: 76.9, 0.4: 19.1},
            'AudioKV (without SSS)': {0.8: 83.1, 0.6: 82.1, 0.4: 0.0},
            'AudioKV': {0.8: 83.4, 0.6: 82.9, 0.4: 78.7}
        },
        'Qwen3-30B': {
            'snapkv': {0.8: 68.7, 0.6: 0.0, 0.4: 0.0},
            'SnapKV': {0.8: 95.1, 0.6: 89.8, 0.4: 0.0},
            'AudioKV (without SSS)': {0.8: 95.6, 0.6: 95.2, 0.4: 0.0},
            'AudioKV': {0.8: 95.6, 0.6: 95.5, 0.4: 89.2}
        },
        'Gemma-E2B': {
            'snapkv': {0.8: 33.8, 0.6: 0.0, 0.4: 0.0},
            'SnapKV': {0.8: 61.5, 0.6: 47.2, 0.4: 0.0},
            'AudioKV (without SSS)': {0.8: 63.3, 0.6: 54.7, 0.4: 20.9},
            'AudioKV': {0.8: 66.8, 0.6: 64.9, 0.4: 28.9}
        },
        'Gemma-E4B': {
            'snapkv': {0.8: 57.8, 0.6: 0.0, 0.4: 0.0},
            'SnapKV': {0.8: 71.4, 0.6: 59.1, 0.4: 0.0},
            'AudioKV (without SSS)': {0.8: 75.5, 0.6: 74.4, 0.4: 35.4},
            'AudioKV': {0.8: 75.5, 0.6: 75.2, 0.4: 35.5}
        }
    }
    
    return manual_data

# 合并FR数据到主数据结构
def merge_fr_data(data, fr_data):
    """将FR数据合并到主数据结构中"""
    # French数据集的vanilla基准值（从左到右：Gemma-E2B, Gemma-E4B, Qwen2.5-3B, Qwen2.5-7B, Qwen3-30B）
    vanilla_values = {
        'Gemma-E2B': 57.2,
        'Gemma-E4B': 60.5,
        'Qwen2.5-3B': 90.9,
        'Qwen2.5-7B': 83.9,
        'Qwen3-30B': 95.6
    }
    
    for model in fr_data:
        if model not in data:
            data[model] = {}
        if 'FR_librispeech' not in data[model]:
            data[model]['FR_librispeech'] = {}
        
        # 添加vanilla值
        if model in vanilla_values:
            data[model]['FR_librispeech']['vanilla'] = vanilla_values[model]
        
        for method in fr_data[model]:
            if method not in data[model]['FR_librispeech']:
                data[model]['FR_librispeech'][method] = {}
            data[model]['FR_librispeech'][method].update(fr_data[model][method])
    
    return data

# 绘制单个子图
def plot_subplot(ax, model, dataset, data, retain_ratios, model_idx=0, dataset_display_names=None):
    """绘制单个子图"""
    if dataset_display_names is None:
        dataset_display_names = {}
    methods = ['vanilla', 'snapkv', 'AudioKV (without SSS)', 'SnapKV', 'AudioKV']
    colors = {
        'vanilla': '#2E5C8A',      # 深蓝色 (高对比度版本) - 对应 Full KV
        'snapkv': '#C66B3A',       # 珊瑚橙色 (高对比度版本) - 对应 SnapKV
        'AudioKV (without SSS)': '#3D8B4F',     # 森林绿色 (高对比度版本) - 对应 AdaKV 配色
        'SnapKV': '#6B5A9A',   # 紫色 (高对比度版本) - 与 snapkv 区分
        'AudioKV': '#B83A3F'  # 砖红色 (高对比度版本) - 对应 AudioKV
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
        'vanilla': 'Full KV',
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
        'Qwen2.5-7B': 'Qwen2.5-Omni-7B',
        'Qwen3-30B': 'Qwen3-Omni-30B-A3B-Instruct'
    }
    
    # 设置标签和标题
    ax.set_xlabel('Retain Ratio', fontsize=22, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=22, fontweight='bold')
    
    # 使用规范模型名称和数据集显示名称
    display_model = model_name_map.get(model, model)
    if dataset_display_names:
        display_dataset = dataset_display_names.get(dataset, dataset)
    else:
        display_dataset = dataset
    title = f'{display_model}\n{display_dataset}'
    ax.set_title(title, fontsize=20, fontweight='bold', pad=12)
    
    # 设置x轴刻度（从高到低）
    ax.set_xticks(retain_ratios)
    ax.set_xticklabels([f'{r:.1f}' for r in retain_ratios], fontsize=19)
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
        'vanilla': 'Full KV',
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
        label.set_fontsize(19)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(19)
    
    return ordered_handles, ordered_labels

# 主函数
def main():
    # 读取数据
    table_path = Path(__file__).parent / 'table.txt'
    data = parse_table(table_path)
    
    # 从ablation.tex提取FR数据
    tex_path = Path(__file__).parent / 'ablation.tex'
    if tex_path.exists():
        fr_data = parse_tex_fr(tex_path)
        data = merge_fr_data(data, fr_data)
    
    # 模型和数据集
    models = ['Gemma-E2B', 'Gemma-E4B', 'Qwen2.5-3B', 'Qwen2.5-7B', 'Qwen3-30B']
    datasets = ['LibriSpeech-long', 'SP_librispeech', 'FR_librispeech']  # 数据键名
    dataset_display_names = {
        'LibriSpeech-long': 'LibriSpeech-long',
        'SP_librispeech': 'LibriSpeech-Spanish',
        'FR_librispeech': 'LibriSpeech-French'
    }
    retain_ratios = [0.8, 0.6, 0.4]  # 从高到低，左边显示高值
    
    # 创建3x5的子图布局（3行5列：3个数据集，5个模型）
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    
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
        'vanilla': 'Full KV',
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
              loc='lower center', bbox_to_anchor=(0.5, -0.02),
              ncol=5, fontsize=24, framealpha=0.95,
              edgecolor='gray', fancybox=True, shadow=False)
    # 设置图例文本加粗
    for text in legend.get_texts():
        text.set_fontweight('bold')
    
    # 调整布局，为底部图例留出空间（折中的间距）
    plt.tight_layout(pad=3.0, rect=[0, 0.08, 1, 1])
    # 进一步调整子图间距（在tight_layout之后调整，折中值）
    # 增加水平间距以解决标题截断问题
    plt.subplots_adjust(left=0.05, right=0.95, top=0.96, bottom=0.10, 
                       wspace=0.4, hspace=0.5)
    
    # 保存图片
    output_dir = Path(__file__).parent
    
    # 保存PNG格式
    # png_path = output_dir / 'ablation_plots.png'
    # plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    # print(f"PNG图片已保存至: {png_path}")
    
    # # 保存SVG格式
    # svg_path = output_dir / 'ablation_plots.svg'
    # plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    # print(f"SVG矢量图已保存至: {svg_path}")
    
    # 保存PDF格式
    pdf_path = output_dir / 'ablation_plots.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"PDF文件已保存至: {pdf_path}")
    
    plt.close()

if __name__ == '__main__':
    main()

