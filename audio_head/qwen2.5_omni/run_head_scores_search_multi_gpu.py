#!/usr/bin/env python3
"""
多 GPU 并行运行 head scores 生成和参数搜索
将样本分析分配到多个 GPU，最大化利用计算资源
"""

import os
import sys
import json
import subprocess
import time
import threading
from queue import Queue
from pathlib import Path
from collections import defaultdict

# GPU列表 - 如果未指定，将自动检测所有可用 GPU
GPUS = [0, 2,3]  # 默认值，如果自动检测失败则使用此列表

# 基础路径
BASE_DIR = "/mnt/users/wangyuxuan-20250915/Audio"
SCRIPT_PATH = f"{BASE_DIR}/Audio_kvcache/audio_head/qwen2.5_omni/generate_head_scores_with_search.py"

# 默认参数
DEFAULT_MODEL_PATH = "/mnt/users/wangyuxuan-20250915/Models/Qwen2.5-Omni-3B"
DEFAULT_AUDIO_DIR = "/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/output_audio"
DEFAULT_WHISPERX_DIR = "/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/whisperx_parquet_output"
DEFAULT_OUTPUT_DIR = "/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni"

# 锁用于线程安全
results_lock = threading.Lock()

# 全局结果存储
all_sample_results = []  # 存储所有 GPU 的分析结果
completed_samples = set()  # 已完成的样本 ID


def detect_available_gpus():
    """自动检测所有可用的 GPU"""
    gpus = []
    
    # 方法1: 使用 nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--list-gpus'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    # 格式通常是 "GPU 0: NVIDIA GeForce RTX 3090 (UUID: ...)"
                    parts = line.split(':')
                    if len(parts) > 0:
                        gpu_id_str = parts[0].replace('GPU', '').strip()
                        try:
                            gpu_id = int(gpu_id_str)
                            gpus.append(gpu_id)
                        except ValueError:
                            continue
            if gpus:
                return sorted(gpus)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        pass
    
    # 方法2: 使用 PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpus = list(range(num_gpus))
            if gpus:
                return gpus
    except ImportError:
        pass
    except Exception as e:
        pass
    
    # 如果都失败了，返回默认值或空列表
    if not gpus:
        return GPUS
    
    return sorted(gpus)


def get_conda_init_script():
    """查找 conda 初始化脚本"""
    possible_paths = [
        '/mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh',
        os.path.expanduser('~/miniconda/etc/profile.d/conda.sh'),
        os.path.expanduser('~/anaconda3/etc/profile.d/conda.sh'),
        os.path.expanduser('~/miniconda3/etc/profile.d/conda.sh'),
        '/opt/conda/etc/profile.d/conda.sh',
    ]
    
    if 'CONDA_PREFIX' in os.environ:
        conda_base = os.path.dirname(os.path.dirname(os.environ['CONDA_PREFIX']))
        conda_init_script = f'{conda_base}/etc/profile.d/conda.sh'
        if os.path.exists(conda_init_script):
            return conda_init_script
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def find_samples(audio_dir, whisperx_dir, num_samples=None, start_idx=0, end_idx=-1):
    """查找所有需要分析的样本"""
    audio_dir = Path(audio_dir)
    whisperx_dir = Path(whisperx_dir)
    
    all_audio_files = sorted(audio_dir.glob("*.flac"))
    
    if start_idx != 0 or end_idx != -1:
        start = start_idx
        end = end_idx if end_idx > 0 else None
        audio_files = all_audio_files[start:end]
    elif num_samples and num_samples > 0:
        audio_files = all_audio_files[:num_samples]
    else:
        audio_files = all_audio_files
    
    samples = []
    for audio_file in audio_files:
        sample_id = audio_file.stem
        json_file = whisperx_dir / f"{sample_id}_whisperx.json"
        
        if json_file.exists():
            samples.append({
                'audio_path': str(audio_file),
                'json_path': str(json_file),
                'sample_id': sample_id
            })
    
    return samples


def check_sample_completed(sample_id, output_dir):
    """检查样本是否已经完成分析"""
    output_dir = Path(output_dir)
    word_attention_file = output_dir / "word_attention_analysis.json"
    
    if word_attention_file.exists():
        try:
            with open(word_attention_file, 'r') as f:
                sample_results = json.load(f)
                for sample in sample_results:
                    if sample.get('sample_name') == sample_id:
                        return True
        except:
            pass
    
    return False


def run_sample_range_analysis(start_idx, end_idx, gpu_id, output_dir, model_path, audio_dir, whisperx_dir,
                              score_threshold=0.95, topk=32):
    """在指定 GPU 上运行样本范围的分析"""
    # 直接使用统一的输出目录，不创建 GPU 特定目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建命令
    conda_init = get_conda_init_script()
    
    cmd_parts = [
        f"CUDA_VISIBLE_DEVICES={gpu_id}",
        f"python {SCRIPT_PATH}",
        f"--model_path {model_path}",
        f"--audio_dir {audio_dir}",
        f"--whisperx_dir {whisperx_dir}",
        f"--output_dir {output_dir}",
        f"--start_idx {start_idx}",
        f"--end_idx {end_idx}",
        f"--score_threshold {score_threshold}",
        f"--topk {topk}",
        f"--device cuda",
        f"--alpha 1.0",  # 分析阶段不需要参数搜索
        f"--beta 1.0",
        f"--gamma 1.0"
    ]
    
    cmd = " ".join(cmd_parts)
    
    if conda_init and os.path.exists(conda_init):
        cmd = f"source {conda_init} && conda activate Qwen3_2 && {cmd}"
    
    error_file = output_dir / f"analysis_range_{start_idx}_{end_idx}_error.log"
    
    start_time = time.time()
    
    try:
        # 将详细输出重定向到 /dev/null，只保留错误信息
        # 这样日志文件中就不会有样本级别的详细输出了，只显示任务级别的信息
        with open(error_file, 'w', encoding='utf-8') as err_file:
            process = subprocess.Popen(
                cmd,
                shell=True,
                executable='/bin/bash',
                stdout=subprocess.DEVNULL,  # 抑制详细输出
                stderr=err_file,
                text=True,
                cwd=str(Path(SCRIPT_PATH).parent)
            )
            
            # 等待完成，定期检查状态
            while True:
                return_code = process.poll()
                if return_code is not None:
                    break
                time.sleep(30)
        
        elapsed_time = time.time() - start_time
        
        # 检查结果文件
        result_file = output_dir / "word_attention_analysis.json"
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    results = json.load(f)
                    if results and len(results) > 0:
                        num_samples = len(results)
                        return True, num_samples
            except Exception:
                pass
        
        return False, 0
        
    except Exception:
        return False, 0


def worker(gpu_id, task_queue, output_dir, model_path, audio_dir, whisperx_dir,
           score_threshold, topk, progress_counter):
    """工作线程：从队列中获取任务并在指定 GPU 上执行（并行执行）"""
    while True:
        task = task_queue.get()
        if task is None:  # 结束信号
            break
        
        task_idx, start_idx, end_idx = task
        
        success, num_samples = run_sample_range_analysis(
            start_idx, end_idx, gpu_id, output_dir, model_path, audio_dir, whisperx_dir,
            score_threshold, topk
        )
        
        if success:
            with results_lock:
                completed_samples.add(f"range_{start_idx}_{end_idx}")
        
        # 更新进度（任务级别，不是样本级别）
        with progress_counter['lock']:
            progress_counter['completed'] += num_samples if success else 0
            progress_counter['ranges_completed'] += 1
        
        task_queue.task_done()


def load_results(output_dir):
    """加载分析结果（所有结果都在统一的输出目录中）"""
    output_dir = Path(output_dir)
    result_file = output_dir / "word_attention_analysis.json"
    
    all_sample_results = []
    
    if result_file.exists():
        try:
            with open(result_file, 'r') as f:
                all_sample_results = json.load(f)
        except Exception:
            pass
    
    # 不保存合并后的结果文件（文件太大）
    merged_output_dir = Path(output_dir) / "merged_results"
    merged_output_dir.mkdir(parents=True, exist_ok=True)
    
    return merged_output_dir, all_sample_results


def run_single_parameter_search(alpha, beta, gamma, gpu_id, output_dir, 
                                layer_head_data, topk, num_layers, num_heads):
    """在指定 GPU 上运行单个参数组合的搜索"""
    import numpy as np
    
    search_dir = Path(output_dir) / "parameter_search"
    search_dir.mkdir(parents=True, exist_ok=True)
    
    param_str = f"alpha_{alpha:.2f}_beta_{beta:.2f}_gamma_{gamma:.2f}".replace('.', '_')
    output_file = search_dir / f"audio_head_scores_aggregated_{param_str}.json"
    
    # 检查是否已经存在结果
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                complete_format = json.load(f)
            scores_list = [v[0] for v in complete_format.values()]
            mean_score = np.mean(scores_list)
            sorted_scores = sorted(scores_list, reverse=True)
            top_20_mean = np.mean(sorted_scores[:20])
            return {
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'statistics': {
                    'mean_score': float(mean_score),
                    'std_score': float(np.std(scores_list)),
                    'top_20_mean': float(top_20_mean)
                }
            }
        except:
            pass
    
    start_time = time.time()
    
    try:
        # 计算每个 layer/head 的统计
        layer_head_summary = {}
        for key, data in layer_head_data.items():
            if not data['topk_hits_on_word']:
                continue
            
            attn_ratios = [
                w / a if a > 0 else 0.0
                for w, a in zip(data['attn_weight_on_word'], data['attn_weight_on_audio'])
            ]
            
            hit_ratios = [
                hits / topk if topk > 0 else 0.0
                for hits in data['topk_hits_on_word']
            ]
            
            coverage_ratios = [
                hits / num_tokens if num_tokens > 0 else 0.0
                for hits, num_tokens in zip(data['topk_hits_on_word'], data['num_word_audio_tokens'])
            ]
            
            S_hit = float(np.mean(hit_ratios))
            S_attn = float(np.mean(attn_ratios))
            S_cov = float(np.mean(coverage_ratios))
            
            importance_score = alpha * S_hit + beta * S_attn + gamma * S_cov
            
            layer_head_summary[key] = {
                'avg_attn_ratio': S_attn,
                'median_attn_ratio': float(np.median(attn_ratios)),
                'importance_score': importance_score
            }
        
        # 生成聚合格式
        visual_format = {}
        for key, stats in layer_head_summary.items():
            if key.startswith("L") and "_H" in key:
                parts = key[1:].split("_H")
                if len(parts) == 2:
                    layer = int(parts[0])
                    head = int(parts[1])
                    visual_key = f"{layer}-{head}"
                    visual_format[visual_key] = [stats['avg_attn_ratio'], stats['median_attn_ratio']]
        
        # 确保所有头都存在
        complete_format = {}
        for layer in range(num_layers):
            for head in range(num_heads):
                key = f"{layer}-{head}"
                if key in visual_format:
                    complete_format[key] = visual_format[key]
                else:
                    complete_format[key] = [0.0, 0.0]
        
        # 计算统计信息
        scores_list = [v[0] for v in complete_format.values()]
        mean_score = np.mean(scores_list)
        std_score = np.std(scores_list)
        sorted_scores = sorted(scores_list, reverse=True)
        top_20_mean = np.mean(sorted_scores[:20])
        
        result = {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'statistics': {
                'mean_score': float(mean_score),
                'std_score': float(std_score),
                'top_20_mean': float(top_20_mean)
            }
        }
        
        # 保存结果
        with open(output_file, 'w') as f:
            json.dump(complete_format, f, indent=4)
        
        return result
        
    except Exception:
        return None


def search_worker(gpu_id, task_queue, output_dir, layer_head_data, 
                  topk, num_layers, num_heads, progress_counter):
    """参数搜索工作线程：从队列中获取参数组合任务并执行"""
    while True:
        task = task_queue.get()
        if task is None:  # 结束信号
            break
        
        alpha, beta, gamma, task_idx, total_tasks = task
        result = run_single_parameter_search(
            alpha, beta, gamma, gpu_id, output_dir,
            layer_head_data, topk, num_layers, num_heads
        )
        
        # 更新进度
        with progress_counter['lock']:
            progress_counter['completed'] += 1
            pass
        
        # 保存结果（线程安全）
        if result:
            with results_lock:
                progress_counter['results'].append(result)
        
        task_queue.task_done()


def run_parameter_search_on_merged(merged_output_dir, output_dir, alpha_range, beta_range, gamma_range,
                                   num_layers=36, num_heads=16, gpus=None, topk=32, sample_results=None):
    """在合并结果上运行参数搜索 - 使用 GPU 池并行处理每个参数组合"""
    import numpy as np
    import itertools
    
    # 如果提供了 sample_results 数据，直接使用；否则尝试从文件加载
    if sample_results is None:
        merged_file = merged_output_dir / "word_attention_analysis.json"
        if not merged_file.exists():
            return
        with open(merged_file, 'r') as f:
            sample_results = json.load(f)
    
    # 从 sample_results 中提取数据
    all_tokens = [token for sample in sample_results for token in sample.get('tokens', [])]
    
    if not all_tokens:
        return
    
    # 计算每个 layer/head 的统计（一次性计算，所有参数组合共享）
    layer_head_data = defaultdict(lambda: {
        'topk_hits_on_word': [],
        'topk_hits_on_audio': [],
        'attn_weight_on_word': [],
        'attn_weight_on_audio': [],
        'num_word_audio_tokens': []
    })
    
    for token in all_tokens:
        num_word_audio_tokens = token.get('num_word_audio_tokens', 0)
        layer_head_stats = token.get('layer_head_stats', {})
        
        for key, stats in layer_head_stats.items():
            layer_head_data[key]['topk_hits_on_word'].append(stats.get('topk_hits_on_word', 0))
            layer_head_data[key]['topk_hits_on_audio'].append(stats.get('topk_hits_on_audio', 0))
            layer_head_data[key]['attn_weight_on_word'].append(stats.get('attn_weight_on_word', 0.0))
            layer_head_data[key]['attn_weight_on_audio'].append(stats.get('attn_weight_on_audio', 0.0))
            layer_head_data[key]['num_word_audio_tokens'].append(num_word_audio_tokens)
    
    # 创建所有参数组合任务
    all_combinations = list(itertools.product(alpha_range, beta_range, gamma_range))
    total_combinations = len(all_combinations)
    
    # 如果未指定GPU，使用默认GPU列表
    if gpus is None:
        gpus = GPUS
    
    # 创建任务队列
    task_queue = Queue()
    for idx, (alpha, beta, gamma) in enumerate(all_combinations, 1):
        task_queue.put((alpha, beta, gamma, idx, total_combinations))
    
    # 进度计数器
    progress_counter = {
        'total': total_combinations,
        'completed': 0,
        'results': [],
        'lock': threading.Lock()
    }
    
    # 启动工作线程（每个 GPU 一个线程）
    threads = []
    for gpu_id in gpus:
        t = threading.Thread(
            target=search_worker,
            args=(gpu_id, task_queue, output_dir, layer_head_data,
                  topk, num_layers, num_heads, progress_counter)
        )
        t.start()
        threads.append(t)
    
    # 等待所有任务完成
    task_queue.join()
    
    # 停止所有工作线程
    for _ in gpus:
        task_queue.put(None)
    
    for t in threads:
        t.join()
    
    # 保存搜索结果汇总
    search_dir = Path(output_dir) / "parameter_search"
    search_dir.mkdir(parents=True, exist_ok=True)
    
    search_results = progress_counter['results']
    search_summary = {
        'alpha_range': list(alpha_range),
        'beta_range': list(beta_range),
        'gamma_range': list(gamma_range),
        'total_combinations': total_combinations,
        'results': search_results
    }
    search_summary_file = search_dir / "parameter_search_summary.json"
    with open(search_summary_file, 'w') as f:
        json.dump(search_summary, f, indent=2)
    
    # 找出最佳参数
    if search_results:
        search_results_sorted = sorted(search_results, key=lambda x: x['statistics']['top_20_mean'], reverse=True)
        best_result = search_results_sorted[0]
        
        # 生成最终文件（使用参数作为文件名后缀）
        best_param_str = f"alpha_{best_result['alpha']:.2f}_beta_{best_result['beta']:.2f}_gamma_{best_result['gamma']:.2f}".replace('.', '_')
        best_file = search_dir / f"audio_head_scores_aggregated_{best_param_str}.json"
        final_output_file = Path(output_dir) / "head_score" / f"audio_head_scores_aggregated_{best_param_str}.json"
        final_output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if best_file.exists():
            import shutil
            shutil.copy(best_file, final_output_file)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-GPU head scores generation and parameter search')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--audio_dir', type=str, default=DEFAULT_AUDIO_DIR)
    parser.add_argument('--whisperx_dir', type=str, default=DEFAULT_WHISPERX_DIR)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--score_threshold', type=float, default=0.95)
    parser.add_argument('--topk', type=int, default=32)
    parser.add_argument('--gpus', type=str, default=None, help='Comma-separated GPU IDs (e.g., "0,1,4,6"). If not specified, will auto-detect all available GPUs.')
    parser.add_argument('--enable_search', action='store_true')
    parser.add_argument('--alpha_range', type=str, default='0.0,0.5,1.0,1.5,2.0')
    parser.add_argument('--beta_range', type=str, default='0.0,0.5,1.0,1.5,2.0')
    parser.add_argument('--gamma_range', type=str, default='0.0,0.5,1.0,1.5,2.0')
    parser.add_argument('--num_layers', type=int, default=36)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--skip_analysis', action='store_true', help='Skip analysis, only do parameter search')
    
    args = parser.parse_args()
    
    # 解析 GPU 列表：优先使用用户指定的，否则自动检测
    if args.gpus:
        # 用户明确指定了 GPU
        gpus = [int(x.strip()) for x in args.gpus.split(',')]
    else:
        # 自动检测所有可用的 GPU
        gpus = detect_available_gpus()
    
    if not gpus:
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 步骤1: 样本分析（如果未跳过）
    if not args.skip_analysis:
        samples = find_samples(args.audio_dir, args.whisperx_dir, 
                              args.num_samples, args.start_idx, args.end_idx)
        
        if not samples:
            return
        
        # 将样本分成多个任务（每个任务是一个样本范围）
        # 多个任务会并行执行，每个 GPU 同时处理一个任务
        num_tasks = len(gpus)  # 任务数量等于 GPU 数量
        samples_per_task = len(samples) // num_tasks if num_tasks > 0 else len(samples)
        remainder = len(samples) % num_tasks if num_tasks > 0 else 0
        
        # 创建任务队列（每个任务是一个样本范围）
        task_queue = Queue()
        current_idx = 0
        
        for i in range(num_tasks):
            # 分配样本范围
            range_size = samples_per_task + (1 if i < remainder else 0)
            if range_size > 0:
                start_idx = current_idx
                end_idx = current_idx + range_size
                task_queue.put((i + 1, start_idx, end_idx))  # (task_idx, start_idx, end_idx)
                current_idx = end_idx
        
        # 进度计数器
        progress_counter = {
            'total': len(samples),
            'completed': 0,
            'total_ranges': task_queue.qsize(),
            'ranges_completed': 0,
            'lock': threading.Lock()
        }
        
        # 并行执行任务：多个 GPU 同时工作
        # 启动工作线程（每个 GPU 一个线程）
        threads = []
        for gpu_id in gpus:
            t = threading.Thread(
                target=worker,
                args=(gpu_id, task_queue, args.output_dir, args.model_path,
                     args.audio_dir, args.whisperx_dir, args.score_threshold,
                     args.topk, progress_counter)
            )
            t.start()
            threads.append(t)
        
        # 等待所有任务完成
        task_queue.join()
        
        # 停止所有工作线程
        for _ in gpus:
            task_queue.put(None)
        
        for t in threads:
            t.join()
        
        # 加载结果
        merged_output_dir, merged_results = load_results(args.output_dir)
    else:
        # 如果跳过分析，直接使用已有的合并结果
        merged_output_dir = Path(args.output_dir) / "merged_results"
        # 如果跳过分析，需要从文件加载数据（如果文件存在）
        merged_file = merged_output_dir / "word_attention_analysis.json"
        if merged_file.exists():
            with open(merged_file, 'r') as f:
                merged_results = json.load(f)
        else:
            merged_results = None
    
    # 步骤2: 参数搜索（如果启用）
    if args.enable_search:
        alpha_range = [float(x.strip()) for x in args.alpha_range.split(',')]
        beta_range = [float(x.strip()) for x in args.beta_range.split(',')]
        gamma_range = [float(x.strip()) for x in args.gamma_range.split(',')]
        
        if merged_results is not None:
            run_parameter_search_on_merged(
                merged_output_dir, args.output_dir,
                alpha_range, beta_range, gamma_range,
                args.num_layers, args.num_heads,
                gpus=gpus, topk=args.topk, sample_results=merged_results
            )


if __name__ == "__main__":
    main()

