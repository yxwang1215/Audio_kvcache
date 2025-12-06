#!/usr/bin/env python3
"""
超参数搜索脚本：搜索图中空白部分的 alpha 和 cutoff 组合
使用4个GPU轮询分配，完成实验后记录结果到txt文件
"""

import os
import sys
import json
import subprocess
import time
import threading
from queue import Queue
from collections import defaultdict
from pathlib import Path

# 根据图中的数据，定义已有的组合（不需要重新搜索）
EXISTING_COMBINATIONS = {
    0: [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8,0.9,1.0],
    0.3: [0.45,0.5,0.55],
    0.4: [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8],
    0.5: [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8],
    0.55: [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8],
    0.6: [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8],
    0.65: [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8],
    0.7: [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8],
    0.8: [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8],
    0.9: [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8],
}

# 所有需要搜索的参数值
ALPHA_VALUES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1]
CUTOFF_VALUES = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]

# GPU列表
GPUS = [0, 5, 6, 7]

# 基础路径
BASE_DIR = "/mnt/users/wangyuxuan-20250915/Audio"
RESULTS_BASE_DIR = f"{BASE_DIR}/Results/LibriSpeech_Long_Results/SAH_FFT"
SCRIPT_PATH = f"{BASE_DIR}/AudioMarathon/src/Qwen_2.5_Omni/sparsekv_fft/LibriSpeech_qwen2.5.py"
LOG_FILE = f"{BASE_DIR}/Audio_kvcache/qwen/sparsekv_fft/hyperparameter_search_results_lowpass.txt"

# 锁用于线程安全地更新日志文件
log_lock = threading.Lock()
results_lock = threading.Lock()


def generate_search_combinations():
    """生成所有需要搜索的 (alpha, cutoff) 组合"""
    combinations = []
    for alpha in ALPHA_VALUES:
        existing_cutoffs = set(EXISTING_COMBINATIONS.get(alpha, []))
        for cutoff in CUTOFF_VALUES:
            if cutoff not in existing_cutoffs:
                combinations.append((alpha, cutoff))
    return combinations


def get_results_dir(alpha, cutoff):
    """生成结果目录路径"""
    cutoff_str = str(cutoff).replace('.', '')
    return f"{RESULTS_BASE_DIR}/LibriSpeech_budget0.5_fft_lowpass_cutoff{cutoff}_alpha{alpha}"


def get_result_json_path(alpha, cutoff):
    """获取结果JSON文件路径"""
    results_dir = get_results_dir(alpha, cutoff)
    return f"{results_dir}/librispeech_results_qwen25.json"


def extract_accuracy(result_json_path):
    """从结果JSON文件中提取准确率"""
    try:
        with open(result_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            accuracy = data.get('word_accuracy', None)
            return accuracy
    except Exception as e:
        print(f"Error reading {result_json_path}: {e}")
        return None


def run_experiment(alpha, cutoff, gpu_id):
    """运行单个实验"""
    results_dir = get_results_dir(alpha, cutoff)
    result_json = get_result_json_path(alpha, cutoff)
    
    # 确保结果目录存在
    os.makedirs(results_dir, exist_ok=True)
    
    # 检查是否已经完成
    if os.path.exists(result_json):
        try:
            with open(result_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get('status') == 'completed':
                    accuracy = data.get('word_accuracy', None)
                    if accuracy is not None:
                        log_result(alpha, cutoff, accuracy, gpu_id, status='already_completed')
                        return True, accuracy
        except:
            pass  # 如果文件损坏，重新运行
    
    # 构建命令 - 使用source conda.sh的方式激活环境
    # 尝试找到conda的初始化脚本
    conda_init_script = None
    possible_paths = [
        '/mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh',
        os.path.expanduser('~/miniconda/etc/profile.d/conda.sh'),
        os.path.expanduser('~/anaconda3/etc/profile.d/conda.sh'),
        os.path.expanduser('~/miniconda3/etc/profile.d/conda.sh'),
        '/opt/conda/etc/profile.d/conda.sh',
    ]
    
    # 从环境变量中获取
    if 'CONDA_PREFIX' in os.environ:
        conda_base = os.path.dirname(os.path.dirname(os.environ['CONDA_PREFIX']))
        conda_init_script = f'{conda_base}/etc/profile.d/conda.sh'
    
    # 如果还没找到，尝试常见路径
    if not conda_init_script or not os.path.exists(conda_init_script):
        for path in possible_paths:
            if os.path.exists(path):
                conda_init_script = path
                break
    
    # 构建命令
    if conda_init_script and os.path.exists(conda_init_script):
        cmd = f'source {conda_init_script} && conda activate Qwen3_2 && CUDA_VISIBLE_DEVICES={gpu_id} USE_SPARSEKV=true USE_FFT_FILTER=true FFT_FILTER_TYPE=low_pass FFT_CUTOFF_RATIO={cutoff} FFT_USE_SALIENCY=true FFT_ALPHA={alpha} FFT_ADAPTIVE_CUTOFF=true BUDGET=0.5 RATIO=0.5 WINDOW_SIZE=32 HEAD_SCORE_FILE=aggregated RESULTS_DIR={results_dir} python {SCRIPT_PATH}'
    else:
        # 回退：直接使用python（假设已经在正确的环境中）
        cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} USE_SPARSEKV=true USE_FFT_FILTER=true FFT_FILTER_TYPE=low_pass FFT_CUTOFF_RATIO={cutoff} FFT_USE_SALIENCY=true FFT_ALPHA={alpha} FFT_ADAPTIVE_CUTOFF=true BUDGET=0.5 RATIO=0.5 WINDOW_SIZE=32 HEAD_SCORE_FILE=aggregated RESULTS_DIR={results_dir} python {SCRIPT_PATH}'
    
    print(f"[GPU {gpu_id}] Starting experiment: alpha={alpha}, cutoff={cutoff}")
    start_time = time.time()
    
    # 创建错误日志文件
    error_log_file = f"{results_dir}/error.log"
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # 运行实验
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        
        # 保存错误输出（如果有）
        if result.returncode != 0 or result.stderr:
            with open(error_log_file, 'w', encoding='utf-8') as f:
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"STDOUT:\n{result.stdout}\n")
                f.write(f"STDERR:\n{result.stderr}\n")
        
        # 检查结果
        if os.path.exists(result_json):
            accuracy = extract_accuracy(result_json)
            if accuracy is not None:
                log_result(alpha, cutoff, accuracy, gpu_id, elapsed_time, status='completed')
                print(f"[GPU {gpu_id}] Completed: alpha={alpha}, cutoff={cutoff}, accuracy={accuracy:.2f}%, time={elapsed_time:.1f}s")
                return True, accuracy
            else:
                log_result(alpha, cutoff, None, gpu_id, elapsed_time, status='failed_no_accuracy')
                print(f"[GPU {gpu_id}] Failed: alpha={alpha}, cutoff={cutoff} - No accuracy in results")
                if result.returncode != 0:
                    print(f"[GPU {gpu_id}] Command failed with return code {result.returncode}")
                return False, None
        else:
            log_result(alpha, cutoff, None, gpu_id, elapsed_time, status='failed_no_file')
            print(f"[GPU {gpu_id}] Failed: alpha={alpha}, cutoff={cutoff} - No result file")
            if result.returncode != 0:
                print(f"[GPU {gpu_id}] Command failed with return code {result.returncode}")
                if result.stderr:
                    print(f"[GPU {gpu_id}] Error: {result.stderr[:200]}")
            return False, None
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        log_result(alpha, cutoff, None, gpu_id, elapsed_time, status=f'error: {str(e)}')
        print(f"[GPU {gpu_id}] Error: alpha={alpha}, cutoff={cutoff} - {e}")
        return False, None


def log_result(alpha, cutoff, accuracy, gpu_id, elapsed_time=None, status='completed'):
    """记录结果到日志文件"""
    with log_lock:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            if accuracy is not None:
                f.write(f"{timestamp} | GPU {gpu_id} | alpha={alpha:.2f} | cutoff={cutoff:.2f} | accuracy={accuracy:.2f}% | {status}")
            else:
                f.write(f"{timestamp} | GPU {gpu_id} | alpha={alpha:.2f} | cutoff={cutoff:.2f} | accuracy=None | {status}")
            if elapsed_time is not None:
                f.write(f" | time={elapsed_time:.1f}s")
            f.write("\n")
            f.flush()


def worker(gpu_id, task_queue, results_dict, progress_counter):
    """工作线程：从队列中获取任务并执行"""
    while True:
        task = task_queue.get()
        if task is None:  # 结束信号
            break
        
        alpha, cutoff = task
        success, accuracy = run_experiment(alpha, cutoff, gpu_id)
        
        if success and accuracy is not None:
            with results_lock:
                results_dict[(alpha, cutoff)] = accuracy
        
        # 更新进度
        with progress_counter['lock']:
            progress_counter['completed'] += 1
            total = progress_counter['total']
            completed = progress_counter['completed']
            print(f"\n[Progress] {completed}/{total} ({100*completed/total:.1f}%) completed\n")
        
        task_queue.task_done()


def main():
    """主函数"""
    # 生成需要搜索的组合
    combinations = generate_search_combinations()
    print(f"Total combinations to search: {len(combinations)}")
    print(f"Existing combinations: {sum(len(v) for v in EXISTING_COMBINATIONS.values())}")
    
    # 初始化日志文件
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Hyperparameter Search Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total combinations: {len(combinations)}\n")
        f.write("=" * 100 + "\n")
    
    # 创建任务队列
    task_queue = Queue()
    for combo in combinations:
        task_queue.put(combo)
    
    # 用于存储结果
    results_dict = {}
    
    # 进度计数器
    progress_counter = {
        'total': len(combinations),
        'completed': 0,
        'lock': threading.Lock()
    }
    
    # 启动工作线程
    threads = []
    for gpu_id in GPUS:
        t = threading.Thread(target=worker, args=(gpu_id, task_queue, results_dict, progress_counter))
        t.start()
        threads.append(t)
    
    # 等待所有任务完成
    task_queue.join()
    
    # 停止所有工作线程
    for _ in GPUS:
        task_queue.put(None)
    
    for t in threads:
        t.join()
    
    # 生成最终报告
    print("\n" + "=" * 80)
    print("Search completed!")
    print(f"Total combinations: {len(combinations)}")
    print(f"Successful: {len(results_dict)}")
    print(f"Failed: {len(combinations) - len(results_dict)}")
    
    # 保存结果摘要
    summary_file = LOG_FILE.replace('.txt', '_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Hyperparameter Search Summary\n")
        f.write("=" * 100 + "\n")
        f.write(f"Total combinations: {len(combinations)}\n")
        f.write(f"Successful: {len(results_dict)}\n")
        f.write(f"Failed: {len(combinations) - len(results_dict)}\n\n")
        
        # 按alpha分组输出
        for alpha in sorted(set(a for a, _ in results_dict.keys())):
            f.write(f"\nAlpha = {alpha}:\n")
            cutoffs = sorted([c for a, c in results_dict.keys() if a == alpha])
            for cutoff in cutoffs:
                accuracy = results_dict[(alpha, cutoff)]
                f.write(f"  Cutoff = {cutoff:.2f}: {accuracy:.2f}%\n")
    
    print(f"\nResults saved to: {LOG_FILE}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()

