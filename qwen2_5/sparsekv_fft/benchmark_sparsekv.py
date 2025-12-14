#!/usr/bin/env python3
"""
SparseKV 性能诊断脚本

用于分析 KVCache 加速不明显的原因，定位性能瓶颈
"""

import os
import sys
import time
import torch
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

# 设置环境
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

@dataclass
class TimingResult:
    name: str
    total_ms: float = 0.0
    count: int = 0
    samples: List[float] = field(default_factory=list)
    
    @property
    def avg_ms(self):
        return self.total_ms / self.count if self.count > 0 else 0.0
    
    def add(self, elapsed_ms: float):
        self.total_ms += elapsed_ms
        self.count += 1
        self.samples.append(elapsed_ms)


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.timings: Dict[str, TimingResult] = {}
        self.cuda_events: Dict[str, tuple] = {}
        self.cuda_available = torch.cuda.is_available()
    
    def start(self, name: str):
        if self.cuda_available:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.cuda_events[name] = (start, end)
    
    def stop(self, name: str) -> float:
        if name not in self.cuda_events:
            return 0.0
        
        start, end = self.cuda_events[name]
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        
        if name not in self.timings:
            self.timings[name] = TimingResult(name)
        self.timings[name].add(elapsed)
        
        del self.cuda_events[name]
        return elapsed
    
    def report(self):
        print("\n" + "="*60)
        print("Performance Profile Report")
        print("="*60)
        
        sorted_timings = sorted(self.timings.values(), key=lambda x: x.total_ms, reverse=True)
        
        total_time = sum(t.total_ms for t in sorted_timings)
        
        print(f"{'Operation':<40} {'Total(ms)':<12} {'Avg(ms)':<10} {'Count':<8} {'%':<8}")
        print("-"*80)
        
        for t in sorted_timings:
            pct = (t.total_ms / total_time * 100) if total_time > 0 else 0
            print(f"{t.name:<40} {t.total_ms:<12.2f} {t.avg_ms:<10.3f} {t.count:<8} {pct:<8.1f}")
        
        print("-"*80)
        print(f"{'TOTAL':<40} {total_time:<12.2f}")
        print("="*60)


def test_cuda_kernel_availability():
    """测试 CUDA Kernel 是否可用"""
    print("\n" + "="*60)
    print("Test 1: CUDA Kernel Availability")
    print("="*60)
    
    try:
        from tiny_api_cuda import update_flatten_view
        print("✓ tiny_api_cuda imported successfully")
        
        # 测试 kernel 调用
        cache = torch.randn(1024, 128, device='cuda', dtype=torch.float16)
        state = torch.randn(4, 128, device='cuda', dtype=torch.float16)
        head_lens = torch.tensor([256, 256, 256, 256], dtype=torch.int32, device='cuda')
        cu_klen = torch.tensor([0, 256, 512, 768, 1024], dtype=torch.int32, device='cuda')
        
        result = update_flatten_view(cache, state, head_lens, cu_klen)
        print(f"✓ CUDA kernel executed successfully")
        print(f"  Input cache shape: {cache.shape}")
        print(f"  Output shape: {result.shape}")
        print(f"  Expected shape: ({1024 + 4}, 128)")
        
        return True
        
    except ImportError as e:
        print(f"✗ CUDA kernel not available: {e}")
        print("  This is a major performance bottleneck!")
        print("  Please compile tiny_api_cuda:")
        print("  cd SparseMM/csrc && pip install -e .")
        return False
    except Exception as e:
        print(f"✗ CUDA kernel execution failed: {e}")
        return False


def benchmark_cache_update():
    """对比 CUDA kernel 和 PyTorch fallback 的性能"""
    print("\n" + "="*60)
    print("Test 2: Cache Update Performance")
    print("="*60)
    
    # 模拟参数
    num_heads = 8
    head_dim = 128
    cache_len = 4096  # 模拟 4096 tokens 的 cache
    
    cache = torch.randn(cache_len, head_dim, device='cuda', dtype=torch.float16)
    state = torch.randn(num_heads, head_dim, device='cuda', dtype=torch.float16)
    head_lens = torch.tensor([cache_len // num_heads] * num_heads, dtype=torch.int32, device='cuda')
    cu_klen_values = [0] + [head_lens[:i+1].sum().item() for i in range(num_heads)]
    cu_klen = torch.tensor(cu_klen_values, dtype=torch.int32, device='cuda')
    
    num_iterations = 100
    
    # Test PyTorch fallback
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    for _ in range(10):
        _ = torch.cat([cache, state], dim=0)
    torch.cuda.synchronize()
    
    start_event.record()
    for _ in range(num_iterations):
        result = torch.cat([cache, state], dim=0)
    end_event.record()
    torch.cuda.synchronize()
    
    pytorch_time = start_event.elapsed_time(end_event) / num_iterations
    print(f"PyTorch torch.cat: {pytorch_time:.3f} ms/iter")
    
    # Test CUDA kernel
    try:
        from tiny_api_cuda import update_flatten_view
        
        # Warmup
        for _ in range(10):
            _ = update_flatten_view(cache, state, head_lens, cu_klen)
        torch.cuda.synchronize()
        
        start_event.record()
        for _ in range(num_iterations):
            result = update_flatten_view(cache, state, head_lens, cu_klen)
        end_event.record()
        torch.cuda.synchronize()
        
        cuda_time = start_event.elapsed_time(end_event) / num_iterations
        print(f"CUDA kernel: {cuda_time:.3f} ms/iter")
        print(f"Speedup: {pytorch_time / cuda_time:.2f}x")
        
    except ImportError:
        print("CUDA kernel not available, skipping")


def benchmark_attention_score():
    """测试 attention score 计算的开销"""
    print("\n" + "="*60)
    print("Test 3: Attention Score Computation")
    print("="*60)
    
    # 模拟参数 (Qwen2.5-Omni 7B)
    batch_size = 1
    num_heads = 28  # num_attention_heads
    num_kv_heads = 4  # num_key_value_heads
    head_dim = 128
    seq_len = 3000  # 音频 tokens
    window_size = 64
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
    key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
    
    # Expand key for GQA
    n_rep = num_heads // num_kv_heads
    key_expanded = key.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(batch_size, num_heads, seq_len, head_dim)
    
    num_iterations = 10
    
    # Warmup
    for _ in range(3):
        attn = torch.matmul(query[..., -window_size:, :], key_expanded.transpose(-2, -1))
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        # 计算 attention score (只用最后 window_size 的 query)
        attn = torch.matmul(query[..., -window_size:, :], key_expanded.transpose(-2, -1))
        attn = attn / (head_dim ** 0.5)
        attn = torch.softmax(attn.float(), dim=-1).to(query.dtype)
    end_event.record()
    torch.cuda.synchronize()
    
    attn_time = start_event.elapsed_time(end_event) / num_iterations
    print(f"Attention score computation: {attn_time:.3f} ms/iter")
    print(f"  Query shape: {query.shape}")
    print(f"  Key shape: {key.shape}")
    print(f"  Window size: {window_size}")


def benchmark_gather_operation():
    """测试 gather 操作的开销"""
    print("\n" + "="*60)
    print("Test 4: Gather Operation Performance")
    print("="*60)
    
    # 模拟参数
    batch_size = 1
    num_kv_heads = 4
    seq_len = 3000
    head_dim = 128
    topk = 256
    
    key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
    indices = torch.randint(0, seq_len, (batch_size, num_kv_heads, topk), device='cuda')
    
    num_iterations = 100
    
    # Method 1: Per-head gather (current implementation)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    for _ in range(10):
        for h in range(num_kv_heads):
            idx = indices[:, h:h+1, :].unsqueeze(-1).expand(-1, -1, -1, head_dim)
            _ = key[:, h:h+1, :, :].gather(2, idx)
    torch.cuda.synchronize()
    
    start_event.record()
    for _ in range(num_iterations):
        results = []
        for h in range(num_kv_heads):
            idx = indices[:, h:h+1, :].unsqueeze(-1).expand(-1, -1, -1, head_dim)
            results.append(key[:, h:h+1, :, :].gather(2, idx))
    end_event.record()
    torch.cuda.synchronize()
    
    per_head_time = start_event.elapsed_time(end_event) / num_iterations
    print(f"Per-head gather ({num_kv_heads} heads): {per_head_time:.3f} ms/iter")
    
    # Method 2: Batched gather
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    
    # Warmup
    for _ in range(10):
        _ = key.gather(2, indices_expanded)
    torch.cuda.synchronize()
    
    start_event.record()
    for _ in range(num_iterations):
        result = key.gather(2, indices_expanded)
    end_event.record()
    torch.cuda.synchronize()
    
    batched_time = start_event.elapsed_time(end_event) / num_iterations
    print(f"Batched gather: {batched_time:.3f} ms/iter")
    print(f"Speedup: {per_head_time / batched_time:.2f}x")


def benchmark_topk_operation():
    """测试 TopK 操作的开销"""
    print("\n" + "="*60)
    print("Test 5: TopK Operation Performance")
    print("="*60)
    
    # 模拟参数
    batch_size = 1
    num_kv_heads = 4
    seq_len = 3000 - 64  # 扣除 window
    topk = 256
    
    scores = torch.randn(batch_size, num_kv_heads, seq_len, device='cuda', dtype=torch.float32)
    
    num_iterations = 100
    
    # Warmup
    for _ in range(10):
        _, _ = torch.topk(scores, k=topk, dim=-1)
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # TopK sorted=True
    start_event.record()
    for _ in range(num_iterations):
        _, indices = torch.topk(scores, k=topk, dim=-1, sorted=True)
    end_event.record()
    torch.cuda.synchronize()
    
    sorted_time = start_event.elapsed_time(end_event) / num_iterations
    print(f"TopK (sorted=True): {sorted_time:.3f} ms/iter")
    
    # TopK sorted=False
    start_event.record()
    for _ in range(num_iterations):
        _, indices = torch.topk(scores, k=topk, dim=-1, sorted=False)
    end_event.record()
    torch.cuda.synchronize()
    
    unsorted_time = start_event.elapsed_time(end_event) / num_iterations
    print(f"TopK (sorted=False): {unsorted_time:.3f} ms/iter")
    print(f"Speedup: {sorted_time / unsorted_time:.2f}x")


def memory_usage_test():
    """测试内存使用情况"""
    print("\n" + "="*60)
    print("Test 6: Memory Usage")
    print("="*60)
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    initial_memory = torch.cuda.memory_allocated()
    print(f"Initial memory: {initial_memory / 1024**3:.2f} GB")
    
    # 模拟 full KV cache
    batch_size = 1
    num_kv_heads = 4
    seq_len = 3000
    head_dim = 128
    num_layers = 36
    
    full_cache_size = batch_size * num_kv_heads * seq_len * head_dim * 2 * num_layers * 2  # K + V, float16
    print(f"Full KV cache size (theoretical): {full_cache_size / 1024**3:.2f} GB")
    
    # 模拟压缩后的 cache
    compressed_len = 256 + 64  # topk + window
    compressed_cache_size = batch_size * num_kv_heads * compressed_len * head_dim * 2 * num_layers * 2
    print(f"Compressed cache size (theoretical): {compressed_cache_size / 1024**3:.2f} GB")
    print(f"Compression ratio: {compressed_cache_size / full_cache_size:.2%}")
    
    # 实际创建 tensor 测试
    key_caches = []
    for _ in range(num_layers):
        k = torch.randn(batch_size, num_kv_heads, compressed_len, head_dim, device='cuda', dtype=torch.float16)
        key_caches.append(k)
    
    after_alloc = torch.cuda.memory_allocated()
    print(f"After allocation: {after_alloc / 1024**3:.2f} GB")
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("SparseKV Performance Diagnostic Tool")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return
    
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    test_cuda_kernel_availability()
    benchmark_cache_update()
    benchmark_attention_score()
    benchmark_gather_operation()
    benchmark_topk_operation()
    memory_usage_test()
    
    print("\n" + "="*60)
    print("Diagnosis Complete")
    print("="*60)
    print("\nRecommendations:")
    print("1. If CUDA kernel is not available, compile tiny_api_cuda")
    print("2. If gather is slow, consider batched gather optimization")
    print("3. Use sorted=False in topk when order doesn't matter")
    print("4. Monitor peak memory to ensure compression is effective")


if __name__ == "__main__":
    run_all_tests()

# CUDA_VISIBLE_DEVICES=6 python3 Audio_kvcache/qwen2_5/sparsekv_fft/benchmark_sparsekv.py