# SparseKV KVCache 加速优化指南

## 问题分析

经过代码分析，发现当前 SparseKV 方案加速不明显的主要原因：

### 1. CUDA Kernel 回退问题

当前 `DynamicCacheSplitHeadFlatten.update()` 方法在 decode 阶段有多个回退路径：

```python
# 问题代码 (sparsemm_utils.py:350-380)
if head_lens is None or cu_klen is None:
    # 回退到 PyTorch 实现
    self.key_cache[layer_idx] = torch.cat([...])
```

**问题**：
- `head_lens`/`cu_klen` 未正确初始化时回退
- dtype 不是 int32 时回退
- CUDA kernel 调用失败时回退

每次回退到 `torch.cat` 会产生：
- 内存分配开销
- 数据复制开销
- GPU kernel launch overhead

### 2. Per-Head 循环的低效率

```python
# 问题代码 (sparsemm_utils.py:780-820)
for head_idx in range(num_heads//self.num_key_value_groups):
    cache_index = indices[head_idx][...,:head_capacity]
    top_Kcache = origin_heads_key_states[head_idx].gather(dim=2, index=cache_index)
    # ...
```

**问题**：
- 72 次循环 (36 layers × 2 gather ops)
- 每次 gather 都是一个独立的 CUDA kernel
- 无法利用 GPU 的批量并行能力

### 3. Attention Score 重复计算

每个 sample 的 prefill 阶段都完整计算 attention score：
```python
attn_weights = torch.matmul(query_states[..., -self.window_size:, :], 
                            key_states.transpose(2, 3)) / math.sqrt(head_dim)
```

对于长音频（3000+ tokens），这部分计算量可观。

### 4. 内存分配频繁

每次 `update_kv` 都创建新的 tensor 列表：
```python
heads_key_states = []
heads_value_states = []
# ... 循环中 append
heads_key_states = torch.cat(heads_key_states, dim=0)
```

---

## 优化方案

### 方案 1: 确保 CUDA Kernel 正确调用

**检查清单**：

```python
# 1. 确认 tiny_api_cuda 已正确编译安装
import tiny_api_cuda
print("✓ CUDA kernel available")

# 2. 验证 metadata 初始化
def verify_metadata(kv_cluster):
    assert kv_cluster.head_lens is not None
    assert kv_cluster.head_lens.dtype == torch.int32
    assert kv_cluster.cu_klen.dtype == torch.int32
    print(f"✓ head_lens: {kv_cluster.head_lens.shape}, dtype={kv_cluster.head_lens.dtype}")
    print(f"✓ cu_klen: {kv_cluster.cu_klen.shape}, dtype={kv_cluster.cu_klen.dtype}")
```

**修复**：在 `init_metadata` 中强制使用 int32：

```python
def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
    # 确保 dtype 正确
    self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
    self.cu_klen = torch.cat([
        self.cu_klen,
        torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)
    ], dim=0)
```

### 方案 2: 批量化 Gather 操作

**优化思路**：将所有 head 的 TopK 索引合并为一次 gather

```python
def batched_gather_kv(key_states, value_states, all_indices, head_capacities, window_size):
    """
    key_states: [bs, num_kv_heads, seq_len, head_dim]
    all_indices: [bs, num_kv_heads, max_k] - 预先排好的 TopK 索引
    head_capacities: [num_kv_heads] - 每个 head 的实际 k 值
    """
    bs, num_kv_heads, seq_len, head_dim = key_states.shape
    
    # 使用高级索引一次性 gather
    # 构建 batch indices
    batch_idx = torch.arange(bs, device=key_states.device).view(-1, 1, 1, 1)
    head_idx = torch.arange(num_kv_heads, device=key_states.device).view(1, -1, 1, 1)
    
    # 扩展 indices
    indices_expanded = all_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    
    # 一次性 gather
    gathered = key_states.gather(2, indices_expanded)
    
    return gathered
```

### 方案 3: 预计算和缓存

**缓存 Attention Mask**：

```python
class SparseMM_Optimized:
    def __init__(self, ...):
        self._cached_mask = None
        self._cached_mask_size = 0
    
    def get_cached_mask(self, window_size, device, dtype):
        if self._cached_mask is None or self._cached_mask_size != window_size:
            mask = torch.full((window_size, window_size), 
                            torch.finfo(dtype).min, device=device)
            mask_cond = torch.arange(mask.size(-1), device=device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(-1, 1), 0)
            self._cached_mask = mask[None, None, :, :]
            self._cached_mask_size = window_size
        return self._cached_mask.to(device=device, dtype=dtype)
```

**缓存 Head Capacity**：

```python
# 在初始化时计算好，避免每次 update_kv 重复计算
self.head_adaptive_capacity = self._compute_head_capacity()
self.head_adaptive_capacity = self.head_adaptive_capacity.to('cuda')  # 预先放到 GPU
```

### 方案 4: 使用 Triton 自定义 Kernel

对于频繁调用的操作，可以用 Triton 编写高效 kernel：

```python
import triton
import triton.language as tl

@triton.jit
def topk_gather_kernel(
    key_ptr, output_ptr, indices_ptr,
    seq_len, head_dim, k,
    BLOCK_SIZE: tl.constexpr
):
    """
    融合 TopK 选择和 Gather 的自定义 kernel
    """
    pid = tl.program_id(0)
    
    # 计算当前处理的位置
    row = pid // k
    col = pid % k
    
    # 读取索引
    idx = tl.load(indices_ptr + row * k + col)
    
    # Gather 数据
    for d in range(0, head_dim, BLOCK_SIZE):
        offsets = d + tl.arange(0, BLOCK_SIZE)
        mask = offsets < head_dim
        
        src_ptr = key_ptr + row * seq_len * head_dim + idx * head_dim + offsets
        dst_ptr = output_ptr + row * k * head_dim + col * head_dim + offsets
        
        data = tl.load(src_ptr, mask=mask)
        tl.store(dst_ptr, data, mask=mask)
```

### 方案 5: 异步流水线

利用 CUDA Stream 重叠计算和内存传输：

```python
class AsyncKVCache:
    def __init__(self):
        self.compute_stream = torch.cuda.Stream()
        self.copy_stream = torch.cuda.Stream()
    
    def update_async(self, key_states, value_states, indices):
        # 在 compute_stream 上计算 attention score
        with torch.cuda.stream(self.compute_stream):
            scores = self.compute_attention_scores(...)
            indices = scores.topk(k, dim=-1).indices
        
        # 等待 indices 准备好
        self.copy_stream.wait_stream(self.compute_stream)
        
        # 在 copy_stream 上执行 gather
        with torch.cuda.stream(self.copy_stream):
            gathered = self.gather_kv(key_states, indices)
        
        return gathered
```

### 方案 6: 分层压缩策略

不同层使用不同的压缩策略：

```python
class LayerAdaptiveCompression:
    """
    观察：
    - 浅层（0-10）：倾向于保留更多 audio token
    - 中层（10-25）：可以更激进压缩
    - 深层（25-35）：保留语义关键 token
    """
    
    def get_layer_strategy(self, layer_idx, num_layers):
        if layer_idx < num_layers * 0.3:
            # 浅层：保守压缩
            return {'compression_ratio': 0.7, 'window_boost': 1.2}
        elif layer_idx < num_layers * 0.7:
            # 中层：激进压缩
            return {'compression_ratio': 0.4, 'window_boost': 1.0}
        else:
            # 深层：中等压缩
            return {'compression_ratio': 0.5, 'window_boost': 1.1}
```

---

## 性能测试方法

### 1. 隔离测试 CUDA Kernel

```python
import torch
from tiny_api_cuda import update_flatten_view
import time

def benchmark_cuda_kernel():
    # 模拟 decode 阶段的 cache 更新
    cache = torch.randn(8192, 128, device='cuda', dtype=torch.float16)
    state = torch.randn(8, 128, device='cuda', dtype=torch.float16)
    head_lens = torch.tensor([1024] * 8, dtype=torch.int32, device='cuda')
    cu_klen = torch.cumsum(head_lens, dim=0, dtype=torch.int32)
    cu_klen = torch.cat([torch.tensor([0], dtype=torch.int32, device='cuda'), cu_klen])
    
    # Warmup
    for _ in range(10):
        _ = update_flatten_view(cache, state, head_lens, cu_klen)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        _ = update_flatten_view(cache, state, head_lens, cu_klen)
    end.record()
    
    torch.cuda.synchronize()
    print(f"CUDA kernel avg: {start.elapsed_time(end) / 100:.3f} ms")

benchmark_cuda_kernel()
```

### 2. 对比 PyTorch Fallback

```python
def benchmark_pytorch_fallback():
    cache = torch.randn(8192, 128, device='cuda', dtype=torch.float16)
    state = torch.randn(8, 128, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        _ = torch.cat([cache, state], dim=0)
    
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        result = torch.cat([cache, state], dim=0)
        cache = result[:-8]  # 模拟下次使用
    end.record()
    
    torch.cuda.synchronize()
    print(f"PyTorch cat avg: {start.elapsed_time(end) / 100:.3f} ms")

benchmark_pytorch_fallback()
```

### 3. Profiling 整体 Generate

```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, 
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    output = model.generate(...)

# 查看 Top 操作
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# 导出 Chrome trace
prof.export_chrome_trace("trace.json")
```

---

## 快速检查清单

1. **检查 CUDA Kernel 是否被调用**
   ```python
   # 在 DynamicCacheSplitHeadFlatten.update 中添加
   if not self._cuda_kernel_used_logged:
       print("[SparseMM] ✓ Using CUDA kernel")
       self._cuda_kernel_used_logged = True
   ```

2. **检查 metadata 初始化**
   ```python
   print(f"head_lens: {kv_cluster.head_lens}")
   print(f"cu_klen: {kv_cluster.cu_klen}")
   ```

3. **测量各阶段耗时**
   ```python
   # Prefill attention score 计算
   # TopK 选择
   # Gather 操作
   # Cache 更新
   ```

4. **内存使用监控**
   ```python
   print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
   ```

---

## 推荐优化顺序

1. **立即修复**：确保 CUDA kernel 被正确调用（检查 metadata dtype）
2. **快速优化**：添加 attention mask 缓存
3. **中期优化**：实现批量化 gather
4. **长期优化**：编写 Triton 自定义 kernel

预计优化效果：
- 修复 CUDA kernel 回退：~20-30% 加速
- 批量化 gather：~10-20% 加速  
- Mask 缓存：~5-10% 加速
- Triton kernel：~30-50% 加速（取决于实现质量）
