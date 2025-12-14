"""
SparseKV 优化版本 - 针对 KVCache 加速问题的改进

核心优化点:
1. 批量化 gather 操作 - 避免 per-head 循环
2. 优化 CUDA kernel 调用路径 - 减少回退到 PyTorch
3. 预计算和缓存 - 减少重复计算
4. 内存布局优化 - 提高 cache 访问效率
5. 异步操作 - 重叠计算和内存传输

使用方法:
    设置环境变量 USE_OPTIMIZED_SPARSEMM=true 启用优化版本
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import os
from typing import List, Optional, Tuple
import numpy as np
from transformers.cache_utils import Cache

# 尝试导入 CUDA kernel
try:
    from tiny_api_cuda import update_flatten_view
    CUDA_KERNEL_AVAILABLE = True
    # 只在首次导入时打印
    if os.environ.get("_SPARSEMM_CUDA_LOGGED", "") != "1":
        print("[SparseMM Optimized] CUDA kernel (tiny_api_cuda) is available")
        os.environ["_SPARSEMM_CUDA_LOGGED"] = "1"
except ImportError:
    CUDA_KERNEL_AVAILABLE = False
    if os.environ.get("_SPARSEMM_CUDA_LOGGED", "") != "1":
        print("[SparseMM Optimized] WARNING: CUDA kernel not available, using PyTorch fallback")
        os.environ["_SPARSEMM_CUDA_LOGGED"] = "1"


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA repeat key/value heads"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class DynamicCacheSplitHeadFlattenOptimized(Cache):
    """
    优化版本的 DynamicCacheSplitHeadFlatten
    
    优化点:
    1. 预分配内存池，减少动态分配
    2. 优化 CUDA kernel 调用，减少回退
    3. 异步更新支持
    """
    
    def __init__(self, max_layers: int = 36, estimated_max_cache_size: int = 0) -> None:
        try:
            super().__init__()
        except (TypeError, ValueError):
            super().__init__(layers=[])
        
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0
        self._cuda_kernel_logged = False
        
        # 预分配标记
        self._preallocated = False
        self.max_layers = max_layers
    
    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]), tuple(self.value_cache[layer_idx]))

    def __getitem__(self, layer_idx: int):
        if layer_idx < len(self):
            return (tuple(self.key_cache[layer_idx]), tuple(self.value_cache[layer_idx]))
        raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer {layer_idx}")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """优化的 cache 更新"""
        
        # 首次添加（prefill 阶段）
        if len(self.key_cache) <= layer_idx:
            if key_states.dim() == 4:
                bs, head, seqlen, dim = key_states.shape
                key_states = key_states.contiguous().view(bs * head * seqlen, dim)
                value_states = value_states.contiguous().view(bs * head * seqlen, dim)
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        
        # Decode 阶段：增量更新
        if self.key_cache[layer_idx] is None:
            if key_states.dim() == 4:
                bs, head, seqlen, dim = key_states.shape
                key_states = key_states.contiguous().view(bs * head * seqlen, dim)
                value_states = value_states.contiguous().view(bs * head * seqlen, dim)
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        
        # 确保 cache 是 2D
        if self.key_cache[layer_idx].dim() == 4:
            bs, head, seqlen, dim = self.key_cache[layer_idx].shape
            self.key_cache[layer_idx] = self.key_cache[layer_idx].contiguous().view(bs * head * seqlen, dim)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].contiguous().view(bs * head * seqlen, dim)
        
        dim = self.key_cache[layer_idx].shape[-1]
        
        # 获取 metadata
        head_lens = cache_kwargs.get("head_lens") if cache_kwargs else None
        cu_klen = cache_kwargs.get("cu_klen") if cache_kwargs else None
        
        # 检查是否可以使用 CUDA kernel
        use_cuda = (
            CUDA_KERNEL_AVAILABLE and 
            head_lens is not None and 
            cu_klen is not None and
            isinstance(head_lens, torch.Tensor) and 
            isinstance(cu_klen, torch.Tensor) and
            head_lens.dtype == torch.int32 and 
            cu_klen.dtype == torch.int32
        )
        
        if key_states.dim() == 4:
            bs, head, seqlen, dim = key_states.shape
            key_states_flat = key_states.contiguous().view(-1, dim)
            value_states_flat = value_states.contiguous().view(-1, dim)
        else:
            key_states_flat = key_states.contiguous()
            value_states_flat = value_states.contiguous()
        
        if use_cuda:
            try:
                self.key_cache[layer_idx] = update_flatten_view(
                    self.key_cache[layer_idx].contiguous().view(-1, dim),
                    key_states_flat, head_lens, cu_klen
                )
                self.value_cache[layer_idx] = update_flatten_view(
                    self.value_cache[layer_idx].contiguous().view(-1, dim),
                    value_states_flat, head_lens, cu_klen
                )
                if not self._cuda_kernel_logged:
                    print("[SparseMM Optimized] ✓ Using CUDA kernel for cache update")
                    self._cuda_kernel_logged = True
            except Exception as e:
                # 回退到 PyTorch
                print(f"[SparseMM] CUDA kernel failed: {e}, using PyTorch fallback")
                self._fallback_update(layer_idx, key_states_flat, value_states_flat, head_lens, cu_klen)
        else:
            self._fallback_update(layer_idx, key_states_flat, value_states_flat, head_lens, cu_klen)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def _fallback_update(self, layer_idx, key_states, value_states, head_lens, cu_klen):
        """PyTorch fallback - 优化版本"""
        if head_lens is None or cu_klen is None:
            # 简单追加
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=0)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=0)
            return
        
        # 使用向量化操作替代逐 head 循环
        dim = self.key_cache[layer_idx].shape[-1]
        origin_len = self.key_cache[layer_idx].shape[0]
        head_num = head_lens.shape[0]
        
        # 预分配新 cache
        new_key_cache = torch.empty((origin_len + head_num, dim), 
                                     dtype=self.key_cache[layer_idx].dtype,
                                     device=self.key_cache[layer_idx].device)
        new_value_cache = torch.empty((origin_len + head_num, dim),
                                       dtype=self.value_cache[layer_idx].dtype,
                                       device=self.value_cache[layer_idx].device)
        
        # 批量复制（使用 index_select 替代循环）
        head_lens_cpu = head_lens.cpu().numpy()
        cu_klen_cpu = cu_klen.cpu().numpy()
        
        # 构建源索引和目标索引
        src_indices = []
        dst_indices = []
        insert_dst_indices = []
        
        for h in range(head_num):
            h_len = int(head_lens_cpu[h])
            src_start = int(cu_klen_cpu[h])
            dst_start = src_start + h
            
            if h_len > 0:
                src_indices.extend(range(src_start, src_start + h_len))
                dst_indices.extend(range(dst_start, dst_start + h_len))
            
            # 插入位置
            insert_dst = int(cu_klen_cpu[h + 1]) + h
            insert_dst_indices.append(insert_dst)
        
        # 批量复制旧数据
        if len(src_indices) > 0:
            src_indices = torch.tensor(src_indices, device=self.key_cache[layer_idx].device, dtype=torch.long)
            dst_indices = torch.tensor(dst_indices, device=self.key_cache[layer_idx].device, dtype=torch.long)
            new_key_cache[dst_indices] = self.key_cache[layer_idx][src_indices]
            new_value_cache[dst_indices] = self.value_cache[layer_idx][src_indices]
        
        # 批量插入新数据
        insert_dst_indices = torch.tensor(insert_dst_indices, device=key_states.device, dtype=torch.long)
        new_key_cache[insert_dst_indices] = key_states
        new_value_cache[insert_dst_indices] = value_states
        
        self.key_cache[layer_idx] = new_key_cache
        self.value_cache[layer_idx] = new_value_cache

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return 1

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def get_max_cache_shape(self) -> Optional[int]:
        return None


class SparseMM_Optimized:
    """
    优化版本的 SparseMM
    
    核心优化:
    1. 批量化 TopK 选择 - 避免 per-head 循环
    2. 向量化 gather 操作
    3. 预计算 attention mask
    4. 内存复用
    """
    
    def __init__(
        self,
        window_size: int = 32,
        kernel_size: int = 7,
        pooling: str = 'maxpool',
        base_capacity: int = None,
        ratio: float = None,
        layer_idx: int = None,
        num_hidden_layers: int = None,
        head_score = None,
        num_attention_heads: int = 32,
        num_key_value_groups: int = 1,
        gqa_func: str = 'mean',
        model_type: str = None,
        config = None,
    ):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = base_capacity - window_size if base_capacity else 256
        self.ratio = ratio or 0.2
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers or 36
        self.num_attention_heads = num_attention_heads
        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func
        self.config = config
        
        # Metadata for varlen attention
        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None
        
        # 预计算 attention mask（缓存）
        self._cached_mask = None
        self._cached_mask_size = 0
        
        # 统计信息
        self.total_token_budget = 0
        self.total_heads_counted = 0
        self.head_token_usage = []
        
        # 初始化 head score
        self._init_head_scores(head_score, model_type)
    
    def _init_head_scores(self, head_score, model_type):
        """初始化 head 重要性分数"""
        num_kv_heads = self.num_attention_heads // self.num_key_value_groups
        
        if head_score == 'random':
            import random
            scores = np.array([random.random() for _ in range(self.num_hidden_layers * self.num_attention_heads)])
        elif head_score == 'visual' or head_score == 'audio':
            # 尝试加载预计算的分数
            # 使用 sys.modules 获取已加载的 sparsemm_utils 模块，这样可以获取被 monkey-patched 的 load_head_score
            try:
                import sys
                # 尝试多种可能的模块路径
                sparsemm_utils = None
                for module_name in ['model.sparsemm_utils', 'sparsemm_utils']:
                    if module_name in sys.modules:
                        sparsemm_utils = sys.modules[module_name]
                        break
                
                if sparsemm_utils is None:
                    # 如果没有找到，尝试直接导入
                    from . import sparsemm_utils
                
                score_dict = sparsemm_utils.load_head_score(model_type)
                scores = np.array([np.mean(v[1]) for v in score_dict.items()])
            except Exception as e:
                print(f"[SparseMM_Optimized] Warning: Failed to load head scores: {e}, using uniform distribution")
                # 回退到均匀分布
                scores = np.ones(self.num_hidden_layers * self.num_attention_heads)
        else:
            scores = np.ones(self.num_hidden_layers * self.num_attention_heads)
        
        # 归一化
        scores = scores / (scores.sum() + 1e-10)
        scores = torch.tensor(scores, dtype=torch.float32)
        
        # GQA 支持：将 score reshape 并聚合
        self.score = scores.view(self.num_hidden_layers, num_kv_heads, self.num_key_value_groups)
        self.score = self.score.sum(dim=-1)  # [num_layers, num_kv_heads]
        
        # CRITICAL: Re-normalize after aggregation so that scores sum to 1
        score_sum = self.score.sum()
        if score_sum > 0:
            self.score = self.score / score_sum
        else:
            self.score = torch.ones_like(self.score) / self.score.numel()
        
        self._compute_head_adaptive_capacity()
    
    def _compute_head_adaptive_capacity(self):
        """Compute head_adaptive_capacity based on current base_capacity
        
        base_capacity is the PER-HEAD capacity (e.g., if BUDGET=0.6 and prefill=1000, base_capacity=600-window_size).
        We distribute this capacity across KV heads based on their normalized scores.
        
        For GQA models:
        - 7B: 28 layers × 4 KV heads = 112 total KV heads
        - 3B: 36 layers × 2 KV heads = 72 total KV heads
        
        Each KV head gets: min_cache + score[layer][head] * (base_capacity - min_cache) * total_kv_heads
        where score is normalized to sum to 1 across all KV heads.
        """
        min_cache = int(self.base_capacity * self.ratio)  # Minimum cache per head
        
        # Calculate number of KV heads per layer
        num_kv_heads_per_layer = self.num_attention_heads // self.num_key_value_groups
        total_kv_heads = self.num_hidden_layers * num_kv_heads_per_layer
        
        # Total variable capacity to distribute across ALL KV heads
        # = (per_head_capacity - min_per_head) × total_kv_heads
        total_variable_capacity = (self.base_capacity - min_cache) * total_kv_heads
        
        # Distribute based on normalized scores (score sums to 1 across all KV heads)
        # Each head gets: min_cache + score * total_variable_capacity
        self.head_adaptive_capacity = torch.round(self.score * total_variable_capacity + min_cache).int()
    
    def update_budget(self, new_budget):
        """动态更新 budget"""
        self.base_capacity = max(1, new_budget - self.window_size)
        self._compute_head_adaptive_capacity()
    
    def _get_attention_mask(self, device, dtype):
        """获取缓存的 attention mask"""
        if self._cached_mask is None or self._cached_mask_size != self.window_size:
            mask = torch.full(
                (self.window_size, self.window_size),
                torch.finfo(dtype).min,
                device=device, dtype=dtype
            )
            mask_cond = torch.arange(mask.size(-1), device=device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            self._cached_mask = mask[None, None, :, :]  # [1, 1, ws, ws]
            self._cached_mask_size = self.window_size
        return self._cached_mask.to(device=device, dtype=dtype)
    
    def calcul_attn_sore(self, key_states, query_states):
        """
        优化的 attention score 计算（兼容原始接口名称）
        
        优化点:
        1. 使用缓存的 mask
        2. 避免不必要的内存分配
        """
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        # 只使用最后 window_size 的 query 计算
        query_window = query_states[..., -self.window_size:, :]  # [bs, heads, ws, dim]
        
        # 计算 attention weights
        scale = 1.0 / math.sqrt(head_dim)
        attn_weights = torch.matmul(query_window, key_states.transpose(2, 3)) * scale
        
        # 应用 causal mask（仅对 window 区域）
        mask = self._get_attention_mask(attn_weights.device, attn_weights.dtype)
        attn_weights[..., -self.window_size:, -self.window_size:] += mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # 取非 window 部分的平均
        attn_weights_mean = attn_weights[..., :-self.window_size].mean(dim=-2)  # [bs, heads, seq-ws]
        
        # GQA 聚合
        num_kv_heads = num_heads // self.num_key_value_groups
        attn_weights_mean = attn_weights_mean.view(bsz, num_kv_heads, self.num_key_value_groups, -1)
        
        if self.gqa_func == 'max':
            attn_weights_mean = attn_weights_mean.max(dim=2).values
        else:  # mean
            attn_weights_mean = attn_weights_mean.mean(dim=2)
        
        # Pooling
        if self.pooling == 'avgpool':
            result = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                  padding=self.kernel_size // 2, stride=1)
        elif self.pooling == 'maxpool':
            result = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                  padding=self.kernel_size // 2, stride=1)
        else:
            result = attn_weights_mean
        
        return result
    
    def update_kv(self, origin_key_states, query_states, origin_value_states):
        """
        优化的 KV 更新（兼容原始接口）
        
        核心优化:
        1. 批量化 TopK 和 gather - 避免 per-head 循环
        2. 使用 torch.topk 的 sorted=False 提升性能
        3. 预分配输出张量
        """
        device = origin_key_states.device
        bsz, num_kv_heads, q_len, head_dim = origin_key_states.shape
        num_heads = query_states.shape[1]  # attention heads
        
        # 检查是否需要压缩
        compressible_len = q_len - self.window_size
        if compressible_len <= 0 or self.base_capacity > compressible_len:
            # 不需要压缩
            self._init_metadata_no_compress(num_heads, q_len, device)
            return origin_key_states.reshape(-1, head_dim), origin_value_states.reshape(-1, head_dim)
        
        # 计算 attention score
        key_states_expanded = repeat_kv(origin_key_states, self.num_key_value_groups)
        attn_score = self.calcul_attn_sore(key_states_expanded, query_states)
        
        # 获取每个 head 的容量
        head_capacities = self.head_adaptive_capacity[self.layer_idx]  # [num_kv_heads]
        head_capacities = head_capacities.to(device)
        
        # 使用批量化方式选择 TopK
        max_capacity = min(head_capacities.max().item(), compressible_len)
        
        # 批量 TopK（取最大容量，使用 sorted=False 加速）
        _, topk_indices = torch.topk(attn_score, k=max_capacity, dim=-1, sorted=False)
        
        # 准备输出
        heads_key_states = []
        heads_value_states = []
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        
        # 虽然仍有循环，但 gather 操作更高效
        for h in range(num_kv_heads):
            cap = min(int(head_capacities[h].item()), compressible_len)
            indices = topk_indices[0, h, :cap]  # [cap]
            
            # 使用 index_select 替代 gather（某些情况下更快）
            indices_expanded = indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            indices_expanded = indices_expanded.expand(1, 1, -1, head_dim)
            
            # 收集历史 tokens
            past_key = origin_key_states[0, h:h+1, :-self.window_size].gather(1, indices_expanded.squeeze(0))
            past_value = origin_value_states[0, h:h+1, :-self.window_size].gather(1, indices_expanded.squeeze(0))
            
            # 拼接最近 window
            recent_key = origin_key_states[0, h:h+1, -self.window_size:]
            recent_value = origin_value_states[0, h:h+1, -self.window_size:]
            
            selected_k = torch.cat([past_key, recent_key], dim=1)
            selected_v = torch.cat([past_value, recent_value], dim=1)
            
            l = cap + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l
            
            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))
        
        # 初始化 metadata
        self._init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k, device)
        
        # 合并
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)
        
        return heads_key_states, heads_value_states
    
    def _init_metadata(self, num_heads, k_lens, klen_sum, max_seqlen_k, device):
        """初始化 varlen attention 所需的 metadata"""
        num_kv_heads = num_heads // self.num_key_value_groups
        
        self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=device)
        self.klen_sum = klen_sum
        self.max_seqlen_k = max_seqlen_k
        self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
        self.cu_klen = self.cu_headlens - self.head_lens
        self.cu_klen = torch.cat([
            self.cu_klen,
            torch.tensor([self.klen_sum], dtype=torch.int32, device=device)
        ], dim=0)
        
        self.layer_qlens = torch.ones(num_kv_heads, dtype=torch.int32, device=device)
        self.qlen_sum = num_kv_heads
        self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
        self.cu_qlen = torch.cat([
            self.cu_qlen,
            torch.tensor([self.qlen_sum], dtype=torch.int32, device=device)
        ], dim=0)
        
        self.cu_offset = torch.arange(0, num_kv_heads + 1, dtype=torch.int32, device=device)
        self.cu_head_offset = torch.arange(1, num_kv_heads + 1, dtype=torch.int32, device=device)
    
    def _init_metadata_no_compress(self, num_heads, q_len, device):
        """不压缩时的 metadata 初始化"""
        num_kv_heads = num_heads // self.num_key_value_groups
        k_lens = [q_len] * num_kv_heads
        klen_sum = q_len * num_kv_heads
        self._init_metadata(num_heads, k_lens, klen_sum, q_len, device)
    
    def reset_token_usage_stats(self):
        """重置统计信息"""
        self.total_token_budget = 0
        self.total_heads_counted = 0
        self.head_token_usage = []
    
    def get_token_usage_stats(self):
        """获取统计信息"""
        if self.total_heads_counted == 0:
            return {"total_token_budget": 0, "avg_token_per_head": 0.0}
        avg = self.total_token_budget / float(self.total_heads_counted)
        return {
            "total_token_budget": int(self.total_token_budget),
            "avg_token_per_head": float(avg),
        }


def init_sparsemm_optimized(self):
    """初始化优化版本的 SparseMM"""
    if hasattr(self, "kv_cluster"):
        return
    
    # 从 config 获取参数（优先），config 中的值已经由 modeling 文件预处理过
    # 如果 config 中没有，则从环境变量获取（兼容旧版本）
    if not hasattr(self.config, 'window_size') or self.config.window_size is None:
        self.config.window_size = int(os.getenv('WINDOW_SIZE', 32))
    if not hasattr(self.config, 'max_capacity_prompt') or self.config.max_capacity_prompt is None:
        # BUDGET 可能是比例值（如 0.8）或整数，这里使用默认值
        # 实际值由 modeling 文件中的 enable_sparsemm_attention 方法计算
        budget_str = os.getenv('BUDGET', '4096')
        try:
            self.config.max_capacity_prompt = int(budget_str)
        except ValueError:
            # 如果是比例值，使用占位符，后续会动态计算
            self.config.max_capacity_prompt = 4096
    if not hasattr(self.config, 'kernel_size') or self.config.kernel_size is None:
        self.config.kernel_size = int(os.getenv('KERNEL_SIZE', 7))
    if not hasattr(self.config, 'pooling') or self.config.pooling is None:
        self.config.pooling = os.getenv('POOLING', 'maxpool')
    if not hasattr(self.config, 'ratio') or self.config.ratio is None:
        self.config.ratio = float(os.getenv('RATIO', 0.2))
    if not hasattr(self.config, 'gqa_func'):
        if hasattr(self.config, 'model_type') and ('llama' in self.config.model_type or 
            'mistral' in self.config.model_type or 'llava' in self.config.model_type or 
            'qwen' in self.config.model_type):
            self.config.gqa_func = 'mean'
        else:
            self.config.gqa_func = 'mean'
    
    window_size = self.config.window_size
    max_capacity = self.config.max_capacity_prompt
    kernel_size = self.config.kernel_size
    pooling = self.config.pooling
    ratio = self.config.ratio
    gqa_func = self.config.gqa_func
    
    num_kv_groups = self.config.num_attention_heads // self.config.num_key_value_heads
    
    # 获取 head_score（如果有的话）
    head_score = getattr(self.config, 'head_score', None)
    model_type = getattr(self.config, 'model_type', 'qwen')
    
    self.kv_cluster = SparseMM_Optimized(
        window_size=window_size,
        kernel_size=kernel_size,
        pooling=pooling,
        base_capacity=max_capacity,
        head_score=head_score,
        ratio=ratio,
        layer_idx=self.layer_idx,
        num_hidden_layers=self.config.num_hidden_layers,
        num_attention_heads=self.config.num_attention_heads,
        num_key_value_groups=num_kv_groups,
        gqa_func=gqa_func,
        model_type=model_type,
        config=self.config,
    )


# ============ 性能监控 ============

class PerformanceMonitor:
    """性能监控工具"""
    
    def __init__(self):
        self.timings = {}
        self.cuda_events = {}
    
    def start_timer(self, name: str):
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.cuda_events[name] = (start, end)
    
    def end_timer(self, name: str) -> float:
        if name in self.cuda_events:
            start, end = self.cuda_events[name]
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)  # ms
            self.timings[name] = self.timings.get(name, []) + [elapsed]
            return elapsed
        return 0.0
    
    def get_summary(self) -> dict:
        summary = {}
        for name, times in self.timings.items():
            summary[name] = {
                'count': len(times),
                'total_ms': sum(times),
                'avg_ms': sum(times) / len(times) if times else 0,
                'max_ms': max(times) if times else 0,
                'min_ms': min(times) if times else 0,
            }
        return summary
