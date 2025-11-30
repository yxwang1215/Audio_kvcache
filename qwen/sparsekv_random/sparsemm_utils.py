import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
import os
from typing import List
import random
import numpy as np
import json
import warnings
from typing import List, Optional, Tuple
from transformers.cache_utils import Cache

def load_head_score(model_type):
    if 'llava' in model_type:
        if 'mistral' not in model_type:
            head_score_path = './visual_head/head_score/llava-v1.6.json'
        else:
            head_score_path = './visual_head/head_score/llava-mistral-v1.6.json'
    elif 'qwen' in model_type:
        head_score_path = './visual_head/head_score/qwen.json'
    else:
        raise NotImplementedError
    with open(head_score_path, 'r') as f:
        head_score = json.load(f)
    return head_score

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def spectral_attention_smoothing(
    attention_scores: torch.Tensor,
    filter_type: str = "lowpass",
    cutoff_ratio: float = 0.3,
    alpha: float = 0.5,
    use_saliency: bool = False,
    adaptive_cutoff: bool = True
) -> torch.Tensor:
    """
    Spectral Attention Smoothing (SAS) using FFT-based frequency domain filtering.
    
    This function replaces Conv1d smoothing with FFT-based spectral filtering, which provides:
    1. Global receptive field (vs. local Conv kernel)
    2. Adaptive frequency cutoff based on signal characteristics
    3. Better separation of semantic trends (low-freq) vs. noise (high-freq)
    
    Args:
        attention_scores: Input attention scores [batch, heads, seq_len] or [heads, seq_len]
        filter_type: Type of filter - "lowpass", "highpass", "bandpass", or "saliency"
        cutoff_ratio: Ratio of frequencies to keep (0.0-1.0). For lowpass, keeps frequencies [0, cutoff_ratio*N]
        alpha: Mixing ratio between original and smoothed signal (0.0 = original, 1.0 = fully smoothed)
        use_saliency: If True, apply spectral residual-based saliency detection
        adaptive_cutoff: If True, dynamically determine cutoff based on energy distribution
    
    Returns:
        Smoothed attention scores with same shape as input
    """
    original_shape = attention_scores.shape
    device = attention_scores.device
    dtype = attention_scores.dtype
    
    # Handle different input dimensions
    if attention_scores.dim() == 2:
        # [heads, seq_len] -> [1, heads, seq_len]
        attention_scores = attention_scores.unsqueeze(0)
        added_batch_dim = True
    elif attention_scores.dim() == 3:
        # [batch, heads, seq_len]
        added_batch_dim = False
    else:
        raise ValueError(f"Unexpected attention_scores dimension: {attention_scores.dim()}")
    
    batch_size, num_heads, seq_len = attention_scores.shape
    
    # Store original for mixing
    original_scores = attention_scores.clone()
    
    # Process each head independently
    smoothed_scores = []
    for b in range(batch_size):
        batch_smoothed = []
        for h in range(num_heads):
            signal = attention_scores[b, h, :]  # [seq_len]
            
            # Skip FFT if sequence is too short
            if seq_len < 4:
                batch_smoothed.append(signal)
                continue
            
            # Step 1: FFT to frequency domain
            # Convert to float32 for FFT operations (BFloat16 not supported by torch.fft)
            signal_fp32 = signal.to(torch.float32)
            # Use rfft for real-valued signals (more efficient)
            fft_result = torch.fft.rfft(signal_fp32, dim=0)  # [seq_len//2 + 1] (complex)
            
            # Step 2: Compute magnitude and phase
            magnitude = torch.abs(fft_result)
            phase = torch.angle(fft_result)
            
            # Step 3: Adaptive cutoff frequency selection
            if adaptive_cutoff:
                # Calculate energy distribution
                energy = magnitude ** 2
                cumulative_energy = torch.cumsum(energy, dim=0)
                total_energy = cumulative_energy[-1]
                
                # Find cutoff based on energy distribution (use cutoff_ratio as energy threshold)
                if total_energy > 0:
                    energy_threshold = total_energy * cutoff_ratio
                    cutoff_idx = torch.searchsorted(cumulative_energy, energy_threshold).item()
                    cutoff_idx = min(cutoff_idx, len(magnitude) - 1)
                    # Ensure at least some frequencies are kept
                    cutoff_idx = max(cutoff_idx, 1)
                else:
                    cutoff_idx = max(1, int(len(magnitude) * cutoff_ratio))
            else:
                cutoff_idx = max(1, int(len(magnitude) * cutoff_ratio))
            
            # Step 4: Apply frequency domain filter
            if filter_type == "lowpass":
                # Low-pass filter: keep low frequencies (semantic trends)
                mask = torch.zeros_like(magnitude)
                mask[:cutoff_idx] = 1.0
                # Apply smooth transition to avoid ringing artifacts
                if cutoff_idx > 0 and cutoff_idx < len(mask):
                    transition_width = max(1, cutoff_idx // 10)
                    start_transition = max(0, cutoff_idx - transition_width)
                    for i in range(start_transition, cutoff_idx):
                        mask[i] = 0.5 * (1.0 + math.cos(math.pi * (cutoff_idx - i) / transition_width))
                
            elif filter_type == "saliency" and use_saliency:
                # Spectral Residual method: detect salient frequencies
                # Log spectrum
                log_magnitude = torch.log(magnitude + 1e-10)
                # Average log spectrum (trend)
                avg_log = torch.mean(log_magnitude)
                # Spectral residual (saliency)
                spectral_residual = log_magnitude - avg_log
                # Keep frequencies with high residual (salient changes)
                threshold = torch.quantile(spectral_residual, 1.0 - cutoff_ratio)
                mask = (spectral_residual > threshold).float()
                # Also keep low frequencies (base trend)
                mask[:cutoff_idx] = 1.0
                
            elif filter_type == "highpass":
                # High-pass filter: keep high frequencies (details/onsets)
                mask = torch.ones_like(magnitude)
                mask[:cutoff_idx] = 0.0
                # Smooth transition
                if cutoff_idx > 0 and cutoff_idx < len(mask):
                    transition_width = max(1, cutoff_idx // 10)
                    end_transition = min(len(mask), cutoff_idx + transition_width)
                    for i in range(cutoff_idx, end_transition):
                        mask[i] = 0.5 * (1.0 - math.cos(math.pi * (i - cutoff_idx) / transition_width))
                
            else:
                # Default: lowpass
                mask = torch.zeros_like(magnitude)
                mask[:cutoff_idx] = 1.0
            
            # Step 5: Apply mask to frequency domain
            # fft_result is complex, mask is real - multiply magnitude and keep phase
            filtered_fft = fft_result * mask
            
            # Step 6: Inverse FFT back to time domain
            smoothed_signal = torch.fft.irfft(filtered_fft, n=seq_len, dim=0)
            # Convert back to original dtype
            smoothed_signal = smoothed_signal.to(dtype)
            
            # Step 7: Mix original and smoothed signals
            final_signal = (1.0 - alpha) * signal + alpha * smoothed_signal
            
            batch_smoothed.append(final_signal)
        
        smoothed_scores.append(torch.stack(batch_smoothed, dim=0))  # [heads, seq_len]
    
    result = torch.stack(smoothed_scores, dim=0)  # [batch, heads, seq_len]
    
    # Remove batch dimension if it was added
    if added_batch_dim:
        result = result.squeeze(0)  # [heads, seq_len]
    
    return result

def merge_kv(key_states, value_states, indices, window_size, merge):
    # merge methods in LOOK-M

    bsz, num_heads, k_len, head_dim = key_states.shape

    # kv-selected
    selected_keys = key_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]
    selected_values = value_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]

    # kv-drop
    all_indices = torch.arange(k_len, device=key_states.device).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, k_len)
    all_indices_flattened = all_indices.flatten()  # [bsz * num_heads * (k_len-window_size)]
    selected_indices_flattened = indices.flatten()  # [bsz * num_heads * topk_len]
    is_selected = torch.isin(all_indices_flattened, selected_indices_flattened)
    drop_indices_flattened = all_indices_flattened[~is_selected]
    drop_len = drop_indices_flattened.shape[0] // (all_indices.shape[0] * all_indices.shape[1])
    drop_indices = drop_indices_flattened.reshape(all_indices.shape[0], all_indices.shape[1], drop_len) # [bsz * num_heads * (k_len-window_size-topk_len)]
    drop_indices = drop_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [bsz, num_heads, (k_len-window_size-topk_len), head_dim]
    drop_keys = key_states.gather(dim=2, index=drop_indices)
    drop_values = value_states.gather(dim=2, index=drop_indices)

    # kv-recent
    recent_keys = key_states[:, :, -window_size:, :]

    # prepare for merge
    k_hh_pruned = drop_keys  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    k_hh_recent = torch.cat([recent_keys, selected_keys], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    v_hh_pruned = drop_values  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    v_hh_recent = torch.cat([selected_values, value_states[:, :, -window_size:, :]], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    # similarity matrix
    similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ ((k_hh_recent / (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128))).transpose(-1, -2)) # cosin
    max_values, max_indices = similarity.max(dim=-1)

    # pivot merge
    if merge=="pivot":
        merged_indices = max_indices.unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_selected = torch.gather(input=k_hh_recent, dim=2, index=merged_indices)
        k_hh_merged = (k_hh_pruned + k_hh_selected)/2
        k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_merged, reduce='mean', include_self=True) # include_self=True seems decrease the performance
        v_hh_selected = torch.gather(input=v_hh_recent, dim=2, index=merged_indices)
        v_hh_merged = (v_hh_pruned + v_hh_selected)/2
        v_hh_recent = torch.scatter_reduce(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_merged, reduce='mean', include_self=True)

    return k_hh_recent, v_hh_recent


class DynamicCacheSplitHeadFlatten(Cache):
    '''
    adapt from https://github.com/FFY0/AdaKV.
    '''
    def __init__(self, layers: Optional[List] = None, layer_class_to_replicate: Optional[type] = None) -> None:
        try:
            # Try to call super().__init__() without arguments for older transformers versions
            super().__init__()
        except (TypeError, ValueError) as e:
            # For newer transformers versions, we need to provide layers or layer_class_to_replicate
            # Since we don't need them, we'll pass an empty list
            if "layers" in str(e) or "layer_class_to_replicate" in str(e):
                super().__init__(layers=[])
            else:
                raise
        self.key_cache: List[List[torch.Tensor]] = []
        self.value_cache: List[List[torch.Tensor]] = []
        self._seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))

    def __getitem__(self, layer_idx: int) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            # First time: append to cache (can be 2D or 4D)
            # If 4D, flatten to 2D for consistency
            if key_states.dim() == 4:
                # Flatten from [bs, head, seqlen, dim] to [total_len, dim]
                bs, head, seqlen, dim = key_states.shape
                key_states = key_states.contiguous().reshape(bs * head * seqlen, dim)
                value_states = value_states.contiguous().reshape(bs * head * seqlen, dim)
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Decode phase: update existing cache
            # Handle case where cache might be None or have wrong dimensions
            if self.key_cache[layer_idx] is None:
                # Re-initialize if None
                if key_states.dim() == 4:
                    bs, head, seqlen, dim = key_states.shape
                    key_states = key_states.contiguous().reshape(bs * head * seqlen, dim)
                    value_states = value_states.contiguous().reshape(bs * head * seqlen, dim)
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
                return
            
            # Convert to 2D if needed
            if self.key_cache[layer_idx].dim() != 2:
                if self.key_cache[layer_idx].dim() == 4:
                    # Flatten 4D to 2D
                    bs, head, seqlen, dim = self.key_cache[layer_idx].shape
                    self.key_cache[layer_idx] = self.key_cache[layer_idx].contiguous().reshape(bs * head * seqlen, dim)
                    self.value_cache[layer_idx] = self.value_cache[layer_idx].contiguous().reshape(bs * head * seqlen, dim)
                else:
                    raise ValueError(f"Unexpected cache dimension: {self.key_cache[layer_idx].dim()}")
            
            assert self.key_cache[layer_idx] is not None and self.key_cache[layer_idx].dim() == 2
            
            # Handle both 2D and 4D input for key_states
            if key_states.dim() == 2:
                # Already flattened (2D): [total_len, head_dim]
                # This should not happen in normal decode flow, but handle it gracefully
                # This might occur if prefill phase hasn't completed for this layer
                # In this case, we need to check if cache_kwargs is available for proper insertion
                head_lens = cache_kwargs.get("head_lens") if cache_kwargs is not None else None
                cu_klen = cache_kwargs.get("cu_klen") if cache_kwargs is not None else None
                
                # Check if head_lens and cu_klen are valid tensors (not None and correct dtype)
                use_cuda_kernel = False
                if head_lens is not None and cu_klen is not None:
                    # Check if they are tensors with correct dtype
                    if isinstance(head_lens, torch.Tensor) and isinstance(cu_klen, torch.Tensor):
                        # Ensure they are int32 dtype as required by CUDA kernel
                        if head_lens.dtype == torch.int32 and cu_klen.dtype == torch.int32:
                            try:
                                dim = key_states.shape[1]  # Get head_dim from second dimension
                                from tiny_api_cuda import update_flatten_view
                                new_key_cache = update_flatten_view(self.key_cache[layer_idx].contiguous().reshape(-1, dim), key_states.contiguous(), head_lens, cu_klen)
                                new_value_cache = update_flatten_view(self.value_cache[layer_idx].contiguous().reshape(-1, dim), value_states.contiguous(), head_lens, cu_klen)
                                use_cuda_kernel = True
                                self.key_cache[layer_idx] = new_key_cache
                                self.value_cache[layer_idx] = new_value_cache
                            except Exception as e:
                                # Fallback to PyTorch implementation if CUDA kernel fails
                                print(f"[SparseMM] Warning: CUDA kernel failed ({e}), falling back to PyTorch implementation")
                                use_cuda_kernel = False
                
                # Fallback to PyTorch implementation if CUDA kernel is not available or failed
                if not use_cuda_kernel:
                    # Fallback: simple concatenation (not ideal, but better than crashing)
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=0)
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=0)
            elif key_states.dim() == 4:
                # Normal decode: 4D tensor [bs, head, seqlen, dim]
                bs, head, seqlen, dim = key_states.shape
                assert bs == 1 and seqlen == 1
                head_lens = cache_kwargs.get("head_lens")
                cu_klen = cache_kwargs.get("cu_klen")

                # Check if head_lens and cu_klen are valid tensors (not None and correct dtype)
                use_cuda_kernel = False
                if head_lens is not None and cu_klen is not None:
                    # Check if they are tensors with correct dtype
                    if isinstance(head_lens, torch.Tensor) and isinstance(cu_klen, torch.Tensor):
                        # Ensure they are int32 dtype as required by CUDA kernel
                        if head_lens.dtype == torch.int32 and cu_klen.dtype == torch.int32:
                            try:
                                from tiny_api_cuda import update_flatten_view
                                new_key_cache = update_flatten_view(self.key_cache[layer_idx].contiguous().reshape(-1,dim), key_states.contiguous().reshape(-1, dim), head_lens, cu_klen)
                                new_value_cache = update_flatten_view(self.value_cache[layer_idx].contiguous().reshape(-1,dim), value_states.contiguous().reshape(-1, dim), head_lens, cu_klen)
                                use_cuda_kernel = True
                                # 在 解码阶段，把新生成的一个 token 的 KV 高效地追加到 flatten KV Cache 中
                                if not hasattr(self, '_cuda_kernel_used_logged'):
                                    print("[SparseMM] ✓ Using CUDA kernel (tiny_api_cuda) for cache update - optimal performance")
                                    self._cuda_kernel_used_logged = True
                                self.key_cache[layer_idx] = new_key_cache
                                self.value_cache[layer_idx] = new_value_cache
                            except Exception as e:
                                # Fallback to PyTorch implementation if CUDA kernel fails
                                print(f"[SparseMM] Warning: CUDA kernel failed ({e}), falling back to PyTorch implementation")
                                use_cuda_kernel = False
                
                # Fallback to PyTorch implementation if CUDA kernel is not available or failed
                if not use_cuda_kernel:
                    # Use PyTorch fallback: simple concatenation
                    # This is less efficient but more robust
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states.contiguous().reshape(-1, dim)], dim=0)
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states.contiguous().reshape(-1, dim)], dim=0)
            else:
                raise ValueError(f"Unexpected key_states dimension: {key_states.dim()}, shape: {key_states.shape}")

        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def _update_flatten_view_pytorch(self, cache, state, head_lens, cu_klen):
        """
        PyTorch fallback implementation of update_flatten_view.
        Replicates the CUDA kernel logic: copies old cache data and inserts new state at the end of each head.
        """
        origin_len = cache.shape[0]
        head_num = head_lens.shape[0]
        dim = cache.shape[1]
        
        # Create new cache with size origin_len + head_num
        new_cache = torch.empty((origin_len + head_num, dim), dtype=cache.dtype, device=cache.device)
        
        # Convert to CPU tensors for indexing if needed (for scalar access)
        head_lens_cpu = head_lens.cpu() if head_lens.is_cuda else head_lens
        cu_klen_cpu = cu_klen.cpu() if cu_klen.is_cuda else cu_klen
        
        # For each head, copy old data and insert new state
        for head_idx in range(head_num):
            head_len = int(head_lens_cpu[head_idx].item())
            cu_klen_idx = int(cu_klen_cpu[head_idx].item())
            cu_klen_next = int(cu_klen_cpu[head_idx + 1].item())
            
            src_cum_off = cu_klen_idx * dim
            dst_cum_off = src_cum_off + head_idx * dim
            insert_off = (cu_klen_next + head_idx) * dim
            
            src_start_row = src_cum_off // dim
            dst_start_row = dst_cum_off // dim
            insert_row = insert_off // dim
            
            if head_len > 0:
                actual_available = cache.shape[0] - src_start_row
                actual_head_len = min(head_len, actual_available)
                
                if actual_head_len > 0:
                    new_cache[dst_start_row:dst_start_row + actual_head_len] = cache[src_start_row:src_start_row + actual_head_len]
                    if actual_head_len < head_len:
                        remaining = head_len - actual_head_len
                        new_cache[dst_start_row + actual_head_len:dst_start_row + head_len] = 0
            
            new_cache[insert_row:insert_row + 1] = state[head_idx:head_idx + 1]
        
        return new_cache

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        # TODO: return 1 to means has content for now
        return 1

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """
        Given the sequence length of the new inputs, returns the usable length of the cache.
        For DynamicCacheSplitHeadFlatten, all cached content is usable since there's no size limit.
        """
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCacheEachHead":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', layer_idx = None, num_hidden_layers = None, 
                 pyram_mode = False, pyram_beta = 20,num_key_value_groups = 1, gqa_func='mean'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

        self.pyram_init = False
        self.pyram_mode = pyram_mode
        self.pyram_beta = pyram_beta
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, origin_key_states, query_states, origin_value_states):   
        # support gqa
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        value_states = repeat_kv(origin_value_states, self.num_key_value_groups)
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        # compute pyramidal capacity
        if self.pyram_mode and not self.pyram_init:
            base_capacity = self.max_capacity_prompt - self.window_size
            min_num = base_capacity // self.pyram_beta
            max_num = base_capacity * 2 - min_num
                
            if max_num >= q_len - self.window_size:
                max_num = q_len - self.window_size
                min_num = base_capacity * 2 - max_num
        
            steps = (max_num - min_num) // (self.num_hidden_layers - 1)

            self.max_capacity_prompt = max_num - self.layer_idx * steps + self.window_size
            self.pyram_init = True
            print(f"Pyram mode adaptive capacity, layer: {self.layer_idx}, max_capacity_prompt: {self.max_capacity_prompt}, base_capacity: {self.max_capacity_prompt - self.window_size}", flush=True)

        if q_len < self.max_capacity_prompt:
            return origin_key_states, origin_value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim = -2)
            
            attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0], -1, self.num_key_value_groups, attn_weights_mean.shape[-1])
            if self.gqa_func == 'max':
                attn_weights_mean = attn_weights_mean.max(dim=-2).values
            elif self.gqa_func == 'mean':
                attn_weights_mean = attn_weights_mean.mean(dim=-2)
            else:
                raise ValueError('gqa_func not supported')
                
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_mean, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_mean, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')

            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            
            k_past_compress = origin_key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = origin_value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = origin_key_states[:, :, -self.window_size:, :]
            v_cur = origin_value_states[:, :, -self.window_size:, :]

            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states

class AdaKVCluster():
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool',base_capacity=None,floor_alpha = None,skip = None,normalize=None, 
                 layer_idx = None, num_hidden_layers = None, pyram_mode = False, pyram_beta = 20, num_key_value_groups=1, gqa_func='mean', config=None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = base_capacity - window_size
        self.floor_ratio = floor_alpha
        self.floor_capacity = int(self.base_capacity * self.floor_ratio)
        self.adaptive_capacity = self.base_capacity - self.floor_capacity
        self.skip = skip

        self.normalize = normalize
        self.pyram_init = False
        self.pyram_mode = pyram_mode
        self.pyram_beta = pyram_beta
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        self.config = config  # Store config for audio-aware selection
        
        # Debug flag to track if we've printed audio token info
        self._debug_printed_audio_info = False

        # NOTE: layer-wise meta-data
        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None

        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func

    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)

        attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0],num_heads//self.num_key_value_groups,self.num_key_value_groups,-1)
        if self.gqa_func == 'max':
            attn_weights_mean = attn_weights_mean.max(dim=-2).values
        elif self.gqa_func == 'mean':
            attn_weights_mean = attn_weights_mean.mean(dim=-2)
        else:
            raise ValueError('gqa_func not supported')

        # Check if FFT-based spectral filtering is enabled
        use_fft_filter = os.environ.get("USE_FFT_FILTER", "false").lower() == "true"
        
        if use_fft_filter:
            # Use FFT-based spectral filtering instead of Conv1d
            filter_type = os.environ.get("FFT_FILTER_TYPE", "lowpass")  # lowpass, highpass, saliency
            cutoff_ratio = float(os.environ.get("FFT_CUTOFF_RATIO", "0.3"))
            alpha = float(os.environ.get("FFT_ALPHA", "0.5"))  # Mixing ratio
            use_saliency = os.environ.get("FFT_USE_SALIENCY", "false").lower() == "true"
            adaptive_cutoff = os.environ.get("FFT_ADAPTIVE_CUTOFF", "true").lower() == "true"
            
            # Apply FFT spectral filtering
            attn_weights_mean_pooling = spectral_attention_smoothing(
                attn_weights_mean,
                filter_type=filter_type,
                cutoff_ratio=cutoff_ratio,
                alpha=alpha,
                use_saliency=use_saliency,
                adaptive_cutoff=adaptive_cutoff
            )
        else:
            # Original Conv1d-based pooling (fallback)
            if self.pooling == 'avgpool':
                attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                         padding=self.kernel_size // 2,
                                                         stride=1)
            elif self.pooling == 'maxpool':
                attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                         padding=self.kernel_size // 2,
                                                         stride=1)
            else:
                raise ValueError('Pooling method not supported')
        
        # ===== 新增：应用音频感知增强（基于 WHY_IMPROVEMENTS.md） =====
        # 改进点：边界增强、Layer自适应时序偏置、时序连续性平滑
        if hasattr(self, 'config') and self.config is not None:
            use_audio_aware = os.environ.get("USE_AUDIO_AWARE_TOPK", "false").lower() == "true"
            
            if use_audio_aware:
                try:
                    # 动态导入音频感知选择器（避免循环依赖）
                    import sys
                    # 尝试多个可能的路径
                    possible_paths = [
                        # 从 SparseMM 目录出发的相对路径
                        os.path.join(
                            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            'src', 'Qwen_2.5_Omni', 'sparsekv', 'cache_selection'
                        ),
                        # 从项目根目录出发的绝对路径
                        '/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv/cache_selection',
                        # 当前工作目录的相对路径
                        os.path.join(os.getcwd(), 'src', 'Qwen_2.5_Omni', 'sparsekv', 'cache_selection'),
                    ]
                    
                    imported = False
                    for cache_selection_path in possible_paths:
                        if os.path.exists(cache_selection_path) and os.path.exists(
                            os.path.join(cache_selection_path, 'audio_aware_topk_selection.py')
                        ):
                            if cache_selection_path not in sys.path:
                                sys.path.insert(0, cache_selection_path)
                            try:
                                from audio_aware_topk_selection import create_audio_aware_selector
                                imported = True
                                break
                            except ImportError:
                                continue
                    
                    if not imported:
                        raise ImportError("Could not find audio_aware_topk_selection module")
                    
                    # 获取配置参数（从环境变量读取，与LibriSpeech_qwen2.5.py保持一致）
                    boundary_boost = float(os.environ.get("AUDIO_BOUNDARY_BOOST", "1.3"))
                    continuity_weight = float(os.environ.get("AUDIO_CONTINUITY_WEIGHT", "0.1"))
                    smooth_window = int(os.environ.get("AUDIO_SMOOTH_WINDOW", "5"))
                    
                    # 调试信息：检查audio token信息是否正确传递（只打印一次每层）
                    if not hasattr(self, '_debug_printed_audio_info') or not self._debug_printed_audio_info:
                        audio_token_start = getattr(self.config, 'audio_token_start', None)
                        audio_token_num = getattr(self.config, 'audio_token_num', None)
                        if self.layer_idx == 0:  # 只在第0层打印，避免过多输出
                            print(f"[AdaKVCluster Audio-Aware] Layer {self.layer_idx}: "
                                  f"audio_token_start={audio_token_start}, "
                                  f"audio_token_num={audio_token_num}, "
                                  f"boundary_boost={boundary_boost}")
                            if audio_token_start is None or audio_token_num is None:
                                print(f"[AdaKVCluster WARNING] audio_token_start or audio_token_num is None! "
                                      f"Boundary boost will NOT be applied.")
                        self._debug_printed_audio_info = True
                    
                    # 创建选择器
                    selector = create_audio_aware_selector(
                        self.config,
                        self.layer_idx if self.layer_idx is not None else 0,
                        self.num_hidden_layers if self.num_hidden_layers is not None else 36,
                        boundary_boost=boundary_boost,
                        continuity_weight=continuity_weight,
                        smooth_window=smooth_window
                    )
                    
                    # 计算序列长度（需要考虑window_size）
                    # attn_weights_mean_pooling shape: [bsz, num_kv_heads, seq_len]
                    seq_length = attn_weights_mean_pooling.shape[-1] + self.window_size
                    
                    # 应用增强（需要添加batch维度）
                    if attn_weights_mean_pooling.dim() == 2:
                        # [num_kv_heads, seq_len] -> [1, num_kv_heads, seq_len]
                        attn_weights_mean_pooling = attn_weights_mean_pooling.unsqueeze(0)
                        added_batch_dim = True
                    else:
                        added_batch_dim = False
                    
                    # 应用音频感知增强
                    attn_weights_mean_pooling = selector.enhance_audio_aware_selection(
                        attn_weights_mean_pooling,
                        seq_length,
                        self.layer_idx if self.layer_idx is not None else 0,
                        self.num_hidden_layers if self.num_hidden_layers is not None else 36
                    )
                    
                    # 移除batch维度（如果之前添加了）
                    if added_batch_dim and attn_weights_mean_pooling.dim() == 3 and attn_weights_mean_pooling.shape[0] == 1:
                        attn_weights_mean_pooling = attn_weights_mean_pooling.squeeze(0)
                        
                except Exception as e:
                    # 如果导入或使用失败，回退到原始方法
                    import warnings
                    import traceback
                    warnings.warn(f"Audio-aware TopK selection failed: {e}, using original method")
                    traceback.print_exc()
        
        return attn_weights_mean_pooling

    def update_kv(self, origin_key_states, query_states, origin_value_states):
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score= self.calcul_attn_sore(key_states,query_states)
        origin_heads_key_states = torch.split(origin_key_states, 1, dim=1)
        origin_heads_value_states = torch.split(origin_value_states, 1, dim=1)

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.layer_qlens = torch.ones(num_heads//self.num_key_value_groups, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads//self.num_key_value_groups
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.cu_offset = torch.arange(0, num_heads//self.num_key_value_groups + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads//self.num_key_value_groups +1, dtype=torch.int32, device=_device)

        if attn_score.size(-1) == 0:
            if self.head_lens is None:
                num_kv_heads = num_heads // self.num_key_value_groups
                init_metadata(num_heads, [q_len] * num_kv_heads, q_len * num_kv_heads, q_len)
            return origin_key_states.reshape(-1, head_dim), origin_value_states.reshape(-1, head_dim)
        
        if self.base_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * (num_heads//self.num_key_value_groups), q_len * (num_heads//self.num_key_value_groups), q_len)
            return origin_key_states.reshape(-1, head_dim), origin_value_states.reshape(-1, head_dim)

        sorted_attn_score,sorted_attn_score_indices = attn_score.sort(dim=-1,descending=True)
        if self.layer_idx >= self.skip:
            adaptive_attn_score = sorted_attn_score
            length = adaptive_attn_score.size(dim=-1)
            if self.normalize:
                ratio_weight = sorted_attn_score[...,:self.base_capacity].sum(dim=-1,keepdim=True)/sorted_attn_score.sum(dim=-1,keepdim=True)
                adaptive_attn_score = adaptive_attn_score*ratio_weight
            adaptive_attn_score = adaptive_attn_score.reshape(bsz,length*num_heads//self.num_key_value_groups)
            sorted_indices = torch.topk(adaptive_attn_score,k=num_heads*self.base_capacity//self.num_key_value_groups,dim=-1).indices
            sorted_indices = sorted_indices//length

            head_adaptive_capacity = torch.zeros((bsz,num_heads//self.num_key_value_groups),device=_device,dtype = sorted_indices.dtype)
            head_adaptive_capacity.scatter_add_(-1,sorted_indices,torch.ones_like(sorted_indices,dtype=head_adaptive_capacity.dtype))
            assert head_adaptive_capacity.sum().item() == num_heads*self.base_capacity//self.num_key_value_groups
            head_adaptive_capacity = torch.round(head_adaptive_capacity * (1-self.floor_ratio) + self.floor_capacity).int()
        else:
            head_adaptive_capacity = torch.ones((bsz,num_heads),device=_device,dtype = sorted_attn_score_indices.dtype) * self.base_capacity
        sorted_attn_score_indices = sorted_attn_score_indices.split(1,dim=1)

        heads_key_states = []
        heads_value_states = []
        assert bsz == 1

        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0

        for head_idx in range(num_heads//self.num_key_value_groups):
            cache_index = sorted_attn_score_indices[head_idx][...,:head_adaptive_capacity[0][head_idx]]
            l = cache_index.shape[-1] + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2,index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2,index=cache_index)
            selected_k = torch.cat([top_Kcache,origin_heads_key_states[head_idx][:, :, -self.window_size:, :]],dim=2)
            selected_v = torch.cat([top_Vcache,origin_heads_value_states[head_idx][:, :, -self.window_size:, :]],dim=2)

            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

        return heads_key_states, heads_value_states

class SparseMM():
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool', base_capacity=None, ratio=None, normalize=None, 
                 layer_idx = None, num_hidden_layers = None, head_score=None, num_attention_heads=32, num_key_value_groups=1, gqa_func='mean', model_type=None, config=None, use_random_selection=False):
        
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = base_capacity - window_size
        self.ratio = ratio

        self.normalize = normalize
        self.layer_idx = layer_idx
        self.num_attention_heads = num_attention_heads  
        self.num_hidden_layers = num_hidden_layers
        self.config = config  # Store config for audio-aware selection
        self.use_random_selection = use_random_selection  # Use random selection for tokens before window_size
        
        # Debug flag to track if we've printed audio token info
        self._debug_printed_audio_info = False

        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None

        self.num_key_value_groups = num_key_value_groups # Qwen2.5 Omni 3B = 8
        self.gqa_func = gqa_func

        if head_score == 'random':
            head_score_list = np.array([random.random() for _ in range(self.num_hidden_layers * self.num_attention_heads)])
        elif head_score == 'visual':
            head_score = load_head_score(model_type)
            head_score_list = [np.mean(l[1]) for l in head_score.items()]
        elif head_score == 'audio':
            head_score = load_head_score(model_type)
            head_score_list = [np.mean(l[1]) for l in head_score.items()]
        else:
            raise ValueError('head_score type not supported')
            
        head_score_list = torch.tensor(head_score_list / sum(head_score_list))
        # GQA support
        self.score = head_score_list.view(self.num_hidden_layers, self.num_attention_heads//self.num_key_value_groups, self.num_key_value_groups)
        self.score = self.score.sum(dim=-1)

        self._compute_head_adaptive_capacity()
        
        self.total_token_budget = 0
        self.total_heads_counted = 0
        self.head_token_usage = []

    def _compute_head_adaptive_capacity(self):
        """Compute head_adaptive_capacity based on current base_capacity"""
        min_cache = int(self.base_capacity * self.ratio)
        remain_capacity = (self.base_capacity - min_cache) * self.num_hidden_layers * self.num_attention_heads // self.num_key_value_groups
        self.head_adaptive_capacity = torch.round(self.score * remain_capacity + min_cache).int()
    
    def update_budget(self, new_budget):
        """
        Update base_capacity and recompute head_adaptive_capacity
        
        Args:
            new_budget: New total budget (base_capacity = new_budget - window_size)
        """
        self.base_capacity = new_budget - self.window_size
        if self.base_capacity <= 0:
            self.base_capacity = 1
        self._compute_head_adaptive_capacity()

    def calcul_attn_sore(self, key_states, query_states): 
        bsz, num_heads, q_len, head_dim = query_states.shape    
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim) 
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]
        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)
        attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0],num_heads//self.num_key_value_groups,self.num_key_value_groups,-1)
        
        if self.gqa_func == 'max':
            attn_weights_mean = attn_weights_mean.max(dim=-2).values
        elif self.gqa_func == 'mean':
            attn_weights_mean = attn_weights_mean.mean(dim=-2)
        else:
            raise ValueError('gqa_func not supported')
        
        # Check if FFT-based spectral filtering is enabled
        use_fft_filter = os.environ.get("USE_FFT_FILTER", "false").lower() == "true"
        
        if use_fft_filter:
            # Use FFT-based spectral filtering instead of Conv1d
            filter_type = os.environ.get("FFT_FILTER_TYPE", "lowpass")  # lowpass, highpass, saliency
            cutoff_ratio = float(os.environ.get("FFT_CUTOFF_RATIO", "0.3"))
            alpha = float(os.environ.get("FFT_ALPHA", "0.5"))  # Mixing ratio
            use_saliency = os.environ.get("FFT_USE_SALIENCY", "false").lower() == "true"
            adaptive_cutoff = os.environ.get("FFT_ADAPTIVE_CUTOFF", "true").lower() == "true"
            
            # Debug: Print FFT filter usage (only once per layer)
            if not hasattr(self, '_fft_debug_printed') or not self._fft_debug_printed:
                if self.layer_idx == 0:
                    print(f"[SparseMM FFT] Layer {self.layer_idx}: Using FFT spectral filtering with "
                          f"filter_type={filter_type}, cutoff_ratio={cutoff_ratio}, alpha={alpha}")
                self._fft_debug_printed = True
            
            # Apply FFT spectral filtering
            attn_weights_mean_pooling = spectral_attention_smoothing(
                attn_weights_mean,
                filter_type=filter_type,
                cutoff_ratio=cutoff_ratio,
                alpha=alpha,
                use_saliency=use_saliency,
                adaptive_cutoff=adaptive_cutoff
            )
        else:
            # Original Conv1d-based pooling (fallback)
            if self.pooling == 'avgpool':
                attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,padding=self.kernel_size // 2,stride=1)
            elif self.pooling == 'maxpool':
                attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,padding=self.kernel_size // 2,stride=1)
            else:
                raise ValueError('Pooling method not supported')
        
        # ===== 新增：应用音频感知增强（基于 WHY_IMPROVEMENTS.md） =====
        # 改进点：边界增强、Layer自适应时序偏置、时序连续性平滑
        if hasattr(self, 'config') and self.config is not None:
            use_audio_aware = os.environ.get("USE_AUDIO_AWARE_TOPK", "false").lower() == "true"
            
            if use_audio_aware:
                try:
                    # 动态导入音频感知选择器（避免循环依赖）
                    import sys
                    # 尝试多个可能的路径
                    possible_paths = [
                        # 从 SparseMM 目录出发的相对路径
                        os.path.join(
                            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            'src', 'Qwen_2.5_Omni', 'sparsekv', 'cache_selection'
                        ),
                        # 从项目根目录出发的绝对路径
                        '/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv/cache_selection',
                        # 当前工作目录的相对路径
                        os.path.join(os.getcwd(), 'src', 'Qwen_2.5_Omni', 'sparsekv', 'cache_selection'),
                    ]
                    
                    imported = False
                    for cache_selection_path in possible_paths:
                        if os.path.exists(cache_selection_path) and os.path.exists(
                            os.path.join(cache_selection_path, 'audio_aware_topk_selection.py')
                        ):
                            if cache_selection_path not in sys.path:
                                sys.path.insert(0, cache_selection_path)
                            try:
                                from audio_aware_topk_selection import create_audio_aware_selector
                                imported = True
                                break
                            except ImportError:
                                continue
                    
                    if not imported:
                        raise ImportError("Could not find audio_aware_topk_selection module")
                    
                    # 获取配置参数（从环境变量读取，与LibriSpeech_qwen2.5.py保持一致）
                    boundary_boost = float(os.environ.get("AUDIO_BOUNDARY_BOOST", "1.3"))
                    continuity_weight = float(os.environ.get("AUDIO_CONTINUITY_WEIGHT", "0.1"))
                    smooth_window = int(os.environ.get("AUDIO_SMOOTH_WINDOW", "5"))
                    
                    # 调试信息：检查audio token信息是否正确传递（只打印一次每层）
                    if not hasattr(self, '_debug_printed_audio_info') or not self._debug_printed_audio_info:
                        audio_token_start = getattr(self.config, 'audio_token_start', None)
                        audio_token_num = getattr(self.config, 'audio_token_num', None)
                        if self.layer_idx == 0:  # 只在第0层打印，避免过多输出
                            print(f"[SparseMM Audio-Aware] Layer {self.layer_idx}: "
                                  f"audio_token_start={audio_token_start}, "
                                  f"audio_token_num={audio_token_num}, "
                                  f"boundary_boost={boundary_boost}")
                            if audio_token_start is None or audio_token_num is None:
                                print(f"[SparseMM WARNING] audio_token_start or audio_token_num is None! "
                                      f"Boundary boost will NOT be applied.")
                        self._debug_printed_audio_info = True
                    
                    # 创建选择器
                    selector = create_audio_aware_selector(
                        self.config,
                        self.layer_idx if self.layer_idx is not None else 0,
                        self.num_hidden_layers if self.num_hidden_layers is not None else 36,
                        boundary_boost=boundary_boost,
                        continuity_weight=continuity_weight,
                        smooth_window=smooth_window
                    )
                    
                    # 计算序列长度（需要考虑window_size）
                    # attn_weights_mean_pooling shape: [bsz, num_kv_heads, seq_len]
                    seq_length = attn_weights_mean_pooling.shape[-1] + self.window_size
                    
                    # 应用增强（需要添加batch维度）
                    if attn_weights_mean_pooling.dim() == 2:
                        # [num_kv_heads, seq_len] -> [1, num_kv_heads, seq_len]
                        attn_weights_mean_pooling = attn_weights_mean_pooling.unsqueeze(0)
                        added_batch_dim = True
                    else:
                        added_batch_dim = False
                    
                    # 应用音频感知增强
                    attn_weights_mean_pooling = selector.enhance_audio_aware_selection(
                        attn_weights_mean_pooling,
                        seq_length,
                        self.layer_idx if self.layer_idx is not None else 0,
                        self.num_hidden_layers if self.num_hidden_layers is not None else 36
                    )
                    
                    # 移除batch维度（如果之前添加了）
                    if added_batch_dim and attn_weights_mean_pooling.dim() == 3 and attn_weights_mean_pooling.shape[0] == 1:
                        attn_weights_mean_pooling = attn_weights_mean_pooling.squeeze(0)
                        
                except Exception as e:
                    # 如果导入或使用失败，回退到原始方法
                    import warnings
                    import traceback
                    warnings.warn(f"Audio-aware TopK selection failed: {e}, using original method")
                    traceback.print_exc()
        
        return attn_weights_mean_pooling    

    def update_kv(self, origin_key_states, query_states, origin_value_states):
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        if getattr(self, 'head_lens', None) is None:
            self.reset_token_usage_stats()
            self._current_sample_started = True

        compressible_len = q_len - self.window_size
        if compressible_len <= 0 or self.base_capacity > compressible_len:
            # 序列过短或者整体 budget 足够大，无需压缩，直接按真实 q_len 计入
            if bsz == 1:
                num_kv_heads = num_heads // self.num_key_value_groups
                used_per_head = q_len
                total_used = num_kv_heads * used_per_head
                self.total_token_budget += total_used
                self.total_heads_counted += num_kv_heads
                self.head_token_usage.extend([used_per_head] * num_kv_heads)

            return origin_key_states.reshape(-1, head_dim), origin_value_states.reshape(-1, head_dim)
      
        # Calculate attention scores only if not using random selection
        if self.use_random_selection:
            # For random selection, we don't need to compute attention scores
            # We'll generate random indices for the compressible tokens
            attn_score = None
        else:
            attn_score = self.calcul_attn_sore(key_states, query_states)
        
        origin_heads_key_states = torch.split(origin_key_states, 1, dim=1)
        origin_heads_value_states = torch.split(origin_value_states, 1, dim=1)

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.layer_qlens = torch.ones(num_heads//self.num_key_value_groups, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads//self.num_key_value_groups
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.cu_offset = torch.arange(0, num_heads//self.num_key_value_groups + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads//self.num_key_value_groups +1, dtype=torch.int32, device=_device)

        # Generate indices based on selection method
        if self.use_random_selection:
            # Random selection: generate random indices for each head
            # We'll create indices per head in the loop below
            indices = None  # Will be generated per head in the loop
        else:
            # Original topk selection based on attention scores
            _, indices = attn_score.sort(dim=-1, descending=True)
            indices = indices.split(1, dim=1)

        heads_key_states = []
        heads_value_states = []
        assert bsz == 1
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0

        for head_idx in range(num_heads//self.num_key_value_groups):
            head_capacity = self.head_adaptive_capacity[self.layer_idx][head_idx].item()
            
            # Get cache indices based on selection method
            if self.use_random_selection:
                # Random selection: generate random indices for this head
                # Randomly select head_capacity indices from compressible_len tokens
                actual_capacity = min(head_capacity, compressible_len)
                if actual_capacity > 0:
                    # Generate random permutation and take first actual_capacity
                    random_indices = torch.randperm(compressible_len, device=_device)[:actual_capacity]
                    # Sort to maintain order (helps with cache locality)
                    random_indices, _ = torch.sort(random_indices)
                    # Reshape to match expected format: [1, 1, actual_capacity]
                    cache_index = random_indices.unsqueeze(0).unsqueeze(0)
                else:
                    # No tokens to select
                    cache_index = torch.empty((1, 1, 0), dtype=torch.long, device=_device)
            else:
                # Original topk selection: use attention score-based indices
                cache_index = indices[head_idx][..., :head_capacity]
            
            selected_old_len = cache_index.shape[-1]
            l = selected_old_len + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            # 这里是统计的关键：使用真实 l，而不是 head_capacity
            self.total_token_budget += l
            self.total_heads_counted += 1
            self.head_token_usage.append(l)
            
            # Reshape cache_index for gathering
            # cache_index should be [1, 1, selected_old_len, 1] for gather operation
            # Then expand to [1, 1, selected_old_len, head_dim] to match key/value dimensions
            if self.use_random_selection:
                # For random selection, cache_index is [1, 1, selected_old_len]
                # We need to add a dimension for head_dim and expand
                cache_index_expanded = cache_index.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            else:
                # For topk selection, cache_index is [1, 1, selected_old_len]
                # Same treatment
                cache_index_expanded = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
            
            # Gather selected tokens from the compressible part (excluding window_size)
            # Note: origin_heads_key_states[head_idx] shape is [bsz, 1, q_len, head_dim]
            # We need to select from indices [0:compressible_len], not including window_size
            compressible_keys = origin_heads_key_states[head_idx][:, :, :compressible_len, :]
            compressible_values = origin_heads_value_states[head_idx][:, :, :compressible_len, :]
            
            # Adjust cache_index to be relative to compressible_keys (which starts at 0)
            # cache_index already contains indices in range [0, compressible_len)
            top_Kcache = compressible_keys.gather(dim=2, index=cache_index_expanded)
            top_Vcache = compressible_values.gather(dim=2, index=cache_index_expanded)
            
            # Concatenate with window_size tokens (recent tokens)
            selected_k = torch.cat([top_Kcache, origin_heads_key_states[head_idx][:, :, -self.window_size:, :]], dim=2)
            selected_v = torch.cat([top_Vcache, origin_heads_value_states[head_idx][:, :, -self.window_size:, :]], dim=2)
            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

        return heads_key_states, heads_value_states

    def get_token_usage_stats(self):
        """
        返回当前 SparseMM 实际 KV cache 使用统计：
        - total_token_budget: 所有 head 合计使用的 token 数
        - avg_token_per_head: 平均每个 head 使用的 token 数
        """
        if self.total_heads_counted == 0:
            return {"total_token_budget": 0, "avg_token_per_head": 0.0}
        avg = self.total_token_budget / float(self.total_heads_counted)
        return {
            "total_token_budget": int(self.total_token_budget),
            "avg_token_per_head": float(avg),
        }

    def reset_token_usage_stats(self):
        """
        重置用于统计每样本 KV 使用情况的计数器。调用时应在样本输出后执行。
        """
        self.total_token_budget = 0
        self.total_heads_counted = 0
        self.head_token_usage = []

def init_pyramidkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = int(os.getenv('BUDGET'))
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config,'num_hidden_layers'):
            raise ValueError('num_hidden_layers should be set')
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'

    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = SnapKVCluster(
            window_size = self.config.window_size,
            max_capacity_prompt = self.config.max_capacity_prompt,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            pyram_mode=True,
            pyram_beta=20,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func=self.config.gqa_func
        )

def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = int(os.getenv('BUDGET'))
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config,'num_hidden_layers'):
            raise ValueError('num_hidden_layers should be set')
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'

    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = SnapKVCluster(
            window_size = self.config.window_size,
            max_capacity_prompt = self.config.max_capacity_prompt,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func=self.config.gqa_func
        )

def init_adakv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = int(os.getenv('BUDGET'))
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'floor_ratio'):
            self.config.floor_ratio = 0.2
        if not hasattr(self.config, 'normalize'):
            self.config.normalize = True
        if not hasattr(self.config, 'num_hidden_layers'):
            raise ValueError('num_hidden_layers should be set')
        if not hasattr(self.config, 'skip'):
            self.config.skip = 0
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'

    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = AdaKVCluster(
            window_size = self.config.window_size,
            base_capacity=self.config.max_capacity_prompt,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            floor_alpha= self.config.floor_ratio,
            skip = self.config.skip,
            layer_idx = self.layer_idx,
            normalize = self.config.normalize,
            num_hidden_layers = self.config.num_hidden_layers,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func = self.config.gqa_func,
            config = self.config  # Pass config for audio-aware selection
        )

def init_sparsemm(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size') or self.config.window_size is None:
            self.config.window_size = int(os.getenv('WINDOW_SIZE', 32))
        if not hasattr(self.config, 'max_capacity_prompt') or self.config.max_capacity_prompt is None:
            self.config.max_capacity_prompt = int(os.getenv('BUDGET', 64))
        if not hasattr(self.config, 'kernel_size') or self.config.kernel_size is None:
            self.config.kernel_size = int(os.getenv('KERNEL_SIZE', 7))
        if not hasattr(self.config, 'pooling') or self.config.pooling is None:
            self.config.pooling = os.getenv('POOLING', 'maxpool')
        if not hasattr(self.config, 'ratio') or self.config.ratio is None:
            self.config.ratio = float(os.getenv('RATIO', 0.2))
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'
        
        # Read USE_RANDOM_SELECTION from environment variable
        use_random_selection = os.getenv('USE_RANDOM_SELECTION', 'false').lower() == 'true'
        if use_random_selection:
            print(f"[SparseMM] Random selection enabled: tokens before window_size will be selected randomly")
        
        self.kv_cluster = SparseMM(
            window_size = self.config.window_size,
            base_capacity=self.config.max_capacity_prompt,
            head_score=self.config.head_score,
            ratio=self.config.ratio,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func = self.config.gqa_func,
            model_type=self.config.model_type,
            config=self.config,  # Pass config for audio-aware selection
            use_random_selection=use_random_selection  # Pass random selection flag
        )

def init_mask(self):
    if not hasattr(self, "head_list"):
        method = os.getenv('METHOD', None)

        head_score = load_head_score(self.config.model_type)
        head_list = [(l[0], np.mean(l[1])) for l in head_score.items()]
        head_list = sorted(head_list, key=lambda x: x[1], reverse=True) 

        if method == 'mask':
            ratio = float(os.getenv('MASK_RATIO'))
            num = int(ratio * len(head_list))
            print(f"mask ratio: {ratio}, num: {num}")
            head_list = [[int(ll) for ll in l[0].split("-")] for l in head_list][:num]
            self.head_list = head_list
        else:
            ratio = float(os.getenv('MASK_RATIO'))
            layer_num = 32 if 'llava' in self.config.model_type else 28
            head_num = 32 if 'llava' in self.config.model_type else 32
            num = int(ratio * layer_num * head_num)
            print(f"mask random ratio: {ratio}, num: {num}")
            head_list = [[int(ll) for ll in l[0].split("-")] for l in head_list][:num]
            self.head_list = []
            seed_list = [i  for i in range(layer_num)]
            random.shuffle(seed_list)
            while len(self.head_list) < num:
                l, h = random.choices(seed_list, k=2)
                if (l, h) in self.head_list or (h, l) in head_list:
                    continue
                else:
                    self.head_list.append((l, h))
