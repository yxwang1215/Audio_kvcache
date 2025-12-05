# -*- coding: utf-8 -*-
import os
import re
import json
import sys
import time
import glob
import traceback
import argparse
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm

# Torch / HF
import torch
import random
from transformers import __version__ as TRANSFORMERS_VERSION
from transformers.cache_utils import DynamicCache, Cache

# Set up path for imports - must be done before importing model modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Get parent directory (sparsekv_fft_dataset)
# Don't chdir to current_dir to avoid conflicts with Python's 'code' module
# Just add parent_dir to sys.path so we can import from model directory
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)  # Add parent directory to path for model imports

# 根据环境变量决定是否使用 sparsekv 版本
USE_SPARSEKV = os.environ.get("USE_SPARSEKV", "true").lower() == "true"

if USE_SPARSEKV:
    # Import from model directory
    from model.modeling_gemma3n import Gemma3nForConditionalGeneration, Gemma3nTextAttentionSparseMM
    print("Using sparsekv version of the model")
else:
    from model.modeling_gemma3n import Gemma3nForConditionalGeneration
    print("Using standard version of the model (sparsekv disabled)")

from model.processing_gemma3n import Gemma3nProcessor
from model.configuration_gemma3n import Gemma3nConfig

# Import SparseMM utilities from model directory
try:
    from model.sparsemm_utils import (
        SparseMM,
        DynamicCacheSplitHeadFlatten,
        init_sparsemm
    )
    SPARSEM_AVAILABLE = True
except ImportError:
    SPARSEM_AVAILABLE = False
    print("[WARNING] sparsemm_utils not available, SparseMM features will be disabled")

# Import audio head utilities for SparseMM (if available)
try:
    from model.audio_head_utils import (
        load_audio_head_scores,
        process_audio_head_scores,
        get_audio_head_score_path
    )
    AUDIO_HEAD_UTILS_AVAILABLE = True
except ImportError:
    AUDIO_HEAD_UTILS_AVAILABLE = False
    print("[WARNING] audio_head_utils not available, using default head scores")

# 音频（官方建议：16kHz、float32、傅里叶重采样）
import librosa
import soundfile as sf
from scipy import signal

# ------------------------------
# 基本配置（可由环境变量覆盖）
# ------------------------------
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/mnt/public/models/gemma-3n-E2B-it")
LIBRISPEECH_DIR = os.environ.get("LIBRISPEECH_DIR", "/mnt/users/wangyuxuan-20250915/Datasets/AudioMarathon/librispeech-long")
SPLIT = os.environ.get("SPLIT", "test-clean")  # 或 "test-other"

# 可配置分片窗口（官方建议单段≤30s）
CHUNK_WINDOW_SEC = float(os.environ.get("CHUNK_WINDOW_SEC", "24"))
CHUNK_OVERLAP_SEC = float(os.environ.get("CHUNK_OVERLAP_SEC", "4"))

# token 预算：官方 6.25 token/s；另加 buffer（默认 E4B 稍大）
TOKENS_PER_SEC = float(os.environ.get("TOKENS_PER_SEC", "6.25"))

# 针对 E4B 的更保守 buffer 默认（可被环境变量覆盖）
_model_hint = os.path.basename(LOCAL_MODEL_PATH).lower()
_is_e4b = ("e4b" in _model_hint)
_default_token_buffer = 192 if _is_e4b else 128

TOKEN_BUFFER = int(os.environ.get("TOKEN_BUFFER", str(_default_token_buffer)))
MAX_TOKEN_FLOOR = int(os.environ.get("MAX_TOKEN_FLOOR", "64"))
MAX_TOKEN_CEIL = int(os.environ.get("MAX_TOKEN_CEIL", "2048"))

# 解码与清洗策略
# 默认开启“跳过特殊 token”，E4B 输出里特殊标记更多；锚点对 E4B 默认关闭（可通过环境变量启用）
DECODE_SKIP_SPECIAL = os.environ.get("DECODE_SKIP_SPECIAL_TOKENS", "1") == "1"
ANCHOR_TAG = os.environ.get("ANCHOR_TAG", "Spoken Content:")
_env_use_anchor = os.environ.get("USE_ANCHOR", None)
if _env_use_anchor is None:
    # 未显式设置时，对 E4B 关闭，对 E2B 依旧启用
    USE_ANCHOR = (not _is_e4b) and len(ANCHOR_TAG) > 0
else:
    USE_ANCHOR = (_env_use_anchor == "1") and len(ANCHOR_TAG) > 0

# librosa 读入细节（官方建议傅里叶 resample）
LIBROSA_RES_TYPE = os.environ.get("LIBROSA_RES_TYPE", "scipy")  # 推荐 "scipy"
LIBROSA_DTYPE = np.float32  # 官方 float32 [-1,1]

# 设备
DEVICE_ID = int(os.environ.get("DEVICE_ID", 0))
if torch.cuda.is_available():
    device = torch.device(f"cuda:{DEVICE_ID}")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


# 运行控制
SAMPLE_LIMIT = int(os.environ.get("SAMPLE_LIMIT", 0))
# 若 >0 则强制使用该值；=0 表示按时长动态估算（推荐）
MAX_NEW_TOKENS_ENV = int(os.environ.get("MAX_NEW_TOKENS", 0))
ENABLE_CHUNKING = os.environ.get("ENABLE_CHUNKING", "1") == "1"  # 默认开启
ENABLE_COMPILE = os.environ.get("COMPILE", "0") == "1"

# SparseMM configuration
ENV_BUDGET_RAW = os.environ.get("BUDGET", "4096")
try:
    ENV_BUDGET = float(ENV_BUDGET_RAW)
    ENV_BUDGET_IS_RATIO = (ENV_BUDGET <= 1.0)  # Fixed: include 1.0 as ratio (100%)
except ValueError:
    ENV_BUDGET = 4096.0
    ENV_BUDGET_IS_RATIO = False
    print(f"Warning: Invalid BUDGET value '{ENV_BUDGET_RAW}', using default 4096")

ENV_RATIO = float(os.environ.get("RATIO", 0.2))
ENV_WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", 64))  # Default 64
ENV_KERNEL_SIZE = int(os.environ.get("KERNEL_SIZE", 7))
ENV_POOLING = os.environ.get("POOLING", "maxpool")
ENV_HEAD_SCORE_FILE = os.environ.get("HEAD_SCORE_FILE", "aggregated")
ENV_HEAD_SCORE_BATCH = os.environ.get("HEAD_SCORE_BATCH", None)

# Audio-aware TopK selection parameters (from WHY_IMPROVEMENTS.md)
USE_AUDIO_AWARE_TOPK = os.environ.get("USE_AUDIO_AWARE_TOPK", "false").lower() == "true"
ENV_AUDIO_BOUNDARY_BOOST = float(os.environ.get("AUDIO_BOUNDARY_BOOST", "1.3"))
ENV_AUDIO_CONTINUITY_WEIGHT = float(os.environ.get("AUDIO_CONTINUITY_WEIGHT", "0.1"))
ENV_AUDIO_SMOOTH_WINDOW = int(os.environ.get("AUDIO_SMOOTH_WINDOW", "5"))

# FFT filter configuration
USE_FFT_FILTER = os.environ.get("USE_FFT_FILTER", "false").lower() == "true"
FFT_FILTER_TYPE = os.environ.get("FFT_FILTER_TYPE", "lowpass")
FFT_CUTOFF_RATIO = float(os.environ.get("FFT_CUTOFF_RATIO", "0.3"))
FFT_ALPHA = float(os.environ.get("FFT_ALPHA", "0.5"))
FFT_USE_SALIENCY = os.environ.get("FFT_USE_SALIENCY", "false").lower() == "true"
FFT_ADAPTIVE_CUTOFF = os.environ.get("FFT_ADAPTIVE_CUTOFF", "true").lower() == "true"

# Set environment variables for sparsemm_utils
if USE_FFT_FILTER:
    os.environ["USE_FFT_FILTER"] = "true"
    os.environ["FFT_FILTER_TYPE"] = FFT_FILTER_TYPE
    os.environ["FFT_CUTOFF_RATIO"] = str(FFT_CUTOFF_RATIO)
    os.environ["FFT_ALPHA"] = str(FFT_ALPHA)
    os.environ["FFT_USE_SALIENCY"] = "true" if FFT_USE_SALIENCY else "false"
    os.environ["FFT_ADAPTIVE_CUTOFF"] = "true" if FFT_ADAPTIVE_CUTOFF else "false"

if USE_AUDIO_AWARE_TOPK:
    os.environ["USE_AUDIO_AWARE_TOPK"] = "true"
    os.environ["AUDIO_BOUNDARY_BOOST"] = str(ENV_AUDIO_BOUNDARY_BOOST)
    os.environ["AUDIO_CONTINUITY_WEIGHT"] = str(ENV_AUDIO_CONTINUITY_WEIGHT)
    os.environ["AUDIO_SMOOTH_WINDOW"] = str(ENV_AUDIO_SMOOTH_WINDOW)


# 生成控制（ASR 友好：默认不做 ngram/惩罚）
NO_REPEAT_NGRAM_SIZE = int(os.environ.get("NO_REPEAT_NGRAM_SIZE", "0"))  # ← 改为 0
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", "1.0"))  # ← 改为 1.0
LENGTH_PENALTY = float(os.environ.get("LENGTH_PENALTY", "1.0"))

# 结果输出路径（将在 main() 函数中通过命令行参数设置）
output_file = None
timing_output_file = None
result_dir = None

# ------------------------------
# 文本规范化 & 指标
# ------------------------------
_DEF_PUNC_RE = re.compile(r"[.!?,;:\"()\[\]{}]")
_DEF_HYP_APOS_RE = re.compile(r"[\-\']")
_DEF_SPACE_RE = re.compile(r"\s+")

def standardize_text(text: str) -> str:
    """小写、去常见标点、去连字符和撇号、规范空白。"""
    if not text:
        return ""
    text = text.lower()
    text = _DEF_PUNC_RE.sub(" ", text)
    text = _DEF_HYP_APOS_RE.sub("", text)
    text = _DEF_SPACE_RE.sub(" ", text).strip()
    return text

def _edit_distance(ref_tokens: List[str], hyp_tokens: List[str]) -> int:
    n, m = len(ref_tokens), len(hyp_tokens)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp_tokens[i - 1] == ref_tokens[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)
    return d[m][n]

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) using jiwer with proper normalization.
    Returns WER as percentage (0-100), consistent with Qwen implementation.
    """
    try:
        import jiwer
        
        transforms = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ])
        
        ref_standardized = transforms(reference)
        hyp_standardized = transforms(hypothesis)
        
        if not ref_standardized.strip() or not hyp_standardized.strip():
            if not ref_standardized.strip() and hyp_standardized.strip():
                return 100.0
            elif not ref_standardized.strip() and not hyp_standardized.strip():
                return 0.0
            elif ref_standardized.strip() and not hyp_standardized.strip():
                return 100.0
        
        wer = jiwer.wer(ref_standardized, hyp_standardized)
        return wer * 100  # Convert ratio to percentage (0-100)
    except ImportError:
        # Fallback to original implementation if jiwer not available
        print("Warning: jiwer library not installed, using fallback WER calculation")
        ref = standardize_text(reference)
        hyp = standardize_text(hypothesis)
        if ref == hyp:
            return 0.0
        if not ref or not hyp:
            return 100.0  # Return percentage for consistency
        ref_words = ref.split()
        hyp_words = hyp.split()
        if len(ref_words) == 0:
            return 0.0 if len(hyp_words) == 0 else 100.0
        dist = _edit_distance(ref_words, hyp_words)
        return (dist / len(ref_words)) * 100  # Convert to percentage
    except Exception as e:
        print(f"Error calculating WER: {e}")
        traceback.print_exc()
        return 100.0

def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER).
    Returns CER as percentage (0-100), consistent with WER format.
    """
    try:
        import jiwer
        
        transforms = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ])
        
        ref_standardized = transforms(reference)
        hyp_standardized = transforms(hypothesis)
        
        if not ref_standardized.strip() or not hyp_standardized.strip():
            if not ref_standardized.strip() and hyp_standardized.strip():
                return 100.0
            elif not ref_standardized.strip() and not hyp_standardized.strip():
                return 0.0
            elif ref_standardized.strip() and not hyp_standardized.strip():
                return 100.0
        
        cer = jiwer.cer(ref_standardized, hyp_standardized)
        return cer * 100  # Convert ratio to percentage (0-100)
    except ImportError:
        # Fallback to original implementation if jiwer not available
        ref = standardize_text(reference).replace(" ", "")
        hyp = standardize_text(hypothesis).replace(" ", "")
        if ref == hyp:
            return 0.0
        if not ref or not hyp:
            return 100.0  # Return percentage for consistency
        ref_chars = list(ref)
        hyp_chars = list(hyp)
        dist = _edit_distance(ref_chars, hyp_chars)
        return (dist / len(ref_chars)) * 100  # Convert to percentage
    except Exception as e:
        print(f"Error calculating CER: {e}")
        traceback.print_exc()
        return 100.0

def set_f1_per_sample(reference: str, hypothesis: str) -> float:
    """基于词集合（忽略顺序）的 F1，用于补充宏 F1 指标。"""
    ref_tokens = set(standardize_text(reference).split())
    hyp_tokens = set(standardize_text(hypothesis).split())
    if len(ref_tokens) == 0 and len(hyp_tokens) == 0:
        return 1.0
    if len(hyp_tokens) == 0:
        return 0.0
    inter = len(ref_tokens & hyp_tokens)
    prec = inter / max(1, len(hyp_tokens))
    rec = inter / max(1, len(ref_tokens))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types to ensure JSON compatibility"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

class BudgetLimitedKVCache(DynamicCache):
    """
    A DynamicCache wrapper that limits kvcache size based on budget.
    Uses attention-score based selection during prefill, then keeps those positions fixed during decode.
    """
    def __init__(self, budget: float = 1.0, seed: int = None):
        """
        Args:
            budget: Budget ratio (0.0-1.0) or absolute value (>1.0)
                   If <= 1.0, it's treated as a ratio of sequence length (1.0 = 100%)
                   If > 1.0, it's treated as an absolute budget value
            seed: Random seed for position selection (None for random)
        """
        super().__init__()
        # Ensure key_cache and value_cache are initialized (for compatibility with different transformers versions)
        if not hasattr(self, 'key_cache') or self.key_cache is None:
            self.key_cache: List[torch.Tensor] = []
        if not hasattr(self, 'value_cache') or self.value_cache is None:
            self.value_cache: List[torch.Tensor] = []
        self.budget = budget
        self.budget_is_ratio = (budget <= 1.0)  # Fixed: include 1.0 as ratio (100%)
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random
        self.input_length = None  # Track input sequence length (prefill length)
        self.prefill_done = False  # Track if prefill is complete
        self.budget_applied = False  # Track if budget has been applied
        # Store selected indices per layer for decode phase
        self._selected_indices: Dict[int, List[torch.Tensor]] = {}
        
    def _get_sequence_length(self):
        """Get current sequence length from cache"""
        if len(self.key_cache) == 0 or self.key_cache[0] is None:
            return 0
        # key_cache shape: [batch_size, num_heads, seq_len, head_dim]
        return self.key_cache[0].shape[2]
    
    def _calculate_budget_size(self, seq_len: int) -> int:
        """Calculate the budget size based on current sequence length"""
        if self.budget_is_ratio:
            # Budget is a ratio: use seq_len * budget
            budget_size = int(seq_len * self.budget)
        else:
            # Budget is absolute: use min(budget, seq_len)
            budget_size = int(min(self.budget, seq_len))
        
        # Ensure at least 1 token is kept
        return max(1, budget_size)
    
    def _random_select_indices(self, seq_len: int, budget_size: int, num_heads: int) -> List[torch.Tensor]:
        """
        Randomly select indices to keep for each head independently.
        Returns a list of indices tensors, one for each head.
        """
        if seq_len <= budget_size:
            # No need to select, keep all - return same indices for all heads
            indices = torch.arange(seq_len, dtype=torch.long)
            return [indices.clone() for _ in range(num_heads)]
        
        # For each head, independently randomly select budget_size indices
        head_indices = []
        for head_idx in range(num_heads):
            # Randomly select budget_size indices from [0, seq_len)
            # Use self.rng to ensure each head gets independent random selection
            selected = self.rng.sample(range(seq_len), budget_size)
            indices = torch.tensor(selected, dtype=torch.long)
            # Sort indices to maintain order
            indices, _ = torch.sort(indices)
            head_indices.append(indices)
        
        return head_indices
    
    def _apply_budget_limit_once(self):
        """
        Apply budget limit to current cache by randomly selecting positions.
        This is called ONCE after prefill to select which positions to keep.
        During decode, new tokens are simply appended without re-selection.
        """
        if self.budget_applied:
            return
            
        seq_len = self._get_sequence_length()
        if seq_len == 0:
            return
        
        budget_size = self._calculate_budget_size(seq_len)
        if seq_len <= budget_size:
            # No need to limit
            self.budget_applied = True
            return
        
        # Apply to all layers
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is None:
                continue
                
            # Get shape: [batch_size, num_heads, seq_len, head_dim]
            key_cache = self.key_cache[layer_idx]
            value_cache = self.value_cache[layer_idx] if self.value_cache[layer_idx] is not None else None
            
            batch_size, num_heads, _, head_dim = key_cache.shape
            
            # Get random indices for each head independently
            device = key_cache.device
            head_indices_list = self._random_select_indices(seq_len, budget_size, num_heads)
            
            # Store indices for this layer (for potential decode phase use)
            self._selected_indices[layer_idx] = head_indices_list
            
            # Move indices to device
            head_indices_list = [indices.to(device) for indices in head_indices_list]
            
            # For each head, select the indices
            # We need to process each head separately since each has different indices
            selected_key_caches = []
            selected_value_caches = []
            
            for head_idx in range(num_heads):
                head_indices = head_indices_list[head_idx]
                # Select for this head: [batch_size, 1, budget_size, head_dim]
                head_key = key_cache[:, head_idx:head_idx+1, :, :].index_select(2, head_indices)
                selected_key_caches.append(head_key)
                
                if value_cache is not None:
                    head_value = value_cache[:, head_idx:head_idx+1, :, :].index_select(2, head_indices)
                    selected_value_caches.append(head_value)
            
            # Concatenate all heads back together
            # Stack along head dimension: [batch_size, num_heads, budget_size, head_dim]
            self.key_cache[layer_idx] = torch.cat(selected_key_caches, dim=1)
            
            if value_cache is not None:
                self.value_cache[layer_idx] = torch.cat(selected_value_caches, dim=1)
        
        self.budget_applied = True
    
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Update cache and apply budget limit"""
        # Ensure key_cache and value_cache exist before calling parent update
        if not hasattr(self, 'key_cache') or self.key_cache is None:
            self.key_cache: List[torch.Tensor] = []
        if not hasattr(self, 'value_cache') or self.value_cache is None:
            self.value_cache: List[torch.Tensor] = []
        
        # Detect if this is prefill (multiple tokens) or decode (single token)
        is_prefill = key_states.shape[2] > 1
        
        # Call parent update
        super().update(key_states, value_states, layer_idx, cache_kwargs)
        
        # Track input length on first prefill update (layer 0)
        if self.input_length is None and layer_idx == 0 and is_prefill:
            self.input_length = self._get_sequence_length()
        
        # Apply budget limit ONCE after prefill is complete
        # We detect prefill completion when we see a decode step (single token)
        if not self.prefill_done and not is_prefill:
            self.prefill_done = True
            # Apply budget limit now that prefill is done
            self._apply_budget_limit_once()
        
        # CRITICAL: Always return the (potentially modified) cache for this layer
        # This ensures the model uses the budget-limited cache
        if layer_idx < len(self.key_cache) and layer_idx < len(self.value_cache):
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        
        # Fallback: return the input states (shouldn't happen normally)
        return key_states, value_states

class GlobalTimingStats:
    """Simplified global timing statistics class"""
    
    def __init__(self):
        self.timing_records = []
        self.first_sample_skipped = False
        
    def add_sample(self, wall_time, prefill_time=None, total_gpu_time=None, decode_time=None, decode_token_count=None):
        """Add sample timing record, skip the first sample"""
        
        if not self.first_sample_skipped:
            self.first_sample_skipped = True
            return
            
        record = {
            "wall_time": wall_time,
            "total_gpu_time": total_gpu_time if total_gpu_time is not None else wall_time
        }
        
        # Add prefill_time if provided
        if prefill_time is not None:
            record["prefill_time"] = prefill_time
        
        # Add decode time and token count if provided
        if decode_time is not None:
            record["decode_time"] = decode_time
        if decode_token_count is not None:
            record["decode_token_count"] = decode_token_count
            # Calculate TPS (Tokens Per Second) for this sample
            # Use decode_time if available, otherwise use wall_time (total time)
            if decode_time is not None and decode_time > 0:
                record["tps"] = decode_token_count / decode_time
            elif wall_time > 0:
                record["tps"] = decode_token_count / wall_time
            else:
                record["tps"] = 0.0
        
        self.timing_records.append(record)
    
    def get_summary(self):
        """Get statistical summary"""
        if len(self.timing_records) == 0:
            return {
                "count": 0,
                "avg_wall_time": 0.0,
                "avg_prefill_time": 0.0,
                "avg_total_gpu_time": 0.0,
                "avg_decode_time": 0.0,
                "total_decode_tokens": 0,
                "avg_tps": 0.0
            }
        
        total_wall = sum(r["wall_time"] for r in self.timing_records)
        total_prefill = sum(r.get("prefill_time", 0.0) for r in self.timing_records if "prefill_time" in r)
        total_gpu = sum(r["total_gpu_time"] for r in self.timing_records)
        count = len(self.timing_records)
        
        # Calculate decode time and TPS statistics
        decode_times = [r.get("decode_time", 0.0) for r in self.timing_records if "decode_time" in r]
        decode_token_counts = [r.get("decode_token_count", 0) for r in self.timing_records if "decode_token_count" in r]
        tps_values = [r.get("tps", 0.0) for r in self.timing_records if "tps" in r]
        
        avg_decode_time = sum(decode_times) / len(decode_times) if len(decode_times) > 0 else 0.0
        total_decode_tokens = sum(decode_token_counts) if len(decode_token_counts) > 0 else 0
        avg_tps = sum(tps_values) / len(tps_values) if len(tps_values) > 0 else 0.0
        
        # Also calculate overall TPS from total tokens and total decode time
        total_decode_time = sum(decode_times) if len(decode_times) > 0 else 0.0
        overall_tps = total_decode_tokens / total_decode_time if total_decode_time > 0 else 0.0
        
        # Calculate avg_prefill_time only if we have prefill_time records
        prefill_times = [r.get("prefill_time", 0.0) for r in self.timing_records if "prefill_time" in r]
        avg_prefill_time = sum(prefill_times) / len(prefill_times) if len(prefill_times) > 0 else (total_prefill / count if count > 0 else 0.0)
        
        return {
            "count": count,
            "avg_wall_time": total_wall / count,
            "avg_prefill_time": avg_prefill_time,
            "avg_total_gpu_time": total_gpu / count,
            "avg_decode_time": avg_decode_time,
            "total_decode_tokens": total_decode_tokens,
            "avg_tps": avg_tps,
            "overall_tps": overall_tps
        }
    
    def export_to_json(self, output_file):
        """Export statistical data to JSON file"""
        result = {
            "global_summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        # Convert numpy types to ensure JSON compatibility
        result = convert_numpy_types(result)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file

def save_results_realtime(output_file, results_dict, timing_stats, dataset_size):
    """
    Save results to JSON file in real-time, updating statistics as we go
    This allows monitoring progress and viewing results before completion
    """
    try:
        # Calculate current statistics from results_dict
        samples = results_dict.get("samples", [])
        total_samples = results_dict["summary"]["total_samples"]
        
        # Calculate WER statistics from actual samples (not from accumulated totals)
        # This ensures accuracy is calculated correctly without double counting
        wer_scores = [r["wer"] for r in samples if "wer" in r]
        cer_scores = [r["cer"] for r in samples if "cer" in r]
        
        if len(wer_scores) > 0:
            # WER is stored as percentage (0-100), calculate mean directly from samples
            wer_mean = np.mean(wer_scores)
            wer_std = np.std(wer_scores)
            wer_min = np.min(wer_scores)
            wer_max = np.max(wer_scores)
            perfect_predictions = len([wer for wer in wer_scores if wer == 0.0])
            # Word accuracy = 100 - WER (both in percentage)
            word_accuracy = 100.0 - wer_mean
        else:
            wer_mean = 100.0  # WER is percentage (0-100)
            wer_std = 0.0
            wer_min = 100.0
            wer_max = 100.0
            perfect_predictions = 0
            word_accuracy = 0.0
        
        if len(cer_scores) > 0:
            cer_mean = np.mean(cer_scores)
        else:
            cer_mean = 1.0
        
        # Calculate speaker-level statistics
        from collections import defaultdict
        speaker_wer = defaultdict(list)
        for r in samples:
            if "wer" in r:
                speaker_wer[r.get("speaker_id", "unknown")].append(r["wer"])
        
        speaker_avg_wer = {spk: (sum(w)/len(w) if w else 0.0) for spk, w in speaker_wer.items()}
        
        # Build real-time results structure (maintain original structure but update summary)
        realtime_results = results_dict.copy()
        # Use mean from actual samples, not from accumulated totals (avoids double counting)
        realtime_results["summary"]["avg_wer"] = float(wer_mean)
        realtime_results["summary"]["word_accuracy"] = float(word_accuracy)
        realtime_results["summary"]["avg_cer"] = float(cer_mean)
        realtime_results["summary"]["speaker_wer"] = speaker_avg_wer
        realtime_results["summary"]["progress"] = {
            "processed": len(samples),
            "total": dataset_size,
            "percentage": len(samples) / dataset_size * 100 if dataset_size > 0 else 0
        }
        realtime_results["summary"]["wer_statistics"] = {
            "wer_mean": float(wer_mean),
            "wer_std": float(wer_std),
            "wer_min": float(wer_min),
            "wer_max": float(wer_max),
            "perfect_predictions": int(perfect_predictions)
        }
        # Update timing info in summary (preserve existing timing structure)
        timing_summary = timing_stats.get_summary()
        if "timing" in realtime_results["summary"]:
            realtime_results["summary"]["timing"].update({
                "avg_response_time": timing_summary.get("avg_wall_time", 0.0),
                "total_response_time": timing_summary.get("avg_wall_time", 0.0) * len(samples) if len(samples) > 0 else 0.0
            })
        realtime_results["summary"]["timing_stats"] = timing_summary
        realtime_results["status"] = "in_progress"
        
        # Convert numpy types
        realtime_results = convert_numpy_types(realtime_results)
        
        # Write to file (this will overwrite the file each time, ensuring latest state)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(realtime_results, f, ensure_ascii=False, indent=2)
            f.flush()  # Explicitly flush to ensure data is written immediately
            
    except Exception as e:
        print(f"Warning: Failed to save real-time results: {e}")
        traceback.print_exc()

# ------------------------------
# 数据加载（与官方读取要求对齐）
# ------------------------------

def _read_transcription_for(flac_path: str, speaker_id: str, chapter_id: str) -> str:
    base = os.path.splitext(os.path.basename(flac_path))[0]  # e.g. 84-121550-0000
    # 1) 同名 txt
    txt_file = os.path.join(os.path.dirname(flac_path), f"{base}.txt")
    if os.path.exists(txt_file):
        with open(txt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    # 2) 章节 trans.txt
    trans_file = os.path.join(os.path.dirname(flac_path), f"{speaker_id}-{chapter_id}.trans.txt")
    if os.path.exists(trans_file):
        with open(trans_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ', 1)
                if len(parts) > 1 and parts[0] == base:
                    return parts[1].strip()
    return ""

def load_librispeech_long_dataset(base_dir: str, split: str = "test-clean") -> List[Dict]:
    dataset = []
    split_dir = os.path.join(base_dir, split)
    if not os.path.exists(split_dir):
        print(f"错误: 数据集路径不存在: {split_dir}")
        return []

    print(f"扫描数据集目录: {split_dir}")
    speaker_dirs = sorted([d for d in glob.glob(os.path.join(split_dir, "*")) if os.path.isdir(d)])

    for spk_dir in speaker_dirs:
        speaker_id = os.path.basename(spk_dir)
        chapter_dirs = sorted([d for d in glob.glob(os.path.join(spk_dir, "*")) if os.path.isdir(d)])
        for ch_dir in chapter_dirs:
            chapter_id = os.path.basename(ch_dir)
            flac_files = sorted(glob.glob(os.path.join(ch_dir, "*.flac")))
            for flac_file in flac_files:
                transcription = _read_transcription_for(flac_file, speaker_id, chapter_id)
                if not transcription:
                    continue
                # 统一到 16k 单声道 float32，傅里叶法重采样
                try:
                    try:
                        y, sr = librosa.load(flac_file, sr=16000, mono=True, dtype=LIBROSA_DTYPE, res_type=LIBROSA_RES_TYPE)
                    except Exception as e:
                        # 回退到 soundfile
                        try:
                            audio, sample_rate = sf.read(flac_file)
                            if len(audio.shape) > 1 and audio.shape[1] > 1:
                                audio = np.mean(audio, axis=1)
                            if sample_rate != 16000:
                                audio = signal.resample(audio, int(len(audio) * 16000 / sample_rate))
                            y = audio.astype(np.float32)
                            sr = 16000
                        except Exception:
                            # 最后的回退：使用零数组
                            y = np.zeros(16000 * 3, dtype=np.float32)
                            sr = 16000
                    duration = y.shape[0] / float(sr)
                except Exception:
                    # 兜底：按文件大小粗略估计（16kHz * 2 bytes）
                    sz = os.path.getsize(flac_file)
                    duration = sz / (16000 * 2)
                item = {
                    "path": flac_file,
                    "audio": {"path": flac_file, "sampling_rate": 16000},
                    "transcription": transcription,
                    "duration": duration,
                    "speaker_id": speaker_id,
                    "chapter_id": chapter_id,
                    "language": "en",
                    "id": f"{speaker_id}_{chapter_id}_{os.path.splitext(os.path.basename(flac_file))[0]}"
                }
                dataset.append(item)
    print(f"加载了 {len(dataset)} 个音频样本")
    return dataset

# ------------------------------
# 模型初始化 / 生成
# ------------------------------
# 回退模板（仅在官方缺失时使用；包含 <audio_soft_token>）
GEMMA3N_CHAT_TEMPLATE = r"""{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
  {%- set first_user_prefix = (messages[0]['content'][0]['text'] if messages[0]['content'] is iterable else messages[0]['content']) + ' ' -%}
  {%- set loop_messages = messages[1:] -%}
{%- else -%}
  {%- set first_user_prefix = "" -%}
  {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
  {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
    {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
  {%- endif -%}
  {%- set role = "assistant" if message['role']=="assistant" else message['role'] -%}
  {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else "") }}
  {%- if message['content'] is string -%}
    {{ message['content'] | trim }}
  {%- elif message['content'] is iterable -%}
    {%- for item in message['content'] -%}
      {%- if item['type'] == 'audio' -%}
        {{ '<audio_soft_token>' }}
      {%- elif item['type'] == 'image' -%}
        {{ '<image_soft_token>' }}
      {%- elif item['type'] == 'text' -%}
        {{ item['text'] | trim }}
      {%- endif -%}
    {%- endfor -%}
  {%- else -%}
    {{ raise_exception("Invalid content type") }}
  {%- endif -%}
  {{ '<end_of_turn>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
  {{ '<start_of_turn>assistant\n' }}
{%- endif -%}
"""

def _preferred_dtype():
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def reset_sparsemm_state(model):
    """
    Reset SparseMM KVCache state across all layers
    This prevents state accumulation between samples that can cause accuracy degradation
    Optimized to minimize overhead when SparseMM is not actually enabled.
    """
    try:
        base_model = model.model.language_model  # Gemma3nTextModel
        if not hasattr(base_model, 'layers'):
            return
        
        # Fast path: check if any layer has kv_cluster before iterating
        has_kv_cluster = False
        for layer in base_model.layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'kv_cluster'):
                has_kv_cluster = True
                break
        
        if not has_kv_cluster:
            return  # No SparseMM state to reset, exit early
        
        # Reset state for all layers that have kv_cluster
        for layer in base_model.layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'kv_cluster'):
                kv_cluster = layer.self_attn.kv_cluster
                # Reset all internal state tensors and metadata
                if hasattr(kv_cluster, 'head_lens'):
                    kv_cluster.head_lens = None
                if hasattr(kv_cluster, 'indices'):
                    kv_cluster.indices = None
                if hasattr(kv_cluster, 'scores'):
                    kv_cluster.scores = None
                # Reset metadata used in update_kv
                if hasattr(kv_cluster, 'max_seqlen_k'):
                    kv_cluster.max_seqlen_k = 0
                if hasattr(kv_cluster, 'klen_sum'):
                    kv_cluster.klen_sum = 0
                if hasattr(kv_cluster, 'cu_klen'):
                    kv_cluster.cu_klen = 0
                if hasattr(kv_cluster, 'cu_offset'):
                    kv_cluster.cu_offset = None
                if hasattr(kv_cluster, 'cu_headlens'):
                    kv_cluster.cu_headlens = None
                # Reset token usage statistics
                if hasattr(kv_cluster, 'reset_token_usage_stats'):
                    kv_cluster.reset_token_usage_stats()
            
    except (AttributeError, KeyError):
        # Model structure doesn't match expected format, skip reset
        pass

def init_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"模型路径不存在: {LOCAL_MODEL_PATH}")
    
    processor = Gemma3nProcessor.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)

    # 优先使用官方模板；缺失时回退
    ct_tok = getattr(getattr(processor, "tokenizer", None), "chat_template", None)
    ct_proc = getattr(processor, "chat_template", None)
    if not ct_tok and not ct_proc:
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.chat_template = GEMMA3N_CHAT_TEMPLATE
        processor.chat_template = GEMMA3N_CHAT_TEMPLATE

    if hasattr(processor, "tokenizer"):
        tok = processor.tokenizer
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        try:
            tok.clean_up_tokenization_spaces = False
        except Exception:
            pass

    model = Gemma3nForConditionalGeneration.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype=_preferred_dtype(),
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()

    # Track if SparseMM is actually enabled (for performance optimization)
    SPARSEM_ENABLED = False
    
    # Enable SparseMM if available and USE_SPARSEKV is True
    if USE_SPARSEKV and SPARSEM_AVAILABLE:
        print("[SparseMM] Enabling SparseMM attention...")
        try:
            # Load head scores
            if AUDIO_HEAD_UTILS_AVAILABLE:
                # Use correct parameter order: base_dir (default), score_type, batch
                head_score_path = get_audio_head_score_path(
                    score_type=ENV_HEAD_SCORE_FILE,
                    batch=ENV_HEAD_SCORE_BATCH
                )
                raw_head_scores = load_audio_head_scores(head_score_path)
                # Process head scores with model config
                num_layers = model.config.text_config.num_hidden_layers
                num_heads = model.config.text_config.num_attention_heads
                num_key_value_heads = getattr(model.config.text_config, 'num_key_value_heads', num_heads)
                num_key_value_groups = num_heads // num_key_value_heads
                head_score_tensor = process_audio_head_scores(
                    raw_head_scores,
                    num_hidden_layers=num_layers,
                    num_attention_heads=num_heads,
                    num_key_value_groups=num_key_value_groups
                )
                # Convert to list for compatibility
                head_score_list = head_score_tensor.tolist()
                head_score_data = {
                    'scores': head_score_list,
                    'tensor': head_score_tensor
                }
            else:
                # Use default head scores (uniform)
                num_layers = model.config.text_config.num_hidden_layers
                num_heads = model.config.text_config.num_attention_heads
                num_total_heads = num_layers * num_heads
                head_score_data = {
                    'scores': [1.0 / num_total_heads] * num_total_heads,
                    'tensor': torch.ones(num_total_heads) / num_total_heads
                }
            
            # Prepare SparseMM config
            # Handle budget as ratio vs absolute value
            if ENV_BUDGET_IS_RATIO:
                # For ratio mode, we need to compute budget based on expected sequence length
                # Use a reasonable default sequence length estimate for audio (e.g., 1024 tokens)
                # The actual budget will be dynamically adjusted in update_kv based on real seq_len
                estimated_seq_len = 1024  # Reasonable estimate for audio ASR
                budget_value = max(64, int(estimated_seq_len * ENV_BUDGET))
                print(f"[SparseMM] Budget ratio mode: {ENV_BUDGET} -> estimated budget: {budget_value}")
            else:
                budget_value = int(ENV_BUDGET)
                print(f"[SparseMM] Budget absolute mode: {budget_value}")
            
            sparsemm_config = {
                'budget': budget_value,
                'budget_ratio': ENV_BUDGET if ENV_BUDGET_IS_RATIO else None,  # Store ratio for dynamic adjustment
                'ratio': ENV_RATIO,
                'window_size': ENV_WINDOW_SIZE,
                'kernel_size': ENV_KERNEL_SIZE,
                'pooling': ENV_POOLING,
            }
            
            # Enable SparseMM
            model.enable_sparsemm_attention(head_score_data, sparsemm_config)
            SPARSEM_ENABLED = True
            print("[SparseMM] SparseMM attention enabled successfully")
        except Exception as e:
            print(f"[SparseMM ERROR] Failed to enable SparseMM: {e}")
            import traceback
            traceback.print_exc()
            print("[SparseMM] Continuing without SparseMM...")
            SPARSEM_ENABLED = False

    if ENABLE_COMPILE and device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        except Exception:
            pass

    return processor, model, SPARSEM_ENABLED

def _move_to_device_safely(batch, device, float_dtype):
    AUDIO_FLOAT_KEYS = {"input_features"}
    AUDIO_BOOL_INT_KEYS = {"input_features_mask"}

    if device.type == "cuda":
        for k, v in list(batch.items()):
            if torch.is_tensor(v) and v.device.type == "cpu":
                try:
                    batch[k] = v.pin_memory()
                except Exception:
                    pass

    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            non_block = device.type == "cuda"
            if k in AUDIO_FLOAT_KEYS:
                batch[k] = v.to(device=device, dtype=torch.float32, non_blocking=non_block)
            elif k in AUDIO_BOOL_INT_KEYS:
                batch[k] = v.to(device=device, non_blocking=non_block)
            else:
                if v.is_floating_point():
                    batch[k] = v.to(device=device, dtype=float_dtype, non_blocking=non_block)
                else:
                    batch[k] = v.to(device=device, non_blocking=non_block)
    return batch

# ------------------------------
# 分片 + 重叠 + 对齐拼接
# ------------------------------

def _chunk_indices_overlap(n_samples: int, sr: int, window_sec: float, overlap_sec: float) -> List[Tuple[int, int]]:
    if window_sec <= 0:
        return [(0, n_samples)]
    win = int(window_sec * sr)
    hop = int(max(0.5, window_sec - overlap_sec) * sr)
    if win >= n_samples:
        return [(0, n_samples)]
    spans = []
    start = 0
    while start < n_samples:
        end = min(start + win, n_samples)
        spans.append((start, end))
        if end >= n_samples:
            break
        start += hop
    return spans

def estimate_tokens(duration_sec: float,
                    tokens_per_sec: float = TOKENS_PER_SEC,
                    buffer_tokens: int = TOKEN_BUFFER,
                    floor: int = MAX_TOKEN_FLOOR,
                    ceil: int = MAX_TOKEN_CEIL) -> int:
    """依据 6.25 token/s + buffer，限制在 [floor, ceil]。"""
    est = int(duration_sec * tokens_per_sec) + int(buffer_tokens)
    return max(floor, min(ceil, est))

def _seq_jaccard(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 1.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union

def _merge_with_overlap_soft(prev_text: str, new_text: str, max_overlap_words: int = 20,
                             max_rel_edit: float = 0.25, min_jaccard: float = 0.5) -> str:
    """
    软重叠拼接：
    - 在标准化词序列上寻找最长“近似相等”的后缀-前缀（允许小编辑距离）
    - 同时要求 Jaccard 相似度达到阈值，避免误判
    """
    prev_clean = standardize_text(prev_text)
    new_clean = standardize_text(new_text)
    if not prev_clean:
        return new_text.strip()

    prev_words = prev_clean.split()
    new_words = new_clean.split()

    max_k = min(max_overlap_words, len(prev_words), len(new_words))
    best_k = 0
    for k in range(max_k, 0, -1):
        tail = prev_words[-k:]
        head = new_words[:k]
        ed = _edit_distance(tail, head)
        rel = ed / max(1, k)
        jac = _seq_jaccard(tail, head)
        if rel <= max_rel_edit and jac >= min_jaccard:
            best_k = k
            break

    if best_k > 0:
        trimmed = ' '.join(new_clean.split()[best_k:])
        if not trimmed:
            return prev_text.strip()
        return (prev_text.strip() + ' ' + trimmed.strip()).strip()
    else:
        return (prev_text.strip() + ' ' + new_clean.strip()).strip()

# ------------------------------
# ASR 生成
# ------------------------------
def _get_asr_prompt() -> str:
    """Generate ASR prompt following official format.
    
    Key requirements:
    - Faithful transcription (no translation, no summarization)
    - No punctuation marks (ground truth has no punctuation)
    - Preserve all words exactly as spoken
    - Only transcription text (no explanations, no additional text)
    """
    return "Transcribe faithfully in English without translation or summarization. Do not use punctuation marks. Preserve all words exactly as spoken. Return only the transcription without explanations or additional text."

def _build_asr_messages_with_audio() -> List[Dict]:
    """
    Build ASR messages following the official Gemma-3n format.
    
    Note: Audio is passed separately via processor(audio=...) call,
    so we use None as placeholder in the message.
    """
    prompt_text = _get_asr_prompt()
    return [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": None},  # Placeholder, actual audio passed via processor
            {"type": "text", "text": prompt_text},
        ],
    }]

# 更保守：不要用会误删正文的 <sep> 规则
_SPECIAL_TAG_RE = re.compile(
    r"(?:^```.*?\n|```$|<start_of_turn>|<end_of_turn>|<eot>|<\|eot\|>|<\|endoftext\|>|<\w+_soft_token>)",
    flags=re.IGNORECASE | re.DOTALL
)

def _clean_asr_output(text: str) -> str:
    """Clean ASR output, extracting only the model's transcription response.
    
    Removes:
    - Special tokens and markers
    - Punctuation marks (to match ground truth format)
    - Extra whitespace
    """
    if not text:
        return ""
    s = text.strip()
    
    # Try to extract content after "model\n" marker (Gemma format)
    model_marker = "model\n"
    model_idx = s.lower().rfind(model_marker)
    if model_idx != -1:
        s = s[model_idx + len(model_marker):].strip()
    
    # Also try "assistant\n" marker
    assistant_marker = "assistant\n"
    assistant_idx = s.lower().rfind(assistant_marker)
    if assistant_idx != -1:
        s = s[assistant_idx + len(assistant_marker):].strip()

    if USE_ANCHOR:
        lower = s.lower()
        marker = ANCHOR_TAG.lower()
        idx = lower.find(marker)
        if idx != -1:
            s = s[idx + len(marker):].strip()

    # 去掉围栏/特殊占位与常见特殊 token
    s = _SPECIAL_TAG_RE.sub("", s).strip()

    # 剥离成对引号
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'""''":
        s = s[1:-1].strip()
    
    # Remove punctuation marks (ground truth has no punctuation)
    # Remove apostrophes
    s = re.sub(r"'", '', s)
    
    # Remove other punctuation
    s = re.sub(r'[.!?,;:"()\[\]{}]', ' ', s)
    s = re.sub(r'[\-]', '', s)  # Remove hyphens
    
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    
    return s

def _collect_terminators(tokenizer) -> List[int]:
    """组装稳健的终止 token 列表"""
    terms = set()
    try:
        if tokenizer.eos_token_id is not None:
            terms.add(int(tokenizer.eos_token_id))
    except Exception:
        pass

    for attr in ("eot_token_id", "end_of_turn_token_id"):
        try:
            tid = getattr(tokenizer, attr, None)
            if isinstance(tid, int):
                terms.add(tid)
        except Exception:
            pass

    for tok in ("<eot>", "<end_of_turn>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid != getattr(tokenizer, "unk_token_id", -1):
                terms.add(tid)
        except Exception:
            pass

    # 尽量返回非空集合
    if terms:
        return list(terms)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        return [tokenizer.eos_token_id]
    return []

@torch.inference_mode()
def asr_transcribe(processor, model, audio_np: np.ndarray, sr: int, max_new_tokens: int, sparsem_enabled: bool = False) -> Tuple[str, Dict]:
    """
    返回：
      text: 清洗后的转录
      info: 诊断信息（是否截断、是否命中 terminator 等）
    
    Following official Gemma-3n format:
    - Single user message with audio + text prompt
    - Audio passed as {"array": audio, "sampling_rate": sr}
    - Text prompt in target language
    """
    # Reset SparseMM state before each sample ONLY if SparseMM is actually enabled
    # This avoids unnecessary overhead when SparseMM is not in use
    if sparsem_enabled:
        reset_sparsemm_state(model)
    
    # === Use official Gemma-3n format ===
    # Build messages with prompt (audio passed separately)
    messages = _build_asr_messages_with_audio()
    
    # Step 1: Apply chat template to get text with <audio_soft_token> placeholder
    templated = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    # Step 2: Process with both text and audio
    # This is the standard way: processor replaces <audio_soft_token> with actual audio features
    model_inputs = processor(
        text=[templated],
        audio=[audio_np],
        sampling_rate=sr,
        return_tensors="pt",
        padding=False
    )

    float_dtype = getattr(model, "dtype", None) or _preferred_dtype()
    model_inputs = _move_to_device_safely(model_inputs, device, float_dtype)

    terminators = None
    try:
        terminators = _collect_terminators(processor.tokenizer)
    except Exception:
        terminators = None

    # Generate kwargs - optimized for ASR
    # Note: ASR typically doesn't need repetition penalty, but we keep it configurable
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
    }
    
    # Add eos_token_id if available (helps with proper stopping)
    # BUT: For ASR, we want to avoid early stopping, so we're more lenient
    if terminators:
        generate_kwargs["eos_token_id"] = terminators
    
    # CRITICAL: Don't use early_stopping for ASR - we want full transcription
    # Early stopping can cause truncation when model is uncertain
    generate_kwargs["early_stopping"] = False
    
    # Optional: Add repetition penalty if configured (default: 1.0 = no penalty)
    # For ASR, we usually don't want penalty, but slight penalty (1.01-1.05) can help avoid loops
    if REPETITION_PENALTY != 1.0:
        generate_kwargs["repetition_penalty"] = float(REPETITION_PENALTY)
    
    # Optional: Add length penalty if configured
    # For ASR, length_penalty < 1.0 encourages longer outputs (helps with truncation)
    if LENGTH_PENALTY != 1.0:
        generate_kwargs["length_penalty"] = float(LENGTH_PENALTY)
    else:
        # Default: slight length penalty to encourage longer outputs for ASR
        generate_kwargs["length_penalty"] = 0.95  # < 1.0 encourages longer sequences
    
    # Optional: Add no_repeat_ngram_size if configured
    if NO_REPEAT_NGRAM_SIZE > 0:
        generate_kwargs["no_repeat_ngram_size"] = int(NO_REPEAT_NGRAM_SIZE)
    
    # Only create custom cache if needed (for performance optimization)
    # If SparseMM is enabled, use DynamicCacheSplitHeadFlatten
    # If budget is set (and not default), use BudgetLimitedKVCache
    # Otherwise, let model use default cache (fastest)
    if sparsem_enabled:
        # Use DynamicCacheSplitHeadFlatten for SparseMM
        generate_kwargs["past_key_values"] = DynamicCacheSplitHeadFlatten()
    elif ENV_BUDGET_RAW != "4096" or ENV_BUDGET != 4096.0:
        # Budget is set, use budget-limited cache
        generate_kwargs["past_key_values"] = BudgetLimitedKVCache(budget=ENV_BUDGET, seed=42)
    # else: use default cache (don't set past_key_values for best performance)
    
    # Generate
    gen = model.generate(
        **model_inputs,
        **generate_kwargs
    )

    # 解码 - 使用 skip_special_tokens=True
    out = processor.batch_decode(gen, skip_special_tokens=True)
    raw_text = out[0] if out else ""
    
    # 清理输出（去除可能的提示词回显等）
    text = _clean_asr_output(raw_text)

    # 诊断信息 - 改进截断检测
    info = {"terminated": False, "truncated": False, "used_max_new_tokens": int(max_new_tokens), "gen_len": 0, "early_stop": False}
    try:
        if isinstance(gen, torch.Tensor):
            # 计算生成的 token 数量
            input_len = model_inputs.get("input_ids", torch.tensor([[]])).shape[-1]
            info["gen_len"] = gen.shape[-1] - input_len
            
            # Check if truncated (reached max_new_tokens)
            if info["gen_len"] >= max_new_tokens:
                info["truncated"] = True
                info["terminated"] = False  # Not naturally terminated
            
            # Check if early stopped (ended before max_new_tokens, likely hit EOS)
            elif info["gen_len"] < max_new_tokens:
                # Check if last token is EOS/terminator
                if terminators and len(gen[0]) > input_len:
                    last_token = gen[0][-1].item()
                    if last_token in terminators:
                        info["terminated"] = True
                        # If terminated very early (less than 80% of max), mark as potential early stop
                        if info["gen_len"] < max_new_tokens * 0.8:
                            info["early_stop"] = True
                    else:
                        # Didn't hit terminator but stopped early - might be truncation
                        info["truncated"] = True
                else:
                    info["terminated"] = True
    except Exception:
        pass

    return text, info

# ------------------------------
# 分片转录（可配窗口/重叠）+ 软对齐拼接
# ------------------------------

def asr_transcribe_with_overlap(processor, model, audio_np: np.ndarray, sr: int, sparsem_enabled: bool = False) -> Tuple[str, Dict]:
    n = len(audio_np)
    spans = _chunk_indices_overlap(n, sr, window_sec=CHUNK_WINDOW_SEC, overlap_sec=CHUNK_OVERLAP_SEC)
    assembled = ""
    diag = {"chunks": []}
    for ci, (a, b) in enumerate(spans):
        chunk = audio_np[a:b].astype(LIBROSA_DTYPE, copy=False)
        dur = (b - a) / float(sr)
        if MAX_NEW_TOKENS_ENV > 0:
            dyn_max = MAX_NEW_TOKENS_ENV
        else:
            dyn_max = estimate_tokens(dur)
        txt, info = asr_transcribe(processor, model, chunk, sr, dyn_max, sparsem_enabled)
        info.update({"chunk_index": ci, "start_sec": a / sr, "end_sec": b / sr, "duration_sec": dur})
        diag["chunks"].append(info)
        if txt:
            assembled = _merge_with_overlap_soft(assembled, txt, max_overlap_words=20)
    return re.sub(r"\s+", " ", assembled).strip(), diag

# ------------------------------
# 主流程
# ------------------------------

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='LibriSpeech-long ASR Evaluation with Gemma-3n')
    parser.add_argument('--output-dir', '-d', type=str, required=True,
                        help='Output directory for results (required)')
    parser.add_argument('--timing-output-file', '-t', type=str, default=None,
                        help='Output file path for timing statistics JSON (optional, defaults to {output-dir}/timing_stats.json)')
    parser.add_argument('--model-path', '-m', type=str, default=None,
                        help='Path to the model directory (optional, overrides LOCAL_MODEL_PATH)')
    args = parser.parse_args()
    
    # 设置输出路径（只使用 --output-dir）
    global output_file, timing_output_file, result_dir
    result_dir = args.output_dir
    os.makedirs(result_dir, exist_ok=True)
    
    # 设置输出文件路径
    output_file = os.path.join(result_dir, f'LibriSpeech_ASR_results_{SPLIT}.json')
    if args.timing_output_file:
        timing_output_file = args.timing_output_file
    else:
        timing_output_file = os.path.join(result_dir, 'timing_stats.json')
    
    # 覆盖模型路径（如果提供）
    if args.model_path:
        global LOCAL_MODEL_PATH
        LOCAL_MODEL_PATH = args.model_path
    
    print(f"结果将保存到: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")
    
    print("开始 LibriSpeech × Gemma-3n 本地 ASR 评测（官方对齐增强版 / E4B 优化）……")
    print(f"Transformers 版本: {TRANSFORMERS_VERSION}")
    print(f"使用设备: {device}")
    print(f"数据根目录: {LIBRISPEECH_DIR}  |  子集: {SPLIT}")
    print(f"分片窗口/重叠: {CHUNK_WINDOW_SEC}s / {CHUNK_OVERLAP_SEC}s")
    print(f"token 预算: {TOKENS_PER_SEC} tok/s, buffer={TOKEN_BUFFER}, 上下限=[{MAX_TOKEN_FLOOR},{MAX_TOKEN_CEIL}]")
    print(f"解码: skip_special_tokens={DECODE_SKIP_SPECIAL}, 锚点清洗={USE_ANCHOR}('{ANCHOR_TAG}')")
    print(f"生成约束: no_repeat_ngram_size={NO_REPEAT_NGRAM_SIZE}, repetition_penalty={REPETITION_PENALTY}, length_penalty={LENGTH_PENALTY}")
    print(f"librosa: sr=16000, mono=True, dtype=float32, res_type={LIBROSA_RES_TYPE}")
    
    # Print SparseMM and FFT configuration
    if USE_SPARSEKV and SPARSEM_AVAILABLE:
        use_fft_filter = os.environ.get("USE_FFT_FILTER", "false").lower() == "true"
        print(f"FFT-based spectral filtering: {'ENABLED' if use_fft_filter else 'DISABLED (using Conv1d)'}")
        if use_fft_filter:
            print(f"  - Filter type: {os.environ.get('FFT_FILTER_TYPE', 'lowpass')}")
            print(f"  - Cutoff ratio: {os.environ.get('FFT_CUTOFF_RATIO', '0.3')}")
            print(f"  - Alpha (mixing): {os.environ.get('FFT_ALPHA', '0.5')}")
            print(f"  - Adaptive cutoff: {os.environ.get('FFT_ADAPTIVE_CUTOFF', 'true')}")
            print(f"  - Use saliency: {os.environ.get('FFT_USE_SALIENCY', 'false')}")
        print(f"Audio-aware TopK selection: {'ENABLED' if USE_AUDIO_AWARE_TOPK else 'DISABLED'}")
        if USE_AUDIO_AWARE_TOPK:
            print(f"  - Boundary boost: {ENV_AUDIO_BOUNDARY_BOOST}")
            print(f"  - Continuity weight: {ENV_AUDIO_CONTINUITY_WEIGHT}")
            print(f"  - Smooth window: {ENV_AUDIO_SMOOTH_WINDOW}")
    print("=" * 80)

    samples = load_librispeech_long_dataset(LIBRISPEECH_DIR, SPLIT)
    if not samples:
        print("没有找到有效的样本数据，退出……")
        return

    if SAMPLE_LIMIT > 0 and len(samples) > SAMPLE_LIMIT:
        samples = samples[:SAMPLE_LIMIT]
        print(f"应用样本限制，将处理 {len(samples)} 个样本")

    from collections import defaultdict
    speaker_stats = defaultdict(int)
    for s in samples:
        speaker_stats[s.get("speaker_id", "unknown")] += 1

    results = {
        "meta": {
            "transformers": TRANSFORMERS_VERSION,
            "chunk_window_sec": CHUNK_WINDOW_SEC,
            "chunk_overlap_sec": CHUNK_OVERLAP_SEC,
            "tokens_per_sec": TOKENS_PER_SEC,
            "token_buffer": TOKEN_BUFFER,
            "token_floor": MAX_TOKEN_FLOOR,
            "token_ceil": MAX_TOKEN_CEIL,
            "decode_skip_special": DECODE_SKIP_SPECIAL,
            "use_anchor": USE_ANCHOR,
            "anchor_tag": ANCHOR_TAG,
            "librosa_res_type": LIBROSA_RES_TYPE,
            "model_path": LOCAL_MODEL_PATH,
            "split": SPLIT,
            "gen_constraints": {
                "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
                "repetition_penalty": REPETITION_PENALTY,
                "length_penalty": LENGTH_PENALTY
            }
        },
        "samples": [],
        "summary": {
            "total_samples": 0,
            "total_wer": 0.0,
            "total_cer": 0.0,
            "avg_wer": 0.0,
            "word_accuracy": 0.0,
            "avg_cer": 0.0,
            "speaker_count": len(speaker_stats),
            "timing": {"avg_response_time": 0.0, "total_response_time": 0.0},
            "macro_set_f1": 0.0,
            "truncated_chunks": 0,
            "total_chunks": 0
        }
    }

    processor, model, sparsem_enabled = init_model()

    # Initialize timing statistics
    timing_stats = GlobalTimingStats()

    is_screen_env = (not sys.stdout.isatty()) or ('TERM' in os.environ and os.environ['TERM'] == 'screen')
    if is_screen_env:
        print("检测到 screen 或非交互式环境，使用简化进度显示")

    tqdm_kwargs = {"ascii": True, "dynamic_ncols": True, "file": sys.stdout}

    set_f1_list = []

    with tqdm(total=len(samples), desc="处理 LibriSpeech 样本", position=0, leave=True, **tqdm_kwargs) as pbar:
        for i, item in enumerate(samples):
            audio_path = item['path']
            reference = item['transcription']
            speaker_id = item.get('speaker_id', 'unknown')

            pred_text = ""
            wer = 100.0  # WER is percentage (0-100)
            cer = 100.0  # CER is percentage (0-100)
            resp_time = 0.0

            try:
                # 官方对齐：统一读取到 16k 单声道 float32，傅里叶法重采样
                # 先尝试使用 librosa，如果失败则回退到 soundfile
                try:
                    y, sr = librosa.load(audio_path, sr=16000, mono=True, dtype=LIBROSA_DTYPE, res_type=LIBROSA_RES_TYPE)
                except Exception as e:
                    print(f"[Audio] librosa loading failed: {e}, falling back to soundfile")
                    # 回退到 soundfile
                    try:
                        audio, sample_rate = sf.read(audio_path)
                        # 转换为单声道
                        if len(audio.shape) > 1 and audio.shape[1] > 1:
                            audio = np.mean(audio, axis=1)
                        # 重采样到目标采样率
                        if sample_rate != 16000:
                            audio = signal.resample(audio, int(len(audio) * 16000 / sample_rate))
                        # 转换为 float32
                        y = audio.astype(np.float32)
                        sr = 16000
                        print(f"[Audio] soundfile processing successful: shape={y.shape}, sample rate={sr}Hz")
                    except Exception as e2:
                        print(f"[Audio] soundfile loading also failed: {e2}, using zero array")
                        # 最后的回退：使用零数组
                        y = np.zeros(16000 * 3, dtype=np.float32)
                        sr = 16000
                
                # === CRITICAL: Ensure proper audio format ===
                # Official docs: "Audio processing rate for the tokenizer is 16kHz"
                # https://ai.google.dev/gemma/docs/capabilities/audio
                TARGET_SR = 16000
                if sr != TARGET_SR:
                    # Use scipy.signal.resample (Fourier method) as recommended
                    num_samples = int(len(y) * TARGET_SR / sr)
                    y = signal.resample(y, num_samples)
                    sr = TARGET_SR
                
                # Ensure audio is float32 in [-1, 1] range
                if y.dtype != np.float32:
                    y = y.astype(np.float32)
                
                # Normalize to [-1, 1] if needed (for int16 source data)
                if np.abs(y).max() > 1.0:
                    y = y / 32768.0
                
                # Ensure mono (average channels if stereo)
                if len(y.shape) > 1 and y.shape[-1] > 1:
                    y = np.mean(y, axis=-1)

                start = time.time()
                if ENABLE_CHUNKING:
                    pred_text, diag = asr_transcribe_with_overlap(processor, model, y, sr, sparsem_enabled)
                    # 统计截断
                    results["summary"]["total_chunks"] += len(diag.get("chunks", []))
                    results["summary"]["truncated_chunks"] += sum(1 for c in diag.get("chunks", []) if c.get("truncated"))
                else:
                    dur = len(y) / float(sr)
                    dyn_max = MAX_NEW_TOKENS_ENV if MAX_NEW_TOKENS_ENV > 0 else estimate_tokens(dur)
                    pred_text, diag_single = asr_transcribe(processor, model, y, sr, dyn_max, sparsem_enabled)
                    results["summary"]["total_chunks"] += 1
                    results["summary"]["truncated_chunks"] += 1 if diag_single.get("truncated") else 0
                resp_time = time.time() - start

                wer = calculate_wer(reference, pred_text)
                cer = calculate_cer(reference, pred_text)
                set_f1 = set_f1_per_sample(reference, pred_text)
                set_f1_list.append(set_f1)

                results["summary"]["total_samples"] += 1
                results["summary"]["total_wer"] += wer
                results["summary"]["total_cer"] += cer
                results["summary"]["timing"]["total_response_time"] += resp_time
                
                # Add timing statistics (skip first sample)
                timing_stats.add_sample(
                    wall_time=resp_time,
                    total_gpu_time=resp_time,
                    decode_time=resp_time,  # For Gemma, we don't separate prefill/decode, so use total time
                    decode_token_count=None  # We don't track token count in current implementation
                )

            except Exception as e:
                print(f"处理错误: {e}")
                traceback.print_exc()
                pred_text = "ERROR"
                wer = 100.0  # WER is percentage (0-100)
                cer = 100.0  # CER is percentage (0-100)
                resp_time = 0.0
                set_f1_list.append(0.0)

            results["samples"].append({
                "audio_file": os.path.basename(audio_path),
                "speaker_id": speaker_id,
                "duration": item.get('duration', 0),
                "ground_truth": reference,
                "prediction": pred_text,
                "wer": wer,
                "cer": cer,
                "response_time": resp_time
            })

            # 进度显示
            cnt = i + 1
            # Calculate from actual samples to avoid double counting
            if cnt > 0:
                wer_scores = [r["wer"] for r in results["samples"] if "wer" in r]
                if len(wer_scores) > 0:
                    cur_avg_wer = np.mean(wer_scores)
                    # Word accuracy = 100 - WER (both in percentage)
                    cur_word_acc = 100.0 - cur_avg_wer
                else:
                    cur_avg_wer = 100.0
                    cur_word_acc = 0.0
            else:
                cur_avg_wer = 100.0
                cur_word_acc = 0.0
            pbar.set_postfix_str(f"平均WER:{cur_avg_wer:.3f} | Word Acc:{cur_word_acc:.2f}%")
            if is_screen_env and (cnt % 10 == 0 or cnt == len(samples)):
                print(f"  进度: {cnt}/{len(samples)} ({cnt/len(samples)*100:.1f}%), 平均WER: {cur_avg_wer:.3f}, Word Acc: {cur_word_acc:.2f}%")
            
            # Real-time save: Update JSON file after each sample (or every N samples for performance)
            save_interval = 1  # Save after every sample for real-time monitoring
            if cnt % save_interval == 0 or cnt == len(samples):
                save_results_realtime(output_file, results, timing_stats, len(samples))
                # Also save timing stats periodically
                if cnt % save_interval == 0 or cnt == len(samples):
                    timing_stats.export_to_json(timing_output_file)
            
            pbar.update()

    total = results["summary"]["total_samples"]
    if total > 0:
        # Calculate statistics from actual samples (not from accumulated totals)
        # This ensures accuracy is calculated correctly without double counting
        wer_scores = [r["wer"] for r in results["samples"] if "wer" in r]
        cer_scores = [r["cer"] for r in results["samples"] if "cer" in r]
        
        if len(wer_scores) > 0:
            # WER is stored as percentage (0-100), calculate mean directly from samples
            results["summary"]["avg_wer"] = float(np.mean(wer_scores))
            # Word accuracy = 100 - WER (both in percentage)
            results["summary"]["word_accuracy"] = 100.0 - results["summary"]["avg_wer"]
        else:
            results["summary"]["avg_wer"] = 100.0
            results["summary"]["word_accuracy"] = 0.0
        
        if len(cer_scores) > 0:
            results["summary"]["avg_cer"] = float(np.mean(cer_scores))
        else:
            results["summary"]["avg_cer"] = 1.0
        
        results["summary"]["timing"]["avg_response_time"] = (
            results["summary"]["timing"]["total_response_time"] / total
        )
        results["summary"]["macro_set_f1"] = float(np.mean(set_f1_list))

    # 按说话人统计 WER
    from collections import defaultdict
    speaker_wer = defaultdict(list)
    for r in results["samples"]:
        speaker_wer[r.get("speaker_id", "unknown")].append(r["wer"])
    speaker_avg_wer = {spk: (sum(w)/len(w) if w else 0.0) for spk, w in speaker_wer.items()}
    results["summary"]["speaker_wer"] = speaker_avg_wer

    # Final save: Update status to "completed" and save final results
    results["status"] = "completed"
    results["summary"]["progress"] = {
        "processed": total,
        "total": len(samples),
        "percentage": 100.0
    }
    
    # Convert numpy types before saving
    results = convert_numpy_types(results)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.flush()  # Explicitly flush to ensure data is written

    # Export final timing statistics to separate JSON file
    timing_stats.export_to_json(timing_output_file)

    trunc_info = f"{results['summary'].get('truncated_chunks', 0)}/{results['summary'].get('total_chunks', 0)} chunks truncated"
    print("\n=== LibriSpeech × Gemma-3n 本地 ASR 评测结果摘要（E4B 优化版） ===")
    print(f"总样本数: {total}")
    print(f"说话人数量: {len(speaker_avg_wer)}")
    print(f"平均 WER: {results['summary']['avg_wer']:.4f}")
    print(f"Word Accuracy: {results['summary']['word_accuracy']:.4f}%")
    print(f"平均 CER: {results['summary']['avg_cer']:.4f}")
    print(f"宏平均 set-F1: {results['summary']['macro_set_f1']:.4f}")
    print(f"平均响应时间: {results['summary']['timing']['avg_response_time']:.4f} 秒")
    print(f"分片窗口/重叠: {CHUNK_WINDOW_SEC}s / {CHUNK_OVERLAP_SEC}s | token 预算: {TOKENS_PER_SEC} tok/s + {TOKEN_BUFFER} | {trunc_info}")
    print(f"skip_special={DECODE_SKIP_SPECIAL}, anchor={USE_ANCHOR}('{ANCHOR_TAG}'), no_repeat_ngram={NO_REPEAT_NGRAM_SIZE}, rep_penalty={REPETITION_PENALTY}")
    print(f"结果已保存到: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")
    
    # Print timing statistics summary
    timing_summary = timing_stats.get_summary()
    if timing_summary["count"] > 0:
        print(f"\n=== Inference Time Statistics ===")
        print(f"Statistical sample count: {timing_summary['count']} (excluding first sample)")
        print(f"Average inference time: {timing_summary['avg_wall_time']:.4f} seconds")
        if timing_summary.get('avg_prefill_time', 0) > 0:
            print(f"Average Prefill time: {timing_summary['avg_prefill_time']:.4f} seconds")
        if timing_summary.get('avg_decode_time', 0) > 0:
            print(f"Average Decode time: {timing_summary['avg_decode_time']:.4f} seconds")
        print(f"Average total GPU time: {timing_summary['avg_total_gpu_time']:.4f} seconds")
        if timing_summary.get('total_decode_tokens', 0) > 0:
            print(f"\n=== TPS (Tokens Per Second) Statistics ===")
            print(f"Total decoded tokens: {timing_summary['total_decode_tokens']}")
            print(f"Average TPS (per sample): {timing_summary['avg_tps']:.2f} tokens/second")
            print(f"Overall TPS (total tokens / total decode time): {timing_summary['overall_tps']:.2f} tokens/second")

if __name__ == "__main__":
    main()
