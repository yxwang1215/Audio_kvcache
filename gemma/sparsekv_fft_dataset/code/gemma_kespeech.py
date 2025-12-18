# -*- coding: utf-8 -*-
import os
import re
import json
import sys
import time
import glob
import traceback
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
os.chdir(current_dir)
sys.path.insert(0, parent_dir)  # Add parent directory to path for model imports

# 根据环境变量决定是否使用 sparsekv 版本
# KV Cache method selection: sparsekv, snapkv, adakv, pyramidkv, mixkv, vanilla
# sparsekv is alias for sparsemm, vanilla means no KV compression
KV_METHOD = os.environ.get("KV_METHOD", "sparsekv").lower()
# USE_SPARSEKV is derived from KV_METHOD for backward compatibility
USE_SPARSEKV = KV_METHOD != "vanilla"

# KV Cache method selection: sparsemm, snapkv, adakv, pyramidkv, mixkv

if USE_SPARSEKV:
    # Import from model directory
    from model.modeling_gemma3n import (
        Gemma3nForConditionalGeneration, 
        Gemma3nTextAttentionSparseMM,
        Gemma3nTextAttentionSnapKV,
        Gemma3nTextAttentionAdaKV,
        Gemma3nTextAttentionPyramidKV,
        Gemma3nTextAttentionMixKV
    )
    print(f"Using sparsekv version of the model (KV_METHOD={KV_METHOD})")
else:
    from model.modeling_gemma3n import Gemma3nForConditionalGeneration
    print("Using standard version of the model (sparsekv disabled)")

from model.processing_gemma3n import Gemma3nProcessor
from model.configuration_gemma3n import Gemma3nConfig
# 必须导入 feature_extractor 类，以便 AutoFeatureExtractor 能够找到它
from model.feature_extraction_gemma3n import Gemma3nAudioFeatureExtractor

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
LOCAL_MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/public/models/gemma-3n-E2B-it")
KESPEECH_DIR = os.environ.get("KESPEECH_DIR", "/mnt/users/wangyuxuan-20250915/Audio/UltraEval-Audio/raw/TwinkStart/kespeech")

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
MAX_TOKEN_FLOOR = int(os.environ.get("MAX_TOKEN_FLOOR", "16"))
# Limit MAX_TOKEN_CEIL to prevent unlimited output for single-sentence ASR
# For KeSpeech (single sentence), typical length is 10-50 chars, requiring ~20-150 tokens
# Even for long sentences, 512 tokens should be sufficient
MAX_TOKEN_CEIL = int(os.environ.get("MAX_TOKEN_CEIL", "512"))  # Reduced from 3072 to 512 for single-sentence ASR

# 解码与清洗策略
# 默认开启"跳过特殊 token"，E4B 输出里特殊标记更多；锚点对 E4B 默认关闭（可通过环境变量启用）
DECODE_SKIP_SPECIAL = os.environ.get("DECODE_SKIP_SPECIAL_TOKENS", "1") == "1"
# 根据语言设置默认锚点标签（KeSpeech 是中文数据集，默认使用中文锚点）
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "chinese").lower()
ANCHOR_TAG = os.environ.get("ANCHOR_TAG", "转录内容：" if (ASR_LANGUAGE == "chinese" or ASR_LANGUAGE == "zh") else "Spoken Content:")

# 生成控制（ASR 友好：默认不做 ngram/惩罚）
NO_REPEAT_NGRAM_SIZE = int(os.environ.get("NO_REPEAT_NGRAM_SIZE", "0"))  # ← 改为 0
# 对于中文 ASR，轻微重复惩罚（1.02-1.05）可以帮助避免重复字符，但不要太高
# 参考代码使用 1.0（无惩罚），但中文可能需要轻微惩罚来避免循环
_default_rep_penalty = 1.02 if (ASR_LANGUAGE in ["chinese", "zh"]) else 1.0  # 中文轻微惩罚
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", str(_default_rep_penalty)))
LENGTH_PENALTY = float(os.environ.get("LENGTH_PENALTY", "1.0"))
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


# ASR 语言配置（默认中文，因为 KeSpeech 是中文数据集）
# 必须在 ANCHOR_TAG 之前定义，因为 ANCHOR_TAG 会根据语言设置默认值
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "chinese").lower()

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
# Head score file path - direct path takes priority
ENV_HEAD_SCORE_PATH = os.environ.get("HEAD_SCORE_PATH", None)
# Fallback: use file type and parameters to construct path
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
# 对于中文 ASR，使用稍高的重复惩罚以防止重复字符（如"转转转"）
_default_rep_penalty = 1.1 if (ASR_LANGUAGE in ["chinese", "zh"]) else 1.0
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", str(_default_rep_penalty)))
LENGTH_PENALTY = float(os.environ.get("LENGTH_PENALTY", "1.0"))

# 结果输出
# 默认输出路径（可通过环境变量或命令行参数覆盖）
RESULTS_DIR_ENV = os.environ.get("RESULTS_DIR", None)
if RESULTS_DIR_ENV:
    result_dir = RESULTS_DIR_ENV
    os.makedirs(result_dir, exist_ok=True)
    output_file = f'{result_dir}/kespeech_results.json'
    timing_output_file = f'{result_dir}/timing_stats.json'
else:
    # 默认使用当前目录，不创建新文件夹
    result_dir = "."  # 使用当前目录
    output_file = f'{result_dir}/kespeech_results.json'
    timing_output_file = f'{result_dir}/timing_stats.json'
# Note: Output paths will be printed in main() after parsing command line arguments

# ------------------------------
# 文本规范化 & 指标
# ------------------------------
_DEF_PUNC_RE = re.compile(r"[.!?,;:\"()\[\]{}]")
_DEF_HYP_APOS_RE = re.compile(r"[\-\'\_]")  # Added underscore (_) for completeness
_DEF_SPACE_RE = re.compile(r"\s+")

# 中文标点符号正则
_CHINESE_PUNC_RE = re.compile(r'[，。！？、；：""''（）【】《》…—～·]')

def standardize_text(text: str, language: str = "chinese") -> str:
    """
    文本规范化，用于公平比较和 WER/CER 计算。
    对中文：移除标点、空格，不转换大小写。
    对英文：转小写、移除标点、规范空白。
    """
    if not text:
        return ""
    
    lang_lower = language.lower()
    if lang_lower in ["chinese", "zh"]:
        # 中文：移除中英文标点和空格
        text = _CHINESE_PUNC_RE.sub("", text)
        text = _DEF_PUNC_RE.sub("", text)
        text = _DEF_HYP_APOS_RE.sub("", text)
        text = _DEF_SPACE_RE.sub("", text)  # 移除所有空格
        return text.strip()
    else:
        # 英文：转小写、移除标点、规范空白
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

def calculate_wer(reference: str, hypothesis: str, language: str = "chinese") -> float:
    """Calculate Word Error Rate (WER) with language-specific handling.
    
    For Chinese: WER is calculated at character level (same as CER), since Chinese
    doesn't have word boundaries like English.
    
    For English: WER is calculated at word level.
    
    Returns WER as percentage (0-100).
    """
    lang_lower = language.lower()
    
    try:
        # Standardize text with language-specific handling
        ref_standardized = standardize_text(reference, language)
        hyp_standardized = standardize_text(hypothesis, language)
        
        if not ref_standardized or not hyp_standardized:
            if not ref_standardized and hyp_standardized:
                return 100.0
            elif not ref_standardized and not hyp_standardized:
                return 0.0
            elif ref_standardized and not hyp_standardized:
                return 100.0
        
        if ref_standardized == hyp_standardized:
            return 0.0
        
        if lang_lower in ["chinese", "zh"]:
            # Chinese: Calculate at character level (same as CER)
            ref_chars = list(ref_standardized)
            hyp_chars = list(hyp_standardized)
            if len(ref_chars) == 0:
                return 0.0 if len(hyp_chars) == 0 else 100.0
            dist = _edit_distance(ref_chars, hyp_chars)
            return (dist / len(ref_chars)) * 100
        else:
            # English: Calculate at word level
            ref_words = ref_standardized.split()
            hyp_words = hyp_standardized.split()
            if len(ref_words) == 0:
                return 0.0 if len(hyp_words) == 0 else 100.0
            dist = _edit_distance(ref_words, hyp_words)
            return (dist / len(ref_words)) * 100
    except Exception as e:
        print(f"Error calculating WER: {e}")
        traceback.print_exc()
        return 100.0

def calculate_cer(reference: str, hypothesis: str, language: str = "chinese") -> float:
    """Calculate Character Error Rate (CER) with language-specific handling.
    
    CER is always calculated at character level, but text normalization
    is language-specific.
    
    Returns CER as percentage (0-100), capped at 100%.
    """
    try:
        # Standardize text with language-specific handling
        ref_standardized = standardize_text(reference, language)
        hyp_standardized = standardize_text(hypothesis, language)
        
        if not ref_standardized or not hyp_standardized:
            if not ref_standardized and hyp_standardized:
                return 100.0
            elif not ref_standardized and not hyp_standardized:
                return 0.0
            elif ref_standardized and not hyp_standardized:
                return 100.0
        
        if ref_standardized == hyp_standardized:
            return 0.0
        
        # For CER, always use character-level comparison
        ref_chars = list(ref_standardized)
        hyp_chars = list(hyp_standardized)
        
        if len(ref_chars) == 0:
            return 0.0 if len(hyp_chars) == 0 else 100.0
        
        dist = _edit_distance(ref_chars, hyp_chars)
        cer_percent = (dist / len(ref_chars)) * 100
        
        # CER can exceed 100% when there are many insertions
        # CER = (S + D + I) / N, where N is the number of characters in reference
        # If insertions are very high, CER can be > 100%, but we cap it at 100%
        if cer_percent > 100.0:
            # Calculate detailed metrics for debugging
            try:
                import jiwer
                # For character-level, we need to use space-separated characters
                ref_chars_str = ' '.join(ref_chars)
                hyp_chars_str = ' '.join(hyp_chars)
                if hasattr(jiwer, 'compute_measures'):
                    measures = jiwer.compute_measures(ref_chars_str, hyp_chars_str)
                    substitutions = measures.get('substitutions', 0)
                    deletions = measures.get('deletions', 0)
                    insertions = measures.get('insertions', 0)
                    hits = measures.get('hits', 0)
                    print(f"[CER Analysis] Raw CER={cer_percent:.2f}% (capped at 100%), Ref chars={len(ref_chars)}, Hyp chars={len(hyp_chars)}")
                    print(f"  Substitutions={substitutions}, Deletions={deletions}, Insertions={insertions}, Hits={hits}")
                    if insertions > len(ref_chars):
                        print(f"  [Issue] Insertions ({insertions}) > Reference chars ({len(ref_chars)}) - CER capped at 100%")
                else:
                    # Fallback: just print basic info
                    print(f"[CER Analysis] Raw CER={cer_percent:.2f}% (capped at 100%), Ref chars={len(ref_chars)}, Hyp chars={len(hyp_chars)}")
                    if len(hyp_chars) > len(ref_chars):
                        print(f"  [Issue] Hypothesis has more chars ({len(hyp_chars)}) than reference ({len(ref_chars)}) - CER capped at 100%")
            except (ImportError, AttributeError):
                # Fallback if jiwer not available or compute_measures doesn't exist
                print(f"[CER Analysis] Raw CER={cer_percent:.2f}% (capped at 100%), Ref chars={len(ref_chars)}, Hyp chars={len(hyp_chars)}")
                if len(hyp_chars) > len(ref_chars):
                    print(f"  [Issue] Hypothesis has more chars ({len(hyp_chars)}) than reference ({len(ref_chars)}) - CER capped at 100%")
            # Cap CER at 100%
            cer_percent = 100.0
        
        return cer_percent
    except Exception as e:
        print(f"Error calculating CER: {e}")
        traceback.print_exc()
        return 100.0

def set_f1_per_sample(reference: str, hypothesis: str, language: str = "chinese") -> float:
    """基于集合（忽略顺序）的 F1，用于补充宏 F1 指标。
    
    对于中文：使用字符集合
    对于英文：使用词集合
    """
    ref_std = standardize_text(reference, language)
    hyp_std = standardize_text(hypothesis, language)
    
    lang_lower = language.lower()
    if lang_lower in ["chinese", "zh"]:
        # 中文：按字符计算
        ref_tokens = set(ref_std)
        hyp_tokens = set(hyp_std)
    else:
        # 英文：按词计算
        ref_tokens = set(ref_std.split())
        hyp_tokens = set(hyp_std.split())
    
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
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            f.flush()  # Explicitly flush to ensure data is written immediately
            os.fsync(f.fileno())  # Force write to disk for file system level sync
        
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
        
        # Calculate CER and WER statistics from actual samples (not from accumulated totals)
        # This ensures accuracy is calculated correctly without double counting
        # For Chinese ASR, CER is the primary metric
        wer_scores = [r["wer"] for r in samples if "wer" in r]
        cer_scores = [r["cer"] for r in samples if "cer" in r]
        
        # Calculate CER (primary metric for Chinese ASR)
        if len(cer_scores) > 0:
            # Ensure all CER scores are capped at 100%
            cer_scores = [min(cer, 100.0) for cer in cer_scores]
            # CER is stored as percentage (0-100), calculate mean directly from samples
            cer_mean = np.mean(cer_scores)
            cer_std = np.std(cer_scores)
            cer_min = np.min(cer_scores)
            cer_max = min(np.max(cer_scores), 100.0)  # Limit cer_max to 100%
            perfect_predictions_cer = len([cer for cer in cer_scores if cer == 0.0])
            # Character accuracy = 100 - CER (both in percentage)
            char_accuracy = max(100.0 - cer_mean, 0.0)  # Ensure accuracy is not negative
        else:
            cer_mean = 100.0  # CER is percentage (0-100)
            cer_std = 0.0
            cer_min = 100.0
            cer_max = 100.0
            perfect_predictions_cer = 0
            char_accuracy = 0.0
        
        # Calculate WER (secondary metric, still useful for reference)
        if len(wer_scores) > 0:
            # WER is stored as percentage (0-100), calculate mean directly from samples
            wer_mean = np.mean(wer_scores)
            wer_std = np.std(wer_scores)
            wer_min = np.min(wer_scores)
            wer_max = np.max(wer_scores)
            perfect_predictions_wer = len([wer for wer in wer_scores if wer == 0.0])
            # Word accuracy = 100 - WER (both in percentage)
            word_accuracy = 100.0 - wer_mean
        else:
            wer_mean = 100.0  # WER is percentage (0-100)
            wer_std = 0.0
            wer_min = 100.0
            wer_max = 100.0
            perfect_predictions_wer = 0
            word_accuracy = 0.0
        
        # Calculate speaker-level statistics
        from collections import defaultdict
        speaker_cer = defaultdict(list)
        speaker_wer = defaultdict(list)
        for r in samples:
            speaker_id = r.get("speaker_id", "unknown")
            if "cer" in r:
                speaker_cer[speaker_id].append(r["cer"])
            if "wer" in r:
                speaker_wer[speaker_id].append(r["wer"])
        
        speaker_avg_cer = {spk: (sum(c)/len(c) if c else 0.0) for spk, c in speaker_cer.items()}
        speaker_avg_wer = {spk: (sum(w)/len(w) if w else 0.0) for spk, w in speaker_wer.items()}
        
        # Build real-time results structure (maintain original structure but update summary)
        realtime_results = results_dict.copy()
        # Use mean from actual samples, not from accumulated totals (avoids double counting)
        # CER is primary metric for Chinese ASR
        realtime_results["summary"]["avg_cer"] = float(cer_mean)
        realtime_results["summary"]["char_accuracy"] = float(char_accuracy)
        realtime_results["summary"]["avg_wer"] = float(wer_mean)
        realtime_results["summary"]["word_accuracy"] = float(word_accuracy)
        realtime_results["summary"]["speaker_cer"] = speaker_avg_cer
        realtime_results["summary"]["speaker_wer"] = speaker_avg_wer
        realtime_results["summary"]["progress"] = {
            "processed": len(samples),
            "total": dataset_size,
            "percentage": len(samples) / dataset_size * 100 if dataset_size > 0 else 0
        }
        realtime_results["summary"]["cer_statistics"] = {
            "cer_mean": float(cer_mean),
            "cer_std": float(cer_std),
            "cer_min": float(cer_min),
            "cer_max": float(cer_max),
            "perfect_predictions": int(perfect_predictions_cer)
        }
        realtime_results["summary"]["wer_statistics"] = {
            "wer_mean": float(wer_mean),
            "wer_std": float(wer_std),
            "wer_min": float(wer_min),
            "wer_max": float(wer_max),
            "perfect_predictions": int(perfect_predictions_wer)
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
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Write to file (this will overwrite the file each time, ensuring latest state)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(realtime_results, f, ensure_ascii=False, indent=2)
            f.flush()  # Explicitly flush to ensure data is written immediately
            os.fsync(f.fileno())  # Force write to disk for file system level sync
            
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

def load_kespeech_dataset(base_dir, split="test"):
    """Load KeSpeech dataset from local path only, no network download"""
    dataset = []
    split_dir = os.path.join(base_dir, split)
    
    print(f"[Dataset Loading] Base directory: {base_dir}")
    print(f"[Dataset Loading] Split directory: {split_dir}")
    
    # Check if split directory exists
    if not os.path.exists(split_dir):
        print(f"Error: Dataset path does not exist: {split_dir}")
        return []
    
    # ========== Priority 0: Load from local metadata.json ==========
    # This is the fastest and most reliable method
    metadata_json_path = os.path.join(base_dir, "metadata.json")
    if os.path.exists(metadata_json_path):
        print(f"Found local metadata.json: {metadata_json_path}")
        try:
            with open(metadata_json_path, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
            print(f"Loaded {len(metadata_list)} samples from metadata.json")
            
            # Build dataset from metadata
            for item in metadata_list:
                idx = int(item.get("id", 0))
                audio_id = item.get("audio_id", f"kespeech_{idx}")
                transcription = item.get("Text", item.get("text", ""))
                speaker_id = item.get("speaker_id", "unknown")
                chapter_id = item.get("chapter_id", "")
                duration = item.get("duration", 0.0)
                
                # Find audio file path
                local_audio_path = os.path.join(split_dir, f"{idx}.wav")
                if not os.path.exists(local_audio_path):
                    # Try path from metadata
                    path_from_meta = item.get("path", "")
                    if path_from_meta:
                        if os.path.isabs(path_from_meta):
                            local_audio_path = path_from_meta
                        else:
                            local_audio_path = os.path.join(split_dir, path_from_meta)
                
                if not os.path.exists(local_audio_path):
                    print(f"Warning: Audio file not found for index {idx}: {local_audio_path}, skipping")
                    continue
                
                # Get duration if not in metadata
                if duration == 0.0:
                    try:
                        audio_info = sf.info(local_audio_path)
                        duration = audio_info.duration
                    except:
                        duration = 0.0
                
                dataset.append({
                    "id": audio_id,
                    "speaker_id": speaker_id,
                    "chapter_id": chapter_id,
                    "path": local_audio_path,
                    "transcription": transcription,
                    "duration": duration,
                    "audio": {
                        "path": local_audio_path,
                        "array": None,
                        "sampling_rate": None
                    }
                })
            
            print(f"Successfully loaded {len(dataset)} KeSpeech samples from metadata.json")
            return dataset
        except Exception as e:
            print(f"Failed to load from metadata.json: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== Priority 1: Load from saved dataset ==========
    # Try to load from local saved dataset first (if exists)
    hf_dataset = None
    try:
        from datasets import load_from_disk
        saved_dataset_path = os.path.join(base_dir, "saved_dataset")
        if os.path.exists(saved_dataset_path):
            print(f"Attempting to load saved dataset from {saved_dataset_path}...")
            saved_dataset = load_from_disk(saved_dataset_path)
            if split in saved_dataset:
                hf_dataset = saved_dataset[split]
                print(f"Loaded {len(hf_dataset)} samples from saved dataset")
    except Exception as e:
        print(f"Could not load from saved dataset: {e}")
        hf_dataset = None
    
    # ========== Priority 2: Load from HuggingFace cache ==========
    # If saved dataset not available, try to load from HuggingFace cache (without downloading)
    if hf_dataset is None:
        try:
            from datasets import load_dataset
            print(f"Attempting to load KeSpeech dataset from HuggingFace cache (no download)...")
            # Use download_mode to avoid downloading if not in cache
            # This will raise an error if dataset is not in cache, which we'll catch
            hf_dataset = load_dataset("TwinkStart/kespeech", split=split, download_mode="reuse_cache_if_exists")
            print(f"Loaded {len(hf_dataset)} samples from HuggingFace cache")
        except Exception as e:
            print(f"Failed to load from HuggingFace cache: {e}")
            print(f"Note: Transcription will be empty. This may affect evaluation accuracy.")
            print(f"To get transcriptions, you can:")
            print(f"  1. Download dataset metadata from HuggingFace: datasets.load_dataset('TwinkStart/kespeech', split='{split}')")
            print(f"  2. Or provide .txt files with same names as .wav files in {split_dir}")
            hf_dataset = None
    
    # If we have HuggingFace dataset metadata, use it
    if hf_dataset is not None:
        print("Using HuggingFace dataset metadata from local cache...")
        print(f"[Dataset Loading] Audio files will be loaded from: {split_dir}")
        for idx, item in enumerate(hf_dataset):
            audio_id = str(item.get("id", item.get("audio_id", f"kespeech_{idx}")))
            transcription = item.get("Text", item.get("text", ""))
            
            # Try to find local audio file by index (strictly from split_dir)
            local_audio_path = os.path.join(split_dir, f"{idx}.wav")
            if not os.path.exists(local_audio_path):
                # Try to find by audio_id if index doesn't work
                local_audio_path = None
            
            if not local_audio_path or not os.path.exists(local_audio_path):
                print(f"Warning: Could not find audio file for index {idx}, ID {audio_id}, skipping")
                continue
            
            try:
                audio_info = sf.info(local_audio_path)
                duration = audio_info.duration
            except:
                duration = 0.0
            
            dataset.append({
                "id": audio_id,
                "speaker_id": item.get("speaker_id", "unknown"),
                "chapter_id": item.get("chapter_id", ""),
                "path": local_audio_path,
                "transcription": transcription,
                "duration": duration,
                "audio": {
                    "path": local_audio_path,
                    "array": None,
                    "sampling_rate": None
                }
            })
        
        print(f"Successfully loaded {len(dataset)} KeSpeech samples from local files with metadata")
        return dataset
    
    # If no HuggingFace metadata, load directly from local files
    print("No HuggingFace metadata found, loading directly from local audio files...")
    print(f"[Dataset Loading] Loading audio files from: {split_dir}")
    print("Note: Transcription will be empty. This may affect evaluation accuracy.")
    
    # Get all .wav files in the split directory (strictly from split_dir)
    import glob
    audio_files = sorted(glob.glob(os.path.join(split_dir, "*.wav")), 
                        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]) if os.path.splitext(os.path.basename(x))[0].isdigit() else float('inf'))
    
    for audio_path in audio_files:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        audio_id = f"kespeech_{base_name}"
        
        # Try to find corresponding .txt file for transcription
        txt_path = os.path.join(split_dir, f"{base_name}.txt")
        transcription = ""
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    transcription = f.read().strip()
            except Exception as e:
                print(f"Warning: Could not read transcription from {txt_path}: {e}")
        
        try:
            audio_info = sf.info(audio_path)
            duration = audio_info.duration
        except:
            duration = 0.0
        
        dataset.append({
            "id": audio_id,
            "speaker_id": "unknown",
            "chapter_id": "",
            "path": audio_path,
            "transcription": transcription,
            "duration": duration,
            "audio": {
                "path": audio_path,
                "array": None,
                "sampling_rate": None
            }
        })
    
    print(f"Successfully loaded {len(dataset)} KeSpeech samples from local files")
    if len(dataset) > 0 and not dataset[0]["transcription"]:
        print("Warning: No transcription found. Evaluation metrics may not be accurate.")
    
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
                # Priority 1: Use direct path if provided
                if ENV_HEAD_SCORE_PATH:
                    head_score_path = ENV_HEAD_SCORE_PATH
                    print(f"[SparseMM] Using direct head score path: {head_score_path}")
                # Priority 2: Check if using uniform distribution
                elif ENV_HEAD_SCORE_FILE == "uniform":
                    head_score_path = "uniform"
                    print(f"[SparseMM] Using uniform head score distribution")
                else:
                    # Priority 3: Construct path from file type and parameters
                    head_score_path = get_audio_head_score_path(
                        score_type=ENV_HEAD_SCORE_FILE,
                        batch=ENV_HEAD_SCORE_BATCH
                    )
                    print(f"[SparseMM] Constructed head score path: {head_score_path}")
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
            
            # Enable based on KV_METHOD
            if KV_METHOD in ("sparsekv", "sparsemm"):
                model.enable_sparsemm_attention(head_score_data, sparsemm_config)
                print("[SparseMM] SparseMM attention enabled successfully")
            elif KV_METHOD == "snapkv":
                model.enable_snapkv_attention(sparsemm_config)
                print("[SnapKV] SnapKV attention enabled successfully")
            elif KV_METHOD == "adakv":
                model.enable_adakv_attention(sparsemm_config)
                print("[AdaKV] AdaKV attention enabled successfully")
            elif KV_METHOD == "pyramidkv":
                model.enable_pyramidkv_attention(sparsemm_config)
                print("[PyramidKV] PyramidKV attention enabled successfully")
            elif KV_METHOD == "mixkv":
                select_method = os.environ.get("SELECT_METHOD", "mixkv")
                sparsemm_config['select_method'] = select_method
                model.enable_mixkv_attention(sparsemm_config)
                print(f"[MixKV] MixKV attention enabled successfully with select_method={select_method}")
            else:
                print(f"[KVCache WARNING] Unknown KV_METHOD '{KV_METHOD}', falling back to SparseMM")
                model.enable_sparsemm_attention(head_score_data, sparsemm_config)
            
            SPARSEM_ENABLED = True
        except Exception as e:
            print(f"[KVCache ERROR] Failed to enable {KV_METHOD}: {e}")
            import traceback
            traceback.print_exc()
            print("[KVCache] Continuing without KVCache compression...")
            SPARSEM_ENABLED = False

    if ENABLE_COMPILE and device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        except Exception:
            pass

    return processor, model, SPARSEM_ENABLED

def update_sparsemm_budget(model, prefill_token_count):
    """
    Update SparseMM/SnapKV/AdaKV/PyramidKV budget for all layers based on prefill token count.
    If ENV_BUDGET_IS_RATIO is True, calculate budget as ratio of prefill tokens.
    Otherwise, use the absolute budget value.
    
    This function should be called BEFORE each inference when budget is a ratio,
    to ensure the budget is computed based on actual sequence length.
    
    Args:
        model: The model with KV compression enabled
        prefill_token_count: Number of tokens in prefill phase
    
    Returns:
        The actual budget value used (integer)
    """
    if not ENV_BUDGET_IS_RATIO:
        # Use absolute budget value, but still need to update it for AdaKV and other methods
        actual_budget = int(ENV_BUDGET)
    else:
        # Calculate budget as ratio of prefill tokens
        budget_ratio = ENV_BUDGET
        actual_budget = int(prefill_token_count * budget_ratio)
    
    # Ensure budget is at least window_size + 1 to avoid errors
    min_budget = ENV_WINDOW_SIZE + 1
    if actual_budget < min_budget:
        actual_budget = min_budget
        if ENV_BUDGET_IS_RATIO:
            print(f"[KVCache] Warning: Calculated budget {int(prefill_token_count * ENV_BUDGET)} is too small, using minimum {min_budget}")
    
    # Update budget in all layers
    try:
        base_model = model.model.language_model  # Gemma3nTextModel
        if hasattr(base_model, 'layers'):
            updated_count = 0
            total_layers = len(base_model.layers)
            for layer_idx, layer in enumerate(base_model.layers):
                has_self_attn = hasattr(layer, 'self_attn')
                has_kv_cluster = has_self_attn and hasattr(layer.self_attn, 'kv_cluster')
                if has_self_attn and has_kv_cluster:
                    kv_cluster = layer.self_attn.kv_cluster
                    # CRITICAL: Always use update_budget() if available to properly reset pyram_init
                    # This is essential for PyramidKV and MixKV to recalculate pyramidal capacity
                    if hasattr(kv_cluster, 'update_budget'):
                        kv_cluster.update_budget(actual_budget)
                        updated_count += 1
                    elif hasattr(kv_cluster, 'base_capacity'):
                        # Fallback for older versions without update_budget method
                        # For AdaKVCluster, we need to update floor_capacity and adaptive_capacity too
                        kv_cluster.base_capacity = actual_budget - kv_cluster.window_size
                        if kv_cluster.base_capacity <= 0:
                            kv_cluster.base_capacity = 1
                        # Update floor_capacity and adaptive_capacity if they exist
                        if hasattr(kv_cluster, 'floor_ratio'):
                            kv_cluster.floor_capacity = int(kv_cluster.base_capacity * kv_cluster.floor_ratio)
                            kv_cluster.adaptive_capacity = kv_cluster.base_capacity - kv_cluster.floor_capacity
                        updated_count += 1
                    elif hasattr(kv_cluster, 'max_capacity_prompt'):
                        # Fallback: Direct assignment (should not happen if update_budget exists)
                        # For PyramidKV/MixKV, this won't reset pyram_init, so it's not ideal
                        # But we still update the budget value
                        kv_cluster.max_capacity_prompt = actual_budget
                        # Try to reset pyram_init if pyram_mode is enabled
                        if hasattr(kv_cluster, 'pyram_mode') and kv_cluster.pyram_mode:
                            kv_cluster.pyram_init = False
                        updated_count += 1
            
            # Also update model config for consistency
            if hasattr(base_model, 'config'):
                base_model.config.max_capacity_prompt = actual_budget
            
            if ENV_BUDGET_IS_RATIO:
                print(f"[KVCache] Updated budget: {prefill_token_count} tokens * {ENV_BUDGET:.2%} = {actual_budget} "
                      f"(base_capacity={actual_budget - ENV_WINDOW_SIZE}, updated {updated_count}/{total_layers} layers)")
            else:
                print(f"[KVCache] Updated budget: {actual_budget} (absolute value, base_capacity={actual_budget - ENV_WINDOW_SIZE}, "
                      f"updated {updated_count}/{total_layers} layers)")
    except (AttributeError, KeyError) as e:
        print(f"[KVCache] Warning: Failed to update budget: {e}")
    
    return actual_budget

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
                    ceil: int = MAX_TOKEN_CEIL,
                    language: str = "chinese") -> int:
    """依据 token/s + buffer 估算所需 tokens，限制在 [floor, ceil]。
    
    For Chinese and other non-English languages, we use a more conservative estimate 
    as they may require more tokens per second (e.g., different tokenization).
    
    For single-sentence ASR (like KeSpeech), the output is limited to prevent unlimited generation.
    """
    lang_lower = language.lower() if language else ASR_LANGUAGE.lower()
    
    # For Chinese and other non-English languages, use a higher token rate estimate
    # Chinese characters may need 1.3-1.5x tokens compared to English
    if lang_lower in ["chinese", "zh"]:
        tokens_per_sec = tokens_per_sec * 1.4  # 40% more tokens for Chinese
        buffer_tokens = int(buffer_tokens * 1.3)  # 30% more buffer
    elif lang_lower in ["french", "german", "spanish"]:
        tokens_per_sec = tokens_per_sec * 1.3  # 30% more tokens for non-English
        buffer_tokens = int(buffer_tokens * 1.2)  # 20% more buffer
    
    est = int(duration_sec * tokens_per_sec) + int(buffer_tokens)
    # Ensure the estimate is capped at MAX_TOKEN_CEIL to prevent unlimited output
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
# ASR 系统提示词（根据语言配置）
def _get_language_prompt(language: str = "chinese") -> str:
    """Generate language-specific transcription prompt following official format.
    
    Key requirements:
    - Faithful transcription (no translation, no summarization)
    - No punctuation marks (ground truth has no punctuation)
    - Preserve all words exactly as spoken
    - Only transcription text (no explanations, no additional text)
    
    This matches the format from gemma_multilingual_librispeech.py which has been verified to work correctly.
    For Chinese, we use English prompt format but specify Chinese language, as the model was trained with English prompts.
    """
    # Normalize language code
    lang_lower = language.lower()
    if lang_lower in ["zh", "chinese"]:
        lang_lower = "chinese"
    elif lang_lower in ["en", "english"]:
        lang_lower = "english"
    
    # Use English prompt format for all languages (matching reference implementation)
    # The model was trained with English prompts, so this works better than Chinese prompts
    # Use detailed format matching reference implementation for better accuracy
    # IMPORTANT: Explicitly specify Simplified Chinese and prohibit other languages/scripts
    # Emphasize complete and accurate transcription for better results
    language_prompts = {
        "chinese": "You are an automatic speech recognizer. Transcribe the complete Chinese speech accurately in Simplified Chinese (简体中文) only. Use ONLY Simplified Chinese characters (汉字), no Traditional Chinese (繁体中文), no English, no other languages or scripts. Transcribe every word and character exactly as spoken. Do not skip any words. Do not summarize or translate. Do not use punctuation marks. Return only the complete transcription in Simplified Chinese without explanations or additional text.",
        "english": "Transcribe faithfully in English without translation or summarization. Do not use punctuation marks. Preserve all words exactly as spoken. Return only the transcription without explanations or additional text."
    }
    
    return language_prompts.get(lang_lower, language_prompts["chinese"])

def _build_asr_messages_with_audio(language: str = "chinese") -> List[Dict]:
    """
    Build ASR messages following the official Gemma-3n format.
    
    Note: Audio is passed separately via processor(audio=...) call,
    so we use None as placeholder in the message.
    This matches the working implementation in gemma_multilingual_librispeech.py
    """
    prompt_text = _get_language_prompt(language)
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

def _clean_asr_output(text: str, language: str = "chinese") -> str:
    """Clean ASR output, extracting only the model's transcription response.
    
    Removes:
    - Special tokens and markers
    - Punctuation marks (to match ground truth format)
    - Extra whitespace
    
    Args:
        text: Raw output text from model
        language: Language code for language-specific processing
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
    
    # Remove punctuation marks (ground truth has no punctuation for Chinese)
    # Remove Chinese punctuation
    s = re.sub(r'[，。！？、；：""''（）【】《》…—～·]', '', s)
    
    # Remove English punctuation
    s = re.sub(r"'", '', s)
    s = re.sub(r'[.!?,;:"()\[\]{}]', '', s)
    s = re.sub(r'[\-]', '', s)  # Remove hyphens
    
    # 检测并处理明显的重复字符模式（如"但是但是但是..."、"转转转"等）
    # 如果连续重复超过3次，可能是模型陷入循环，需要更严格的检测
    if len(s) > 10:
        # 方法1: 检测连续重复的字符（如"但是但是但是"）
        # 查找连续重复超过3次的字符序列
        import re as re_module
        # 匹配连续重复的字符（至少3次）
        repeated_pattern = re_module.compile(r'(.{1,10}?)\1{2,}')
        matches = repeated_pattern.findall(s)
        if matches:
            # 如果找到重复模式，尝试提取第一个出现的位置之前的内容
            first_match = repeated_pattern.search(s)
            if first_match:
                # 返回重复模式开始之前的内容
                s = s[:first_match.start()].strip()
                if len(s) < 5:  # 如果太短，可能是无效输出
                    return ""
        
        # 方法2: 检测是否主要是重复字符（整体重复率）
        char_counts = {}
        for char in s[:100]:  # 检查前100个字符
            char_counts[char] = char_counts.get(char, 0) + 1
        if char_counts:
            max_char, max_count = max(char_counts.items(), key=lambda x: x[1])
            # 如果某个字符占比超过60%，可能是重复循环
            if max_count > len(s[:100]) * 0.6 and max_count > 15:
                # 返回空，因为这是明显的重复循环
                return ""

    # For Chinese: Filter out non-Chinese characters (English, other scripts, etc.)
    if language.lower() in ["chinese", "zh"]:
        # Keep only Simplified Chinese characters (CJK Unified Ideographs)
        # Remove English letters, numbers, and other non-Chinese characters
        # But keep Chinese characters (Unicode range: \u4e00-\u9fff)
        chinese_only = []
        for char in s:
            # Keep Chinese characters (CJK Unified Ideographs)
            if '\u4e00' <= char <= '\u9fff':
                chinese_only.append(char)
            # Keep spaces (but we'll normalize them later)
            elif char == ' ':
                chinese_only.append(char)
        s = ''.join(chinese_only)
        
        # If after filtering we have very few Chinese characters, it might be invalid
        chinese_chars = [c for c in s if '\u4e00' <= c <= '\u9fff']
        if len(chinese_chars) < 2 and len(s) > 0:
            # If we have mostly non-Chinese content, return empty
            return ""
    
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
    
    按照官方文档实现：https://ai.google.dev/gemma/docs/capabilities/audio
    官方示例使用 apply_chat_template 的 tokenize=True, return_dict=True 方式
    """
    # Ensure max_new_tokens is capped to prevent unlimited output
    # For single-sentence ASR, limit to MAX_TOKEN_CEIL
    max_new_tokens = min(max_new_tokens, MAX_TOKEN_CEIL)
    
    # Reset SparseMM state before each sample ONLY if SparseMM is actually enabled
    # This avoids unnecessary overhead when SparseMM is not in use
    if sparsem_enabled:
        reset_sparsemm_state(model)
    
    # === Use same format as multilingual version (which works correctly) ===
    # Build messages with language-specific prompt (audio passed separately)
    # Use ASR_LANGUAGE to ensure correct prompt is used
    messages = _build_asr_messages_with_audio(language=ASR_LANGUAGE)
    
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

    # 收集终止符 - 确保模型能正确停止生成
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
    # Update KV compression budget based on prefill token count
    # For ratio budgets, calculate based on prefill tokens; for absolute budgets, use the value directly
    if sparsem_enabled:
        if hasattr(model_inputs, 'input_ids'):
            prefill_token_count = model_inputs.input_ids.shape[1] if len(model_inputs.input_ids.shape) > 1 else model_inputs.input_ids.shape[0]
        elif 'input_ids' in model_inputs:
            prefill_token_count = model_inputs['input_ids'].shape[1] if len(model_inputs['input_ids'].shape) > 1 else model_inputs['input_ids'].shape[0]
        else:
            prefill_token_count = 1024  # Default estimate
        update_sparsemm_budget(model, prefill_token_count)
    
    # For ASR, length_penalty < 1.0 encourages longer outputs (helps with truncation)
    if LENGTH_PENALTY != 1.0:
        generate_kwargs["length_penalty"] = float(LENGTH_PENALTY)
    else:
        # Default: slight length penalty to encourage longer outputs for ASR
        # For Chinese, use slightly lower penalty to encourage more complete transcription
        length_penalty_value = 0.92 if (ASR_LANGUAGE in ["chinese", "zh"]) else 0.95
        generate_kwargs["length_penalty"] = length_penalty_value  # < 1.0 encourages longer sequences
    
    # NOTE: Do NOT set past_key_values here - let the model use its default cache
    # Setting custom cache can cause generation to fail or produce empty outputs
    
    # Generate
    gen = model.generate(
        **model_inputs,
        **generate_kwargs
    )

    # 解码 - 使用 skip_special_tokens=True
    out = processor.batch_decode(gen, skip_special_tokens=True)
    raw_text = out[0] if out else ""
    
    # Debug: Print raw output for first sample
    if not hasattr(asr_transcribe, '_debug_raw_printed'):
        asr_transcribe._debug_raw_printed = True
        print(f"\n[DEBUG] Generated shape: {gen.shape if isinstance(gen, torch.Tensor) else type(gen)}")
        print(f"[DEBUG] Input length: {model_inputs.get('input_ids', torch.tensor([[]])).shape[-1]}")
        print(f"[DEBUG] Generated length: {gen.shape[-1] if isinstance(gen, torch.Tensor) else 'N/A'}")
        print(f"[DEBUG] Raw decoded text (first 200 chars): {raw_text[:200]}")
    
    # 清理输出（去除可能的提示词回显等，使用语言特定的处理）
    # Normalize language code for cleaning
    clean_language = ASR_LANGUAGE.lower()
    if clean_language in ["zh", "chinese"]:
        clean_language = "chinese"
    elif clean_language in ["en", "english"]:
        clean_language = "english"
    text = _clean_asr_output(raw_text, language=clean_language)
    
    # Debug: Print cleaned output for first sample
    if not hasattr(asr_transcribe, '_debug_cleaned_printed'):
        asr_transcribe._debug_cleaned_printed = True
        print(f"[DEBUG] Cleaned text (first 200 chars): {text[:200]}")

    # 诊断信息 - 改进截断检测（匹配多语言版本）
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
            dyn_max = estimate_tokens(dur, language=ASR_LANGUAGE)
        # Ensure max_new_tokens is capped to prevent unlimited output
        dyn_max = min(dyn_max, MAX_TOKEN_CEIL)
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
    import argparse
    parser = argparse.ArgumentParser(description='KeSpeech ASR Evaluation with Gemma-3n')
    parser.add_argument('--output-file', '-o', type=str, default=None,
                        help='Output file path for results JSON')
    parser.add_argument('--output-dir', '-d', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--timing-output-file', '-t', type=str, default=None,
                        help='Output file path for timing statistics JSON')
    parser.add_argument('--model-path', '-m', type=str, default=None,
                        help='Path to the model directory')
    args = parser.parse_args()
    
    # Override model path if provided
    if args.model_path:
        global LOCAL_MODEL_PATH
        LOCAL_MODEL_PATH = args.model_path
    
    # Override output paths if provided via command line arguments
    global output_file, timing_output_file, result_dir
    if args.output_dir:
        # If output-dir is provided, use it as the base directory
        result_dir = args.output_dir
        os.makedirs(result_dir, exist_ok=True)
        if args.output_file:
            output_file = args.output_file
        else:
            output_file = os.path.join(result_dir, "kespeech_results.json")
        if args.timing_output_file:
            timing_output_file = args.timing_output_file
        else:
            timing_output_file = os.path.join(result_dir, "timing_stats.json")
    elif args.output_file:
        # If only output-file is provided, use it directly
        output_file = args.output_file
        result_dir = os.path.dirname(output_file) or "."
        os.makedirs(result_dir, exist_ok=True)
        if args.timing_output_file:
            timing_output_file = args.timing_output_file
        else:
            timing_output_file = os.path.join(result_dir, "timing_stats.json")
    elif args.timing_output_file:
        # If only timing-output-file is provided
        timing_output_file = args.timing_output_file
        result_dir = os.path.dirname(timing_output_file) or "."
        os.makedirs(result_dir, exist_ok=True)
    
    print(f"结果将保存到: {output_file}")
    print(f"Timing statistics will be saved to: {timing_output_file}")
    
    print("开始 KeSpeech × Gemma-3n 本地 ASR 评测……")
    print(f"Transformers 版本: {TRANSFORMERS_VERSION}")
    print(f"使用设备: {device}")
    print(f"数据根目录: {KESPEECH_DIR}")
    print(f"ASR 语言: {ASR_LANGUAGE} (锚点标签: '{ANCHOR_TAG}')")
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

    samples = load_kespeech_dataset(KESPEECH_DIR, "test")
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
            "split": "test",
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
            "avg_cer": 0.0,
            "char_accuracy": 0.0,
            "avg_wer": 0.0,
            "word_accuracy": 0.0,
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

    with tqdm(total=len(samples), desc="处理 KeSpeech 样本", position=0, leave=True, **tqdm_kwargs) as pbar:
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

                start = time.time()
                if ENABLE_CHUNKING:
                    pred_text, diag = asr_transcribe_with_overlap(processor, model, y, sr, sparsem_enabled)
                    # 统计截断
                    results["summary"]["total_chunks"] += len(diag.get("chunks", []))
                    results["summary"]["truncated_chunks"] += sum(1 for c in diag.get("chunks", []) if c.get("truncated"))
                else:
                    dur = len(y) / float(sr)
                    dyn_max = MAX_NEW_TOKENS_ENV if MAX_NEW_TOKENS_ENV > 0 else estimate_tokens(dur, language=ASR_LANGUAGE)
                    # Ensure max_new_tokens is capped to prevent unlimited output
                    dyn_max = min(dyn_max, MAX_TOKEN_CEIL)
                    pred_text, diag_single = asr_transcribe(processor, model, y, sr, dyn_max, sparsem_enabled)
                    results["summary"]["total_chunks"] += 1
                    results["summary"]["truncated_chunks"] += 1 if diag_single.get("truncated") else 0
                resp_time = time.time() - start

                # Calculate WER/CER with Chinese language setting
                wer = calculate_wer(reference, pred_text, language="chinese")
                cer = calculate_cer(reference, pred_text, language="chinese")
                # Ensure CER is capped at 100% (calculate_cer should already do this, but double-check)
                cer = min(cer, 100.0)
                set_f1 = set_f1_per_sample(reference, pred_text, language="chinese")
                set_f1_list.append(set_f1)
                
                # Print transcription results for debugging
                print(f"\n{'='*80}")
                print(f"[样本 {i+1}] CER: {cer:.2f}%, WER: {wer:.2f}%")
                print(f"[参考文本]: {reference[:100]}{'...' if len(reference) > 100 else ''}")
                print(f"[预测文本]: {pred_text[:100]}{'...' if len(pred_text) > 100 else ''}")
                print(f"{'='*80}")

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

            # 进度显示（cnt已在上面定义）
            # Calculate from actual samples to avoid double counting
            # For Chinese ASR, CER is the primary metric
            if cnt > 0:
                cer_scores = [r["cer"] for r in results["samples"] if "cer" in r]
                if len(cer_scores) > 0:
                    # Ensure all CER scores are capped at 100%
                    cer_scores = [min(cer, 100.0) for cer in cer_scores]
                    cur_avg_cer = min(np.mean(cer_scores), 100.0)  # Limit average CER to 100%
                    # Character accuracy = 100 - CER (both in percentage)
                    cur_char_acc = max(100.0 - cur_avg_cer, 0.0)  # Ensure accuracy is not negative
                else:
                    cur_avg_cer = 100.0
                    cur_char_acc = 0.0
            else:
                cur_avg_cer = 100.0
                cur_char_acc = 0.0
            pbar.set_postfix_str(f"平均CER:{cur_avg_cer:.3f} | Char Acc:{cur_char_acc:.2f}%")
            if is_screen_env and (cnt % 10 == 0 or cnt == len(samples)):
                print(f"  进度: {cnt}/{len(samples)} ({cnt/len(samples)*100:.1f}%), 平均CER: {cur_avg_cer:.3f}, Char Acc: {cur_char_acc:.2f}%")
            
            # Real-time save: Update JSON file after each sample to ensure progress is saved immediately
            # This ensures that if the process is interrupted, all processed samples are preserved
            save_results_realtime(output_file, results, timing_stats, len(samples))
            timing_stats.export_to_json(timing_output_file)
            
            pbar.update()

    total = results["summary"]["total_samples"]
    if total > 0:
        # Calculate statistics from actual samples (not from accumulated totals)
        # This ensures accuracy is calculated correctly without double counting
        # For Chinese ASR, CER is the primary metric
        wer_scores = [r["wer"] for r in results["samples"] if "wer" in r]
        cer_scores = [r["cer"] for r in results["samples"] if "cer" in r]
        
        # Calculate CER (primary metric for Chinese ASR)
        if len(cer_scores) > 0:
            # Ensure all CER scores are capped at 100%
            cer_scores = [min(cer, 100.0) for cer in cer_scores]
            # CER is stored as percentage (0-100), calculate mean directly from samples
            results["summary"]["avg_cer"] = float(np.mean(cer_scores))
            # Character accuracy = 100 - CER (both in percentage)
            results["summary"]["char_accuracy"] = max(100.0 - results["summary"]["avg_cer"], 0.0)  # Ensure accuracy is not negative
        else:
            results["summary"]["avg_cer"] = 100.0
            results["summary"]["char_accuracy"] = 0.0
        
        # Calculate WER (secondary metric, still useful for reference)
        if len(wer_scores) > 0:
            # WER is stored as percentage (0-100), calculate mean directly from samples
            results["summary"]["avg_wer"] = float(np.mean(wer_scores))
            # Word accuracy = 100 - WER (both in percentage)
            results["summary"]["word_accuracy"] = 100.0 - results["summary"]["avg_wer"]
        else:
            results["summary"]["avg_wer"] = 100.0
            results["summary"]["word_accuracy"] = 0.0
        
        results["summary"]["timing"]["avg_response_time"] = (
            results["summary"]["timing"]["total_response_time"] / total
        )
        results["summary"]["macro_set_f1"] = float(np.mean(set_f1_list))

    # 按说话人统计 CER 和 WER
    from collections import defaultdict
    speaker_cer = defaultdict(list)
    speaker_wer = defaultdict(list)
    for r in results["samples"]:
        speaker_id = r.get("speaker_id", "unknown")
        if "cer" in r:
            speaker_cer[speaker_id].append(r["cer"])
        if "wer" in r:
            speaker_wer[speaker_id].append(r["wer"])
    speaker_avg_cer = {spk: (sum(c)/len(c) if c else 0.0) for spk, c in speaker_cer.items()}
    speaker_avg_wer = {spk: (sum(w)/len(w) if w else 0.0) for spk, w in speaker_wer.items()}
    results["summary"]["speaker_cer"] = speaker_avg_cer
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
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.flush()  # Explicitly flush to ensure data is written
        os.fsync(f.fileno())  # Force write to disk for file system level sync

    # Export final timing statistics to separate JSON file
    timing_stats.export_to_json(timing_output_file)

    trunc_info = f"{results['summary'].get('truncated_chunks', 0)}/{results['summary'].get('total_chunks', 0)} chunks truncated"
    print("\n=== KeSpeech × Gemma-3n 本地 ASR 评测结果摘要（E4B 优化版） ===")
    print(f"总样本数: {total}")
    print(f"说话人数量: {len(speaker_avg_cer)}")
    print(f"平均 CER: {results['summary']['avg_cer']:.4f}% (主要指标)")
    print(f"Character Accuracy: {results['summary']['char_accuracy']:.4f}%")
    print(f"平均 WER: {results['summary']['avg_wer']:.4f}% (参考指标)")
    print(f"Word Accuracy: {results['summary']['word_accuracy']:.4f}%")
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
