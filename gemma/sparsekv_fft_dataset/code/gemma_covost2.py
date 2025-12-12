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
from datasets import load_dataset, load_from_disk
import sacrebleu

# Set up path for imports - must be done before importing model modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Get parent directory (sparsekv_fft_dataset)
os.chdir(current_dir)
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
# 解析命令行参数（优先于环境变量）
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='CoVoST2 Speech Translation Evaluation with Gemma-3n')
    parser.add_argument('--direction', type=str, default=None,
                        help='Translation direction: en_zh or zh_en (overrides COVOST2_DIRECTION env var)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to the model (overrides MODEL_PATH env var)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (overrides RESULTS_DIR env var)')
    parser.add_argument('--timing-output-file', type=str, default=None,
                        help='Output file for timing statistics')
    return parser.parse_args()

# 解析命令行参数
args = parse_args()

# ------------------------------
# 基本配置（命令行参数优先，然后环境变量，最后默认值）
# ------------------------------
LOCAL_MODEL_PATH = args.model_path if args.model_path else os.environ.get("MODEL_PATH", "/mnt/public/models/gemma-3n-E2B-it")
# CoVoST2 dataset configuration
COVOST2_DIRECTION = args.direction if args.direction else os.environ.get("COVOST2_DIRECTION", "en_zh")  # "en_zh" or "zh_en"
COVOST2_DATASET_PATH = os.environ.get("COVOST2_DATASET_PATH", None)

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


# 生成控制（ST 友好：默认不做 ngram/惩罚）
NO_REPEAT_NGRAM_SIZE = int(os.environ.get("NO_REPEAT_NGRAM_SIZE", "0"))  # ← 改为 0
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", "1.0"))  # ← 改为 1.0
LENGTH_PENALTY = float(os.environ.get("LENGTH_PENALTY", "1.0"))

# 结果输出 - 优先使用命令行参数
if args.output_dir:
    result_dir = args.output_dir
    os.makedirs(result_dir, exist_ok=True)
    output_file = f'{result_dir}/CoVoST2_{COVOST2_DIRECTION.upper()}_results_gemma.json'
    if args.timing_output_file:
        timing_output_file = args.timing_output_file
    else:
        timing_output_file = f'{result_dir}/timing_stats_gemma_{COVOST2_DIRECTION}.json'
elif os.environ.get("RESULTS_DIR", None):
    result_dir = os.environ.get("RESULTS_DIR")
    os.makedirs(result_dir, exist_ok=True)
    output_file = f'{result_dir}/CoVoST2_{COVOST2_DIRECTION.upper()}_results_gemma.json'
    timing_output_file = f'{result_dir}/timing_stats_gemma_{COVOST2_DIRECTION}.json'
else:
    # 如果没有指定输出目录，使用默认位置（但不在 code 目录下）
    base_output_dir = "/mnt/users/wangyuxuan-20250915/Audio/Results/Gemma_3n_E2B_it/datasets/CoVoST2"
    if COVOST2_DIRECTION == "en_zh":
        result_dir = os.path.join(base_output_dir, "EN_ZH", "vanilla", "full_cache")
    else:  # zh_en
        result_dir = os.path.join(base_output_dir, "ZH_EN", "vanilla", "full_cache")
    os.makedirs(result_dir, exist_ok=True)
    output_file = f'{result_dir}/CoVoST2_{COVOST2_DIRECTION.upper()}_results_gemma.json'
    timing_output_file = f'{result_dir}/timing_stats_gemma_{COVOST2_DIRECTION}.json'

# 根据 ST 方向设置输出语言
if COVOST2_DIRECTION == "en_zh":
    OUTPUT_LANGUAGE = "zh"  # 英语到中文，输出中文
    print("Translation direction: English -> Chinese (输出中文)")
else:  # zh_en
    OUTPUT_LANGUAGE = "en"  # 中文到英语，输出英文
    print("Translation direction: Chinese -> English (输出英文)")

print(f"结果将保存到: {output_file}")
print(f"Timing statistics will be saved to: {timing_output_file}")

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

def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER).
    Returns CER as percentage (0-100).
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

def calculate_bleu_score(predictions, references, direction="en_zh"):
    """Calculate BLEU score following AudioBench style
    
    Args:
        predictions: List of prediction strings
        references: List of reference strings
        direction: Translation direction ('en_zh' or 'zh_en')
                   - 'en_zh': English to Chinese, use 'zh' tokenizer
                   - 'zh_en': Chinese to English, use '13a' tokenizer
    """
    try:
        # Validate inputs
        if not predictions or not references:
            return 0.0
        
        if len(predictions) != len(references):
            print(f"Error: predictions ({len(predictions)}) and references ({len(references)}) have different lengths")
            return 0.0
        
        # Prepare predictions and references following AudioBench style
        # Replace empty strings with "empty" placeholder
        clean_predictions = []
        clean_references = []
        for pred, ref in zip(predictions, references):
            pred_str = str(pred) if pred else ""
            ref_str = str(ref) if ref else ""
            
            if len(pred_str) == 0:
                pred_str = "empty"
            if len(ref_str) == 0:
                ref_str = "empty"
            
            clean_predictions.append(pred_str)
            clean_references.append(ref_str)
        
        # Select tokenizer based on target language
        # en_zh: target is Chinese, use 'zh' tokenizer
        # zh_en: target is English, use '13a' tokenizer (default mteval)
        if direction == "en_zh":
            tokenize_option = "zh"
        else:  # zh_en
            tokenize_option = "13a"
        
        # Use sacrebleu.corpus_bleu with correct format
        # references should be [[ref1, ref2, ref3, ...]] - one reference set containing all references
        result = sacrebleu.corpus_bleu(clean_predictions, [clean_references], tokenize=tokenize_option)
        return result.score
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        traceback.print_exc()
        return 0.0

def calculate_chrf_score(predictions, references):
    """Calculate chrF score following AudioBench style
    
    Args:
        predictions: List of prediction strings
        references: List of reference strings
    """
    try:
        # Validate inputs
        if not predictions or not references:
            return 0.0
        
        if len(predictions) != len(references):
            print(f"Error: predictions ({len(predictions)}) and references ({len(references)}) have different lengths")
            return 0.0
        
        # Prepare predictions and references following AudioBench style
        # Replace empty strings with "empty" placeholder
        clean_predictions = []
        clean_references = []
        for pred, ref in zip(predictions, references):
            pred_str = str(pred) if pred else ""
            ref_str = str(ref) if ref else ""
            
            if len(pred_str) == 0:
                pred_str = "empty"
            if len(ref_str) == 0:
                ref_str = "empty"
            
            clean_predictions.append(pred_str)
            clean_references.append(ref_str)
        
        # Use sacrebleu.corpus_chrf with correct format
        # references should be [[ref1, ref2, ref3, ...]] - one reference set containing all references
        result = sacrebleu.corpus_chrf(clean_predictions, [clean_references])
        # chrF score is returned as 0-1, multiply by 100 to match BLEU format (0-100)
        return result.score * 100.0
    except Exception as e:
        print(f"Error calculating chrF score: {e}")
        traceback.print_exc()
        return 0.0

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
        
        # Calculate CER statistics from actual samples (not from accumulated totals)
        # This ensures accuracy is calculated correctly without double counting
        cer_scores = [r["cer"] for r in samples if "cer" in r]
        bleu_scores = [r["bleu"] for r in samples if "bleu" in r]
        chrf_scores = [r["chrf"] for r in samples if "chrf" in r]
        
        if len(cer_scores) > 0:
            cer_mean = np.mean(cer_scores)
        else:
            cer_mean = 1.0
        
        if len(bleu_scores) > 0:
            bleu_mean = np.mean(bleu_scores)
        else:
            bleu_mean = 0.0
        
        if len(chrf_scores) > 0:
            chrf_mean = np.mean(chrf_scores)
        else:
            chrf_mean = 0.0
        
        # Calculate overall BLEU and chrF
        predictions = [r.get("prediction", "") for r in samples]
        references = [r.get("ground_truth", "") for r in samples]
        overall_bleu = calculate_bleu_score(predictions, references, COVOST2_DIRECTION) if len(predictions) > 0 else 0.0
        overall_chrf = calculate_chrf_score(predictions, references) if len(predictions) > 0 else 0.0
        
        # Note: CoVoST2 doesn't have speaker_id, so we skip speaker-level statistics
        
        # Build real-time results structure (maintain original structure but update summary)
        realtime_results = results_dict.copy()
        # Use mean from actual samples, not from accumulated totals (avoids double counting)
        realtime_results["summary"]["avg_cer"] = float(cer_mean)
        realtime_results["summary"]["avg_bleu"] = float(bleu_mean)
        realtime_results["summary"]["avg_chrf"] = float(chrf_mean)
        realtime_results["summary"]["overall_bleu"] = float(overall_bleu)
        realtime_results["summary"]["overall_chrf"] = float(overall_chrf)
        realtime_results["summary"]["progress"] = {
            "processed": len(samples),
            "total": dataset_size,
            "percentage": len(samples) / dataset_size * 100 if dataset_size > 0 else 0
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
# 数据加载（CoVoST2 数据集）
# ------------------------------

def load_covost2_dataset(dataset_path, direction="en_zh"):
    """Load CoVoST2 dataset from local path"""
    try:
        # Try loading from disk first
        if os.path.exists(dataset_path):
            print(f"Loading CoVoST2 dataset from local path: {dataset_path}")
            dataset = load_from_disk(dataset_path)
            # If it's a DatasetDict, get the 'test' split
            if hasattr(dataset, 'keys') and 'test' in dataset:
                dataset = dataset['test']
            elif hasattr(dataset, 'split') and dataset.split == 'test':
                pass  # Already the test split
            print(f"Loaded {len(dataset)} samples from CoVoST2-{direction}")
            return dataset
        else:
            print(f"Error: Dataset path does not exist: {dataset_path}")
            return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        traceback.print_exc()
        return None

def covost2_doc_to_audio(doc):
    """Load audio data from CoVoST2 documents"""
    # Check if doc is a dictionary
    if not isinstance(doc, dict):
        print(f"Error: doc is not a dictionary, got type: {type(doc)}")
        if isinstance(doc, str):
            print(f"  doc content (first 100 chars): {doc[:100]}")
        return None
    
    if "context" not in doc:
        return None
    
    audio_data = doc["context"]
    
    # Handle different audio formats
    if isinstance(audio_data, dict):
        # If it's a dict with 'array' and 'sampling_rate'
        if "array" in audio_data:
            audio_np = np.array(audio_data["array"], dtype=np.float32)
            sr = audio_data.get("sampling_rate", 16000)
            return audio_np, sr
        elif "path" in audio_data:
            # Load from file path
            try:
                audio_np, sr = librosa.load(audio_data["path"], sr=16000, mono=True)
                return audio_np.astype(np.float32), sr
            except Exception as e:
                print(f"Error loading audio from path: {e}")
                return None
    elif isinstance(audio_data, np.ndarray):
        # Direct numpy array
        return audio_data.astype(np.float32), 16000
    elif isinstance(audio_data, (list, tuple)):
        # List or tuple of audio data
        audio_np = np.array(audio_data, dtype=np.float32)
        return audio_np, 16000
    else:
        print(f"Unknown audio format: {type(audio_data)}")
        return None

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
                estimated_seq_len = 1024  # Reasonable estimate for audio ST
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
# ST (Speech Translation) 生成
# ------------------------------
def _get_st_sys_prompt(direction: str = None) -> str:
    """生成语音翻译系统提示，明确指定输出语言要求"""
    if direction is None:
        direction = COVOST2_DIRECTION
    
    if direction == "en_zh":
        # 英语到中文：明确要求简体中文，使用中英文双语强调
        return (
            "You are a speech translation model. "
            "When translating English speech to Chinese, you MUST output ONLY Simplified Chinese (简体中文). "
            "Do NOT use Traditional Chinese (繁体中文), English, or any other language. "
            "Output only the translation text in Simplified Chinese, without any prefixes, labels, or explanations."
        )
    else:  # zh_en - 两阶段：先ASR，再翻译
        # 第一阶段：中文ASR系统提示
        return (
            "You are a Chinese speech recognition (ASR) model. "
            "Your task is to transcribe Chinese speech into Chinese text. "
            "You MUST output ONLY Simplified Chinese characters. "
            "Do NOT output English, pinyin, or any other language. "
            "Output only the transcribed Chinese text, without any explanations, labels, or markers."
        )

def _get_st_user_instruction(direction: str) -> str:
    """根据翻译方向生成用户指令，确保语言一致"""
    if direction == "en_zh":
        # 英语到中文：明确要求简体中文，多次强调
        return (
            "Listen to the provided English speech and translate it into Simplified Chinese (简体中文). "
            "You must use Simplified Chinese characters only. "
            "Do not use Traditional Chinese characters. "
            "Output only the translation text in Simplified Chinese, without any English text, explanations, labels, or markers."
        )
    else:  # zh_en - 第一阶段：中文ASR用户指令
        return (
            "Listen to the provided Chinese speech and transcribe it into Simplified Chinese text. "
            "Output only the transcribed Chinese text, without any English, pinyin, explanations, or markers."
        )

def _get_asr_sys_prompt(direction: str) -> str:
    """ASR系统提示（第一阶段）"""
    if direction == "zh_en":
        # 中文ASR
        return (
            "You are a Chinese speech recognition (ASR) model. "
            "Your task is to transcribe Chinese speech into Chinese text. "
            "You MUST output ONLY Simplified Chinese characters. "
            "Do NOT output English, pinyin, or any other language. "
            "Output only the transcribed Chinese text, without any explanations, labels, or markers."
        )
    else:  # en_zh
        # 英文ASR
        return (
            "You are an English speech recognition (ASR) model. "
            "Your task is to transcribe English speech into English text. "
            "You MUST output ONLY English text. "
            "Do NOT output Chinese characters, pinyin, or any other language. "
            "Output only the transcribed English text, without any explanations, labels, or markers."
        )

def _get_asr_user_instruction(direction: str) -> str:
    """ASR用户指令（第一阶段）"""
    if direction == "zh_en":
        # 中文ASR
        return (
            "Listen to the provided Chinese speech and transcribe it into Simplified Chinese text. "
            "Output only the transcribed Chinese text, without any English, pinyin, explanations, or markers."
        )
    else:  # en_zh
        # 英文ASR
        return (
            "Listen to the provided English speech and transcribe it into English text. "
            "Output only the transcribed English text, without any Chinese characters, explanations, or markers."
        )

def _get_translation_sys_prompt(direction: str) -> str:
    """翻译系统提示（第二阶段）"""
    if direction == "zh_en":
        # 中文到英文翻译
        return (
            "You are a translation model. "
            "Your task is to translate Chinese text into English. "
            "You MUST output ONLY English text. "
            "Do NOT output Chinese characters, pinyin, or any other language. "
            "Output only the English translation, without any explanations, labels, or markers."
        )
    else:  # en_zh
        # 英文到中文翻译
        return (
            "You are a translation model. "
            "Your task is to translate English text into Simplified Chinese (简体中文). "
            "You MUST output ONLY Simplified Chinese characters. "
            "Do NOT use Traditional Chinese (繁体中文), English, or any other language. "
            "Output only the Simplified Chinese translation, without any explanations, labels, or markers."
        )

def _get_translation_user_instruction(source_text: str, direction: str) -> str:
    """翻译用户指令（第二阶段）"""
    if direction == "zh_en":
        # 中文到英文翻译
        return (
            f"Translate the following Chinese text into English: {source_text}\n"
            "Output only the English translation, without any Chinese characters, explanations, or markers."
        )
    else:  # en_zh
        # 英文到中文翻译
        return (
            f"Translate the following English text into Simplified Chinese (简体中文): {source_text}\n"
            "You must use Simplified Chinese characters only. "
            "Do not use Traditional Chinese characters. "
            "Output only the Simplified Chinese translation, without any English text, explanations, labels, or markers."
        )

def _build_st_messages(audio_np: np.ndarray, direction: str = None) -> List[Dict]:
    """构建语音翻译消息，根据 direction 设置正确的输出语言"""
    if direction is None:
        direction = COVOST2_DIRECTION
    
    system_prompt = _get_st_sys_prompt(direction)
    user_instruction = _get_st_user_instruction(direction)
    
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_np},
                {"type": "text", "text": user_instruction}
            ]
        },
    ]

def _build_asr_messages(audio_np: np.ndarray, direction: str) -> List[Dict]:
    """构建ASR消息（第一阶段）"""
    return [
        {"role": "system", "content": [{"type": "text", "text": _get_asr_sys_prompt(direction)}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_np},
                {"type": "text", "text": _get_asr_user_instruction(direction)}
            ]
        },
    ]

def _build_translation_messages(source_text: str, direction: str) -> List[Dict]:
    """构建翻译消息（第二阶段）"""
    return [
        {"role": "system", "content": [{"type": "text", "text": _get_translation_sys_prompt(direction)}]},
        {
            "role": "user",
            "content": [{"type": "text", "text": _get_translation_user_instruction(source_text, direction)}]
        },
    ]

# 更保守：不要用会误删正文的 <sep> 规则
_SPECIAL_TAG_RE = re.compile(
    r"(?:^```.*?\n|```$|<start_of_turn>|<end_of_turn>|<eot>|<\|eot\|>|<\|endoftext\|>|<\w+_soft_token>)",
    flags=re.IGNORECASE | re.DOTALL
)

def _clean_asr_output(text: str, direction: str = None) -> str:
    """清洗ASR输出"""
    if not text:
        return ""
    
    if direction is None:
        direction = COVOST2_DIRECTION
    
    s = text.strip()
    
    # 移除常见的ASR标记和前缀
    if direction == "zh_en":
        # 中文ASR标记
        asr_markers = [
            "transcription:", "transcribed text:", "chinese:", "中文:", 
            "transcription text:", "中文转录:", "转录文本:",
            "simplified chinese:", "简体中文:"
        ]
    else:  # en_zh
        # 英文ASR标记
        asr_markers = [
            "transcription:", "transcribed text:", "english:", "英文:",
            "transcription text:", "英文转录:", "english transcription:",
            "english text:"
        ]
    
    for marker in asr_markers:
        if s.lower().startswith(marker.lower()):
            s = s[len(marker):].strip()
            s = re.sub(r"^[:\-\s]+", "", s)
            break
    
    # 去掉特殊标记
    s = _SPECIAL_TAG_RE.sub("", s).strip()
    
    # 规范化空白字符
    s = re.sub(r"\s+", " ", s)
    
    return s

def _clean_st_output(text: str, direction: str = None) -> str:
    """清洗翻译输出，确保语言正确"""
    if not text:
        return ""
    if direction is None:
        direction = COVOST2_DIRECTION
    
    s = text.strip()

    # 移除常见的翻译标记和前缀（但要小心不要误删中文内容）
    # 只移除出现在开头的标记
    translation_markers = [
        "translation:", "translated text:", "chinese:", "english:", 
        "中文:", "英文:", "翻译:", "译文:",
        "chinese translation:", "english translation:",
        "中文翻译:", "英文翻译:", "简体中文:", "繁体中文:",
        "simplified chinese:", "traditional chinese:"
    ]
    for marker in translation_markers:
        # 只检查开头的标记（不区分大小写）
        if s.lower().startswith(marker.lower()):
            s = s[len(marker):].strip()
            # 移除可能的分隔符（冒号、破折号等）
            s = re.sub(r"^[:\-\s]+", "", s)
            break

    # 去掉围栏/特殊占位与常见特殊 token
    s = _SPECIAL_TAG_RE.sub("", s).strip()

    # 剥离成对引号（但要小心，中文引号可能是有意义的）
    # 只移除英文引号
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()

    # 规范化空白字符
    s = re.sub(r"\s+", " ", s)
    
    # 对于 en_zh 方向，确保输出是中文
    # 如果输出完全是英文或数字，可能是模型理解错误
    if direction == "en_zh":
        original_s = s
        
        # 1. 检测并过滤重复字符/单词（复读问题）
        # 检测重复的中文字符（如 "拉 拉 拉 拉..."）
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', s)
        if len(chinese_chars) > 0:
            # 检查是否有单个字符重复超过5次
            char_counts = {}
            for char in chinese_chars:
                char_counts[char] = char_counts.get(char, 0) + 1
            
            # 如果某个字符出现频率超过50%，且总字符数>10，可能是复读
            total_chars = len(chinese_chars)
            if total_chars > 10:
                max_char_count = max(char_counts.values())
                if max_char_count > total_chars * 0.5:
                    # 找到重复最多的字符
                    most_common_char = max(char_counts, key=char_counts.get)
                    # 检查是否真的是重复模式（如 "拉 拉 拉" 或 "拉拉拉"）
                    repeat_pattern = re.compile(rf'[{most_common_char}\s]+')
                    if repeat_pattern.search(s):
                        print(f"  ⚠️  Warning: Output contains repetitive characters (direction=en_zh). Detected '{most_common_char}' repeated {max_char_count} times. Original: {s[:150]}")
                        s = ""
                        return s
        
        # 检测重复的单词（如 "跑 跑 跑"）
        words = s.split()
        if len(words) > 5:
            # 检查是否有单词重复超过总单词数的50%
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            max_word_count = max(word_counts.values())
            if max_word_count > len(words) * 0.5:
                most_common_word = max(word_counts, key=word_counts.get)
                print(f"  ⚠️  Warning: Output contains repetitive words (direction=en_zh). Detected '{most_common_word}' repeated {max_word_count} times. Original: {s[:150]}")
                s = ""
                return s
        
        # 2. 检测重复指令（模型直接输出指令而不是翻译）
        instruction_keywords = [
            "听取提供的英语语音",
            "将其翻译成简体中文",
            "只使用简体中文字符",
            "不要使用繁体中文",
            "只输出简体中文翻译文本"
        ]
        instruction_count = sum(1 for keyword in instruction_keywords if keyword in s)
        if instruction_count >= 3:
            print(f"  ⚠️  Warning: Output appears to be repeating the instruction instead of translating (direction=en_zh). Original: {s[:150]}")
            s = ""
            return s
        
        # 3. 检查是否包含中文字符
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', s))
        if not has_chinese and len(s) > 0:
            # 如果没有任何中文字符，可能是模型输出了英文
            # 记录警告以便调试
            print(f"  ⚠️  Warning: Output contains no Chinese characters (direction=en_zh). Output: {s[:100]}")
        # 检查是否包含繁体中文（可选，如果需要可以添加繁体转简体）
        # 这里只做检测，不转换（需要额外库如 opencc）
        has_traditional = bool(re.search(r'[繁體]', s))  # 简单的繁体检测
        if has_traditional:
            print(f"  ⚠️  Warning: Output may contain Traditional Chinese characters. Consider converting to Simplified Chinese.")
    
    # 对于 zh_en 方向，确保输出是英文
    # 如果输出包含中文字符，必须移除（因为这是中文到英文的翻译）
    elif direction == "zh_en":
        # 首先快速检测拼音（在检查中文之前）
        words = s.lower().split()
        if len(words) > 0:
            # 快速拼音检测：检查常见拼音词
            quick_pinyin_words = ['pian', 'tuo', 'yici', 'luo', 'qiyuan', 'yuanzi', 'zhongwen', 'pingqiu', 'fushi', 'qiao', 'shu', 'lei', 'wei', 'shi', 'guo', 'ma', 'le', 'ne', 'ta', 'wo', 'ni', 'zhe', 'zhi', 'chu', 'ding', 'yu', 'qi', 'yi', 'de']
            quick_pinyin_count = sum(1 for w in words if w in quick_pinyin_words)
            # 如果包含多个拼音词，很可能是拼音输出
            if quick_pinyin_count >= 3:
                print(f"  ⚠️  Warning: Output appears to be pinyin (detected {quick_pinyin_count} pinyin words). Original: {s[:150]}")
                s = ""
                return s
        
        # 检查是否包含中文字符
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', s))
        if has_chinese:
            original_s = s
            
            # 智能处理：尝试识别常见的"中文：英文"或"中文 英文"格式
            # 查找分隔符（冒号、破折号、换行等）后的英文部分
            # 使用正则表达式查找最后一个包含英文的部分
            # 匹配模式：中文 + 分隔符 + 英文
            patterns = [
                r'[\u4e00-\u9fff]+[：:]\s*([A-Za-z].*)',  # 中文：英文
                r'[\u4e00-\u9fff]+[—\-]\s*([A-Za-z].*)',  # 中文—英文
                r'[\u4e00-\u9fff]+\s+([A-Za-z].*)',       # 中文 英文
            ]
            
            extracted_english = None
            for pattern in patterns:
                match = re.search(pattern, s)
                if match:
                    extracted_english = match.group(1).strip()
                    break
            
            if extracted_english and len(extracted_english) > 0:
                # 如果成功提取了英文部分，使用提取的部分
                s = extracted_english
                # 规范化空白字符
                s = re.sub(r"\s+", " ", s)
                print(f"  ⚠️  Warning: Output contained Chinese characters (direction=zh_en). Extracted English part. Original: {original_s[:150]}, Extracted: {s[:150]}")
            else:
                # 如果没有找到明显的分隔格式，直接移除所有中文字符
                s_no_chinese = re.sub(r'[\u4e00-\u9fff]+', '', s).strip()
                
                if len(s_no_chinese) > 0:
                    s = s_no_chinese
                    # 规范化空白字符
                    s = re.sub(r"\s+", " ", s)
                    print(f"  ⚠️  Warning: Output contained Chinese characters (direction=zh_en). Removed Chinese parts. Original: {original_s[:150]}, Cleaned: {s[:150]}")
                else:
                    # 如果移除中文后为空，说明输出全是中文，返回空
                    s = ""
                    print(f"  ⚠️  Warning: Output contained only Chinese characters (direction=zh_en). Removed all Chinese. Original: {original_s[:150]}")
        else:
            # 没有中文字符，但需要检查是否是有效的英文输出
            original_s = s
            
            # 1. 检测并移除拒绝翻译的回复
            rejection_patterns = [
                r"i'?m\s+(not\s+)?(unable|sorry|cannot|can'?t)",
                r"i\s+(am\s+)?(unable|sorry|cannot|can'?t)",
                r"unable\s+to\s+fulfill",
                r"cannot\s+(fulfill|translate|provide)",
                r"please\s+provide\s+(the\s+)?(speech|audio|chinese)",
                r"need\s+(the\s+)?(chinese\s+)?speech",
            ]
            for pattern in rejection_patterns:
                if re.search(pattern, s, re.IGNORECASE):
                    print(f"  ⚠️  Warning: Output appears to be a rejection message (direction=zh_en). Original: {s[:150]}")
                    s = ""
                    return s
            
            # 2. 检测拼音输出（包含大量拼音特征）- 更严格的检测
            words = s.lower().split()
            if len(words) > 0:
                # 检查是否包含声调符号
                has_pinyin_tones = bool(re.search(r'[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]', s))
                
                # 扩展的拼音词列表（包括更多常见拼音）
                pinyin_words = [
                    'shi', 'de', 'yi', 'wei', 'yu', 'qi', 'ma', 'le', 'ne', 'ta', 'wo', 'ni', 'zhe', 'zhi', 
                    'chu', 'guo', 'yuan', 'ding', 'pian', 'tuo', 'yici', 'luo', 'qiyuan', 'yuanzi', 'la', 
                    'zhongwen', 'pingqiu', 'fushi', 'qiao', 'ke', 'shu', 'lei', 'wei', 'shi', 'guo', 'luo',
                    'ma', 'qiyuan', 'yuanzi', 'la', 'ding', 'yu', 'zhongwen', 'yi', 'wei', 'dian'
                ]
                pinyin_words_count = sum(1 for w in words if w in pinyin_words)
                
                # 检查是否主要是短单词（拼音特征：大部分是1-4个字母）
                if len(words) > 2:
                    short_words_ratio = sum(1 for w in words if len(w) <= 4) / len(words)
                else:
                    short_words_ratio = 0
                
                # 检查是否包含连续的拼音模式（如 "pian tuo", "zhongwen yi"）
                pinyin_patterns = [
                    r'\b(pian|tuo|yici|luo|qiyuan|yuanzi|zhongwen|pingqiu|fushi|qiao|shu|lei)\s+\w+',
                    r'\w+\s+(pian|tuo|yici|luo|qiyuan|yuanzi|zhongwen|pingqiu|fushi|qiao|shu|lei)\b',
                ]
                has_pinyin_pattern = any(re.search(pattern, s, re.IGNORECASE) for pattern in pinyin_patterns)
                
                # 更严格的判定：包含声调符号，或者（短单词比例高且拼音词多），或者有明显的拼音模式
                is_pinyin = (
                    has_pinyin_tones or 
                    (short_words_ratio > 0.5 and pinyin_words_count > len(words) * 0.25) or
                    (has_pinyin_pattern and pinyin_words_count > 2)
                )
                
                if is_pinyin:
                    print(f"  ⚠️  Warning: Output appears to be pinyin instead of English (direction=zh_en). Original: {s[:150]}")
                    s = ""
                    return s
            
            # 3. 检测无意义输出（如数字序列、重复单词等）
            if len(words) >= 5:
                # 检测数字序列（如 "one two three four five"）
                number_words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
                if all(w in number_words for w in words[:min(8, len(words))]):
                    print(f"  ⚠️  Warning: Output appears to be a number sequence (direction=zh_en). Original: {s[:150]}")
                    s = ""
                    return s
                
                # 检测重复单词
                unique_words = set(words)
                if len(unique_words) <= 2 and len(words) > 5:
                    print(f"  ⚠️  Warning: Output appears to be repetitive (direction=zh_en). Original: {s[:150]}")
                    s = ""
                    return s
    
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
def _generate_text(processor, model, messages: List[Dict], max_new_tokens: int, 
                   sparsem_enabled: bool = False, audio_np: np.ndarray = None, 
                   sr: int = None) -> Tuple[str, Dict]:
    """
    通用生成函数，支持音频+文本或纯文本输入
    返回：
      text: 生成的文本
      info: 诊断信息
    """
    templated = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    if audio_np is not None and sr is not None:
        # 音频+文本输入
        model_inputs = processor(
            text=[templated],
            audio=[audio_np],
            sampling_rate=sr,
            return_tensors="pt",
            padding=False
        )
    else:
        # 纯文本输入
        model_inputs = processor(
            text=[templated],
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

    generate_kwargs = {
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": terminators if terminators else processor.tokenizer.eos_token_id,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "use_cache": True,
    }
    
    if sparsem_enabled:
        generate_kwargs["past_key_values"] = DynamicCacheSplitHeadFlatten()
    elif ENV_BUDGET_RAW != "4096" or ENV_BUDGET != 4096.0:
        generate_kwargs["past_key_values"] = BudgetLimitedKVCache(budget=ENV_BUDGET, seed=42)
    
    gen = model.generate(
        **model_inputs,
        **generate_kwargs,
        no_repeat_ngram_size=int(NO_REPEAT_NGRAM_SIZE) if NO_REPEAT_NGRAM_SIZE > 0 else None,
        repetition_penalty=float(REPETITION_PENALTY) if REPETITION_PENALTY != 1.0 else None,
        length_penalty=float(LENGTH_PENALTY),
        early_stopping=True,
        return_dict_in_generate=False
    )

    # 只解码新增 tokens
    new_ids = None
    if "input_ids" in model_inputs and isinstance(gen, torch.Tensor):
        new_ids = gen[:, model_inputs["input_ids"].shape[-1]:]
    elif isinstance(gen, torch.Tensor):
        new_ids = gen
    else:
        try:
            new_ids = gen
        except Exception:
            new_ids = None

    # 解码
    out = processor.batch_decode(
        new_ids if isinstance(new_ids, torch.Tensor) else gen,
        skip_special_tokens=DECODE_SKIP_SPECIAL,
        clean_up_tokenization_spaces=False
    )
    raw_text = out[0] if out else ""

    # 诊断信息
    info = {"terminated": False, "truncated": False, "used_max_new_tokens": int(max_new_tokens), "gen_len": 0}
    try:
        if isinstance(new_ids, torch.Tensor):
            seq = new_ids[0].tolist()
            info["gen_len"] = len(seq)
            if terminators:
                info["terminated"] = any(t in seq for t in terminators)
            if not info["terminated"] and len(seq) >= max_new_tokens:
                info["truncated"] = True
    except Exception:
        pass

    return raw_text, info

@torch.inference_mode()
def st_translate(processor, model, audio_np: np.ndarray, sr: int, max_new_tokens: int, sparsem_enabled: bool = False, direction: str = None) -> Tuple[str, Dict]:
    """
    返回：
      text: 清洗后的翻译结果
      info: 诊断信息（是否截断、是否命中 terminator 等）
    """
    # Reset SparseMM state before each sample ONLY if SparseMM is actually enabled
    if sparsem_enabled:
        reset_sparsemm_state(model)
    
    if direction is None:
        direction = COVOST2_DIRECTION
    
    # 两阶段流程：先ASR，再翻译（适用于zh_en和en_zh）
    # 第一阶段：ASR（语音识别）
    asr_messages = _build_asr_messages(audio_np, direction)
    # ASR阶段通常需要更多tokens
    asr_max_tokens = max(max_new_tokens, 512)  # 至少512 tokens用于ASR
    source_text, asr_info = _generate_text(
        processor, model, asr_messages, asr_max_tokens, 
        sparsem_enabled, audio_np, sr
    )
    # 清洗ASR输出
    source_text = _clean_asr_output(source_text, direction)
    
    if not source_text:
        # ASR失败，返回空结果
        return "", {"asr_failed": True, "asr_info": asr_info}
    
    # 第二阶段：文本翻译
    translation_messages = _build_translation_messages(source_text, direction)
    # 翻译阶段通常需要更少tokens
    translation_max_tokens = max_new_tokens
    target_text, translation_info = _generate_text(
        processor, model, translation_messages, translation_max_tokens,
        sparsem_enabled, audio_np=None, sr=None
    )
    
    # 清洗翻译输出
    text = _clean_st_output(target_text, direction)
    
    # 合并诊断信息
    info = {
        "asr_stage": asr_info,
        "translation_stage": translation_info,
        "source_text": source_text,  # 第一阶段识别的文本（中文或英文）
        "terminated": translation_info.get("terminated", False),
        "truncated": translation_info.get("truncated", False),
        "used_max_new_tokens": translation_info.get("used_max_new_tokens", max_new_tokens),
        "gen_len": translation_info.get("gen_len", 0)
    }
    
    return text, info

# ------------------------------
# 分片翻译（可配窗口/重叠）+ 软对齐拼接
# ------------------------------

def st_translate_with_overlap(processor, model, audio_np: np.ndarray, sr: int, sparsem_enabled: bool = False, direction: str = None) -> Tuple[str, Dict]:
    if direction is None:
        direction = COVOST2_DIRECTION
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
        txt, info = st_translate(processor, model, chunk, sr, dyn_max, sparsem_enabled, direction=direction)
        info.update({"chunk_index": ci, "start_sec": a / sr, "end_sec": b / sr, "duration_sec": dur})
        diag["chunks"].append(info)
        if txt:
            assembled = _merge_with_overlap_soft(assembled, txt, max_overlap_words=20)
    return re.sub(r"\s+", " ", assembled).strip(), diag

# ------------------------------
# 主流程
# ------------------------------

def main():
    print("开始 CoVoST2 × Gemma-3n 本地语音翻译评测（官方对齐增强版 / E4B 优化）……")
    print(f"Transformers 版本: {TRANSFORMERS_VERSION}")
    print(f"使用设备: {device}")
    
    # Determine dataset path
    if COVOST2_DATASET_PATH:
        dataset_path = COVOST2_DATASET_PATH
    else:
        # Auto-detect based on direction
        base_datasets_dir = "/mnt/users/wangyuxuan-20250915/Audio/Datasets"
        if COVOST2_DIRECTION == "en_zh":
            dataset_path = os.path.join(base_datasets_dir, "covost2_en_zh_test")
        else:  # zh_en
            dataset_path = os.path.join(base_datasets_dir, "covost2_zh_en_test")
    
    print(f"数据集路径: {dataset_path}  |  方向: {COVOST2_DIRECTION}")
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

    print("\n正在加载数据集...")
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集路径不存在: {dataset_path}")
        print(f"请确保数据集已下载到: {dataset_path}")
        return
    
    dataset = load_covost2_dataset(dataset_path, COVOST2_DIRECTION)
    if not dataset:
        print("❌ 没有找到有效的样本数据，退出……")
        return

    if SAMPLE_LIMIT > 0 and len(dataset) > SAMPLE_LIMIT:
        # Use select() for HuggingFace datasets instead of slicing
        if hasattr(dataset, 'select'):
            dataset = dataset.select(range(SAMPLE_LIMIT))
        else:
            dataset = dataset[:SAMPLE_LIMIT]
        print(f"⚠️  应用样本限制，将处理 {len(dataset)} 个样本")
    
    print(f"✓ 数据集加载完成: {len(dataset)} 个样本")

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
            "dataset_path": dataset_path,
            "direction": COVOST2_DIRECTION,
            "gen_constraints": {
                "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
                "repetition_penalty": REPETITION_PENALTY,
                "length_penalty": LENGTH_PENALTY
            }
        },
        "samples": [],
        "summary": {
            "total_samples": 0,
            "total_cer": 0.0,
            "avg_cer": 0.0,
            "speaker_count": 0,  # CoVoST2 doesn't have speaker information
            "timing": {"avg_response_time": 0.0, "total_response_time": 0.0},
            "macro_set_f1": 0.0,
            "truncated_chunks": 0,
            "total_chunks": 0,
            "avg_bleu": 0.0,
            "avg_chrf": 0.0,
            "overall_bleu": 0.0,
            "overall_chrf": 0.0
        }
    }

    print("\n正在初始化模型...")
    processor, model, sparsem_enabled = init_model()
    print(f"✓ 模型初始化完成 | SparseMM: {'启用' if sparsem_enabled else '禁用'}")

    # Initialize timing statistics
    timing_stats = GlobalTimingStats()
    print(f"✓ 统计模块初始化完成")

    is_screen_env = (not sys.stdout.isatty()) or ('TERM' in os.environ and os.environ['TERM'] == 'screen')
    if is_screen_env:
        print("检测到 screen 或非交互式环境，使用简化进度显示")

    tqdm_kwargs = {"ascii": True, "dynamic_ncols": True, "file": sys.stdout}

    set_f1_list = []

    print(f"\n开始处理 {len(dataset)} 个样本...")
    print("=" * 80)
    
    with tqdm(total=len(dataset), desc=f"处理 CoVoST2-{COVOST2_DIRECTION.upper()} 样本", position=0, leave=True, **tqdm_kwargs) as pbar:
        for i, doc in enumerate(dataset):
            # Check if doc is a dictionary
            if not isinstance(doc, dict):
                print(f"Warning: Sample {i} is not a dictionary (type: {type(doc)}), skipping...")
                if isinstance(doc, str):
                    print(f"  Content (first 100 chars): {doc[:100]}")
                continue
            
            # Get reference translation
            reference = doc.get("answer", "")
            if not reference:
                reference = doc.get("reference", "")
            
            audio_data_result = covost2_doc_to_audio(doc)
            if audio_data_result is None:
                continue
            
            audio_np, sr = audio_data_result
            duration = len(audio_np) / float(sr) if audio_np is not None else 0.0

            pred_text = ""
            cer = 100.0  # CER is percentage (0-100)
            resp_time = 0.0
            bleu_score = 0.0
            chrf_score = 0.0

            # 打印当前样本信息
            sample_id = doc.get("id", f"sample_{i}")
            print(f"\n[{i+1}/{len(dataset)}] 处理样本: {sample_id}")
            print(f"  时长: {duration:.2f}s")
            if len(reference) > 100:
                print(f"  参考文本: {reference[:100]}...")
            else:
                print(f"  参考文本: {reference}")

            try:
                start = time.time()
                if ENABLE_CHUNKING:
                    pred_text, diag = st_translate_with_overlap(processor, model, audio_np, sr, sparsem_enabled, direction=COVOST2_DIRECTION)
                    # 统计截断
                    results["summary"]["total_chunks"] += len(diag.get("chunks", []))
                    results["summary"]["truncated_chunks"] += sum(1 for c in diag.get("chunks", []) if c.get("truncated"))
                else:
                    dur = len(audio_np) / float(sr) if audio_np is not None else 0.0
                    dyn_max = MAX_NEW_TOKENS_ENV if MAX_NEW_TOKENS_ENV > 0 else estimate_tokens(dur)
                    pred_text, diag_single = st_translate(processor, model, audio_np, sr, dyn_max, sparsem_enabled, direction=COVOST2_DIRECTION)
                    results["summary"]["total_chunks"] += 1
                    results["summary"]["truncated_chunks"] += 1 if diag_single.get("truncated") else 0
                resp_time = time.time() - start

                cer = calculate_cer(reference, pred_text)
                set_f1 = set_f1_per_sample(reference, pred_text)
                set_f1_list.append(set_f1)
                
                # Calculate BLEU and chrF for this sample
                bleu_score = calculate_bleu_score([pred_text], [reference], COVOST2_DIRECTION)
                chrf_score = calculate_chrf_score([pred_text], [reference])

                # 打印翻译结果
                print(f"  翻译结果: {pred_text[:200]}{'...' if len(pred_text) > 200 else ''}")
                print(f"  处理时间: {resp_time:.3f}s | CER: {cer:.2f}% | BLEU: {bleu_score:.2f} | chrF: {chrf_score:.2f}")
                if ENABLE_CHUNKING and 'chunks' in diag:
                    num_chunks = len(diag.get("chunks", []))
                    truncated = sum(1 for c in diag.get("chunks", []) if c.get("truncated"))
                    print(f"  分片数: {num_chunks} | 截断分片: {truncated}")

                results["summary"]["total_samples"] += 1
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
                print(f"  ❌ 处理错误: {e}")
                traceback.print_exc()
                pred_text = "ERROR"
                cer = 100.0  # CER is percentage (0-100)
                resp_time = 0.0
                bleu_score = 0.0
                chrf_score = 0.0
                set_f1_list.append(0.0)

            results["samples"].append({
                "id": sample_id,
                "duration": duration,
                "ground_truth": reference,
                "prediction": pred_text,
                "cer": cer,
                "bleu": bleu_score,
                "chrf": chrf_score,
                "response_time": resp_time
            })

            # 进度显示
            cnt = i + 1
            # Calculate from actual samples to avoid double counting
            if cnt > 0:
                cer_scores = [r["cer"] for r in results["samples"] if "cer" in r]
                bleu_scores = [r["bleu"] for r in results["samples"] if "bleu" in r]
                chrf_scores = [r["chrf"] for r in results["samples"] if "chrf" in r]
                if len(cer_scores) > 0:
                    cur_avg_cer = np.mean(cer_scores)
                else:
                    cur_avg_cer = 100.0
                if len(bleu_scores) > 0:
                    cur_avg_bleu = np.mean(bleu_scores)
                else:
                    cur_avg_bleu = 0.0
                if len(chrf_scores) > 0:
                    cur_avg_chrf = np.mean(chrf_scores)
                else:
                    cur_avg_chrf = 0.0
            else:
                cur_avg_cer = 100.0
                cur_avg_bleu = 0.0
                cur_avg_chrf = 0.0
            pbar.set_postfix_str(f"CER:{cur_avg_cer:.2f}% BLEU:{cur_avg_bleu:.2f} chrF:{cur_avg_chrf:.2f}")
            
            # 每10个样本或每个样本输出一次统计信息（根据环境调整）
            print_interval = 10 if not is_screen_env else 1
            if cnt % print_interval == 0 or cnt == len(dataset):
                print(f"\n  📊 当前统计 [{cnt}/{len(dataset)}] ({cnt/len(dataset)*100:.1f}%):")
                print(f"     平均CER: {cur_avg_cer:.3f}%")
                print(f"     平均BLEU: {cur_avg_bleu:.2f}")
                print(f"     平均chrF: {cur_avg_chrf:.2f}")
                if cnt > 0:
                    avg_time = results["summary"]["timing"]["total_response_time"] / cnt
                    print(f"     平均处理时间: {avg_time:.3f}s")
                print("-" * 80)
            
            # Real-time save: Update JSON file after each sample (or every N samples for performance)
            save_interval = 1  # Save after every sample for real-time monitoring
            if cnt % save_interval == 0 or cnt == len(dataset):
                try:
                    save_results_realtime(output_file, results, timing_stats, len(dataset))
                    # Also save timing stats periodically
                    if cnt % save_interval == 0 or cnt == len(dataset):
                        timing_stats.export_to_json(timing_output_file)
                    if cnt % 10 == 0 or cnt == len(dataset):
                        print(f"  💾 结果已保存到: {output_file}")
                except Exception as e:
                    print(f"  ⚠️  保存结果时出错: {e}")
            
            pbar.update()

    total = results["summary"]["total_samples"]
    if total > 0:
        # Calculate statistics from actual samples (not from accumulated totals)
        # This ensures accuracy is calculated correctly without double counting
        cer_scores = [r["cer"] for r in results["samples"] if "cer" in r]
        bleu_scores = [r["bleu"] for r in results["samples"] if "bleu" in r]
        chrf_scores = [r["chrf"] for r in results["samples"] if "chrf" in r]
        
        if len(cer_scores) > 0:
            results["summary"]["avg_cer"] = float(np.mean(cer_scores))
        else:
            results["summary"]["avg_cer"] = 1.0
        
        if len(bleu_scores) > 0:
            results["summary"]["avg_bleu"] = float(np.mean(bleu_scores))
        else:
            results["summary"]["avg_bleu"] = 0.0
        
        if len(chrf_scores) > 0:
            results["summary"]["avg_chrf"] = float(np.mean(chrf_scores))
        else:
            results["summary"]["avg_chrf"] = 0.0
        
        # Calculate overall BLEU and chrF using all predictions and references
        predictions = [r.get("prediction", "") for r in results["samples"]]
        references = [r.get("ground_truth", "") for r in results["samples"]]
        results["summary"]["overall_bleu"] = float(calculate_bleu_score(predictions, references, COVOST2_DIRECTION))
        results["summary"]["overall_chrf"] = float(calculate_chrf_score(predictions, references))
        
        results["summary"]["timing"]["avg_response_time"] = (
            results["summary"]["timing"]["total_response_time"] / total
        )
        results["summary"]["macro_set_f1"] = float(np.mean(set_f1_list))

    # Note: CoVoST2 doesn't have speaker_id, so we skip speaker-level statistics

    # Final save: Update status to "completed" and save final results
    results["status"] = "completed"
    results["summary"]["progress"] = {
        "processed": total,
        "total": len(dataset),
        "percentage": 100.0
    }
    
    # Convert numpy types before saving
    print("\n正在保存最终结果...")
    results = convert_numpy_types(results)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.flush()  # Explicitly flush to ensure data is written
    print(f"✓ 最终结果已保存到: {output_file}")

    # Export final timing statistics to separate JSON file
    timing_stats.export_to_json(timing_output_file)
    print(f"✓ 时间统计已保存到: {timing_output_file}")

    trunc_info = f"{results['summary'].get('truncated_chunks', 0)}/{results['summary'].get('total_chunks', 0)} chunks truncated"
    print("\n=== CoVoST2 × Gemma-3n 本地语音翻译评测结果摘要（E4B 优化版） ===")
    print(f"总样本数: {total}")
    print(f"翻译方向: {COVOST2_DIRECTION}")
    print(f"\n=== 翻译质量指标 ===")
    print(f"平均 CER: {results['summary']['avg_cer']:.4f}%")
    print(f"平均 BLEU: {results['summary']['avg_bleu']:.2f}")
    print(f"总体 BLEU: {results['summary']['overall_bleu']:.2f}")
    print(f"平均 chrF: {results['summary']['avg_chrf']:.2f}")
    print(f"总体 chrF: {results['summary']['overall_chrf']:.2f}")
    print(f"宏平均 set-F1: {results['summary']['macro_set_f1']:.4f}")
    print(f"\n=== 性能指标 ===")
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
