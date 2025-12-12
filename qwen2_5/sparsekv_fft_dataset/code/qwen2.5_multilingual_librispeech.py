import os
import sys
import json
import time
import torch
import glob
import soundfile as sf
import numpy as np
import pandas as pd
from transformers import logging
from tqdm import tqdm
from collections import defaultdict
import warnings
import gc
import re
import traceback
import subprocess
import tempfile
from scipy.io import wavfile
from scipy import signal
import librosa
from io import BytesIO
from urllib.request import urlopen
import random
import jiwer
import argparse
from typing import Dict, Any, List, Tuple
import pyarrow.parquet as pq
from pathlib import Path


random.seed(42)

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


# 根据环境变量决定是否使用 sparsekv 版本
USE_SPARSEKV = os.environ.get("USE_SPARSEKV", "true").lower() == "true"

_AUDIO_TOKEN_ID = 151646        
_AUDIO_BOS_TOKEN_ID = 151647      
_AUDIO_EOS_TOKEN_ID = 151648      

ENV_BUDGET_RAW = os.environ.get("BUDGET", "4096")
try:
    ENV_BUDGET = float(ENV_BUDGET_RAW)
    ENV_BUDGET_IS_RATIO = (ENV_BUDGET < 1.0)
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
# Alpha, Beta, Gamma parameters for head score file selection (only used if HEAD_SCORE_PATH is not set)
ENV_HEAD_SCORE_ALPHA = os.environ.get("HEAD_SCORE_ALPHA", None)
ENV_HEAD_SCORE_BETA = os.environ.get("HEAD_SCORE_BETA", None)
ENV_HEAD_SCORE_GAMMA = os.environ.get("HEAD_SCORE_GAMMA", None)
# Aggregation method for multiple scores in JSON (mean, first, second, max, min, sum)
ENV_HEAD_SCORE_AGGREGATION = os.environ.get("HEAD_SCORE_AGGREGATION", "first")

# Audio-aware TopK selection parameters
USE_AUDIO_AWARE_TOPK = os.environ.get("USE_AUDIO_AWARE_TOPK", "false").lower() == "true"
ENV_AUDIO_BOUNDARY_BOOST = float(os.environ.get("AUDIO_BOUNDARY_BOOST", "1.3"))
ENV_AUDIO_CONTINUITY_WEIGHT = float(os.environ.get("AUDIO_CONTINUITY_WEIGHT", "0.1"))
ENV_AUDIO_SMOOTH_WINDOW = int(os.environ.get("AUDIO_SMOOTH_WINDOW", "5"))

# System prompt configuration
# According to official Qwen2.5-Omni README: "No system prompt should be set for any evaluation benchmark"
# Set USE_SYSTEM_PROMPT=true to enable system prompt (for compatibility with UltraEval-Audio)
USE_SYSTEM_PROMPT = os.environ.get("USE_SYSTEM_PROMPT", "false").lower() == "true"

# Language-specific max_new_tokens configuration
# German words are typically longer, so it needs more tokens for accurate transcription
# Default values can be overridden by environment variables: MAX_NEW_TOKENS_<LANGUAGE>
LANGUAGE_MAX_NEW_TOKENS = {
    "german": int(os.environ.get("MAX_NEW_TOKENS_GERMAN", "1500")),  # German words are longer
    "french": int(os.environ.get("MAX_NEW_TOKENS_FRENCH", "1100")),
    "spanish": int(os.environ.get("MAX_NEW_TOKENS_SPANISH", "1100")),
    "english": int(os.environ.get("MAX_NEW_TOKENS_ENGLISH", "1100"))
}
# Global default (can be overridden by language-specific config)
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1100"))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Set up path for imports - must be done before importing model modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Get parent directory (sparsekv_fft_dataset)
os.chdir(current_dir)
sys.path.insert(0, parent_dir)  # Add parent directory to path for model imports

# Import model modules after setting up path
if USE_SPARSEKV:
    # Import from model directory
    from model.modeling_qwen2_5_omni_sparsekv import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniFlashAttention2SparseMM
    print("Using sparsekv version of the model")
else:
    # Try importing from transformers first, then fallback to local file
    try:
        from transformers import Qwen2_5OmniForConditionalGeneration
        print("Using standard version of the model from transformers (sparsekv disabled)")
    except ImportError:
        # Fallback to local modeling file
        modeling_path = "/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Modeling"
        if modeling_path not in sys.path:
            sys.path.insert(0, modeling_path)
        try:
            from modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
            print("Using standard version of the model from local file (sparsekv disabled)")
        except ImportError as e:
            print(f"Error importing modeling_qwen2_5_omni from {modeling_path}: {e}")
            print(f"Current sys.path: {sys.path[:5]}")
            raise

from model.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
# Import SparseMM utilities from model directory
from model.sparsemm_utils import (
    SparseMM,
    DynamicCacheSplitHeadFlatten,
    init_sparsemm
)
from transformers.cache_utils import DynamicCache

# Import audio head utilities for SparseMM
from model.audio_head_utils import (
    load_audio_head_scores,
    process_audio_head_scores,
    get_audio_head_score_path
)

# Import MultilingualLibriSpeechLoader
multilingual_librispeech_path = "/mnt/users/wangyuxuan-20250915/Audio/Datasets/multilingual_librispeech"
if multilingual_librispeech_path not in sys.path:
    sys.path.insert(0, multilingual_librispeech_path)
try:
    from example_usage import MultilingualLibriSpeechLoader
except ImportError:
    # Fallback: define the loader class directly if import fails
    class MultilingualLibriSpeechLoader:
        """Multilingual LibriSpeech dataset loader"""
        def __init__(self, base_dir: str = "/mnt/users/wangyuxuan-20250915/Audio/Datasets/multilingual_librispeech", languages: List[str] = None):
            self.base_dir = Path(base_dir)
            # 如果未指定语言，默认加载所有语言
            if languages is None:
                self.languages = ['spanish', 'french', 'german']
            else:
                self.languages = languages
            self.parquet_files = {}
            # 只检查并打印指定语言的 parquet 文件路径
            for lang in self.languages:
                lang_dir = self.base_dir / lang
                parquet_file = lang_dir / "test-00000-of-00001.parquet"
                if parquet_file.exists():
                    self.parquet_files[lang] = str(parquet_file)
                    print(f"✓ 找到 {lang} 数据集: {parquet_file}")
                else:
                    print(f"⚠ 未找到 {lang} 数据集: {parquet_file}")
        
        def load_audio_from_parquet(self, language: str, index: int = 0) -> Tuple[np.ndarray, int, Dict]:
            if language not in self.parquet_files:
                raise ValueError(f"Language {language} not found")
            parquet_path = self.parquet_files[language]
            table = pq.read_table(parquet_path)
            df = table.to_pandas()
            if index >= len(df):
                raise ValueError(f"Index {index} out of range")
            row = df.iloc[index]
            audio_bytes = row['audio']['bytes']
            audio_io = BytesIO(audio_bytes)
            audio_array, sample_rate = sf.read(audio_io)
            metadata = {
                'id': row['id'],
                'transcript': row['transcript'],
                'speaker_id': row['speaker_id'],
                'chapter_id': row['chapter_id'],
                'audio_duration': row['audio_duration'],
                'language': language
            }
            return audio_array, sample_rate, metadata
        
        def get_dataset_info(self, language: str) -> Dict:
            if language not in self.parquet_files:
                raise ValueError(f"Language {language} not found")
            parquet_path = self.parquet_files[language]
            table = pq.read_table(parquet_path)
            df = table.to_pandas()
            info = {
                'language': language,
                'total_samples': len(df),
                'unique_speakers': df['speaker_id'].nunique() if 'speaker_id' in df.columns else None,
                'total_duration': df['audio_duration'].sum() if 'audio_duration' in df.columns else None,
            }
            return info

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  
        reserved = torch.cuda.memory_reserved() / 1024**3    
        return allocated, reserved
    return 0, 0

def reset_sparsemm_state(model):
    """
    Reset SparseMM KVCache state across all layers
    This prevents state accumulation between samples that can cause accuracy degradation
    """
    base_model = model.thinker.model 
    if hasattr(base_model, 'layers'):
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
                if hasattr(kv_cluster, 'layer_qlens'):
                    kv_cluster.layer_qlens = None
                if hasattr(kv_cluster, 'qlen_sum'):
                    kv_cluster.qlen_sum = 0
                if hasattr(kv_cluster, 'cu_qlen'):
                    kv_cluster.cu_qlen = None
                if hasattr(kv_cluster, 'cu_head_offset'):
                    kv_cluster.cu_head_offset = None
        return True
    return False

def clear_model_cache(model):
    """
    Clear all past_key_values cache from the model to ensure fresh state for each sample
    This is critical to prevent cache accumulation between samples
    """
    # Clear any cached past_key_values in the model
    if hasattr(model, '_past_key_values') and model._past_key_values is not None:
        model._past_key_values = None
    if hasattr(model, 'past_key_values') and model.past_key_values is not None:
        model.past_key_values = None
    
    # Clear cache in thinker model if exists
    if hasattr(model, 'thinker'):
        if hasattr(model.thinker, '_past_key_values') and model.thinker._past_key_values is not None:
            model.thinker._past_key_values = None
        if hasattr(model.thinker, 'past_key_values') and model.thinker.past_key_values is not None:
            model.thinker.past_key_values = None
        if hasattr(model.thinker, 'model'):
            base_model = model.thinker.model
            if hasattr(base_model, '_past_key_values') and base_model._past_key_values is not None:
                base_model._past_key_values = None
            if hasattr(base_model, 'past_key_values') and base_model.past_key_values is not None:
                base_model.past_key_values = None
    
    # Clear DynamicCacheSplitHeadFlatten cache if it exists in any layer
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
        base_model = model.thinker.model
        if hasattr(base_model, 'layers'):
            if len(base_model.layers) == 0:
                return
            first_layer = base_model.layers[0]
            if hasattr(first_layer, 'self_attn'):
                first_attn = first_layer.self_attn
                has_cache = (hasattr(first_attn, '_cache') and first_attn._cache is not None) or \
                           (hasattr(first_attn, 'cache') and first_attn.cache is not None)
                
                if has_cache:
                    for layer in base_model.layers:
                        if hasattr(layer, 'self_attn'):
                            attn = layer.self_attn
                            if hasattr(attn, '_cache') and attn._cache is not None:
                                if hasattr(attn._cache, 'key_cache'):
                                    attn._cache.key_cache.clear()
                                if hasattr(attn._cache, 'value_cache'):
                                    attn._cache.value_cache.clear()
                                attn._cache = None
                            if hasattr(attn, 'cache') and attn.cache is not None:
                                if hasattr(attn.cache, 'key_cache'):
                                    attn.cache.key_cache.clear()
                                if hasattr(attn.cache, 'value_cache'):
                                    attn.cache.value_cache.clear()
                                attn.cache = None

def update_sparsemm_budget(model, prefill_token_count):
    """
    Update SparseMM budget for all layers based on prefill token count
    If ENV_BUDGET_IS_RATIO is True, calculate budget as ratio of prefill tokens
    Otherwise, use the absolute budget value
    
    Args:
        model: The model with SparseMM enabled
        prefill_token_count: Number of tokens in prefill phase
    
    Returns:
        The actual budget value used (integer)
    """
    if not ENV_BUDGET_IS_RATIO:
        # Use absolute budget value, no update needed
        return int(ENV_BUDGET)
    
    # Calculate budget as ratio of prefill tokens
    budget_ratio = ENV_BUDGET
    actual_budget = int(prefill_token_count * budget_ratio)
    
    # Ensure budget is at least window_size to avoid errors
    min_budget = ENV_WINDOW_SIZE + 1
    if actual_budget < min_budget:
        actual_budget = min_budget
        print(f"[SparseMM] Warning: Calculated budget {int(prefill_token_count * budget_ratio)} is too small, using minimum {min_budget}")
    
    # Update budget in all layers
    base_model = model.thinker.model
    if hasattr(base_model, 'layers'):
        for layer in base_model.layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'kv_cluster'):
                kv_cluster = layer.self_attn.kv_cluster
                # Use the new update_budget method to properly recalculate head_adaptive_capacity
                if hasattr(kv_cluster, 'update_budget'):
                    kv_cluster.update_budget(actual_budget)
                elif hasattr(kv_cluster, 'base_capacity'):
                    # Fallback for older versions without update_budget method
                    kv_cluster.base_capacity = actual_budget - kv_cluster.window_size
                    if kv_cluster.base_capacity <= 0:
                        kv_cluster.base_capacity = 1
        # Also update model config for consistency
        if hasattr(base_model, 'config'):
            base_model.config.max_capacity_prompt = actual_budget
    
    print(f"[SparseMM] Updated budget: {prefill_token_count} tokens * {budget_ratio:.2%} = {actual_budget} (base_capacity={actual_budget - ENV_WINDOW_SIZE})")
    return actual_budget

def load_audio_head_scores_for_sparsemm(
    score_file: str,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_groups: int = 1,
    aggregation: str = "mean"
) -> Dict[str, Any]:
    """
    Load and process audio head scores for SparseMM
    Returns a dictionary compatible with SparseMM's head_score format
    
    Args:
        score_file: Path to the JSON file containing head scores
        num_hidden_layers: Number of hidden layers
        num_attention_heads: Number of attention heads per layer
        num_key_value_groups: Number of key-value groups (for GQA)
        aggregation: Method to aggregate multiple scores in JSON
                    Options: "mean" (default), "first", "second", "max", "min", "sum"
    """
    head_scores = load_audio_head_scores(score_file, use_mean=True)

    # Handle compact uniform specification: {"__uniform__": value}
    if isinstance(head_scores, dict) and "__uniform__" in head_scores:
        try:
            uniform_val = float(head_scores["__uniform__"])
        except Exception:
            uniform_val = 1.0
        head_score_list = [uniform_val] * (num_hidden_layers * num_attention_heads)
    else:
        # Convert to list format expected by SparseMM
        # SparseMM expects: [score for layer0_head0, layer0_head1, ..., layerN_headM]
        head_score_list = []
        for layer_idx in range(num_hidden_layers):
            for head_idx in range(num_attention_heads):
                key = f"{layer_idx}-{head_idx}"
                if key in head_scores:
                    score_value = head_scores[key]
                    if isinstance(score_value, list):
                        # Apply aggregation method
                        if aggregation == "mean":
                            score = np.mean(score_value)
                        elif aggregation == "first":
                            score = float(score_value[0]) if len(score_value) > 0 else 0.0
                        elif aggregation == "second":
                            score = float(score_value[1]) if len(score_value) > 1 else float(score_value[0]) if len(score_value) > 0 else 0.0
                        elif aggregation == "max":
                            score = np.max(score_value)
                        elif aggregation == "min":
                            score = np.min(score_value)
                        elif aggregation == "sum":
                            score = np.sum(score_value)
                        else:
                            # Default to mean if unknown method
                            print(f"[SparseMM] Warning: Unknown aggregation method '{aggregation}', using mean")
                            score = np.mean(score_value)
                    else:
                        score = float(score_value)
                else:
                    score = 0.0
                head_score_list.append(score)
    
    # Normalize
    head_score_array = np.array(head_score_list)
    if head_score_array.sum() > 0:
        head_score_array = head_score_array / head_score_array.sum()
    else:
        head_score_array = np.ones_like(head_score_array) / len(head_score_array)
    
    return {
        'scores': head_score_array.tolist(),
        'tensor': torch.tensor(head_score_array, dtype=torch.float32)
    }

class GlobalTimingStats:
    """Simplified global timing statistics class"""
    
    def __init__(self):
        self.timing_records = []
        self.first_sample_skipped = False
        
    def add_sample(self, wall_time, prefill_time, total_gpu_time, decode_time=None, decode_token_count=None):
        """Add sample timing record, skip the first sample"""
        
        if not self.first_sample_skipped:
            self.first_sample_skipped = True
            return
            
        record = {
            "wall_time": wall_time,
            "total_gpu_time": total_gpu_time
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
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file

class LazyAudioLoader:
    """Lazy loader for audio data from parquet files - loads audio on demand and caches"""
    def __init__(self):
        self._df_cache = {}  # Cache DataFrames (audio column only) by parquet path
    
    def load_audio(self, parquet_path: str, index: int) -> Tuple[np.ndarray, int]:
        """Load audio data from parquet file on demand"""
        # Cache DataFrame (audio column only) to avoid reloading
        # This is efficient because:
        # 1. First access loads the audio column once and caches it
        # 2. Subsequent accesses use cached DataFrame (very fast)
        # 3. Only audio column is loaded, not all metadata
        if parquet_path not in self._df_cache:
            # Read only audio column for efficiency
            table = pq.read_table(parquet_path, columns=['audio'])
            self._df_cache[parquet_path] = table.to_pandas()
        
        df = self._df_cache[parquet_path]
        row = df.iloc[index]
        
        # Extract and decode audio
        audio_bytes = row['audio']['bytes']
        audio_io = BytesIO(audio_bytes)
        audio_array, sample_rate = sf.read(audio_io)
        
        return audio_array, sample_rate

# Global lazy loader instance
_lazy_loader = LazyAudioLoader()

def load_multilingual_librispeech_dataset(base_dir: str, languages: List[str] = None):
    """
    Load Multilingual LibriSpeech dataset from parquet files (lazy loading)
    Only loads metadata, audio is loaded on-demand during evaluation
    
    Args:
        base_dir: Base directory containing multilingual_librispeech dataset
        languages: List of languages to load (default: ['spanish', 'french', 'german'])
    
    Returns:
        List of dataset samples with language information (audio loaded lazily)
    """
    if languages is None:
        languages = ['spanish', 'french', 'german']
    
    loader = MultilingualLibriSpeechLoader(base_dir=base_dir, languages=languages)
    dataset = []
    
    for lang in languages:
        if lang not in loader.parquet_files:
            print(f"Warning: Language {lang} not found, skipping")
            continue
        
        # Get dataset info (this reads parquet once to get count)
        info = loader.get_dataset_info(lang)
        total_samples = info['total_samples']
        
        print(f"Loading {lang} dataset metadata: {total_samples} samples")
        
        # Read parquet file once to extract metadata only
        parquet_path = loader.parquet_files[lang]
        print(f"Reading parquet file metadata: {parquet_path}")
        table = pq.read_table(parquet_path, columns=['id', 'transcript', 'speaker_id', 'chapter_id', 
                                                      'file', 'original_path', 'begin_time', 'end_time', 
                                                      'audio_duration'])
        df = table.to_pandas()
        print(f"Metadata loaded, preparing {len(df)} samples (audio will be loaded on-demand)...")
        
        # Store only metadata, audio will be loaded lazily
        for idx in range(total_samples):
            try:
                row = df.iloc[idx]
                
                # Extract metadata only (no audio loading)
                metadata = {
                    'id': row['id'],
                    'transcript': row.get('transcript', ''),
                    'speaker_id': row.get('speaker_id', 'unknown'),
                    'chapter_id': row.get('chapter_id', ''),
                    'file': row.get('file', ''),
                    'original_path': row.get('original_path', ''),
                    'begin_time': row.get('begin_time', 0.0),
                    'end_time': row.get('end_time', 0.0),
                    'audio_duration': row.get('audio_duration', 0.0),
                    'language': lang
                }
                
                dataset.append({
                    "id": metadata['id'],
                    "language": lang,
                    "speaker_id": metadata.get('speaker_id', 'unknown'),
                    "chapter_id": metadata.get('chapter_id', ''),
                    "transcript": metadata.get('transcript', ''),
                    "audio_array": None,  # Will be loaded lazily
                    "sample_rate": None,  # Will be loaded lazily
                    "duration": metadata.get('audio_duration', 0.0),
                    "metadata": metadata,
                    # Store reference for lazy loading
                    "_parquet_path": parquet_path,
                    "_parquet_index": idx
                })
            except Exception as e:
                print(f"Error loading {lang} sample {idx} metadata: {e}")
                continue
    
    print(f"Successfully prepared {len(dataset)} samples from {len(languages)} languages (lazy loading enabled)")
    return dataset

def asr_doc_to_text(doc, kwargs=None):
    """
    Generate prompts for multilingual ASR tasks
    Following official Qwen2.5-Omni prompt format:
    - For other languages: "Transcribe the <source_language> audio into text."
    - UltraEval-Audio format includes "without any punctuation marks"
    """
    if kwargs is None:
        kwargs = {}
    
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    
    language = doc.get("language", "english").lower()
    
    # Language-specific prompts following official format
    # Format: "Transcribe the <language> audio into text without any punctuation marks."
    # Reference: UltraEval-Audio/registry/prompt/qwen-omni.yaml
    language_prompts = {
        "spanish": "Transcribe the Spanish audio into text without any punctuation marks.",
        "french": "Transcribe the French audio into text without any punctuation marks.",
        "german": "Transcribe the German audio into text without any punctuation marks.",
        "english": "Transcribe the English audio into text without any punctuation marks."
    }
    
    instruction = language_prompts.get(language, "Transcribe the audio into text without any punctuation marks.")
    
    prompt_text = f"{pre_prompt}{instruction}{post_prompt}"
    
    return prompt_text

def clean_response(response):
    """Clean ASR response"""
    if not response or response.strip() == "":
        return ""
    
    # Try to extract JSON content if present
    try:
        import json
        # Look for JSON content in the response
        json_match = re.search(r'\{[^{}]*"content"[^{}]*\}', response)
        if json_match:
            json_str = json_match.group(0)
            json_obj = json.loads(json_str)
            if "content" in json_obj:
                response = json_obj["content"]
    except:
        pass
    
    # Remove common markers
    for marker in ["spoken content:", "content:", "transcription:", "transcript:", 
                   "内容:", "转录:", "文本:", "识别结果:"]:
        if marker.lower() in response.lower():
            parts = re.split(re.escape(marker), response, flags=re.IGNORECASE)
            if len(parts) > 1:
                response = parts[1].strip()
                break
    
    # Remove placeholder text
    response = re.sub(r'<transcribed text here>', '', response)
    response = re.sub(r'<sep>.*?($|<|$)', '', response)
    # Remove leading markers
    response = re.sub(r'(?i)^(spoken\s+(?:text|content)|content|transcript|transcription|内容|转录|文本|识别结果):\s*', '', response.strip())
    
    return response.strip()

def standardize_text(text, language="english"):
    """
    Normalize text for fair comparison and WER calculation
    Language-specific handling for special characters (e.g., German ä, ö, ü, ß)
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Language-specific normalization
    if language.lower() == "german":
        # German: Keep umlauts (ä, ö, ü) and ß as they are meaningful
        # But normalize them for comparison: ä->ae, ö->oe, ü->ue, ß->ss
        # However, for WER calculation, we might want to keep them as-is
        # Let's keep them for now, but remove other punctuation
        pass
    elif language.lower() == "french":
        # French: Keep accented characters (é, è, ê, etc.) as they are meaningful
        pass
    elif language.lower() == "spanish":
        # Spanish: Keep accented characters (á, é, í, ó, ú, ñ) as they are meaningful
        pass
    
    # Remove punctuation (but preserve language-specific characters)
    text = re.sub(r'[.!?,;:"()\[\]{}]', ' ', text)
    text = re.sub(r'[\-\']', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_wer(reference, hypothesis, language="english"):
    """Calculate Word Error Rate (WER)"""
    import jiwer
    
    # Standardize text with language-specific handling
    ref_standardized = standardize_text(reference, language)
    hyp_standardized = standardize_text(hypothesis, language)
    
    # Calculate WER using jiwer
    wer = jiwer.wer(ref_standardized, hyp_standardized)
    wer_percent = wer * 100
    
    # WER can exceed 100% when there are many insertions
    # WER = (S + D + I) / N, where N is the number of words in reference
    # If insertions are very high, WER can be > 100%
    if wer_percent > 100.0:
        ref_words = ref_standardized.split()
        hyp_words = hyp_standardized.split()
        # Calculate detailed metrics (if available in jiwer version)
        try:
            if hasattr(jiwer, 'compute_measures'):
                measures = jiwer.compute_measures(ref_standardized, hyp_standardized)
                substitutions = measures.get('substitutions', 0)
                deletions = measures.get('deletions', 0)
                insertions = measures.get('insertions', 0)
                hits = measures.get('hits', 0)
                print(f"[WER Analysis] WER={wer_percent:.2f}%, Ref words={len(ref_words)}, Hyp words={len(hyp_words)}")
                print(f"  Substitutions={substitutions}, Deletions={deletions}, Insertions={insertions}, Hits={hits}")
                if insertions > len(ref_words):
                    print(f"  [Issue] Insertions ({insertions}) > Reference words ({len(ref_words)}) - this causes WER > 100%")
            else:
                # Fallback: just print basic info if compute_measures is not available
                print(f"[WER Analysis] WER={wer_percent:.2f}%, Ref words={len(ref_words)}, Hyp words={len(hyp_words)}")
                print(f"  [Note] Detailed metrics unavailable (jiwer.compute_measures not available in this version)")
                if len(hyp_words) > len(ref_words):
                    print(f"  [Issue] Hypothesis has more words ({len(hyp_words)}) than reference ({len(ref_words)}) - this may cause WER > 100%")
        except AttributeError:
            # Fallback if compute_measures doesn't exist
            print(f"[WER Analysis] WER={wer_percent:.2f}%, Ref words={len(ref_words)}, Hyp words={len(hyp_words)}")
            print(f"  [Note] Detailed metrics unavailable (jiwer.compute_measures not available in this version)")
            if len(hyp_words) > len(ref_words):
                print(f"  [Issue] Hypothesis has more words ({len(hyp_words)}) than reference ({len(ref_words)}) - this may cause WER > 100%")

    return wer_percent


def save_results_realtime(output_file, results, total_wer, processed_samples, speaker_stats, 
                          timing_stats, gpu_id, prune_ratio, method_is, 
                          sample_limit, dataset_path, dataset_size, language_stats):
    """
    Save results to JSON file in real-time, updating statistics as we go
    This allows monitoring progress and viewing results before completion
    """
    try:
        # Calculate current statistics
        current_avg_wer = total_wer / processed_samples if processed_samples > 0 else 100.0
        current_avg_accuracy = 100.0 - current_avg_wer
        
        # Calculate speaker-level statistics
        speaker_wer = defaultdict(list)
        for result in results:
            speaker_wer[result["speaker_id"]].append(result["wer"])
        
        speaker_avg_wer = {}
        for speaker_id, wers in speaker_wer.items():
            speaker_avg_wer[speaker_id] = sum(wers) / len(wers)
        
        # Calculate language-level statistics
        language_wer = defaultdict(list)
        for result in results:
            language_wer[result["language"]].append(result["wer"])
        
        language_avg_wer = {}
        for lang, wers in language_wer.items():
            language_avg_wer[lang] = sum(wers) / len(wers)
        
        # Calculate detailed metrics from current results
        references = [r["reference"] for r in results]
        hypotheses = [r["hypothesis"] for r in results]
        
        # Calculate WER statistics
        wer_scores = [r["wer"] for r in results]
        if len(wer_scores) > 0:
            wer_mean = np.mean(wer_scores)
            wer_std = np.std(wer_scores)
            wer_min = np.min(wer_scores)
            wer_max = np.max(wer_scores)
            perfect_predictions = len([wer for wer in wer_scores if wer == 0.0])
            word_accuracy = 100.0 - wer_mean
        else:
            wer_mean = 100.0
            wer_std = 0.0
            wer_min = 100.0
            wer_max = 100.0
            perfect_predictions = 0
            word_accuracy = 0.0
        
        # Build real-time results structure
        realtime_results = {
            "model_name": f"Qwen2.5-Omni-3B",
            "dataset": "Multilingual LibriSpeech",
            "status": "in_progress",
            "progress": {
                "processed": len(results),
                "total": dataset_size,
                "percentage": len(results) / dataset_size * 100 if dataset_size > 0 else 0
            },
            "current_statistics": {
                "wer_mean": float(wer_mean),
                "word_accuracy": float(word_accuracy),
                "wer_std": float(wer_std),
                "wer_min": float(wer_min),
                "wer_max": float(wer_max),
                "perfect_predictions": int(perfect_predictions),
                "valid_samples": len(results),
                "current_avg_wer": float(current_avg_wer),
                "current_avg_accuracy": float(current_avg_accuracy)
            },
            "speaker_count": len(speaker_stats),
            "speaker_wer": speaker_avg_wer,
            "language_count": len(language_stats),
            "language_wer": language_avg_wer,
            "timing_stats": timing_stats.get_summary(),
            "config": {
                "device": f"cuda:{gpu_id}",
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "gpu_id": str(os.environ.get('CUDA_VISIBLE_DEVICES', 'default')),
                "pruning_config": {
                    "prune_ratio": prune_ratio, 
                    "prune_method": method_is
                },
                "sample_limit": sample_limit,
                "data_path": dataset_path
            },
            "detailed_results": results
        }
        
        # Convert numpy types
        realtime_results = convert_numpy_types(realtime_results)
        
        # Write to file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(realtime_results, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Warning: Failed to save real-time results: {e}")
        traceback.print_exc()

def calculate_multilingual_metrics(references, hypotheses):
    """
    Multilingual LibriSpeech ASR metrics calculation using WER (Word Error Rate)
    
    Args:
        references: List of true transcriptions
        hypotheses: List of predicted transcriptions
        
    Returns:
        dict: Dictionary containing WER-related metrics
    """
    
    valid_indices = []
    clean_references = []
    clean_hypotheses = []
    wer_scores = []
    
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        if ref and hyp:  
            valid_indices.append(i)
            clean_references.append(ref)
            clean_hypotheses.append(hyp)
            
            # Calculate WER (language will be inferred from context or use default)
            # Note: For batch processing, we use default "english" - individual samples use language-specific
            wer = calculate_wer(ref, hyp, language="english")
            wer_scores.append(wer)
    
    if len(clean_references) == 0:
        return {
            'wer_mean': 100.0,
            'wer_std': 0.0,
            'wer_min': 100.0,
            'wer_max': 100.0,
            'perfect_predictions': 0,
            'valid_samples': 0,
            'total_samples': len(references),
            'word_accuracy': 0.0
        }
    
    # Calculate statistics
    wer_mean = np.mean(wer_scores)
    wer_std = np.std(wer_scores)
    wer_min = np.min(wer_scores)
    wer_max = np.max(wer_scores)
    
    # Count perfect predictions
    perfect_predictions = len([wer for wer in wer_scores if wer == 0.0])
    
    # Word accuracy = 100 - WER
    word_accuracy = 100.0 - wer_mean
    
    return {
        'wer_mean': float(wer_mean),
        'wer_std': float(wer_std),
        'wer_min': float(wer_min),
        'wer_max': float(wer_max),
        'perfect_predictions': int(perfect_predictions),
        'valid_samples': len(clean_references),
        'total_samples': len(references),
        'word_accuracy': float(word_accuracy),
        'wer_scores': wer_scores
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multilingual LibriSpeech ASR Evaluation with Qwen2.5-Omni')
    parser.add_argument('--output-file', '-o', type=str, default=None,
                        help='Output file path for results JSON (overrides --output-dir)')
    parser.add_argument('--output-dir', '-d', type=str, default=None,
                        help='Output directory for results (default: ./MultilingualLibriSpeech_Results or from RESULTS_DIR env)')
    parser.add_argument('--timing-output-file', '-t', type=str, default=None,
                        help='Output file path for timing statistics JSON')
    parser.add_argument('--languages', '-l', type=str, nargs='+', default=['spanish', 'french', 'german'],
                        help='Languages to evaluate (default: spanish french german)')
    parser.add_argument('--model-path', '-m', type=str, default=None,
                        help='Path to the model directory (default: from MODEL_PATH env or hardcoded path)')
    args = parser.parse_args()
    
    gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_id = gpu_temp[-1] if gpu_temp else "0"
    print(f"Using GPU ID: {gpu_id}")

    # Pruning configuration
    prune_ratio = float(os.environ.get("PRUNE_RATIO", 0))
    prune_method = os.environ.get("PRUNE_METHOD", "base")

    use_random = (prune_method == "random")
    use_frame = (prune_method == "frame")
    if use_random == False and use_frame == False:
        prune_method = "fast_v"
    
    if prune_ratio == 0:
        method_is = "base"
    else:
        method_is = prune_method

    # Sample limit
    sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
    if sample_limit > 0:
        print(f"Sample limit set to: {sample_limit}")

    # Set default dataset path
    default_dataset_path = '/mnt/users/wangyuxuan-20250915/Audio/Datasets/multilingual_librispeech'
    
    # Check if default path exists
    if os.path.exists(default_dataset_path):
        dataset_path = default_dataset_path
        print(f"Using dataset path: {dataset_path}")
    else:
        dataset_path = default_dataset_path
        print(f"Dataset path not found: {dataset_path}, will attempt to load anyway")
    
    # Determine output file paths
    # Priority: --output-file > --output-dir > RESULTS_DIR env > default
    if args.output_file:
        # Direct file path specified
        output_file = os.path.abspath(args.output_file)
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Use directory-based approach
        if args.output_dir:
            result_dir = args.output_dir
        else:
            result_dir = os.environ.get("RESULTS_DIR", './MultilingualLibriSpeech_Results')
        
        dataset_path = os.path.abspath(dataset_path)
        result_dir = os.path.abspath(result_dir)
        os.makedirs(result_dir, exist_ok=True)
        output_file = f'{result_dir}/multilingual_librispeech_results_qwen25.json'
    
    # Determine timing output file path
    if args.timing_output_file:
        timing_output_file = os.path.abspath(args.timing_output_file)
        timing_output_dir = os.path.dirname(timing_output_file)
        os.makedirs(timing_output_dir, exist_ok=True)
    else:
        # Use same directory as output_file
        timing_output_dir = os.path.dirname(output_file)
        timing_output_file = f'{timing_output_dir}/timing_stats_qwen25_{method_is}_{prune_ratio}.json'
    
    print(f"Results will be saved to: {output_file} (updated in real-time)")
    print(f"Timing statistics will be saved to: {timing_output_file}")

    # Initialize timing stats
    timing_stats = GlobalTimingStats()

    print(f"\n=== Multilingual LibriSpeech ASR Evaluation Configuration (Qwen2.5-Omni) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Pruning ratio: {prune_ratio}")
    print(f"Pruning method: {method_is}")
    print(f"Data path: {dataset_path}")
    print(f"Languages: {', '.join(args.languages)}")
    print(f"System prompt: {'ENABLED' if USE_SYSTEM_PROMPT else 'DISABLED (following official README)'}")
    print(f"Language-specific max_new_tokens: {dict(LANGUAGE_MAX_NEW_TOKENS)}")
    if sample_limit > 0:
        print(f"Sample limit: {sample_limit}")
    if USE_SPARSEKV:
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
    print("=" * 40)

    # Load model
    print("Loading Qwen2.5-Omni model...")
    # Get model path from command line argument, environment variable, or use default
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.environ.get("MODEL_PATH", "/mnt/users/wangyuxuan-20250915/Models/Qwen2.5-Omni-3B")
    print(f"Using model path: {model_path}")
    device_map = {"": 0}  
    
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.disable_talker()
    
    # Configure sparse attention
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
        model.thinker.model.config.sparse_attention_config = {'prune_ratio': prune_ratio, 'prune_method': prune_method}
        print(f"Sparse attention config set: prune_ratio={prune_ratio}, prune_method={prune_method}")
    else:
        print("Warning: thinker model config not found, using default parameters")
    
    # Initialize thinker model config
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
        if not hasattr(model.thinker.model.config, 'image_layer_idx'):
            model.thinker.model.config.image_layer_idx = False
        if not hasattr(model.thinker.model.config, 'audio_layer_idx'):
            model.thinker.model.config.audio_layer_idx = None
        if not hasattr(model.thinker.model.config, 'audio_token_num'):
            model.thinker.model.config.audio_token_num = None
        if not hasattr(model.thinker.model.config, 'audio_token_start'):
            model.thinker.model.config.audio_token_start = None
        if not hasattr(model.thinker.model.config, 'audio_prune_ratio'):
            model.thinker.model.config.audio_prune_ratio = 0
        if not hasattr(model.thinker.model.config, 'random'):
            model.thinker.model.config.random = False
        if not hasattr(model.thinker.model.config, 'frame'):
            model.thinker.model.config.frame = False
    print(f"Initializing thinker.model.config pruning configuration parameters")
    
    # Enable SparseMM if USE_SPARSEKV is true
    sparsemm_enabled = False
    if USE_SPARSEKV and hasattr(model, 'enable_sparsemm_attention'):
        # Get model config for head score loading
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            base_model = model.thinker.model
            num_hidden_layers = base_model.config.num_hidden_layers
            num_attention_heads = base_model.config.num_attention_heads
            
            # Load head score file
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
                head_score_type = ENV_HEAD_SCORE_FILE
                head_score_batch = ENV_HEAD_SCORE_BATCH if ENV_HEAD_SCORE_BATCH else None
                
                # Parse alpha, beta, gamma if provided
                head_score_alpha = None
                head_score_beta = None
                head_score_gamma = None
                if ENV_HEAD_SCORE_ALPHA is not None:
                    try:
                        head_score_alpha = float(ENV_HEAD_SCORE_ALPHA)
                    except ValueError:
                        print(f"[SparseMM] Warning: Invalid HEAD_SCORE_ALPHA value '{ENV_HEAD_SCORE_ALPHA}', ignoring")
                if ENV_HEAD_SCORE_BETA is not None:
                    try:
                        head_score_beta = float(ENV_HEAD_SCORE_BETA)
                    except ValueError:
                        print(f"[SparseMM] Warning: Invalid HEAD_SCORE_BETA value '{ENV_HEAD_SCORE_BETA}', ignoring")
                if ENV_HEAD_SCORE_GAMMA is not None:
                    try:
                        head_score_gamma = float(ENV_HEAD_SCORE_GAMMA)
                    except ValueError:
                        print(f"[SparseMM] Warning: Invalid HEAD_SCORE_GAMMA value '{ENV_HEAD_SCORE_GAMMA}', ignoring")
                
                # Use the correct base directory for qwen2.5_omni head scores
                head_score_base_dir = "/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/head_score"
                head_score_path = get_audio_head_score_path(
                    base_dir=head_score_base_dir,
                    score_type=head_score_type,
                    batch=head_score_batch,
                    alpha=head_score_alpha,
                    beta=head_score_beta,
                    gamma=head_score_gamma
                )
                print(f"[SparseMM] Constructed head score path: {head_score_path}")
            
            print(f"[SparseMM] Loading audio head scores from: {head_score_path}")
            print(f"[SparseMM] Using aggregation method: {ENV_HEAD_SCORE_AGGREGATION}")
            head_score_data = load_audio_head_scores_for_sparsemm(
                head_score_path,
                num_hidden_layers,
                num_attention_heads,
                aggregation=ENV_HEAD_SCORE_AGGREGATION
            )
            
            # Configure SparseMM
            if ENV_BUDGET_IS_RATIO:
                init_budget = 4096  # Default initialization budget
                print(f"[SparseMM] Budget is a ratio ({ENV_BUDGET:.2%}), using placeholder {init_budget} for initialization")
            else:
                init_budget = int(ENV_BUDGET)
                print(f"[SparseMM] Budget is absolute value: {init_budget}")
            
            sparsemm_config = {
                'budget': init_budget,
                'ratio': ENV_RATIO,
                'window_size': ENV_WINDOW_SIZE,
                'kernel_size': ENV_KERNEL_SIZE,
                'pooling': ENV_POOLING
            }
            
            print(f"[SparseMM] Enabling SparseMM with config: {sparsemm_config}")
            model.enable_sparsemm_attention(head_score_data, sparsemm_config)
            sparsemm_enabled = True
            print(f"[SparseMM] SparseMM enabled successfully")
        else:
            print("[SparseMM] Warning: Cannot find thinker model, skipping SparseMM initialization")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Load dataset
    print(f"Loading Multilingual LibriSpeech dataset: {dataset_path}")
    dataset = load_multilingual_librispeech_dataset(dataset_path, languages=args.languages)
    
    if not dataset:
        print("Error: Failed to load any data")
        return
    
    # Apply sample limit if specified
    if sample_limit > 0 and len(dataset) > sample_limit:
        dataset = dataset[:sample_limit]
        print(f"Applying sample limit, processing {len(dataset)} samples")

    # Collect statistics
    speaker_stats = defaultdict(int)
    language_stats = defaultdict(int)
    for sample in dataset:
        speaker_id = sample.get("speaker_id", "unknown")
        language = sample.get("language", "unknown")
        speaker_stats[speaker_id] += 1
        language_stats[language] += 1
    
    print(f"Speaker count: {len(speaker_stats)} speakers")
    print(f"Language distribution: {dict(language_stats)}")
    print(f"Sample distribution by speaker: {dict(list(speaker_stats.items())[:5])}...")

    results = []
    total_wer = 0.0
    processed_samples = 0

    # Check if running in screen/non-interactive environment
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
    
    tqdm_kwargs = {
        'ascii': True,      
        'dynamic_ncols': True, 
        'file': sys.stdout    
    }

    print(f"Starting evaluation of {len(dataset)} samples...")
    
    # Print GPU memory usage
    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model loading - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    progress_bar = tqdm(enumerate(dataset), total=len(dataset), desc="Multilingual LibriSpeech ASR evaluation (Qwen2.5)", **tqdm_kwargs)

    for idx, doc in progress_bar:
        try:
            # CRITICAL: Reset SparseMM state before each sample to prevent state accumulation
            if USE_SPARSEKV and sparsemm_enabled:
                # Check if reset is actually needed
                needs_reset = False
                if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
                    base_model = model.thinker.model
                    if hasattr(base_model, 'layers'):
                        for layer in base_model.layers:
                            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'kv_cluster'):
                                kv_cluster = layer.self_attn.kv_cluster
                                if kv_cluster is not None and hasattr(kv_cluster, 'head_lens'):
                                    if kv_cluster.head_lens is not None:
                                        needs_reset = True
                                        break
                
                if needs_reset or idx == 0:
                    reset_sparsemm_state(model)
                clear_model_cache(model)
            
            # Get audio data (lazy loading)
            audio_np = doc.get("audio_array")
            sr = doc.get("sample_rate")
            
            # Lazy load audio if not already loaded
            if audio_np is None:
                parquet_path = doc.get("_parquet_path")
                parquet_index = doc.get("_parquet_index")
                if parquet_path is not None and parquet_index is not None:
                    try:
                        audio_np, sr = _lazy_loader.load_audio(parquet_path, parquet_index)
                        # Cache the loaded audio
                        doc["audio_array"] = audio_np
                        doc["sample_rate"] = sr
                    except Exception as e:
                        print(f"Error loading audio for sample {idx}: {e}")
                        continue
                else:
                    print(f"Warning: No audio data for sample {idx}, skipping")
                    continue
            
            # Ensure sample_rate has a default
            if sr is None:
                sr = 16000
            
            # Get reference transcript
            reference = doc.get("transcript", "")
            speaker_id = doc.get("speaker_id", "unknown")
            language = doc.get("language", "unknown")
            
            # Generate prompt
            prompt_text = asr_doc_to_text(doc)

            # Create messages
            # According to official Qwen2.5-Omni README: "No system prompt should be set for any evaluation benchmark"
            # But UltraEval-Audio uses system prompt, so we provide an option to enable it
            messages = []
            if USE_SYSTEM_PROMPT:
                messages.append({
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a speech recognition model."}
                    ]
                })
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_np},
                    {"type": "text", "text": prompt_text}
                ]
            })
            
            # Process multimodal info
            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
            
            # Apply chat template
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            if isinstance(text, list):
                text = text[0] if len(text) > 0 else ""
            
            # Process inputs
            inputs = processor(
                text=text, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=True
            )
            inputs = inputs.to(model.device).to(model.dtype)
            
            # Detect audio tokens
            audio_token_length = 0
            audio_token_start = 0
            input_token_length = inputs.input_ids.shape[1] if hasattr(inputs, 'input_ids') else 0
            
            audio_detected = False
            
            if hasattr(inputs, 'input_ids'):
                token_ids = inputs.input_ids[0].tolist()
                
                bos_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_BOS_TOKEN_ID]
                eos_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_EOS_TOKEN_ID]
                
                if bos_positions and eos_positions:
                    audio_token_start = bos_positions[0]
                    audio_token_end = eos_positions[0]
                    audio_token_length = audio_token_end - audio_token_start + 1
                    
                    audio_detected = True
                    
                    model.thinker.model.config.image_layer_idx = False  
                    model.thinker.model.config.audio_token_num = audio_token_length
                    model.thinker.model.config.audio_token_start = audio_token_start
                    model.thinker.model.config.audio_prune_ratio = prune_ratio
                    model.thinker.model.config.random = use_random
                    model.thinker.model.config.frame = use_frame 
                    
            if not audio_detected:
                model.thinker.model.config.audio_prune_ratio = 0

            # Update SparseMM budget based on prefill token count if budget is a ratio
            if USE_SPARSEKV and sparsemm_enabled and ENV_BUDGET_IS_RATIO:
                if hasattr(inputs, 'input_ids'):
                    prefill_token_count = inputs.input_ids.shape[1] if len(inputs.input_ids.shape) > 1 else inputs.input_ids.shape[0]
                    update_sparsemm_budget(model, prefill_token_count)
            
            # Compute audio token count
            audio_tokens = 0
            try:
                if hasattr(processor.tokenizer, 'audio_bos_token_id') and hasattr(processor.tokenizer, 'audio_eos_token_id') and 'input_ids' in inputs:
                    input_ids = inputs['input_ids'][0]
                    audio_bos_positions = (input_ids == processor.tokenizer.audio_bos_token_id).nonzero(as_tuple=True)[0]
                    audio_eos_positions = (input_ids == processor.tokenizer.audio_eos_token_id).nonzero(as_tuple=True)[0]

                    if len(audio_bos_positions) > 0 and len(audio_eos_positions) > 0:
                        for bos_pos in audio_bos_positions:
                            eos_candidates = audio_eos_positions[audio_eos_positions > bos_pos]
                            if len(eos_candidates) > 0:
                                eos_pos = eos_candidates[0]
                                audio_tokens += int(eos_pos - bos_pos - 1)

                # Set config field if available
                if hasattr(model, 'thinker') and hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
                    if hasattr(model.thinker.model.config, 'sparse_attention_config'):
                        model.thinker.model.config.sparse_attention_config['audio_tokens'] = audio_tokens.item() if hasattr(audio_tokens, 'item') else audio_tokens
            except Exception:
                audio_tokens = 0

            # Print number of tokens in the input before prefill starts
            try:
                if hasattr(inputs, 'input_ids'):
                    total_input_tokens = int(inputs.input_ids.shape[1])
                else:
                    total_input_tokens = int(input_token_length)
                print(f"[SparseMM] Prefill input tokens: total_input_tokens={total_input_tokens}, audio_tokens={audio_tokens}")
            except Exception:
                print("[SparseMM] Prefill input tokens: unable to determine token count")
            
            # Initialize cache for SparseMM if enabled
            past_key_values = None
            if USE_SPARSEKV and sparsemm_enabled:
                past_key_values = DynamicCacheSplitHeadFlatten()
            
            # Measure prefill time: generate only 1 token to measure prefill phase
            prefill_start_event = torch.cuda.Event(enable_timing=True)
            prefill_end_event = torch.cuda.Event(enable_timing=True)
            
            prefill_start_event.record()
            with torch.no_grad():
                # First generate call: only 1 token to measure prefill time
                _ = model.generate(
                    **inputs,
                    use_audio_in_video=True,
                    return_audio=False,
                    thinker_max_new_tokens=1,  
                    thinker_do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    past_key_values=past_key_values
                )
            prefill_end_event.record()
            
            # Measure decode time: generate full output
            # Use language-specific max_new_tokens (German words are longer, need more tokens)
            max_new_tokens = LANGUAGE_MAX_NEW_TOKENS.get(language.lower(), DEFAULT_MAX_NEW_TOKENS)
            if language.lower() == "german":
                print(f"[German Optimization] Using max_new_tokens={max_new_tokens} (default for other languages: {DEFAULT_MAX_NEW_TOKENS})")
            
            decode_start_event = torch.cuda.Event(enable_timing=True)
            decode_end_event = torch.cuda.Event(enable_timing=True)
            
            decode_start_event.record()
            with torch.no_grad():
                # Second generate call: full generation to measure decode time
                out_ids = model.generate(
                    **inputs,
                    use_audio_in_video=True,
                    return_audio=False,
                    thinker_max_new_tokens=max_new_tokens,  
                    thinker_do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    past_key_values=past_key_values
                )
            decode_end_event.record()
            
            # Synchronize and calculate times
            torch.cuda.synchronize()
            prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0
            decode_time = decode_start_event.elapsed_time(decode_end_event) / 1000.0
            total_time = prefill_time + decode_time
            
            # Decode response
            resp = processor.batch_decode(
                out_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            # Extract assistant response
            if "assistant\n" in resp:
                assistant_start = resp.rfind("assistant\n") + len("assistant\n")
                resp = resp[assistant_start:].strip()
            
            # Clean response
            hypothesis = clean_response(resp)
            
            # Calculate decode token count
            input_token_count = inputs.input_ids.shape[1] if hasattr(inputs, 'input_ids') else 0
            output_token_count = out_ids.shape[1] if hasattr(out_ids, 'shape') else len(out_ids[0]) if isinstance(out_ids, list) else 0
            decode_token_count = max(0, output_token_count - input_token_count)
            
            # Calculate TPS
            sample_tps = decode_token_count / decode_time if decode_time > 0 else 0.0
            
            # Calculate text lengths
            hypothesis_char_len = len(hypothesis)
            hypothesis_word_len = len(hypothesis.split()) if hypothesis else 0
            reference_char_len = len(reference)
            reference_word_len = len(reference.split()) if reference else 0
            
            # Print inference statistics
            print(f"[Sample {idx+1}] Language={language}, Generated tokens: {decode_token_count}, "
                  f"Prefill time: {prefill_time:.4f}s, Decode time: {decode_time:.4f}s, "
                  f"Total time: {total_time:.4f}s, TPS: {sample_tps:.2f}, "
                  f"Hypothesis: {hypothesis_char_len} chars / {hypothesis_word_len} words, "
                  f"Ground truth: {reference_char_len} chars / {reference_word_len} words")
            
            # Calculate WER with language-specific text normalization
            wer = calculate_wer(reference, hypothesis, language)

            total_wer += wer
            processed_samples += 1

            current_avg_wer = total_wer / processed_samples
            current_avg_accuracy = 100.0 - current_avg_wer  # Word Accuracy = 100 - WER
            
            # Print results
            print(f"\n{'='*80}")
            print(f"[Sample {idx+1}/{len(dataset)}] ID={doc.get('id', f'sample_{idx}')} | "
                  f"Language={language} | Speaker={speaker_id} | Duration={doc.get('duration', 0):.1f}s")
            print(f"{'='*80}")
            print(f"  📊 实时 WER (当前样本): {wer:.2f}%")
            print(f"  📈 平均 WER (已处理 {processed_samples} 个样本): {current_avg_wer:.2f}%")
            print(f"  ✅ 平均准确率: {current_avg_accuracy:.2f}%")
            print(f"{'='*80}")
            
            # Print reference and hypothesis (truncated if too long)
            ref_display = reference[:100] + "..." if len(reference) > 100 else reference
            hyp_display = hypothesis[:100] + "..." if len(hypothesis) > 100 else hypothesis
            print(f"  📝 参考文本:  {ref_display}")
            print(f"  💬 识别结果: {hyp_display}")
            print(f"{'='*80}\n")
            
            # Update progress bar
            update_interval = 10 if is_screen_env else 1
            sample_count = idx + 1
            
            progress_bar.set_postfix({
                '实时WER': f'{wer:.2f}%',
                '平均WER': f'{current_avg_wer:.2f}%',
                '准确率': f'{current_avg_accuracy:.2f}%',
                '样本': f'{sample_count}/{len(dataset)}'
            })
            
            if sample_count % update_interval == 0 or sample_count == len(dataset):
                if is_screen_env:
                    print(f"  📊 进度更新: {sample_count}/{len(dataset)} ({sample_count/len(dataset)*100:.1f}%)")
                    print(f"  📈 当前平均 WER: {current_avg_wer:.2f}% | 平均准确率: {current_avg_accuracy:.2f}%")

            # Store results
            results.append({
                "idx": idx,
                "id": doc.get("id", f"sample_{idx}"),
                "language": language,
                "speaker_id": speaker_id,
                "chapter_id": doc.get("chapter_id", ""),
                "duration": doc.get("duration", 0),
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": wer,
                "response_text": resp,
                "decode_token_count": decode_token_count,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": total_time,
                "tps": sample_tps
            })

            # Use actual measured prefill_time and decode_time
            wall_time = prefill_time + decode_time
            timing_stats.add_sample(wall_time, prefill_time, wall_time, decode_time, decode_token_count)
            
            # Real-time save
            save_interval = 1  # Save after every sample for real-time monitoring
            if (idx + 1) % save_interval == 0 or (idx + 1) == len(dataset):
                save_results_realtime(
                    output_file, 
                    results, 
                    total_wer, 
                    processed_samples,
                    speaker_stats,
                    timing_stats,
                    gpu_id,
                    prune_ratio,
                    method_is,
                    sample_limit,
                    dataset_path,
                    len(dataset),
                    language_stats
                )

            # CRITICAL: Clear cache and reset state after each sample
            if USE_SPARSEKV and sparsemm_enabled:
                # Clear past_key_values to free memory
                if past_key_values is not None:
                    if hasattr(past_key_values, 'key_cache'):
                        for layer_cache in past_key_values.key_cache:
                            if isinstance(layer_cache, list):
                                for cache in layer_cache:
                                    if isinstance(cache, torch.Tensor):
                                        del cache
                            elif isinstance(layer_cache, torch.Tensor):
                                del layer_cache
                    if hasattr(past_key_values, 'value_cache'):
                        for layer_cache in past_key_values.value_cache:
                            if isinstance(layer_cache, list):
                                for cache in layer_cache:
                                    if isinstance(cache, torch.Tensor):
                                        del cache
                            elif isinstance(layer_cache, torch.Tensor):
                                del layer_cache
                past_key_values = None
            
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Periodic garbage collection
            if (idx + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            traceback.print_exc()
            
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            continue

    # Calculate final statistics
    final_wer = total_wer / processed_samples if processed_samples > 0 else 100.0

    # Calculate speaker-level statistics
    speaker_wer = defaultdict(list)
    for result in results:
        speaker_wer[result["speaker_id"]].append(result["wer"])
    
    speaker_avg_wer = {}
    for speaker_id, wers in speaker_wer.items():
        speaker_avg_wer[speaker_id] = sum(wers) / len(wers)

    # Calculate language-level statistics
    language_wer = defaultdict(list)
    for result in results:
        language_wer[result["language"]].append(result["wer"])
    
    language_avg_wer = {}
    for lang, wers in language_wer.items():
        language_avg_wer[lang] = sum(wers) / len(wers)

    # Calculate detailed metrics
    references = [r["reference"] for r in results]
    hypotheses = [r["hypothesis"] for r in results]
    
    detailed_metrics = calculate_multilingual_metrics(references, hypotheses)

    # Build final results
    final_results = {
        "model_name": f"Qwen2.5-Omni-3B",
        "dataset": "Multilingual LibriSpeech",
        "status": "completed",
        "progress": {
            "processed": len(results),
            "total": len(dataset),
            "percentage": 100.0
        },
        "total_samples": len(results),
        "valid_samples": detailed_metrics['valid_samples'],
        
        # WER metrics
        "wer_mean": detailed_metrics['wer_mean'],
        "word_accuracy": detailed_metrics['word_accuracy'],
        "wer_std": detailed_metrics['wer_std'],
        "wer_min": detailed_metrics['wer_min'],
        "wer_max": detailed_metrics['wer_max'],
        "perfect_predictions": detailed_metrics['perfect_predictions'],
        
        # Speaker statistics
        "speaker_count": len(speaker_stats),
        "speaker_wer": speaker_avg_wer,
        
        # Language statistics
        "language_count": len(language_stats),
        "language_wer": language_avg_wer,
        
        # Timing statistics
        "timing_stats": timing_stats.get_summary(),
        
        # Configuration
        "config": {
            "device": f"cuda:{gpu_id}",
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "gpu_id": str(os.environ.get('CUDA_VISIBLE_DEVICES', 'default')),
            "pruning_config": {
                "prune_ratio": prune_ratio, 
                "prune_method": method_is
            },
            "sample_limit": sample_limit,
            "data_path": dataset_path,
            "languages": args.languages
        },
        
        # Detailed results
        "detailed_results": results
    }
    
    # Convert numpy types
    final_results = convert_numpy_types(final_results)
    
    # Save final results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    # Export timing statistics
    timing_stats.export_to_json(timing_output_file)

    # Print summary
    print(f"\n=== Multilingual LibriSpeech ASR Evaluation Results Summary (Qwen2.5-Omni) ===")
    print(f"Total samples: {len(results)}")
    print(f"Processed samples: {processed_samples}")
    print(f"Average WER: {detailed_metrics['wer_mean']:.2f}%")
    print(f"Word accuracy: {detailed_metrics['word_accuracy']:.2f}%")
    print(f"Number of speakers: {len(speaker_stats)}")
    print(f"Number of languages: {len(language_stats)}")
    
    # Print WER statistics
    print(f"\n=== WER Statistics ===")
    print(f"WER standard deviation: {detailed_metrics['wer_std']:.2f}%")
    print(f"Minimum WER: {detailed_metrics['wer_min']:.2f}%")
    print(f"Maximum WER: {detailed_metrics['wer_max']:.2f}%")
    print(f"Perfect predictions: {detailed_metrics['perfect_predictions']} ({detailed_metrics['perfect_predictions']/detailed_metrics['valid_samples']*100:.1f}%)")
    
    # Print language-level statistics
    print(f"\n=== Language-level WER ===")
    for lang, avg_wer in sorted(language_avg_wer.items()):
        sample_count = language_stats[lang]
        print(f"  {lang}: {avg_wer:.2f}% ({sample_count} samples)")
    
    print(f"\nTop 5 speakers by WER:")
    for speaker_id, avg_wer in list(speaker_avg_wer.items())[:5]:
        sample_count = speaker_stats[speaker_id]
        print(f"  {speaker_id}: {avg_wer:.2f}% ({sample_count} samples)")
    
    timing_summary = timing_stats.get_summary()
    print(f"\n=== Inference Time Statistics ===")
    print(f"Statistical sample count: {timing_summary['count']} (excluding first sample)")
    print(f"Average inference time: {timing_summary['avg_wall_time']:.4f} seconds")
    print(f"Average Prefill time: {timing_summary['avg_prefill_time']:.4f} seconds")
    print(f"Average Decode time: {timing_summary['avg_decode_time']:.4f} seconds")
    print(f"Average total GPU time: {timing_summary['avg_total_gpu_time']:.4f} seconds")
    print(f"\n=== TPS (Tokens Per Second) Statistics ===")
    print(f"Total decoded tokens: {timing_summary['total_decode_tokens']}")
    print(f"Average TPS (per sample): {timing_summary['avg_tps']:.2f} tokens/second")
    print(f"Overall TPS (total tokens / total decode time): {timing_summary['overall_tps']:.2f} tokens/second")
    
    print(f"\nResults saved to: {output_file}")
    print(f"Timing statistics saved to: {timing_output_file}")
    print("="*80)

if __name__ == "__main__":
    main()

