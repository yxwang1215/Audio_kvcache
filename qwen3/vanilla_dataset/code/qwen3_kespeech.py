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
from typing import Dict, Any
import editdistance as ed


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
ENV_HEAD_SCORE_FILE = os.environ.get("HEAD_SCORE_FILE", "aggregated")
ENV_HEAD_SCORE_BATCH = os.environ.get("HEAD_SCORE_BATCH", None)

# Audio-aware TopK selection parameters (from WHY_IMPROVEMENTS.md)
USE_AUDIO_AWARE_TOPK = os.environ.get("USE_AUDIO_AWARE_TOPK", "false").lower() == "true"
ENV_AUDIO_BOUNDARY_BOOST = float(os.environ.get("AUDIO_BOUNDARY_BOOST", "1.3"))
ENV_AUDIO_CONTINUITY_WEIGHT = float(os.environ.get("AUDIO_CONTINUITY_WEIGHT", "0.1"))
ENV_AUDIO_SMOOTH_WINDOW = int(os.environ.get("AUDIO_SMOOTH_WINDOW", "5"))

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
    # Import from model directory (if sparsekv version exists for qwen3)
    try:
        from model.modeling_qwen3_omni_sparsekv import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeFlashAttention2SparseMM
        print("Using sparsekv version of the model")
    except ImportError:
        # Fallback to standard qwen3 model if sparsekv version doesn't exist
        print("SparseKV version not found, using standard Qwen3 model")
        USE_SPARSEKV = False
        from transformers import Qwen3OmniMoeForConditionalGeneration
        print("Using standard version of the model from transformers (sparsekv disabled)")
else:
    # Try importing from transformers first, then fallback to local file
    try:
        from transformers import Qwen3OmniMoeForConditionalGeneration
        print("Using standard version of the model from transformers (sparsekv disabled)")
    except ImportError:
        # Fallback to local modeling file
        try:
            from modular_qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration
            print("Using standard version of the model from local file (sparsekv disabled)")
        except ImportError as e:
            print(f"Error importing Qwen3OmniMoeForConditionalGeneration: {e}")
            print(f"Current sys.path: {sys.path[:5]}")
            raise

try:
    from processing_qwen3_omni_moe import Qwen3OmniMoeProcessor
except ImportError:
    from model.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor
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
    num_key_value_groups: int = 1
) -> Dict[str, Any]:
    """
    Load and process audio head scores for SparseMM
    Returns a dictionary compatible with SparseMM's head_score format
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

def load_kespeech_dataset(base_dir, split="test"):
    """Load KeSpeech dataset from local path or HuggingFace, download if not exists"""
    dataset = []
    split_dir = os.path.join(base_dir, split)
    
    # Default download directory
    default_download_dir = "/mnt/users/wangyuxuan-20250915/Audio/Datasets"
    download_base_dir = default_download_dir
    download_split_dir = os.path.join(download_base_dir, "TwinkStart", "kespeech", split)
    
    # Check if local audio files exist
    if os.path.exists(split_dir):
        wav_files = glob.glob(os.path.join(split_dir, "*.wav"))
        if len(wav_files) > 0:
            print(f"Found {len(wav_files)} local audio files in {split_dir}")
    
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
    # Try to load from local saved dataset first
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
            print(f"\n{'='*80}")
            print(f"说明: 需要下载数据集的元数据文件 (Parquet 格式)")
            print(f"  - Parquet 文件包含: 音频ID、转录文本、speaker_id 等标注信息")
            print(f"  - 虽然本地已有音频文件 (.wav), 但缺少这些元数据")
            print(f"  - 没有转录文本就无法进行 ASR 评估 (无法计算 CER)")
            print(f"  - 这些元数据文件较小, 只包含文本信息, 不包含音频数据")
            print(f"  - 下载位置: {download_base_dir}")
            print(f"{'='*80}\n")
            print(f"Dataset not found in cache. Attempting to download metadata to {download_base_dir}...")
            
            # Download dataset metadata to the specified directory
            # Note: This downloads only metadata (parquet files), not audio files
            # Audio files are already available locally
            try:
                from datasets import load_dataset, DatasetDict
                os.makedirs(download_base_dir, exist_ok=True)
                
                # Set cache directory temporarily for this download
                original_cache_dir = os.environ.get("HF_DATASETS_CACHE", None)
                try:
                    # Set cache directory to our download location
                    os.environ["HF_DATASETS_CACHE"] = download_base_dir
                    print(f"Downloading dataset metadata (parquet files) to {download_base_dir}...")
                    print(f"  (This will only download metadata, not audio files)")
                    hf_dataset = load_dataset(
                        "TwinkStart/kespeech", 
                        split=split
                    )
                    print(f"Successfully downloaded metadata for {len(hf_dataset)} samples")
                    
                    # Find and display actual cache location
                    print(f"\n{'='*80}")
                    print(f"查找元数据实际存储位置...")
                    try:
                        # HuggingFace datasets stores files in a specific structure under HF_DATASETS_CACHE
                        # The structure is usually: HF_DATASETS_CACHE/datasets/{dataset_name}/{hash}/...
                        
                        # Look for parquet files in the cache directory
                        cache_pattern = os.path.join(download_base_dir, "**", "*.parquet")
                        parquet_files = glob.glob(cache_pattern, recursive=True)
                        
                        if not parquet_files:
                            # Also check in the datasets subdirectory
                            datasets_cache_pattern = os.path.join(download_base_dir, "datasets", "**", "*.parquet")
                            parquet_files = glob.glob(datasets_cache_pattern, recursive=True)
                        
                        if parquet_files:
                            # Get the directory containing parquet files
                            actual_cache_dir = os.path.dirname(parquet_files[0])
                            print(f"✓ 元数据实际存储位置:")
                            print(f"  {actual_cache_dir}")
                            print(f"  ✓ 包含 {len(parquet_files)} 个 parquet 文件")
                            # Calculate total size
                            total_size = sum(os.path.getsize(f) for f in parquet_files)
                            total_size_mb = total_size / (1024 * 1024)
                            total_size_gb = total_size_mb / 1024
                            if total_size_gb >= 1:
                                print(f"  ✓ 总大小: {total_size_gb:.2f} GB ({total_size_mb:.2f} MB)")
                            else:
                                print(f"  ✓ 总大小: {total_size_mb:.2f} MB")
                        else:
                            # Check default HuggingFace cache location
                            default_hf_cache = os.path.expanduser("~/.cache/huggingface/datasets")
                            default_pattern = os.path.join(default_hf_cache, "**", "*kespeech*", "*.parquet")
                            default_parquet = glob.glob(default_pattern, recursive=True)
                            if default_parquet:
                                actual_cache_dir = os.path.dirname(default_parquet[0])
                                print(f"⚠ 注意: 元数据存储在默认 HuggingFace 缓存位置:")
                                print(f"  {actual_cache_dir}")
                            else:
                                # Check environment variable
                                current_cache = os.environ.get("HF_DATASETS_CACHE", "未设置")
                                print(f"⚠ 无法找到 parquet 文件")
                                print(f"  当前 HF_DATASETS_CACHE: {current_cache}")
                                print(f"  请检查: {download_base_dir}")
                                print(f"  或默认位置: {default_hf_cache}")
                    except Exception as cache_info_e:
                        print(f"⚠ 无法确定实际缓存位置: {cache_info_e}")
                        print(f"  设置的缓存目录: {download_base_dir}")
                        print(f"  请手动检查该目录下的文件")
                    print(f"{'='*80}\n")
                finally:
                    # Restore original cache directory setting
                    if original_cache_dir is not None:
                        os.environ["HF_DATASETS_CACHE"] = original_cache_dir
                    elif "HF_DATASETS_CACHE" in os.environ:
                        del os.environ["HF_DATASETS_CACHE"]
                
                # Save dataset to disk for future use
                try:
                    saved_dataset_path = os.path.join(download_base_dir, "TwinkStart", "kespeech", "saved_dataset")
                    os.makedirs(os.path.dirname(saved_dataset_path), exist_ok=True)
                    if not os.path.exists(saved_dataset_path):
                        print(f"Saving dataset to {saved_dataset_path} for future use...")
                        dataset_dict = DatasetDict({split: hf_dataset})
                        dataset_dict.save_to_disk(saved_dataset_path)
                        print(f"Dataset saved successfully")
                except Exception as save_e:
                    print(f"Warning: Could not save dataset to disk: {save_e}")
                    
            except Exception as download_e:
                print(f"Failed to download dataset: {download_e}")
                traceback.print_exc()
                print(f"Falling back to local directory scan: {split_dir}")
                hf_dataset = None
    
    # Process the dataset if we have it
    if hf_dataset is not None:
        # Map HuggingFace dataset to local audio files
        # Use index-based matching: local files are named as {index}.wav (e.g., 0.wav, 1.wav, ...)
        # This matches UltraEval-Audio's save_audio_to_local function behavior
        for idx, item in enumerate(hf_dataset):
            audio_id = str(item.get("id", item.get("audio_id", f"kespeech_{idx}")))
            transcription = item.get("Text", item.get("text", ""))
            
            # Priority 1: Use index-based naming (UltraEval-Audio format: {index}.wav)
            # This is the most reliable way to match local files with HuggingFace dataset
            local_audio_path = os.path.join(split_dir, f"{idx}.wav")
            if not os.path.exists(local_audio_path):
                # Try alternative paths
                search_paths = [
                    split_dir,
                    download_split_dir,
                    os.path.join(base_dir, split),
                    os.path.join(download_base_dir, "TwinkStart", "kespeech", split)
                ]
                found = False
                for search_dir in search_paths:
                    if os.path.exists(search_dir):
                        potential_path = os.path.join(search_dir, f"{idx}.wav")
                        if os.path.exists(potential_path):
                            local_audio_path = potential_path
                            found = True
                            break
                if not found:
                    local_audio_path = None
            
            # Priority 2: If index-based matching fails, try HuggingFace audio path
            if not local_audio_path or not os.path.exists(local_audio_path):
                hf_audio_path = None
                if "audio" in item and "path" in item["audio"]:
                    hf_audio_path = item["audio"]["path"]
                elif "path" in item:
                    hf_audio_path = item["path"]
                
                if hf_audio_path:
                    # Convert relative path to absolute path
                    if not os.path.isabs(hf_audio_path):
                        # Try to resolve relative path from various possible base directories
                        possible_bases = [
                            split_dir,
                            base_dir,
                            download_split_dir,
                            download_base_dir,
                            os.path.join(download_base_dir, "TwinkStart", "kespeech", split)
                        ]
                        for base in possible_bases:
                            abs_path = os.path.join(base, hf_audio_path)
                            if os.path.exists(abs_path):
                                local_audio_path = abs_path
                                break
                        if not local_audio_path:
                            # Try using the relative path as-is from split_dir
                            local_audio_path = os.path.join(split_dir, hf_audio_path)
                            if not os.path.exists(local_audio_path):
                                # Last resort: use the relative path as filename
                                filename = os.path.basename(hf_audio_path)
                                local_audio_path = os.path.join(split_dir, filename)
                    else:
                        local_audio_path = hf_audio_path
            
            # Priority 3: If still not found, try using audio_id (fallback)
            if not local_audio_path or not os.path.exists(local_audio_path):
                local_audio_path = os.path.join(split_dir, f"{audio_id}.wav")
                if not os.path.exists(local_audio_path):
                    local_audio_path = os.path.join(split_dir, f"{audio_id}.mp3")
                    if not os.path.exists(local_audio_path):
                        local_audio_path = None
            
            if not local_audio_path or not os.path.exists(local_audio_path):
                print(f"Warning: Could not find audio file for index {idx}, ID {audio_id}, skipping")
                continue
            
            # Ensure path is absolute
            if not os.path.isabs(local_audio_path):
                local_audio_path = os.path.abspath(local_audio_path)
            
            # Final check: if file still doesn't exist, skip this sample
            if not os.path.exists(local_audio_path):
                print(f"Warning: Audio file not found for index {idx}, ID {audio_id}: {local_audio_path}, skipping")
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
        
        print(f"Successfully loaded {len(dataset)} KeSpeech samples")
        return dataset
    
    # Fallback: load from local directory (audio files only, no annotations)
    if not os.path.exists(split_dir):
        # Also check download directory
        if os.path.exists(download_split_dir):
            split_dir = download_split_dir
            print(f"Using downloaded dataset directory: {split_dir}")
        else:
            print(f"Error: Dataset path does not exist: {split_dir} or {download_split_dir}")
            return []
    
    print(f"Scanning local dataset directory: {split_dir}")
    
    # Load all wav files from directory
    wav_files = sorted(glob.glob(os.path.join(split_dir, "*.wav")))
    
    if len(wav_files) == 0:
        print(f"Warning: No wav files found in {split_dir}")
        return []
    
    print(f"Found {len(wav_files)} audio files, but no annotations available.")
    print("Please ensure HuggingFace datasets library can access TwinkStart/kespeech for annotations.")
    
    # If we can't get annotations, we can't evaluate, so return empty
    # Or we could try to load annotations from a separate file if it exists
    return []

def prepare_audio_for_qwen_omni(audio_path, target_sr=16000):
    """Process audio files according to Qwen3-Omni requirements"""
    
    # Check if file exists first
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file does not exist: {audio_path}")
        print(f"  Current working directory: {os.getcwd()}")
        print(f"  Absolute path: {os.path.abspath(audio_path)}")
        # Try to find the file by filename
        filename = os.path.basename(audio_path)
        print(f"  Attempting to find file by name: {filename}")
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return silence
    
    try:
        
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            # Only print success message for first few files to avoid spam
            # print(f"Successfully loaded with librosa: shape={audio.shape}, sample rate={sr}Hz")
        except Exception as e:
            print(f"librosa loading failed: {e}")
            print(f"  File path: {audio_path}")
            print(f"  File exists: {os.path.exists(audio_path)}")
            
            
            try:
                audio, sample_rate = sf.read(audio_path)
                
                
                if len(audio.shape) > 1 and audio.shape[1] > 1:
                    audio = np.mean(audio, axis=1)
                
                
                if sample_rate != target_sr:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * target_sr / sample_rate))
                    
                audio = audio.astype(np.float32)
                sr = target_sr
                # print(f"soundfile processing successful: shape={audio.shape}, sample rate={sr}Hz")
                
            except Exception as e:
                print(f"soundfile loading also failed: {e}")
                print(f"  File path: {audio_path}")
                print(f"  File exists: {os.path.exists(audio_path)}")
                
                audio = np.zeros(target_sr * 3, dtype=np.float32)
                sr = target_sr
                print("WARNING: Generating silent replacement audio - this will cause high CER!")
        
        audio = audio.astype(np.float32)
        
        return audio
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        print(f"  File path: {audio_path}")
        traceback.print_exc()
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return silence

def kespeech_doc_to_audio(doc):
    """Load audio data from KeSpeech documents"""
    if "audio" not in doc:
        return None
    
    if doc["audio"]["array"] is None:
        
        try:
            audio_data = prepare_audio_for_qwen_omni(doc["audio"]["path"])
            doc["audio"]["array"] = audio_data
            doc["audio"]["sampling_rate"] = 16000  
        except Exception as e:
            print(f"Unable to load audio file {doc['audio']['path']}: {e}")
            return None
    
    return doc["audio"]["array"], doc["audio"]["sampling_rate"]

def asr_doc_to_text(doc, kwargs=None):
    """Generate prompts for Chinese ASR tasks (KeSpeech)"""
    if kwargs is None:
        kwargs = {}
    
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    
    
    # Use Chinese prompt following UltraEval-Audio format for qwen-omni-asr-zh
    instruction = "请将这段中文语音转换为纯文本，去掉标点符号。"
    
    
    prompt_text = f"{pre_prompt}{instruction}{post_prompt}"
    
    return prompt_text

def clean_response(response):
    """Clean ASR response (supports both Chinese and English)"""
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
    
    # Remove common markers (both English and Chinese)
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

def standardize_text_for_cer(text):
    """Normalize text for CER calculation (Chinese character-level)"""
    if not text:
        return ""
    
    # Remove punctuation marks (both Chinese and English)
    # Common Chinese punctuation: ，。！？；：""''（）【】《》、· etc.
    # Common English punctuation: , . ! ? ; : " ' ( ) [ ] { } - _ etc.
    punctuation_pattern = r'[，。！？；：""''（）【】《》、·,\.!?;:"\'\(\)\[\]\{\}\-\_\s]+'
    text = re.sub(punctuation_pattern, '', text)
    
    # Remove extra whitespace (in case any remains)
    text = re.sub(r'\s+', '', text).strip()
    
    return text

def calculate_cer(reference, hypothesis):
    """Calculate Character Error Rate (CER) for Chinese text"""
    
    # Standardize text (minimal processing for Chinese)
    ref_standardized = standardize_text_for_cer(reference)
    hyp_standardized = standardize_text_for_cer(hypothesis)
    
    # For Chinese CER, split into characters
    # Remove spaces and split by character
    ref_chars = list(ref_standardized.replace(' ', ''))
    hyp_chars = list(hyp_standardized.replace(' ', ''))
    
    # Calculate character-level edit distance
    if len(ref_chars) == 0:
        if len(hyp_chars) == 0:
            return 0.0
        else:
            return 100.0
    
    # Calculate CER using edit distance
    distance = ed.eval(ref_chars, hyp_chars)
    cer = distance / len(ref_chars)
    cer_percent = cer * 100
    
    # CER can exceed 100% when there are many insertions
    # CER = (S + D + I) / N, where N is the number of characters in reference
    # If insertions are very high, CER can be > 100%
    if cer_percent > 100.0:
        # Calculate detailed metrics for debugging
        try:
            if hasattr(jiwer, 'compute_measures'):
                # For character-level, we need to use space-separated characters
                ref_chars_str = ' '.join(ref_chars)
                hyp_chars_str = ' '.join(hyp_chars)
                measures = jiwer.compute_measures(ref_chars_str, hyp_chars_str)
                substitutions = measures.get('substitutions', 0)
                deletions = measures.get('deletions', 0)
                insertions = measures.get('insertions', 0)
                hits = measures.get('hits', 0)
                print(f"[CER Analysis] CER={cer_percent:.2f}%, Ref chars={len(ref_chars)}, Hyp chars={len(hyp_chars)}")
                print(f"  Substitutions={substitutions}, Deletions={deletions}, Insertions={insertions}, Hits={hits}")
                if insertions > len(ref_chars):
                    print(f"  [Issue] Insertions ({insertions}) > Reference chars ({len(ref_chars)}) - this causes CER > 100%")
            else:
                # Fallback: just print basic info
                print(f"[CER Analysis] CER={cer_percent:.2f}%, Ref chars={len(ref_chars)}, Hyp chars={len(hyp_chars)}")
                print(f"  [Note] Detailed metrics unavailable (jiwer.compute_measures not available in this version)")
                if len(hyp_chars) > len(ref_chars):
                    print(f"  [Issue] Hypothesis has more chars ({len(hyp_chars)}) than reference ({len(ref_chars)}) - this may cause CER > 100%")
        except AttributeError:
            # Fallback if compute_measures doesn't exist
            print(f"[CER Analysis] CER={cer_percent:.2f}%, Ref chars={len(ref_chars)}, Hyp chars={len(hyp_chars)}")
            print(f"  [Note] Detailed metrics unavailable (jiwer.compute_measures not available in this version)")
            if len(hyp_chars) > len(ref_chars):
                print(f"  [Issue] Hypothesis has more chars ({len(hyp_chars)}) than reference ({len(ref_chars)}) - this may cause CER > 100%")

    return cer_percent


def save_results_realtime(output_file, results, total_cer, processed_samples, speaker_stats, 
                          timing_stats, gpu_id, prune_ratio, method_is, 
                          sample_limit, kespeech_path, dataset_size):
    """
    Save results to JSON file in real-time, updating statistics as we go
    This allows monitoring progress and viewing results before completion
    """
    try:
        # Calculate current statistics
        current_avg_cer = total_cer / processed_samples if processed_samples > 0 else 100.0
        current_avg_accuracy = 100.0 - current_avg_cer
        
        # Calculate speaker-level statistics
        speaker_cer = defaultdict(list)
        for result in results:
            speaker_cer[result["speaker_id"]].append(result["cer"])
        
        speaker_avg_cer = {}
        for speaker_id, cers in speaker_cer.items():
            speaker_avg_cer[speaker_id] = sum(cers) / len(cers)
        
        # Calculate detailed metrics from current results
        references = [r["reference"] for r in results]
        hypotheses = [r["hypothesis"] for r in results]
        
        # Calculate CER statistics
        cer_scores = [r["cer"] for r in results]
        if len(cer_scores) > 0:
            cer_mean = np.mean(cer_scores)
            cer_std = np.std(cer_scores)
            cer_min = np.min(cer_scores)
            cer_max = np.max(cer_scores)
            perfect_predictions = len([cer for cer in cer_scores if cer == 0.0])
            char_accuracy = 100.0 - cer_mean
        else:
            cer_mean = 100.0
            cer_std = 0.0
            cer_min = 100.0
            cer_max = 100.0
            perfect_predictions = 0
            char_accuracy = 0.0
        
        # Build real-time results structure
        realtime_results = {
            "model_name": f"Qwen3-Omni-30B-A3B-Instruct",
            "dataset": "KeSpeech",
            "status": "in_progress",
            "progress": {
                "processed": len(results),
                "total": dataset_size,
                "percentage": len(results) / dataset_size * 100 if dataset_size > 0 else 0
            },
            "current_statistics": {
                "cer_mean": float(cer_mean),
                "char_accuracy": float(char_accuracy),
                "cer_std": float(cer_std),
                "cer_min": float(cer_min),
                "cer_max": float(cer_max),
                "perfect_predictions": int(perfect_predictions),
                "valid_samples": len(results),
                "current_avg_cer": float(current_avg_cer),
                "current_avg_accuracy": float(current_avg_accuracy)
            },
            "speaker_count": len(speaker_stats),
            "speaker_cer": speaker_avg_cer,
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
                "data_path": kespeech_path
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

def calculate_kespeech_metrics(references, hypotheses):
    """
    KeSpeech ASR metrics calculation using CER (Character Error Rate)
    
    Args:
        references: List of true transcriptions
        hypotheses: List of predicted transcriptions
        
    Returns:
        dict: Dictionary containing CER-related metrics
    """
    
    valid_indices = []
    clean_references = []
    clean_hypotheses = []
    cer_scores = []
    
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        if ref and hyp:  
            valid_indices.append(i)
            clean_references.append(ref)
            clean_hypotheses.append(hyp)
            
            
            cer = calculate_cer(ref, hyp)
            cer_scores.append(cer)
    
    if len(clean_references) == 0:
        return {
            'cer_mean': 100.0,
            'cer_std': 0.0,
            'cer_min': 100.0,
            'cer_max': 100.0,
            'perfect_predictions': 0,
            'valid_samples': 0,
            'total_samples': len(references),
            'char_accuracy': 0.0
        }
    
    
    cer_mean = np.mean(cer_scores)
    cer_std = np.std(cer_scores)
    cer_min = np.min(cer_scores)
    cer_max = np.max(cer_scores)
    
    
    perfect_predictions = len([cer for cer in cer_scores if cer == 0.0])
    
    
    char_accuracy = 100.0 - cer_mean
    
    return {
        'cer_mean': float(cer_mean),
        'cer_std': float(cer_std),
        'cer_min': float(cer_min),
        'cer_max': float(cer_max),
        'perfect_predictions': int(perfect_predictions),
        'valid_samples': len(clean_references),
        'total_samples': len(references),
        'char_accuracy': float(char_accuracy),
        'cer_scores': cer_scores
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='KeSpeech ASR Evaluation with Qwen3-Omni')
    parser.add_argument('--output-file', '-o', type=str, default=None,
                        help='Output file path for results JSON (overrides --output-dir)')
    parser.add_argument('--output-dir', '-d', type=str, default=None,
                        help='Output directory for results (default: ./LibriSpeech_Results or from RESULTS_DIR env)')
    parser.add_argument('--timing-output-file', '-t', type=str, default=None,
                        help='Output file path for timing statistics JSON')
    parser.add_argument('--model-path', '-m', type=str, default=None,
                        help='Path to the model directory (default: from MODEL_PATH env or hardcoded path)')
    args = parser.parse_args()
    
    gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_id = gpu_temp[-1] if gpu_temp else "0"
    print(f"Using GPU ID: {gpu_id}")

    
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

    
    sample_limit = int(os.environ.get("SAMPLE_LIMIT", 0))
    if sample_limit > 0:
        print(f"Sample limit set to: {sample_limit}")

    
    # Set default dataset path
    default_kespeech_path = '/mnt/users/wangyuxuan-20250915/Audio/UltraEval-Audio/raw/TwinkStart/kespeech'
    download_base_dir = '/mnt/users/wangyuxuan-20250915/Audio/Datasets'
    download_kespeech_path = os.path.join(download_base_dir, 'TwinkStart', 'kespeech')
    
    # Check if default path exists, otherwise use download path
    if os.path.exists(default_kespeech_path):
        kespeech_path = default_kespeech_path
        print(f"Using default dataset path: {kespeech_path}")
    elif os.path.exists(download_kespeech_path):
        kespeech_path = download_kespeech_path
        print(f"Using downloaded dataset path: {kespeech_path}")
    else:
        # Will be downloaded in load_kespeech_dataset if needed
        kespeech_path = default_kespeech_path
        print(f"Dataset path not found, will download to: {download_base_dir} if needed")
    
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
            result_dir = os.environ.get("RESULTS_DIR", './KeSpeech_Results')
        
        kespeech_path = os.path.abspath(kespeech_path)
        result_dir = os.path.abspath(result_dir)
        os.makedirs(result_dir, exist_ok=True)
        output_file = f'{result_dir}/kespeech_results_qwen3.json'
    
    # Determine timing output file path
    if args.timing_output_file:
        timing_output_file = os.path.abspath(args.timing_output_file)
        timing_output_dir = os.path.dirname(timing_output_file)
        os.makedirs(timing_output_dir, exist_ok=True)
    else:
        # Use same directory as output_file
        timing_output_dir = os.path.dirname(output_file)
        timing_output_file = f'{timing_output_dir}/timing_stats_qwen3_{method_is}_{prune_ratio}.json'
    
    print(f"Results will be saved to: {output_file} (updated in real-time)")
    print(f"Timing statistics will be saved to: {timing_output_file}")

    
    timing_stats = GlobalTimingStats()

    print(f"\n=== KeSpeech ASR Evaluation Configuration (Qwen3-Omni) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Pruning ratio: {prune_ratio}")
    print(f"Pruning method: {method_is}")
    print(f"Data path: {kespeech_path}")
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

    
    print("Loading Qwen3-Omni model...")
    # Get model path from command line argument, environment variable, or use default
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.environ.get("MODEL_PATH", "/mnt/public/models/Qwen3-Omni-30B-A3B-Instruct")
    print(f"Using model path: {model_path}")
    device_map = {"": 0}  
    
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.disable_talker()
    
    
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
        model.thinker.model.config.sparse_attention_config = {'prune_ratio': prune_ratio, 'prune_method': prune_method}
        print(f"Sparse attention config set: prune_ratio={prune_ratio}, prune_method={prune_method}")
    else:
        print("Warning: thinker model config not found, using default parameters")
    
    
    
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
            head_score_type = ENV_HEAD_SCORE_FILE
            head_score_batch = ENV_HEAD_SCORE_BATCH if ENV_HEAD_SCORE_BATCH else None
            head_score_path = get_audio_head_score_path(
                score_type=head_score_type,
                batch=head_score_batch
            )
            
            print(f"[SparseMM] Loading audio head scores from: {head_score_path}")
            head_score_data = load_audio_head_scores_for_sparsemm(
                head_score_path,
                num_hidden_layers,
                num_attention_heads
            )
            
            # Configure SparseMM
            # If budget is a ratio, use a placeholder value for initialization
            # The actual budget will be calculated dynamically based on prefill token count
            if ENV_BUDGET_IS_RATIO:
                # Use a reasonable default for initialization (will be updated per sample)
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

    
    print(f"Loading KeSpeech dataset: {kespeech_path}")
    dataset = load_kespeech_dataset(kespeech_path, "test")
    
    if not dataset:
        print("Error: Failed to load any data")
        return
    
    
    if sample_limit > 0 and len(dataset) > sample_limit:
        dataset = dataset[:sample_limit]
        print(f"Applying sample limit, processing {len(dataset)} samples")

    
    speaker_stats = defaultdict(int)
    for sample in dataset:
        speaker_id = sample.get("speaker_id", "unknown")
        speaker_stats[speaker_id] += 1
    
    print(f"Speaker count: {len(speaker_stats)} speakers")
    print(f"Sample distribution: {dict(list(speaker_stats.items())[:5])}...")

    results = []
    total_cer = 0.0
    processed_samples = 0

    
    is_screen_env = not sys.stdout.isatty() or 'TERM' in os.environ and os.environ['TERM'] == 'screen'
    if is_screen_env:
        print("Detected screen or non-interactive environment, using simplified progress display")
    
    tqdm_kwargs = {
        'ascii': True,      
        'dynamic_ncols': True, 
        'file': sys.stdout    
    }

    print(f"Starting evaluation of {len(dataset)} samples...")
    
    
    allocated, reserved = get_gpu_memory_usage()
    print(f"GPU memory after model loading - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    progress_bar = tqdm(enumerate(dataset), total=len(dataset), desc="KeSpeech ASR evaluation (Qwen3)", **tqdm_kwargs)

    for idx, doc in progress_bar:
        try:
            # CRITICAL: Reset SparseMM state before each sample to prevent state accumulation
            # This is essential for performance - accumulated state can cause significant slowdown
            # OPTIMIZATION: Only reset if actually needed (not on first sample or if state exists)
            if USE_SPARSEKV and sparsemm_enabled:
                # Check if reset is actually needed by checking if any layer has non-None state
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
            
            audio_data_result = kespeech_doc_to_audio(doc)
            if audio_data_result is None:
                continue
            
            audio_np, sr = audio_data_result
            
            
            reference = doc.get("transcription", "")
            speaker_id = doc.get("speaker_id", "unknown")
            
            
            prompt_text = asr_doc_to_text(doc)

            
            # For Chinese ASR, use the prompt directly (already contains instruction)
            # Following UltraEval-Audio format: system message + user prompt with audio
            full_user_prompt = prompt_text

            
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a speech recognition model."}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_np},
                        {"type": "text", "text": full_user_prompt}
                    ]
                }
            ]
            
            
            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
            
            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            
            if isinstance(text, list):
                text = text[0] if len(text) > 0 else ""
            
            
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
            
            # Compute audio token count (if any) before prefill so we can print it
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

            # Print number of tokens in the input before prefill starts (for debugging)
            try:
                # input_token_length computed earlier as a fallback; prefer inputs.input_ids if available
                if hasattr(inputs, 'input_ids'):
                    total_input_tokens = int(inputs.input_ids.shape[1])
                else:
                    total_input_tokens = int(input_token_length)
                print(f"[SparseMM] Prefill input tokens: total_input_tokens={total_input_tokens}, audio_tokens={audio_tokens}")
            except Exception:
                print("[SparseMM] Prefill input tokens: unable to determine token count")
            
            # Initialize cache for SparseMM if enabled
            # NOTE: Create fresh cache for each sample to avoid state leakage
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
            
            # CRITICAL FIX: Reset SparseMM state and create fresh cache for second generate
            # The first generate call pollutes the cache state, so we MUST reset before the actual generation
            if USE_SPARSEKV and sparsemm_enabled:
                reset_sparsemm_state(model)
                past_key_values = DynamicCacheSplitHeadFlatten()  # Create fresh cache
            
            # Measure decode time: generate full output
            decode_start_event = torch.cuda.Event(enable_timing=True)
            decode_end_event = torch.cuda.Event(enable_timing=True)
            
            decode_start_event.record()
            with torch.no_grad():
                # Second generate call: full generation to measure decode time
                out_ids = model.generate(
                    **inputs,
                    use_audio_in_video=True,
                    return_audio=False,
                    thinker_max_new_tokens=1100,  
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
            
            # Handle different return types from model.generate()
            # The return structure should be: GenerateDecoderOnlyOutput with sequences attribute
            if hasattr(out_ids, 'sequences'):
                generated_sequences = out_ids.sequences
            elif isinstance(out_ids, tuple) and len(out_ids) > 0:
                # If it's a tuple, check if first element has sequences
                first_element = out_ids[0]
                if hasattr(first_element, 'sequences'):
                    generated_sequences = first_element.sequences
                elif isinstance(first_element, torch.Tensor):
                    generated_sequences = first_element
                else:
                    # Fallback: try to extract tensor from nested structure
                    generated_sequences = first_element
                    while isinstance(generated_sequences, tuple) and len(generated_sequences) > 0:
                        generated_sequences = generated_sequences[0]
            elif isinstance(out_ids, torch.Tensor):
                generated_sequences = out_ids
            else:
                # Last resort: try to access sequences directly
                generated_sequences = out_ids.sequences if hasattr(out_ids, 'sequences') else out_ids
            
            # Get input length
            input_ids_tensor = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
            input_length = input_ids_tensor.shape[1]
            
            # Extract generated tokens (excluding input tokens)
            generated_tokens = generated_sequences[:, input_length:]
            
            # Decode only the generated tokens (not the full sequence including input)
            resp = processor.batch_decode(
                generated_tokens, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            
            if "assistant\n" in resp:
                
                assistant_start = resp.rfind("assistant\n") + len("assistant\n")
                resp = resp[assistant_start:].strip()
            
            # Store original response for debugging
            original_resp = resp[:300] if len(resp) > 300 else resp
            hypothesis = clean_response(resp)
            
            # Calculate decode token count (generated tokens only, excluding input)
            generated_length = generated_sequences.shape[1]
            decode_token_count = generated_length - input_length
            
            # Calculate TPS (Tokens Per Second) using decode time
            sample_tps = decode_token_count / decode_time if decode_time > 0 else 0.0
            
            # Calculate text lengths
            hypothesis_char_len = len(hypothesis)
            hypothesis_word_len = len(hypothesis.split()) if hypothesis else 0
            reference_char_len = len(reference)
            reference_word_len = len(reference.split()) if reference else 0
            
            # Print inference statistics with TPS
            print(f"[Sample {idx+1}] Generated tokens: {decode_token_count}, "
                  f"Prefill time: {prefill_time:.4f}s, Decode time: {decode_time:.4f}s, "
                  f"Total time: {total_time:.4f}s, TPS: {sample_tps:.2f}, "
                  f"Hypothesis: {hypothesis_char_len} chars / {hypothesis_word_len} words, "
                  f"Ground truth: {reference_char_len} chars / {reference_word_len} words")
            
            cer = calculate_cer(reference, hypothesis)

            total_cer += cer
            processed_samples += 1

            current_avg_cer = total_cer / processed_samples
            current_avg_accuracy = 100.0 - current_avg_cer  # Character Accuracy = 100 - CER
            
            # ===== 实时输出 CER 和平均 CER =====
            # 在每个样本处理后立即输出，格式清晰醒目
            print(f"\n{'='*80}")
            print(f"[Sample {idx+1}/{len(dataset)}] ID={doc.get('id', f'sample_{idx}')} | "
                  f"Speaker={speaker_id} | Duration={doc.get('duration', 0):.1f}s")
            print(f"{'='*80}")
            print(f"  📊 实时 CER (当前样本): {cer:.2f}%")
            print(f"  📈 平均 CER (已处理 {processed_samples} 个样本): {current_avg_cer:.2f}%")
            print(f"  ✅ 平均准确率: {current_avg_accuracy:.2f}%")
            print(f"{'='*80}")
            
            # Print reference and hypothesis (truncated if too long)
            ref_display = reference[:100] + "..." if len(reference) > 100 else reference
            hyp_display = hypothesis[:100] + "..." if len(hypothesis) > 100 else hypothesis
            print(f"  📝 参考文本:  {ref_display}")
            print(f"  💬 识别结果: {hyp_display}")
            print(f"{'='*80}\n")
            
            # 更新进度条，始终显示最新的 CER 信息
            update_interval = 10 if is_screen_env else 1
            sample_count = idx + 1
            
            # 进度条始终显示最新的 CER 和平均 CER
            progress_bar.set_postfix({
                '实时CER': f'{cer:.2f}%',
                '平均CER': f'{current_avg_cer:.2f}%',
                '准确率': f'{current_avg_accuracy:.2f}%',
                '样本': f'{sample_count}/{len(dataset)}'
            })
            
            if sample_count % update_interval == 0 or sample_count == len(dataset):
                if is_screen_env:
                    # 在非交互式环境中，额外输出进度信息
                    print(f"  📊 进度更新: {sample_count}/{len(dataset)} ({sample_count/len(dataset)*100:.1f}%)")
                    print(f"  📈 当前平均 CER: {current_avg_cer:.2f}% | 平均准确率: {current_avg_accuracy:.2f}%")

            
            results.append({
                "idx": idx,
                "id": doc.get("id", f"sample_{idx}"),
                "speaker_id": speaker_id,
                "chapter_id": doc.get("chapter_id", ""),
                "path": doc.get("path", ""),
                "duration": doc.get("duration", 0),
                "reference": reference,
                "hypothesis": hypothesis,
                "cer": cer,
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
            
            # Real-time save: Update JSON file after each sample (or every N samples for performance)
            save_interval = 1  # Save after every sample for real-time monitoring
            if (idx + 1) % save_interval == 0 or (idx + 1) == len(dataset):
                save_results_realtime(
                    output_file, 
                    results, 
                    total_cer, 
                    processed_samples,
                    speaker_stats,
                    timing_stats,
                    gpu_id,
                    prune_ratio,
                    method_is,
                    sample_limit,
                    kespeech_path,
                    len(dataset)
                )

            
            # CRITICAL: Clear cache and reset state after each sample to prevent memory leaks
            # This is especially important for SparseMM which maintains complex state
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

    
    final_cer = total_cer / processed_samples if processed_samples > 0 else 100.0

    
    speaker_cer = defaultdict(list)
    for result in results:
        speaker_cer[result["speaker_id"]].append(result["cer"])
    
    speaker_avg_cer = {}
    for speaker_id, cers in speaker_cer.items():
        speaker_avg_cer[speaker_id] = sum(cers) / len(cers)

    
    references = [r["reference"] for r in results]
    hypotheses = [r["hypothesis"] for r in results]
    
    
    detailed_metrics = calculate_kespeech_metrics(references, hypotheses)

    
    final_results = {
        "model_name": f"Qwen2.5-Omni-3B",
        "dataset": "KeSpeech",
        "status": "completed",
        "progress": {
            "processed": len(results),
            "total": len(dataset),
            "percentage": 100.0
        },
        "total_samples": len(results),
        "valid_samples": detailed_metrics['valid_samples'],
        
        
        "cer_mean": detailed_metrics['cer_mean'],
        "char_accuracy": detailed_metrics['char_accuracy'],
        "cer_std": detailed_metrics['cer_std'],
        "cer_min": detailed_metrics['cer_min'],
        "cer_max": detailed_metrics['cer_max'],
        "perfect_predictions": detailed_metrics['perfect_predictions'],
        
        
        "speaker_count": len(speaker_stats),
        "speaker_cer": speaker_avg_cer,
        
        
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
            "data_path": kespeech_path
        },
        
        
        "detailed_results": results
    }
    
    
    final_results = convert_numpy_types(final_results)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    
    timing_stats.export_to_json(timing_output_file)

    
    print(f"\n=== KeSpeech ASR Evaluation Results Summary (Qwen3-Omni) ===")
    print(f"Total samples: {len(results)}")
    print(f"Processed samples: {processed_samples}")
    print(f"Average CER: {detailed_metrics['cer_mean']:.2f}%")
    print(f"Character accuracy: {detailed_metrics['char_accuracy']:.2f}%")
    print(f"Number of speakers: {len(speaker_stats)}")
    
    
    print(f"\n=== CER Statistics ===")
    print(f"CER standard deviation: {detailed_metrics['cer_std']:.2f}%")
    print(f"Minimum CER: {detailed_metrics['cer_min']:.2f}%")
    print(f"Maximum CER: {detailed_metrics['cer_max']:.2f}%")
    print(f"Perfect predictions: {detailed_metrics['perfect_predictions']} ({detailed_metrics['perfect_predictions']/detailed_metrics['valid_samples']*100:.1f}%)")
    
    print(f"\nTop 5 speakers by CER:")
    for speaker_id, avg_cer in list(speaker_avg_cer.items())[:5]:
        sample_count = speaker_stats[speaker_id]
        print(f"  {speaker_id}: {avg_cer:.2f}% ({sample_count} samples)")
    
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
    