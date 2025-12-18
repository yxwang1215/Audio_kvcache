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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import random
import jiwer
import argparse
from typing import Dict, Any


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
        from modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
        print("Using standard version of the model from local file (sparsekv disabled)")

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
        # Use absolute budget value, but still need to update it for AdaKV and other methods
        actual_budget = int(ENV_BUDGET)
    else:
        # Calculate budget as ratio of prefill tokens
        budget_ratio = ENV_BUDGET
        actual_budget = int(prefill_token_count * budget_ratio)
    
    # Ensure budget is at least window_size to avoid errors
    min_budget = ENV_WINDOW_SIZE + 1
    if actual_budget < min_budget:
        actual_budget = min_budget
        if ENV_BUDGET_IS_RATIO:
            print(f"[SparseMM] Warning: Calculated budget {int(prefill_token_count * ENV_BUDGET)} is too small, using minimum {min_budget}")
    
    # Update budget in all layers
    base_model = model.thinker.model
    if hasattr(base_model, 'layers'):
        for layer in base_model.layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'kv_cluster'):
                kv_cluster = layer.self_attn.kv_cluster
                # Use the new update_budget method to properly recalculate head_adaptive_capacity
                if hasattr(kv_cluster, 'update_budget'):
                    kv_cluster.update_budget(actual_budget)
                elif hasattr(kv_cluster, 'max_capacity_prompt'):
                    # For SnapKVCluster (used by pyramidkv, snapkv, and mixkv)
                    kv_cluster.max_capacity_prompt = actual_budget
                    # Reset pyram_init so pyramid mode recalculates with new budget
                    if hasattr(kv_cluster, 'pyram_mode') and kv_cluster.pyram_mode:
                        kv_cluster.pyram_init = False
                elif hasattr(kv_cluster, 'base_capacity'):
                    # For AdaKVCluster and SparseMM (they use base_capacity)
                    kv_cluster.base_capacity = actual_budget - kv_cluster.window_size
                    if kv_cluster.base_capacity <= 0:
                        kv_cluster.base_capacity = 1
                    # For AdaKV, recompute floor_capacity and adaptive_capacity if they exist
                    if hasattr(kv_cluster, 'floor_ratio'):
                        kv_cluster.floor_capacity = int(kv_cluster.base_capacity * kv_cluster.floor_ratio)
                        kv_cluster.adaptive_capacity = kv_cluster.base_capacity - kv_cluster.floor_capacity
                    # For SparseMM, recompute head_adaptive_capacity if method exists
                    if hasattr(kv_cluster, '_compute_head_adaptive_capacity'):
                        kv_cluster._compute_head_adaptive_capacity()
        # Also update model config for consistency
        if hasattr(base_model, 'config'):
            base_model.config.max_capacity_prompt = actual_budget
    
    if ENV_BUDGET_IS_RATIO:
        print(f"[SparseMM] Updated budget: {prefill_token_count} tokens * {ENV_BUDGET:.2%} = {actual_budget} (base_capacity={actual_budget - ENV_WINDOW_SIZE})")
    else:
        print(f"[SparseMM] Updated budget: {actual_budget} (absolute value, base_capacity={actual_budget - ENV_WINDOW_SIZE})")
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

def load_wavcaps_audioset_dataset(split="test", local_path=""):
    """Load wavcaps-audioset dataset using UltraEval-Audio registry system
    
    Args:
        split: Dataset split to load (default: "test")
        local_path: Local path to UltraEval-Audio root directory (optional)
    
    Returns:
        List of dataset samples with audio and caption fields
    """
    import glob
    import traceback
    
    # UltraEval-Audio root directory
    ultraeval_root = local_path if local_path else "/mnt/users/wangyuxuan-20250915/Audio/UltraEval-Audio"
    
    # Try to find UltraEval-Audio root
    possible_paths = [
        ultraeval_root,
        os.environ.get("ULTRAEVAL_ROOT", ""),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../UltraEval-Audio"),
        "/mnt/users/wangyuxuan-20250915/Audio/UltraEval-Audio"
    ]
    
    ultraeval_root = None
    for path in possible_paths:
        if path and os.path.exists(path) and os.path.exists(os.path.join(path, "audio_evals")):
            ultraeval_root = path
            break
    
    if not ultraeval_root:
        print("Error: UltraEval-Audio root directory not found. Please set ULTRAEVAL_ROOT environment variable or provide local_path.")
        print("Attempted paths:", possible_paths)
        return []
    
    # Check if local audio files exist
    default_test_dir = os.path.join(ultraeval_root, "raw", "TwinkStart", "wavcaps-audioset", split)
    local_audio_files = []
    if os.path.exists(default_test_dir):
        local_audio_files = sorted(glob.glob(os.path.join(default_test_dir, "*.wav")), 
                                  key=lambda x: int(os.path.splitext(os.path.basename(x))[0]) if os.path.splitext(os.path.basename(x))[0].isdigit() else float('inf'))
        print(f"Found {len(local_audio_files)} local audio files in {default_test_dir}")
    
    # Add UltraEval-Audio to Python path
    if ultraeval_root not in sys.path:
        sys.path.insert(0, ultraeval_root)
    
    # Default download directory
    default_download_dir = "/mnt/users/wangyuxuan-20250915/Audio/Datasets"
    download_base_dir = os.path.join(default_download_dir, "TwinkStart", "wavcaps-audioset")
    
    dataset_data = None
    
    # Try to load from HuggingFace cache first (without downloading)
    try:
        from datasets import load_dataset
        print(f"Attempting to load wavcaps-audioset from HuggingFace cache (no download)...")
        hf_dataset = load_dataset("TwinkStart/wavcaps-audioset", split=split, download_mode="reuse_cache_if_exists")
        print(f"Successfully loaded {len(hf_dataset)} samples from HuggingFace cache")
        
        # Convert to list format
        dataset_data = []
        for idx, item in enumerate(hf_dataset):
            sample = {
                'id': item.get('id', f"audioset_{idx}"),
                'caption': item.get('caption', item.get('text', '')),
                'WavPath': None
            }
            # Try to map to local audio file
            if local_audio_files and idx < len(local_audio_files):
                sample['WavPath'] = local_audio_files[idx]
            elif 'audio' in item and isinstance(item['audio'], dict):
                sample['WavPath'] = item['audio'].get('path')
            dataset_data.append(sample)
            
    except Exception as e:
        print(f"Could not load from HuggingFace cache: {e}")
        print(f"Dataset not found in cache. Attempting to download to {download_base_dir}...")
        # Try to download to the specified directory
        try:
            os.makedirs(download_base_dir, exist_ok=True)
            print(f"Downloading wavcaps-audioset dataset to {download_base_dir}...")
            hf_dataset = load_dataset(
                "TwinkStart/wavcaps-audioset",
                split=split,
                cache_dir=download_base_dir,
                download_mode="force_redownload" if os.path.exists(download_base_dir) else None
            )
            print(f"Successfully downloaded {len(hf_dataset)} samples to {download_base_dir}")
            
            # Convert to list format
            dataset_data = []
            for idx, item in enumerate(hf_dataset):
                sample = {
                    'id': item.get('id', f"audioset_{idx}"),
                    'caption': item.get('caption', item.get('text', '')),
                    'WavPath': None
                }
                # Try to map to local audio file
                if local_audio_files and idx < len(local_audio_files):
                    sample['WavPath'] = local_audio_files[idx]
                elif 'audio' in item and isinstance(item['audio'], dict):
                    sample['WavPath'] = item['audio'].get('path')
                dataset_data.append(sample)
        except Exception as download_e:
            print(f"Failed to download dataset: {download_e}")
            print("Falling back to loading from registry or local files only...")
            dataset_data = None
    
    # If HuggingFace cache failed, try registry (but this might download)
    if dataset_data is None:
        try:
            from audio_evals.registry import registry
            
            print(f"Loading wavcaps-audioset dataset from UltraEval-Audio registry...")
            print(f"UltraEval-Audio root: {ultraeval_root}")
            
            # Get dataset from registry
            dataset_obj = registry.get_dataset("wavcaps-audioset")
            if dataset_obj is None:
                print(f"Error: Dataset 'wavcaps-audioset' not found in registry")
                # Fall back to local files only
                if not local_audio_files:
                    return []
                dataset_data = []
                for idx, audio_file in enumerate(local_audio_files):
                    dataset_data.append({
                        'id': f"audioset_{idx}",
                        'caption': '',  # No caption available from local files only
                        'WavPath': audio_file
                    })
            else:
                # Try to load with download_mode to avoid downloading
                try:
                    # Modify the dataset object to use local_path if available
                    # Check if there's a saved dataset
                    saved_dataset_path = os.path.join(ultraeval_root, "raw", "TwinkStart", "wavcaps-audioset", "saved_dataset")
                    if os.path.exists(saved_dataset_path):
                        from datasets import load_from_disk
                        print(f"Found saved dataset at {saved_dataset_path}, loading...")
                        saved_dataset = load_from_disk(saved_dataset_path)
                        if split in saved_dataset:
                            hf_dataset = saved_dataset[split]
                            dataset_data = []
                            for idx, item in enumerate(hf_dataset):
                                sample = {
                                    'id': item.get('id', f"audioset_{idx}"),
                                    'caption': item.get('caption', item.get('text', '')),
                                    'WavPath': None
                                }
                                if local_audio_files and idx < len(local_audio_files):
                                    sample['WavPath'] = local_audio_files[idx]
                                elif 'WavPath' in item:
                                    sample['WavPath'] = item['WavPath']
                                dataset_data.append(sample)
                        else:
                            raise ValueError(f"Split {split} not found in saved dataset")
                    else:
                        # Last resort: try to load from registry (might download)
                        print("Warning: Attempting to load from registry, this may download data if not in cache...")
                        dataset_data = dataset_obj.load(limit=0)
                except Exception as e:
                    print(f"Error loading from registry: {e}")
                    traceback.print_exc()
                    # Fall back to local files only
                    if not local_audio_files:
                        return []
                    dataset_data = []
                    for idx, audio_file in enumerate(local_audio_files):
                        dataset_data.append({
                            'id': f"audioset_{idx}",
                            'caption': '',  # No caption available from local files only
                            'WavPath': audio_file
                        })
        except Exception as e:
            print(f"Error accessing registry: {e}")
            traceback.print_exc()
            # Fall back to local files only
            if not local_audio_files:
                return []
            dataset_data = []
            for idx, audio_file in enumerate(local_audio_files):
                dataset_data.append({
                    'id': f"audioset_{idx}",
                    'caption': '',  # No caption available from local files only
                    'WavPath': audio_file
                })
    
    if not dataset_data:
        print("Error: No dataset data loaded")
        return []
    
    print(f"Loaded {len(dataset_data)} samples")
    
    # Convert UltraEval format to our format
    dataset = []
    ultraeval_root_for_path = ultraeval_root
    
    # Get ref_col from dataset_obj if available, otherwise default to 'caption'
    try:
        from audio_evals.registry import registry
        dataset_obj = registry.get_dataset("wavcaps-audioset")
        ref_col = dataset_obj.ref_col if dataset_obj else 'caption'
    except:
        ref_col = 'caption'
    
    for idx, sample in enumerate(dataset_data):
        # Get audio path from UltraEval format
        audio_path = None
        if 'WavPath' in sample:
            audio_path = sample['WavPath']
        elif 'audio' in sample and isinstance(sample['audio'], dict):
            audio_path = sample['audio'].get('path') or sample['audio'].get('WavPath')
        elif 'audio_path' in sample:
            audio_path = sample['audio_path']
        
        # Convert relative path to absolute path if needed
        if audio_path:
            if not os.path.isabs(audio_path):
                # Try multiple possible base directories
                possible_bases = [
                    ultraeval_root_for_path,
                    os.getcwd(),
                    os.path.dirname(os.path.abspath(__file__)),
                ]
                
                resolved = False
                for base_dir in possible_bases:
                    if not base_dir:
                        continue
                    abs_path = os.path.normpath(os.path.join(base_dir, audio_path))
                    if os.path.exists(abs_path):
                        audio_path = abs_path
                        resolved = True
                        break
                
                if not resolved:
                    # If still not resolved, try the default test directory
                    default_test_dir = os.path.join(ultraeval_root_for_path, "raw", "TwinkStart", "wavcaps-audioset", split)
                    if os.path.exists(default_test_dir):
                        # Try index-based naming
                        index_based_path = os.path.join(default_test_dir, f"{idx}.wav")
                        if os.path.exists(index_based_path):
                            audio_path = index_based_path
                        else:
                            print(f"Warning: Sample {idx} audio path not found: {audio_path}, trying index-based: {index_based_path}")
            else:
                # Already absolute path, just check if it exists
                if not os.path.exists(audio_path):
                    # Try index-based path in default test directory
                    default_test_dir = os.path.join(ultraeval_root_for_path, "raw", "TwinkStart", "wavcaps-audioset", split)
                    index_based_path = os.path.join(default_test_dir, f"{idx}.wav")
                    if os.path.exists(index_based_path):
                        audio_path = index_based_path
        
        # If still no valid audio path, try default test directory with index
        if not audio_path or not os.path.exists(audio_path):
            default_test_dir = os.path.join(ultraeval_root_for_path, "raw", "TwinkStart", "wavcaps-audioset", split)
            index_based_path = os.path.join(default_test_dir, f"{idx}.wav")
            if os.path.exists(index_based_path):
                audio_path = index_based_path
            else:
                print(f"Warning: Sample {idx} missing or invalid audio path, skipping")
                continue
        
        # Get caption (reference text) from UltraEval format
        # UltraEval uses ref_col which is 'caption' for wavcaps-audioset
        caption = sample.get(ref_col, sample.get('caption', sample.get('text', sample.get('transcription', ''))))
        
        # Calculate duration
        duration = 0.0
        try:
            audio_info = sf.info(audio_path)
            duration = audio_info.duration
        except:
            pass
        
        # Load audio array (optional, for compatibility)
        audio_array = None
        sampling_rate = 16000
        try:
            audio_array, sampling_rate = sf.read(audio_path)
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            audio_array = audio_array.astype(np.float32)
        except Exception as e:
            print(f"Warning: Failed to load audio array for sample {idx}: {e}")
        
        # Create sample entry compatible with existing code structure
        sample_entry = {
            "id": sample.get("id", f"audioset_{idx}"),
            "speaker_id": "unknown",  # wavcaps-audioset doesn't have speaker info
            "chapter_id": "",  # wavcaps-audioset doesn't have chapter info
            "path": audio_path,
            "caption": caption,  # Use caption as reference for caption evaluation
            "transcription": caption,  # Keep for compatibility
            "duration": duration,
            "audio": {
                "path": audio_path,
                "array": audio_array,  # Store audio array if loaded
                "sampling_rate": sampling_rate
            }
        }
        
        dataset.append(sample_entry)
    
    print(f"Successfully loaded {len(dataset)} wavcaps-audioset samples")
    return dataset

def prepare_audio_for_qwen_omni(audio_path, target_sr=16000):
    """Process audio files according to Qwen2.5-Omni requirements"""
    
    try:
        
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            print(f"Successfully loaded with librosa: shape={audio.shape}, sample rate={sr}Hz")
        except Exception as e:
            print(f"librosa loading failed: {e}")
            
            
            try:
                audio, sample_rate = sf.read(audio_path)
                
                
                if len(audio.shape) > 1 and audio.shape[1] > 1:
                    audio = np.mean(audio, axis=1)
                
                
                if sample_rate != target_sr:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * target_sr / sample_rate))
                    
                audio = audio.astype(np.float32)
                sr = target_sr
                print(f"soundfile processing successful: shape={audio.shape}, sample rate={sr}Hz")
                
            except Exception as e:
                print(f"soundfile loading also failed: {e}")
                
                audio = np.zeros(target_sr * 3, dtype=np.float32)
                sr = target_sr
                print("Generating silent replacement audio")
        
        audio = audio.astype(np.float32)
        
        return audio
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        traceback.print_exc()
        silence = np.zeros(target_sr * 3, dtype=np.float32)
        return silence

def wavcaps_doc_to_audio(doc):
    """Load audio data from wavcaps documents (supports both old and UltraEval formats)"""
    # Try to get audio path from different possible locations
    audio_path = None
    audio_array = None
    sampling_rate = 16000
    
    # Check if audio array is already loaded (old format)
    if "audio" in doc and isinstance(doc["audio"], dict):
        if doc["audio"].get("array") is not None:
            return doc["audio"]["array"], doc["audio"].get("sampling_rate", 16000)
        audio_path = doc["audio"].get("path")
        if audio_path:
            sampling_rate = doc["audio"].get("sampling_rate", 16000)
    
    # Check direct path field (UltraEval format or our converted format)
    if not audio_path and "path" in doc:
        audio_path = doc["path"]
    
    # Check WavPath field (UltraEval format)
    if not audio_path and "WavPath" in doc:
        audio_path = doc["WavPath"]
    
    # If we have a path, load the audio
    if audio_path:
        try:
            audio_data = prepare_audio_for_qwen_omni(audio_path)
            # Cache the loaded audio in doc for future use
            if "audio" not in doc:
                doc["audio"] = {}
            doc["audio"]["array"] = audio_data
            doc["audio"]["path"] = audio_path
            doc["audio"]["sampling_rate"] = sampling_rate
            return audio_data, sampling_rate
        except Exception as e:
            print(f"Unable to load audio file {audio_path}: {e}")
            return None
    
    # If no audio found, return None
    return None

def caption_doc_to_text(doc, kwargs=None):
    """Generate prompts for audio captioning tasks"""
    if kwargs is None:
        kwargs = {}
    
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    
    
    instruction = "Please describe the audio content in detail. Provide a clear and accurate caption of what you hear."
    
    
    prompt_text = f"{pre_prompt}{instruction}{post_prompt}"
    
    return prompt_text

def clean_response(response):
    """Clean caption response"""
    if not response or response.strip() == "":
        return ""
    
    
    for marker in ["caption:", "description:", "audio content:", "content:"]:
        if marker.lower() in response.lower():
            parts = re.split(re.escape(marker), response, flags=re.IGNORECASE)
            if len(parts) > 1:
                response = parts[1].strip()
                break
    
    
    response = re.sub(r'<sep>.*?($|<|$)', '', response)
    response = re.sub(r'(?i)^(caption|description|audio\s+content|content):\s*', '', response.strip())
    
    return response.strip()

def calculate_caption_metrics(reference, hypothesis):
    """Calculate COCO caption metrics (BLEU, METEOR, ROUGE, CIDEr, SPICE)
    
    Args:
        reference: Reference caption (ground truth)
        hypothesis: Predicted caption
        
    Returns:
        dict: Dictionary containing caption metrics
    """
    try:
        # Try to import COCO evaluation tools
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.meteor.meteor import Meteor
            from pycocoevalcap.rouge.rouge import Rouge
            from pycocoevalcap.cider.cider import Cider
            from pycocoevalcap.spice.spice import Spice
            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
            coco_available = True
        except ImportError:
            coco_available = False
            print("Warning: pycocoevalcap not available, using simplified metrics")
        
        if not coco_available:
            # Fallback to simple BLEU using sacrebleu if available
            try:
                import sacrebleu
                bleu_score = sacrebleu.corpus_bleu([hypothesis], [[reference]]).score
                return {
                    'Bleu_1': bleu_score,
                    'Bleu_2': bleu_score,
                    'Bleu_3': bleu_score,
                    'Bleu_4': bleu_score,
                    'METEOR': 0.0,
                    'ROUGE_L': 0.0,
                    'CIDEr': 0.0,
                    'SPICE': 0.0
                }
            except ImportError:
                # Ultimate fallback: return zeros
                return {
                    'Bleu_1': 0.0,
                    'Bleu_2': 0.0,
                    'Bleu_3': 0.0,
                    'Bleu_4': 0.0,
                    'METEOR': 0.0,
                    'ROUGE_L': 0.0,
                    'CIDEr': 0.0,
                    'SPICE': 0.0
                }
        
        # Format data for COCO evaluation
        # COCO evaluator expects: gts = {id: [{'caption': text}, ...]}, res = {id: [{'caption': text}]}
        # Each caption must be a dictionary with 'caption' key, not a string
        gts = {0: [{'caption': reference}]}
        res = {0: [{'caption': hypothesis}]}
        
        # Tokenize
        tokenizer = PTBTokenizer()
        gts_tokenized = tokenizer.tokenize(gts)
        res_tokenized = tokenizer.tokenize(res)
        
        # Calculate metrics
        metrics = {}
        
        # BLEU
        try:
            scorer = Bleu(4)
            score, scores = scorer.compute_score(gts_tokenized, res_tokenized)
            metrics['Bleu_1'] = score[0] * 100
            metrics['Bleu_2'] = score[1] * 100
            metrics['Bleu_3'] = score[2] * 100
            metrics['Bleu_4'] = score[3] * 100
        except Exception as e:
            print(f"Warning: BLEU calculation failed: {e}")
            metrics['Bleu_1'] = metrics['Bleu_2'] = metrics['Bleu_3'] = metrics['Bleu_4'] = 0.0
        
        # METEOR
        try:
            scorer = Meteor()
            score, scores = scorer.compute_score(gts_tokenized, res_tokenized)
            metrics['METEOR'] = score * 100
        except Exception as e:
            print(f"Warning: METEOR calculation failed: {e}")
            metrics['METEOR'] = 0.0
        
        # ROUGE
        try:
            scorer = Rouge()
            score, scores = scorer.compute_score(gts_tokenized, res_tokenized)
            metrics['ROUGE_L'] = score * 100
        except Exception as e:
            print(f"Warning: ROUGE calculation failed: {e}")
            metrics['ROUGE_L'] = 0.0
        
        # CIDEr
        try:
            scorer = Cider()
            score, scores = scorer.compute_score(gts_tokenized, res_tokenized)
            metrics['CIDEr'] = score * 100
        except Exception as e:
            print(f"Warning: CIDEr calculation failed: {e}")
            metrics['CIDEr'] = 0.0
        
        # SPICE
        try:
            scorer = Spice()
            score, scores = scorer.compute_score(gts_tokenized, res_tokenized)
            metrics['SPICE'] = score * 100
        except Exception as e:
            print(f"Warning: SPICE calculation failed: {e}")
            metrics['SPICE'] = 0.0
        
        # Calculate SPIDEr (average of CIDEr and SPICE)
        if 'CIDEr' in metrics and 'SPICE' in metrics:
            metrics['SPIDEr'] = (metrics['CIDEr'] + metrics['SPICE']) / 2.0
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating caption metrics: {e}")
        traceback.print_exc()
        return {
            'Bleu_1': 0.0,
            'Bleu_2': 0.0,
            'Bleu_3': 0.0,
            'Bleu_4': 0.0,
            'METEOR': 0.0,
            'ROUGE_L': 0.0,
            'CIDEr': 0.0,
            'SPICE': 0.0,
            'SPIDEr': 0.0
        }


def save_results_realtime(output_file, results, processed_samples, speaker_stats, 
                          timing_stats, gpu_id, prune_ratio, method_is, 
                          sample_limit, dataset_path, dataset_size):
    """
    Save results to JSON file in real-time, updating statistics as we go
    This allows monitoring progress and viewing results before completion
    """
    try:
        # Calculate detailed metrics from current results
        references = [r["reference"] for r in results]
        hypotheses = [r["hypothesis"] for r in results]
        
        # Calculate caption metrics
        current_metrics = calculate_wavcaps_metrics(references, hypotheses)
        
        # Build real-time results structure
        realtime_results = {
            "model_name": f"Qwen2.5-Omni-3B",
            "dataset": "wavcaps-audioset",
            "status": "in_progress",
            "progress": {
                "processed": len(results),
                "total": dataset_size,
                "percentage": len(results) / dataset_size * 100 if dataset_size > 0 else 0
            },
            "current_statistics": {
                "Bleu_1": current_metrics.get('Bleu_1', 0.0),
                "Bleu_2": current_metrics.get('Bleu_2', 0.0),
                "Bleu_3": current_metrics.get('Bleu_3', 0.0),
                "Bleu_4": current_metrics.get('Bleu_4', 0.0),
                "METEOR": current_metrics.get('METEOR', 0.0),
                "ROUGE_L": current_metrics.get('ROUGE_L', 0.0),
                "CIDEr": current_metrics.get('CIDEr', 0.0),
                "SPICE": current_metrics.get('SPICE', 0.0),
                "SPIDEr": current_metrics.get('SPIDEr', 0.0),
                "valid_samples": len(results)
            },
            "speaker_count": len(speaker_stats),
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

def calculate_wavcaps_metrics(references, hypotheses):
    """
    Calculate wavcaps-audioset caption metrics (COCO metrics)
    
    Args:
        references: List of true captions
        hypotheses: List of predicted captions
        
    Returns:
        dict: Dictionary containing caption metrics
    """
    
    valid_indices = []
    clean_references = []
    clean_hypotheses = []
    all_metrics = []
    
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        if ref and hyp:  
            valid_indices.append(i)
            clean_references.append(ref)
            clean_hypotheses.append(hyp)
            
            # Calculate caption metrics for this pair
            metrics = calculate_caption_metrics(ref, hyp)
            all_metrics.append(metrics)
    
    if len(clean_references) == 0:
        return {
            'Bleu_1': 0.0,
            'Bleu_2': 0.0,
            'Bleu_3': 0.0,
            'Bleu_4': 0.0,
            'METEOR': 0.0,
            'ROUGE_L': 0.0,
            'CIDEr': 0.0,
            'SPICE': 0.0,
            'SPIDEr': 0.0,
            'valid_samples': 0,
            'total_samples': len(references)
        }
    
    # Aggregate metrics across all samples
    aggregated_metrics = {
        'Bleu_1': np.mean([m.get('Bleu_1', 0.0) for m in all_metrics]),
        'Bleu_2': np.mean([m.get('Bleu_2', 0.0) for m in all_metrics]),
        'Bleu_3': np.mean([m.get('Bleu_3', 0.0) for m in all_metrics]),
        'Bleu_4': np.mean([m.get('Bleu_4', 0.0) for m in all_metrics]),
        'METEOR': np.mean([m.get('METEOR', 0.0) for m in all_metrics]),
        'ROUGE_L': np.mean([m.get('ROUGE_L', 0.0) for m in all_metrics]),
        'CIDEr': np.mean([m.get('CIDEr', 0.0) for m in all_metrics]),
        'SPICE': np.mean([m.get('SPICE', 0.0) for m in all_metrics]),
        'SPIDEr': np.mean([m.get('SPIDEr', 0.0) for m in all_metrics]),
        'valid_samples': len(clean_references),
        'total_samples': len(references)
    }
    
    # Convert to float for JSON serialization
    for key in aggregated_metrics:
        if isinstance(aggregated_metrics[key], (np.floating, np.integer)):
            aggregated_metrics[key] = float(aggregated_metrics[key])
    
    return aggregated_metrics

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='wavcaps-audioset Caption Evaluation with Qwen2.5-Omni')
    parser.add_argument('--output-file', '-o', type=str, default=None,
                        help='Output file path for results JSON (overrides --output-dir)')
    parser.add_argument('--output-dir', '-d', type=str, default=None,
                        help='Output directory for results (default: ./wavcaps_audioset_Results or from RESULTS_DIR env)')
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

    
    # UltraEval-Audio root directory (not the dataset directory)
    ultraeval_root = '/mnt/users/wangyuxuan-20250915/Audio/UltraEval-Audio'
    # Dataset path for reference (used in output)
    wavcaps_path = '/mnt/users/wangyuxuan-20250915/Audio/UltraEval-Audio/raw/TwinkStart/wavcaps-audioset/test'
    
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
            result_dir = os.environ.get("RESULTS_DIR", './wavcaps_audioset_Results')
        
        wavcaps_path = os.path.abspath(wavcaps_path)
        result_dir = os.path.abspath(result_dir)
        os.makedirs(result_dir, exist_ok=True)
        output_file = f'{result_dir}/wavcaps_audioset_results_qwen25.json'
    
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

    
    timing_stats = GlobalTimingStats()

    print(f"\n=== wavcaps-audioset Caption Evaluation Configuration (Qwen2.5-Omni) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Pruning ratio: {prune_ratio}")
    print(f"Pruning method: {method_is}")
    print(f"Data path: {wavcaps_path}")
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

    
    print(f"Loading wavcaps-audioset dataset using UltraEval-Audio registry")
    print(f"UltraEval-Audio root: {ultraeval_root}")
    print(f"Dataset audio files: {wavcaps_path}")
    dataset = load_wavcaps_audioset_dataset("test", ultraeval_root)
    
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

    progress_bar = tqdm(enumerate(dataset), total=len(dataset), desc="wavcaps-audioset Caption evaluation (Qwen2.5)", **tqdm_kwargs)

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
            
            audio_data_result = wavcaps_doc_to_audio(doc)
            if audio_data_result is None:
                continue
            
            audio_np, sr = audio_data_result
            
            
            reference = doc.get("caption", doc.get("transcription", ""))
            speaker_id = doc.get("speaker_id", "unknown")
            
            
            prompt_text = caption_doc_to_text(doc)

            
            task_instruction = "You are a helpful assistant that describes audio content. Please listen carefully and provide a detailed caption describing what you hear in the audio."
            full_user_prompt = f"{task_instruction}\n\n{prompt_text}"

            
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
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
            # Update KV compression budget based on prefill token count
            # For ratio budgets, calculate based on prefill tokens; for absolute budgets, use the value directly
            if USE_SPARSEKV and sparsemm_enabled:
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
            
            resp = processor.batch_decode(
                out_ids, 
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
            input_token_count = inputs.input_ids.shape[1] if hasattr(inputs, 'input_ids') else 0
            output_token_count = out_ids.shape[1] if hasattr(out_ids, 'shape') else len(out_ids[0]) if isinstance(out_ids, list) else 0
            decode_token_count = max(0, output_token_count - input_token_count)
            
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
            
            # Calculate caption metrics
            caption_metrics = calculate_caption_metrics(reference, hypothesis)
            processed_samples += 1

            # Calculate current average metrics
            current_references = [r["reference"] for r in results] + [reference]
            current_hypotheses = [r["hypothesis"] for r in results] + [hypothesis]
            current_avg_metrics = calculate_wavcaps_metrics(current_references, current_hypotheses)
            
            # ===== 实时输出 Caption 指标 =====
            # 在每个样本处理后立即输出，格式清晰醒目
            print(f"\n{'='*80}")
            print(f"[Sample {idx+1}/{len(dataset)}] ID={doc.get('id', f'sample_{idx}')} | "
                  f"Speaker={speaker_id} | Duration={doc.get('duration', 0):.1f}s")
            print(f"{'='*80}")
            print(f"  📊 当前样本指标:")
            print(f"     BLEU-4: {caption_metrics.get('Bleu_4', 0.0):.2f} | METEOR: {caption_metrics.get('METEOR', 0.0):.2f} | CIDEr: {caption_metrics.get('CIDEr', 0.0):.2f}")
            print(f"  📈 平均指标 (已处理 {processed_samples} 个样本):")
            print(f"     BLEU-4: {current_avg_metrics.get('Bleu_4', 0.0):.2f} | METEOR: {current_avg_metrics.get('METEOR', 0.0):.2f} | CIDEr: {current_avg_metrics.get('CIDEr', 0.0):.2f}")
            print(f"{'='*80}")
            
            # Print reference and hypothesis (truncated if too long)
            ref_display = reference[:100] + "..." if len(reference) > 100 else reference
            hyp_display = hypothesis[:100] + "..." if len(hypothesis) > 100 else hypothesis
            print(f"  📝 参考文本:  {ref_display}")
            print(f"  💬 生成结果: {hyp_display}")
            print(f"{'='*80}\n")
            
            # 更新进度条，始终显示最新的指标信息
            update_interval = 10 if is_screen_env else 1
            sample_count = idx + 1
            
            # 进度条始终显示最新的指标
            progress_bar.set_postfix({
                'BLEU-4': f'{current_avg_metrics.get("Bleu_4", 0.0):.2f}',
                'METEOR': f'{current_avg_metrics.get("METEOR", 0.0):.2f}',
                'CIDEr': f'{current_avg_metrics.get("CIDEr", 0.0):.2f}',
                '样本': f'{sample_count}/{len(dataset)}'
            })
            
            if sample_count % update_interval == 0 or sample_count == len(dataset):
                if is_screen_env:
                    # 在非交互式环境中，额外输出进度信息
                    print(f"  📊 进度更新: {sample_count}/{len(dataset)} ({sample_count/len(dataset)*100:.1f}%)")
                    print(f"  📈 当前平均 BLEU-4: {current_avg_metrics.get('Bleu_4', 0.0):.2f} | METEOR: {current_avg_metrics.get('METEOR', 0.0):.2f} | CIDEr: {current_avg_metrics.get('CIDEr', 0.0):.2f}")

            
            results.append({
                "idx": idx,
                "id": doc.get("id", f"sample_{idx}"),
                "speaker_id": speaker_id,
                "chapter_id": doc.get("chapter_id", ""),
                "path": doc.get("path", ""),
                "duration": doc.get("duration", 0),
                "reference": reference,
                "hypothesis": hypothesis,
                "caption_metrics": caption_metrics,
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
                    processed_samples,
                    speaker_stats,
                    timing_stats,
                    gpu_id,
                    prune_ratio,
                    method_is,
                    sample_limit,
                    wavcaps_path,
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

    
    references = [r["reference"] for r in results]
    hypotheses = [r["hypothesis"] for r in results]
    
    
    detailed_metrics = calculate_wavcaps_metrics(references, hypotheses)

    
    final_results = {
        "model_name": f"Qwen2.5-Omni-3B",
        "dataset": "wavcaps-audioset",
        "status": "completed",
        "progress": {
            "processed": len(results),
            "total": len(dataset),
            "percentage": 100.0
        },
        "total_samples": len(results),
        "valid_samples": detailed_metrics['valid_samples'],
        
        
        "Bleu_1": detailed_metrics.get('Bleu_1', 0.0),
        "Bleu_2": detailed_metrics.get('Bleu_2', 0.0),
        "Bleu_3": detailed_metrics.get('Bleu_3', 0.0),
        "Bleu_4": detailed_metrics.get('Bleu_4', 0.0),
        "METEOR": detailed_metrics.get('METEOR', 0.0),
        "ROUGE_L": detailed_metrics.get('ROUGE_L', 0.0),
        "CIDEr": detailed_metrics.get('CIDEr', 0.0),
        "SPICE": detailed_metrics.get('SPICE', 0.0),
        "SPIDEr": detailed_metrics.get('SPIDEr', 0.0),
        
        
        "speaker_count": len(speaker_stats),
        
        
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
            "data_path": wavcaps_path
        },
        
        
        "detailed_results": results
    }
    
    
    final_results = convert_numpy_types(final_results)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    
    timing_stats.export_to_json(timing_output_file)

    
    print(f"\n=== wavcaps-audioset Caption Evaluation Results Summary (Qwen2.5-Omni) ===")
    print(f"Total samples: {len(results)}")
    print(f"Processed samples: {processed_samples}")
    print(f"Number of speakers: {len(speaker_stats)}")
    
    
    print(f"\n=== Caption Metrics (COCO) ===")
    print(f"BLEU-1: {detailed_metrics.get('Bleu_1', 0.0):.2f}")
    print(f"BLEU-2: {detailed_metrics.get('Bleu_2', 0.0):.2f}")
    print(f"BLEU-3: {detailed_metrics.get('Bleu_3', 0.0):.2f}")
    print(f"BLEU-4: {detailed_metrics.get('Bleu_4', 0.0):.2f}")
    print(f"METEOR: {detailed_metrics.get('METEOR', 0.0):.2f}")
    print(f"ROUGE-L: {detailed_metrics.get('ROUGE_L', 0.0):.2f}")
    print(f"CIDEr: {detailed_metrics.get('CIDEr', 0.0):.2f}")
    print(f"SPICE: {detailed_metrics.get('SPICE', 0.0):.2f}")
    print(f"SPIDEr: {detailed_metrics.get('SPIDEr', 0.0):.2f}")
    
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
    