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
from datasets import load_dataset
import ast
from copy import deepcopy


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
ENV_HEAD_SCORE_PATH = os.environ.get("HEAD_SCORE_PATH", None)
ENV_HEAD_SCORE_FILE = os.environ.get("HEAD_SCORE_FILE", "aggregated")
ENV_HEAD_SCORE_BATCH = os.environ.get("HEAD_SCORE_BATCH", None)
# Aggregation method for multiple scores in JSON (mean, first, second, max, min, sum)
ENV_HEAD_SCORE_AGGREGATION = os.environ.get("HEAD_SCORE_AGGREGATION", "first")

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

# Safe makedirs function that prevents creating 'raw' folder in code directory
def safe_makedirs(path, exist_ok=False):
    """Create directory, but prevent creating 'raw' folder in code directory"""
    if path:
        abs_path = os.path.abspath(path)
        code_dir = os.path.abspath(current_dir)
        # Check if the path is trying to create 'raw' in code directory
        if abs_path.startswith(code_dir):
            # Get relative path from code directory
            try:
                rel_path = os.path.relpath(abs_path, code_dir)
                # Check if path ends with 'raw' or contains 'raw' as a direct subdirectory
                path_parts = rel_path.split(os.sep)
                if 'raw' in path_parts and path_parts.index('raw') == 0:
                    # Trying to create 'raw' directly in code directory
                    raise ValueError(f"Cannot create 'raw' folder in code directory: {abs_path}")
            except ValueError:
                raise
        os.makedirs(path, exist_ok=exist_ok)

# Import model modules after setting up path
# The modeling_qwen3_omni_moe.py file contains both standard and SparseMM versions
if USE_SPARSEKV:
    # Import from model directory
    from model.modeling_qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration
    print("Using sparsekv version of the model (SparseMM enabled)")
else:
    # Try importing from transformers first, then fallback to local file
    try:
        from transformers import Qwen3OmniMoeForConditionalGeneration
        print("Using standard version of the model from transformers (sparsekv disabled)")
    except ImportError:
        # Fallback to local modeling file
        modeling_path = "/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Modeling"
        if modeling_path not in sys.path:
            sys.path.insert(0, modeling_path)
        from modeling_qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration
        print("Using standard version of the model from local file (sparsekv disabled)")

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
    num_key_value_groups: int = 1,
    aggregation: str = "first"
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
                    Options: "first" (default), "mean", "second", "max", "min", "sum"
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
                            print(f"Warning: Unknown aggregation method '{aggregation}', using 'first'")
                            score = float(score_value[0]) if len(score_value) > 0 else 0.0
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

def load_speech_web_questions_dataset(split="test", twinkstart_path=None):
    """Load speech-web-questions dataset from local path, UltraEval-Audio registry, or HuggingFace directly
    
    Args:
        split: Dataset split to load (default: "test")
        twinkstart_path: Path to TwinkStart root directory (default: /mnt/users/wangyuxuan-20250915/Audio/UltraEval-Audio/raw/TwinkStart)
    """
    print(f"Loading speech-web-questions dataset (split={split})...")
    
    # Default TwinkStart path
    default_twinkstart_path = "/mnt/users/wangyuxuan-20250915/Audio/UltraEval-Audio/raw/TwinkStart"
    if twinkstart_path is None:
        twinkstart_path = default_twinkstart_path
    
    # Try loading from local path first
    dataset = None
    local_dataset_path = os.path.join(twinkstart_path, "speech-web-questions", split)
    if os.path.exists(local_dataset_path):
        try:
            from datasets import load_from_disk
            print(f"Attempting to load speech-web-questions from local path: {local_dataset_path}")
            dataset = load_from_disk(local_dataset_path)
            print(f"Successfully loaded {len(dataset)} samples from local path")
            
            # Convert to list format
            dataset_list = []
            for idx, item in enumerate(dataset):
                # Extract required fields
                audio_data = item.get("audio", {})
                question_text = item.get("QuestionText", item.get("question", ""))
                answer = item.get("answer", item.get("answers", ""))
                
                # Get audio array and sampling rate
                audio_array = audio_data.get("array", None) if isinstance(audio_data, dict) else None
                sampling_rate = audio_data.get("sampling_rate", 16000) if isinstance(audio_data, dict) else 16000
                
                # Calculate duration if audio array is available
                duration = 0.0
                if audio_array is not None:
                    try:
                        duration = len(audio_array) / sampling_rate
                    except:
                        duration = 0.0
                
                dataset_list.append({
                    "id": item.get("id", f"sample_{idx}"),
                    "QuestionText": question_text,
                    "answer": answer,
                    "duration": duration,
                    "audio": {
                        "array": audio_array,
                        "sampling_rate": sampling_rate,
                        "path": None
                    }
                })
            
            print(f"Converted {len(dataset_list)} samples to list format")
            return dataset_list
        except Exception as e:
            print(f"Warning: Failed to load from local path: {e}")
            print("Falling back to UltraEval-Audio registry...")
            dataset = None
    else:
        print(f"Local path not found: {local_dataset_path}, trying UltraEval-Audio registry...")
        dataset = None
    
    # Try UltraEval-Audio registry if local path failed or doesn't exist
    if dataset is None:
        # Try UltraEval-Audio registry first
        ultraeval_root = "/mnt/users/wangyuxuan-20250915/Audio/UltraEval-Audio"
    
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
    
    # Try loading from UltraEval-Audio registry
    if ultraeval_root:
        try:
            # Add UltraEval-Audio to Python path
            if ultraeval_root not in sys.path:
                sys.path.insert(0, ultraeval_root)
            
            from audio_evals.registry import registry
            
            print(f"Loading speech-web-questions dataset from UltraEval-Audio registry...")
            dataset_obj = registry.get_dataset("speech-web-questions")
            if dataset_obj is not None:
                # Load dataset data
                dataset_data = dataset_obj.load(limit=0)
                print(f"Loaded {len(dataset_data)} samples from UltraEval-Audio registry")
                
                # Convert to list format
                dataset_list = []
                for idx, item in enumerate(dataset_data):
                    # Extract required fields
                    audio_data = item.get("audio", {})
                    question_text = item.get("QuestionText", item.get("question", ""))
                    answer = item.get("answer", item.get("answers", ""))
                    
                    # Get audio array and sampling rate
                    audio_array = audio_data.get("array", None) if isinstance(audio_data, dict) else None
                    sampling_rate = audio_data.get("sampling_rate", 16000) if isinstance(audio_data, dict) else 16000
                    
                    # Calculate duration if audio array is available
                    duration = 0.0
                    if audio_array is not None:
                        try:
                            duration = len(audio_array) / sampling_rate
                        except:
                            duration = 0.0
                    
                    dataset_list.append({
                        "id": item.get("id", f"sample_{idx}"),
                        "QuestionText": question_text,
                        "answer": answer,
                        "duration": duration,
                        "audio": {
                            "array": audio_array,
                            "sampling_rate": sampling_rate,
                            "path": None
                        }
                    })
                
                print(f"Converted {len(dataset_list)} samples to list format")
                return dataset_list
        except Exception as e:
            print(f"Warning: Failed to load from UltraEval-Audio registry: {e}")
            print("Falling back to HuggingFace direct loading...")
    
    # Fallback: Load directly from HuggingFace
    try:
        print(f"Loading speech-web-questions dataset directly from HuggingFace: TwinkStart/speech-web-questions")
        dataset = load_dataset("TwinkStart/speech-web-questions", split=split, trust_remote_code=True)
        print(f"Successfully loaded {len(dataset)} samples from HuggingFace")
        
        # Convert to list format
        dataset_list = []
        for idx, item in enumerate(dataset):
            # Extract required fields
            audio_data = item.get("audio", {})
            question_text = item.get("QuestionText", item.get("question", ""))
            answer = item.get("answer", item.get("answers", ""))
            
            # Get audio array and sampling rate
            audio_array = audio_data.get("array", None) if isinstance(audio_data, dict) else None
            sampling_rate = audio_data.get("sampling_rate", 16000) if isinstance(audio_data, dict) else 16000
            
            # Calculate duration if audio array is available
            duration = 0.0
            if audio_array is not None:
                try:
                    duration = len(audio_array) / sampling_rate
                except:
                    duration = 0.0
            
            dataset_list.append({
                "id": item.get("id", f"sample_{idx}"),
                "QuestionText": question_text,
                "answer": answer,
                "duration": duration,
                "audio": {
                    "array": audio_array,
                    "sampling_rate": sampling_rate,
                    "path": None
                }
            })
        
        print(f"Converted {len(dataset_list)} samples to list format")
        return dataset_list
        
    except Exception as e:
        print(f"Error loading speech-web-questions dataset from HuggingFace: {e}")
        traceback.print_exc()
        return []

def prepare_audio_for_qwen_omni(audio_path, target_sr=16000):
    """Process audio files according to Qwen3-Omni requirements"""
    
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

def speech_web_questions_doc_to_audio(doc):
    """Load audio data from speech-web-questions documents"""
    if "audio" not in doc:
        return None
    
    audio_array = doc["audio"].get("array", None)
    sampling_rate = doc["audio"].get("sampling_rate", 16000)
    
    # If audio array is None, try to load from path if available
    if audio_array is None:
        audio_path = doc["audio"].get("path", None)
        if audio_path and os.path.exists(audio_path):
            try:
                audio_array = prepare_audio_for_qwen_omni(audio_path)
                doc["audio"]["array"] = audio_array
                doc["audio"]["sampling_rate"] = 16000
            except Exception as e:
                print(f"Unable to load audio file {audio_path}: {e}")
                return None
        else:
            print(f"Warning: No audio data available for sample {doc.get('id', 'unknown')}")
            return None
    
    # Ensure audio is in the correct format
    if audio_array is not None:
        # Convert to numpy array if needed
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)
        
        # Resample to 16kHz if needed
        if sampling_rate != 16000:
            try:
                audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
                sampling_rate = 16000
            except:
                pass
        
        # Ensure mono channel
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        audio_array = audio_array.astype(np.float32)
    
    return audio_array, sampling_rate

def aqa_doc_to_text(doc, kwargs=None):
    """Generate prompts for AQA (Audio Question Answering) tasks (following UltraEval-Audio format)"""
    if kwargs is None:
        kwargs = {}
    
    # Get question text from document
    question_text = doc.get("QuestionText", doc.get("question", ""))
    
    # AQA prompt template from UltraEval-Audio: "answer the question without explain\n# Question: {{QuestionText}}"
    prompt_text = f"answer the question without explain\n# Question: {question_text}\n\nDo not output phrases like 'If you want to know more about...' or 'feel free to ask' or any similar invitations to ask more questions. Just provide the direct answer."
    
    return prompt_text

def clean_response(response):
    """Clean AQA response (extract answer from model output)"""
    if not response or response.strip() == "":
        return ""
    
    # For AQA, we just need to clean up the response
    # Remove common prefixes that might be added by the model
    response = response.strip()
    
    # Remove any "Answer:" or similar prefixes
    for marker in ["answer:", "answer is:", "the answer is:", "response:"]:
        if marker.lower() in response.lower():
            parts = re.split(re.escape(marker), response, flags=re.IGNORECASE)
            if len(parts) > 1:
                response = parts[1].strip()
                break
    
    return response.strip()

def prefix_match(pred, label, ignore_case=True):
    """
    Prefix match evaluator (following UltraEval-Audio PrefixMatch implementation)
    Returns 1 if prediction prefix matches label, 0 otherwise
    """
    if ignore_case:
        pred = pred.lower().strip()
        label = str(label).lower().strip()
    else:
        pred = pred.strip()
        label = str(label).strip()
    
    n = len(label)
    m = 1 if pred[:n] == label else 0
    
    # If no match and label has spaces but pred doesn't, try without spaces
    if m == 0 and " " in label and " " not in pred:
        label_no_spaces = label.replace(" ", "")
        n = len(label_no_spaces)
        m = 1 if pred[:n] == label_no_spaces else 0
    
    return m


def save_results_realtime(output_file, results, total_correct, processed_samples, 
                          timing_stats, gpu_id, prune_ratio, method_is, 
                          sample_limit, dataset_size):
    """
    Save results to JSON file in real-time, updating statistics as we go
    This allows monitoring progress and viewing results before completion
    """
    try:
        # For inference mode: skip accuracy calculation if total_correct is None
        # Calculate current statistics only if total_correct is provided
        if total_correct is not None and processed_samples > 0:
            current_accuracy = (total_correct / processed_samples * 100.0)
        else:
            current_accuracy = None
        
        # Calculate detailed metrics from current results (only if match field exists)
        match_scores = [r.get("match") for r in results if "match" in r]
        if len(match_scores) > 0:
            accuracy = np.mean(match_scores) * 100.0
            correct_count = int(sum(match_scores))
            total_count = len(match_scores)
        else:
            # Inference mode: no accuracy metrics
            accuracy = None
            correct_count = None
            total_count = None
        
        # Build real-time results structure
        current_statistics = {}
        if accuracy is not None:
            current_statistics["accuracy"] = float(accuracy)
            current_statistics["correct_count"] = int(correct_count)
            current_statistics["total_count"] = int(total_count)
        if current_accuracy is not None:
            current_statistics["current_avg_accuracy"] = float(current_accuracy)
        
        realtime_results = {
            "model_name": f"Qwen3-Omni-3B",
            "dataset": "speech-web-questions",
            "task": "AQA (Audio Question Answering)",
            "status": "in_progress",
            "progress": {
                "processed": len(results),
                "total": dataset_size,
                "percentage": len(results) / dataset_size * 100 if dataset_size > 0 else 0
            },
            "current_statistics": current_statistics if current_statistics else None,
            "timing_stats": timing_stats.get_summary(),
            "config": {
                "device": f"cuda:{gpu_id}",
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "gpu_id": str(os.environ.get('CUDA_VISIBLE_DEVICES', 'default')),
                "pruning_config": {
                    "prune_ratio": prune_ratio, 
                    "prune_method": method_is
                },
                "sample_limit": sample_limit
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

def calculate_speech_web_questions_metrics(results):
    """
    Calculate speech-web-questions AQA metrics (accuracy using prefix match)
    
    Args:
        results: List of result dictionaries with 'match' field
        
    Returns:
        dict: Dictionary containing accuracy-related metrics
    """
    if len(results) == 0:
        return {
            'accuracy': 0.0,
            'correct_count': 0,
            'total_count': 0,
            'valid_samples': 0
        }
    
    match_scores = [r.get("match", 0) for r in results]
    correct_count = sum(match_scores)
    total_count = len(match_scores)
    accuracy = (correct_count / total_count * 100.0) if total_count > 0 else 0.0
    
    return {
        'accuracy': float(accuracy),
        'correct_count': int(correct_count),
        'total_count': int(total_count),
        'valid_samples': int(total_count)
    }

# ==================== Judge Functions ====================
# LLM as Judge prompt for evaluating answer quality against ground truth (reference UltraEval)
# Updated based on GPT-5.1 feedback: balanced strictness and reasonableness
JUDGE_PROMPT = """<|im_start|>system
You are a helpful assistant that evaluates the quality of model-generated answers by comparing them to the provided ground truth reference answer.
<|im_end|>
<|im_start|>user
Please evaluate the following model response against the ground truth reference answer.

Question/Instruction:
{{
    "instruction": ""\"{instruction}\"\"",
}}

Ground Truth Reference Answer:
{{
    "reference": ""\"{reference}\"\"",
}}

Model Output:
{{
    "prediction": ""\"{prediction}\"\"",
}}

Your task is to determine whether the model output correctly answers the question based on the ground truth reference answer.

**CORE EVALUATION PRINCIPLE:**

Score = 1 (CORRECT) if:
- The prediction **explicitly contains the correct answer** (can be synonyms, aliases, or equivalent expressions)
- AND there are **no contradictory factual errors** in the core facts of the answer
- Extra information, verbosity, or conversational style does NOT reduce the score

Score = 0 (INCORRECT) if:
- **Key information is wrong** (wrong person name, place name, country, year, date, etc.)
- The prediction says "I don't know", "I'm not sure", or **asks the user to look it up** (e.g., "check the website", "you might want to check", "look it up")
- The prediction mentions the correct keyword BUT also contains **contradictory core facts** (e.g., saying someone "became VP" when they were already VP and became President)
- **Fails to provide requested specific information**: If the question asks for specific information (e.g., a schedule, a list, dates, times, a complete answer) and the prediction only provides vague, incomplete, or speculative information without the core requested details, score 0.
  - Example: Question asks "what is the 2012 football schedule?", prediction mentions a few possible opponents but provides no actual schedule, dates, or complete list → Score 0
  - Example: Question asks for a specific list or schedule, prediction says "they played some teams" or "check the website" → Score 0
- **Vague or speculative answers**: If the prediction is vague, uncertain, or speculative about facts that should be concrete, score 0.
  - Example: "They had some tough games" or "schedules can change sometimes" when asked for a specific past schedule → Score 0

**DETAILED GUIDELINES:**

**When to give Score = 1 (Be Reasonable, Not Overly Strict):**

1. **Synonyms and Aliases**: If the prediction uses a synonym or alias of the reference answer, score 1.
   - Example: "Ross Bagdasarian Sr." = "David Seville" (same person, different name) → Score 1
   - Example: "French" when reference is "French language" or "French and English" → Score 1

2. **Core Fact Correct**: If the prediction captures the core fact correctly, even if not word-for-word identical, score 1.
   - Example: Question asks what lachrymatory collects, answer says "tears" → Score 1 (core fact correct)
   - Example: Question asks about bufflehead, answer says "a small diving duck" or "a type of duck" → Score 1 (core fact correct)

3. **Partial Matches**: If the reference contains multiple parts but the prediction contains the key correct answer, score 1.
   - Example: Reference is "Challenger Deep, Mariana Trench", prediction says "Mariana Trench" → Score 1
   - Example: Reference is "English and French", prediction says "French" (when question asks "besides English") → Score 1
   - **IMPORTANT for Lists/Multiple Items**: If the reference contains multiple items (e.g., a list of exercises, actors, foods) and the prediction contains **multiple correct items** from the reference, score 1, even if some items are missing.
     - Example: Reference is "walking, swimming, biking, squats, lunges, push-ups, planks, sit-ups" (best exercises for beginners), prediction says "push-ups, squats, lunges, planks, sit-ups" → Score 1 (contains multiple correct answers)
     - Example: Reference lists 10 actors, prediction lists 5 of them correctly → Score 1 (contains multiple correct answers)
     - The key principle: If the prediction contains **multiple correct answers** from a list-type reference, it demonstrates understanding and should score 1, even if not exhaustive.
   - **CRITICAL EXCEPTION**: If the question explicitly asks for a **complete schedule, complete list, or specific structured information** (e.g., "what is the 2012 schedule?", "list all items"), and the prediction only provides vague mentions or partial information without the core requested structure, score 0, even if it mentions some correct items.
     - Example: Question asks "what is the 2012 football schedule?", prediction says "they played some teams like X, Y, Z" without providing actual schedule → Score 0 (missing core requested information: the schedule itself)

4. **Hierarchical Relationships**: If the reference is specific but prediction gives a valid parent/general category, score 1.
   - Example: Reference is "bufflehead", prediction says "duck" or "diving duck" → Score 1

5. **Additional Information**: If the prediction contains the correct answer plus extra correct information, score 1.
   - Example: Reference is "Chicago", prediction says "Chicago, Illinois" → Score 1
   - Verbosity and conversational style are acceptable and do not reduce the score

6. **Case Variations**: Different capitalization, punctuation, or formatting should not affect the score if the core answer is correct.

**When to give Score = 0 (Be Strict on Factual Errors):**

1. **Wrong Key Information**: If critical facts are wrong, score 0.
   - Wrong person: "Elizabeth I" → "Queen Victoria" → Score 0
   - Wrong person: "David Bowie" → "David Essex" → Score 0
   - Wrong person: "Steve Biko" → "Nelson Mandela" → Score 0
   - Wrong place: "Australia" → "United States" → Score 0
   - Wrong place: "Kenya" → "Malaya" → Score 0
   - Wrong place: "Amsterdam" → "Paris" → Score 0
   - Wrong year: "1970" → "1985" → Score 0
   - Wrong entity: "Mallard" → "Bluebell" → Score 0
   - Wrong title: "Live and Let Die" → "The Man with the Golden Gun" → Score 0

2. **Contradictory Core Facts**: If the prediction mentions the correct keyword but states contradictory facts, score 0.
   - Example: Question asks "who was vice president after Kennedy died?", prediction says "Lyndon B. Johnson became the vice president" → Score 0 (he was already VP, became President)

3. **No Answer or Uncertainty**: If the prediction says "I don't know", "I'm not sure", or asks the user to look it up (e.g., "check the website", "you might want to check"), score 0.

4. **Incomplete or Vague Answers for Specific Requests**: If the question asks for specific information (schedule, list, dates, complete answer) and the prediction:
   - Only provides vague mentions without concrete details → Score 0
   - Says "some teams" or "various opponents" without listing them → Score 0
   - Provides speculative information ("they might have played", "probably") → Score 0
   - Redirects the user to external sources instead of answering → Score 0
   - Example: Question asks "what is the 2012 football schedule?", prediction says "they played some teams like X, Y, Z, but check the website for details" → Score 0 (incomplete + redirects)

5. **Fundamentally Wrong**: If the prediction is completely off-topic or fundamentally incorrect, score 0.

**EVALUATION CHECKLIST:**

Before scoring, ask yourself:
1. Does the question ask for **specific information** (schedule, list, dates, complete answer)? 
   - If YES: Does the prediction provide that specific information, or just vague mentions? → If vague/incomplete, Score 0
2. Does the prediction **redirect the user** to external sources instead of answering? → If YES, Score 0
3. Does the prediction contain the correct answer (or its synonym/alias)? → If YES, likely Score 1 (unless it fails checks 1-2)
4. Are there any contradictory factual errors in key information (person, place, year, etc.)? → If YES, Score 0
5. Is the core fact correct even if wording differs? → If YES, Score 1 (unless it fails checks 1-2)
6. Is the prediction just verbose or conversational but factually correct? → If YES, Score 1 (unless it fails checks 1-2)

**Remember:**
- Focus on **core factual correctness**, not exact wording
- Allow synonyms, aliases, and equivalent expressions
- Be strict on **factual errors** (wrong names, places, years)
- Verbosity and extra correct information are acceptable

Please evaluate and return your judgment in the following JSON format:
{{
    "score": <0 or 1>
}}

IMPORTANT: Your response must be valid JSON format (use double quotes, not single quotes) and should contain nothing else. Do not wrap it in code blocks. Return only the JSON object with the score field.
<|im_end|>"""


def repair_and_parse_json(text):
    """Robust JSON parsing function (following alimeeting pattern)"""
    if not text or not text.strip():
        raise ValueError("Empty text")
    
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
    
    text = text.strip()
    
    # Remove redacted reasoning blocks and other noise
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # Handle cases where response contains multiple code blocks or malformed content
    # Try to extract the first valid JSON-like structure
    code_block_patterns = [
        r"```(?:json|python)?\s*(\[.*?\])\s*```",  # List in code block
        r"```(?:json|python)?\s*(\{.*?\})\s*```",  # Dict in code block
        r"```(?:json|python)?\s*(.*?)\s*```",      # Any content in code block
    ]
    
    for pattern in code_block_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            if extracted:
                text = extracted
                break
    
    # If still contains code block markers, try to clean them
    if "```" in text:
        # Remove any remaining code block markers
        text = re.sub(r"```(?:json|python)?", "", text).strip()
        text = re.sub(r"```", "", text).strip()
    
    # Try json.loads first
    try:
        return json.loads(text)
    except:
        pass
    
    # Try ast.literal_eval
    try:
        return ast.literal_eval(text)
    except:
        pass
    
    # Try to extract dict from text (for judge response format: {"score": 0 or 1, "explanation": "..."})
    if text.startswith("{") or "score" in text.lower():
        # First try non-greedy match for dict
        match = re.search(r"(\{.*?\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                try:
                    return ast.literal_eval(match.group(1))
                except:
                    pass
        # If that fails, try greedy match (might capture more, including truncated)
        match = re.search(r"(\{.*)", text, re.DOTALL)
        if match:
            extracted = match.group(1)
            # Try to fix incomplete JSON (handle truncation)
            if not extracted.strip().endswith("}"):
                # Try to complete truncated JSON - simpler approach: just extract score
                if '"score"' in extracted or "'score'" in extracted:
                    # Extract score if available
                    score_match = re.search(r'"score"\s*:\s*(\d+)', extracted)
                    if not score_match:
                        score_match = re.search(r"'score'\s*:\s*(\d+)", extracted)
                    if score_match:
                        try:
                            score_val = int(score_match.group(1))
                            # Return minimal valid JSON with just score (more robust)
                            return {"score": score_val}
                        except:
                            pass
            else:
                # Complete JSON, try to parse
                try:
                    return json.loads(extracted)
                except:
                    try:
                        return ast.literal_eval(extracted)
                    except:
                        pass
    
    # Try to extract list from text (use non-greedy first, then greedy)
    if text.startswith("[") or "model" in text.lower() or "rank" in text.lower():
        # First try non-greedy match
        match = re.search(r"(\[.*?\])", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                try:
                    return ast.literal_eval(match.group(1))
                except:
                    pass
        # If that fails, try greedy match (might capture more)
        match = re.search(r"(\[.*\])", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                try:
                    return ast.literal_eval(match.group(1))
                except:
                    pass
    
    # Try to fix incomplete JSON (handle truncation)
    if text.strip().startswith("[") and not text.strip().endswith("]"):
        # Try various completion strategies
        completion_strategies = [
            text + "]",  # Simple closing bracket
            text + "}]",  # Close dict then list
            text + "}]}]",  # Close two dicts then list
        ]
        
        # If we can see partial model/rank info, try to complete it
        if "'model'" in text or '"model"' in text:
            # Try to extract what we have and complete it
            if "'model'" in text:
                # Python dict format - try to convert to JSON and complete
                try:
                    # Find the last complete or partial entry
                    last_comma = text.rfind(",")
                    if last_comma > 0:
                        # Try to complete the last entry
                        partial = text[:last_comma + 1]
                        # Try to find rank value even if incomplete
                        rank_match = re.search(r"'rank'\s*:\s*(\d+)", text)
                        if rank_match:
                            rank_val = rank_match.group(1)
                            model_match = re.search(r"'model'\s*:\s*['\"]?model_(\d+)['\"]?", text)
                            if model_match:
                                model_num = model_match.group(1)
                                # Try to construct complete JSON
                                completion_strategies.append(
                                    f'[{{"model": "model_{model_num}", "rank": {rank_val}}}]'
                                )
                except:
                    pass
        
        for strategy in completion_strategies:
            try:
                return json.loads(strategy)
            except:
                try:
                    return ast.literal_eval(strategy)
                except:
                    pass
    
    # Last resort: try to extract ranking info via regex (handle both dict and list formats)
    # Try to find model_1 and model_2 ranks (handle both complete and truncated responses)
    
    # More flexible patterns that work even with truncated responses
    model_1_rank_match = re.search(r'"model"\s*:\s*["\']?model_1["\']?\s*[,:}][^}]*"rank"\s*:\s*(\d+)', text)
    if not model_1_rank_match:
        model_1_rank_match = re.search(r"'model'\s*:\s*['\"]?model_1['\"]?\s*[,:}][^}]*'rank'\s*:\s*(\d+)", text)
    if not model_1_rank_match:
        # Try even more flexible pattern for truncated responses
        model_1_rank_match = re.search(r"model_1[^}]*?rank['\"]?\s*:\s*(\d+)", text, re.IGNORECASE)
    if not model_1_rank_match:
        model_1_rank_match = re.search(r"model_1.*?rank.*?(\d+)", text, re.IGNORECASE)
    
    model_2_rank_match = re.search(r'"model"\s*:\s*["\']?model_2["\']?\s*[,:}][^}]*"rank"\s*:\s*(\d+)', text)
    if not model_2_rank_match:
        model_2_rank_match = re.search(r"'model'\s*:\s*['\"]?model_2['\"]?\s*[,:}][^}]*'rank'\s*:\s*(\d+)", text)
    if not model_2_rank_match:
        # Try even more flexible pattern for truncated responses
        model_2_rank_match = re.search(r"model_2[^}]*?rank['\"]?\s*:\s*(\d+)", text, re.IGNORECASE)
    if not model_2_rank_match:
        model_2_rank_match = re.search(r"model_2.*?rank.*?(\d+)", text, re.IGNORECASE)
    
    if model_1_rank_match and model_2_rank_match:
        try:
            return [
                {"model": "model_1", "rank": int(model_1_rank_match.group(1))},
                {"model": "model_2", "rank": int(model_2_rank_match.group(1))}
            ]
        except:
            pass
    
    # Try to extract just the ranks if we can find them (even from truncated responses)
    all_ranks = re.findall(r'"rank"\s*:\s*(\d+)', text)
    if not all_ranks:
        all_ranks = re.findall(r"'rank'\s*:\s*(\d+)", text)
    if not all_ranks:
        # Try even more flexible pattern
        all_ranks = re.findall(r"rank['\"]?\s*:\s*(\d+)", text, re.IGNORECASE)
    
    # If we found ranks, try to match them with models
    if len(all_ranks) >= 2:
        try:
            # Try to determine which rank belongs to which model
            model_1_pos = text.find("model_1")
            model_2_pos = text.find("model_2")
            
            if model_1_pos >= 0 and model_2_pos >= 0:
                # Find ranks near each model
                rank_1_match = re.search(r"model_1[^}]*?rank['\"]?\s*:\s*(\d+)", text, re.IGNORECASE)
                rank_2_match = re.search(r"model_2[^}]*?rank['\"]?\s*:\s*(\d+)", text, re.IGNORECASE)
                
                if rank_1_match and rank_2_match:
                    return [
                        {"model": "model_1", "rank": int(rank_1_match.group(1))},
                        {"model": "model_2", "rank": int(rank_2_match.group(1))}
                    ]
            
            # Fallback: assume first rank is model_1, second is model_2
            return [
                {"model": "model_1", "rank": int(all_ranks[0])},
                {"model": "model_2", "rank": int(all_ranks[1])}
            ]
        except:
            pass
    elif len(all_ranks) == 1:
        # Only found one rank - might be truncated, try to infer the other
        try:
            found_rank = int(all_ranks[0])
            # If we found model_1's rank, assume model_2 has rank 2 (or vice versa)
            if "model_1" in text.lower() and found_rank == 1:
                return [
                    {"model": "model_1", "rank": 1},
                    {"model": "model_2", "rank": 2}
                ]
            elif "model_2" in text.lower() and found_rank == 1:
                return [
                    {"model": "model_1", "rank": 2},
                    {"model": "model_2", "rank": 1}
                ]
        except:
            pass
    
    # Try to extract score from text (for judge response format: {"score": 0 or 1})
    # This is a last resort for truncated responses
    # Use simpler approach like speech-triviaqa: just return score, don't try to extract explanation
    # This is more robust for truncated responses
    score_match = re.search(r'"score"\s*:\s*(\d+)', text)
    if not score_match:
        score_match = re.search(r"'score'\s*:\s*(\d+)", text)
    if score_match:
        try:
            score_val = int(score_match.group(1))
            # Just return score, don't try to extract explanation (more robust for truncated responses)
            return {"score": score_val}
        except:
            pass
    
    preview = text[:300] + "..." + text[-100:] if len(text) > 400 else text
    raise ValueError(f"JSON Parse Failed. Preview:\n{preview}")


def call_api_judge(api_client, model_name, instruction, prediction, reference, max_retries=3):
    """Call API model as judge to evaluate prediction against ground truth reference"""
    for retry in range(max_retries):
        try:
            # Format prompt
            prompt = deepcopy(JUDGE_PROMPT)
            prompt = prompt.replace("{instruction}", instruction)
            prompt = prompt.replace("{prediction}", prediction)
            prompt = prompt.replace("{reference}", reference)
            
            # Convert ChatML format to messages format for OpenAI API
            # Parse <|im_start|>system and <|im_start|>user tags
            messages = []
            if "<|im_start|>system" in prompt:
                parts = prompt.split("<|im_start|>")
                for part in parts:
                    if part.strip().startswith("system"):
                        content = part.split("<|im_end|>")[0].replace("system", "", 1).strip()
                        if content:
                            messages.append({"role": "system", "content": content})
                    elif part.strip().startswith("user"):
                        content = part.split("<|im_end|>")[0].replace("user", "", 1).strip()
                        if content:
                            messages.append({"role": "user", "content": content})
            else:
                # If no ChatML format, treat entire prompt as user message
                messages = [{"role": "user", "content": prompt}]
            
            # Call API directly using OpenAI client (like alimeeting's simple approach)
            try:
                response = api_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=2000  # Increased to prevent truncation of JSON responses
                )
                res = response.choices[0].message.content
                
                # Check if response is None or empty
                if res is None or not res.strip():
                    if retry < max_retries - 1:
                        print(f"Warning: Empty response from API judge (retry {retry + 1}/{max_retries})")
                        print(f"  Model: {model_name}, Response object: {response}")
                        # Increase retry delay for empty responses (might be rate limiting)
                        time.sleep(2 * (retry + 1))  # Exponential backoff: 2s, 4s, 6s
                        continue
                    print(f"Warning: Empty response from API judge after {max_retries} retries")
                    print(f"  Model: {model_name}, Last response object: {response}")
                    return 0
            except Exception as api_error:
                if retry < max_retries - 1:
                    print(f"API call failed (retry {retry + 1}/{max_retries}): {api_error}")
                    # Exponential backoff for API errors
                    time.sleep(2 * (retry + 1))  # 2s, 4s, 6s
                    continue
                print(f"Error calling API judge after {max_retries} retries: {api_error}")
                traceback.print_exc()
                return 0
            
            
            # Use robust parsing function
            try:
                parsed_res = repair_and_parse_json(res)
                
                # Handle new response format: {"score": 0 or 1, "explanation": "..."}
                if isinstance(parsed_res, dict):
                    # Extract score from the response
                    score = parsed_res.get("score")
                    if score is not None:
                        # Ensure score is 0 or 1
                        score = int(score)
                        if score not in [0, 1]:
                            # Normalize: any non-zero becomes 1
                            score = 1 if score > 0 else 0
                        return score
                    # Try alternative keys
                    for key in ["judge_score", "correct", "is_correct", "result"]:
                        if key in parsed_res:
                            val = parsed_res[key]
                            if isinstance(val, (int, float, bool)):
                                return 1 if val else 0
                            elif isinstance(val, str):
                                val_lower = val.lower()
                                if val_lower in ["yes", "true", "correct", "1"]:
                                    return 1
                                elif val_lower in ["no", "false", "incorrect", "0"]:
                                    return 0
                
                # If we got here, parsing succeeded but format is unexpected
                if retry < max_retries - 1:
                    time.sleep(1)
                    continue
                print(f"Warning: Unexpected response format: {parsed_res}")
                return 0
                
            except ValueError as e:
                # Parsing failed, retry if possible
                if retry < max_retries - 1:
                    print(f"Parsing failed (retry {retry + 1}/{max_retries}): {str(e)[:100]}")
                    time.sleep(1)
                    continue
                # Show more context about the failed response
                response_preview = res[:500] if res and len(res) > 500 else (res if res else '(empty)')
                print(f"Error parsing judge response after {max_retries} retries: {e}")
                print(f"Response preview: {response_preview}")
                return 0
                
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if retry < max_retries - 1:
                time.sleep(1)
                continue
            print(f"Error calling API judge after {max_retries} retries: {e}")
            traceback.print_exc()
            return 0
    
    return 0


def call_local_judge(model, tokenizer, instruction, prediction, reference):
    """Call local model as judge to evaluate prediction against ground truth reference"""
    try:
        # Format prompt
        prompt = deepcopy(JUDGE_PROMPT)
        prompt = prompt.replace("{instruction}", instruction)
        prompt = prompt.replace("{prediction}", prediction)
        prompt = prompt.replace("{reference}", reference)
        
        # Convert ChatML format to messages format
        messages = []
        if "<|im_start|>system" in prompt:
            parts = prompt.split("<|im_start|>")
            for part in parts:
                if part.strip().startswith("system"):
                    content = part.split("<|im_end|>")[0].replace("system", "", 1).strip()
                    if content:
                        messages.append({"role": "system", "content": content})
                elif part.strip().startswith("user"):
                    content = part.split("<|im_end|>")[0].replace("user", "", 1).strip()
                    if content:
                        messages.append({"role": "user", "content": content})
        else:
            # If no ChatML format, treat entire prompt as user message
            messages = [{"role": "user", "content": prompt}]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=4096,
                do_sample=False,
                temperature=0.1
            )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_inputs.input_ids, generated_ids)]
        res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Parse response
        if res.startswith("```python"):
            res = res[9:-3].strip()
        elif res.startswith("```"):
            res = res[3:-3].strip()
        
        try:
            parsed_res = repair_and_parse_json(res)
            
            # Handle new response format: {"score": 0 or 1, "explanation": "..."}
            if isinstance(parsed_res, dict):
                # Extract score from the response
                score = parsed_res.get("score")
                if score is not None:
                    # Ensure score is 0 or 1
                    score = int(score)
                    if score not in [0, 1]:
                        # Normalize: any non-zero becomes 1
                        score = 1 if score > 0 else 0
                    return score
                # Try alternative keys
                for key in ["judge_score", "correct", "is_correct", "result"]:
                    if key in parsed_res:
                        val = parsed_res[key]
                        if isinstance(val, (int, float, bool)):
                            return 1 if val else 0
                        elif isinstance(val, str):
                            val_lower = val.lower()
                            if val_lower in ["yes", "true", "correct", "1"]:
                                return 1
                            elif val_lower in ["no", "false", "incorrect", "0"]:
                                return 0
            
            return 0
        except Exception as e:
            print(f"Error parsing judge response: {e}, response: {res[:200]}")
            return 0
    except Exception as e:
        print(f"Error calling local judge: {e}")
        traceback.print_exc()
        return 0


def run_llm_judge(judge_model_path=None, judge_api_model=None, judge_api_key=None, judge_api_base_url=None, input_file=None, output_file=None, force_rejudge=False):
    """Run LLM judge evaluation (supports both API and local models)"""
    print(f"=== Starting LLM Judge Evaluation ===")
    
    # Determine judge type
    use_api = judge_api_model is not None
    use_local = judge_model_path is not None
    
    if not use_api and not use_local:
        print("Error: Must provide either judge_model_path (for local model) or judge_api_model (for API model)")
        return
    
    # Load local model if needed
    judge_model = None
    judge_tokenizer = None
    if use_local:
        print(f"Loading Local Judge Model: {judge_model_path} ...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_path, trust_remote_code=True)
            judge_model = AutoModelForCausalLM.from_pretrained(
                judge_model_path, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True
            ).eval()
            print("Local judge model loaded successfully")
        except Exception as e:
            print(f"Local model load failed: {e}")
            return
    
    # Load API client if needed (following alimeeting's simple approach)
    api_client = None
    if use_api:
        print(f"Using API Judge Model: {judge_api_model}")
        if judge_api_key:
            print(f"Using custom API Key: {judge_api_key[:10]}...")
        if judge_api_base_url:
            print(f"Using custom Base URL: {judge_api_base_url}")
        
        try:
            # Directly create OpenAI client (like alimeeting's simple approach)
            from openai import OpenAI
            
            api_key = judge_api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("Error: No API key provided. Please set --judge-api-key or OPENAI_API_KEY environment variable.")
                if not use_local:
                    return
            else:
                api_client = OpenAI(
                    api_key=api_key,
                    base_url=judge_api_base_url if judge_api_base_url else None
                )
                print(f"API judge client created successfully for model: {judge_api_model}")
        except Exception as e:
            print(f"API client creation failed: {e}")
            traceback.print_exc()
            if not use_local:
                return
    
    if not api_client and not judge_model:
        print("Error: Failed to load any judge model")
        return
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    # Read input file (inference results JSON)
    with open(input_file, 'r', encoding='utf-8') as f_in:
        content = f_in.read().strip()
    
    try:
        # Parse inference results JSON
        inference_data = json.loads(content)
        
        # Extract metadata and detailed_results
        if isinstance(inference_data, dict) and "detailed_results" in inference_data:
            # Full JSON structure with metadata
            metadata = {k: v for k, v in inference_data.items() if k != "detailed_results"}
            data_list = inference_data["detailed_results"]
        elif isinstance(inference_data, list):
            # Just a list of results (legacy format)
            metadata = {}
            data_list = inference_data
        else:
            # Single result object
            metadata = {}
            data_list = [inference_data]
    except Exception as e:
        print(f"Error reading input file: {e}")
        traceback.print_exc()
        return
    
    # Check if output file exists (for resume)
    processed_indices = {}
    if output_file and os.path.exists(output_file) and not force_rejudge:
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if isinstance(existing_data, dict) and "detailed_results" in existing_data:
                    existing_results = existing_data["detailed_results"]
                    # Find already processed samples by checking for judge_score
                    for item in existing_results:
                        if "judge_score" in item and "idx" in item:
                            processed_indices[item["idx"]] = item
            print(f"Resuming from {len(processed_indices)} already processed samples...")
        except Exception:
            print("Could not read existing output file, starting fresh...")
    elif force_rejudge:
        print("Force rejudge enabled: will re-process all samples...")
    
    results = []
    total_correct = 0
    total_processed = 0
    
    try:
        # Create progress bar with postfix for statistics
        pbar = tqdm(data_list, desc="Judge Evaluating")
        for idx, data in enumerate(pbar):
            sample_idx = data.get("idx", idx)
            
            # Check if already processed
            if sample_idx in processed_indices:
                existing_item = processed_indices[sample_idx]
                results.append(existing_item)
                if existing_item.get("judge_score") == 1:
                    total_correct += 1
                if existing_item.get("judge_score") is not None:
                    total_processed += 1
                # Update progress bar even for skipped samples (real-time accuracy)
                current_acc = (total_correct / total_processed * 100.0) if total_processed > 0 else 0.0
                pbar.set_postfix({
                    'acc': f'{current_acc:.2f}%',
                    'correct': f'{total_correct}/{total_processed}'
                })
                pbar.refresh()  # Force immediate update
                continue
            
            try:
                # Extract data
                instruction = data.get("question", data.get("QuestionText", ""))
                pred = data.get("hypothesis", data.get("response_text", ""))
                ref_raw = data.get("reference", data.get("answer", ""))
                
                # Handle reference: convert list to string if needed
                if isinstance(ref_raw, list):
                    # If reference is a list, join them with " or " or use the first one
                    if len(ref_raw) > 0:
                        ref = " or ".join(str(r) for r in ref_raw) if len(ref_raw) > 1 else str(ref_raw[0])
                    else:
                        ref = ""
                else:
                    ref = str(ref_raw) if ref_raw else ""
                
                if not instruction or not pred or not ref:
                    print(f"Warning: Missing data at index {idx}")
                    # Still add the result without judge_score
                    output_item = data.copy()
                    output_item["judge_score"] = None
                    output_item["judge_method"] = "api" if use_api else "local"
                    output_item["judge_error"] = "Missing data"
                    results.append(output_item)
                    continue
                
                # Call judge
                if use_api and api_client:
                    judge_score = call_api_judge(api_client, judge_api_model, instruction, pred, ref)
                elif use_local and judge_model:
                    judge_score = call_local_judge(judge_model, judge_tokenizer, instruction, pred, ref)
                else:
                    judge_score = 0
                
                total_correct += judge_score
                total_processed += 1
                
                # Update progress bar with statistics (real-time accuracy)
                current_acc = (total_correct / total_processed * 100.0) if total_processed > 0 else 0.0
                pbar.set_postfix({
                    'acc': f'{current_acc:.2f}%',
                    'correct': f'{total_correct}/{total_processed}'
                })
                pbar.refresh()  # Force immediate update
                
                # Update result with judge information
                output_item = data.copy()
                output_item["judge_score"] = judge_score
                output_item["judge_method"] = "api" if use_api else "local"
                
                results.append(output_item)
                
                # Save results in real-time (same format as inference)
                if output_file and (idx + 1) % 10 == 0:  # Save every 10 samples
                    final_results = {
                        "model_name": metadata.get("model_name", "Qwen3-Omni-3B"),
                        "dataset": metadata.get("dataset", "speech-web-questions"),
                        "task": metadata.get("task", "AQA (Audio Question Answering)"),
                        "status": "in_progress",
                        "progress": {
                            "processed": len(results),
                            "total": len(data_list),
                            "percentage": len(results) / len(data_list) * 100 if len(data_list) > 0 else 0
                        },
                        "judge_statistics": {
                            "accuracy": (total_correct / total_processed * 100.0) if total_processed > 0 else 0.0,
                            "correct_count": total_correct,
                            "total_processed": total_processed,
                            "judge_method": "api" if use_api else "local",
                            "judge_model": judge_api_model if use_api else judge_model_path
                        },
                        "config": metadata.get("config", {}),
                        "detailed_results": results
                    }
                    final_results = convert_numpy_types(final_results)
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(final_results, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                # Error details are logged via traceback, no need to print to console
                traceback.print_exc()
                # Still add the result with error info
                output_item = data.copy()
                output_item["judge_score"] = None
                output_item["judge_method"] = "api" if use_api else "local"
                output_item["judge_error"] = str(e)
                results.append(output_item)
                # Update progress bar even on error (don't count as correct, but count as processed)
                total_processed += 1
                current_acc = (total_correct / total_processed * 100.0) if total_processed > 0 else 0.0
                pbar.set_postfix({
                    'acc': f'{current_acc:.2f}%',
                    'correct': f'{total_correct}/{total_processed}'
                })
                pbar.refresh()  # Force immediate update
                continue
            
            if (idx + 1) % 5 == 0 and use_local:
                torch.cuda.empty_cache()
    
    finally:
        pass
    
    # Build final results in same format as inference
    final_results = {
        "model_name": metadata.get("model_name", "Qwen3-Omni-3B"),
        "dataset": metadata.get("dataset", "speech-web-questions"),
        "task": metadata.get("task", "AQA (Audio Question Answering)"),
        "status": "completed",
        "progress": {
            "processed": len(results),
            "total": len(data_list),
            "percentage": 100.0
        },
        "judge_statistics": {
            "accuracy": (total_correct / total_processed * 100.0) if total_processed > 0 else 0.0,
            "correct_count": total_correct,
            "total_processed": total_processed,
            "judge_method": "api" if use_api else "local",
            "judge_model": judge_api_model if use_api else judge_model_path
        },
        "config": metadata.get("config", {}),
        "timing_stats": metadata.get("timing_stats", {}),
        "detailed_results": results
    }
    
    # Convert numpy types
    final_results = convert_numpy_types(final_results)
    
    # Save final results
    if output_file:
        safe_makedirs(os.path.dirname(output_file), exist_ok=True) if os.path.dirname(output_file) else None
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    if total_processed > 0:
        accuracy = (total_correct / total_processed * 100.0)
        print(f"\n=== Judge Evaluation Results ===")
        print(f"Total processed: {total_processed}")
        print(f"Correct (prediction preferred): {total_correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Results saved to: {output_file}")
        print("=" * 40)
    else:
        print("No samples processed")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='speech-web-questions AQA Evaluation with Qwen3-Omni')
    parser.add_argument('--mode', type=str, default='inference',
                        choices=['inference', 'judge'],
                        help='运行模式: inference (推理) 或 judge (打分)')
    parser.add_argument('--output-file', '-o', type=str, default=None,
                        help='Output file path for results JSON (overrides --output-dir)')
    parser.add_argument('--output-dir', '-d', type=str, default=None,
                        help='Output directory for results (default: ./ClothoAQA_Results or from RESULTS_DIR env)')
    parser.add_argument('--timing-output-file', '-t', type=str, default=None,
                        help='Output file path for timing statistics JSON')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to use (default: test)')
    parser.add_argument('--model-path', '-m', type=str, default=None,
                        help='Path to the model directory (default: from MODEL_PATH env or hardcoded path)')
    parser.add_argument('--twinkstart-path', type=str, default=None,
                        help='Path to TwinkStart root directory (default: /mnt/users/wangyuxuan-20250915/Audio/UltraEval-Audio/raw/TwinkStart)')
    # Judge mode arguments
    parser.add_argument('--judge-model-path', type=str, default=None,
                        help='Judge model path (for local model, used in judge mode)')
    parser.add_argument('--judge-api-model', type=str, default=None,
                        help='Judge API model name (e.g., gpt4o-mini, qwen-audio, used in judge mode)')
    parser.add_argument('--judge-api-key', type=str, default=None,
                        help='API Key for judge API model (optional, can also use OPENAI_API_KEY env var)')
    parser.add_argument('--judge-api-base-url', type=str, default=None,
                        help='Base URL for judge API (optional, e.g., http://123.129.219.111:3000/v1)')
    parser.add_argument('--input-file', type=str, default=None,
                        help='Input file path (for judge mode, should be inference results JSON)')
    parser.add_argument('--force-rejudge', action='store_true',
                        help='Force rejudge all samples even if they have been processed before')
    args = parser.parse_args()
    
    # Handle judge mode
    if args.mode == 'judge':
        if not args.input_file:
            print("Error: judge mode requires --input-file")
            return
        if not args.output_file:
            if args.output_dir:
                output_file = os.path.join(args.output_dir, 'judge_results.jsonl')
            else:
                output_file = './judge_results.jsonl'
        else:
            output_file = args.output_file
        
        run_llm_judge(
            judge_model_path=args.judge_model_path,
            judge_api_model=args.judge_api_model,
            judge_api_key=args.judge_api_key,
            judge_api_base_url=args.judge_api_base_url,
            input_file=args.input_file,
            output_file=output_file,
            force_rejudge=args.force_rejudge
        )
        return
    
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

    
    # Determine output file paths
    # Priority: --output-file > --output-dir > RESULTS_DIR env > default
    if args.output_file:
        # Direct file path specified
        output_file = os.path.abspath(args.output_file)
        output_dir = os.path.dirname(output_file)
        safe_makedirs(output_dir, exist_ok=True)
    else:
        # Use directory-based approach
        if args.output_dir:
            result_dir = args.output_dir
        else:
            result_dir = os.environ.get("RESULTS_DIR", './speech_web_questions_Results')
        
        result_dir = os.path.abspath(result_dir)
        safe_makedirs(result_dir, exist_ok=True)
        output_file = f'{result_dir}/speech_web_questions_results_qwen3.json'
    
    # Determine timing output file path
    if args.timing_output_file:
        timing_output_file = os.path.abspath(args.timing_output_file)
        timing_output_dir = os.path.dirname(timing_output_file)
        safe_makedirs(timing_output_dir, exist_ok=True)
    else:
        # Use same directory as output_file
        timing_output_dir = os.path.dirname(output_file)
        timing_output_file = f'{timing_output_dir}/timing_stats_qwen3_{method_is}_{prune_ratio}.json'
    
    print(f"Results will be saved to: {output_file} (updated in real-time)")
    print(f"Timing statistics will be saved to: {timing_output_file}")

    
    timing_stats = GlobalTimingStats()

    print(f"\n=== speech-web-questions AQA Evaluation Configuration (Qwen3-Omni) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Pruning ratio: {prune_ratio}")
    print(f"Pruning method: {method_is}")
    print(f"Dataset: speech-web-questions")
    print(f"Split: {args.split}")
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
        model_path = os.environ.get("MODEL_PATH", "/mnt/users/wangyuxuan-20250915/Models/Qwen3-Omni-3B")
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
                head_score_path = get_audio_head_score_path(
                    score_type=head_score_type,
                    batch=head_score_batch
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

    
    print(f"Loading speech-web-questions dataset (split={args.split})")
    dataset = load_speech_web_questions_dataset(split=args.split, twinkstart_path=args.twinkstart_path)
    
    if not dataset:
        print("Error: Failed to load any data")
        return
    
    
    if sample_limit > 0 and len(dataset) > sample_limit:
        dataset = dataset[:sample_limit]
        print(f"Applying sample limit, processing {len(dataset)} samples")

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

    progress_bar = tqdm(enumerate(dataset), total=len(dataset), desc="speech-web-questions AQA evaluation (Qwen3)", **tqdm_kwargs)

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
            
            audio_data_result = speech_web_questions_doc_to_audio(doc)
            if audio_data_result is None:
                continue
            
            audio_np, sr = audio_data_result
            
            
            reference = doc.get("answer", "")
            question_text = doc.get("QuestionText", "")
            
            
            prompt_text = aqa_doc_to_text(doc)
            
            # AQA prompt: "answer the question without explain\n# Question: {{QuestionText}}"
            full_user_prompt = prompt_text

            
            # NOTE: Only audio is provided to the model - question text is removed to force model to extract question from audio
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
                        {"type": "audio", "audio": audio_np}
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
            
            # Decode the full sequence (including input) for AQA tasks
            resp = processor.batch_decode(
                generated_sequences, 
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
            
            # Inference mode: no evaluation, just save results
            processed_samples += 1
            
            # Print sample info
            print(f"\n{'='*80}")
            print(f"[Sample {idx+1}/{len(dataset)}] ID={doc.get('id', f'sample_{idx}')} | "
                  f"Duration={doc.get('duration', 0):.1f}s")
            print(f"{'='*80}")
            
            # Print question, reference and hypothesis (full text)
            print(f"  ❓ 问题: {question_text}")
            print(f"  📝 参考答案 (Ground Truth): {reference}")
            print(f"  💬 模型回答: {hypothesis}")
            print(f"{'='*80}\n")
            
            # Update progress bar
            sample_count = idx + 1
            progress_bar.set_postfix({
                '样本': f'{sample_count}/{len(dataset)}'
            })

            
            results.append({
                "idx": idx,
                "id": doc.get("id", f"sample_{idx}"),
                "question": question_text,
                "reference": reference,
                "hypothesis": hypothesis,
                "response_text": resp,
                "duration": doc.get("duration", 0),
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
                    None,  # total_correct: None for inference mode (no accuracy calculation)
                    processed_samples,
                    timing_stats,
                    gpu_id,
                    prune_ratio,
                    method_is,
                    sample_limit,
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

    
    # Inference mode: save results without evaluation metrics
    final_results = {
        "model_name": f"Qwen3-Omni-3B",
        "dataset": "speech-web-questions",
        "task": "AQA (Audio Question Answering)",
        "status": "completed",
        "progress": {
            "processed": len(results),
            "total": len(dataset),
            "percentage": 100.0
        },
        "total_samples": len(results),
        
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
            "dataset_source": "speech-web-questions",
            "split": args.split
        },
        
        # Detailed results
        "detailed_results": results
    }
    
    
    final_results = convert_numpy_types(final_results)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    
    timing_stats.export_to_json(timing_output_file)

    
    print(f"\n=== speech-web-questions Inference Results Summary (Qwen3-Omni) ===")
    print(f"Total samples: {len(results)}")
    print(f"Processed samples: {processed_samples}")
    print(f"Note: Evaluation will be performed in judge mode")
    
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
    