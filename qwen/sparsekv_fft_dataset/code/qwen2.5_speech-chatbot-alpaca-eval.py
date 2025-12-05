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

def load_speech_chatbot_alpaca_eval_dataset(split="test"):
    """Load speech-chatbot-alpaca-eval dataset from local path with metadata from HuggingFace"""
    print(f"Loading speech-chatbot-alpaca-eval dataset (split={split})...")
    
    # UltraEval-Audio root directory
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
    
    if not ultraeval_root:
        print("Error: UltraEval-Audio root directory not found.")
        return []
    
    # Check if local audio files exist
    local_audio_dir = os.path.join(ultraeval_root, "raw", "TwinkStart", "speech-chatbot-alpaca-eval", split)
    local_audio_files = []
    if os.path.exists(local_audio_dir):
        local_audio_files = sorted(glob.glob(os.path.join(local_audio_dir, "*.wav")), 
                                  key=lambda x: int(os.path.splitext(os.path.basename(x))[0]) if os.path.splitext(os.path.basename(x))[0].isdigit() else float('inf'))
        print(f"Found {len(local_audio_files)} local audio files in {local_audio_dir}")
    else:
        print(f"Warning: Local audio directory not found: {local_audio_dir}")
    
    # Add UltraEval-Audio to Python path
    if ultraeval_root not in sys.path:
        sys.path.insert(0, ultraeval_root)
    
    dataset_data = None
    
    # Try to load from HuggingFace cache first (without downloading)
    try:
        from datasets import load_dataset
        print(f"Attempting to load speech-chatbot-alpaca-eval from HuggingFace cache (no download)...")
        hf_dataset = load_dataset("TwinkStart/speech-chatbot-alpaca-eval", split=split, download_mode="reuse_cache_if_exists")
        print(f"Successfully loaded {len(hf_dataset)} samples from HuggingFace cache")
        
        # Convert to list format
        dataset_data = []
        for idx, item in enumerate(hf_dataset):
            # Extract question text - prioritize instruction (chatbot format), then input, then QuestionText
            question_text = item.get("instruction", "")
            if not question_text:
                question_text = item.get("input", "")
            if not question_text:
                question_text = item.get("QuestionText", "")
            if not question_text:
                # Last resort: try other common field names
                question_text = item.get("question", item.get("prompt", ""))
            
            # Extract answer - according to registry, ref_col is "output"
            answer = item.get("output", "")
            if not answer:
                # Fallback to "answer" if "output" is not available
                answer = item.get("answer", "")
            
            # Use local audio file if available, otherwise use audio from HuggingFace dataset
            audio_path = None
            audio_array = None
            sampling_rate = 16000
            
            if local_audio_files and idx < len(local_audio_files):
                audio_path = local_audio_files[idx]
                # Load audio from local file
                try:
                    # Try librosa first, then soundfile
                    try:
                        audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
                        sampling_rate = 16000
                    except Exception:
                        audio, sample_rate = sf.read(audio_path)
                        if len(audio.shape) > 1 and audio.shape[1] > 1:
                            audio = np.mean(audio, axis=1)
                        if sample_rate != 16000:
                            from scipy import signal
                            audio = signal.resample(audio, int(len(audio) * 16000 / sample_rate))
                        audio_array = audio.astype(np.float32)
                        sampling_rate = 16000
                except Exception as e:
                    print(f"Warning: Failed to load audio from {audio_path}: {e}")
                    audio_array = None
            elif 'audio' in item and isinstance(item['audio'], dict):
                audio_path = item['audio'].get('path')
                audio_array = item['audio'].get('array')
                sampling_rate = item['audio'].get('sampling_rate', 16000)
            
            # Calculate duration if audio array is available
            duration = 0.0
            if audio_array is not None:
                try:
                    duration = len(audio_array) / sampling_rate
                except:
                    duration = 0.0
            
            dataset_data.append({
                "id": item.get("id", f"sample_{idx}"),
                "QuestionText": question_text,
                "answer": answer,
                "duration": duration,
                "audio": {
                    "array": audio_array,
                    "sampling_rate": sampling_rate,
                    "path": audio_path
                }
            })
            
    except Exception as e:
        print(f"Could not load from HuggingFace cache: {e}")
        print(f"Attempting to load from UltraEval-Audio registry...")
        
        # Fallback to registry
        try:
            from audio_evals.registry import registry
            
            dataset_obj = registry.get_dataset("speech-chatbot-alpaca-eval")
            if dataset_obj is None:
                print(f"Error: Dataset 'speech-chatbot-alpaca-eval' not found in registry")
                # Fall back to local files only
                if not local_audio_files:
                    return []
                dataset_data = []
                for idx, audio_file in enumerate(local_audio_files):
                    try:
                        # Try librosa first, then soundfile
                        try:
                            audio_array, sr = librosa.load(audio_file, sr=16000, mono=True)
                        except Exception:
                            audio, sample_rate = sf.read(audio_file)
                            if len(audio.shape) > 1 and audio.shape[1] > 1:
                                audio = np.mean(audio, axis=1)
                            if sample_rate != 16000:
                                from scipy import signal
                                audio = signal.resample(audio, int(len(audio) * 16000 / sample_rate))
                            audio_array = audio.astype(np.float32)
                        duration = len(audio_array) / 16000.0
                    except:
                        audio_array = None
                        duration = 0.0
                    dataset_data.append({
                        'id': f"sample_{idx}",
                        'QuestionText': '',  # No question available from local files only
                        'answer': '',  # No answer available from local files only
                        'duration': duration,
                        'audio': {
                            'array': audio_array,
                            'sampling_rate': 16000,
                            'path': audio_file
                        }
                    })
            else:
                # Load dataset data from registry
                registry_data = dataset_obj.load(limit=0)
                print(f"Loaded {len(registry_data)} samples from registry")
                
                # Convert to list format and map to local audio files
                dataset_data = []
                for idx, item in enumerate(registry_data):
                    # Extract question text
                    question_text = item.get("instruction", "")
                    if not question_text:
                        question_text = item.get("input", "")
                    if not question_text:
                        question_text = item.get("QuestionText", "")
                    if not question_text:
                        question_text = item.get("question", item.get("prompt", ""))
                    
                    # Extract answer
                    answer = item.get("output", "")
                    if not answer:
                        answer = item.get("answer", "")
                    
                    # Use local audio file if available
                    audio_path = None
                    audio_array = None
                    sampling_rate = 16000
                    
                    if local_audio_files and idx < len(local_audio_files):
                        audio_path = local_audio_files[idx]
                        try:
                            # Try librosa first, then soundfile
                            try:
                                audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
                                sampling_rate = 16000
                            except Exception:
                                audio, sample_rate = sf.read(audio_path)
                                if len(audio.shape) > 1 and audio.shape[1] > 1:
                                    audio = np.mean(audio, axis=1)
                                if sample_rate != 16000:
                                    from scipy import signal
                                    audio = signal.resample(audio, int(len(audio) * 16000 / sample_rate))
                                audio_array = audio.astype(np.float32)
                                sampling_rate = 16000
                        except Exception as e:
                            print(f"Warning: Failed to load audio from {audio_path}: {e}")
                            audio_array = None
                    else:
                        # Use audio from registry data
                        audio_data = item.get("audio", {})
                        if isinstance(audio_data, dict):
                            audio_array = audio_data.get("array", None)
                            sampling_rate = audio_data.get("sampling_rate", 16000)
                            audio_path = audio_data.get("path", None)
                    
                    # Calculate duration
                    duration = 0.0
                    if audio_array is not None:
                        try:
                            duration = len(audio_array) / sampling_rate
                        except:
                            duration = 0.0
                    
                    dataset_data.append({
                        "id": item.get("id", f"sample_{idx}"),
                        "QuestionText": question_text,
                        "answer": answer,
                        "duration": duration,
                        "audio": {
                            "array": audio_array,
                            "sampling_rate": sampling_rate,
                            "path": audio_path
                        }
                    })
                    
        except Exception as registry_e:
            print(f"Error loading from registry: {registry_e}")
            traceback.print_exc()
            return []
    
    if dataset_data is None or len(dataset_data) == 0:
        print("Error: Failed to load any dataset data")
        return []
    
    # Debug: Print available fields from first sample to verify dataset structure
    if len(dataset_data) > 0:
        first_item = dataset_data[0]
        print(f"Debug: Available fields in dataset sample: {list(first_item.keys())}")
        print(f"Debug: Sample fields preview: QuestionText={first_item.get('QuestionText', 'N/A')[:50] if first_item.get('QuestionText') else 'N/A'}, "
              f"answer={first_item.get('answer', 'N/A')[:50] if first_item.get('answer') else 'N/A'}, "
              f"audio_path={first_item.get('audio', {}).get('path', 'N/A')}")
    
    print(f"Successfully loaded {len(dataset_data)} samples")
    return dataset_data

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

def speech_chatbot_alpaca_eval_doc_to_audio(doc):
    """Load audio data from speech-chatbot-alpaca-eval documents"""
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
    prompt_text = f"answer the question without explain\n# Question: {question_text}"
    
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
    
    # Remove common conversational prefixes (Well, So, Actually, etc.)
    # These are often added by chat models but shouldn't affect matching
    conversational_prefixes = [
        r"^well,\s*",
        r"^so,\s*",
        r"^actually,\s*",
        r"^you know,\s*",
        r"^i think\s+",
        r"^i believe\s+",
        r"^in my opinion,\s*",
        r"^to answer your question,\s*",
        r"^the answer is\s+",
        r"^basically,\s*",
        r"^essentially,\s*",
        r"^there are\s+",  # Add more patterns
        r"^some\s+",  # Add more patterns
    ]
    
    for prefix_pattern in conversational_prefixes:
        response = re.sub(prefix_pattern, "", response, flags=re.IGNORECASE)
        response = response.strip()
    
    # Remove trailing conversational phrases that don't add value
    trailing_phrases = [
        r"\s*if you want to know more.*$",
        r"\s*feel free to ask.*$",
        r"\s*let me know if.*$",
        r"\s*is there anything else.*$",
    ]
    
    for phrase_pattern in trailing_phrases:
        response = re.sub(phrase_pattern, "", response, flags=re.IGNORECASE)
        response = response.strip()
    
    return response.strip()

def prefix_match(pred, label, ignore_case=True):
    """
    Prefix match evaluator (following UltraEval-Audio PrefixMatch implementation)
    Returns 1 if prediction prefix matches label, 0 otherwise
    
    Enhanced version that handles format differences:
    - Tries direct prefix match first (original behavior)
    - If that fails, tries to find substantial key content from reference within prediction
    """
    if ignore_case:
        pred = pred.lower().strip()
        label = str(label).lower().strip()
    else:
        pred = pred.strip()
        label = str(label).strip()
    
    # Handle empty label case
    if len(label) == 0:
        return 1 if len(pred) == 0 else 0
    
    # Normalize whitespace: replace multiple whitespace (including newlines) with single space
    pred_normalized = re.sub(r'\s+', ' ', pred)
    label_normalized = re.sub(r'\s+', ' ', label)
    
    # Direct prefix match (original behavior)
    n = len(label_normalized)
    if len(pred_normalized) < n:
        m = 0
    else:
        m = 1 if pred_normalized[:n] == label_normalized else 0
    
    # If no match and label has spaces but pred doesn't, try without spaces
    if m == 0 and " " in label_normalized and " " not in pred_normalized:
        label_no_spaces = label_normalized.replace(" ", "")
        n_no_spaces = len(label_no_spaces)
        if len(pred_normalized) >= n_no_spaces:
            m = 1 if pred_normalized[:n_no_spaces] == label_no_spaces else 0
    
    # Enhanced matching: if direct prefix match fails, try to find key content
    # Extract meaningful phrases from reference and check if they appear in prediction
    if m == 0:
        # Try to find substantial content from the first part of reference in prediction
        # Take first 100 chars of reference as key content
        key_content = label_normalized[:100].strip()
        if len(key_content) > 20:
            # Extract key phrases (sequences of 3+ words) from key content
            words = key_content.split()
            if len(words) >= 3:
                # Try matching first 3-5 words as a phrase
                for phrase_len in range(min(5, len(words)), 2, -1):
                    phrase = ' '.join(words[:phrase_len])
                    # Check if this phrase appears in prediction (within first 150 chars to keep it prefix-like)
                    if phrase in pred_normalized[:150]:
                        m = 1
                        break
                
                # If still no match, try finding individual key words (longer words, excluding common words)
                if m == 0:
                    common_words = {'the', 'and', 'are', 'from', 'that', 'with', 'this', 'have', 'been', 'were', 'will', 'would', 'could', 'should', 'some', 'their', 'names', 'get', 'states', 'here', 'what', 'need', 'you', 'your', 'can', 'help', 'sure', 'happy', 'basic', 'rundown', 'how', 'play', 'game', 'similar', 'baseball', 'large', 'rubber', 'ball', 'instead', 'bat', 'cool', 'music', 'includes', 'jazz', 'classics', 'like', 'start', 'gathering', 'supplies', 'wrap', 'present', 'such', 'wrapping', 'paper', 'tape'}
                    key_words = [w for w in words if len(w) > 3 and w not in common_words]
                    if len(key_words) >= 2:
                        # Check if at least 2 key words appear in prediction (check in first 200 chars, more lenient)
                        found_count = sum(1 for word in key_words if word in pred_normalized[:200])
                        if found_count >= 2:
                            m = 1
                
                # Additional check: look for key topic words anywhere in prediction (not just prefix)
                # This handles cases where model uses different phrasing but same topic
                if m == 0:
                    # Extract topic-specific words (nouns, important verbs) from entire reference, not just first 100 chars
                    # This is more lenient - allows different sentence structure
                    all_words = label_normalized.split()
                    topic_words = []
                    for word in all_words:
                        # Skip very common words and short words
                        if len(word) > 4 and word not in common_words:
                            topic_words.append(word)
                    
                    if len(topic_words) >= 2:
                        # Check if topic words appear anywhere in prediction (not just prefix)
                        found_topic_words = sum(1 for word in topic_words if word in pred_normalized)
                        # Need at least 2 topic words to match, or 50% of topic words if more than 4
                        threshold = max(2, int(len(topic_words) * 0.5)) if len(topic_words) > 4 else 2
                        if found_topic_words >= threshold:
                            m = 1
    
    return m


def save_results_realtime(output_file, results, total_correct, processed_samples, 
                          timing_stats, gpu_id, prune_ratio, method_is, 
                          sample_limit, dataset_size):
    """
    Save results to JSON file in real-time, updating statistics as we go
    This allows monitoring progress and viewing results before completion
    """
    try:
        # Calculate current statistics
        current_accuracy = (total_correct / processed_samples * 100.0) if processed_samples > 0 else 0.0
        
        # Calculate detailed metrics from current results
        # Note: match 字段可能不存在（推理阶段不计算对错判断，由 judge 阶段负责）
        match_scores = [r.get("match") for r in results if "match" in r]
        if len(match_scores) > 0:
            accuracy = np.mean(match_scores) * 100.0
            correct_count = int(sum(match_scores))
            total_count = len(match_scores)
        else:
            # 推理阶段不计算准确率，显示 N/A
            accuracy = None
            correct_count = 0
            total_count = len(results)
        
        # Build real-time results structure
        realtime_results = {
            "model_name": f"Qwen2.5-Omni-3B",
            "dataset": "speech-chatbot-alpaca-eval",
            "task": "AQA (Audio Question Answering)",
            "status": "in_progress",
            "progress": {
                "processed": len(results),
                "total": dataset_size,
                "percentage": len(results) / dataset_size * 100 if dataset_size > 0 else 0
            },
            "current_statistics": {
                "accuracy": float(accuracy) if accuracy is not None else None,
                "correct_count": int(correct_count),
                "total_count": int(total_count),
                "current_avg_accuracy": float(current_accuracy) if processed_samples > 0 else None,
                "note": "推理阶段不计算准确率，请使用 judge 模式进行评估" if accuracy is None else None
            },
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

def calculate_speech_chatbot_alpaca_eval_metrics(results):
    """
    Calculate speech-chatbot-alpaca-eval AQA metrics (accuracy using prefix match)
    
    Args:
        results: List of result dictionaries with optional 'match' field
        Note: In inference mode, 'match' field may not exist (judge mode handles evaluation)
        
    Returns:
        dict: Dictionary containing accuracy-related metrics (accuracy may be None if match field is missing)
    """
    if len(results) == 0:
        return {
            'accuracy': None,
            'correct_count': 0,
            'total_count': 0,
            'valid_samples': 0
        }
    
    # Check if match field exists in any result
    has_match_field = any("match" in r for r in results)
    
    if not has_match_field:
        # Inference mode: no match field, accuracy calculation deferred to judge mode
        return {
            'accuracy': None,
            'correct_count': 0,
            'total_count': len(results),
            'valid_samples': len(results)
        }
    
    # Judge mode: match field exists, calculate accuracy
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
# Alpaca eval prompt from UltraEval-Audio
ALPACA_EVAL_PROMPT = """<|im_start|>system
You are a helpful assistant, that ranks models by the quality of their answers.
<|im_end|>
<|im_start|>user
I want you to create a leaderboard of different of large-language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be python dictionaries.

Here is the prompt:
{{
    "instruction": ""\"{instruction}\"\"",
}}

Here are the outputs of the models:
[
    {{
        "model": "model_1",
        "answer": ""\"{output_1}\"\""
    }},
    {{
        "model": "model_2",
        "answer": ""\"{output_2}\"\""
    }}
]

Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:
[
    {{'model': <model-name>, 'rank': <model-rank>}},
    {{'model': <model-name>, 'rank': <model-rank>}}
]

Your response must be a valid Python dictionary and should contain nothing else because we will directly execute it in Python. Please provide the ranking that the majority of humans would give.
<|im_end|>"""


def call_api_judge(api_model, instruction, output_1, output_2):
    """Call API model as judge (following UltraEval-Audio AlpacaEvaluator pattern)"""
    try:
        # Format prompt
        prompt = deepcopy(ALPACA_EVAL_PROMPT)
        prompt = prompt.replace("{instruction}", instruction)
        prompt = prompt.replace("{output_1}", output_1)
        prompt = prompt.replace("{output_2}", output_2)
        
        # Call API model
        res = api_model.inference(prompt, temperature=0, maxTokens=100)
        
        # Parse response (following AlpacaEvaluator logic)
        if res.startswith("```python"):
            res = res[9:-3].strip()
        elif res.startswith("```"):
            res = res[3:-3].strip()
        
        try:
            res = ast.literal_eval(res)
            if isinstance(res, dict):
                for k in res:
                    res = res[k]
                    break
            # Return 1 if model_1 (prediction) is ranked higher, 0 otherwise
            if isinstance(res, list) and len(res) >= 2:
                model_1_rank = next((item.get("rank") for item in res if item.get("model") == "model_1"), None)
                model_2_rank = next((item.get("rank") for item in res if item.get("model") == "model_2"), None)
                if model_1_rank is not None and model_2_rank is not None:
                    # Lower rank number means better (rank 1 is best)
                    return 1 if model_1_rank < model_2_rank else 0
            return 0
        except Exception as e:
            print(f"Error parsing judge response: {e}, response: {res[:200]}")
            return 0
    except Exception as e:
        print(f"Error calling API judge: {e}")
        traceback.print_exc()
        return 0


def call_local_judge(model, tokenizer, instruction, output_1, output_2):
    """Call local model as judge (following alimeeting pattern)"""
    try:
        # Format prompt
        prompt = deepcopy(ALPACA_EVAL_PROMPT)
        prompt = prompt.replace("{instruction}", instruction)
        prompt = prompt.replace("{output_1}", output_1)
        prompt = prompt.replace("{output_2}", output_2)
        
        # Convert to messages format
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
        
        # Parse response (following AlpacaEvaluator logic)
        if res.startswith("```python"):
            res = res[9:-3].strip()
        elif res.startswith("```"):
            res = res[3:-3].strip()
        
        try:
            res = ast.literal_eval(res)
            if isinstance(res, dict):
                for k in res:
                    res = res[k]
                    break
            # Return 1 if model_1 (prediction) is ranked higher, 0 otherwise
            if isinstance(res, list) and len(res) >= 2:
                model_1_rank = next((item.get("rank") for item in res if item.get("model") == "model_1"), None)
                model_2_rank = next((item.get("rank") for item in res if item.get("model") == "model_2"), None)
                if model_1_rank is not None and model_2_rank is not None:
                    # Lower rank number means better (rank 1 is best)
                    return 1 if model_1_rank < model_2_rank else 0
            return 0
        except Exception as e:
            print(f"Error parsing judge response: {e}, response: {res[:200]}")
            return 0
    except Exception as e:
        print(f"Error calling local judge: {e}")
        traceback.print_exc()
        return 0


def run_llm_judge(judge_model_path=None, judge_api_model=None, input_file=None, output_file=None):
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
    
    # Load API model if needed
    api_model = None
    if use_api:
        print(f"Using API Judge Model: {judge_api_model}")
        try:
            # Try to load from UltraEval-Audio registry
            ultraeval_root = "/mnt/users/wangyuxuan-20250915/Audio/UltraEval-Audio"
            if os.path.exists(ultraeval_root):
                if ultraeval_root not in sys.path:
                    sys.path.insert(0, ultraeval_root)
                from audio_evals.registry import registry
                api_model = registry.get_model(judge_api_model)
                if api_model is None:
                    print(f"Warning: Could not load API model '{judge_api_model}' from registry")
                else:
                    print(f"API judge model '{judge_api_model}' loaded successfully")
            else:
                print(f"Warning: UltraEval-Audio root not found at {ultraeval_root}")
        except Exception as e:
            print(f"API model load failed: {e}")
            traceback.print_exc()
            if not use_local:
                return
    
    if not api_model and not judge_model:
        print("Error: Failed to load any judge model")
        return
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True) if output_file else None
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    # Process input file
    processed_count = 0
    if output_file and os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            processed_count = sum(1 for _ in f)
    print(f"Resuming from line {processed_count}...")
    
    results = []
    total_correct = 0
    total_processed = 0
    
    # Read input file (could be single JSON or JSONL)
    with open(input_file, 'r', encoding='utf-8') as f_in:
        content = f_in.read().strip()
    
    try:
        # Try to parse as JSON first
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "detailed_results" in data:
                data_list = data["detailed_results"]
            elif isinstance(data, list):
                data_list = data
            else:
                # Single result object
                data_list = [data]
        except json.JSONDecodeError:
            # Try JSONL format
            data_list = []
            with open(input_file, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    if line.strip():
                        try:
                            data_list.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        print(f"Error reading input file: {e}")
        traceback.print_exc()
        return
    
    # Open output file
    if output_file:
        f_out = open(output_file, 'a', encoding='utf-8')
    else:
        f_out = None
    
    try:
            
            for idx, data in enumerate(tqdm(data_list, desc="Judge Evaluating")):
                if idx < processed_count:
                    continue
                
                try:
                    # Extract data
                    instruction = data.get("question", data.get("QuestionText", ""))
                    pred = data.get("hypothesis", data.get("response_text", ""))
                    ref = data.get("reference", data.get("answer", ""))
                    
                    if not instruction or not pred or not ref:
                        print(f"Warning: Missing data at index {idx}")
                        continue
                    
                    # Call judge
                    if use_api and api_model:
                        judge_score = call_api_judge(api_model, instruction, pred, ref)
                    elif use_local and judge_model:
                        judge_score = call_local_judge(judge_model, judge_tokenizer, instruction, pred, ref)
                    else:
                        judge_score = 0
                    
                    total_correct += judge_score
                    total_processed += 1
                    
                    # Update result
                    output_item = data.copy()
                    output_item["judge_score"] = judge_score
                    output_item["judge_method"] = "api" if use_api else "local"
                    
                    results.append(output_item)
                    
                    if f_out:
                        f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                        f_out.flush()
                    
                except Exception as e:
                    print(f"\n[Error processing index {idx}]: {e}")
                    traceback.print_exc()
                    continue
                
                if (idx + 1) % 5 == 0 and use_local:
                    torch.cuda.empty_cache()
    
    finally:
        if f_out:
            f_out.close()
    
    # Print statistics
    if total_processed > 0:
        accuracy = (total_correct / total_processed * 100.0)
        print(f"\n=== Judge Evaluation Results ===")
        print(f"Total processed: {total_processed}")
        print(f"Correct (prediction preferred): {total_correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print("=" * 40)
    else:
        print("No samples processed")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='speech-chatbot-alpaca-eval AQA Evaluation with Qwen2.5-Omni')
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
    # Judge mode arguments
    parser.add_argument('--judge-model-path', type=str, default=None,
                        help='Judge model path (for local model, used in judge mode)')
    parser.add_argument('--judge-api-model', type=str, default=None,
                        help='Judge API model name (e.g., gpt4o-mini, qwen-audio, used in judge mode)')
    parser.add_argument('--input-file', type=str, default=None,
                        help='Input file path (for judge mode, should be inference results JSON)')
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
            input_file=args.input_file,
            output_file=output_file
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
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Use directory-based approach
        if args.output_dir:
            result_dir = args.output_dir
        else:
            result_dir = os.environ.get("RESULTS_DIR", './speech_chatbot_alpaca_eval_Results')
        
        result_dir = os.path.abspath(result_dir)
        os.makedirs(result_dir, exist_ok=True)
        output_file = f'{result_dir}/speech_chatbot_alpaca_eval_results_qwen25.json'
    
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

    print(f"\n=== speech-chatbot-alpaca-eval AQA Evaluation Configuration (Qwen2.5-Omni) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Pruning ratio: {prune_ratio}")
    print(f"Pruning method: {method_is}")
    print(f"Dataset: speech-chatbot-alpaca-eval")
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

    
    print(f"Loading speech-chatbot-alpaca-eval dataset (split={args.split})")
    dataset = load_speech_chatbot_alpaca_eval_dataset(split=args.split)
    
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

    progress_bar = tqdm(enumerate(dataset), total=len(dataset), desc="speech-chatbot-alpaca-eval AQA evaluation (Qwen2.5)", **tqdm_kwargs)

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
            
            audio_data_result = speech_chatbot_alpaca_eval_doc_to_audio(doc)
            if audio_data_result is None:
                continue
            
            audio_np, sr = audio_data_result
            
            
            reference = doc.get("answer", "")
            question_text = doc.get("QuestionText", "")
            
            
            prompt_text = aqa_doc_to_text(doc)
            
            # AQA prompt: "answer the question without explain\n# Question: {{QuestionText}}"
            full_user_prompt = prompt_text

            
            # For AQA tasks, use a more direct system prompt to encourage concise answers
            # This helps match the reference answer format better
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant. Answer questions directly and concisely without unnecessary explanations or conversational fillers."}
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
            
            # Note: 对错判断已移至独立的 judge 步骤，推理阶段只负责生成回答
            processed_samples += 1
            
            # ===== 实时输出样本信息 =====
            # 在每个样本处理后立即输出，格式清晰醒目
            print(f"\n{'='*80}")
            print(f"[Sample {idx+1}/{len(dataset)}] ID={doc.get('id', f'sample_{idx}')} | "
                  f"Duration={doc.get('duration', 0):.1f}s")
            print(f"{'='*80}")
            
            # Print question, reference and hypothesis (truncated if too long)
            question_display = question_text[:100] + "..." if len(question_text) > 100 else question_text
            ref_display = reference[:100] + "..." if len(reference) > 100 else reference
            hyp_display = hypothesis[:100] + "..." if len(hypothesis) > 100 else hypothesis
            print(f"  ❓ 问题: {question_display}")
            print(f"  📝 参考答案: {ref_display}")
            print(f"  💬 模型回答: {hyp_display}")
            print(f"{'='*80}\n")
            
            # 更新进度条
            update_interval = 10 if is_screen_env else 1
            sample_count = idx + 1
            
            # 进度条显示进度信息（不显示准确率，因为对错判断在 judge 阶段）
            progress_bar.set_postfix({
                '样本': f'{sample_count}/{len(dataset)}'
            })
            
            if sample_count % update_interval == 0 or sample_count == len(dataset):
                if is_screen_env:
                    # 在非交互式环境中，额外输出进度信息
                    print(f"  📊 进度更新: {sample_count}/{len(dataset)} ({sample_count/len(dataset)*100:.1f}%)")

            
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
                    0,  # total_correct: 推理阶段不计算准确率，由 judge 阶段负责
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

    
    # Note: 推理阶段不计算最终准确率，对错判断由独立的 judge 步骤负责
    # Calculate detailed metrics
    detailed_metrics = calculate_speech_chatbot_alpaca_eval_metrics(results)

    
    final_results = {
        "model_name": f"Qwen2.5-Omni-3B",
        "dataset": "speech-chatbot-alpaca-eval",
        "task": "AQA (Audio Question Answering)",
        "status": "completed",
        "progress": {
            "processed": len(results),
            "total": len(dataset),
            "percentage": 100.0
        },
        "total_samples": len(results),
        "valid_samples": detailed_metrics['valid_samples'],
        
        # AQA metrics (may be None in inference mode, calculated in judge mode)
        "accuracy": detailed_metrics['accuracy'],
        "correct_count": detailed_metrics['correct_count'],
        "total_count": detailed_metrics['total_count'],
        "note": "推理阶段不计算准确率，请使用 judge 模式进行评估" if detailed_metrics['accuracy'] is None else None,
        
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
            "dataset_source": "speech-chatbot-alpaca-eval",
            "split": args.split
        },
        
        # Detailed results
        "detailed_results": results
    }
    
    
    final_results = convert_numpy_types(final_results)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    
    timing_stats.export_to_json(timing_output_file)

    
    print(f"\n=== speech-chatbot-alpaca-eval AQA Evaluation Results Summary (Qwen2.5-Omni) ===")
    print(f"Total samples: {len(results)}")
    print(f"Processed samples: {processed_samples}")
    if detailed_metrics['accuracy'] is not None:
        print(f"Accuracy: {detailed_metrics['accuracy']:.2f}%")
        print(f"Correct: {detailed_metrics['correct_count']}/{detailed_metrics['total_count']}")
    else:
        print(f"Accuracy: N/A (推理阶段不计算准确率)")
        print(f"Note: 请使用 judge 模式对推理结果进行评估")
        print(f"      bash run_speech-chatbot-alpaca-eval_judge.sh")
    
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
    