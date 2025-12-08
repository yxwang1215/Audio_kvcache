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
from datasets import load_dataset, load_from_disk
import evaluate
import sacrebleu


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
        from modeling_qwen3_omni import Qwen3OmniMoeForConditionalGeneration
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

# Translation instructions (improved based on official Qwen3-Omni prompts)
# Reference: UltraEval-Audio/registry/prompt/qwen-omni.yaml
# 
# Why multiple variants?
# - AudioBench originally used 20 variants to test model robustness across different phrasings
# - However, too many similar variants can confuse the model and reduce translation quality
# - Official Qwen3-Omni examples use a single, clear, explicit prompt
# 
# Current approach: Use 1-2 most effective prompts for consistency and clarity
# If you want to test robustness, you can add more variants, but they should be:
# 1. Explicit about source and target languages
# 2. Clear about output format (text)
# 3. Direct and unambiguous

# Primary prompt (recommended, matches official format)
ST_INSTRUCTIONS_EN_ZH = [
    "Listen to the provided English speech and produce a translation in Chinese text."
]

ST_INSTRUCTIONS_ZH_EN = [
    "Listen to the provided Chinese speech and produce a translation in English text."
]

# Alternative: If you want to test with multiple variants (uncomment to use)
# ST_INSTRUCTIONS_EN_ZH = [
#     "Listen to the provided English speech and produce a translation in Chinese text.",
#     "Translate the English speech audio into Chinese text.",
# ]
# 
# ST_INSTRUCTIONS_ZH_EN = [
#     "Listen to the provided Chinese speech and produce a translation in English text.",
#     "Translate the Chinese speech audio into English text.",
# ]

# System messages for speech translation (based on official Qwen3-Omni format)
ST_SYSTEM_MESSAGE = "You are a speech translation model."

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
                "avg_tps": 0.0,
                "overall_tps": 0.0
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
    # First, try dict-like access (works for both dict and AudioDecoder objects)
    try:
        # Try to access 'array' key (works for dict and AudioDecoder)
        if hasattr(audio_data, '__getitem__'):
            audio_array = audio_data['array']
            audio_np = np.array(audio_array, dtype=np.float32)
            # Try to get sampling_rate
            if hasattr(audio_data, 'get'):
                sr = audio_data.get('sampling_rate', 16000)
            else:
                try:
                    sr = audio_data['sampling_rate']
                except (KeyError, TypeError):
                    sr = 16000
            return audio_np, sr
    except (KeyError, TypeError, AttributeError) as e:
        # If dict-like access fails, continue to other methods
        pass
    
    # Handle dict type explicitly
    if isinstance(audio_data, dict):
        if "path" in audio_data:
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
    
    # If all else fails, print error
    print(f"Unknown audio format: {type(audio_data)}")
    return None

def clean_translation_response(response):
    """Clean translation response"""
    if not response or response.strip() == "":
        return ""
    
    # Remove common prefixes
    response = response.strip()
    
    # Remove translation markers
    for marker in ["translation:", "translated text:", "chinese:", "english:", "中文:", "英文:"]:
        if marker.lower() in response.lower():
            parts = re.split(re.escape(marker), response, flags=re.IGNORECASE)
            if len(parts) > 1:
                response = parts[1].strip()
                break
    
    # Remove extra whitespace
    response = re.sub(r'\s+', ' ', response).strip()
    
    return response

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

def save_results_realtime(output_file, results, total_bleu, total_chrf, processed_samples, 
                          timing_stats, gpu_id, prune_ratio, method_is, 
                          sample_limit, dataset_path, dataset_size, direction):
    """
    Save results to JSON file in real-time, updating statistics as we go
    """
    try:
        # Calculate current statistics
        current_avg_bleu = total_bleu / processed_samples if processed_samples > 0 else 0.0
        current_avg_chrf = total_chrf / processed_samples if processed_samples > 0 else 0.0
        
        # Calculate detailed metrics from current results
        predictions = [r.get("prediction", "") for r in results]
        references = [r.get("reference", "") for r in results]
        
        # Calculate BLEU statistics
        bleu_scores = [r.get("bleu", 0.0) for r in results if "bleu" in r]
        if len(bleu_scores) > 0:
            bleu_mean = np.mean(bleu_scores)
            bleu_std = np.std(bleu_scores)
            bleu_min = np.min(bleu_scores)
            bleu_max = np.max(bleu_scores)
        else:
            bleu_mean = 0.0
            bleu_std = 0.0
            bleu_min = 0.0
            bleu_max = 0.0
        
        # Calculate chrF statistics
        chrf_scores = [r.get("chrf", 0.0) for r in results if "chrf" in r]
        if len(chrf_scores) > 0:
            chrf_mean = np.mean(chrf_scores)
            chrf_std = np.std(chrf_scores)
            chrf_min = np.min(chrf_scores)
            chrf_max = np.max(chrf_scores)
        else:
            chrf_mean = 0.0
            chrf_std = 0.0
            chrf_min = 0.0
            chrf_max = 0.0
        
        # Calculate overall BLEU and chrF using sacrebleu (functions handle empty values internally)
        overall_bleu = calculate_bleu_score(predictions, references, direction) if len(predictions) > 0 else 0.0
        overall_chrf = calculate_chrf_score(predictions, references) if len(predictions) > 0 else 0.0
        
        # Build real-time results structure
        realtime_results = {
            "model_name": f"Qwen3-Omni",
            "dataset": f"CoVoST2-{direction.upper()}",
            "task": "Speech Translation",
            "status": "in_progress",
            "progress": {
                "processed": len(results),
                "total": dataset_size,
                "percentage": len(results) / dataset_size * 100 if dataset_size > 0 else 0
            },
            "current_statistics": {
                "bleu_mean": float(bleu_mean),
                "bleu_std": float(bleu_std),
                "bleu_min": float(bleu_min),
                "bleu_max": float(bleu_max),
                "overall_bleu": float(overall_bleu),
                "current_avg_bleu": float(current_avg_bleu),
                "chrf_mean": float(chrf_mean),
                "chrf_std": float(chrf_std),
                "chrf_min": float(chrf_min),
                "chrf_max": float(chrf_max),
                "overall_chrf": float(overall_chrf),
                "current_avg_chrf": float(current_avg_chrf),
                "valid_samples": len(results)
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
                "sample_limit": sample_limit,
                "data_path": dataset_path,
                "direction": direction
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CoVoST2 Speech Translation Evaluation with Qwen3-Omni')
    parser.add_argument('--direction', type=str, choices=['en_zh', 'zh_en'], required=True,
                        help='Translation direction: en_zh (English to Chinese) or zh_en (Chinese to English)')
    parser.add_argument('--output-file', '-o', type=str, default=None,
                        help='Output file path for results JSON (overrides --output-dir)')
    parser.add_argument('--output-dir', '-d', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--timing-output-file', '-t', type=str, default=None,
                        help='Output file path for timing statistics JSON')
    parser.add_argument('--model-path', '-m', type=str, default=None,
                        help='Path to the model directory (default: from MODEL_PATH env or hardcoded path)')
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path to CoVoST2 dataset (default: from DATASET_PATH env or auto-detect)')
    args = parser.parse_args()
    
    direction = args.direction
    gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_id = gpu_temp[-1] if gpu_temp else "0"
    print(f"Using GPU ID: {gpu_id}")

    # Determine dataset path
    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        # Auto-detect based on direction
        base_datasets_dir = "/mnt/users/wangyuxuan-20250915/Audio/Datasets"
        if direction == "en_zh":
            dataset_path = os.path.join(base_datasets_dir, "covost2_en_zh_test")
        else:  # zh_en
            dataset_path = os.path.join(base_datasets_dir, "covost2_zh_en_test")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path does not exist: {dataset_path}")
        print(f"Please ensure the dataset is downloaded to: {dataset_path}")
        return
    
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
    if args.output_file:
        output_file = os.path.abspath(args.output_file)
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
    else:
        if args.output_dir:
            result_dir = args.output_dir
        else:
            result_dir = os.environ.get("RESULTS_DIR", f'./CoVoST2_{direction.upper()}_Results')
        
        result_dir = os.path.abspath(result_dir)
        os.makedirs(result_dir, exist_ok=True)
        output_file = f'{result_dir}/covost2_{direction}_results_qwen3.json'
    
    # Determine timing output file path
    if args.timing_output_file:
        timing_output_file = os.path.abspath(args.timing_output_file)
        timing_output_dir = os.path.dirname(timing_output_file)
        os.makedirs(timing_output_dir, exist_ok=True)
    else:
        timing_output_dir = os.path.dirname(output_file)
        timing_output_file = f'{timing_output_dir}/timing_stats_qwen3_{method_is}_{prune_ratio}.json'
    
    print(f"Results will be saved to: {output_file} (updated in real-time)")
    print(f"Timing statistics will be saved to: {timing_output_file}")

    timing_stats = GlobalTimingStats()

    print(f"\n=== CoVoST2-{direction.upper()} Speech Translation Evaluation Configuration (Qwen3-Omni) ===")
    print(f"GPU ID: {gpu_id}")
    print(f"Direction: {direction}")
    print(f"Pruning ratio: {prune_ratio}")
    print(f"Pruning method: {method_is}")
    print(f"Dataset path: {dataset_path}")
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
    print("Loading Qwen3-Omni model...")
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
    
    # Configure sparse attention
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
        model.thinker.model.config.sparse_attention_config = {'prune_ratio': prune_ratio, 'prune_method': prune_method}
        print(f"Sparse attention config set: prune_ratio={prune_ratio}, prune_method={prune_method}")
    else:
        print("Warning: thinker model config not found, using default parameters")
    
    # Initialize config parameters
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
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            base_model = model.thinker.model
            num_hidden_layers = base_model.config.num_hidden_layers
            num_attention_heads = base_model.config.num_attention_heads
            
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
            
            if ENV_BUDGET_IS_RATIO:
                init_budget = 4096
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
    print(f"Loading CoVoST2-{direction} dataset from: {dataset_path}")
    dataset = load_covost2_dataset(dataset_path, direction)
    
    if not dataset:
        print("Error: Failed to load any data")
        return
    
    if sample_limit > 0:
        # Use select() for HuggingFace datasets instead of slicing
        # Slicing with small numbers returns a dict instead of Dataset
        # Limit to min(sample_limit, len(dataset)) to handle cases where dataset is smaller
        original_size = len(dataset)
        actual_limit = min(sample_limit, original_size)
        if hasattr(dataset, 'select'):
            dataset = dataset.select(range(actual_limit))
        else:
            dataset = dataset[:actual_limit]
        print(f"Applying sample limit, processing {len(dataset)} samples (limit: {sample_limit}, original dataset size: {original_size})")

    # Select instructions based on direction
    if direction == "en_zh":
        instructions = ST_INSTRUCTIONS_EN_ZH
    else:
        instructions = ST_INSTRUCTIONS_ZH_EN

    results = []
    total_bleu = 0.0
    total_chrf = 0.0
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

    progress_bar = tqdm(enumerate(dataset), total=len(dataset), desc=f"CoVoST2-{direction.upper()} ST evaluation (Qwen3)", **tqdm_kwargs)

    for idx, doc in progress_bar:
        try:
            # Reset SparseMM state before each sample
            if USE_SPARSEKV and sparsemm_enabled:
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
            
            audio_data_result = covost2_doc_to_audio(doc)
            if audio_data_result is None:
                continue
            
            audio_np, sr = audio_data_result
            
            # Get reference translation
            reference = doc.get("answer", "")
            if not reference:
                reference = doc.get("reference", "")
            if not reference:
                reference = doc.get("translation", "")
            
            # Select random instruction (using improved prompts)
            instruction = random.choice(instructions)
            
            # Prepare messages with improved system message and prompt
            # System message explicitly defines the model as a speech translation model
            # User prompt is more direct and explicit about the task
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": ST_SYSTEM_MESSAGE}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_np},
                        {"type": "text", "text": instruction}
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
            
            # Initialize cache for SparseMM if enabled
            past_key_values = None
            if USE_SPARSEKV and sparsemm_enabled:
                past_key_values = DynamicCacheSplitHeadFlatten()
            
            # Measure prefill time
            prefill_start_event = torch.cuda.Event(enable_timing=True)
            prefill_end_event = torch.cuda.Event(enable_timing=True)
            
            prefill_start_event.record()
            with torch.no_grad():
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
            
            prediction = clean_translation_response(resp)
            
            # Calculate decode token count (generated tokens only, excluding input)
            generated_length = generated_sequences.shape[1]
            decode_token_count = generated_length - input_length
            
            # Calculate TPS
            sample_tps = decode_token_count / decode_time if decode_time > 0 else 0.0
            
            # Calculate BLEU and chrF for this sample (using single reference)
            sample_bleu = calculate_bleu_score([prediction], [reference], direction)
            sample_chrf = calculate_chrf_score([prediction], [reference])
            
            total_bleu += sample_bleu
            total_chrf += sample_chrf
            processed_samples += 1
            
            current_avg_bleu = total_bleu / processed_samples
            current_avg_chrf = total_chrf / processed_samples
            
            print(f"\n[Sample {idx+1}/{len(dataset)}]")
            print(f"  BLEU (current): {sample_bleu:.2f}")
            print(f"  chrF (current): {sample_chrf:.2f}")
            print(f"  Average BLEU: {current_avg_bleu:.2f}")
            print(f"  Average chrF: {current_avg_chrf:.2f}")
            print(f"  Reference: {reference[:100]}...")
            print(f"  Prediction: {prediction[:100]}...")
            
            progress_bar.set_postfix({
                'BLEU': f'{sample_bleu:.2f}',
                'chrF': f'{sample_chrf:.2f}',
                'Avg BLEU': f'{current_avg_bleu:.2f}',
                'Avg chrF': f'{current_avg_chrf:.2f}',
                'Sample': f'{idx+1}/{len(dataset)}'
            })
            
            results.append({
                "idx": idx,
                "id": doc.get("id", f"sample_{idx}"),
                "reference": reference,
                "prediction": prediction,
                "bleu": sample_bleu,
                "chrf": sample_chrf,
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
            save_interval = 1
            if (idx + 1) % save_interval == 0 or (idx + 1) == len(dataset):
                save_results_realtime(
                    output_file, 
                    results, 
                    total_bleu,
                    total_chrf,
                    processed_samples,
                    timing_stats,
                    gpu_id,
                    prune_ratio,
                    method_is,
                    sample_limit,
                    dataset_path,
                    len(dataset),
                    direction
                )
            
            # Clear cache
            if USE_SPARSEKV and sparsemm_enabled:
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

    # Calculate final metrics
    predictions = [r.get("prediction", "") for r in results]
    references = [r.get("reference", "") for r in results]
    
    # Calculate final BLEU and chrF (functions handle empty values internally)
    final_bleu = calculate_bleu_score(predictions, references, direction)
    final_chrf = calculate_chrf_score(predictions, references)
    
    # Calculate detailed metrics
    bleu_scores = [r.get("bleu", 0.0) for r in results if "bleu" in r]
    if len(bleu_scores) > 0:
        bleu_mean = np.mean(bleu_scores)
        bleu_std = np.std(bleu_scores)
        bleu_min = np.min(bleu_scores)
        bleu_max = np.max(bleu_scores)
    else:
        bleu_mean = 0.0
        bleu_std = 0.0
        bleu_min = 0.0
        bleu_max = 0.0
    
    # Calculate detailed chrF metrics
    chrf_scores = [r.get("chrf", 0.0) for r in results if "chrf" in r]
    if len(chrf_scores) > 0:
        chrf_mean = np.mean(chrf_scores)
        chrf_std = np.std(chrf_scores)
        chrf_min = np.min(chrf_scores)
        chrf_max = np.max(chrf_scores)
    else:
        chrf_mean = 0.0
        chrf_std = 0.0
        chrf_min = 0.0
        chrf_max = 0.0
    
    final_results = {
        "model_name": f"Qwen3-Omni",
        "dataset": f"CoVoST2-{direction.upper()}",
        "task": "Speech Translation",
        "status": "completed",
        "progress": {
            "processed": len(results),
            "total": len(dataset),
            "percentage": 100.0
        },
        "total_samples": len(results),
        "valid_samples": len(results),
        "bleu": float(final_bleu),
        "bleu_mean": float(bleu_mean),
        "bleu_std": float(bleu_std),
        "bleu_min": float(bleu_min),
        "bleu_max": float(bleu_max),
        "chrf": float(final_chrf),
        "chrf_mean": float(chrf_mean),
        "chrf_std": float(chrf_std),
        "chrf_min": float(chrf_min),
        "chrf_max": float(chrf_max),
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
            "data_path": dataset_path,
            "direction": direction
        },
        "detailed_results": results
    }
    
    final_results = convert_numpy_types(final_results)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    timing_stats.export_to_json(timing_output_file)

    print(f"\n=== CoVoST2-{direction.upper()} Speech Translation Evaluation Results Summary (Qwen3-Omni) ===")
    print(f"Total samples: {len(results)}")
    print(f"Processed samples: {processed_samples}")
    print(f"\n=== BLEU Metrics ===")
    print(f"BLEU Score: {final_bleu:.2f}")
    print(f"BLEU Mean: {bleu_mean:.2f}")
    print(f"BLEU Std: {bleu_std:.2f}")
    print(f"BLEU Min: {bleu_min:.2f}")
    print(f"BLEU Max: {bleu_max:.2f}")
    print(f"\n=== chrF Metrics ===")
    print(f"chrF Score: {final_chrf:.2f}")
    print(f"chrF Mean: {chrf_mean:.2f}")
    print(f"chrF Std: {chrf_std:.2f}")
    print(f"chrF Min: {chrf_min:.2f}")
    print(f"chrF Max: {chrf_max:.2f}")
    
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

