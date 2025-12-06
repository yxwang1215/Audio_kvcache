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

sys.path.append("/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Modeling")
sys.path.append("/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/SparseMM")
sys.path.append("/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/AudioAdaptiveKVCache")
sys.path.append("/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Modeling/Qwen2_5_Omni")

# 根据环境变量决定是否使用 sparsekv 版本
USE_SPARSEKV = os.environ.get("USE_SPARSEKV", "true").lower() == "true"

if USE_SPARSEKV:
    from modeling_qwen2_5_omni_sparsekv import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniFlashAttention2SparseMM
    print("Using sparsekv version of the model")
else:
    from modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
    print("Using standard version of the model (sparsekv disabled)")

from processing_qwen2_5_omni import Qwen2_5OmniProcessor

sys.path.append("/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Effallm/utils")
from qwen_omni_utils import process_mm_info

# Import SparseMM utilities
from sparsemm.sparsemm_utils import (
    SparseMM,
    DynamicCacheSplitHeadFlatten,
    init_sparsemm
)
from transformers.cache_utils import DynamicCache

# Import audio head utilities for SparseMM
from audio_head_utils import (
    load_audio_head_scores,
    process_audio_head_scores,
    get_audio_head_score_path
)


_AUDIO_TOKEN_ID = 151646        
_AUDIO_BOS_TOKEN_ID = 151647      
_AUDIO_EOS_TOKEN_ID = 151648      

# Environment variables for SparseMM
# BUDGET can be either a ratio (0.0-1.0) or an absolute value (>=1)
# If < 1.0, it's treated as a ratio of prefill input tokens
# If >= 1.0, it's treated as an absolute budget value
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

logging.set_verbosity_error()
warnings.filterwarnings("ignore")


current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
sys.path.insert(0, current_dir)

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  
        reserved = torch.cuda.memory_reserved() / 1024**3    
        return allocated, reserved
    return 0, 0

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
        
    def add_sample(self, wall_time, prefill_time, total_gpu_time):
        """Add sample timing record, skip the first sample"""
        
        if not self.first_sample_skipped:
            self.first_sample_skipped = True
            return
            
        record = {
            "wall_time": wall_time,
            "prefill_time": prefill_time,
            "total_gpu_time": total_gpu_time
        }
        
        self.timing_records.append(record)
    
    def get_summary(self):
        """Get statistical summary"""
        if len(self.timing_records) == 0:
            return {
                "count": 0,
                "avg_wall_time": 0.0,
                "avg_prefill_time": 0.0,
                "avg_total_gpu_time": 0.0
            }
        
        total_wall = sum(r["wall_time"] for r in self.timing_records)
        total_prefill = sum(r["prefill_time"] for r in self.timing_records)
        total_gpu = sum(r["total_gpu_time"] for r in self.timing_records)
        count = len(self.timing_records)
        
        return {
            "count": count,
            "avg_wall_time": total_wall / count,
            "avg_prefill_time": total_prefill / count,
            "avg_total_gpu_time": total_gpu / count
        }
    
    def export_to_json(self, output_file):
        """Export statistical data to JSON file - HAD style"""
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def export_to_json(self, output_file):
        """Export statistical data to JSON file"""
        result = {
            "global_summary": self.get_summary(),
            "detailed_records": self.timing_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_file

def load_jsonl_dataset(jsonl_path, wav_path_col="WavPath", ref_col="Transcript", id_col=None):
    """
    Load UltraEval-Audio style JSONL dataset
    
    Args:
        jsonl_path: Path to JSONL file
        wav_path_col: Column name for audio file path (default: "WavPath")
        ref_col: Column name for reference transcription/answer (default: "Transcript")
        id_col: Column name for sample ID (optional, will use index if not provided)
    
    Returns:
        List of dataset samples in the same format as load_librispeech_long_dataset
    """
    dataset = []
    
    if not os.path.exists(jsonl_path):
        print(f"Error: JSONL file does not exist: {jsonl_path}")
        return []
    
    print(f"Loading JSONL dataset from: {jsonl_path}")
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    sample = json.loads(line)
                    
                    # Get audio path
                    audio_path = sample.get(wav_path_col)
                    if not audio_path:
                        print(f"Warning: Sample {idx} missing {wav_path_col}, skipping")
                        continue
                    
                    # Handle relative paths - if path doesn't exist, try relative to JSONL file directory
                    if not os.path.isabs(audio_path) and not os.path.exists(audio_path):
                        jsonl_dir = os.path.dirname(os.path.abspath(jsonl_path))
                        audio_path = os.path.join(jsonl_dir, audio_path)
                    
                    if not os.path.exists(audio_path):
                        print(f"Warning: Audio file not found for sample {idx}: {audio_path}, skipping")
                        continue
                    
                    # Get reference transcription/answer
                    transcription = sample.get(ref_col, "")
                    if not transcription:
                        # Try common alternative column names
                        for alt_col in ["text", "Text", "transcription", "Transcription", "answer", "Answer"]:
                            if alt_col in sample:
                                transcription = sample[alt_col]
                                break
                    
                    # Get sample ID
                    sample_id = sample.get(id_col) if id_col and id_col in sample else f"sample_{idx}"
                    
                    # Get duration
                    try:
                        audio_info = sf.info(audio_path)
                        duration = audio_info.duration
                    except:
                        duration = 0.0
                    
                    # Get speaker_id if available (for compatibility)
                    speaker_id = sample.get("speaker_id", sample.get("speaker", "unknown"))
                    
                    dataset.append({
                        "id": str(sample_id),
                        "speaker_id": str(speaker_id),
                        "chapter_id": sample.get("chapter_id", ""),
                        "path": audio_path,
                        "transcription": str(transcription) if transcription else "",
                        "duration": duration,
                        "audio": {
                            "path": audio_path,
                            "array": None,
                            "sampling_rate": None
                        }
                    })
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON on line {idx}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing sample {idx}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
        traceback.print_exc()
        return []
    
    print(f"Loaded {len(dataset)} audio samples from JSONL")
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

def asr_doc_to_text(doc, kwargs=None):
    """Generate prompts for English ASR tasks"""
    if kwargs is None:
        kwargs = {}
    
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    
    
    instruction = "Transcribe this audio accurately. Remove hesitation words like 'um', 'uh'."
    format_text = "Your response should be formatted as follows: Spoken Content:"
    
    
    prompt_text = f"{pre_prompt}{instruction} {format_text} <transcribed text here>{post_prompt}"
    
    return prompt_text

def clean_response(response):
    """Clean ASR response"""
    if not response or response.strip() == "":
        return ""
    
    
    for marker in ["spoken content:", "content:", "transcription:", "transcript:"]:
        if marker.lower() in response.lower():
            parts = re.split(re.escape(marker), response, flags=re.IGNORECASE)
            if len(parts) > 1:
                response = parts[1].strip()
                break
    
    
    response = re.sub(r'<transcribed text here>', '', response)
    response = re.sub(r'<sep>.*?($|<|$)', '', response)
    response = re.sub(r'(?i)^(spoken\s+(?:text|content)|content|transcript|transcription):\s*', '', response.strip())
    
    return response.strip()

def standardize_text(text):
    """Normalize text for fair comparison and WER calculation
    
    This function standardizes text by:
    1. Converting to lowercase
    2. Removing punctuation marks
    3. Normalizing whitespace
    
    This ensures fair WER comparison between reference (often lowercase, no punctuation)
    and hypothesis (often has capitalization and punctuation from the model).
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation marks
    text = re.sub(r'[.!?,;:"()\[\]{}]', ' ', text)
    text = re.sub(r'[\-\']', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def main():
    # Parse command line arguments - UltraEval-Audio compatible interface
    parser = argparse.ArgumentParser(description='Qwen2.5-Omni with SparseMM for UltraEval-Audio')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--speech', action='store_true', default=False,
                        help='Whether to use speech output')
    parser.add_argument('--dataset', type=str, default='',
                        help='Dataset name (e.g., librispeech-test-clean, clotho-aqa). If provided, will use UltraEval-Audio registry to load dataset automatically.')
    args = parser.parse_args()
    
    gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_id = gpu_temp[-1] if gpu_temp else "0"
    print(f"Using GPU ID: {gpu_id}")

    
    prune_layer_idx = int(os.environ.get("PRUNE_LAYER_IDX", 1))
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

    
    print("Loading Qwen2.5-Omni model...")
    model_path = args.path
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
    # SAME as LibriSpeech_qwen2.5.py - always disable talker
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
                # This should be large enough to avoid errors during initialization
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
            
            model.enable_sparsemm_attention(head_score_data, sparsemm_config)
            sparsemm_enabled = True

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print("Model loaded from checkpoint: {}".format(model_path), flush=True)
    
    # If dataset name is provided, load dataset using UltraEval-Audio registry
    if args.dataset:
        # Add UltraEval-Audio to path to use registry
        # Try multiple possible paths
        possible_paths = [
            os.environ.get("ULTRAEVAL_AUDIO_ROOT"),
            "/mnt/users/wangyuxuan-20250915/Audio/UltraEval-Audio",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "UltraEval-Audio"),
        ]
        
        ultraeval_root = None
        for path in possible_paths:
            if path and os.path.exists(path) and os.path.exists(os.path.join(path, "audio_evals")):
                ultraeval_root = path
                break
        
        if ultraeval_root and ultraeval_root not in sys.path:
            sys.path.insert(0, ultraeval_root)
        
        try:
            from audio_evals.registry import registry
            
            print(f"Loading dataset '{args.dataset}' from UltraEval-Audio registry...", flush=True)
            dataset_obj = registry.get_dataset(args.dataset)
            if dataset_obj is None:
                print(f"Error: Dataset '{args.dataset}' not found in registry", flush=True)
                sys.exit(1)
            
            # Load dataset data
            dataset_data = dataset_obj.load(limit=0)
            print(f"Loaded {len(dataset_data)} samples from dataset '{args.dataset}'", flush=True)
            
            # Process dataset samples
            # Get UltraEval-Audio root directory for resolving relative paths
            ultraeval_root_for_path = ultraeval_root if ultraeval_root else os.getcwd()
            
            # Initialize WER tracking
            wer_accumulator = {
                'total_samples': 0,
                'total_errors': 0,
                'total_words': 0,
                'all_references': [],
                'all_hypotheses': [],
                'all_wers': []  # Store individual sample WERs for averaging
            }
            
            for idx, sample in enumerate(dataset_data):
                try:
                    # Convert sample to conversation format
                    # Get audio path from sample
                    audio_path = None
                    if 'WavPath' in sample:
                        audio_path = sample['WavPath']
                    elif 'audio' in sample and isinstance(sample['audio'], dict):
                        audio_path = sample['audio'].get('path') or sample['audio'].get('WavPath')
                    elif 'audio_path' in sample:
                        audio_path = sample['audio_path']
                    
                    # Convert relative path to absolute path if needed
                    original_path = audio_path if audio_path else None
                    if audio_path:
                        resolved = False
                        # If path is relative, try to resolve it relative to UltraEval-Audio root
                        if not os.path.isabs(audio_path):
                            # Try multiple possible base directories
                            possible_bases = [
                                ultraeval_root_for_path,
                                os.getcwd(),
                                os.path.dirname(os.path.abspath(__file__)),
                            ]
                            
                            for base_dir in possible_bases:
                                if not base_dir:
                                    continue
                                abs_path = os.path.normpath(os.path.join(base_dir, audio_path))
                                if os.path.exists(abs_path):
                                    audio_path = abs_path
                                    resolved = True
                                    break
                        else:
                            # Already absolute path, just check if it exists
                            if os.path.exists(audio_path):
                                resolved = True
                    
                    if not audio_path or not os.path.exists(audio_path):
                        print(f"Warning: Sample {idx} missing or invalid audio path: {audio_path} (original: {original_path}, ultraeval_root: {ultraeval_root_for_path}, cwd: {os.getcwd()})", flush=True)
                        continue
                    
                    # Get reference text (if available)
                    ref_text = sample.get(dataset_obj.ref_col, sample.get('text', sample.get('answer', '')))
                    
                    # Generate ASR prompt text (same as reference file)
                    # Create a doc-like object for asr_doc_to_text function
                    doc_for_prompt = {"transcription": ref_text} if ref_text else {}
                    prompt_text = asr_doc_to_text(doc_for_prompt)
                    
                    # Task instruction for ASR
                    task_instruction = "You are a helpful assistant that transcribes speech audio. Please listen carefully and provide the exact transcription of what is spoken in the audio."
                    full_user_prompt = f"{task_instruction}\n\n{prompt_text}"
                    
                    # Load audio into numpy array - SAME as LibriSpeech_qwen2.5.py
                    # This is the key difference that affects accuracy!
                    audio_np = prepare_audio_for_qwen_omni(audio_path, target_sr=16000)
                    
                    # Build conversation format - SAME as LibriSpeech_qwen2.5.py
                    conversation = [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                                }
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "audio",
                                    "audio": audio_np  # Use numpy array like LibriSpeech_qwen2.5.py
                                },
                                {
                                    "type": "text",
                                    "text": full_user_prompt
                                }
                            ]
                        }
                    ]
                    
                    # Reset SparseMM state before each inference
                    if USE_SPARSEKV and sparsemm_enabled:
                        reset_sparsemm_state(model)
                        clear_model_cache(model)
                    
                    # Process conversation
                    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
                    text = processor.apply_chat_template(
                        conversation, add_generation_prompt=True, tokenize=False
                    )
                    
                    if isinstance(text, list):
                        text = text[0] if len(text) > 0 else ""
                    
                    # SAME as LibriSpeech_qwen2.5.py - use use_audio_in_video=True
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
                    
                    # Update SparseMM budget based on prefill token count if budget is a ratio
                    if USE_SPARSEKV and sparsemm_enabled and ENV_BUDGET_IS_RATIO:
                        if hasattr(inputs, 'input_ids'):
                            prefill_token_count = inputs.input_ids.shape[1] if len(inputs.input_ids.shape) > 1 else inputs.input_ids.shape[0]
                            update_sparsemm_budget(model, prefill_token_count)
                    
                    # Configure audio pruning if audio tokens detected
                    if hasattr(inputs, 'input_ids'):
                        token_ids = inputs.input_ids[0].tolist()
                        bos_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_BOS_TOKEN_ID]
                        eos_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_EOS_TOKEN_ID]
                        
                        if bos_positions and eos_positions:
                            audio_token_start = bos_positions[0]
                            audio_token_end = eos_positions[0]
                            audio_token_length = audio_token_end - audio_token_start + 1
                            
                            model.thinker.model.config.image_layer_idx = False
                            model.thinker.model.config.audio_layer_idx = prune_layer_idx
                            model.thinker.model.config.audio_token_num = audio_token_length
                            model.thinker.model.config.audio_token_start = audio_token_start
                            model.thinker.model.config.audio_prune_ratio = prune_ratio
                            model.thinker.model.config.random = use_random
                            model.thinker.model.config.frame = use_frame
                        else:
                            model.thinker.model.config.audio_layer_idx = None
                            model.thinker.model.config.audio_prune_ratio = 0
                    
                    # Initialize cache for SparseMM if enabled
                    past_key_values = None
                    if USE_SPARSEKV and sparsemm_enabled:
                        past_key_values = DynamicCacheSplitHeadFlatten()
                    
                    # Generate - use OFFICIAL parameters when not using SparseMM
                    # Official UltraEval-Audio does NOT use thinker_max_new_tokens
                    with torch.no_grad():
                        if args.speech:
                            if USE_SPARSEKV and sparsemm_enabled:
                                generate_result = model.generate(
                                    **inputs, 
                                    use_audio_in_video=True, 
                                    return_audio=True,
                                    thinker_do_sample=False,
                                    past_key_values=past_key_values
                                )
                            else:
                                # Official implementation - no extra parameters
                                generate_result = model.generate(
                                    **inputs, 
                                    use_audio_in_video=True,
                                    thinker_do_sample=False
                                )
                            text_ids, audio = generate_result
                            text = processor.batch_decode(
                                text_ids,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            )
                            full_text = text[0] if isinstance(text, list) and len(text) > 0 else ""
                        else:
                            if USE_SPARSEKV and sparsemm_enabled:
                                text_ids = model.generate(
                                    **inputs,
                                    use_audio_in_video=True,
                                    return_audio=False,
                                    thinker_max_new_tokens=1100,
                                    thinker_do_sample=False,
                                    pad_token_id=processor.tokenizer.eos_token_id,
                                    past_key_values=past_key_values
                                )
                            else:
                                # SAME as LibriSpeech_qwen2.5.py - use thinker_max_new_tokens and pad_token_id
                                text_ids = model.generate(
                                    **inputs,
                                    use_audio_in_video=True,
                                    return_audio=False,
                                    thinker_max_new_tokens=1100,
                                    thinker_do_sample=False,
                                    pad_token_id=processor.tokenizer.eos_token_id
                                )
                            text = processor.batch_decode(
                                text_ids,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            )
                            full_text = text[0] if isinstance(text, list) and len(text) > 0 else ""
                    
                    # Extract response - SAME as LibriSpeech_qwen2.5.py
                    # Use "assistant\n" (with newline) not just "assistant"
                    generated_text = full_text
                    if "assistant\n" in generated_text:
                        assistant_start = generated_text.rfind("assistant\n") + len("assistant\n")
                        generated_text = generated_text[assistant_start:].strip()
                    
                    # Clean up the generated text using clean_response function
                    generated_text = clean_response(generated_text)
                    
                    # Calculate WER if reference text is available
                    if ref_text and generated_text:
                        try:
                            # Standardize both reference and hypothesis for fair WER comparison
                            # This handles case differences and punctuation
                            ref_standardized = standardize_text(ref_text)
                            hyp_standardized = standardize_text(generated_text)
                            
                            # Calculate WER for current sample using standardized text
                            current_wer = jiwer.wer(ref_standardized, hyp_standardized)
                            
                            # Print reference and generated text for debugging
                            print(f"\n[Sample {idx+1}/{len(dataset_data)}]", flush=True)
                            print(f"Reference: {ref_text.strip()}", flush=True)
                            print(f"Generated: {generated_text}", flush=True)
                            
                            # Update accumulator with standardized text for consistent WER calculation
                            wer_accumulator['all_references'].append(ref_standardized)
                            wer_accumulator['all_hypotheses'].append(hyp_standardized)
                            wer_accumulator['all_wers'].append(current_wer)
                            wer_accumulator['total_samples'] += 1
                            
                            # Calculate average WER and accuracy across all samples
                            avg_wer = sum(wer_accumulator['all_wers']) / len(wer_accumulator['all_wers'])
                            current_acc = 1.0 - current_wer
                            avg_acc = 1.0 - avg_wer
                            
                            # Print real-time accuracy (acc = 1 - WER)
                            print(f"Acc | Current: {current_acc*100:.2f}% | Avg: {avg_acc*100:.2f}% ({wer_accumulator['total_samples']} samples)\n", flush=True)
                        except Exception as e:
                            print(f"[WER] Error calculating WER for sample {idx+1}: {e}", flush=True)
                            import traceback
                            traceback.print_exc()
                    elif not ref_text:
                        print(f"[WER] Sample {idx+1}: No reference text available, skipping WER calculation", flush=True)
                    elif not generated_text:
                        print(f"[WER] Sample {idx+1}: No generated text, skipping WER calculation", flush=True)
                    
                    # Clear cache after inference
                    if USE_SPARSEKV and sparsemm_enabled and past_key_values is not None:
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
                    
                except Exception as e:
                    import traceback
                    print(f"Error processing sample {idx}: {e}", flush=True)
                    traceback.print_exc()
                    continue
            
            # Print final WER statistics
            if wer_accumulator['total_samples'] > 0:
                # Calculate average WER from individual sample WERs
                avg_wer = sum(wer_accumulator['all_wers']) / len(wer_accumulator['all_wers'])
                
                # Also calculate corpus-level WER for reference (all texts concatenated)
                corpus_wer = jiwer.wer(
                    wer_accumulator['all_references'],
                    wer_accumulator['all_hypotheses']
                )
                
                # Calculate other statistics
                import statistics
                wer_std = statistics.stdev(wer_accumulator['all_wers']) if len(wer_accumulator['all_wers']) > 1 else 0.0
                wer_min = min(wer_accumulator['all_wers'])
                wer_max = max(wer_accumulator['all_wers'])
                
                # Also calculate detailed metrics using standardized texts
                try:
                    measures = jiwer.compute_measures(
                        wer_accumulator['all_references'],
                        wer_accumulator['all_hypotheses']
                    )
                    # Calculate accuracy metrics (acc = 1 - WER)
                    avg_acc = 1.0 - avg_wer
                    corpus_acc = 1.0 - corpus_wer
                    acc_std = wer_std  # Same std for accuracy
                    acc_min = 1.0 - wer_max  # Min acc = 1 - max WER
                    acc_max = 1.0 - wer_min  # Max acc = 1 - min WER
                    
                    print(f"\n{'='*60}", flush=True)
                    print(f"[Final Statistics]", flush=True)
                    print(f"{'='*60}", flush=True)
                    print(f"Total samples processed: {wer_accumulator['total_samples']}", flush=True)
                    print(f"Average Accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)", flush=True)
                    print(f"Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)", flush=True)
                    print(f"Corpus Accuracy: {corpus_acc:.4f} ({corpus_acc*100:.2f}%)", flush=True)
                    print(f"Corpus WER: {corpus_wer:.4f} ({corpus_wer*100:.2f}%)", flush=True)
                    print(f"Accuracy Std: {acc_std:.4f} ({acc_std*100:.2f}%)", flush=True)
                    print(f"Accuracy Min: {acc_min:.4f} ({acc_min*100:.2f}%)", flush=True)
                    print(f"Accuracy Max: {acc_max:.4f} ({acc_max*100:.2f}%)", flush=True)
                    print(f"Hits: {measures.get('hits', 0)}", flush=True)
                    print(f"Substitutions: {measures.get('substitutions', 0)}", flush=True)
                    print(f"Deletions: {measures.get('deletions', 0)}", flush=True)
                    print(f"Insertions: {measures.get('insertions', 0)}", flush=True)
                    print(f"{'='*60}\n", flush=True)
                except Exception as e:
                    avg_acc = 1.0 - avg_wer
                    corpus_acc = 1.0 - corpus_wer
                    print(f"[Final Stats] Avg Acc: {avg_acc:.4f} ({avg_acc*100:.2f}%), Avg WER: {avg_wer:.4f} ({avg_wer*100:.2f}%), Corpus Acc: {corpus_acc:.4f} ({corpus_acc*100:.2f}%), Corpus WER: {corpus_wer:.4f} ({corpus_wer*100:.2f}%) - Error computing detailed metrics: {e}", flush=True)
            else:
                print(f"[WER] No samples with valid reference and generated text for WER calculation", flush=True)
            
            print(f"Completed processing {len(dataset_data)} samples from dataset '{args.dataset}'", flush=True)
            return
        
        except ImportError as e:
            print(f"Error: Cannot import UltraEval-Audio registry. Make sure UltraEval-Audio is in PYTHONPATH. Error: {e}", flush=True)
            print("Falling back to stdin/stdout interface...", flush=True)
        except Exception as e:
            print(f"Error loading dataset: {e}", flush=True)
            import traceback
            traceback.print_exc()
            print("Falling back to stdin/stdout interface...", flush=True)
    
    # UltraEval-Audio compatible stdin/stdout interface
    import select
    import uuid
    
    while True:
        try:
            prompt = input()
            
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    "Error: Invalid conversation format, must contains  ->, but {}".format(
                        prompt
                    ),
                    flush=True,
                )
                continue
            prefix = prompt[:anchor].strip() + "->"
            conversation = json.loads(prompt[anchor + 2 :])
            
            # CRITICAL FIX: Process audio fields in conversation before passing to process_mm_info
            # UltraEval-Audio sends audio as file paths or other formats in JSON, 
            # but process_mm_info expects numpy arrays or proper audio format
            # We need to ensure audio fields are properly formatted
            for msg in conversation:
                if msg.get("role") == "user" and "content" in msg:
                    for content_item in msg["content"]:
                        if content_item.get("type") == "audio":
                            audio_data = content_item.get("audio")
                            # If audio is a string (file path), load it as numpy array
                            if isinstance(audio_data, str):
                                if os.path.exists(audio_data):
                                    audio_np = prepare_audio_for_qwen_omni(audio_data, target_sr=16000)
                                    content_item["audio"] = audio_np
                            # If audio is already a list (from JSON deserialization of numpy array), convert back
                            elif isinstance(audio_data, list):
                                audio_np = np.array(audio_data, dtype=np.float32)
                                content_item["audio"] = audio_np
            
            # Reset SparseMM state before each inference
            if USE_SPARSEKV and sparsemm_enabled:
                reset_sparsemm_state(model)
                clear_model_cache(model)
            
            # Process conversation
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
            text = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            
            if isinstance(text, list):
                text = text[0] if len(text) > 0 else ""
            
            # SAME as LibriSpeech_qwen2.5.py - use use_audio_in_video=True
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
            
            # Update SparseMM budget based on prefill token count if budget is a ratio
            if USE_SPARSEKV and sparsemm_enabled and ENV_BUDGET_IS_RATIO:
                if hasattr(inputs, 'input_ids'):
                    prefill_token_count = inputs.input_ids.shape[1] if len(inputs.input_ids.shape) > 1 else inputs.input_ids.shape[0]
                    update_sparsemm_budget(model, prefill_token_count)
            
            # Configure audio pruning if audio tokens detected
            if hasattr(inputs, 'input_ids'):
                token_ids = inputs.input_ids[0].tolist()
                bos_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_BOS_TOKEN_ID]
                eos_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_EOS_TOKEN_ID]
                
                if bos_positions and eos_positions:
                    audio_token_start = bos_positions[0]
                    audio_token_end = eos_positions[0]
                    audio_token_length = audio_token_end - audio_token_start + 1
                    
                    model.thinker.model.config.image_layer_idx = False
                    model.thinker.model.config.audio_layer_idx = prune_layer_idx
                    model.thinker.model.config.audio_token_num = audio_token_length
                    model.thinker.model.config.audio_token_start = audio_token_start
                    model.thinker.model.config.audio_prune_ratio = prune_ratio
                    model.thinker.model.config.random = use_random
                    model.thinker.model.config.frame = use_frame
                else:
                    model.thinker.model.config.audio_layer_idx = None
                    model.thinker.model.config.audio_prune_ratio = 0
            
            # Initialize cache for SparseMM if enabled
            past_key_values = None
            if USE_SPARSEKV and sparsemm_enabled:
                past_key_values = DynamicCacheSplitHeadFlatten()
            
            # Print prefill input token count
            if hasattr(inputs, 'input_ids'):
                prefill_token_count = inputs.input_ids.shape[1] if len(inputs.input_ids.shape) > 1 else inputs.input_ids.shape[0]
                print(f"[Prefill] Input tokens = {prefill_token_count}", flush=True)
            
            # Generate - use OFFICIAL parameters when not using SparseMM
            with torch.no_grad():
                if args.speech:
                    if USE_SPARSEKV and sparsemm_enabled:
                        text_ids, audio = model.generate(
                            **inputs, 
                            use_audio_in_video=True, 
                            return_audio=True,
                            thinker_do_sample=False,
                            past_key_values=past_key_values
                        )
                    else:
                        # Official implementation
                        text_ids, audio = model.generate(
                            **inputs, 
                            use_audio_in_video=True,
                            thinker_do_sample=False
                        )
                    text = processor.batch_decode(
                        text_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        sf.write(
                            f.name,
                            audio.reshape(-1).detach().cpu().numpy(),
                            samplerate=24000,
                        )
                        # Output for speech mode
                        print(prefix + json.dumps({"text": text[0], "audio": f.name}), flush=True)
                else:
                    if USE_SPARSEKV and sparsemm_enabled:
                        text_ids = model.generate(
                            **inputs,
                            use_audio_in_video=True,
                            return_audio=False,
                            thinker_max_new_tokens=1100,
                            thinker_do_sample=False,
                            pad_token_id=processor.tokenizer.eos_token_id,
                            past_key_values=past_key_values
                        )
                    else:
                        # SAME as LibriSpeech_qwen2.5.py - use thinker_max_new_tokens and pad_token_id
                        text_ids = model.generate(
                            **inputs,
                            use_audio_in_video=True,
                            return_audio=False,
                            thinker_max_new_tokens=1100,
                            thinker_do_sample=False,
                            pad_token_id=processor.tokenizer.eos_token_id
                        )
                    text = processor.batch_decode(
                        text_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    
                    # Output full text - UltraEval-Audio will extract "assistant" part on receiving end
                    # This matches the expected format for UltraEval-Audio interface
                    full_text = text[0] if isinstance(text, list) and len(text) > 0 else ""
                    print(prefix + json.dumps({"text": full_text}), flush=True)
                    
            
            # Clear cache after inference
            if USE_SPARSEKV and sparsemm_enabled and past_key_values is not None:
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
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Error:" + str(e), flush=True)

if __name__ == "__main__":
    main()
    
'''
# Example 1: Using absolute budget value (>= 1.0)
CUDA_VISIBLE_DEVICES=0 \
USE_SPARSEKV=true \
BUDGET=4096 \
RATIO=0.5 \
WINDOW_SIZE=32 \
KERNEL_SIZE=5 \
POOLING=maxpool \
HEAD_SCORE_FILE=aggregated \
PYTHONPATH=/mnt/users/wangyuxuan-20250915/Audio/UltraEval-Audio:$PYTHONPATH \
python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv/LibriSpeech_qwen2.5_ultraeval.py \
  --path /mnt/users/wangyuxuan-20250915/Models/Qwen2.5-Omni-3B \
  --dataset librispeech-test-clean

# Example 2: Using budget as ratio of prefill tokens (< 1.0, e.g., 0.3 = 30%)
CUDA_VISIBLE_DEVICES=3 \
USE_SPARSEKV=true \
BUDGET=0.7 \
RATIO=0.4 \
WINDOW_SIZE=16 \
KERNEL_SIZE=5 \
POOLING=maxpool \
HEAD_SCORE_FILE=aggregated \
PYTHONPATH=/mnt/users/wangyuxuan-20250915/Audio/UltraEval-Audio:$PYTHONPATH \
python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv/LibriSpeech_qwen2.5_ultraeval.py \
  --path /mnt/users/wangyuxuan-20250915/Models/Qwen2.5-Omni-3B \
  --dataset librispeech-test-clean
  
CUDA_VISIBLE_DEVICES=3 \
USE_SPARSEKV=false \
PYTHONPATH=/mnt/users/wangyuxuan-20250915/Audio/UltraEval-Audio:$PYTHONPATH \
python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv/LibriSpeech_qwen2.5_ultraeval.py \
  --path /mnt/users/wangyuxuan-20250915/Models/Qwen2.5-Omni-3B \
  --dataset librispeech-test-clean

'''