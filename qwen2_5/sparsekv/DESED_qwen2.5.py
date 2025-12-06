"""
DESED with Qwen2.5-Omni + Audio Adaptive KVCache
Integrates audio head scores with SparseMM adaptive KVCache allocation
"""
import os
import sys
import json
import time
import gc
import argparse
import warnings
import random
import traceback
import contextlib
import re
import numpy as np
import torch
import librosa
import soundfile as sf
import transformers
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Add paths
sys.path.append("/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Modeling")
sys.path.append("/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/SparseMM")
sys.path.append("/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/AudioAdaptiveKVCache")
sys.path.append("/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Modeling/Qwen2_5_Omni")

os.environ['NUMEXPR_MAX_THREADS'] = '64'
random.seed(42) 

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

# Import local utilities
from audio_head_utils import (
    load_audio_head_scores,
    process_audio_head_scores,
    get_audio_head_score_path
)

from modeling_qwen2_5_omni_sparsekv import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniFlashAttention2SparseMM
from processing_qwen2_5_omni import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from sparsemm.sparsemm_utils import (
    SparseMM,
    DynamicCacheSplitHeadFlatten,
    init_sparsemm
)
from transformers.cache_utils import DynamicCache

# Environment variables
ENV_BUDGET = int(os.environ.get("BUDGET", 4096))  
ENV_RATIO = float(os.environ.get("RATIO", 0.2))
ENV_WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", 64))  # Default 64, safe for most cases
ENV_KERNEL_SIZE = int(os.environ.get("KERNEL_SIZE", 7))
ENV_POOLING = os.environ.get("POOLING", "maxpool")
ENV_HEAD_SCORE_FILE = os.environ.get("HEAD_SCORE_FILE", "aggregated")
ENV_HEAD_SCORE_BATCH = os.environ.get("HEAD_SCORE_BATCH", None)
SAMPLE_LIMIT = int(os.environ.get("SAMPLE_LIMIT", 0))
RESULTS_DIR_ENV = os.environ.get("RESULTS_DIR", "DESED_AudioAdaptiveKVCache_Results")

# Debug: Output environment variables
print(f"[BUDGET DEBUG] ===== Environment Variables =====")
print(f"[BUDGET DEBUG] BUDGET (from env) = {os.environ.get('BUDGET', 'NOT SET (using default 4096)')}")
print(f"[BUDGET DEBUG] ENV_BUDGET (parsed) = {ENV_BUDGET}")
print(f"[BUDGET DEBUG] ENV_RATIO = {ENV_RATIO}")
print(f"[BUDGET DEBUG] ENV_WINDOW_SIZE = {ENV_WINDOW_SIZE}")
print(f"[BUDGET DEBUG] ENV_KERNEL_SIZE = {ENV_KERNEL_SIZE}")
print(f"[BUDGET DEBUG] ENV_POOLING = {ENV_POOLING}")

_AUDIO_TOKEN_ID = 151646          # '<|AUDIO|>'
_AUDIO_BOS_TOKEN_ID = 151647      # '<|audio_bos|>'
_AUDIO_EOS_TOKEN_ID = 151648      # '<|audio_eos|>'


def parse_args():
    parser = argparse.ArgumentParser(description="DESED with Qwen2.5-Omni + Audio Adaptive KVCache")
    parser.add_argument("--model-path", type=str, default="/mnt/users/wangyuxuan-20250915/Models/Qwen2.5-Omni-3B")
    parser.add_argument("--dataset-file", type=str, default="/mnt/users/wangyuxuan-20250915/Datasets/AudioMarathon/DESED/DESED_dataset/concatenated_audio/desed_sound_event_detection_task.json")
    parser.add_argument("--audio-base-dir", type=str, default="/mnt/users/wangyuxuan-20250915/Datasets/AudioMarathon/DESED/DESED_dataset/concatenated_audio")
    parser.add_argument("--max-new-tokens", type=int, default=10, help="Maximum generated tokens")
    parser.add_argument("--head-score-file", type=str, default=None, help="Head score file type (aggregated, mean, etc.)")
    parser.add_argument("--head-score-batch", type=str, default=None, help="Head score batch identifier")
    parser.add_argument("--enable-sparsemm", action="store_true", help="Enable SparseMM attention replacement for adaptive KVCache")
    return parser.parse_args()


def convert_to_serializable(obj):
    """Recursively convert objects to JSON serializable format"""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif torch.is_tensor(obj):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)
    else:
        return str(obj) if obj is not None else None


def get_gpu_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0.0, 0.0


def get_head_adaptive_capacity(model, num_attention_heads, num_key_value_groups):
    """
    Get head_adaptive_capacity from SparseMM attention modules
    Returns torch.Tensor of shape [num_layers, num_heads] or None
    """
    base_model = model.thinker.model    # Qwen2_5OmniThinkerTextModel
    if not hasattr(base_model, 'layers') or len(base_model.layers) == 0:
        return None  
    head_capacities = []
    for layer_idx, layer in enumerate(base_model.layers):
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            if hasattr(attn, 'head_adaptive_capacity') and attn.head_adaptive_capacity is not None:
                head_capacities.append(attn.head_adaptive_capacity.cpu())
            elif hasattr(attn, 'kv_cluster') and hasattr(attn.kv_cluster, 'base_capacity'):
                num_kv_heads = num_attention_heads // num_key_value_groups
                base_cap = attn.kv_cluster.base_capacity + attn.kv_cluster.window_size
                head_capacities.append(torch.full((num_kv_heads,), base_cap, dtype=torch.int32))
    if head_capacities:
        return torch.stack(head_capacities, dim=0)
    return None


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
    Optimized: Only clear if cache exists to avoid unnecessary iterations
    """
    # Clear any cached past_key_values in the model (fast checks)
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
    # Only iterate if we have SparseMM enabled (cache might exist)
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
        base_model = model.thinker.model
        if hasattr(base_model, 'layers'):
            # Early exit if no layers
            if len(base_model.layers) == 0:
                return
            # Only check first layer to see if we need to clear (optimization)
            first_layer = base_model.layers[0]
            if hasattr(first_layer, 'self_attn'):
                first_attn = first_layer.self_attn
                has_cache = (hasattr(first_attn, '_cache') and first_attn._cache is not None) or \
                           (hasattr(first_attn, 'cache') and first_attn.cache is not None)
                
                # Only iterate if cache exists
                if has_cache:
                    for layer in base_model.layers:
                        if hasattr(layer, 'self_attn'):
                            attn = layer.self_attn
                            # Check if there's a cached DynamicCacheSplitHeadFlatten object
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


def load_desed_qa_dataset(json_file, audio_base_dir):
    """
    Load data from new DESED task JSON file
    
    Args:
        json_file: DESED task JSON file path
        audio_base_dir: Audio file base directory
    
    Returns:
        dataset: List containing task data
    """
    dataset = []
    
    if not os.path.exists(json_file):
        print(f"Error: JSON file does not exist: {json_file}")
        return []
    
    print(f"Loading DESED task JSON: {json_file}")
    print(f"Audio base directory: {audio_base_dir}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 参考文件只接受字典格式，包含 'tasks' 字段
    if not isinstance(data, dict) or 'tasks' not in data:
        print(f"Error: JSON file format is incorrect, expected dictionary format containing 'tasks' field")
        return []
    
    tasks = data['tasks']
    print(f"Loaded {len(tasks)} tasks from JSON")
    
    task_type_stats = defaultdict(int)
    missing_files = 0
    
    for i, task in enumerate(tasks):
        # 使用相对路径，然后与 audio_base_dir 拼接
        relative_path = task.get("path", "")
        if relative_path:
            full_audio_path = os.path.join(audio_base_dir, relative_path)
        else:
            print(f"Warning: Task missing audio path: {task}")
            continue
        
        if not os.path.exists(full_audio_path):
            missing_files += 1
            if missing_files <= 5:  
                print(f"Warning: Audio file does not exist: {full_audio_path}")
            continue
        
        task_type = task.get("task_type", "unknown")
        question = task.get("question", "")
        answer_gt = task.get("answer_gt", "")
        
        choices = task.get("choices", {})
        choice_a = choices.get("A", "")
        choice_b = choices.get("B", "")
        choice_c = choices.get("C", "")
        choice_d = choices.get("D", "")
        
        # 读取音频信息（可选）
        audio_info = sf.info(full_audio_path)
        duration = audio_info.duration if audio_info else 0
        sample_rate = audio_info.samplerate if audio_info else 16000
        
        item = {
            "audio_path": full_audio_path,
            "question": question,
            "choice_a": choice_a,
            "choice_b": choice_b,
            "choice_c": choice_c,
            "choice_d": choice_d,
            "correct_answer": answer_gt,
            "task_type": task_type,
            "task_id": task.get("uniq_id", f"task_{i}"),
            "duration": duration,
            "sample_rate": sample_rate
        }
        
        dataset.append(item)
        task_type_stats[task_type] += 1
    
    if missing_files > 5:
        print(f"Warning: Total {missing_files} audio files do not exist")
    
    print(f"Loaded {len(dataset)} valid samples")
    print(f"Task type statistics: {dict(task_type_stats)}")
    return dataset


def extract_answer_choice(response):
    """Extract answer choice (A, B, C, D) from model response"""
    if not response:
        return "unknown"
    
    if '<|im_start|>assistant' in response:
        response = response.split('<|im_start|>assistant')[-1]
    elif 'assistant' in response.lower() and ':' in response:
        response = response.split(':', 1)[1]
    
    response = re.sub(r'<\|.*?\|>', '', response)
    response = re.sub(r'<.*?>', '', response)
    response = response.strip().upper()
    
    if len(response) > 50:
        response = response[:50]
    
    for choice in ['A', 'B', 'C', 'D']:
        if f' {choice} ' in f' {response} ' or response.startswith(f'{choice} ') or response.endswith(f' {choice}'):
            return choice
        if f'({choice})' in response or f'{choice})' in response:
            return choice
        if f'OPTION {choice}' in response or f'CHOICE {choice}' in response:
            return choice
    
    if response in ['A', 'B', 'C', 'D']:
        return response
    
    response_words = response.split()
    for word in response_words:
        if word in ['A', 'B', 'C', 'D']:
            return word
    
    return "unknown"


def evaluate_qa_accuracy(predicted_choice, ground_truth_choice):
    """Evaluate QA accuracy"""
    return predicted_choice.upper() == ground_truth_choice.upper()


def calculate_desed_metrics(y_true, y_pred):
    """
    Calculate detailed evaluation metrics for DESED sound event detection
    
    Args:
        y_true: True label list (A/B/C/D)
        y_pred: Predicted label list (A/B/C/D) 
        
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    
    valid_indices = []
    clean_y_true = []
    clean_y_pred = []
    
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label in ['A', 'B', 'C', 'D'] and pred_label in ['A', 'B', 'C', 'D']:
            valid_indices.append(i)
            clean_y_true.append(true_label)
            clean_y_pred.append(pred_label)
    
    if len(clean_y_true) == 0:
        return {
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'precision_weighted': 0.0,
            'recall_weighted': 0.0,
            'f1_weighted': 0.0,
            'classification_report': "No valid predictions",
            'valid_samples': 0,
            'total_samples': len(y_true)
        }
    
    
    accuracy = accuracy_score(clean_y_true, clean_y_pred)
    
    
    labels = ['A', 'B', 'C', 'D']
    precision, recall, f1, support = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, labels=labels, average=None, zero_division=0
    )
    
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, average='macro', zero_division=0
    )
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, average='weighted', zero_division=0
    )
    
    
    report = classification_report(
        clean_y_true, clean_y_pred, 
        labels=labels,
        target_names=['Choice A', 'Choice B', 'Choice C', 'Choice D'],
        zero_division=0,
        digits=4
    )
    
    return {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'classification_report': report,
        'valid_samples': len(clean_y_true),
        'total_samples': len(y_true)
    }


@dataclass
class DESEDTimingStats:
    """Timing statistics for DESED sound event detection tasks"""
    samples: int = 0
    total_prefill_time: float = 0.0
    total_time: float = 0.0  # Total time (prefill + decode), not separate decode time
    # Detailed timing breakdown
    total_audio_load_time: float = 0.0
    total_preprocessing_time: float = 0.0
    total_model_generation_time: float = 0.0
    total_postprocessing_time: float = 0.0
    total_output_tokens: int = 0
    total_input_tokens: int = 0
    total_audio_tokens: int = 0
    peak_gpu_memory: float = 0.0
    initial_gpu_memory: float = 0.0
    correct_predictions: int = 0
    detailed_records: List[Dict] = None
    
    def __post_init__(self):
        if self.detailed_records is None:
            self.detailed_records = []
    
    def record_initial_memory(self):
        """Record initial GPU memory usage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated, _ = get_gpu_memory_usage()
            self.initial_gpu_memory = allocated
            print(f"Initial GPU memory: {self.initial_gpu_memory:.2f} GB")
    
    def add_record(self, total_time, output_tokens, input_tokens=0, audio_tokens=0, 
                   is_correct=False, peak_memory_gb=None, task_type=None,
                   audio_load_time=0.0, preprocessing_time=0.0, 
                   model_generation_time=0.0, postprocessing_time=0.0):
        """Add single inference record with detailed timing breakdown"""
        
        self.samples += 1
        self.total_time += total_time
        self.total_audio_load_time += audio_load_time
        self.total_preprocessing_time += preprocessing_time
        self.total_model_generation_time += model_generation_time
        self.total_postprocessing_time += postprocessing_time
        self.total_output_tokens += output_tokens
        self.total_input_tokens += input_tokens
        self.total_audio_tokens += audio_tokens
        
        if is_correct:
            self.correct_predictions += 1
        
        if peak_memory_gb is not None:
            self.peak_gpu_memory = max(self.peak_gpu_memory, peak_memory_gb)
        
        self.detailed_records.append({
            "sample_id": self.samples,
            "total_time": total_time,
            "audio_load_time": audio_load_time,
            "preprocessing_time": preprocessing_time,
            "model_generation_time": model_generation_time,
            "postprocessing_time": postprocessing_time,
            "output_tokens": output_tokens,
            "input_tokens": input_tokens,
            "audio_tokens": audio_tokens,
            "is_correct": is_correct,
            "task_type": task_type,
            "peak_memory_gb": peak_memory_gb
        })
    
    def get_summary(self):
        """Get summary statistics with detailed timing breakdown"""
        if self.samples == 0:
            return {"samples": 0, "message": "No data"}
        
        avg_prefill = self.total_prefill_time / self.samples if self.samples > 0 else 0.0
        avg_total = self.total_time / self.samples
        avg_audio_load = self.total_audio_load_time / self.samples
        avg_preprocessing = self.total_preprocessing_time / self.samples
        avg_model_generation = self.total_model_generation_time / self.samples
        avg_postprocessing = self.total_postprocessing_time / self.samples
        # Decode time is approximate and may not be accurate due to independent measurements
        avg_decode = max(0, avg_total - avg_prefill)
        avg_output_tokens = self.total_output_tokens / self.samples
        avg_input_tokens = self.total_input_tokens / self.samples
        avg_audio_tokens = self.total_audio_tokens / self.samples
        accuracy = self.correct_predictions / self.samples
        
        # Calculate percentages
        total_avg = avg_total if avg_total > 0 else 1.0
        pct_audio_load = (avg_audio_load / total_avg) * 100
        pct_preprocessing = (avg_preprocessing / total_avg) * 100
        pct_model_generation = (avg_model_generation / total_avg) * 100
        pct_postprocessing = (avg_postprocessing / total_avg) * 100
        
        return {
            "samples": self.samples,
            "accuracy": accuracy,
            "correct_predictions": self.correct_predictions,
            "avg_prefill_time": avg_prefill,
            "avg_decode_time": avg_decode,  # Approximate, may not be accurate
            "avg_total_time": avg_total,
            "avg_audio_load_time": avg_audio_load,
            "avg_preprocessing_time": avg_preprocessing,
            "avg_model_generation_time": avg_model_generation,
            "avg_postprocessing_time": avg_postprocessing,
            "pct_audio_load": pct_audio_load,
            "pct_preprocessing": pct_preprocessing,
            "pct_model_generation": pct_model_generation,
            "pct_postprocessing": pct_postprocessing,
            "avg_output_tokens": avg_output_tokens,
            "avg_input_tokens": avg_input_tokens,
            "avg_audio_tokens": avg_audio_tokens,
            "peak_gpu_memory": self.peak_gpu_memory,
            "initial_gpu_memory": self.initial_gpu_memory,
            "memory_increase": self.peak_gpu_memory - self.initial_gpu_memory
        }
    
    def export_to_json(self, output_file):
        """Export to JSON file"""
        result = {
            "summary": self.get_summary(),
            "detailed_records": self.detailed_records
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(convert_to_serializable(result), f, indent=2, ensure_ascii=False)
        
        return output_file


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


def main():
    """Main function"""
    args = parse_args()
    
    results_dir = RESULTS_DIR_ENV
    os.makedirs(results_dir, exist_ok=True)
    
    output_file = os.path.join(results_dir, f"desed_results_audio_adaptive_kvcache_budget{ENV_BUDGET}_ratio{ENV_RATIO}.json")
    timing_file = os.path.join(results_dir, f"desed_timing_audio_adaptive_kvcache_budget{ENV_BUDGET}_ratio{ENV_RATIO}.json")
    
    timing_stats = DESEDTimingStats()

    print("Loading Qwen2.5-Omni model...")
    device_map = {"": 0}
    processor = Qwen2_5OmniProcessor.from_pretrained(
        args.model_path, 
        trust_remote_code=True
    )
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    model.disable_talker()
    
    print("Model loaded successfully")
    
    base_model = model.thinker.model # Qwen2_5OmniThinkerTextModel
    
    model_config = base_model.config
    num_hidden_layers = model_config.num_hidden_layers
    num_attention_heads = model_config.num_attention_heads
    num_key_value_heads = num_attention_heads
    num_key_value_groups = num_attention_heads // num_key_value_heads
    # Initialize head_score variables (used in config output even if SparseMM is disabled)
    head_score_type = args.head_score_file or ENV_HEAD_SCORE_FILE
    head_score_batch = args.head_score_batch or ENV_HEAD_SCORE_BATCH
    if args.enable_sparsemm:
        head_score_path = get_audio_head_score_path(
            score_type=head_score_type,
            batch=head_score_batch
        )
        print(f"Loading audio head scores from: {head_score_path}")
        # Load and process audio head scores
        head_score_data = load_audio_head_scores_for_sparsemm(
            head_score_path,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_groups
        )
        # Configure SparseMM
        sparsemm_config = {
            'budget': ENV_BUDGET,
            'ratio': ENV_RATIO,
            'window_size': ENV_WINDOW_SIZE,
            'kernel_size': ENV_KERNEL_SIZE,
            'pooling': ENV_POOLING
        }
        if hasattr(model, 'thinker'):
            print(f"[SparseMM] Thinker type: {type(model.thinker)}")
            print(f"[SparseMM] Thinker has model: {hasattr(model.thinker, 'model')}")
        model.enable_sparsemm_attention(head_score_data, sparsemm_config)
        print(f"[SparseMM] enable_sparsemm_attention completed")
    
    sparsemm_enabled = args.enable_sparsemm
    
    timing_stats.record_initial_memory()
    
    # Load dataset
    print("Loading DESED dataset...")
    dataset = load_desed_qa_dataset(args.dataset_file, args.audio_base_dir)
    
    if SAMPLE_LIMIT > 0:
        dataset = dataset[:SAMPLE_LIMIT]
        print(f"Sample limit set to: {len(dataset)}")
    
    print(f"Begin processing {len(dataset)} samples...")
    
    # Use DynamicCacheSplitHeadFlatten for SparseMM
    results = []
    correct_predictions = 0
    total_processed = 0
    
    # Create progress bar with postfix support
    progress_bar = tqdm(dataset, desc="Processing DESED samples")
    
    for idx, sample in enumerate(progress_bar):
        # Reset SparseMM state and clear cache before each sample to prevent state accumulation
        if sparsemm_enabled:
            reset_sparsemm_state(model)
        clear_model_cache(model)
        
        audio_path = sample["audio_path"]
    
        if not os.path.exists(audio_path):
            print(f"Skipping missing audio file: {audio_path}")
            results.append({
                "sample_id": idx,
                "task_id": sample["task_id"],
                "question": sample["question"],
                "choices": {
                    "A": sample["choice_a"],
                    "B": sample["choice_b"],
                    "C": sample["choice_c"],
                    "D": sample["choice_d"]
                },
                "ground_truth": sample["correct_answer"],
                "prediction": "skip",
                "raw_response": "file not found",
                "task_type": sample.get("task_type", "unknown"),
                "status": "skipped",
                "reason": "audio_file_not_found"
            })
            # Update progress bar (skipped samples don't count toward accuracy)
            if total_processed > 0:
                current_acc = correct_predictions / total_processed
                progress_bar.set_postfix({
                    'Acc': f"{current_acc:.3f}",
                    'Correct': f"{correct_predictions}/{total_processed}",
                    'Skipped': '1'
            })
            continue
        
        
        # Construct question text with choices
        question_text = (
            f"{sample['question']}\n"
            f"A. {sample['choice_a']}\n"
            f"B. {sample['choice_b']}\n"
            f"C. {sample['choice_c']}\n"
            f"D. {sample['choice_d']}\n"
            f"Answer with only the letter (A, B, C, or D):"
        )
        
        # Time: Audio loading
        audio_load_start = time.time()
        audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_data = audio_data.astype(np.float32)
        audio_load_time = time.time() - audio_load_start
        
        # Time: Preprocessing
        preprocessing_start = time.time()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_data},  # Use audio data, not path
                    {"type": "text", "text": question_text}
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
            audio=audios,  # Use processed audios from process_mm_info
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True
        )
        inputs = inputs.to(model.device).to(model.dtype)
        preprocessing_time = time.time() - preprocessing_start
        audio_token_length = 0
        if hasattr(inputs, 'input_ids'):
            token_ids = inputs.input_ids[0].tolist()
            bos_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_BOS_TOKEN_ID]
            eos_positions = [i for i, tid in enumerate(token_ids) if tid == _AUDIO_EOS_TOKEN_ID]
            if bos_positions and eos_positions:
                audio_token_start = bos_positions[0]
                audio_token_end = eos_positions[0]
                audio_token_length = audio_token_end - audio_token_start + 1
            
            # DEBUG: Print token length info for first sample
            if idx == 0:
                total_input_length = inputs.input_ids.shape[1]
                print(f"\n[TOKEN LENGTH DEBUG] Sample 0:")
                print(f"  Total input tokens: {total_input_length}")
                print(f"  Audio tokens: {audio_token_length}")
                print(f"  Text tokens: {total_input_length - audio_token_length}")
                print(f"  ENV_BUDGET: {ENV_BUDGET}")
                print(f"  ENV_WINDOW_SIZE: {ENV_WINDOW_SIZE}")
                print(f"  base_capacity = BUDGET - WINDOW_SIZE = {ENV_BUDGET - ENV_WINDOW_SIZE}")
                print(f"  Compression will trigger if: base_capacity < (total_input_length - window_size)")
                print(f"  i.e., {ENV_BUDGET - ENV_WINDOW_SIZE} < {total_input_length - ENV_WINDOW_SIZE}")
                print(f"  Will compress? {ENV_BUDGET - ENV_WINDOW_SIZE < total_input_length - ENV_WINDOW_SIZE}\n")
        # Time: Model generation (using CUDA events for accurate GPU timing)
        model_gen_start_event = torch.cuda.Event(enable_timing=True)
        model_gen_end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        model_gen_start_event.record()
        
        with torch.no_grad():
            # Initialize DynamicCacheSplitHeadFlatten for SparseMM if enabled
            if sparsemm_enabled:
                past_key_values = DynamicCacheSplitHeadFlatten()
            else:
                past_key_values = None # Let model.generate create standard cache

            generate_kwargs = {
                **inputs,
                "use_audio_in_video": True,
                "return_audio": False,
                "thinker_max_new_tokens": args.max_new_tokens,  # Use thinker_max_new_tokens, not max_new_tokens!
                "thinker_do_sample": False,
                "pad_token_id": processor.tokenizer.eos_token_id,
                "past_key_values": past_key_values
            }
            outputs = model.generate(**generate_kwargs)
        
        model_gen_end_event.record()
        torch.cuda.synchronize()
        model_generation_time = model_gen_start_event.elapsed_time(model_gen_end_event) / 1000.0  # Convert to seconds
        
        # Time: Postprocessing
        postprocessing_start = time.time()
        output_text = processor.batch_decode(
            outputs, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        if "assistant\n" in output_text:
            assistant_start = output_text.rfind("assistant\n") + len("assistant\n")
            response = output_text[assistant_start:].strip()
        else:
            response = output_text
        postprocessing_time = time.time() - postprocessing_start
        
        # Total time (sum of all components)
        total_time = audio_load_time + preprocessing_time + model_generation_time + postprocessing_time
        if hasattr(outputs, 'shape') and len(outputs.shape) > 1:
            output_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        else:
            output_tokens = 0
            generated_ids = torch.tensor([]) if isinstance(outputs, torch.Tensor) else []
        predicted_choice = extract_answer_choice(response)
        is_correct = evaluate_qa_accuracy(predicted_choice, sample["correct_answer"])
        if is_correct:
            correct_predictions += 1
        total_processed += 1
        current_acc = correct_predictions / total_processed if total_processed > 0 else 0.0
        progress_bar.set_postfix({
            'Acc': f"{current_acc:.3f}",
            'Correct': f"{correct_predictions}/{total_processed}"
        })
        allocated, reserved = get_gpu_memory_usage()
        timing_stats.add_record(
            total_time=total_time,
            output_tokens=output_tokens,
            input_tokens=inputs['input_ids'].shape[1],
            audio_tokens=audio_token_length,
            is_correct=is_correct,
            peak_memory_gb=allocated,
            task_type=sample.get("task_type", "unknown"),
            audio_load_time=audio_load_time,
            preprocessing_time=preprocessing_time,
            model_generation_time=model_generation_time,
            postprocessing_time=postprocessing_time
        )
        results.append({
            "sample_id": idx,
            "task_id": sample["task_id"],
            "question": sample["question"],
            "choices": {
                "A": sample["choice_a"],
                "B": sample["choice_b"],
                "C": sample["choice_c"],
                "D": sample["choice_d"]
            },
            "ground_truth": sample["correct_answer"],
            "prediction": predicted_choice,
            "raw_response": response,
            "is_correct": is_correct,
            "task_type": sample.get("task_type", "unknown"),
            "total_time": total_time,
            "audio_load_time": audio_load_time,
            "preprocessing_time": preprocessing_time,
            "model_generation_time": model_generation_time,
            "postprocessing_time": postprocessing_time,
            "output_tokens": output_tokens,
            "input_tokens": inputs['input_ids'].shape[1],
            "audio_tokens": audio_token_length,
            "gpu_memory_gb": allocated
        })
    y_true_clean = []
    y_pred_clean = []
    for result in results:
        if "ground_truth" in result and "prediction" in result:
            gt = result["ground_truth"]
            pred = result["prediction"]
            # Only include valid pairs where both are A/B/C/D
            if gt in ['A', 'B', 'C', 'D'] and pred in ['A', 'B', 'C', 'D']:
                y_true_clean.append(gt)
                y_pred_clean.append(pred)
    
    detailed_metrics = calculate_desed_metrics(y_true_clean, y_pred_clean)
    timing_summary = timing_stats.get_summary()
    final_accuracy = correct_predictions / len(dataset) if len(dataset) > 0 else 0.0
    print(f"\n=== DESED Sound Event Detection Evaluation Results (Qwen2.5-Omni + Audio Adaptive KVCache) ===")
    print(f"Total samples: {len(dataset)}")
    print(f"Correct samples: {correct_predictions}")
    print(f"Accuracy: {final_accuracy:.2%}")
    print(f"F1 Score (Macro): {detailed_metrics['f1_macro']:.2%}")
    print(f"F1 Score (Weighted): {detailed_metrics['f1_weighted']:.2%}")
    print(f"Precision: {detailed_metrics['precision_macro']:.2%}")
    print(f"Recall: {detailed_metrics['recall_macro']:.2%}")
    
    print(f"\n=== Detailed Inference Time Statistics ===")
    print(f"Sample count: {timing_summary['samples']}")
    print(f"\n--- Time Breakdown (per sample) ---")
    print(f"  Audio Loading:      {timing_summary['avg_audio_load_time']:.4f} seconds ({timing_summary.get('pct_audio_load', 0):.1f}%)")
    print(f"  Preprocessing:      {timing_summary['avg_preprocessing_time']:.4f} seconds ({timing_summary.get('pct_preprocessing', 0):.1f}%)")
    print(f"  Model Generation:  {timing_summary['avg_model_generation_time']:.4f} seconds ({timing_summary.get('pct_model_generation', 0):.1f}%)")
    print(f"  Postprocessing:    {timing_summary['avg_postprocessing_time']:.4f} seconds ({timing_summary.get('pct_postprocessing', 0):.1f}%)")
    print(f"  Total Time:        {timing_summary['avg_total_time']:.4f} seconds (100.0%)")
    print(f"\n--- Legacy Time Statistics (for compatibility) ---")
    print(f"  Average Prefill time: {timing_summary['avg_prefill_time']:.4f} seconds")
    print(f"  Average Decode time (approximate): {timing_summary['avg_decode_time']:.4f} seconds")
    print(f"\n--- Token Statistics ---")
    print(f"  Average input tokens: {timing_summary['avg_input_tokens']:.1f}")
    print(f"  Average audio tokens: {timing_summary['avg_audio_tokens']:.1f}")
    print(f"  Average output tokens: {timing_summary['avg_output_tokens']:.1f}")
    timing_stats.export_to_json(timing_file)
    print(f"\nTiming statistics saved to: {timing_file}")
    output_data = {
        "summary": {
            "total_samples": len(dataset),
            "correct_predictions": correct_predictions,
            "accuracy": final_accuracy,
            "f1_macro": detailed_metrics.get('f1_macro', 0.0),
            "f1_weighted": detailed_metrics.get('f1_weighted', 0.0),
            "precision_macro": detailed_metrics.get('precision_macro', 0.0),
            "recall_macro": detailed_metrics.get('recall_macro', 0.0),
            "valid_samples": detailed_metrics.get('valid_samples', 0),
            "classification_report": detailed_metrics.get('classification_report', "")
        },
        "config": {
            "budget": ENV_BUDGET,
            "ratio": ENV_RATIO,
            "window_size": ENV_WINDOW_SIZE,
            "kernel_size": ENV_KERNEL_SIZE,
            "pooling": ENV_POOLING,
            "max_new_tokens": args.max_new_tokens,
            "head_score_file": head_score_type,
            "head_score_batch": head_score_batch
        },
        "results": results
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(convert_to_serializable(output_data), f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}")
    

if __name__ == "__main__":
    main()

'''
# Usage with SparseMM enabled (recommended settings for DESED)
CUDA_VISIBLE_DEVICES=3 \
BUDGET=64 \
RATIO=0.2 \
WINDOW_SIZE=32 \
KERNEL_SIZE=7 \
POOLING=maxpool \
python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv/DESED_audio_sparse_kvcache.py \
    --model-path /mnt/users/wangyuxuan-20250915/Models/Qwen2.5-Omni-3B \
    --max-new-tokens 10 \
    --head-score-file aggregated \
    --enable-sparsemm
    
# Usage with batch-specific head scores and SparseMM
CUDA_VISIBLE_DEVICES=7 BUDGET=7168 RATIO=0.2 WINDOW_SIZE=64 KERNEL_SIZE=7 \
python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv/DESED_audio_sparse_kvcache.py \
    --model-path /mnt/users/wangyuxuan-20250915/Models/Qwen2.5-Omni-3B \
    --max-new-tokens 10 \
    --head-score-file aggregated \
    --head-score-batch batch1 \
    --enable-sparsemm

# Usage without SparseMM (standard attention)
CUDA_VISIBLE_DEVICES=2 python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv/DESED_audio_sparse_kvcache.py \
    --model-path /mnt/users/wangyuxuan-20250915/Models/Qwen2.5-Omni-3B \
    --max-new-tokens 10
'''