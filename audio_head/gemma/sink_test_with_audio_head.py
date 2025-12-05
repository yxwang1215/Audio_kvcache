"""
Audio Token Attention Sink Analysis + Audio Head Scoring
集成 SparseMM 风格的音频头打分功能

基于 SparseMM 的 hit_score 机制:
1. 在生成时捕获注意力
2. 检查每个头的 top-k 注意力是否落在音频 token 区域
3. 累积命中分数并排序识别音频头
"""

import os
import sys
import json
import gc
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import librosa
import argparse
import random
import glob
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
from tqdm import tqdm
from collections import defaultdict

try:
    import soundfile as sf
except ImportError:
    sf = None

warnings.filterwarnings("ignore")

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../Modeling'))

# Import Gemma3n model and processor
import sys
import os
sys.path.insert(0, '/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft')
from modeling_gemma3n import Gemma3nForConditionalGeneration
from processing_gemma3n import Gemma3nProcessor

random.seed(42)

# 定义音频 token ID (Gemma3n uses config.audio_token_id, default is 262273)
# We'll get it from the model config after loading
_AUDIO_TOKEN_ID = None  # Will be set from model config


def _parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated string into a list of ints."""
    if value is None:
        return None
    cleaned = [item.strip() for item in value.split(',') if item.strip()]
    if not cleaned:
        return None
    try:
        return [int(item) for item in cleaned]
    except ValueError:
        raise ValueError(f"Invalid sink dimension string: {value}")


@dataclass
class AttnSinkParams:
    """Configuration for attention-sink head selection (See What You Are Told, 2025)."""

    tau: float = 20.0
    rho: float = 0.5
    summ: float = 0.2
    sink_dims: Optional[List[int]] = None
    topk_dims: int = 2
    min_layer: int = 2


@dataclass
class AudioSinkHeadResult:
    """Container for per-layer attention sink statistics."""

    candidate_heads: List[int] = field(default_factory=list)
    sink_portion: List[float] = field(default_factory=list)
    audio_mass: List[float] = field(default_factory=list)
    sink_mass: List[float] = field(default_factory=list)
    total_audio_tokens: int = 0
    sink_token_indices: List[int] = field(default_factory=list)
    used_sink_dims: List[int] = field(default_factory=list)
    max_sink_rms: float = 0.0
    mean_sink_rms: float = 0.0
    p95_sink_rms: float = 0.0

    def to_serializable(self) -> Dict[str, object]:
        return {
            "candidate_heads": self.candidate_heads,
            "sink_portion": [float(x) for x in self.sink_portion],
            "audio_mass": [float(x) for x in self.audio_mass],
            "sink_mass": [float(x) for x in self.sink_mass],
            "total_audio_tokens": int(self.total_audio_tokens),
            "sink_token_indices": [int(idx) for idx in self.sink_token_indices],
            "used_sink_dims": [int(dim) for dim in self.used_sink_dims],
            "max_sink_rms": float(self.max_sink_rms),
            "mean_sink_rms": float(self.mean_sink_rms),
            "p95_sink_rms": float(self.p95_sink_rms),
        }


class AudioSinkHeadSelector:
    """Implements the AttnSink head selection strategy for audio tokens."""

    def __init__(self, params: AttnSinkParams):
        self.params = params
        self.sink_dims: Optional[List[int]] = list(params.sink_dims) if params.sink_dims else None
        self.current_layer_results: Dict[int, AudioSinkHeadResult] = {}

    @staticmethod
    def _rms_norm(hidden_states: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        return hidden_states * torch.rsqrt(variance + eps)

    def start_sample(self) -> None:
        self.current_layer_results = {}

    def finish_sample(self) -> Dict[int, Dict[str, object]]:
        snapshot = {layer: result.to_serializable() for layer, result in self.current_layer_results.items()}
        self.current_layer_results = {}
        return snapshot

    def _maybe_init_sink_dims(self, hidden_states: torch.Tensor) -> None:
        if self.sink_dims is not None:
            return
        rms_norm = self._rms_norm(hidden_states)
        scores = rms_norm.abs().mean(dim=0)
        topk = min(self.params.topk_dims, scores.shape[0])
        selected = torch.topk(scores, k=topk).indices.tolist()
        self.sink_dims = selected
        self.params.sink_dims = selected
        print(f"  Initialized sink dims from hidden states: {self.sink_dims}")

    def update(
        self,
        layer_idx: int,
        attn_output: torch.Tensor,
        attn_weights: torch.Tensor,
        audio_indices: List[int]
    ) -> Optional[AudioSinkHeadResult]:
        if not audio_indices:
            return None
        if attn_weights is None or attn_output is None:
            return None
        if layer_idx < self.params.min_layer:
            return None
        if attn_weights.dim() != 4 or attn_weights.shape[2] == 0:
            return None

        with torch.no_grad():
            hidden_states = attn_output.detach().to(torch.float32)
            if hidden_states.dim() == 3:
                hidden_states = hidden_states[0]
            elif hidden_states.dim() == 2:
                pass
            else:
                hidden_states = hidden_states.view(hidden_states.shape[-2], hidden_states.shape[-1])

            hidden_states_cpu = hidden_states.cpu()
            self._maybe_init_sink_dims(hidden_states_cpu)
            sink_dims = self.sink_dims or []

            rms_hidden = self._rms_norm(hidden_states_cpu)
            if sink_dims:
                sink_scores = torch.abs(rms_hidden[:, sink_dims])
                max_scores = sink_scores.max(dim=-1).values
                sink_mask = max_scores > self.params.tau
                sink_indices = torch.nonzero(sink_mask, as_tuple=False).squeeze(-1).tolist()
                if max_scores.numel() > 0:
                    layer_max_sink_rms = float(max_scores.max().item())
                    layer_mean_sink_rms = float(max_scores.mean().item())
                    np_scores = max_scores.cpu().numpy()
                    layer_p95_sink_rms = float(np.percentile(np_scores, 95)) if np_scores.size > 1 else layer_max_sink_rms
                else:
                    layer_max_sink_rms = layer_mean_sink_rms = layer_p95_sink_rms = 0.0
            else:
                sink_indices = []
                layer_max_sink_rms = layer_mean_sink_rms = layer_p95_sink_rms = 0.0

            audio_set = set(audio_indices)
            audio_sink_indices = [idx for idx in sink_indices if idx in audio_set]

            last_token_attn = attn_weights[0, :, -1, :].detach().to(torch.float32).cpu()
            audio_attn = last_token_attn[:, audio_indices] if audio_indices else torch.zeros_like(last_token_attn[:, :0])
            audio_mass = audio_attn.sum(dim=-1)

            if audio_sink_indices:
                sink_attn = last_token_attn[:, audio_sink_indices]
                sink_mass = sink_attn.sum(dim=-1)
            else:
                sink_mass = torch.zeros_like(audio_mass)

            sink_portion = torch.zeros_like(audio_mass)
            valid_denominator = audio_mass > 0
            sink_portion[valid_denominator] = sink_mass[valid_denominator] / (audio_mass[valid_denominator] + 1e-6)

            qualified_mask = (sink_portion <= self.params.rho) & (audio_mass >= self.params.summ)
            candidate_heads = torch.nonzero(qualified_mask, as_tuple=False).squeeze(-1).tolist()

            result = AudioSinkHeadResult(
                candidate_heads=[int(idx) for idx in candidate_heads],
                sink_portion=[float(x) for x in sink_portion.tolist()],
                audio_mass=[float(x) for x in audio_mass.tolist()],
                sink_mass=[float(x) for x in sink_mass.tolist()],
                total_audio_tokens=len(audio_indices),
                sink_token_indices=[int(idx) for idx in audio_sink_indices],
                used_sink_dims=[int(dim) for dim in sink_dims],
                max_sink_rms=layer_max_sink_rms,
                mean_sink_rms=layer_mean_sink_rms,
                p95_sink_rms=layer_p95_sink_rms,
            )
            self.current_layer_results[layer_idx] = result
            return result


def load_tau_samples(base_dir, num_samples=5, filter_by_duration=True):
    """Load TAU acoustic scene classification samples"""
    meta_file = os.path.join(base_dir, "acoustic_scene_task_meta.json")
    
    if not os.path.exists(meta_file):
        print(f"Error: Metadata file does not exist: {meta_file}")
        return []
    
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    print(f"Loading {num_samples} samples from TAU dataset...")
    
    # 收集所有有效样本及其时长
    valid_samples = []
    for item in metadata:
        rel_path = item["path"]
        # TAU 音频文件在 concatenated_resampled 子目录下
        wav_path = os.path.join(base_dir, "concatenated_resampled", rel_path)
        
        if os.path.exists(wav_path):
            # 获取音频时长
            try:
                if sf is not None:
                    info = sf.info(wav_path)
                    duration = info.duration
                else:
                    duration = librosa.get_duration(path=wav_path)
            except:
                continue
            
            valid_samples.append({
                "audio_file": wav_path,
                "scene_label": item["scene_label"],
                "question": item["question"],
                "choices": {
                    "A": item["choice_a"],
                    "B": item["choice_b"],
                    "C": item["choice_c"],
                    "D": item["choice_d"]
                },
                "answer": item["answer_gt"],
                "duration": round(duration, 1)  # 四舍五入到0.1秒
            })
    
    if filter_by_duration and num_samples > 0:
        # 统计每个时长的样本数量
        duration_counts = defaultdict(list)
        for sample in valid_samples:
            duration_counts[sample["duration"]].append(sample)
        
        # 找到样本数最多的时长
        most_common_duration = max(duration_counts.keys(), key=lambda d: len(duration_counts[d]))
        filtered_samples = duration_counts[most_common_duration]
        
        print(f"Most common duration: {most_common_duration}s with {len(filtered_samples)} samples")
        
        # 从该时长的样本中随机选择
        if num_samples < len(filtered_samples):
            valid_samples = random.sample(filtered_samples, num_samples)
        else:
            valid_samples = filtered_samples[:num_samples]
    elif num_samples > 0 and num_samples < len(valid_samples):
        # 如果不按时长过滤，就随机采样
        valid_samples = random.sample(valid_samples, num_samples)
    
    print(f"Found {len(valid_samples)} valid TAU audio files")
    return valid_samples


def load_librispeech_samples(base_dir, num_samples=10):
    """Load LibriSpeech samples from pre-built mapping file"""
    # 使用预先构建的映射文件
    mapping_file = os.path.join(base_dir, "librispeech_audio_annotation_mapping.json")
    
    if not os.path.exists(mapping_file):
        print(f"Error: Mapping file does not exist: {mapping_file}")
        print(f"  Expected path: {mapping_file}")
        print(f"  Please run create_mapping.py first to generate the mapping file")
        return []
    
    print(f"Loading LibriSpeech samples from mapping file...")
    print(f"  Mapping file: {mapping_file}")
    
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
    except Exception as e:
        print(f"Error loading mapping file: {e}")
        return []
    
    total_files = mapping_data.get("total_audio_files", 0)
    matched_pairs = mapping_data.get("matched_pairs", 0)
    missing_annotations = mapping_data.get("missing_annotations", 0)
    
    print(f"  Total audio files: {total_files}")
    print(f"  Matched pairs: {matched_pairs}")
    print(f"  Missing annotations: {missing_annotations}")
    
    mapping_list = mapping_data.get("mapping", [])
    
    if not mapping_list:
        print("No samples found in mapping file!")
        return []
    
    # 过滤出有标注的样本
    valid_samples = []
    missing_count = 0
    
    for item in mapping_list:
        if not item.get("has_annotation", False):
            continue
        
        audio_file = item.get("audio_file", "")
        json_file = item.get("json_file", "")
        
        # 验证文件是否存在
        if not os.path.exists(audio_file):
            missing_count += 1
            if missing_count <= 3:
                print(f"  Warning: Audio file not found: {audio_file}")
            continue
        
        if not os.path.exists(json_file):
            missing_count += 1
            if missing_count <= 3:
                print(f"  Warning: JSON file not found: {json_file}")
            continue
        
        # 加载 JSON 标注
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"  Warning: Error loading JSON {json_file}: {e}")
            continue
        
        valid_samples.append({
            "audio_file": audio_file,
            "audio_id": item.get("audio_id", ""),
            "json_file": json_file,
            "metadata": item.get("metadata", {}),
            "transcription": json_data.get("transcription", {}),
            "transcription_text": item.get("transcription_text", ""),
            "word_count": item.get("word_count", 0)
        })
    
    if missing_count > 3:
        print(f"  ... and {missing_count - 3} more files not found")
    
    # Randomly sample
    if num_samples > 0 and num_samples < len(valid_samples):
        valid_samples = random.sample(valid_samples, num_samples)
    
    print(f"Found {len(valid_samples)} valid LibriSpeech audio files with annotations")
    
    # 打印样本统计信息
    if valid_samples:
        # 统计不同 speaker 的数量
        speakers = set(s['metadata'].get('speaker_id', 0) for s in valid_samples)
        print(f"  - Unique speakers: {len(speakers)}")
        
        # 统计平均单词数（如果有的话）
        word_counts = [s.get('word_count', 0) for s in valid_samples if s.get('word_count', 0) > 0]
        if word_counts:
            print(f"  - Average word count: {sum(word_counts) / len(word_counts):.1f}")
    
    return valid_samples


def load_vox_samples(base_dir, num_samples=10):
    """Load VoxCeleb gender identification samples"""
    meta_file = os.path.join(base_dir, "concatenated_audio", "gender_id_task_meta.json")
    
    if not os.path.exists(meta_file):
        print(f"Error: Metadata file does not exist: {meta_file}")
        print(f"  Expected path: {meta_file}")
        return []
    
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    sample_desc = "all samples" if num_samples <= 0 else f"{num_samples} samples"
    print(f"Loading {sample_desc} from VoxCeleb dataset...")
    print(f"  Metadata file: {meta_file}")
    print(f"  Total entries in metadata: {len(metadata)}")
    
    valid_samples = []
    missing_count = 0
    for item in metadata:
        # 音频文件在 concatenated_audio/wav/ 子目录下
        rel_path = item["path"]
        wav_path = os.path.join(base_dir, "concatenated_audio", "wav", rel_path)
        
        if os.path.exists(wav_path):
            # 获取 answer_gt (gender)
            gender = item.get("answer_gt", "").lower().strip()
            valid_samples.append({
                "audio_file": wav_path,
                "gender": gender,
                "speaker_id": item.get("speaker_id_original", ""),
                "question": item["question"],
                "choices": {
                    "A": item["choice_a"],
                    "B": item["choice_b"]
                },
                "answer": item["answer_gt"]
            })
        else:
            missing_count += 1
            if missing_count <= 3:  # 只打印前3个缺失文件作为示例
                print(f"  Warning: Audio file not found: {wav_path}")
    
    if missing_count > 3:
        print(f"  ... and {missing_count - 3} more files not found")
    
    # Randomly sample
    if num_samples > 0 and num_samples < len(valid_samples):
        valid_samples = random.sample(valid_samples, num_samples)
    
    print(f"Found {len(valid_samples)} valid VoxCeleb audio files")
    if valid_samples:
        male_count = sum(1 for s in valid_samples if s["gender"] == "male")
        female_count = len(valid_samples) - male_count
        print(f"  - Male: {male_count} samples")
        print(f"  - Female: {female_count} samples")
    
    return valid_samples


def load_vox_age_samples(base_dir, num_samples=10):
    """Load VoxCeleb age classification samples"""
    # 元数据文件在 concatenated_audio 子目录下
    meta_file = os.path.join(base_dir, "concatenated_audio", "age_classification_task_meta.json")
    
    if not os.path.exists(meta_file):
        print(f"Error: Metadata file does not exist: {meta_file}")
        print(f"  Expected path: {meta_file}")
        return []
    
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    sample_desc = "all samples" if num_samples <= 0 else f"{num_samples} samples"
    print(f"Loading {sample_desc} from VoxCeleb Age dataset...")
    print(f"  Metadata file: {meta_file}")
    print(f"  Total entries in metadata: {len(metadata)}")
    
    valid_samples = []
    missing_count = 0
    for item in metadata:
        # 音频文件在 concatenated_audio/wav/ 子目录下
        rel_path = item["path"]
        wav_path = os.path.join(base_dir, "concatenated_audio", "wav", rel_path)
        
        if os.path.exists(wav_path):
            age_group = item["answer_gt"].strip()
            valid_samples.append({
                "audio_file": wav_path,
                "age_group": age_group,
                "speaker_age": item.get("speaker_age", 0),
                "speaker_id": item.get("speaker_id_original", ""),
                "question": item["question"],
                "choices": {
                    "A": item["choice_a"],
                    "B": item["choice_b"],
                    "C": item["choice_c"],
                    "D": item["choice_d"],
                    "E": item["choice_e"]
                },
                "answer": item["answer_gt"]
            })
        else:
            missing_count += 1
            if missing_count <= 3:  # 只打印前3个缺失文件作为示例
                print(f"  Warning: Audio file not found: {wav_path}")
    
    if missing_count > 3:
        print(f"  ... and {missing_count - 3} more files not found")
    
    # Randomly sample
    if num_samples > 0 and num_samples < len(valid_samples):
        valid_samples = random.sample(valid_samples, num_samples)
    
    print(f"Found {len(valid_samples)} valid VoxCeleb Age audio files")
    
    # 统计年龄组分布
    if valid_samples:
        age_group_counts = defaultdict(int)
        for s in valid_samples:
            age_group_counts[s["age_group"]] += 1
        print(f"  Age group distribution:")
        for group in sorted(age_group_counts.keys()):
            print(f"    {group}: {age_group_counts[group]} samples")
    
    return valid_samples


def load_desed_samples(base_dir, num_samples=10):
    """Load DESED sound event detection samples"""
    # DESED 数据在 DESED_dataset/concatenated_audio 子目录下
    audio_dir = os.path.join(base_dir, "DESED_dataset", "concatenated_audio")
    meta_file = os.path.join(audio_dir, "desed_sound_event_detection_task.json")
    
    if not os.path.exists(meta_file):
        print(f"Error: Metadata file does not exist: {meta_file}")
        print(f"  Expected path: {meta_file}")
        return []
    
    print(f"Loading DESED sound event detection samples...")
    print(f"  Metadata file: {meta_file}")
    
    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, dict) or 'tasks' not in data:
        print(f"Error: JSON file format is incorrect, expected {'tasks': [...]} structure")
        return []
    
    tasks = data['tasks']
    sample_desc = "all samples" if num_samples <= 0 else f"{num_samples} samples"
    print(f"  Total tasks in metadata: {len(tasks)}")
    print(f"  Requested: {sample_desc}")
    
    valid_samples = []
    missing_count = 0
    task_type_stats = defaultdict(int)
    
    for task in tasks:
        # 音频文件路径是相对于 audio_dir 的
        rel_path = task.get("path", "")
        if not rel_path:
            continue
            
        audio_path = os.path.join(audio_dir, rel_path)
        
        if os.path.exists(audio_path):
            choices_dict = task.get("choices", {})
            task_type = task.get("task_type", "unknown")
            
            valid_samples.append({
                "audio_file": audio_path,
                "task_type": task_type,
                "question": task.get("question", ""),
                "choices": {
                    "A": choices_dict.get("A", ""),
                    "B": choices_dict.get("B", ""),
                    "C": choices_dict.get("C", ""),
                    "D": choices_dict.get("D", "")
                },
                "answer": task.get("answer_gt", ""),
                "all_events": task.get("all_events", []),
                "primary_event": task.get("primary_event", ""),
                "correct_event": task.get("correct_event", "")
            })
            task_type_stats[task_type] += 1
        else:
            missing_count += 1
            if missing_count <= 3:
                print(f"  Warning: Audio file not found: {audio_path}")
    
    if missing_count > 3:
        print(f"  ... and {missing_count - 3} more files not found")
    
    # Randomly sample
    if num_samples > 0 and num_samples < len(valid_samples):
        valid_samples = random.sample(valid_samples, num_samples)
    
    print(f"Found {len(valid_samples)} valid DESED audio files")
    if task_type_stats:
        print(f"  Task type distribution:")
        for task_type, count in sorted(task_type_stats.items()):
            print(f"    {task_type}: {count} samples")
    
    return valid_samples


def load_gtzan_samples(base_dir, num_samples=10):
    """Load GTZAN music genre classification samples"""
    # GTZAN 数据在 concatenated_audio 子目录下
    audio_dir = os.path.join(base_dir, "concatenated_audio")
    meta_file = os.path.join(audio_dir, "music_genre_classification_meta.json")
    
    if not os.path.exists(meta_file):
        print(f"Error: Metadata file does not exist: {meta_file}")
        print(f"  Expected path: {meta_file}")
        return []
    
    print(f"Loading GTZAN music genre classification samples...")
    print(f"  Metadata file: {meta_file}")
    
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    sample_desc = "all samples" if num_samples <= 0 else f"{num_samples} samples"
    print(f"  Total entries in metadata: {len(metadata)}")
    print(f"  Requested: {sample_desc}")
    
    valid_samples = []
    missing_count = 0
    genre_stats = defaultdict(int)
    
    for item in metadata:
        # 音频文件在 wav 子目录下
        rel_path = item.get("path", "")
        if not rel_path:
            continue
            
        audio_path = os.path.join(audio_dir, "wav", rel_path)
        
        if os.path.exists(audio_path):
            genre = item.get("genre", "unknown")
            valid_samples.append({
                "audio_file": audio_path,
                "genre": genre,
                "question": item["question"],
                "choices": {
                    "A": item["choice_a"],
                    "B": item["choice_b"],
                    "C": item["choice_c"],
                    "D": item["choice_d"]
                },
                "answer": item["answer_gt"]
            })
            genre_stats[genre] += 1
        else:
            missing_count += 1
            if missing_count <= 3:
                print(f"  Warning: Audio file not found: {audio_path}")
    
    if missing_count > 3:
        print(f"  ... and {missing_count - 3} more files not found")
    
    # Randomly sample
    if num_samples > 0 and num_samples < len(valid_samples):
        valid_samples = random.sample(valid_samples, num_samples)
    
    print(f"Found {len(valid_samples)} valid GTZAN audio files")
    if genre_stats:
        print(f"  Genre distribution:")
        for genre, count in sorted(genre_stats.items()):
            print(f"    {genre}: {count} samples")
    
    return valid_samples


def load_had_samples(base_dir, num_samples=10):
    """Load HAD (Human Activity Detection) samples"""
    sample_desc = "all samples" if num_samples <= 0 else f"{num_samples} samples"
    print(f"Loading HAD human activity detection samples...")
    print(f"  Base directory: {base_dir}")
    print(f"  Requested: {sample_desc}")
    
    real_dir = os.path.join(base_dir, "real")
    fake_dir = os.path.join(base_dir, "fake")
    
    print(f"  Real audio directory: {real_dir}")
    print(f"  Fake audio directory: {fake_dir}")
    
    all_samples = []
    
    # Load real samples
    real_count = 0
    if os.path.exists(real_dir):
        real_files = glob.glob(os.path.join(real_dir, "*.wav"))
        print(f"  Found {len(real_files)} real audio files")
        for audio_file in real_files:
            all_samples.append({
                "audio_file": audio_file,
                "label": "real",
                "question": "Is this audio of a human activity generated by AI?",
                "choices": {
                    "A": "Yes, it is AI-generated",
                    "B": "No, it is real human activity"
                },
                "answer": "B"
            })
            real_count += 1
    else:
        print(f"  Warning: Real directory does not exist: {real_dir}")
    
    # Load fake samples
    fake_count = 0
    if os.path.exists(fake_dir):
        fake_files = glob.glob(os.path.join(fake_dir, "*.wav"))
        print(f"  Found {len(fake_files)} fake audio files")
        for audio_file in fake_files:
            all_samples.append({
                "audio_file": audio_file,
                "label": "fake",
                "question": "Is this audio of a human activity generated by AI?",
                "choices": {
                    "A": "Yes, it is AI-generated",
                    "B": "No, it is real human activity"
                },
                "answer": "A"
            })
            fake_count += 1
    else:
        print(f"  Warning: Fake directory does not exist: {fake_dir}")
    
    if len(all_samples) == 0:
        print("  No audio files found in either real or fake directories!")
        return []
    
    # Balance samples
    real_samples = [s for s in all_samples if s["label"] == "real"]
    fake_samples = [s for s in all_samples if s["label"] == "fake"]
    
    print(f"  Before balancing: real={len(real_samples)}, fake={len(fake_samples)}")
    
    min_count = min(len(real_samples), len(fake_samples))
    if min_count > 0:
        all_samples = random.sample(real_samples, min_count) + random.sample(fake_samples, min_count)
        random.shuffle(all_samples)
        print(f"  After balancing: {len(all_samples)} total samples (balanced)")
    else:
        print(f"  Warning: Cannot balance - one category is empty")
    
    # Randomly sample
    if num_samples > 0 and num_samples < len(all_samples):
        all_samples = random.sample(all_samples, num_samples)
    
    print(f"Found {len(all_samples)} valid HAD audio files")
    if all_samples:
        real_count = sum(1 for s in all_samples if s["label"] == "real")
        fake_count = len(all_samples) - real_count
        print(f"  - Real: {real_count} samples")
        print(f"  - Fake: {fake_count} samples")
    
    return all_samples


def load_slue_samples(base_dir, num_samples=10):
    """Load SLUE (Spoken Language Understanding Evaluation) samples"""
    # SLUE 元数据文件名是 merged_audio_data.json
    meta_file = os.path.join(base_dir, "merged_audio_data.json")
    
    if not os.path.exists(meta_file):
        print(f"Error: Metadata file does not exist: {meta_file}")
        print(f"  Expected path: {meta_file}")
        return []
    
    print(f"Loading SLUE spoken language understanding samples...")
    print(f"  Metadata file: {meta_file}")
    
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    sample_desc = "all samples" if num_samples <= 0 else f"{num_samples} samples"
    print(f"  Total entries in metadata: {len(metadata)}")
    print(f"  Requested: {sample_desc}")
    
    valid_samples = []
    missing_count = 0
    task_type_stats = defaultdict(int)
    
    for item in metadata:
        # 音频文件路径是相对于 base_dir 的
        rel_path = item.get("path", "")
        if not rel_path:
            continue
            
        audio_path = os.path.join(base_dir, rel_path)
        
        if os.path.exists(audio_path):
            choices_dict = item.get("choices", {})
            task_type = item.get("task_type", "unknown")
            
            valid_samples.append({
                "audio_file": audio_path,
                "task_type": task_type,
                "question": item.get("question", ""),
                "choices": {
                    "A": choices_dict.get("A", ""),
                    "B": choices_dict.get("B", ""),
                    "C": choices_dict.get("C", ""),
                    "D": choices_dict.get("D", "")
                },
                "answer": item.get("answer_gt", "")
            })
            task_type_stats[task_type] += 1
        else:
            missing_count += 1
            if missing_count <= 3:
                print(f"  Warning: Audio file not found: {audio_path}")
    
    if missing_count > 3:
        print(f"  ... and {missing_count - 3} more files not found")
    
    # Randomly sample
    if num_samples > 0 and num_samples < len(valid_samples):
        valid_samples = random.sample(valid_samples, num_samples)
    
    print(f"Found {len(valid_samples)} valid SLUE audio files")
    if task_type_stats:
        print(f"  Task type distribution:")
        for task_type, count in sorted(task_type_stats.items()):
            print(f"    {task_type}: {count} samples")
    
    return valid_samples


def load_vesus_samples(base_dir, num_samples=10):
    """Load VESUS emotion recognition samples"""
    # VESUS 元数据文件名是 audio_emotion_dataset.json
    meta_file = os.path.join(base_dir, "audio_emotion_dataset.json")
    
    if not os.path.exists(meta_file):
        print(f"Error: Metadata file does not exist: {meta_file}")
        print(f"  Expected path: {meta_file}")
        return []
    
    print(f"Loading VESUS emotion recognition samples...")
    print(f"  Metadata file: {meta_file}")
    
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    sample_desc = "all samples" if num_samples <= 0 else f"{num_samples} samples"
    print(f"  Total entries in metadata: {len(metadata)}")
    print(f"  Requested: {sample_desc}")
    
    valid_samples = []
    missing_count = 0
    filtered_count = 0
    emotion_stats = defaultdict(int)
    
    for item in metadata:
        # Filter out person 2 and 10's happy emotion (数据质量问题)
        person_id_str = str(item.get('person_id', '')).strip()
        emotion_label_lower = str(item.get('emotion_label', '')).strip().lower()
        
        if person_id_str in ['2', '10'] and emotion_label_lower == 'happy':
            filtered_count += 1
            continue
        
        # 音频文件路径是相对于 base_dir 的
        rel_path = item.get("path", "")
        if not rel_path:
            continue
            
        audio_path = os.path.join(base_dir, rel_path)
        
        if os.path.exists(audio_path):
            choices_dict = item.get("choices", {})
            emotion_label = item.get("emotion_label", "unknown")
            
            valid_samples.append({
                "audio_file": audio_path,
                "emotion_label": emotion_label,
                "person_id": item.get("person_id", ""),
                "question": item.get("question", ""),
                "choices": {
                    "A": choices_dict.get("A", ""),
                    "B": choices_dict.get("B", ""),
                    "C": choices_dict.get("C", ""),
                    "D": choices_dict.get("D", "")
                },
                "answer": item.get("answer_gt", "")
            })
            emotion_stats[emotion_label] += 1
        else:
            missing_count += 1
            if missing_count <= 3:
                print(f"  Warning: Audio file not found: {audio_path}")
    
    if missing_count > 3:
        print(f"  ... and {missing_count - 3} more files not found")
    
    if filtered_count > 0:
        print(f"  Filtered {filtered_count} samples (person 2/10's happy emotion)")
    
    # Randomly sample
    if num_samples > 0 and num_samples < len(valid_samples):
        valid_samples = random.sample(valid_samples, num_samples)
    
    print(f"Found {len(valid_samples)} valid VESUS audio files")
    if emotion_stats:
        print(f"  Emotion distribution:")
        for emotion, count in sorted(emotion_stats.items()):
            print(f"    {emotion}: {count} samples")
    
    return valid_samples


def load_race_samples(base_dir, num_samples=10):
    """Load RACE reading comprehension samples with audio narration"""
    meta_file = os.path.join(base_dir, "race_benchmark.json")
    
    if not os.path.exists(meta_file):
        print(f"Error: Metadata file does not exist: {meta_file}")
        print(f"  Expected path: {meta_file}")
        return []
    
    sample_desc = "all samples" if num_samples <= 0 else f"{num_samples} samples"
    print(f"Loading RACE reading comprehension samples...")
    print(f"  Metadata file: {meta_file}")
    
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    print(f"  Total entries in metadata: {len(metadata)}")
    print(f"  Requested: {sample_desc}")
    
    valid_samples = []
    missing_count = 0
    subset_stats = defaultdict(int)
    
    for item in metadata:
        audio_rel_path = item.get("audio_path", "")
        if not audio_rel_path:
            continue
            
        audio_path = os.path.join(base_dir, audio_rel_path)
        
        if os.path.exists(audio_path):
            # Determine subset (high/middle) from audio path
            subset = "high" if "high" in audio_rel_path.lower() else "middle"
            
            valid_samples.append({
                "audio_file": audio_path,
                "article_id": item.get("article_id", ""),
                "question_idx": item.get("question_idx", 0),
                "question": item.get("question", ""),
                "options": item.get("options", []),
                "answer": item.get("answer", ""),
                "subset": subset,
                "article": item.get("article", "")  # 可能包含原文
            })
            subset_stats[subset] += 1
        else:
            missing_count += 1
            if missing_count <= 3:
                print(f"  Warning: Audio file not found: {audio_path}")
    
    if missing_count > 3:
        print(f"  ... and {missing_count - 3} more files not found")
    
    # Randomly sample
    if num_samples > 0 and num_samples < len(valid_samples):
        valid_samples = random.sample(valid_samples, num_samples)
    
    print(f"Found {len(valid_samples)} valid RACE audio files")
    if subset_stats:
        print(f"  Subset distribution:")
        for subset, count in sorted(subset_stats.items()):
            print(f"    {subset}: {count} samples")
    
    return valid_samples


class AudioAttentionSinkAnalyzer:
    """Audio Token Attention Sink Analyzer"""
    
    def __init__(
        self,
        model,
        processor,
        output_dir="./audio_attention_analysis",
        num_layers=36,
        sink_params: Optional[AttnSinkParams] = None,
        skip_heatmaps=False
    ):
        self.model = model
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache model device/dtype
        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype
        
        self.num_layers = num_layers
        self.num_heads = None
        self.enable_head_scoring = False
        self.skip_heatmaps = skip_heatmaps  # Control heatmap generation

        self.sink_params = sink_params or AttnSinkParams()
        self.audio_sink_selector = AudioSinkHeadSelector(self.sink_params)

        # Storage for per-sample attention snapshots
        self.current_sample_attentions = {}
        self.current_audio_start = None
        self.current_audio_end = None
        self.current_audio_indices = None
        self.sample_sink_statistics: List[Dict[str, object]] = []
        self.sink_metric_tracker = {
            'sink_portion': defaultdict(list),
            'audio_mass': defaultdict(list),
            'sink_mass': defaultdict(list),
            'num_qualified': defaultdict(list),
            'max_sink_rms': defaultdict(list),
            'mean_sink_rms': defaultdict(list),
            'p95_sink_rms': defaultdict(list)
        }
        self.sink_head_occurrences = defaultdict(lambda: defaultdict(int))

        self.sample_count = 0
        self.samples_metadata = []

        # Hooks storage
        self.hooks = []

        # Layer-level accumulated statistics (sum; will divide by sample_count later)
        self.accumulated_stats = {
            'first_token_attn': [0.0] * num_layers,
            'audio_mean_attn': [0.0] * num_layers,
            'audio_start_attn': [0.0] * num_layers,
            'non_audio_mean_attn': [0.0] * num_layers,
        }

        # Head-level accumulated statistics (sum per head)
        self.headwise_stats = defaultdict(lambda: defaultdict(lambda: {
            'first_token_attn': 0.0,
            'audio_mean_attn': 0.0,
            'audio_start_attn': 0.0,
            'non_audio_mean_attn': 0.0
        }))
        
        # NEW: Detailed audio attention logging for distribution analysis
        # 记录每个样本的原始音频注意力分布，用于分析和可视化
        self.audio_attention_distributions = []  # List of dicts with per-sample stats
        self.per_token_audio_scores = []  # Store per-token scores for visualization
        
    def prepare_audio(self, audio_path):
        """Load and prepare audio"""
        try:
            audio_np, sr = librosa.load(audio_path, sr=16000, mono=True)
            return audio_np
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None
    
    def register_attention_hooks(self):
        """Register hooks to capture attention weights from all layers - 参考 sink_test.py"""
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                # output is a tuple: (hidden_states, attention_weights, ...)
                # attention_weights shape: [batch, num_heads, seq_len, seq_len]
                # Debug: 检查 output 的格式
                if not isinstance(output, (tuple, list)):
                    if layer_idx == 0:  # 只在第一层打印调试信息，避免过多输出
                        print(f"  DEBUG hook layer {layer_idx}: output is not tuple/list, type={type(output)}")
                    return output
                
                if len(output) <= 1:
                    if layer_idx == 0:
                        print(f"  DEBUG hook layer {layer_idx}: output length={len(output)}, expected >= 2")
                    return output
                
                if output[1] is None:
                    if layer_idx == 0:
                        print(f"  DEBUG hook layer {layer_idx}: output[1] is None, output type={type(output)}, len={len(output)}")
                        # 尝试查看 output 的所有元素
                        if len(output) > 0:
                            print(f"    output[0] type: {type(output[0])}, shape: {output[0].shape if hasattr(output[0], 'shape') else 'N/A'}")
                    return output
                
                if isinstance(output, (tuple, list)) and len(output) > 1 and output[1] is not None:
                    attn_output = output[0]
                    attn_weights = output[1]  # [batch, num_heads, seq_len, seq_len]

                    # 只保存最后一个 token 对所有 token 的注意力
                    batch_size, num_heads, seq_len, _ = attn_weights.shape

                    if self.num_heads is None:
                        self.num_heads = num_heads

                    last_token_attn = attn_weights[0, :, -1, :].detach()

                    # 保存当前步的注意力（用于 sink 统计和最终可视化）
                    for head_idx in range(num_heads):
                        key = (layer_idx, head_idx)
                        self.current_sample_attentions[key] = last_token_attn[head_idx].detach().to(device="cpu")
                    
                    if self.current_audio_indices is not None:
                        self.audio_sink_selector.update(
                            layer_idx=layer_idx,
                            attn_output=attn_output,
                            attn_weights=attn_weights,
                            audio_indices=self.current_audio_indices
                        )

                    # 如果启用音频头打分，立即累积统计（在 GPU 上处理更快）
                    if self.enable_head_scoring and self.current_audio_indices is not None:
                        # 在 layer 0 追踪序列长度变化，用于统计生成步数
                        if layer_idx == 0:
                            current_seq_len = last_token_attn.shape[-1]
                            if self._prev_seq_len is None:
                                # 第一次调用：这是 prompt 处理阶段，记录初始长度
                                self._prev_seq_len = current_seq_len
                                self._is_generation_started = False
                            else:
                                # 后续调用：检查是否是新的生成步骤
                                if current_seq_len > self._prev_seq_len:
                                    # 序列长度增加，说明生成了新 token
                                    if not self._is_generation_started:
                                        # 标记生成阶段开始（跳过 prompt 处理）
                                        self._is_generation_started = True
                                        self._prev_seq_len = current_seq_len
                                    else:
                                        # 生成阶段，增加计数
                                        self.total_generated_tokens += 1
                                        self._prev_seq_len = current_seq_len
                                elif current_seq_len == self._prev_seq_len and self._is_generation_started:
                                    # 同一生成步，但所有层都会调用，只计数一次（在 layer 0）
                                    pass
                        self._accumulate_audio_head_scores_in_hook(layer_idx, last_token_attn)

                    output_list = list(output)
                    output_list[1] = None

                    del attn_weights, last_token_attn

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    return tuple(output_list)
                return output

            return hook
        
        # Register hooks for each layer in Gemma3n model
        # Gemma3n model structure: model.model.language_model.layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model'):
            base_model = self.model.model.language_model
            if hasattr(base_model, 'layers'):
                layers = base_model.layers
                self.num_layers = len(layers)  # Update num_layers if needed
                for layer_idx, layer in enumerate(layers):
                    if hasattr(layer, 'self_attn'):
                        hook = layer.self_attn.register_forward_hook(make_hook(layer_idx))
                        self.hooks.append(hook)
                print(f"Registered attention hooks for {len(self.hooks)} layers")
            else:
                print("Warning: Could not find language_model.layers, hooks not registered")
        else:
            print("Warning: Could not find model.model.language_model, hooks not registered")
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print("Removed all attention hooks")
    
    def _accumulate_statistics(self):
        """Accumulate layer/head statistics for later averaging."""
        if not self.current_sample_attentions:
            print("Warning: No attention captured for current sample")
            return

        audio_indices = self.current_audio_indices

        if not audio_indices:
            print("Warning: Audio region not defined")
            return

        audio_indices = sorted(audio_indices)
        audio_start = audio_indices[0]
        audio_end = audio_indices[-1] + 1

        sample_len = next(iter(self.current_sample_attentions.values())).shape[0]
        
        # 调试信息：检查序列长度和音频区域
        print(f"  DEBUG: sample_len={sample_len}, audio_indices range=[{audio_start}, {audio_end-1}], count={len(audio_indices)}")
        
        # 过滤超出范围的音频索引
        valid_audio_indices = [idx for idx in audio_indices if idx < sample_len]
        if len(valid_audio_indices) != len(audio_indices):
            print(f"  WARNING: {len(audio_indices) - len(valid_audio_indices)} audio indices out of range (>= {sample_len})")
        
        audio_mask = torch.zeros(sample_len, dtype=torch.bool)
        if valid_audio_indices:
            audio_mask[valid_audio_indices] = True
        else:
            print("  ERROR: No valid audio indices in range!")
            return
        non_audio_mask = ~audio_mask
        
        # NEW: 记录当前样本的详细音频注意力分布
        sample_distribution = {
            'sample_id': self.sample_count,
            'audio_token_count': len(valid_audio_indices),
            'layers': {}
        }

        # Accumulate per-head metrics
        for (layer_idx, head_idx), last_token_attn in self.current_sample_attentions.items():
            # Ensure tensor on CPU float for numerical stability
            attn_cpu = last_token_attn.float()

            first_token_attn = float(attn_cpu[0].item()) if attn_cpu.numel() > 0 else 0.0

            audio_region = attn_cpu[audio_mask]
            audio_mean_attn = float(audio_region.mean().item()) if audio_region.numel() > 0 else 0.0
            audio_start_attn = float(attn_cpu[audio_start].item()) if audio_start < attn_cpu.numel() else 0.0

            if non_audio_mask.any():
                non_audio_mean_attn = float(attn_cpu[non_audio_mask].mean().item())
            else:
                non_audio_mean_attn = 0.0
            
            # NEW: 记录详细的音频区域统计（min/max/sum）
            if audio_region.numel() > 0:
                audio_min = float(audio_region.min().item())
                audio_max = float(audio_region.max().item())
                audio_sum = float(audio_region.sum().item())
                audio_std = float(audio_region.std().item())
                
                # 初始化层级字典
                if layer_idx not in sample_distribution['layers']:
                    sample_distribution['layers'][layer_idx] = {
                        'heads': {},
                        'layer_audio_mean': 0.0,
                        'layer_audio_sum': 0.0,
                        'layer_audio_max': 0.0
                    }
                
                # 记录每个头的详细统计
                sample_distribution['layers'][layer_idx]['heads'][head_idx] = {
                    'audio_mean': audio_mean_attn,
                    'audio_min': audio_min,
                    'audio_max': audio_max,
                    'audio_sum': audio_sum,
                    'audio_std': audio_std,
                    'first_token': first_token_attn
                }
                
                # 更新层级统计
                sample_distribution['layers'][layer_idx]['layer_audio_sum'] += audio_sum
                sample_distribution['layers'][layer_idx]['layer_audio_max'] = max(
                    sample_distribution['layers'][layer_idx]['layer_audio_max'], audio_max
                )
            
            # 调试：每层打印第一个头的详细信息
            if layer_idx == 0 and head_idx == 0 and self.sample_count == 0:
                print(f"  DEBUG Layer 0 Head 0 (first sample): audio_mean={audio_mean_attn:.6f}, first_token={first_token_attn:.6f}")
                print(f"  DEBUG: audio_region sum={audio_region.sum():.6f}, max={audio_region.max():.6f}, min={audio_region.min():.6f}")
                print(f"  DEBUG: audio_region size={audio_region.numel()}, values range=[{audio_region.min():.6f}, {audio_region.max():.6f}]")

            head_stats = self.headwise_stats[layer_idx][head_idx]
            head_stats['first_token_attn'] += first_token_attn
            head_stats['audio_mean_attn'] += audio_mean_attn
            head_stats['audio_start_attn'] += audio_start_attn
            head_stats['non_audio_mean_attn'] += non_audio_mean_attn

        # Accumulate per-layer averages
        for layer_idx in range(self.num_layers):
            layer_heads = [self.current_sample_attentions[(l, h)] for (l, h) in self.current_sample_attentions if l == layer_idx]
            if not layer_heads:
                continue

            layer_first = []
            layer_audio_mean = []
            layer_audio_start = []
            layer_non_audio = []

            for attn in layer_heads:
                attn_cpu = attn.float()

                layer_first.append(float(attn_cpu[0].item()) if attn_cpu.numel() > 0 else 0.0)

                # 使用 valid_audio_indices 而不是原始的 audio_indices
                if valid_audio_indices:
                    audio_region = attn_cpu[audio_mask]
                    layer_audio_mean.append(float(audio_region.mean().item()) if audio_region.numel() > 0 else 0.0)
                else:
                    layer_audio_mean.append(0.0)
                    
                layer_audio_start.append(float(attn_cpu[audio_start].item()) if audio_start < attn_cpu.numel() else 0.0)

                if non_audio_mask.any():
                    layer_non_audio.append(float(attn_cpu[non_audio_mask].mean().item()))
                else:
                    layer_non_audio.append(0.0)

            self.accumulated_stats['first_token_attn'][layer_idx] += float(np.mean(layer_first))
            self.accumulated_stats['audio_mean_attn'][layer_idx] += float(np.mean(layer_audio_mean))
            self.accumulated_stats['audio_start_attn'][layer_idx] += float(np.mean(layer_audio_start))
            self.accumulated_stats['non_audio_mean_attn'][layer_idx] += float(np.mean(layer_non_audio))
            
            # 计算层级平均值
            if layer_idx in sample_distribution['layers'] and len(layer_audio_mean) > 0:
                sample_distribution['layers'][layer_idx]['layer_audio_mean'] = float(np.mean(layer_audio_mean))
        
        # 保存当前样本的分布统计
        self.audio_attention_distributions.append(sample_distribution)
        
        # 打印当前样本的摘要
        if sample_distribution['layers']:
            layer_means = [sample_distribution['layers'][l]['layer_audio_mean'] 
                          for l in sorted(sample_distribution['layers'].keys())]
            print(f"  Sample {self.sample_count} audio attention summary:")
            print(f"    Layer-wise audio_mean range: [{min(layer_means):.6f}, {max(layer_means):.6f}]")
            print(f"    Average across layers: {np.mean(layer_means):.6f}")

    def _record_sink_summary(self, sample_name: str, sink_summary: Dict[int, Dict[str, object]]) -> None:
        """Store per-sample AttnSink head statistics and update aggregates."""
        self.sample_sink_statistics.append({
            'sample_name': sample_name,
            'layers': sink_summary
        })

        for layer_idx, stats in sink_summary.items():
            layer_id = int(layer_idx)
            sink_portion = stats.get('sink_portion', [])
            audio_mass = stats.get('audio_mass', [])
            sink_mass = stats.get('sink_mass', [])
            candidate_heads = stats.get('candidate_heads', [])
            max_sink_rms = stats.get('max_sink_rms')
            mean_sink_rms = stats.get('mean_sink_rms')
            p95_sink_rms = stats.get('p95_sink_rms')

            if sink_portion:
                self.sink_metric_tracker['sink_portion'][layer_id].extend(
                    float(val) for val in sink_portion
                )
            if audio_mass:
                self.sink_metric_tracker['audio_mass'][layer_id].extend(
                    float(val) for val in audio_mass
                )
            if sink_mass:
                self.sink_metric_tracker['sink_mass'][layer_id].extend(
                    float(val) for val in sink_mass
                )
            if max_sink_rms is not None:
                self.sink_metric_tracker['max_sink_rms'][layer_id].append(float(max_sink_rms))
            if mean_sink_rms is not None:
                self.sink_metric_tracker['mean_sink_rms'][layer_id].append(float(mean_sink_rms))
            if p95_sink_rms is not None:
                self.sink_metric_tracker['p95_sink_rms'][layer_id].append(float(p95_sink_rms))

            self.sink_metric_tracker['num_qualified'][layer_id].append(len(candidate_heads))

            for head_idx in candidate_heads:
                self.sink_head_occurrences[layer_id][int(head_idx)] += 1
    
    def generate_averaged_visualizations(self):
        """Generate averaged visualization figures for accumulated statistics."""
        if self.sample_count == 0:
            print("No samples processed, skipping visualization.")
            return

        print(f"\n{'='*60}")
        print(f"Generating averaged visualizations from {self.sample_count} samples...")
        print(f"{'='*60}")

        avg_stats = {}
        for key in self.accumulated_stats:
            avg_stats[key] = [val / self.sample_count for val in self.accumulated_stats[key]]

        if self.samples_metadata:
            avg_audio_start = int(np.mean([m['audio_token_start'] for m in self.samples_metadata if m['audio_token_start'] is not None]))
            avg_audio_length = int(np.mean([m['audio_token_length'] for m in self.samples_metadata if m['audio_token_length'] > 0]))
            print(f"Average audio region: start={avg_audio_start}, length={avg_audio_length}")

        def normalize_scores(scores):
            scores_arr = np.array(scores)
            min_val = scores_arr.min()
            max_val = scores_arr.max()
            if max_val > min_val:
                normalized = (scores_arr - min_val) / (max_val - min_val)
            else:
                normalized = scores_arr
            return normalized.tolist()

        normalized_stats = {key: normalize_scores(values) for key, values in avg_stats.items()}

        layers = list(range(self.num_layers))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.plot(layers, avg_stats['first_token_attn'], 'o-', linewidth=2, markersize=4)
        ax.set_title('First Token Attention (Averaged)', fontsize=14)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Attention Score', fontsize=12)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(layers, avg_stats['audio_mean_attn'], 'o-', color='red', linewidth=2, markersize=4)
        ax.set_title('Audio Region Mean Attention (Averaged)', fontsize=14)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Attention Score', fontsize=12)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(layers, avg_stats['audio_start_attn'], 'o-', color='green', linewidth=2, markersize=4)
        ax.set_title('Audio Start Token Attention (Averaged)', fontsize=14)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Attention Score', fontsize=12)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(layers, avg_stats['non_audio_mean_attn'], 'o-', color='blue', linewidth=2, markersize=4)
        ax.set_title('Non-Audio Region Mean Attention (Averaged)', fontsize=14)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Attention Score', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / "averaged_attention_statistics.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.plot(layers, normalized_stats['first_token_attn'], 'o-', linewidth=2, markersize=4)
        ax.set_title('First Token Attention (Normalized)', fontsize=14)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Normalized Score [0,1]', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        ax = axes[0, 1]
        ax.plot(layers, normalized_stats['audio_mean_attn'], 'o-', color='red', linewidth=2, markersize=4)
        ax.set_title('Audio Region Mean Attention (Normalized)', fontsize=14)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Normalized Score [0,1]', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        ax = axes[1, 0]
        ax.plot(layers, normalized_stats['audio_start_attn'], 'o-', color='green', linewidth=2, markersize=4)
        ax.set_title('Audio Start Token Attention (Normalized)', fontsize=14)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Normalized Score [0,1]', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        ax = axes[1, 1]
        ax.plot(layers, normalized_stats['non_audio_mean_attn'], 'o-', color='blue', linewidth=2, markersize=4)
        ax.set_title('Non-Audio Region Mean Attention (Normalized)', fontsize=14)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Normalized Score [0,1]', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        save_path = self.output_dir / "averaged_attention_statistics_normalized.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(layers, avg_stats['first_token_attn'], 'o-', label='First Token', linewidth=2, markersize=4)
        ax.plot(layers, avg_stats['audio_mean_attn'], 's-', label='Audio Region Mean', linewidth=2, markersize=4)
        ax.plot(layers, avg_stats['audio_start_attn'], '^-', label='Audio Start Token', linewidth=2, markersize=4)
        ax.plot(layers, avg_stats['non_audio_mean_attn'], 'd-', label='Non-Audio Region Mean', linewidth=2, markersize=4)
        ax.set_title(f'Attention Statistics Comparison (Averaged over {self.sample_count} samples)', fontsize=14)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Attention Score', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = self.output_dir / "attention_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(layers, normalized_stats['first_token_attn'], 'o-', label='First Token', linewidth=2, markersize=4)
        ax.plot(layers, normalized_stats['audio_mean_attn'], 's-', label='Audio Region Mean', linewidth=2, markersize=4)
        ax.plot(layers, normalized_stats['audio_start_attn'], '^-', label='Audio Start Token', linewidth=2, markersize=4)
        ax.plot(layers, normalized_stats['non_audio_mean_attn'], 'd-', label='Non-Audio Region Mean', linewidth=2, markersize=4)
        ax.set_title(f'Attention Statistics Comparison (Normalized, Averaged over {self.sample_count} samples)', fontsize=14)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Normalized Score [0,1]', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        save_path = self.output_dir / "attention_comparison_normalized.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

        # Generate headwise heatmaps (可能很耗时和内存)
        if not self.skip_heatmaps:
            self._generate_headwise_heatmap()
        else:
            print("\nSkipping head-wise heatmaps (skip_heatmaps=True)")

        print(f"\nGenerated visualizations for averaged statistics (with normalization)")

    def _generate_headwise_heatmap(self):
        if self.num_heads is None or self.num_layers == 0:
            print("No head information available for heatmap generation.")
            return

        print("\nGenerating head-wise heatmaps...")

        metrics = ['first_token_attn', 'audio_mean_attn', 'audio_start_attn', 'non_audio_mean_attn']
        metric_names = ['First Token Attention', 'Audio Region Mean Attention',
                        'Audio Start Token Attention', 'Non-Audio Region Mean Attention']

        for metric, metric_name in zip(metrics, metric_names):
            heatmap_data = np.zeros((self.num_layers, self.num_heads))

            for layer_idx in self.headwise_stats:
                for head_idx in self.headwise_stats[layer_idx]:
                    value = self.headwise_stats[layer_idx][head_idx][metric] / self.sample_count
                    heatmap_data[layer_idx, head_idx] = value

            min_val = heatmap_data.min()
            max_val = heatmap_data.max()
            print(f"  {metric_name}: min={min_val:.6f}, max={max_val:.6f}")

            if max_val > min_val:
                heatmap_data_normalized = (heatmap_data - min_val) / (max_val - min_val)
            else:
                heatmap_data_normalized = heatmap_data

            fig, ax = plt.subplots(figsize=(14, 10))
            im = ax.imshow(heatmap_data, cmap='Greens', aspect='auto', interpolation='nearest')
            ax.set_xlabel('Head Index', fontsize=12)
            ax.set_ylabel('Layer Index', fontsize=12)
            ax.set_title(f'{metric_name} (Averaged over {self.sample_count} samples)', fontsize=14)
            ax.set_xticks(np.arange(0, self.num_heads, max(1, self.num_heads // 16)))
            ax.set_yticks(np.arange(0, self.num_layers, max(1, self.num_layers // 16)))
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Attention Score', rotation=270, labelpad=20, fontsize=11)
            plt.tight_layout()
            save_path = self.output_dir / f"headwise_{metric}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {save_path}")

            fig, ax = plt.subplots(figsize=(14, 10))
            im = ax.imshow(heatmap_data_normalized, cmap='Greens', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
            ax.set_xlabel('Head Index', fontsize=12)
            ax.set_ylabel('Layer Index', fontsize=12)
            ax.set_title(f'{metric_name} (Normalized, Averaged over {self.sample_count} samples)', fontsize=14)
            ax.set_xticks(np.arange(0, self.num_heads, max(1, self.num_heads // 16)))
            ax.set_yticks(np.arange(0, self.num_layers, max(1, self.num_layers // 16)))
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Normalized Score [0,1]', rotation=270, labelpad=20, fontsize=11)
            plt.tight_layout()
            save_path = self.output_dir / f"headwise_{metric}_normalized.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {save_path}")

            del heatmap_data, heatmap_data_normalized
            gc.collect()

        print("Head-wise heatmaps generation completed.")
    
    def visualize_audio_attention_distributions(self):
        """
        生成音频注意力分布的详细可视化
        1. 每个样本的 min/max/sum 统计
        2. 跨层的贡献度热力图
        3. Top contributing heads/layers
        """
        if not self.audio_attention_distributions:
            print("No audio attention distribution data available")
            return
        
        print(f"\n{'='*60}")
        print("Generating Audio Attention Distribution Visualizations")
        print(f"{'='*60}")
        
        # 1. 绘制每个样本的音频注意力统计（min/max/sum/mean）
        num_samples = len(self.audio_attention_distributions)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 收集每层的统计数据
        layer_audio_means = defaultdict(list)
        layer_audio_sums = defaultdict(list)
        layer_audio_maxs = defaultdict(list)
        
        for sample_dist in self.audio_attention_distributions:
            for layer_idx, layer_data in sample_dist['layers'].items():
                layer_audio_means[layer_idx].append(layer_data['layer_audio_mean'])
                layer_audio_sums[layer_idx].append(layer_data['layer_audio_sum'])
                layer_audio_maxs[layer_idx].append(layer_data['layer_audio_max'])
        
        layers = sorted(layer_audio_means.keys())
        
        # Plot 1: Audio Mean per Layer (averaged across samples)
        ax = axes[0, 0]
        means_per_layer = [np.mean(layer_audio_means[l]) for l in layers]
        stds_per_layer = [np.std(layer_audio_means[l]) for l in layers]
        ax.errorbar(layers, means_per_layer, yerr=stds_per_layer, fmt='o-', linewidth=2, 
                    capsize=5, capthick=2, markersize=6)
        ax.set_title('Audio Region Mean Attention per Layer\n(with std across samples)', fontsize=13)
        ax.set_xlabel('Layer Index', fontsize=11)
        ax.set_ylabel('Mean Attention Score', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Audio Sum per Layer
        ax = axes[0, 1]
        sums_per_layer = [np.mean(layer_audio_sums[l]) for l in layers]
        ax.plot(layers, sums_per_layer, 'o-', color='red', linewidth=2, markersize=6)
        ax.set_title('Audio Region Sum Attention per Layer\n(averaged across samples)', fontsize=13)
        ax.set_xlabel('Layer Index', fontsize=11)
        ax.set_ylabel('Sum Attention Score', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Audio Max per Layer
        ax = axes[1, 0]
        maxs_per_layer = [np.mean(layer_audio_maxs[l]) for l in layers]
        ax.plot(layers, maxs_per_layer, 'o-', color='green', linewidth=2, markersize=6)
        ax.set_title('Audio Region Max Attention per Layer\n(averaged across samples)', fontsize=13)
        ax.set_xlabel('Layer Index', fontsize=11)
        ax.set_ylabel('Max Attention Score', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Distribution comparison (box plot)
        ax = axes[1, 1]
        data_for_box = [layer_audio_means[l] for l in layers[::4]]  # Sample every 4 layers
        ax.boxplot(data_for_box, labels=[str(l) for l in layers[::4]])
        ax.set_title('Audio Mean Distribution (sampled layers)', fontsize=13)
        ax.set_xlabel('Layer Index', fontsize=11)
        ax.set_ylabel('Mean Attention Score', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.output_dir / "audio_attention_distributions.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
        
        # 2. 生成层级和头级贡献热力图
        self._visualize_contribution_heatmap()
        
        # 3. 识别并打印 Top Contributing Heads/Layers
        self._identify_top_contributors()
        
        # 4. 保存详细的分布数据为 JSON
        distribution_file = self.output_dir / "audio_attention_distribution_details.json"
        with open(distribution_file, 'w', encoding='utf-8') as f:
            json.dump({
                'samples': self.audio_attention_distributions,
                'layer_statistics': {
                    str(l): {
                        'mean': float(np.mean(layer_audio_means[l])),
                        'std': float(np.std(layer_audio_means[l])),
                        'sum_avg': float(np.mean(layer_audio_sums[l])),
                        'max_avg': float(np.mean(layer_audio_maxs[l]))
                    }
                    for l in layers
                }
            }, f, indent=2)
        print(f"  Saved detailed distribution data: {distribution_file}")

    def save_audio_sink_summary(self) -> None:
        """Persist aggregated AttnSink head statistics to disk."""
        if not self.sample_sink_statistics:
            print("No audio sink head statistics recorded; skipping sink summary save.")
            return

        aggregated = {}
        layer_ids = set()
        for tracker in self.sink_metric_tracker.values():
            layer_ids.update(tracker.keys())

        for layer_id in sorted(layer_ids):
            portions = self.sink_metric_tracker['sink_portion'].get(layer_id, [])
            audio_mass = self.sink_metric_tracker['audio_mass'].get(layer_id, [])
            sink_mass = self.sink_metric_tracker['sink_mass'].get(layer_id, [])
            qualified_counts = self.sink_metric_tracker['num_qualified'].get(layer_id, [])
            max_sink_rms_vals = self.sink_metric_tracker['max_sink_rms'].get(layer_id, [])
            mean_sink_rms_vals = self.sink_metric_tracker['mean_sink_rms'].get(layer_id, [])
            p95_sink_rms_vals = self.sink_metric_tracker['p95_sink_rms'].get(layer_id, [])

            head_occurrence = self.sink_head_occurrences.get(layer_id, {})
            top_heads = sorted(
                head_occurrence.items(),
                key=lambda item: item[1],
                reverse=True
            )

            aggregated[str(layer_id)] = {
                'mean_sink_portion': float(np.mean(portions)) if portions else 0.0,
                'median_sink_portion': float(np.median(portions)) if portions else 0.0,
                'mean_audio_mass': float(np.mean(audio_mass)) if audio_mass else 0.0,
                'mean_sink_mass': float(np.mean(sink_mass)) if sink_mass else 0.0,
                'avg_num_qualified_heads': float(np.mean(qualified_counts)) if qualified_counts else 0.0,
                'max_sink_rms_stats': {
                    'mean': float(np.mean(max_sink_rms_vals)) if max_sink_rms_vals else 0.0,
                    'max': float(np.max(max_sink_rms_vals)) if max_sink_rms_vals else 0.0,
                    'p95': float(np.percentile(max_sink_rms_vals, 95)) if len(max_sink_rms_vals) > 1 else (float(max_sink_rms_vals[0]) if max_sink_rms_vals else 0.0)
                },
                'mean_sink_rms_avg': float(np.mean(mean_sink_rms_vals)) if mean_sink_rms_vals else 0.0,
                'p95_sink_rms_avg': float(np.mean(p95_sink_rms_vals)) if p95_sink_rms_vals else 0.0,
                'top_candidate_heads': [
                    {'head': int(head_idx), 'count': int(count)}
                    for head_idx, count in top_heads[:10]
                ]
            }

        all_max_rms = [
            val
            for values in self.sink_metric_tracker['max_sink_rms'].values()
            for val in values
            if val is not None
        ]
        if all_max_rms:
            recommended_tau = {
                'p90': float(np.percentile(all_max_rms, 90)),
                'p95': float(np.percentile(all_max_rms, 95)),
                'p99': float(np.percentile(all_max_rms, 99)),
                'max': float(np.max(all_max_rms)),
                'mean': float(np.mean(all_max_rms))
            }
        else:
            recommended_tau = {}

        summary = {
            'params': asdict(self.sink_params),
            'num_samples': self.sample_count,
            'per_sample': self.sample_sink_statistics,
            'aggregated': aggregated,
            'recommended_tau': recommended_tau
        }

        summary_path = self.output_dir / "audio_sink_head_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print(f"Audio sink head summary saved to: {summary_path}")
    
    def _visualize_contribution_heatmap(self):
        """生成头级和层级贡献度热力图"""
        if not self.audio_attention_distributions or self.num_heads is None:
            return
        
        print("\nGenerating contribution heatmaps...")
        
        # 计算每个头的平均音频注意力总和（表示贡献度）
        head_contributions = np.zeros((self.num_layers, self.num_heads))
        head_max_attentions = np.zeros((self.num_layers, self.num_heads))
        
        for sample_dist in self.audio_attention_distributions:
            for layer_idx, layer_data in sample_dist['layers'].items():
                for head_idx, head_data in layer_data['heads'].items():
                    head_contributions[layer_idx, head_idx] += head_data['audio_sum']
                    head_max_attentions[layer_idx, head_idx] = max(
                        head_max_attentions[layer_idx, head_idx], 
                        head_data['audio_max']
                    )
        
        # 平均化
        num_samples = len(self.audio_attention_distributions)
        head_contributions /= num_samples
        
        # 绘制热力图
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Heatmap 1: Sum contribution
        ax = axes[0]
        im = ax.imshow(head_contributions, cmap='hot', aspect='auto', interpolation='nearest')
        ax.set_xlabel('Head Index', fontsize=12)
        ax.set_ylabel('Layer Index', fontsize=12)
        ax.set_title('Audio Attention Sum Contribution per Head\n(Higher = More contribution)', fontsize=14)
        ax.set_xticks(np.arange(0, self.num_heads, max(1, self.num_heads // 16)))
        ax.set_yticks(np.arange(0, self.num_layers, max(1, self.num_layers // 8)))
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Sum Attention Score', rotation=270, labelpad=20, fontsize=11)
        
        # Heatmap 2: Max attention
        ax = axes[1]
        im = ax.imshow(head_max_attentions, cmap='hot', aspect='auto', interpolation='nearest')
        ax.set_xlabel('Head Index', fontsize=12)
        ax.set_ylabel('Layer Index', fontsize=12)
        ax.set_title('Audio Attention Max Value per Head\n(Peak attention strength)', fontsize=14)
        ax.set_xticks(np.arange(0, self.num_heads, max(1, self.num_heads // 16)))
        ax.set_yticks(np.arange(0, self.num_layers, max(1, self.num_layers // 8)))
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Max Attention Score', rotation=270, labelpad=20, fontsize=11)
        
        plt.tight_layout()
        save_path = self.output_dir / "audio_contribution_heatmap.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
    
    def _identify_top_contributors(self):
        """识别并报告贡献最大的 heads 和 layers"""
        if not self.audio_attention_distributions or self.num_heads is None:
            return
        
        print(f"\n{'='*60}")
        print("Top Contributing Heads and Layers")
        print(f"{'='*60}")
        
        # 计算每个头的平均贡献
        head_contributions = defaultdict(float)
        layer_contributions = defaultdict(float)
        
        num_samples = len(self.audio_attention_distributions)
        
        for sample_dist in self.audio_attention_distributions:
            for layer_idx, layer_data in sample_dist['layers'].items():
                layer_sum = 0.0
                for head_idx, head_data in layer_data['heads'].items():
                    contrib = head_data['audio_sum']
                    head_contributions[(layer_idx, head_idx)] += contrib
                    layer_sum += contrib
                layer_contributions[layer_idx] += layer_sum
        
        # 平均化
        for key in head_contributions:
            head_contributions[key] /= num_samples
        for key in layer_contributions:
            layer_contributions[key] /= num_samples
        
        # 排序
        top_heads = sorted(head_contributions.items(), key=lambda x: x[1], reverse=True)[:30]
        top_layers = sorted(layer_contributions.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 打印 Top Heads
        print("\nTop 30 Contributing Heads (by average audio attention sum):")
        print(f"{'Rank':<6} {'Layer':<8} {'Head':<8} {'Avg Sum':<15}")
        print("-" * 45)
        for rank, ((layer_idx, head_idx), contrib) in enumerate(top_heads, 1):
            print(f"{rank:<6} {layer_idx:<8} {head_idx:<8} {contrib:<15.6f}")
        
        # 打印 Top Layers
        print("\nTop 10 Contributing Layers (by total audio attention sum):")
        print(f"{'Rank':<6} {'Layer':<8} {'Total Sum':<15}")
        print("-" * 35)
        for rank, (layer_idx, contrib) in enumerate(top_layers, 1):
            print(f"{rank:<6} {layer_idx:<8} {contrib:<15.6f}")
        
        # 保存为 JSON
        contribution_file = self.output_dir / "top_contributors.json"
        with open(contribution_file, 'w', encoding='utf-8') as f:
            json.dump({
                'top_heads': [
                    {
                        'rank': rank,
                        'layer': layer_idx,
                        'head': head_idx,
                        'contribution': float(contrib)
                    }
                    for rank, ((layer_idx, head_idx), contrib) in enumerate(top_heads, 1)
                ],
                'top_layers': [
                    {
                        'rank': rank,
                        'layer': layer_idx,
                        'contribution': float(contrib)
                    }
                    for rank, (layer_idx, contrib) in enumerate(top_layers, 1)
                ]
            }, f, indent=2)
        print(f"\nTop contributors saved to: {contribution_file}")
    
    def analyze_and_save_results(self):
        """Analyze accumulated statistics and persist averaged summaries."""
        if self.sample_count == 0:
            print("No samples processed, skipping analysis.")
            return None

        print(f"\n{'='*60}")
        print(f"Analyzing accumulated statistics from {self.sample_count} samples...")
        print(f"{'='*60}")

        avg_stats = {}
        for key in self.accumulated_stats:
            avg_stats[key] = [val / self.sample_count for val in self.accumulated_stats[key]]

        if self.samples_metadata:
            avg_audio_start = int(np.mean([m['audio_token_start'] for m in self.samples_metadata if m['audio_token_start'] is not None]))
            avg_audio_length = int(np.mean([m['audio_token_length'] for m in self.samples_metadata if m['audio_token_length'] > 0]))
        else:
            avg_audio_start = 0
            avg_audio_length = 0

        analysis_results = {
            'num_samples': self.sample_count,
            'samples_metadata': self.samples_metadata,
            'avg_audio_token_start': avg_audio_start,
            'avg_audio_token_length': avg_audio_length,
            'layers': {}
        }

        all_first_token_attn = []
        all_audio_mean_attn = []
        all_audio_start_attn = []
        all_non_audio_mean = []

        for layer_idx in range(self.num_layers):
            first_token_attn = avg_stats['first_token_attn'][layer_idx]
            audio_mean_attn = avg_stats['audio_mean_attn'][layer_idx]
            audio_start_attn = avg_stats['audio_start_attn'][layer_idx]
            non_audio_mean = avg_stats['non_audio_mean_attn'][layer_idx]

            is_global_sink = first_token_attn > (audio_mean_attn * 2)
            is_audio_start_sink = audio_start_attn > (non_audio_mean * 2)

            layer_stats = {
                'first_token_attn': float(first_token_attn),
                'audio_mean_attn': float(audio_mean_attn),
                'audio_start_attn': float(audio_start_attn),
                'non_audio_mean_attn': float(non_audio_mean),
                'is_global_sink': bool(is_global_sink),
                'is_audio_start_sink': bool(is_audio_start_sink),
            }

            analysis_results['layers'][layer_idx] = layer_stats

            all_first_token_attn.append(first_token_attn)
            all_audio_mean_attn.append(audio_mean_attn)
            all_audio_start_attn.append(audio_start_attn)
            all_non_audio_mean.append(non_audio_mean)

            print(f"  Layer {layer_idx}:")
            print(f"    First token: {first_token_attn:.4f}, Audio mean: {audio_mean_attn:.4f}, Global sink: {is_global_sink}")

        analysis_results['summary'] = {
            'avg_first_token_attn_all_layers': float(np.mean(all_first_token_attn)) if all_first_token_attn else 0.0,
            'avg_audio_mean_attn_all_layers': float(np.mean(all_audio_mean_attn)) if all_audio_mean_attn else 0.0,
            'avg_audio_start_attn_all_layers': float(np.mean(all_audio_start_attn)) if all_audio_start_attn else 0.0,
            'avg_non_audio_mean_all_layers': float(np.mean(all_non_audio_mean)) if all_non_audio_mean else 0.0,
            'num_global_sink_layers': sum(1 for stats in analysis_results['layers'].values() if stats['is_global_sink']),
            'num_audio_start_sink_layers': sum(1 for stats in analysis_results['layers'].values() if stats['is_audio_start_sink']),
        }

        headwise_avg = {}
        for layer_idx in self.headwise_stats:
            headwise_avg[str(layer_idx)] = {}
            for head_idx in self.headwise_stats[layer_idx]:
                headwise_avg[str(layer_idx)][str(head_idx)] = {
                    'first_token_attn': float(self.headwise_stats[layer_idx][head_idx]['first_token_attn'] / self.sample_count),
                    'audio_mean_attn': float(self.headwise_stats[layer_idx][head_idx]['audio_mean_attn'] / self.sample_count),
                    'audio_start_attn': float(self.headwise_stats[layer_idx][head_idx]['audio_start_attn'] / self.sample_count),
                    'non_audio_mean_attn': float(self.headwise_stats[layer_idx][head_idx]['non_audio_mean_attn'] / self.sample_count),
                }

        head_analysis = {
            'num_samples': self.sample_count,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'head_stats': headwise_avg
        }

        head_output_path = self.output_dir / "averaged_head_analysis_results.json"
        with open(head_output_path, 'w', encoding='utf-8') as f:
            json.dump(head_analysis, f, indent=2, ensure_ascii=False)
        print(f"\nHeadwise analysis results saved to: {head_output_path}")

        layer_output_path = self.output_dir / "averaged_analysis_results.json"
        with open(layer_output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        print(f"\nAnalysis results saved to: {layer_output_path}")

        return analysis_results


class AudioAttentionSinkAnalyzerWithHeadScoring(AudioAttentionSinkAnalyzer):
    """扩展原有分析器,增加音频头打分功能 - 基于 SparseMM hit_score 机制"""
    
    def __init__(self, model, processor, output_dir="./audio_attention_analysis", 
                 num_layers=None, enable_head_scoring=True, task_type='librispeech', topk=128,
                 use_simple_prompt=True, sink_params: Optional[AttnSinkParams] = None,
                 skip_heatmaps=False):
        # Get num_layers from model config if not provided
        if num_layers is None:
            if hasattr(model, 'config') and hasattr(model.config, 'text_config'):
                num_layers = model.config.text_config.num_hidden_layers
            elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
                num_layers = model.model.language_model.config.num_hidden_layers
            else:
                num_layers = 28  # Default fallback for Gemma3n-E2B
        
        super().__init__(model, processor, output_dir, num_layers, sink_params=sink_params,
                        skip_heatmaps=skip_heatmaps)
        
        self.enable_head_scoring = enable_head_scoring
        self.task_type = task_type
        self.topk = topk  # top-k attention checking (SparseMM 使用 topk=128 for vision)
        self.use_simple_prompt = use_simple_prompt  # 是否使用简洁 prompt（推荐）

        # 记录生成阶段序列长度,用于统计生成 token 数
        self._prev_seq_len = None
        self._is_generation_started = False
        
        if self.enable_head_scoring:
            # 音频头打分: [num_layers, num_heads, 2]
            # [0]: hit count, [1]: accumulated token string
            # 注意: 实际头数将在运行时确定,这里预分配一个安全的大小
            self.audio_head_scores = None  # 延迟初始化
            
            # 统计信息
            self.total_generated_tokens = 0
            self.total_audio_hits = 0
            
            # Token级别的命中统计 (用于时间片段分析)
            # 格式: {audio_token_index: hit_count}
            self.token_hit_counts = defaultdict(int)
            
            # 用于累积多个生成 token 的注意力
            self.current_generation_attentions = []  # 存储每个生成步的注意力
        
    def analyze_single_audio_with_generation(self, audio_path, sample_name="sample", sample_metadata=None):
        """
        分析单个音频样本 - 严格参考 sink_test.py 模式
        只生成1个token来触发注意力计算,不进行完整生成
        """
        print(f"\n{'='*60}")
        print(f"Analyzing audio: {audio_path}")
        print(f"{'='*60}")
        
        # Prepare audio
        audio_np = self.prepare_audio(audio_path)
        if audio_np is None:
            print("Failed to load audio")
            return None
        
        print(f"Audio duration: {len(audio_np)/16000:.2f}s")
        
        # Build simplified prompt based on task type
        # 参考 SparseMM 和 VLM 最佳实践:
        # 1. 使用简洁、中性的问句，避免在 prompt 中提及答案格式
        # 2. 不引入选项字母 (A/B/C/D)，防止模型直接匹配字母而非理解音频
        # 3. 让第一个生成 token 基于音频内容的深度推理，而非格式化输出
        if self.use_simple_prompt:
            if self.task_type == 'tau':
                # 简洁版：直接问场景，不提及选项
                prompt_text = "What acoustic scene is this?"
            elif self.task_type == 'vox':
                prompt_text = "What is the speaker's gender?"
            elif self.task_type == 'vox_age':
                prompt_text = "What is the speaker's age group?"
            elif self.task_type == 'desed':
                prompt_text = "What sound events can you hear in this audio?"
            elif self.task_type == 'gtzan':
                prompt_text = "What is the music genre of this audio?"
            elif self.task_type == 'had':
                prompt_text = "Is this audio of a human activity AI-generated or real?"
            elif self.task_type == 'slue':
                prompt_text = "Analyze the spoken content in this audio."
            elif self.task_type == 'vesus':
                prompt_text = "What emotion is expressed in this speech?"
            elif self.task_type == 'race':
                # RACE: 阅读理解任务，需要包含问题和选项
                if sample_metadata and 'question' in sample_metadata and 'options' in sample_metadata:
                    question = sample_metadata['question']
                    options = sample_metadata['options']
                    formatted_options = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                    prompt_text = f"Listen to this audio passage and answer the following question:\n\nQuestion: {question}\n\nOptions:\n{formatted_options}\n\nAnswer with only the letter (A, B, C, or D)."
                else:
                    prompt_text = "Listen to this audio passage and answer the question."
            else:  # ASR (librispeech)
                prompt_text = "Transcribe this audio."
        else:
            # 详细版：包含答案格式要求（可能引入偏置）
            if self.task_type == 'tau':
                prompt_text = "Listen to the audio and identify the acoustic scene. Answer with only one letter: A, B, C, or D."
            elif self.task_type == 'vox':
                prompt_text = "Listen to the audio. Is this a male or female voice? Answer with only 'male' or 'female'."
            elif self.task_type == 'vox_age':
                prompt_text = "Listen to the audio. What age group does the speaker belong to? Answer with only one letter: A, B, C, D, or E."
            elif self.task_type == 'desed':
                prompt_text = "Listen to the audio and identify the sound events. Answer with only one letter: A, B, C, or D."
            elif self.task_type == 'gtzan':
                prompt_text = "Listen to the music and identify its genre. Answer with only one letter: A, B, C, or D."
            elif self.task_type == 'had':
                prompt_text = "Listen to the audio. Is this human activity audio AI-generated or real? Answer with only one letter: A or B."
            elif self.task_type == 'slue':
                prompt_text = "Listen to the audio and answer the question. Answer with only one letter: A, B, C, or D."
            elif self.task_type == 'vesus':
                prompt_text = "Listen to the audio and identify the emotion expressed. Answer with only one letter: A, B, C, or D."
            elif self.task_type == 'race':
                # RACE: 阅读理解任务，详细版 prompt
                if sample_metadata and 'question' in sample_metadata and 'options' in sample_metadata:
                    question = sample_metadata['question']
                    options = sample_metadata['options']
                    formatted_options = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                    prompt_text = f"Listen to this audio passage being read aloud, then answer the multiple-choice question based solely on the information from the audio.\n\nQuestion: {question}\n\nOptions:\n{formatted_options}\n\nRespond with only the letter of the correct option (A, B, C, or D)."
                else:
                    prompt_text = "Listen to the audio passage and answer the question. Answer with only one letter: A, B, C, or D."
            else:  # ASR (librispeech)
                prompt_text = "Transcribe this audio."
        
        # Build messages for Gemma3n
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_np},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        # Process inputs for Gemma3n
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            audio=[audio_np],
            sampling_rate=16000,
            return_tensors="pt",
            padding=False
        )
        
        # Move to device
        float_dtype = getattr(self.model, "dtype", None) or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        for k, v in inputs.items():
            if torch.is_tensor(v):
                if k == "input_features":
                    inputs[k] = v.to(device=self.device, dtype=torch.float32)
                elif k == "input_features_mask":
                    inputs[k] = v.to(device=self.device)
                else:
                    if v.is_floating_point():
                        inputs[k] = v.to(device=self.device, dtype=float_dtype)
                    else:
                        inputs[k] = v.to(device=self.device)
        
        # Locate audio tokens (Gemma3n uses config.audio_token_id)
        audio_token_start = None
        audio_token_length = 0
        audio_token_indices = []
        
        # Get audio_token_id from model config
        global _AUDIO_TOKEN_ID
        if _AUDIO_TOKEN_ID is None:
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'audio_token_id'):
                _AUDIO_TOKEN_ID = self.model.config.audio_token_id
            else:
                _AUDIO_TOKEN_ID = 262273  # Default Gemma3n audio_token_id
        
        if hasattr(inputs, 'input_ids'):
            input_ids = inputs.input_ids[0]
            # Gemma3n uses audio_token_id to mark audio positions
            audio_positions = (input_ids == _AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0].tolist()

            if audio_positions:
                # Audio tokens are consecutive in Gemma3n (188 tokens by default)
                audio_token_indices = audio_positions
            else:
                # Try to find audio tokens by checking tokenizer
                if hasattr(self.processor, 'audio_token_id'):
                    audio_token_id = self.processor.audio_token_id
                    audio_positions = (input_ids == audio_token_id).nonzero(as_tuple=True)[0].tolist()
                    if audio_positions:
                        audio_token_indices = audio_positions

            if audio_token_indices:
                audio_token_start = int(audio_token_indices[0])
                audio_token_length = len(audio_token_indices)
                print(f"Detected audio tokens: start={audio_token_start}, length={audio_token_length}")
            else:
                print("Warning: No audio tokens detected")
                del inputs, audio_np
                return None
        
        # 设置当前样本的音频区域信息
        self.current_audio_start = audio_token_start
        self.current_audio_end = (audio_token_indices[-1] + 1) if audio_token_indices else None
        self.current_audio_indices = audio_token_indices
        self.audio_sink_selector.start_sample()
        
        # 清空临时存储
        self.current_sample_attentions = {}

        # 关键修改: 使用 generate(max_new_tokens=64) 来捕获生成阶段的注意力
        # Hook 会在每个生成步捕获注意力，我们计算所有生成 token 的平均
        # 参考 SparseMM：在生成时模型需要回顾输入来产生答案
        max_gen_tokens = 64
        print(f"Running generation with max_new_tokens={max_gen_tokens} (hooks will capture generation attention)...")
        
        # 确保模型配置了 output_attentions（如果存在配置对象）
        # Gemma3n model structure: model.model.language_model.config
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model'):
            base_model = self.model.model.language_model
            if hasattr(base_model, 'config'):
                config = base_model.config
                if hasattr(config, 'attn_implementation') and config.attn_implementation != 'eager':
                    config.attn_implementation = 'eager'
                    print(f"  Set model.config.attn_implementation='eager' (was '{config.attn_implementation}')")
                if hasattr(config, 'output_attentions'):
                    config.output_attentions = True
                    print("  Set model.config.output_attentions=True")
        
        if self.enable_head_scoring:
            # 重置序列长度追踪,确保当前样本的生成步统计准确
            self._prev_seq_len = None
            self._is_generation_started = False
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                input_features=inputs.input_features if hasattr(inputs, 'input_features') else None,
                input_features_mask=inputs.input_features_mask if hasattr(inputs, 'input_features_mask') else None,
                max_new_tokens=max_gen_tokens,  # 生成64个token
                output_attentions=True,
                return_dict_in_generate=True,
                use_cache=True,  # 启用 KV cache 以加速生成
                do_sample=False,
                temperature=0.0,
                top_p=1.0
            )

        # CRITICAL: Delete outputs immediately to free GPU memory
        del outputs
        
        # CRITICAL: Delete inputs immediately after inference to free GPU memory
        del inputs
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache immediately after inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 注意：在 generate 模式下，hook 会被多次调用（每个生成步一次）
        # - current_sample_attentions 只保存最后一个生成步的注意力（用于 sink 统计）
        # - audio_head_scores 在每个生成步通过 hook 累积（平均所有生成 token）
        print(f"Generation completed. Processing final statistics...")
        
        # 记录实际生成的 token 数（假设生成了 max_gen_tokens 个）
        # 注意：实际可能少于 max_gen_tokens（如果遇到 EOS）
        # 但由于我们在 hook 中计数，total_generated_tokens 会准确反映
        if self.enable_head_scoring:
            # Hook 会在每个 forward 时调用，包括 prompt 处理（1次）+ 生成步骤（n次）
            # 我们只想统计生成步骤的 token
            # 这里我们标记生成完成，让 _accumulate_statistics 知道开始计数
            print(f"  Audio head scoring: accumulated {self.total_generated_tokens} generation steps")
        
        # 立即处理捕获的注意力,计算统计量并累积
        # 这里只处理 sink 统计（基于最后一个 token 的注意力）
        self._accumulate_statistics()

        sink_summary = self.audio_sink_selector.finish_sample()
        if sink_summary:
            self._record_sink_summary(sample_name, sink_summary)
        
        # CRITICAL: 清空临时存储以释放内存
        for key in list(self.current_sample_attentions.keys()):
            del self.current_sample_attentions[key]
        self.current_sample_attentions.clear()
        self.current_audio_start = None
        self.current_audio_end = None
        self.current_audio_indices = None
        
        # 强制垃圾回收
        gc.collect()
        
        # 增加样本计数
        self.sample_count += 1
        
        # 保存样本元数据
        metadata = {
            'sample_name': sample_name,
            'audio_path': audio_path,
            'audio_duration': len(audio_np) / 16000,
            'audio_token_start': audio_token_start,
            'audio_token_length': audio_token_length
        }
        
        if sample_metadata:
            metadata.update(sample_metadata)
        
        self.samples_metadata.append(metadata)
        
        # Delete audio data to free CPU memory (参考 sink_test.py)
        del audio_np

        # Force garbage collection to release CPU RAM held by audio buffers
        gc.collect()

        if self.enable_head_scoring:
            # 清空序列长度追踪,避免跨样本污染
            self._prev_seq_len = None
            self._is_generation_started = False

        print(f"Sample {self.sample_count} processed.")

        return metadata

    def hit_score_audio(self, attention_matrix, audio_token_indices, topk=1):
        """SparseMM-style audio head hit scoring helper."""
        if attention_matrix is None or len(audio_token_indices) == 0:
            return 0

        num_heads = attention_matrix.shape[0]
        hit_count = 0

        for head_idx in range(num_heads):
            head_attn = attention_matrix[head_idx]
            _, indices = head_attn.topk(topk)

            for idx in indices:
                if idx.item() in audio_token_indices:
                    hit_count += 1
                    break

        return hit_count
    
    def _accumulate_audio_head_scores_in_hook(self, layer_idx, last_token_attn):
        """
        在 hook 中直接累积音频头分数，避免多次 CPU-GPU 传输
        last_token_attn: [num_heads, seq_len] tensor on GPU/CPU
        """
        # 延迟初始化
        if self.audio_head_scores is None and self.num_heads is not None:
            self.audio_head_scores = [
                [[0.0, ''] for _ in range(self.num_heads)]
                for _ in range(self.num_layers)
            ]
        
        if self.audio_head_scores is None or not self.current_audio_indices:
            return
        
        audio_token_indices = set(self.current_audio_indices)
        num_heads = last_token_attn.shape[0]
        
        # 对该层的每个头进行打分
        for head_idx in range(num_heads):
            head_attn = last_token_attn[head_idx]  # [seq_len]
            
            # 找到 top-k 注意力位置
            topk_val = min(self.topk, head_attn.numel())
            values, indices = head_attn.topk(topk_val)
            
            # 统计命中数（参考 SparseMM: 直接计数，不归一化）
            hit_count = 0
            for idx in indices:
                idx_val = idx.item()
                if idx_val in audio_token_indices:
                    hit_count += 1
                    self.token_hit_counts[idx_val] += 1
            
            # 累积分数
            if hit_count > 0 and layer_idx < len(self.audio_head_scores) and head_idx < len(self.audio_head_scores[layer_idx]):
                self.audio_head_scores[layer_idx][head_idx][0] += float(hit_count)
                self.total_audio_hits += hit_count
    

    def _accumulate_statistics(self):
        """
        重写基类方法，在累积 sink 统计的同时记录生成统计
        - Sink 统计：只使用最后一个生成 token 的注意力
        - Audio head scoring：在生成过程中通过 hook 已经累积，这里只记录元数据
        """
        # 首先调用基类的统计累积逻辑（只处理最后一个 token 的注意力）
        super()._accumulate_statistics()
        
        # 注意：total_generated_tokens 实际会统计所有 forward pass
        # 包括 prompt 处理（1次）和生成步骤（max_new_tokens 次）
        # 但对于 audio head scoring，这个总数是合理的，因为它代表了
        # 模型"查看"输入的总次数
        
        if self.enable_head_scoring and self.audio_head_scores is not None:
            print(f"  Audio head scoring: {self.total_audio_hits} hits across {self.total_generated_tokens} forward passes")

    
    def save_audio_head_scores(self):
        """
        保存音频头分数 - 参考 SparseMM 的保存格式
        格式: {"layer-head": [average_score, token_string]}
        """
        if not self.enable_head_scoring:
            return
        
        if self.audio_head_scores is None:
            print("Warning: audio_head_scores not initialized, skipping save")
            return
        
        print("\n" + "="*60)
        print("Saving Audio Head Scores (SparseMM Format)")
        print("="*60)
        
        # 使用实际的头数
        actual_num_heads = self.num_heads if self.num_heads is not None else len(self.audio_head_scores[0])
        
        # 调试：检查原始分数
        max_score = max((self.audio_head_scores[l][h][0] for l in range(self.num_layers) for h in range(actual_num_heads)), default=0)
        min_score = min((self.audio_head_scores[l][h][0] for l in range(self.num_layers) for h in range(actual_num_heads)), default=0)
        print(f"Raw score range: min={min_score:.4f}, max={max_score:.4f}")

        flat_scores = {}
        nested_scores = {}
        detailed_scores = {}

        for layer_idx in range(self.num_layers):
            nested_scores[str(layer_idx)] = {}
            for head_idx in range(actual_num_heads):
                key = f"{layer_idx}-{head_idx}"
                accumulated_score, token_str = self.audio_head_scores[layer_idx][head_idx]
                
                # 计算平均分数：累积命中数 / 总生成 token 数
                # 参考 SparseMM: 不在累积时归一化，在最后计算平均
                avg_score = accumulated_score / self.total_generated_tokens if self.total_generated_tokens > 0 else 0.0
                
                # 计算归一化分数：平均命中率（占 top-k 的比例）
                # 这表示该头在所有生成步中，平均有多少比例的 top-k 注意力落在音频区域
                normalized_score = avg_score / self.topk if self.topk > 0 else 0.0

                flat_scores[key] = avg_score
                nested_scores[str(layer_idx)][str(head_idx)] = avg_score
                detailed_scores[key] = {
                    "avg_score": avg_score,  # 平均每个生成步的命中数
                    "normalized_score": normalized_score,  # 0-1 范围，表示 top-k 中音频命中比例
                    "accumulated_score": accumulated_score,  # 所有生成步的总命中数
                    "token_history": token_str
                }

        # 处理token级别的命中统计
        token_hits_list = []
        if self.token_hit_counts:
            for token_idx, hit_count in sorted(self.token_hit_counts.items()):
                token_hits_list.append({
                    "token_index": token_idx,
                    "hit_count": hit_count,
                    "hit_rate": hit_count / (self.total_generated_tokens * actual_num_heads * self.num_layers) if self.total_generated_tokens > 0 else 0
                })
        
        payload = {
            "flat_scores": flat_scores,
            "head_scores": nested_scores,
            "details": detailed_scores,
            "token_hits": token_hits_list,
            "metadata": {
                "num_layers": self.num_layers,
                "num_heads": actual_num_heads,
                "total_samples": self.sample_count,
                "total_generated_tokens": self.total_generated_tokens,
                "total_audio_hits": self.total_audio_hits,
                "time_per_token_ms": 40  # 40ms per token
            }
        }

        # 保存为 JSON
        output_file = self.output_dir / "audio_head_scores.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

        print(f"Audio head scores saved to: {output_file}")

        # 计算并打印统计信息
        print(f"\nStatistics:")
        print(f"  Total samples processed: {self.sample_count}")
        print(f"  Total forward passes (generation steps): {self.total_generated_tokens}")
        print(f"  Total audio hits across all steps: {self.total_audio_hits}")
        print(f"  Top-k value used: {self.topk}")
        
        if self.total_generated_tokens > 0:
            # 计算平均命中率
            avg_hits_per_step = self.total_audio_hits / self.total_generated_tokens
            avg_hits_per_head_per_step = avg_hits_per_step / (self.num_layers * actual_num_heads) if (self.num_layers * actual_num_heads) > 0 else 0
            
            print(f"  Average hits per generation step: {avg_hits_per_step:.2f}")
            print(f"  Average hits per head per step: {avg_hits_per_head_per_step:.4f}")
            print(f"  Model: {self.num_layers} layers × {actual_num_heads} heads = {self.num_layers * actual_num_heads} total heads")
            
            # 检查是否有非零分数
            non_zero_heads = sum(1 for layer in self.audio_head_scores for head in layer if head[0] > 0)
            print(f"  Heads with non-zero scores: {non_zero_heads} / {self.num_layers * actual_num_heads}")

        # 按平均分数排序，打印 Top-K 音频头
        head_score_list = [
            ([int(ll) for ll in key.split("-")], score)
            for key, score in flat_scores.items()
        ]
        head_score_list = sorted(head_score_list, key=lambda x: x[1], reverse=True)

        print("\nTop 30 Audio Heads (sorted by average hits per generation step):")
        print(f"{'Rank':<6} {'Layer':<8} {'Head':<8} {'Avg Hits/Step':<15} {'Hit Rate %':<12}")
        print("-" * 65)
        for rank, (pos, score) in enumerate(head_score_list[:30], 1):
            layer_idx, head_idx = pos
            # score 已经是平均每步的命中数
            hit_rate_pct = (score / self.topk * 100) if self.topk > 0 else 0
            print(f"{rank:<6} {layer_idx:<8} {head_idx:<8} {score:<15.4f} {hit_rate_pct:<12.2f}")

        # 也保存排序后的列表（方便后续使用）
        sorted_output_file = self.output_dir / "audio_head_scores_sorted.json"
        sorted_data = {
            "head_scores": [
                {
                    "rank": rank,
                    "layer": pos[0],
                    "head": pos[1],
                    "score": score
                }
                for rank, (pos, score) in enumerate(head_score_list, 1)
            ],
            "statistics": {
                "total_samples": self.sample_count,
                "total_generated_tokens": self.total_generated_tokens,
                "total_audio_hits": self.total_audio_hits,
                "hit_rate": self.total_audio_hits / (self.total_generated_tokens * actual_num_heads * self.num_layers) if self.total_generated_tokens > 0 else 0
            }
        }

        with open(sorted_output_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_data, f, indent=2)

        print(f"Sorted audio head scores saved to: {sorted_output_file}")



def main():
    """Main function with audio head scoring"""
    parser = argparse.ArgumentParser(description='Audio Attention Sink Analysis + Audio Head Scoring')
    parser.add_argument('--task', type=str, default='librispeech', 
                        choices=['librispeech', 'tau', 'vox', 'vox_age', 'desed', 'gtzan', 'had', 'slue', 'vesus', 'race'],
                        help='Task type')
    parser.add_argument('--model_path', type=str, 
                        default="/mnt/petrelfs/wenzichen/hf_models/google/gemma-3n-E2B-it",
                        help='Path to model')
    parser.add_argument('--data_path', type=str, 
                        default="/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon",
                        help='Path to dataset root')
    parser.add_argument('--output_dir', type=str, 
                        default="./audio_attention_with_head_scoring",
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to process (建议先用少量样本测试)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda', 'gpu'],
                        help='Device to run model on')
    parser.add_argument('--gpu_id', type=str, default='7',
                        help='GPU ID(s) to use')
    parser.add_argument('--enable_head_scoring', action='store_true', default=True,
                        help='Enable audio head scoring (SparseMM-style)')
    parser.add_argument('--topk', type=int, default=128,
                        help='Top-k attention positions to check (default: 128, following SparseMM)')
    parser.add_argument('--filter_by_duration', action='store_true', default=True,
                        help='Filter samples by most common audio duration (default: True)')
    parser.add_argument('--use_simple_prompt', action='store_true', default=True,
                        help='Use simple neutral prompts without answer format hints (recommended to avoid bias)')
    parser.add_argument('--sink_tau', type=float, default=20.0,
                        help='Threshold tau for identifying sink tokens via RMS activation (AttnSink)')
    parser.add_argument('--sink_rho', type=float, default=0.5,
                        help='Maximum allowed attention portion allocated to sink tokens for a head to qualify')
    parser.add_argument('--sink_summ', type=float, default=0.2,
                        help='Minimum attention mass required on audio tokens for head qualification')
    parser.add_argument('--sink_dims', type=str, default=None,
                        help='Comma-separated sink dimensions; leave empty for automatic discovery')
    parser.add_argument('--sink_topk_dims', type=int, default=2,
                        help='Number of top RMS dims to use when sink_dims is not provided')
    parser.add_argument('--sink_min_layer', type=int, default=2,
                        help='Minimum decoder layer index to consider for AttnSink head selection')
    
    # Visualization control parameters
    parser.add_argument('--skip_visualization', action='store_true', default=True,
                        help='Skip generating visualization plots (PNG files), only save JSON statistics')
    parser.add_argument('--skip_heatmaps', action='store_true', default=True,
                        help='Skip generating head-wise heatmaps (most memory/time consuming)')
    parser.add_argument('--skip_distribution_plots', action='store_true', default=True,
                        help='Skip generating audio attention distribution plots')
    
    args = parser.parse_args()

    sink_dims = _parse_int_list(args.sink_dims)
    sink_params = AttnSinkParams(
        tau=args.sink_tau,
        rho=args.sink_rho,
        summ=args.sink_summ,
        sink_dims=sink_dims,
        topk_dims=args.sink_topk_dims,
        min_layer=args.sink_min_layer
    )
    
    # 设备配置
    device_type = args.device
    if device_type in ['cuda', 'gpu']:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        compute_device = 'cuda'
    else:
        compute_device = 'cpu'
    
    output_dir = f"{args.output_dir}_{args.task}"
    
    print(f"\n{'='*80}")
    print(f"Audio Head Scoring Analysis - {args.task.upper()}")
    print(f"{'='*80}")
    print(f"Device: {compute_device}")
    print(f"Samples: {args.num_samples}")
    print(f"Output: {output_dir}")
    print(f"Head Scoring: {'Enabled' if args.enable_head_scoring else 'Disabled'}")
    print(f"Visualization: {'Disabled (JSON only)' if args.skip_visualization else 'Enabled'}")
    if not args.skip_visualization:
        if args.skip_heatmaps:
            print(f"  - Heatmaps: Disabled")
        if args.skip_distribution_plots:
            print(f"  - Distribution plots: Disabled")
    print(f"AttnSink params: tau={sink_params.tau}, rho={sink_params.rho}, summ={sink_params.summ}, dims={sink_params.sink_dims or 'auto'}")
    
    # 加载样本
    if args.task == 'librispeech':
        test_audio_samples = load_librispeech_samples(
            os.path.join(args.data_path, "librispeech"),
            num_samples=args.num_samples
        )
    elif args.task == 'tau':
        test_audio_samples = load_tau_samples(
            os.path.join(args.data_path, "TAU"),
            num_samples=args.num_samples,
            filter_by_duration=args.filter_by_duration
        )
    elif args.task == 'vox':
        test_audio_samples = load_vox_samples(
            os.path.join(args.data_path, "VoxCeleb"),
            num_samples=args.num_samples
        )
    elif args.task == 'vox_age':
        test_audio_samples = load_vox_age_samples(
            os.path.join(args.data_path, "VoxCeleb"),
            num_samples=args.num_samples
        )
    elif args.task == 'desed':
        test_audio_samples = load_desed_samples(
            os.path.join(args.data_path, "DESED"),
            num_samples=args.num_samples
        )
    elif args.task == 'gtzan':
        test_audio_samples = load_gtzan_samples(
            os.path.join(args.data_path, "GTZAN"),
            num_samples=args.num_samples
        )
    elif args.task == 'had':
        test_audio_samples = load_had_samples(
            os.path.join(args.data_path, "HAD", "concatenated_audio"),
            num_samples=args.num_samples
        )
    elif args.task == 'slue':
        test_audio_samples = load_slue_samples(
            os.path.join(args.data_path, "SLUE"),
            num_samples=args.num_samples
        )
    elif args.task == 'vesus':
        test_audio_samples = load_vesus_samples(
            os.path.join(args.data_path, "VESUS"),
            num_samples=args.num_samples
        )
    elif args.task == 'race':
        test_audio_samples = load_race_samples(
            os.path.join(args.data_path, "race_audio"),
            num_samples=args.num_samples
        )
    else:
        print(f"Unknown task: {args.task}")
        return
    
    if not test_audio_samples:
        print("No samples loaded!")
        return
    
    # 加载模型
    print("\nLoading model...")
    processor = Gemma3nProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    if device_type == 'cuda':
        float_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = Gemma3nForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=float_dtype,
            device_map='auto',
            trust_remote_code=True
        )
    else:
        model = Gemma3nForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(compute_device)

    model = model.eval()
    
    # 创建分析器（增强版）
    analyzer = AudioAttentionSinkAnalyzerWithHeadScoring(
        model, processor, output_dir,
        enable_head_scoring=args.enable_head_scoring,
        task_type=args.task,
        topk=args.topk,
        use_simple_prompt=args.use_simple_prompt,
        sink_params=sink_params,
        skip_heatmaps=args.skip_heatmaps or args.skip_visualization  # 如果跳过所有可视化，也跳过热力图
    )
    
    # 注册 hooks
    analyzer.register_attention_hooks()
    
    # 分析样本
    for idx, sample in enumerate(test_audio_samples):
        try:
            if args.task == 'librispeech':
                sample_name = f"librispeech_sample_{idx+1:02d}_{sample['audio_id']}"
                metadata = analyzer.analyze_single_audio_with_generation(
                    sample['audio_file'], 
                    sample_name=sample_name, 
                    sample_metadata=sample
                )
            elif args.task == 'tau':
                sample_name = f"tau_sample_{idx+1:02d}_{sample['scene_label']}_{Path(sample['audio_file']).stem}"
                metadata = analyzer.analyze_single_audio_with_generation(
                    sample['audio_file'], 
                    sample_name=sample_name,
                    sample_metadata=sample
                )
            elif args.task == 'vox':
                sample_name = f"vox_sample_{idx+1:02d}_{sample['gender']}_{Path(sample['audio_file']).stem}"
                metadata = analyzer.analyze_single_audio_with_generation(
                    sample['audio_file'],
                    sample_name=sample_name,
                    sample_metadata=sample
                )
            elif args.task == 'vox_age':
                sample_name = f"vox_age_sample_{idx+1:02d}_{sample['age_group']}_{Path(sample['audio_file']).stem}"
                metadata = analyzer.analyze_single_audio_with_generation(
                    sample['audio_file'],
                    sample_name=sample_name,
                    sample_metadata=sample
                )
            elif args.task == 'desed':
                sample_name = f"desed_sample_{idx+1:02d}_{sample['task_type']}_{Path(sample['audio_file']).stem}"
                metadata = analyzer.analyze_single_audio_with_generation(
                    sample['audio_file'],
                    sample_name=sample_name,
                    sample_metadata=sample
                )
            elif args.task == 'gtzan':
                sample_name = f"gtzan_sample_{idx+1:02d}_{sample['genre']}_{Path(sample['audio_file']).stem}"
                metadata = analyzer.analyze_single_audio_with_generation(
                    sample['audio_file'],
                    sample_name=sample_name,
                    sample_metadata=sample
                )
            elif args.task == 'had':
                sample_name = f"had_sample_{idx+1:02d}_{sample['label']}_{Path(sample['audio_file']).stem}"
                metadata = analyzer.analyze_single_audio_with_generation(
                    sample['audio_file'],
                    sample_name=sample_name,
                    sample_metadata=sample
                )
            elif args.task == 'slue':
                sample_name = f"slue_sample_{idx+1:02d}_{sample['task_type']}_{Path(sample['audio_file']).stem}"
                metadata = analyzer.analyze_single_audio_with_generation(
                    sample['audio_file'],
                    sample_name=sample_name,
                    sample_metadata=sample
                )
            elif args.task == 'vesus':
                sample_name = f"vesus_sample_{idx+1:02d}_{sample['emotion_label']}_{Path(sample['audio_file']).stem}"
                metadata = analyzer.analyze_single_audio_with_generation(
                    sample['audio_file'],
                    sample_name=sample_name,
                    sample_metadata=sample
                )
            elif args.task == 'race':
                sample_name = f"race_sample_{idx+1:02d}_{sample['subset']}_{sample['article_id']}_{sample['question_idx']}"
                metadata = analyzer.analyze_single_audio_with_generation(
                    sample['audio_file'],
                    sample_name=sample_name,
                    sample_metadata=sample
                )
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 移除 hooks
    analyzer.remove_hooks()
    
    # 生成可视化和结果（根据参数控制）
    if not args.skip_visualization:
        print("\n" + "="*60)
        print("Generating visualizations...")
        print("="*60)
        analyzer.generate_averaged_visualizations()
    else:
        print("\n" + "="*60)
        print("Skipping visualizations (--skip_visualization enabled)")
        print("="*60)
    
    # 保存统计结果（JSON，总是执行）
    final_results = analyzer.analyze_and_save_results()
    
    # 生成详细的音频注意力分布可视化（根据参数控制）
    if not args.skip_visualization and not args.skip_distribution_plots:
        analyzer.visualize_audio_attention_distributions()
    elif args.skip_distribution_plots:
        print("Skipping audio attention distribution plots (--skip_distribution_plots enabled)")
    
    # 保存 Sink 统计摘要（JSON，总是执行）
    analyzer.save_audio_sink_summary()
    
    # 保存音频头分数（JSON，总是执行）
    if args.enable_head_scoring:
        analyzer.save_audio_head_scores()
    
    print(f"\n{'='*80}")
    print(f"Analysis completed!")
    print(f"Results saved in: {output_dir}")
    if args.skip_visualization:
        print(f"Note: Only JSON files were saved (visualization skipped)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

'''
使用示例 (Usage Examples)
============================

# 基本用法：LibriSpeech 任务 (ASR - Automatic Speech Recognition)
# 注意：librispeech 数据会自动从以下路径加载：
#   - 音频: {data_path}/librispeech/output_audio/*.flac
#   - JSON: {data_path}/librispeech/whisperx_parquet_output/*_whisperx.json

CUDA_VISIBLE_DEVICES=2 python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/audio_head/sink_test_with_audio_head.py \
    --task librispeech \
    --num_samples 10 \
    --gpu_id 0 \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/audio_attention_with_head_scoring/librispeech \
    --skip_visualization

CUDA_VISIBLE_DEVICES=2 python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/audio_head/sink_test_with_audio_head.py \
    --task librispeech \
    --num_samples 10 \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/audio_attention_with_head_scoring/librispeech \
    --skip_visualization

# TAU 任务 (Acoustic Scene Classification)
python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/attn_sink/sink_test_with_audio_head.py \
    --task tau \
    --num_samples 0 \
    --gpu_id 1 \
    --output_dir ./audiohead/tau \
    --skip_visualization

# VoxCeleb 性别识别任务 (Speaker Gender Classification)
CUDA_VISIBLE_DEVICES=2 python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/attn_sink/sink_test_with_audio_head.py \
    --task vox \
    --num_samples 0 \
    --gpu_id 2 \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/attn_sink/audiohead/vox \
    --skip_visualization

# VoxCeleb 年龄分类任务 (Speaker Age Classification)
CUDA_VISIBLE_DEVICES=2 python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/attn_sink/sink_test_with_audio_head.py \
    --task vox_age \
    --num_samples 0 \
    --gpu_id 3 \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/attn_sink/audiohead/vox_age \
    --skip_visualization

# DESED 音频事件检测任务 (Sound Event Detection)
CUDA_VISIBLE_DEVICES=0 python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/attn_sink/sink_test_with_audio_head.py \
    --task desed \
    --num_samples 0 \
    --gpu_id 4 \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/attn_sink/audiohead/desed \
    --skip_visualization

# GTZAN 音乐流派分类任务 (Music Genre Classification)
CUDA_VISIBLE_DEVICES=7 python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/attn_sink/sink_test_with_audio_head.py \
    --task gtzan \
    --num_samples 0 \
    --gpu_id 5 \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/attn_sink/audiohead/gtzan \
    --skip_visualization

# HAD 人类活动检测任务 (Human Activity Detection - Real vs AI-generated)
CUDA_VISIBLE_DEVICES=5 python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/attn_sink/sink_test_with_audio_head.py \
    --task had \
    --num_samples 0 \
    --gpu_id 6 \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/attn_sink/audiohead/had \
    --skip_visualization

# SLUE 语音理解任务 (Spoken Language Understanding Evaluation)
CUDA_VISIBLE_DEVICES=5 python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/attn_sink/sink_test_with_audio_head.py \
    --task slue \
    --num_samples 0 \
    --gpu_id 7 \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/attn_sink/audiohead/slue \
    --skip_visualization

# VESUS 情感识别任务 (Emotion Recognition)
CUDA_VISIBLE_DEVICES=7 python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/attn_sink/sink_test_with_audio_head.py \
    --task vesus \
    --num_samples 0 \
    --gpu_id 0 \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/attn_sink/audiohead/vesus \
    --skip_visualization

# RACE 阅读理解任务 (Reading Comprehension with Audio)
CUDA_VISIBLE_DEVICES=5 python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/attn_sink/sink_test_with_audio_head.py \
    --task race \
    --num_samples 0 \
    --gpu_id 0 \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/attn_sink/audiohead/race \
    --skip_visualization

'''

