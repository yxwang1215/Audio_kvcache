#!/usr/bin/env python3
"""
合并 word_attention_analysis.py 和 generate_head_scores.py 的功能
直接生成 audio_head_scores_aggregated.json 文件
支持对 alpha, beta, gamma 参数进行搜索实验
"""

import os
import sys
import json
import gc
import torch
import numpy as np
from pathlib import Path
import warnings
import librosa
import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import itertools

try:
    import soundfile as sf
except ImportError:
    sf = None

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# 音频 token ID
_AUDIO_TOKEN_ID = 151646
_AUDIO_BOS_TOKEN_ID = 151647
_AUDIO_EOS_TOKEN_ID = 151648


@dataclass
class WordAlignment:
    """单词对齐信息"""
    word: str
    start_time: float  # 秒
    end_time: float
    score: float
    audio_token_start: int  # 对应的音频 token 起始位置
    audio_token_end: int  # 对应的音频 token 结束位置


@dataclass
class TokenAttentionStats:
    """单个 token 的注意力统计 - 包括 top-k 和注意力权重"""
    token_id: int
    token_text: str
    word: str  # 所属单词
    word_score: float  # WhisperX 单词 score
    position: int  # 在生成序列中的位置
    
    # Top-k 注意力统计（对每个 layer/head）
    topk_hits_on_word_audio: Dict[Tuple[int, int], int] = field(default_factory=dict)  # (layer, head) -> hit count in top-k
    topk_hits_on_all_audio: Dict[Tuple[int, int], int] = field(default_factory=dict)  # (layer, head) -> total audio hits in top-k
    
    # 注意力权重统计（对每个 layer/head）
    attn_weight_on_word_audio: Dict[Tuple[int, int], float] = field(default_factory=dict)  # (layer, head) -> sum of attention weights on word audio tokens
    attn_weight_on_all_audio: Dict[Tuple[int, int], float] = field(default_factory=dict)  # (layer, head) -> sum of attention weights on all audio tokens


class WordAttentionAnalyzer:
    """分析生成单词 token 时的注意力分布 - 基于 top-k 注意力"""
    
    def __init__(self, model, processor, output_dir="./word_attention_analysis", num_layers=36, topk=128):
        self.model = model
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_layers = num_layers
        self.num_heads = None
        self.hooks = []
        self.topk = topk  # top-k 注意力位置（参考 SparseMM）
        
        # 当前样本信息
        self.current_audio_indices = None  # prefill 阶段的音频 token 位置
        self.current_word_alignments = []  # 当前样本的高质量单词对齐
        self.prefill_length = 0  # prefill 阶段的序列长度
        
        # 注意力捕获
        self.generation_attentions = []  # 每个生成步的注意力: List[Dict[(layer, head), tensor]]
        self.generated_token_ids = []  # 生成的 token IDs
        self.is_generating = False  # 标记是否在生成阶段
        
        # 统计结果
        self.token_stats = []  # 所有样本的 token 统计
        self.sample_results = []  # 每个样本的汇总结果
        
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
    
    def load_whisperx_alignment(self, json_path: str, score_threshold: float = 0.95) -> Tuple[List[WordAlignment], int]:
        """加载 WhisperX JSON 并筛选高质量单词"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        word_alignments = []
        total_words = 0
        
        if 'transcription' in data and 'word_segments' in data['transcription']:
            total_words = len(data['transcription']['word_segments'])
            for word_data in data['transcription']['word_segments']:
                if word_data['score'] >= score_threshold:
                    alignment = WordAlignment(
                        word=word_data['word'],
                        start_time=word_data['start'],
                        end_time=word_data['end'],
                        score=word_data['score'],
                        audio_token_start=-1,  # 稍后计算
                        audio_token_end=-1
                    )
                    word_alignments.append(alignment)
        
        return word_alignments, total_words
    
    def map_time_to_audio_tokens(
        self, 
        word_alignments: List[WordAlignment],
        audio_duration: float,
        audio_token_indices: List[int]
    ) -> List[WordAlignment]:
        """将时间戳映射到音频 token 位置"""
        if not audio_token_indices:
            return []
        
        num_audio_tokens = len(audio_token_indices)
        audio_start_idx = audio_token_indices[0]
        
        for alignment in word_alignments:
            start_ratio = alignment.start_time / audio_duration
            end_ratio = alignment.end_time / audio_duration
            
            token_start = int(start_ratio * num_audio_tokens)
            token_end = int(end_ratio * num_audio_tokens)
            
            token_start = max(0, min(token_start, num_audio_tokens - 1))
            token_end = max(token_start, min(token_end, num_audio_tokens - 1))
            
            alignment.audio_token_start = audio_start_idx + token_start
            alignment.audio_token_end = audio_start_idx + token_end
        
        return word_alignments
    
    def load_audio(self, audio_path):
        """加载音频文件"""
        try:
            audio_np, sr = librosa.load(audio_path, sr=16000, mono=True)
            if audio_np.ndim > 1:
                audio_np = audio_np[0]
            return audio_np
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None
    
    def register_attention_hooks(self):
        """注册 hooks 捕获生成过程中的注意力"""
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                if not isinstance(output, (tuple, list)) or len(output) <= 1 or output[1] is None:
                    return output
                
                attn_output = output[0]
                attn_weights = output[1]  # [batch, num_heads, seq_len, seq_len]
                
                batch_size, num_heads, seq_len, _ = attn_weights.shape
                
                if self.num_heads is None:
                    self.num_heads = num_heads
                
                if not self.is_generating and self.current_audio_indices:
                    if seq_len > max(self.current_audio_indices) + 1:
                        self.prefill_length = seq_len - 1
                        self.is_generating = True
                
                if self.is_generating:
                    last_token_attn = attn_weights[0, :, -1, :].detach().cpu()  # [num_heads, seq_len]
                    
                    if len(self.generation_attentions) == 0:
                        self.generation_attentions.append({})
                    
                    current_step = self.generation_attentions[-1]
                    if len(current_step) >= self.num_layers * num_heads:
                        self.generation_attentions.append({})
                        current_step = self.generation_attentions[-1]
                    
                    for head_idx in range(num_heads):
                        key = (layer_idx, head_idx)
                        current_step[key] = last_token_attn[head_idx]
                    
                    del last_token_attn
                
                output_list = list(output)
                output_list[1] = None
                del attn_weights
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return tuple(output_list)
            
            return hook
        
        if hasattr(self.model, 'thinker') and hasattr(self.model.thinker, 'model'):
            layers = self.model.thinker.model.layers
            self.num_layers = len(layers)
            for layer_idx, layer in enumerate(layers):
                hook = layer.register_forward_hook(make_hook(layer_idx))
                self.hooks.append(hook)
            print(f"Registered attention hooks for {len(self.hooks)} layers")
        else:
            print("Warning: Could not find thinker.model.layers")
    
    def remove_hooks(self):
        """移除所有 hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def align_tokens_to_words(
        self,
        token_ids: List[int],
        generated_text: str,
        word_alignments: List[WordAlignment],
        max_skip_threshold: int = 10
    ) -> List[Tuple[int, str, Optional[WordAlignment]]]:
        """将生成的 token 对齐到单词"""
        token_texts = []
        for token_id in token_ids:
            token_text = self.processor.tokenizer.decode([token_id])
            token_texts.append(token_text)
        
        token_to_word = []
        current_word_idx = 0
        accumulated_text = ""
        word_start_token_idx = 0
        skipped_words = []
        
        for i, (token_id, token_text) in enumerate(zip(token_ids, token_texts)):
            accumulated_text += token_text
            
            matched_word = None
            if current_word_idx < len(word_alignments):
                word = word_alignments[current_word_idx].word.strip()
                clean_accumulated = accumulated_text.strip().lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
                clean_word = word.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
                
                if clean_word in clean_accumulated:
                    matched_word = word_alignments[current_word_idx]
                    
                    match_start_pos = clean_accumulated.find(clean_word)
                    match_end_pos = match_start_pos + len(clean_word)
                    
                    temp_accumulated = ""
                    word_token_start = word_start_token_idx
                    word_token_end = i
                    
                    for j in range(word_start_token_idx, i + 1):
                        token_txt = token_texts[j]
                        temp_clean = (temp_accumulated + token_txt).strip().lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
                        
                        if len(temp_clean) <= match_start_pos:
                            word_token_start = j + 1
                        
                        temp_accumulated += token_txt
                        
                        temp_clean_after = temp_accumulated.strip().lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
                        if len(temp_clean_after) >= match_end_pos:
                            word_token_end = j
                            break
                    
                    for j in range(word_token_start, word_token_end + 1):
                        if j < len(token_to_word):
                            old_token_id, old_token_text, _ = token_to_word[j]
                            token_to_word[j] = (old_token_id, old_token_text, matched_word)
                        else:
                            if j == i:
                                token_to_word.append((token_id, token_text, matched_word))
                    
                    current_word_idx += 1
                    accumulated_text = ""
                    word_start_token_idx = i + 1
                    continue
                
                tokens_accumulated = i - word_start_token_idx + 1
                if tokens_accumulated > max_skip_threshold:
                    if current_word_idx + 1 < len(word_alignments):
                        next_word = word_alignments[current_word_idx + 1].word.strip()
                        clean_next_word = next_word.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
                        
                        if clean_next_word in clean_accumulated:
                            skipped_words.append(word_alignments[current_word_idx].word)
                            current_word_idx += 1
                            continue
                    
                    if len(clean_accumulated) > len(clean_word) * 3:
                        skipped_words.append(word_alignments[current_word_idx].word)
                        current_word_idx += 1
                        accumulated_text = ""
                        word_start_token_idx = i + 1
            
            if len(token_to_word) == i:
                token_to_word.append((token_id, token_text, None))
        
        if skipped_words:
            print(f"  [Warning] Skipped {len(skipped_words)} WhisperX words: {skipped_words}")
        
        return token_to_word
    
    def analyze_single_sample(
        self,
        audio_path: str,
        whisperx_json_path: str,
        sample_name: str = "sample",
        score_threshold: float = 0.95
    ):
        """分析单个样本"""
        print(f"\n{'='*80}")
        print(f"Analyzing sample: {sample_name}")
        print(f"{'='*80}")
        
        word_alignments, total_words_in_json = self.load_whisperx_alignment(whisperx_json_path, score_threshold)
        
        if not word_alignments:
            print("No high-quality words found, skipping sample")
            return
        
        audio_np = self.load_audio(audio_path)
        if audio_np is None:
            return
        
        audio_duration = len(audio_np) / 16000
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_np},
                    {"type": "text", "text": "Please transcribe the audio."}
                ],
            },
        ]
        
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True
        )
        inputs = inputs.to(device=self.device, dtype=self.dtype)
        
        input_ids = inputs.input_ids[0]
        audio_positions = (input_ids == _AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0].tolist()
        bos_positions = (input_ids == _AUDIO_BOS_TOKEN_ID).nonzero(as_tuple=True)[0].tolist()
        eos_positions = (input_ids == _AUDIO_EOS_TOKEN_ID).nonzero(as_tuple=True)[0].tolist()
        
        if audio_positions:
            audio_token_indices = audio_positions
        elif bos_positions and eos_positions:
            start_idx = bos_positions[0]
            end_idx = eos_positions[-1]
            audio_token_indices = list(range(start_idx, end_idx + 1))
        else:
            print("Warning: No audio tokens detected")
            del inputs, audio_np
            return
        
        word_alignments = self.map_time_to_audio_tokens(
            word_alignments,
            audio_duration,
            audio_token_indices
        )
        
        self.current_audio_indices = audio_token_indices
        self.current_word_alignments = word_alignments
        self.generation_attentions = []
        self.generated_token_ids = []
        self.is_generating = False
        self.prefill_length = 0
        
        print(f"Generating transcription...")
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values if hasattr(inputs, 'pixel_values') else None,
                pixel_values_videos=inputs.pixel_values_videos if hasattr(inputs, 'pixel_values_videos') else None,
                image_grid_thw=inputs.image_grid_thw if hasattr(inputs, 'image_grid_thw') else None,
                video_grid_thw=inputs.video_grid_thw if hasattr(inputs, 'video_grid_thw') else None,
                input_features=inputs.input_features if hasattr(inputs, 'input_features') else None,
                feature_attention_mask=inputs.feature_attention_mask if hasattr(inputs, 'feature_attention_mask') else None,
                use_audio_in_video=True,
                max_new_tokens=128,
                output_attentions=True,
                return_dict_in_generate=True,
                use_cache=True
            )
        
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        generated_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
        
        token_to_word = self.align_tokens_to_words(
            generated_ids.cpu().tolist(),
            generated_text,
            word_alignments
        )
        
        sample_stats = self._analyze_token_attention(token_to_word, sample_name)
        
        del inputs, outputs, audio_np
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return sample_stats
    
    def _analyze_token_attention(
        self,
        token_to_word: List[Tuple[int, str, Optional[WordAlignment]]],
        sample_name: str
    ) -> Dict:
        """分析每个 token 生成时的注意力"""
        
        sample_stats = {
            'sample_name': sample_name,
            'total_tokens': len(token_to_word),
            'high_quality_word_tokens': 0,
            'tokens': []
        }
        
        for step_idx, (token_id, token_text, word_alignment) in enumerate(token_to_word):
            if step_idx >= len(self.generation_attentions):
                continue
            
            if word_alignment is None:
                continue
            
            sample_stats['high_quality_word_tokens'] += 1
            step_attention = self.generation_attentions[step_idx]
            
            token_stat = TokenAttentionStats(
                token_id=token_id,
                token_text=token_text,
                word=word_alignment.word,
                word_score=word_alignment.score,
                position=step_idx
            )
            
            word_audio_indices = set(range(
                word_alignment.audio_token_start,
                word_alignment.audio_token_end + 1
            ))
            num_word_audio_tokens = len(word_audio_indices)  # 目标单词的音频 token 数量
            all_audio_indices = set(self.current_audio_indices)
            
            for (layer_idx, head_idx), attn in step_attention.items():
                topk_val = min(self.topk, attn.numel())
                values, indices = attn.topk(topk_val)
                
                word_hits = sum(1 for idx in indices if idx.item() in word_audio_indices)
                token_stat.topk_hits_on_word_audio[(layer_idx, head_idx)] = word_hits
                
                audio_hits = sum(1 for idx in indices if idx.item() in all_audio_indices)
                token_stat.topk_hits_on_all_audio[(layer_idx, head_idx)] = audio_hits
                
                word_audio_weight = sum(attn[idx].item() for idx in word_audio_indices if idx < len(attn))
                token_stat.attn_weight_on_word_audio[(layer_idx, head_idx)] = word_audio_weight
                
                all_audio_weight = sum(attn[idx].item() for idx in all_audio_indices if idx < len(attn))
                token_stat.attn_weight_on_all_audio[(layer_idx, head_idx)] = all_audio_weight
            
            layer_head_stats = {}
            for (layer_idx, head_idx) in token_stat.topk_hits_on_word_audio.keys():
                key = f"L{layer_idx}_H{head_idx}"
                layer_head_stats[key] = {
                    'topk_hits_on_word': token_stat.topk_hits_on_word_audio[(layer_idx, head_idx)],
                    'topk_hits_on_audio': token_stat.topk_hits_on_all_audio[(layer_idx, head_idx)],
                    'attn_weight_on_word': token_stat.attn_weight_on_word_audio[(layer_idx, head_idx)],
                    'attn_weight_on_audio': token_stat.attn_weight_on_all_audio[(layer_idx, head_idx)],
                }
            
            sample_stats['tokens'].append({
                'token_id': token_stat.token_id,
                'token_text': token_stat.token_text,
                'word': token_stat.word,
                'word_score': token_stat.word_score,
                'position': token_stat.position,
                'num_word_audio_tokens': num_word_audio_tokens,  # 目标单词的音频 token 数量
                'layer_head_stats': layer_head_stats
            })
            
            self.token_stats.append(token_stat)
        
        self.sample_results.append(sample_stats)
        return sample_stats
    
    def _compute_layer_head_statistics(self, all_tokens: List[Dict], alpha=1.0, beta=1.0, gamma=1.0) -> Dict:
        """计算每个 layer/head 的统计信息，使用指定的 alpha, beta, gamma"""
        
        layer_head_data = defaultdict(lambda: {
            'topk_hits_on_word': [],
            'topk_hits_on_audio': [],
            'attn_weight_on_word': [],
            'attn_weight_on_audio': [],
            'num_word_audio_tokens': []
        })
        
        for token in all_tokens:
            num_word_audio_tokens = token.get('num_word_audio_tokens', 0)
            layer_head_stats = token.get('layer_head_stats', {})
            
            for key, stats in layer_head_stats.items():
                if key.startswith("L") and "_H" in key:
                    layer_head_data[key]['topk_hits_on_word'].append(stats.get('topk_hits_on_word', 0))
                    layer_head_data[key]['topk_hits_on_audio'].append(stats.get('topk_hits_on_audio', 0))
                    layer_head_data[key]['attn_weight_on_word'].append(stats.get('attn_weight_on_word', 0.0))
                    layer_head_data[key]['attn_weight_on_audio'].append(stats.get('attn_weight_on_audio', 0.0))
                    layer_head_data[key]['num_word_audio_tokens'].append(num_word_audio_tokens)
        
        layer_head_summary = {}
        for key, data in layer_head_data.items():
            if not data['topk_hits_on_word']:
                continue
            
            attn_ratios = [
                w / a if a > 0 else 0.0
                for w, a in zip(data['attn_weight_on_word'], data['attn_weight_on_audio'])
            ]
            
            hit_ratios = [
                hits / self.topk if self.topk > 0 else 0.0
                for hits in data['topk_hits_on_word']
            ]
            
            coverage_ratios = [
                hits / num_tokens if num_tokens > 0 else 0.0
                for hits, num_tokens in zip(data['topk_hits_on_word'], data['num_word_audio_tokens'])
            ]
            
            S_hit = float(np.mean(hit_ratios))
            S_attn = float(np.mean(attn_ratios))
            S_cov = float(np.mean(coverage_ratios))
            
            importance_score = (
                alpha * S_hit +
                beta * S_attn +
                gamma * S_cov
            )
            
            layer_head_summary[key] = {
                'num_tokens': len(data['topk_hits_on_word']),
                'avg_topk_hits_on_word': float(np.mean(data['topk_hits_on_word'])),
                'avg_topk_hits_on_audio': float(np.mean(data['topk_hits_on_audio'])),
                'avg_attn_weight_on_word': float(np.mean(data['attn_weight_on_word'])),
                'avg_attn_weight_on_audio': float(np.mean(data['attn_weight_on_audio'])),
                'avg_attn_ratio': S_attn,
                'median_attn_ratio': float(np.median(attn_ratios)),
                'avg_hit_ratio': S_hit,
                'median_hit_ratio': float(np.median(hit_ratios)),
                'avg_coverage_ratio': S_cov,
                'median_coverage_ratio': float(np.median(coverage_ratios)),
                'importance_score': importance_score,
                'weights': {
                    'alpha': alpha,
                    'beta': beta,
                    'gamma': gamma
                }
            }
        
        return layer_head_summary
    
    def generate_aggregated_scores(self, alpha=1.0, beta=1.0, gamma=1.0, num_layers=36, num_heads=16) -> Dict:
        """生成聚合的 head scores，格式: {"layer-head": [avg_ratio, median_ratio]}"""
        
        all_tokens = [token for sample in self.sample_results for token in sample['tokens']]
        
        if not all_tokens:
            return {}
        
        layer_head_stats = self._compute_layer_head_statistics(all_tokens, alpha, beta, gamma)
        
        visual_format = {}
        for key, stats in layer_head_stats.items():
            if key.startswith("L") and "_H" in key:
                parts = key[1:].split("_H")
                if len(parts) == 2:
                    layer = int(parts[0])
                    head = int(parts[1])
                    visual_key = f"{layer}-{head}"
                    
                    avg_ratio = stats.get("avg_attn_ratio", 0.0)
                    median_ratio = stats.get("median_attn_ratio", 0.0)
                    
                    visual_format[visual_key] = [avg_ratio, median_ratio]
        
        # 确保所有头都存在
        complete_format = {}
        for layer in range(num_layers):
            for head in range(num_heads):
                key = f"{layer}-{head}"
                if key in visual_format:
                    complete_format[key] = visual_format[key]
                else:
                    complete_format[key] = [0.0, 0.0]
        
        return complete_format


def run_parameter_search(
    analyzer: WordAttentionAnalyzer,
    alpha_range: List[float],
    beta_range: List[float],
    gamma_range: List[float],
    output_dir: Path,
    num_layers: int = 36,
    num_heads: int = 16
):
    """对 alpha, beta, gamma 进行网格搜索"""
    
    search_results = []
    total_combinations = len(alpha_range) * len(beta_range) * len(gamma_range)
    
    print(f"\n{'='*80}")
    print(f"Starting parameter search")
    print(f"{'='*80}")
    print(f"Alpha range: {alpha_range}")
    print(f"Beta range: {beta_range}")
    print(f"Gamma range: {gamma_range}")
    print(f"Total combinations: {total_combinations}")
    print(f"{'='*80}\n")
    
    search_dir = output_dir / "parameter_search"
    search_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, (alpha, beta, gamma) in enumerate(itertools.product(alpha_range, beta_range, gamma_range), 1):
        print(f"[{idx}/{total_combinations}] Testing: α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}", end=" ... ")
        
        # 生成聚合分数
        aggregated_scores = analyzer.generate_aggregated_scores(
            alpha=alpha, beta=beta, gamma=gamma,
            num_layers=num_layers, num_heads=num_heads
        )
        
        # 计算统计信息
        scores_list = [v[0] for v in aggregated_scores.values()]  # 使用 avg_ratio
        mean_score = np.mean(scores_list)
        std_score = np.std(scores_list)
        max_score = np.max(scores_list)
        min_score = np.min(scores_list)
        
        # 计算 top-k heads 的统计
        sorted_scores = sorted(scores_list, reverse=True)
        top_10_mean = np.mean(sorted_scores[:10])
        top_20_mean = np.mean(sorted_scores[:20])
        top_50_mean = np.mean(sorted_scores[:50])
        
        # 计算非零 heads 的比例
        non_zero_ratio = sum(1 for v in scores_list if v > 0) / len(scores_list) if scores_list else 0.0
        
        result = {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'statistics': {
                'mean_score': float(mean_score),
                'std_score': float(std_score),
                'max_score': float(max_score),
                'min_score': float(min_score),
                'top_10_mean': float(top_10_mean),
                'top_20_mean': float(top_20_mean),
                'top_50_mean': float(top_50_mean),
                'non_zero_ratio': float(non_zero_ratio),
                'score_range': float(max_score - min_score)
            }
        }
        
        search_results.append(result)
        
        # 保存当前参数组合的结果
        param_str = f"alpha_{alpha:.2f}_beta_{beta:.2f}_gamma_{gamma:.2f}".replace('.', '_')
        output_file = search_dir / f"audio_head_scores_aggregated_{param_str}.json"
        with open(output_file, 'w') as f:
            json.dump(aggregated_scores, f, indent=4)
        
        print(f"Done (mean={mean_score:.4f}, std={std_score:.4f}, top10={top_10_mean:.4f})")
    
    # 保存搜索结果汇总（包含参数范围信息）
    search_summary = {
        'alpha_range': alpha_range,
        'beta_range': beta_range,
        'gamma_range': gamma_range,
        'total_combinations': total_combinations,
        'results': search_results
    }
    search_summary_file = search_dir / "parameter_search_summary.json"
    with open(search_summary_file, 'w') as f:
        json.dump(search_summary, f, indent=2)
    
    # 按 top_20_mean 排序，找出最佳参数
    search_results_sorted = sorted(search_results, key=lambda x: x['statistics']['top_20_mean'], reverse=True)
    
    print(f"\n{'='*80}")
    print(f"Parameter Search Results")
    print(f"{'='*80}")
    print(f"\nTop 10 parameter combinations (by top_20_mean):")
    print(f"{'Rank':<6} {'Alpha':<8} {'Beta':<8} {'Gamma':<8} {'Top20Mean':<12} {'Mean':<10} {'Std':<10}")
    print(f"{'-'*80}")
    for rank, result in enumerate(search_results_sorted[:10], 1):
        stats = result['statistics']
        print(f"{rank:<6} {result['alpha']:<8.2f} {result['beta']:<8.2f} {result['gamma']:<8.2f} "
              f"{stats['top_20_mean']:<12.6f} {stats['mean_score']:<10.6f} {stats['std_score']:<10.6f}")
    
    # 保存最佳参数的结果
    best_result = search_results_sorted[0]
    best_alpha = best_result['alpha']
    best_beta = best_result['beta']
    best_gamma = best_result['gamma']
    
    print(f"\nBest parameters: α={best_alpha:.2f}, β={best_beta:.2f}, γ={best_gamma:.2f}")
    
    # 生成最佳参数的最终文件
    best_aggregated = analyzer.generate_aggregated_scores(
        alpha=best_alpha, beta=best_beta, gamma=best_gamma,
        num_layers=num_layers, num_heads=num_heads
    )
    
    # 使用参数作为文件名后缀
    param_str = f"alpha_{best_alpha:.2f}_beta_{best_beta:.2f}_gamma_{best_gamma:.2f}".replace('.', '_')
    final_output_file = output_dir / "head_score" / f"audio_head_scores_aggregated_{param_str}.json"
    final_output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(final_output_file, 'w') as f:
        json.dump(best_aggregated, f, indent=4)
    
    print(f"Saved best result to: {final_output_file}")
    
    return search_results_sorted


def main():
    parser = argparse.ArgumentParser(description='Generate audio head scores with parameter search')
    parser.add_argument('--model_path', type=str,
                       default="/mnt/users/wangyuxuan-20250915/Models/Qwen2.5-Omni-3B",
                       help='Path to model')
    parser.add_argument('--audio_dir', type=str,
                       default="/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/output_audio",
                       help='Directory containing audio files')
    parser.add_argument('--whisperx_dir', type=str,
                       default="/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/whisperx_parquet_output",
                       help='Directory containing WhisperX JSON files')
    parser.add_argument('--output_dir', type=str,
                       default="/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni",
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to analyze')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index of samples')
    parser.add_argument('--end_idx', type=int, default=-1,
                       help='Ending index of samples')
    parser.add_argument('--score_threshold', type=float, default=0.95,
                       help='WhisperX score threshold')
    parser.add_argument('--topk', type=int, default=32,
                       help='Top-k attention positions')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    # 参数搜索相关
    parser.add_argument('--enable_search', action='store_true',
                       help='Enable parameter search for alpha, beta, gamma')
    parser.add_argument('--alpha_range', type=str, default='0.0,0.5,1.0,1.5,2.0',
                       help='Comma-separated alpha values for search')
    parser.add_argument('--beta_range', type=str, default='0.0,0.5,1.0,1.5,2.0',
                       help='Comma-separated beta values for search')
    parser.add_argument('--gamma_range', type=str, default='0.0,0.5,1.0,1.5,2.0',
                       help='Comma-separated gamma values for search')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Alpha value (used if search disabled)')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Beta value (used if search disabled)')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Gamma value (used if search disabled)')
    parser.add_argument('--num_layers', type=int, default=36,
                       help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=16,
                       help='Number of heads per layer')
    
    args = parser.parse_args()
    
    if args.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # 查找匹配的音频和 JSON 文件
    audio_dir = Path(args.audio_dir)
    whisperx_dir = Path(args.whisperx_dir)
    
    all_audio_files = sorted(audio_dir.glob("*.flac"))
    total_available = len(all_audio_files)
    
    print(f"Total audio files available: {total_available}")
    
    if args.start_idx != 0 or args.end_idx != -1:
        start = args.start_idx
        end = args.end_idx if args.end_idx > 0 else None
        audio_files = all_audio_files[start:end]
        print(f"[Range Mode] Selected samples range: [{start}:{end if end else 'end'}] = {len(audio_files)} files")
    elif args.num_samples > 0:
        audio_files = all_audio_files[:args.num_samples]
        print(f"[Count Mode] Selected first {len(audio_files)} samples")
    else:
        audio_files = all_audio_files
        print(f"[All Mode] Selected all {len(audio_files)} samples")
    
    samples = []
    for audio_file in audio_files:
        sample_id = audio_file.stem
        json_file = whisperx_dir / f"{sample_id}_whisperx.json"
        
        if json_file.exists():
            samples.append({
                'audio_path': str(audio_file),
                'json_path': str(json_file),
                'sample_id': sample_id
            })
        else:
            print(f"Warning: No WhisperX JSON found for {sample_id}")
    
    print(f"Found {len(samples)} samples with matching audio and WhisperX JSON")
    
    if not samples:
        print("No samples found!")
        return
    
    # 加载模型
    print("\nLoading model...")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    
    if args.device == 'cuda':
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            attn_implementation='eager'
        )
    else:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32,
            attn_implementation='eager'
        ).to(device)
    
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    model = model.eval()
    
    # 创建分析器
    analyzer = WordAttentionAnalyzer(
        model, processor, args.output_dir, 
        topk=args.topk
    )
    
    # 注册 hooks
    analyzer.register_attention_hooks()
    
    # 分析样本
    start_time = time.time()
    total_samples = len(samples)
    
    print(f"\nProcessing {total_samples} samples...\n")
    
    for idx, sample in enumerate(samples, 1):
        try:
            sample_id = sample['sample_id']
            print(f"[{idx}/{total_samples}] Processing: {sample_id}", end=" ... ", flush=True)
            
            sample_start = time.time()
            analyzer.analyze_single_sample(
                audio_path=sample['audio_path'],
                whisperx_json_path=sample['json_path'],
                sample_name=sample_id,
                score_threshold=args.score_threshold
            )
            sample_time = time.time() - sample_start
            
            elapsed_total = time.time() - start_time
            avg_time = elapsed_total / idx
            remaining = avg_time * (total_samples - idx)
            percentage = (idx / total_samples) * 100
            
            print(f"Done in {sample_time:.1f}s (avg: {avg_time:.1f}s, remaining: {remaining/60:.1f}min, {percentage:.1f}%)")
        except Exception as e:
            print(f"\nError analyzing {sample['sample_id']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 移除 hooks
    analyzer.remove_hooks()
    
    # 生成输出文件
    output_dir = Path(args.output_dir)
    head_score_dir = output_dir / "head_score"
    head_score_dir.mkdir(parents=True, exist_ok=True)
    
    if args.enable_search:
        # 解析参数范围
        alpha_range = [float(x.strip()) for x in args.alpha_range.split(',')]
        beta_range = [float(x.strip()) for x in args.beta_range.split(',')]
        gamma_range = [float(x.strip()) for x in args.gamma_range.split(',')]
        
        # 执行参数搜索
        search_results = run_parameter_search(
            analyzer,
            alpha_range,
            beta_range,
            gamma_range,
            output_dir,
            num_layers=args.num_layers,
            num_heads=args.num_heads
        )
    else:
        # 使用指定的参数生成单个文件
        aggregated_scores = analyzer.generate_aggregated_scores(
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            num_layers=args.num_layers,
            num_heads=args.num_heads
        )
        
        # 使用参数作为文件名后缀
        param_str = f"alpha_{args.alpha:.2f}_beta_{args.beta:.2f}_gamma_{args.gamma:.2f}".replace('.', '_')
        output_file = head_score_dir / f"audio_head_scores_aggregated_{param_str}.json"
        with open(output_file, 'w') as f:
            json.dump(aggregated_scores, f, indent=4)
        
        print(f"\nSaved aggregated scores to: {output_file}")
        print(f"Parameters used: α={args.alpha}, β={args.beta}, γ={args.gamma}")
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Average time per sample: {total_time/len(samples):.2f}s")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()


'''
python generate_head_scores_with_search.py \
    --num_samples 10 \
    --enable_search \
    --alpha_range "0.0,0.5,1.0,1.5,2.0" \
    --beta_range "0.0,0.5,1.0,1.5,2.0" \
    --gamma_range "0.0,0.5,1.0,1.5,2.0"
    
'''