"""
Word-Level Attention Analysis with WhisperX Alignment
分析在生成高质量单词的 token 时，注意力是否集中在对应的 prefill 音频 token 位置

流程:
1. 从 WhisperX JSON 获取单词时间标注（筛选 score > 0.95）
2. 将时间戳映射到音频 token 位置
3. 在生成过程中，捕获每个 token 及其注意力权重
4. 统计生成高质量单词时，注意力是否关注对应的音频 token
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

try:
    import soundfile as sf
except ImportError:
    sf = None

warnings.filterwarnings("ignore")

# 设置导入路径：将项目根目录 Audio_kvcache 添加到 sys.path
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
# 从 audio_head/qwen3_omni 回到 Audio_kvcache（项目根目录）
_audio_kvcache_dir = os.path.dirname(os.path.dirname(_current_file_dir))
if _audio_kvcache_dir not in sys.path:
    sys.path.insert(0, _audio_kvcache_dir)

# 将 vanilla 目录添加到 sys.path，以便内部导入能正常工作
_vanilla_dir = os.path.join(_audio_kvcache_dir, 'qwen3', 'vanilla')
if _vanilla_dir not in sys.path:
    sys.path.insert(0, _vanilla_dir)

# 从 vanilla 目录导入
from modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration
)
from processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor
)

from qwen3.qwen_omni_utils import process_mm_info

# 音频 token ID for Qwen3-Omni
_AUDIO_TOKEN_ID = 151675  # <|audio_pad|>
_AUDIO_BOS_TOKEN_ID = 151669  # <|audio_start|>
_AUDIO_EOS_TOKEN_ID = 151670  # <|audio_end|>


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
    
    # 聚合统计
    avg_topk_hits_on_word: float = 0.0  # 平均 top-k 命中对应单词音频 token 的数量
    avg_topk_hits_on_audio: float = 0.0  # 平均 top-k 命中所有音频 token 的数量
    avg_hit_ratio: float = 0.0  # 平均命中率 (word_hits / topk)
    avg_attn_weight_on_word: float = 0.0  # 平均注意力权重（对应单词的音频 token）
    avg_attn_weight_on_audio: float = 0.0  # 平均注意力权重（所有音频 token）
    avg_attn_ratio: float = 0.0  # 平均注意力比例 (word_weight / all_audio_weight)
    word_audio_coverage_ratio: float = 0.0  # 相对于目标单词音频 token 数量的覆盖率 (word_hits / num_word_audio_tokens)
    

class WordAttentionAnalyzer:
    """分析生成单词 token 时的注意力分布 - 基于 top-k 注意力"""
    
    def __init__(self, model, processor, output_dir="./word_attention_analysis", num_layers=36, topk=128,
                 save_detailed_analysis=True, alpha=1.0, beta=1.0, gamma=1.0):
        self.model = model
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_layers = num_layers
        self.num_heads = None
        self.hooks = []
        self.topk = topk  # top-k 注意力位置（参考 SparseMM）
        
        # 权重参数用于计算加权importance score: S = α*S_hit + β*S_attn + γ*S_cov
        # 固定使用默认值，权重信息会保存在输出文件中
        self.alpha = alpha  # weight for hit ratio
        self.beta = beta    # weight for attention ratio
        self.gamma = gamma  # weight for coverage ratio
        
        # 是否保存详细的 token 级别分析文件（最大的文件）
        self.save_detailed_analysis = save_detailed_analysis
        
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
        self.total_words_across_samples = 0  # 所有样本的总单词数（JSON中）
        self.total_high_quality_words = 0  # 所有样本的高质量单词数
        
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
    def load_whisperx_alignment(self, json_path: str, score_threshold: float = 0.95) -> Tuple[List[WordAlignment], int]:
        """加载 WhisperX JSON 并筛选高质量单词
        
        返回: (高质量单词列表, JSON中总单词数)
        """
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
        """将时间戳映射到音频 token 位置
        
        假设音频 token 均匀分布在音频时长上
        """
        if not audio_token_indices:
            return []
        
        num_audio_tokens = len(audio_token_indices)
        audio_start_idx = audio_token_indices[0]
        
        for alignment in word_alignments:
            # 计算在音频中的相对位置
            start_ratio = alignment.start_time / audio_duration
            end_ratio = alignment.end_time / audio_duration
            
            # 映射到 token 索引
            token_start = int(start_ratio * num_audio_tokens)
            token_end = int(end_ratio * num_audio_tokens)
            
            # 确保在范围内
            token_start = max(0, min(token_start, num_audio_tokens - 1))
            token_end = max(token_start, min(token_end, num_audio_tokens - 1))
            
            # 转换为绝对索引
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
                # 调试：只打印第一层的信息
                if layer_idx == 0 and not hasattr(self, '_debug_printed'):
                    print(f"[DEBUG] Hook output type: {type(output)}")
                    if isinstance(output, (tuple, list)):
                        print(f"[DEBUG] Hook output length: {len(output)}")
                        for i, o in enumerate(output):
                            if o is not None:
                                print(f"[DEBUG] output[{i}] type: {type(o)}, shape: {o.shape if hasattr(o, 'shape') else 'N/A'}")
                            else:
                                print(f"[DEBUG] output[{i}] is None")
                    self._debug_printed = True
                
                if not isinstance(output, (tuple, list)) or len(output) <= 1 or output[1] is None:
                    return output
                
                attn_output = output[0]
                attn_weights = output[1]  # [batch, num_heads, seq_len, seq_len]
                
                batch_size, num_heads, seq_len, _ = attn_weights.shape
                
                if self.num_heads is None:
                    self.num_heads = num_heads
                
                # 记录 prefill 长度
                if not self.is_generating and self.current_audio_indices:
                    if seq_len > max(self.current_audio_indices) + 1:
                        self.prefill_length = seq_len - 1
                        self.is_generating = True
                
                # 在生成阶段捕获注意力
                if self.is_generating:
                    # 保存最后一个 token 的注意力
                    last_token_attn = attn_weights[0, :, -1, :].detach().cpu()  # [num_heads, seq_len]
                    
                    # 检查是否需要开始新的生成步
                    # 当前生成步完成的标志：已收集所有 layer 的所有 head
                    if len(self.generation_attentions) == 0:
                        self.generation_attentions.append({})
                    
                    current_step = self.generation_attentions[-1]
                    # 如果当前步骤已经收集了所有 layer*head，开始新步骤
                    if len(current_step) >= self.num_layers * num_heads:
                        self.generation_attentions.append({})
                        current_step = self.generation_attentions[-1]
                    
                    # 保存当前 layer 的注意力
                    for head_idx in range(num_heads):
                        key = (layer_idx, head_idx)
                        current_step[key] = last_token_attn[head_idx]
                    
                    del last_token_attn
                
                # 释放注意力权重
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
                # 注册在 self_attn 上而不是整个 layer 上，因为 layer.forward 只返回 hidden_states
                hook = layer.self_attn.register_forward_hook(make_hook(layer_idx))
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
        max_skip_threshold: int = 10  # 最多累积多少个token仍未匹配就跳过
    ) -> List[Tuple[int, str, Optional[WordAlignment]]]:
        """将生成的 token 对齐到单词
        
        返回: List[(token_id, token_text, word_alignment or None)]
        
        策略：
        1. 解码每个 token
        2. 累积 token 直到匹配到一个单词
        3. 将该单词的所有 token 都标记为属于该单词
        4. 如果累积token数超过阈值仍未匹配，跳过当前WhisperX单词
        """
        # 解码 token 为文本
        token_texts = []
        for token_id in token_ids:
            token_text = self.processor.tokenizer.decode([token_id])
            token_texts.append(token_text)
        
        # 对齐策略：记录每个 token 属于哪个单词
        token_to_word = []
        current_word_idx = 0
        accumulated_text = ""
        word_start_token_idx = 0  # 当前单词的起始 token 索引
        skipped_words = []  # 记录跳过的单词
        
        for i, (token_id, token_text) in enumerate(zip(token_ids, token_texts)):
            accumulated_text += token_text
            
            # 检查是否完成了当前单词
            matched_word = None
            if current_word_idx < len(word_alignments):
                word = word_alignments[current_word_idx].word.strip()
                # 清理accumulated_text进行匹配（去除空格和标点）
                clean_accumulated = accumulated_text.strip().lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
                clean_word = word.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
                
                # 检查是否匹配
                if clean_word in clean_accumulated:
                    matched_word = word_alignments[current_word_idx]
                    
                    # 找到单词在累积文本中的实际位置
                    match_start_pos = clean_accumulated.find(clean_word)
                    match_end_pos = match_start_pos + len(clean_word)
                    
                    # 计算哪些 tokens 真正属于这个单词
                    # 通过重新累积 tokens 来确定边界
                    temp_accumulated = ""
                    word_token_start = word_start_token_idx
                    word_token_end = i
                    
                    for j in range(word_start_token_idx, i + 1):
                        token_txt = token_texts[j]
                        temp_clean = (temp_accumulated + token_txt).strip().lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
                        
                        # 找到单词开始的 token
                        if len(temp_clean) <= match_start_pos:
                            word_token_start = j + 1
                        
                        temp_accumulated += token_txt
                        
                        # 找到单词结束的 token
                        temp_clean_after = temp_accumulated.strip().lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
                        if len(temp_clean_after) >= match_end_pos:
                            word_token_end = j
                            break
                    
                    # 只标记真正属于该单词的 tokens
                    for j in range(word_token_start, word_token_end + 1):
                        if j < len(token_to_word):
                            # 更新已有的 token
                            old_token_id, old_token_text, _ = token_to_word[j]
                            token_to_word[j] = (old_token_id, old_token_text, matched_word)
                        else:
                            # 当前 token
                            if j == i:
                                token_to_word.append((token_id, token_text, matched_word))
                    
                    # 移动到下一个单词
                    current_word_idx += 1
                    accumulated_text = ""
                    word_start_token_idx = i + 1
                    continue
                
                # 跳过机制：如果累积了太多token仍未匹配，尝试跳过当前WhisperX单词
                tokens_accumulated = i - word_start_token_idx + 1
                if tokens_accumulated > max_skip_threshold:
                    # 尝试匹配下一个单词
                    if current_word_idx + 1 < len(word_alignments):
                        next_word = word_alignments[current_word_idx + 1].word.strip()
                        clean_next_word = next_word.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
                        
                        # 如果累积文本包含下一个单词，跳过当前单词
                        if clean_next_word in clean_accumulated:
                            skipped_words.append(word_alignments[current_word_idx].word)
                            print(f"  [Skip] Word '{word}' not found in generated text, skipping...")
                            current_word_idx += 1
                            # 继续尝试匹配新的当前单词（不重置accumulated_text）
                            continue
                    
                    # 如果也匹配不上下一个，或者已经是最后一个单词了
                    # 检查累积长度是否过长，如果是则强制跳过
                    if len(clean_accumulated) > len(clean_word) * 3:
                        skipped_words.append(word_alignments[current_word_idx].word)
                        print(f"  [Skip] Word '{word}' timeout (accumulated {tokens_accumulated} tokens), forcing skip...")
                        current_word_idx += 1
                        accumulated_text = ""
                        word_start_token_idx = i + 1
            
            # 如果还没匹配到单词，先添加为 None
            if len(token_to_word) == i:
                token_to_word.append((token_id, token_text, None))
        
        # 输出跳过信息
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
        
        # 1. 加载 WhisperX 对齐
        word_alignments, total_words_in_json = self.load_whisperx_alignment(whisperx_json_path, score_threshold)
        
        # 累积统计
        self.total_words_across_samples += total_words_in_json
        self.total_high_quality_words += len(word_alignments)
        
        print(f"JSON Statistics:")
        print(f"  - Total words in JSON: {total_words_in_json}")
        print(f"  - High-quality words (score >= {score_threshold}): {len(word_alignments)}")
        print(f"  - Filter ratio: {len(word_alignments)/total_words_in_json*100:.1f}%" if total_words_in_json > 0 else "  - Filter ratio: N/A")
        print(f"\nHigh-quality words preview:")
        for wa in word_alignments[:5]:  # 显示前5个
            print(f"  - {wa.word}: {wa.start_time:.3f}s - {wa.end_time:.3f}s (score: {wa.score:.3f})")
        if len(word_alignments) > 5:
            print(f"  ... and {len(word_alignments) - 5} more")
        
        if not word_alignments:
            print("No high-quality words found, skipping sample")
            return
        
        # 2. 加载音频
        audio_np = self.load_audio(audio_path)
        if audio_np is None:
            return
        
        audio_duration = len(audio_np) / 16000
        print(f"Audio duration: {audio_duration:.2f}s")
        
        # 3. 准备输入 - 使用正确的 conversation 格式
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_np},
                    {"type": "text", "text": "Please transcribe the audio."}
                ],
            },
        ]
        
        # 使用正确的 API 调用 process_mm_info
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        
        # 准备模型输入
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
        
        # 4. 定位音频 tokens
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
        
        print(f"Detected {len(audio_token_indices)} audio tokens at positions {audio_token_indices[0]}-{audio_token_indices[-1]}")
        
        # 5. 映射单词时间到音频 token 位置
        # 5. 映射单词时间到音频 token 位置
        word_alignments = self.map_time_to_audio_tokens(
            word_alignments,
            audio_duration,
            audio_token_indices
        )
        print(f"Mapped words to audio tokens:")
        for wa in word_alignments[:5]:
            print(f"  - {wa.word}: tokens[{wa.audio_token_start}:{wa.audio_token_end+1}]")
        
        # 6. 设置当前样本信息
        self.current_audio_indices = audio_token_indices
        self.current_word_alignments = word_alignments
        self.generation_attentions = []
        self.generated_token_ids = []
        self.is_generating = False  # 重置生成标志
        self.prefill_length = 0
        if hasattr(self, '_debug_printed'):
            delattr(self, '_debug_printed')  # 重置调试标志
        
        # 7. 生成并捕获注意力
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
        
        # 8. 提取生成的 tokens
        # 处理 generate 返回 tuple 的情况 (thinker_result, audio_result)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]  # 去掉 prompt 部分
        generated_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
        print(f"Generated {len(generated_ids)} tokens")
        print(f"Captured {len(self.generation_attentions)} generation steps")
        if len(self.generation_attentions) > 0:
            print(f"  - First step has {len(self.generation_attentions[0])} layer-head pairs")
            print(f"  - Expected: {self.num_layers} layers × {self.num_heads} heads = {self.num_layers * self.num_heads} pairs")
        
        # 9. 对齐 tokens 到单词
        token_to_word = self.align_tokens_to_words(
            generated_ids.cpu().tolist(),
            generated_text,
            word_alignments
        )
        
        # 10. 分析注意力
        print(f"Analyzing attention for {len(token_to_word)} tokens...")
        sample_stats = self._analyze_token_attention(token_to_word, sample_name)
        
        # 11. 清理
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
        """分析每个 token 生成时的注意力 - 基于 top-k 位置而非绝对值"""
        
        sample_stats = {
            'sample_name': sample_name,
            'total_tokens': len(token_to_word),
            'high_quality_word_tokens': 0,
            'tokens': [],
            'word_coverage': {
                'total_words_in_whisperx': len(self.current_word_alignments),
                'words_analyzed': 0,
                'words_missed': [],
                'coverage_ratio': 0.0
            }
        }
        
        # 跟踪哪些单词被分析了
        analyzed_words = set()
        
        for step_idx, (token_id, token_text, word_alignment) in enumerate(token_to_word):
            if step_idx >= len(self.generation_attentions):
                print(f"Warning: No attention captured for step {step_idx}")
                continue
            
            step_attention = self.generation_attentions[step_idx]
            
            # 只分析属于高质量单词的 tokens
            if word_alignment is None:
                continue
            
            sample_stats['high_quality_word_tokens'] += 1
            
            # 记录这个单词被分析了
            analyzed_words.add(word_alignment.word)
            
            # 计算 top-k 注意力统计
            token_stat = TokenAttentionStats(
                token_id=token_id,
                token_text=token_text,
                word=word_alignment.word,
                word_score=word_alignment.score,
                position=step_idx
            )
            
            # 对应单词的音频 token 范围
            word_audio_indices = set(range(
                word_alignment.audio_token_start,
                word_alignment.audio_token_end + 1
            ))
            num_word_audio_tokens = len(word_audio_indices)  # 目标单词的音频 token 数量
            all_audio_indices = set(self.current_audio_indices)
            
            # 统计每个 layer/head 的 top-k 命中和注意力权重
            for (layer_idx, head_idx), attn in step_attention.items():
                # attn shape: [seq_len]
                
                # 找到 top-k 注意力位置
                topk_val = min(self.topk, attn.numel())
                values, indices = attn.topk(topk_val)
                
                # 统计 top-k 中有多少落在对应单词的音频 token 上
                word_hits = sum(1 for idx in indices if idx.item() in word_audio_indices)
                token_stat.topk_hits_on_word_audio[(layer_idx, head_idx)] = word_hits
                
                # 统计 top-k 中有多少落在所有音频 token 上
                audio_hits = sum(1 for idx in indices if idx.item() in all_audio_indices)
                token_stat.topk_hits_on_all_audio[(layer_idx, head_idx)] = audio_hits
                
                # 计算注意力权重和（对应单词的音频 token）
                word_audio_weight = sum(attn[idx].item() for idx in word_audio_indices if idx < len(attn))
                token_stat.attn_weight_on_word_audio[(layer_idx, head_idx)] = word_audio_weight
                
                # 计算注意力权重和（所有音频 token）
                all_audio_weight = sum(attn[idx].item() for idx in all_audio_indices if idx < len(attn))
                token_stat.attn_weight_on_all_audio[(layer_idx, head_idx)] = all_audio_weight
            
            # 计算平均
            if token_stat.topk_hits_on_word_audio:
                token_stat.avg_topk_hits_on_word = np.mean(list(token_stat.topk_hits_on_word_audio.values()))
                token_stat.avg_topk_hits_on_audio = np.mean(list(token_stat.topk_hits_on_all_audio.values()))
                # 命中率：平均每个头的 top-k 中有多少比例关注了对应的单词音频
                token_stat.avg_hit_ratio = token_stat.avg_topk_hits_on_word / self.topk if self.topk > 0 else 0.0
                
                # 计算平均注意力权重
                token_stat.avg_attn_weight_on_word = np.mean(list(token_stat.attn_weight_on_word_audio.values()))
                token_stat.avg_attn_weight_on_audio = np.mean(list(token_stat.attn_weight_on_all_audio.values()))
                # 注意力比例：对应单词的权重 / 所有音频的权重
                token_stat.avg_attn_ratio = (token_stat.avg_attn_weight_on_word / token_stat.avg_attn_weight_on_audio 
                                            if token_stat.avg_attn_weight_on_audio > 0 else 0.0)
                
                # 新增：相对于目标单词音频 token 数量的覆盖率
                # 这个指标更有意义：如果目标单词有 N 个音频 token，top-k 中命中了 M 个，覆盖率 = M/N
                token_stat.word_audio_coverage_ratio = (
                    token_stat.avg_topk_hits_on_word / num_word_audio_tokens 
                    if num_word_audio_tokens > 0 else 0.0
                )
            
            # 保存每个 layer/head 的详细统计
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
                'avg_topk_hits_on_word': token_stat.avg_topk_hits_on_word,
                'avg_topk_hits_on_audio': token_stat.avg_topk_hits_on_audio,
                'hit_ratio': token_stat.avg_hit_ratio,
                'word_audio_coverage_ratio': token_stat.word_audio_coverage_ratio,  # 新增：相对于目标单词的覆盖率
                'avg_attn_weight_on_word': token_stat.avg_attn_weight_on_word,
                'avg_attn_weight_on_audio': token_stat.avg_attn_weight_on_audio,
                'attn_ratio': token_stat.avg_attn_ratio,
                'word_audio_ratio': (token_stat.avg_topk_hits_on_word / token_stat.avg_topk_hits_on_audio 
                                    if token_stat.avg_topk_hits_on_audio > 0 else 0.0),
                'layer_head_stats': layer_head_stats
            })
            
            self.token_stats.append(token_stat)
        
        # 计算单词覆盖率
        all_whisperx_words = [wa.word for wa in self.current_word_alignments]
        missed_words = [w for w in all_whisperx_words if w not in analyzed_words]
        
        sample_stats['word_coverage']['words_analyzed'] = len(analyzed_words)
        sample_stats['word_coverage']['words_missed'] = missed_words
        sample_stats['word_coverage']['coverage_ratio'] = (
            len(analyzed_words) / len(all_whisperx_words) if all_whisperx_words else 0.0
        )
        
        self.sample_results.append(sample_stats)
        
        print(f"  Analyzed {sample_stats['high_quality_word_tokens']} tokens from high-quality words")
        print(f"  Word coverage: {len(analyzed_words)}/{len(all_whisperx_words)} ({sample_stats['word_coverage']['coverage_ratio']*100:.1f}%)")
        if missed_words:
            print(f"  [Warning] Missed words: {missed_words[:5]}" + (f" ... and {len(missed_words)-5} more" if len(missed_words) > 5 else ""))
        
        return sample_stats
    
    def _compute_layer_head_statistics(self, all_tokens: List[Dict]) -> Dict:
        """计算每个 layer/head 的统计信息，包括加权importance score"""
        
        # 收集每个 layer/head 的所有数据
        layer_head_data = defaultdict(lambda: {
            'topk_hits_on_word': [],
            'topk_hits_on_audio': [],
            'attn_weight_on_word': [],
            'attn_weight_on_audio': [],
            'num_word_audio_tokens': []  # 用于计算coverage ratio
        })
        
        for token in all_tokens:
            num_word_audio_tokens = token.get('num_word_audio_tokens', 0)
            for key, stats in token['layer_head_stats'].items():
                layer_head_data[key]['topk_hits_on_word'].append(stats['topk_hits_on_word'])
                layer_head_data[key]['topk_hits_on_audio'].append(stats['topk_hits_on_audio'])
                layer_head_data[key]['attn_weight_on_word'].append(stats['attn_weight_on_word'])
                layer_head_data[key]['attn_weight_on_audio'].append(stats['attn_weight_on_audio'])
                layer_head_data[key]['num_word_audio_tokens'].append(num_word_audio_tokens)
        
        # 计算每个 layer/head 的统计量
        layer_head_summary = {}
        for key, data in layer_head_data.items():
            # 计算注意力比例
            attn_ratios = [
                w / a if a > 0 else 0.0
                for w, a in zip(data['attn_weight_on_word'], data['attn_weight_on_audio'])
            ]
            
            # 计算hit ratio: r_hit = c_word / K
            hit_ratios = [
                hits / self.topk if self.topk > 0 else 0.0
                for hits in data['topk_hits_on_word']
            ]
            
            # 计算coverage ratio: r_cov = c_word / |A(w_i)|
            coverage_ratios = [
                hits / num_tokens if num_tokens > 0 else 0.0
                for hits, num_tokens in zip(data['topk_hits_on_word'], data['num_word_audio_tokens'])
            ]
            
            # 计算三个metric的平均值（对应论文中的 S_hit, S_attn, S_cov）
            S_hit = float(np.mean(hit_ratios))
            S_attn = float(np.mean(attn_ratios))
            S_cov = float(np.mean(coverage_ratios))
            
            # 加权组合: S = α*S_hit + β*S_attn + γ*S_cov
            importance_score = (
                self.alpha * S_hit +
                self.beta * S_attn +
                self.gamma * S_cov
            )
            
            layer_head_summary[key] = {
                'num_tokens': len(data['topk_hits_on_word']),
                'avg_topk_hits_on_word': float(np.mean(data['topk_hits_on_word'])),
                'avg_topk_hits_on_audio': float(np.mean(data['topk_hits_on_audio'])),
                'avg_attn_weight_on_word': float(np.mean(data['attn_weight_on_word'])),
                'avg_attn_weight_on_audio': float(np.mean(data['attn_weight_on_audio'])),
                'avg_attn_ratio': S_attn,
                'median_attn_ratio': float(np.median(attn_ratios)),
                'avg_hit_ratio': S_hit,  # 新增
                'median_hit_ratio': float(np.median(hit_ratios)),  # 新增
                'avg_coverage_ratio': S_cov,  # 新增
                'median_coverage_ratio': float(np.median(coverage_ratios)),  # 新增
                'importance_score': importance_score,  # 加权后的importance score
                'weights': {  # 保存权重信息
                    'alpha': self.alpha,
                    'beta': self.beta,
                    'gamma': self.gamma
                }
            }
        
        return layer_head_summary
    
    def save_results(self):
        """保存分析结果"""
        
        # 生成参数后缀用于文件名（包含topk和权重信息）
        param_suffix = f"_topk{self.topk}_alpha{int(self.alpha) if self.alpha == int(self.alpha) else self.alpha}_beta{int(self.beta) if self.beta == int(self.beta) else self.beta}_gamma{int(self.gamma) if self.gamma == int(self.gamma) else self.gamma}"
        
        # 1. 保存详细的 token 级别统计（可选，文件较大）
        if self.save_detailed_analysis:
            output_file = self.output_dir / f"word_attention_analysis{param_suffix}.json"
            with open(output_file, 'w') as f:
                json.dump(self.sample_results, f, indent=2)
            print(f"Saved detailed results to: {output_file}")
        else:
            print(f"Skipped saving detailed analysis file (word_attention_analysis{param_suffix}.json)")
            print(f"  Note: You can still process attention_score using layer_head_attention_stats{param_suffix}.json")
        
        # 2. 计算汇总统计 - 基于 top-k 命中和注意力权重
        all_tokens = [token for sample in self.sample_results for token in sample['tokens']]
        
        if all_tokens:
            hit_ratios = [t['hit_ratio'] for t in all_tokens]
            word_audio_ratios = [t['word_audio_ratio'] for t in all_tokens]
            word_audio_coverage_ratios = [t['word_audio_coverage_ratio'] for t in all_tokens]  # 新增
            num_word_audio_tokens_list = [t.get('num_word_audio_tokens', 0) for t in all_tokens]  # 新增
            attn_ratios = [t['attn_ratio'] for t in all_tokens]
            attn_weights_on_word = [t['avg_attn_weight_on_word'] for t in all_tokens]
            attn_weights_on_audio = [t['avg_attn_weight_on_audio'] for t in all_tokens]
            
            # 计算全局单词覆盖率
            total_words_in_whisperx = sum(s['word_coverage']['total_words_in_whisperx'] for s in self.sample_results)
            total_words_analyzed = sum(s['word_coverage']['words_analyzed'] for s in self.sample_results)
            global_coverage_ratio = total_words_analyzed / total_words_in_whisperx if total_words_in_whisperx > 0 else 0.0
            
            summary = {
                'topk_value': self.topk,
                'importance_score_weights': {
                    'alpha': self.alpha,
                    'beta': self.beta,
                    'gamma': self.gamma
                },
                'total_samples': len(self.sample_results),
                'total_words_in_json': self.total_words_across_samples,
                'total_high_quality_words': self.total_high_quality_words,
                'high_quality_ratio': self.total_high_quality_words / self.total_words_across_samples if self.total_words_across_samples > 0 else 0.0,
                'word_coverage': {
                    'total_words_in_whisperx': total_words_in_whisperx,
                    'total_words_analyzed': total_words_analyzed,
                    'global_coverage_ratio': global_coverage_ratio,
                    'samples_with_full_coverage': sum(1 for s in self.sample_results if s['word_coverage']['coverage_ratio'] == 1.0),
                    'samples_with_partial_coverage': sum(1 for s in self.sample_results if 0 < s['word_coverage']['coverage_ratio'] < 1.0),
                    'samples_with_no_coverage': sum(1 for s in self.sample_results if s['word_coverage']['coverage_ratio'] == 0.0)
                },
                'total_tokens_analyzed': len(all_tokens),
                'avg_topk_hits_on_word': np.mean([t['avg_topk_hits_on_word'] for t in all_tokens]),
                'avg_topk_hits_on_audio': np.mean([t['avg_topk_hits_on_audio'] for t in all_tokens]),
                'avg_hit_ratio': np.mean(hit_ratios),
                'median_hit_ratio': np.median(hit_ratios),
                'hit_ratio_distribution': {
                    'p25': float(np.percentile(hit_ratios, 25)),
                    'p50': float(np.percentile(hit_ratios, 50)),
                    'p75': float(np.percentile(hit_ratios, 75)),
                    'p95': float(np.percentile(hit_ratios, 95))
                },
                'avg_word_audio_ratio': np.mean(word_audio_ratios),
                'median_word_audio_ratio': np.median(word_audio_ratios),
                'word_audio_ratio_distribution': {
                    'p25': float(np.percentile(word_audio_ratios, 25)),
                    'p50': float(np.percentile(word_audio_ratios, 50)),
                    'p75': float(np.percentile(word_audio_ratios, 75)),
                    'p95': float(np.percentile(word_audio_ratios, 95))
                },
                'avg_num_word_audio_tokens': float(np.mean(num_word_audio_tokens_list)),  # 新增：平均每个单词的音频 token 数量
                'median_num_word_audio_tokens': float(np.median(num_word_audio_tokens_list)),  # 新增
                'avg_word_audio_coverage_ratio': np.mean(word_audio_coverage_ratios),  # 新增：相对于目标单词的覆盖率
                'median_word_audio_coverage_ratio': np.median(word_audio_coverage_ratios),  # 新增
                'word_audio_coverage_ratio_distribution': {  # 新增
                    'p25': float(np.percentile(word_audio_coverage_ratios, 25)),
                    'p50': float(np.percentile(word_audio_coverage_ratios, 50)),
                    'p75': float(np.percentile(word_audio_coverage_ratios, 75)),
                    'p95': float(np.percentile(word_audio_coverage_ratios, 95))
                },
                'avg_attn_weight_on_word': np.mean(attn_weights_on_word),
                'avg_attn_weight_on_audio': np.mean(attn_weights_on_audio),
                'avg_attn_ratio': np.mean(attn_ratios),
                'median_attn_ratio': np.median(attn_ratios),
                'attn_ratio_distribution': {
                    'p25': float(np.percentile(attn_ratios, 25)),
                    'p50': float(np.percentile(attn_ratios, 50)),
                    'p75': float(np.percentile(attn_ratios, 75)),
                    'p95': float(np.percentile(attn_ratios, 95))
                }
            }
            
            summary_file = self.output_dir / f"word_attention_summary{param_suffix}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Saved summary to: {summary_file}")
            
            # 3. 计算并保存每个 layer/head 的统计
            layer_head_summary = self._compute_layer_head_statistics(all_tokens)
            
            # 按importance_score排序
            sorted_heads = sorted(
                layer_head_summary.items(),
                key=lambda x: x[1]['importance_score'],
                reverse=True
            )
            
            # 保存排序后的结果
            layer_head_file = self.output_dir / f"layer_head_attention_stats{param_suffix}.json"
            with open(layer_head_file, 'w') as f:
                json.dump(layer_head_summary, f, indent=2)
            print(f"Saved layer/head statistics to: {layer_head_file}")
            
            # 保存排序后的top heads
            top_heads_file = self.output_dir / f"top_heads_by_importance{param_suffix}.json"
            top_heads_data = {
                'weights': {
                    'alpha': self.alpha,
                    'beta': self.beta,
                    'gamma': self.gamma
                },
                'top_heads': [
                    {
                        'layer_head': key,
                        'importance_score': stats['importance_score'],
                        'avg_hit_ratio': stats['avg_hit_ratio'],
                        'avg_attn_ratio': stats['avg_attn_ratio'],
                        'avg_coverage_ratio': stats['avg_coverage_ratio'],
                        'num_tokens': stats['num_tokens']
                    }
                    for key, stats in sorted_heads
                ]
            }
            with open(top_heads_file, 'w') as f:
                json.dump(top_heads_data, f, indent=2)
            print(f"Saved top heads (sorted by importance) to: {top_heads_file}")
            
            print(f"\n{'='*80}")
            print(f"Summary Statistics (Top-k = {self.topk}):")
            print(f"{'='*80}")
            print(f"Total samples: {summary['total_samples']}")
            print(f"\nWord Statistics from JSON:")
            print(f"  Total words in all JSONs: {summary['total_words_in_json']}")
            print(f"  High-quality words (after filtering): {summary['total_high_quality_words']}")
            print(f"  High-quality ratio: {summary['high_quality_ratio']*100:.1f}%")
            print(f"\nWord Coverage (How many WhisperX words were successfully analyzed):")
            print(f"  Total WhisperX words (high-quality): {summary['word_coverage']['total_words_in_whisperx']}")
            print(f"  Words successfully analyzed: {summary['word_coverage']['total_words_analyzed']}")
            print(f"  Global coverage ratio: {summary['word_coverage']['global_coverage_ratio']*100:.1f}%")
            print(f"  Samples with 100% coverage: {summary['word_coverage']['samples_with_full_coverage']}")
            print(f"  Samples with partial coverage: {summary['word_coverage']['samples_with_partial_coverage']}")
            print(f"  Samples with 0% coverage: {summary['word_coverage']['samples_with_no_coverage']}")
            print(f"\nToken Analysis:")
            print(f"  Total tokens analyzed: {summary['total_tokens_analyzed']}")
            print(f"\nTop-k Attention Hits:")
            print(f"  Avg hits on word audio tokens: {summary['avg_topk_hits_on_word']:.2f} / {self.topk}")
            print(f"  Avg hits on all audio tokens: {summary['avg_topk_hits_on_audio']:.2f} / {self.topk}")
            print(f"  Avg hit ratio (word/topk): {summary['avg_hit_ratio']:.4f} ({summary['avg_hit_ratio']*100:.2f}%)")
            print(f"  Median hit ratio: {summary['median_hit_ratio']:.4f} ({summary['median_hit_ratio']*100:.2f}%)")
            print(f"\nHit Ratio Distribution (对应单词的命中率):")
            print(f"  P25: {summary['hit_ratio_distribution']['p25']:.4f} ({summary['hit_ratio_distribution']['p25']*100:.2f}%)")
            print(f"  P50: {summary['hit_ratio_distribution']['p50']:.4f} ({summary['hit_ratio_distribution']['p50']*100:.2f}%)")
            print(f"  P75: {summary['hit_ratio_distribution']['p75']:.4f} ({summary['hit_ratio_distribution']['p75']*100:.2f}%)")
            print(f"  P95: {summary['hit_ratio_distribution']['p95']:.4f} ({summary['hit_ratio_distribution']['p95']*100:.2f}%)")
            print(f"\nWord/Audio Ratio Distribution (对应单词占所有音频的比例):")
            print(f"  Avg: {summary['avg_word_audio_ratio']:.4f}")
            print(f"  Median: {summary['median_word_audio_ratio']:.4f}")
            print(f"  P75: {summary['word_audio_ratio_distribution']['p75']:.4f}")
            print(f"\nWord Audio Token Statistics (目标单词的音频 token 数量):")
            print(f"  Avg num word audio tokens: {summary['avg_num_word_audio_tokens']:.2f}")
            print(f"  Median num word audio tokens: {summary['median_num_word_audio_tokens']:.2f}")
            print(f"\nWord Audio Coverage Ratio (相对于目标单词的覆盖率 = hits / num_word_tokens):")
            print(f"  Avg coverage ratio: {summary['avg_word_audio_coverage_ratio']:.4f} ({summary['avg_word_audio_coverage_ratio']*100:.2f}%)")
            print(f"  Median coverage ratio: {summary['median_word_audio_coverage_ratio']:.4f} ({summary['median_word_audio_coverage_ratio']*100:.2f}%)")
            print(f"  P25: {summary['word_audio_coverage_ratio_distribution']['p25']:.4f} ({summary['word_audio_coverage_ratio_distribution']['p25']*100:.2f}%)")
            print(f"  P50: {summary['word_audio_coverage_ratio_distribution']['p50']:.4f} ({summary['word_audio_coverage_ratio_distribution']['p50']*100:.2f}%)")
            print(f"  P75: {summary['word_audio_coverage_ratio_distribution']['p75']:.4f} ({summary['word_audio_coverage_ratio_distribution']['p75']*100:.2f}%)")
            print(f"  P95: {summary['word_audio_coverage_ratio_distribution']['p95']:.4f} ({summary['word_audio_coverage_ratio_distribution']['p95']*100:.2f}%)")
            print(f"\nAttention Weight Statistics:")
            print(f"  Avg attention weight on word audio: {summary['avg_attn_weight_on_word']:.4f}")
            print(f"  Avg attention weight on all audio: {summary['avg_attn_weight_on_audio']:.4f}")
            print(f"  Avg attention ratio (word/audio): {summary['avg_attn_ratio']:.4f} ({summary['avg_attn_ratio']*100:.2f}%)")
            print(f"  Median attention ratio: {summary['median_attn_ratio']:.4f} ({summary['median_attn_ratio']*100:.2f}%)")
            print(f"\nAttention Ratio Distribution:")
            print(f"  P25: {summary['attn_ratio_distribution']['p25']:.4f}")
            print(f"  P50: {summary['attn_ratio_distribution']['p50']:.4f}")
            print(f"  P75: {summary['attn_ratio_distribution']['p75']:.4f}")
            print(f"  P95: {summary['attn_ratio_distribution']['p95']:.4f}")
            
            # 显示importance score信息
            print(f"\n{'='*80}")
            print(f"Head Importance Scores (Weighted Combination):")
            print(f"{'='*80}")
            print(f"Weights: α={self.alpha}, β={self.beta}, γ={self.gamma}")
            print(f"Formula: S = α·S_hit + β·S_attn + γ·S_cov")
            print(f"\nTop 20 Audio-Critical Heads (by Importance Score):")
            print(f"{'Rank':<6} {'Layer/Head':<15} {'Importance':<12} {'S_hit':<10} {'S_attn':<10} {'S_cov':<10}")
            print(f"{'-'*80}")
            for rank, (key, stats) in enumerate(sorted_heads[:20], 1):
                print(f"{rank:<6} {key:<15} {stats['importance_score']:<12.6f} "
                      f"{stats['avg_hit_ratio']:<10.4f} {stats['avg_attn_ratio']:<10.4f} "
                      f"{stats['avg_coverage_ratio']:<10.4f}")
            if len(sorted_heads) > 20:
                print(f"... and {len(sorted_heads) - 20} more heads")


def main():
    parser = argparse.ArgumentParser(description='Word-Level Attention Analysis')
    parser.add_argument('--model_path', type=str,
                       default="/mnt/users/wangyuxuan-20250915/Models/Qwen3-Omni-30B-A3B-Instruct",
                       help='Path to model')
    parser.add_argument('--audio_dir', type=str,
                       default="/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/output_audio",
                       help='Directory containing audio files')
    parser.add_argument('--whisperx_dir', type=str,
                       default="/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/whisperx_parquet_output",
                       help='Directory containing WhisperX JSON files')
    parser.add_argument('--output_dir', type=str,
                       default="/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/audio_head/word_attention_results",
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to analyze (set to 0 or -1 for all samples)')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index of samples (0-based, default: 0)')
    parser.add_argument('--end_idx', type=int, default=-1,
                       help='Ending index of samples (exclusive, -1 for no limit, default: -1)')
    parser.add_argument('--score_threshold', type=float, default=0.95,
                       help='WhisperX score threshold for high-quality words')
    parser.add_argument('--topk', type=int, default=32,
                       help='Top-k attention positions to check (default: 32)')
    parser.add_argument('--skip_detailed_analysis', action='store_true',
                       help='Skip saving detailed analysis file (word_attention_analysis_*.json) to save disk space. You can still process attention_score using layer_head_attention_stats_*.json')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Weight for hit ratio in importance score (default: 1.0)')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Weight for attention ratio in importance score (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Weight for coverage ratio in importance score (default: 1.0)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # 查找匹配的音频和 JSON 文件
    audio_dir = Path(args.audio_dir)
    whisperx_dir = Path(args.whisperx_dir)
    
    # 获取所有音频文件（按文件名排序）
    all_audio_files = sorted(audio_dir.glob("*.flac"))
    total_available = len(all_audio_files)
    
    print(f"Total audio files available: {total_available}")
    print(f"Files are sorted by filename")
    
    # 应用样本选择逻辑
    # 优先级: start_idx/end_idx > num_samples
    # 判断用户是否明确指定了范围（start_idx != 默认值0 或 end_idx != 默认值-1）
    if args.start_idx != 0 or args.end_idx != -1:
        # 使用范围选择（优先级最高）
        start = args.start_idx
        end = args.end_idx if args.end_idx > 0 else None
        audio_files = all_audio_files[start:end]
        print(f"[Range Mode] Selected samples range: [{start}:{end if end else 'end'}] = {len(audio_files)} files")
    elif args.num_samples > 0:
        # 使用数量限制（从开头取）
        audio_files = all_audio_files[:args.num_samples]
        print(f"[Count Mode] Selected first {len(audio_files)} samples")
    else:
        # 使用所有样本
        audio_files = all_audio_files
        print(f"[All Mode] Selected all {len(audio_files)} samples")
    
    samples = []
    for audio_file in audio_files:
        # 获取 sample ID (例如: 116-288045-0008)
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
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    
    if args.device == 'cuda':
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            attn_implementation='eager',
            trust_remote_code=True
        )
    else:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32,
            attn_implementation='eager',
            trust_remote_code=True
        ).to(device)
    
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    
    # 启用 attention 输出，这样 hook 才能捕获 attention weights
    model.config.output_attentions = True
    if hasattr(model, 'thinker'):
        if hasattr(model.thinker, 'config'):
            model.thinker.config.output_attentions = True
        if hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
            model.thinker.model.config.output_attentions = True
    
    model = model.eval()
    
    # 创建分析器
    analyzer = WordAttentionAnalyzer(
        model, processor, args.output_dir, 
        topk=args.topk,
        save_detailed_analysis=not args.skip_detailed_analysis,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )
    
    # 注册 hooks
    analyzer.register_attention_hooks()
    
    # 分析样本
    start_time = time.time()
    total_samples = len(samples)
    
    print(f"\nProcessing {total_samples} samples...\n")
    
    for idx, sample in enumerate(samples, 1):
        try:
            # 显示当前进度
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
            
            # 计算统计信息
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
    
    # 保存结果
    analyzer.save_results()
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Average time per sample: {total_time/len(samples):.2f}s")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

'''
使用示例：

# 1. 处理前10个样本
python word_attention_analysis.py --num_samples 10

# 2. 处理所有样本
conda activate Qwen3_2 && \
CUDA_VISIBLE_DEVICES=6 python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen3_omni/word_attention_analysis.py --num_samples 0 --alpha 1.0 --beta 1.0 --gamma 1.0

# 3. 处理第100-200个样本（索引从0开始，左闭右开）
CUDA_VISIBLE_DEVICES=5 python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen3_omni/word_attention_analysis.py --start_idx 100 --end_idx 200 --alpha 1.0 --beta 1.0 --gamma 1.0

# 4. 处理从第500个开始的所有样本
python word_attention_analysis.py --start_idx 500

# 5. 指定输出目录和topk参数
CUDA_VISIBLE_DEVICES=5 python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen3_omni/word_attention_analysis.py \
    --start_idx 1 \
    --end_idx 500 \
    --topk 32 \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen3_omni/results/results_topk32

# 6. 跳过生成详细分析文件（节省磁盘空间，仍可使用 layer_head_attention_stats_*.json 处理 attention_score）
python word_attention_analysis.py \
    --num_samples 20 \
    --topk 64 \
    --skip_detailed_analysis \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen3_omni/results/results_topk64

# 7. 使用不同的topk值进行对比实验
python word_attention_analysis.py \
    --num_samples 0 \
    --topk 16 \
    --skip_detailed_analysis \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen3_omni/results/results_topk16

python word_attention_analysis.py \
    --num_samples 50 \
    --topk 32 \
    --skip_detailed_analysis \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen3_omni/results/results_topk32

python word_attention_analysis.py \
    --num_samples 50 \
    --topk 64 \
    --skip_detailed_analysis \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen3_omni/results/results_topk64

参数说明：
- --topk: Top-k注意力位置数量（默认32，参考SparseMM）
- --skip_detailed_analysis: 跳过生成详细的token级别分析文件（word_attention_analysis_*.json），节省磁盘空间。
  即使跳过该文件，仍可使用 layer_head_attention_stats_*.json 通过 batch_generate_head_scores.py 处理 attention_score。
  注意：如果跳过，generate_head_scores.py 将无法工作（因为它需要每个样本的原始数据）。
- --output_dir: 输出目录，会自动添加参数后缀（如_topk32_alpha1_beta1_gamma1）
- --start_idx: 起始样本索引（从0开始，包含）
- --end_idx: 结束样本索引（不包含，类似Python切片，-1表示无限制）
- --num_samples: 处理的样本数量（0或-1表示处理所有样本）
- --score_threshold: WhisperX单词质量阈值（默认0.95）

注意：
- 样本按文件名排序
- 如果同时指定start_idx/end_idx和num_samples，优先使用start_idx/end_idx
- 输出目录会自动包含参数信息作为后缀，例如：word_attention_results_topk32_alpha1_beta1_gamma1
- importance score计算公式：S = α·S_hit + β·S_attn + γ·S_cov（权重可通过命令行参数指定，权重信息保存在输出文件中）
  - S_hit: 平均hit ratio (c_word / K)
  - S_attn: 平均attention ratio (weight_word / weight_all_audio)
  - S_cov: 平均coverage ratio (c_word / |A(w_i)|)
'''