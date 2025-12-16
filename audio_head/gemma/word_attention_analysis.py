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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Gemma3n model and processor
import sys
import os
sys.path.insert(0, '/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft')
from modeling_gemma3n import Gemma3nForConditionalGeneration
from processing_gemma3n import Gemma3nProcessor

# 音频 token ID (Gemma3n uses config.audio_token_id, default is 262273)
# We'll get it from the model config after loading
_AUDIO_TOKEN_ID = None  # Will be set from model config


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
    
    def __init__(self, model, processor, output_dir="./word_attention_analysis", num_layers=None, topk=128, 
                 alpha=1.0, beta=1.0, gamma=1.0, save_detailed_analysis=True):
        self.model = model
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get num_layers from model config if not provided
        if num_layers is None:
            if hasattr(model, 'config') and hasattr(model.config, 'text_config'):
                self.num_layers = model.config.text_config.num_hidden_layers
            elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
                self.num_layers = model.model.language_model.config.num_hidden_layers
            else:
                self.num_layers = 36  # Default fallback
        else:
            self.num_layers = num_layers
        
        self.num_heads = None
        self.head_dim = None  # 每个 head 的维度，由 hook 动态获取
        self.hooks = []
        self.topk = topk  # top-k 注意力位置（参考 SparseMM）
        
        # 权重参数用于计算加权importance score: S = α*S_hit + β*S_attn + γ*S_cov
        self.alpha = alpha  # weight for hit ratio
        self.beta = beta    # weight for attention ratio
        self.gamma = gamma  # weight for coverage ratio
        
        # 是否保存详细的 token 级别分析文件（最大的文件）
        self.save_detailed_analysis = save_detailed_analysis
        
        # 当前样本信息
        self.current_audio_indices = None  # prefill 阶段的音频 token 位置
        self.current_word_alignments = []  # 当前样本的高质量单词对齐
        self.prefill_length = 0  # prefill 阶段的序列长度
        self.current_model_size = "unknown"  # 当前模型大小（E2B/E4B/unknown）
        
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
    
    def count_audio_tokens(self, audio_np):
        """统计音频 tokenize 后的音频 tokens 数量（不分片）"""
        try:
            # 准备输入 - 使用 Gemma3n 的 conversation 格式
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an automatic speech recognizer. Transcribe the English speech only."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_np},
                        {"type": "text", "text": "Please transcribe the audio."}
                    ],
                },
            ]
            
            # 使用 Gemma3n processor
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
            # 准备模型输入
            inputs = self.processor(
                text=[text],
                audio=[audio_np],
                sampling_rate=16000,
                return_tensors="pt",
                padding=False
            )
            
            # 获取 input_ids
            input_ids = inputs.input_ids[0]
            
            # Get audio_token_id from model config
            global _AUDIO_TOKEN_ID
            if _AUDIO_TOKEN_ID is None:
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'audio_token_id'):
                    _AUDIO_TOKEN_ID = self.model.config.audio_token_id
                else:
                    _AUDIO_TOKEN_ID = 262273  # Default Gemma3n audio_token_id
            
            # 统计音频 tokens
            audio_positions = (input_ids == _AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0].tolist()
            
            if not audio_positions:
                # Try to find audio tokens by checking tokenizer
                if hasattr(self.processor, 'audio_token_id'):
                    audio_token_id = self.processor.audio_token_id
                    audio_positions = (input_ids == audio_token_id).nonzero(as_tuple=True)[0].tolist()
            
            num_audio_tokens = len(audio_positions) if audio_positions else 0
            
            return num_audio_tokens, input_ids.shape[0]  # 返回音频 tokens 数量和总序列长度
        except Exception as e:
            print(f"Error counting audio tokens: {e}")
            return 0, 0
    
    def register_attention_hooks(self):
        """注册 hooks 捕获生成过程中的注意力
        
        改进版本：在 self_attn 模块上 hook，手动从 Q/K 计算 attention weights，
        这样可以使用 sdpa/flash_attention_2 正常运行模型。
        
        使用 with_kwargs=True 来接收 kwargs 参数（transformers 新版本用 kwargs 传参）。
        """
        
        def make_attn_hook(layer_idx):
            """为 self_attn 模块创建 hook，手动计算 attention weights"""
            def hook(module, args, kwargs, output):
                # 获取 hidden_states - 优先从 kwargs，否则从 args
                hidden_states = kwargs.get('hidden_states', None)
                if hidden_states is None and len(args) > 0:
                    hidden_states = args[0]
                
                if hidden_states is None:
                    return output
                
                batch_size, seq_len, hidden_dim = hidden_states.shape
                
                # 检测是否在生成阶段（seq_len == 1 表示 decode step）
                is_decode_step = (seq_len == 1)
                
                # 初始化 num_heads 和 head_dim
                if self.num_heads is None:
                    # Gemma3n 的 attention 配置
                    config = module.config
                    self.num_heads = config.num_attention_heads
                    self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
                
                # 标记生成开始
                if is_decode_step and not self.is_generating:
                    self.is_generating = True
                
                # 只在生成阶段（decode step）捕获注意力
                if not is_decode_step or not self.is_generating:
                    return output
                
                # 手动计算 attention weights
                try:
                    # 从 kwargs 获取 past_key_values
                    past_key_value = kwargs.get('past_key_values', None)
                    if past_key_value is None and len(args) > 3:
                        past_key_value = args[3]
                    
                    # 检查是否有 cache
                    if past_key_value is None:
                        return output
                    
                    # 获取 position_embeddings (cos, sin) for RoPE
                    position_embeddings = kwargs.get('position_embeddings', None)
                    if position_embeddings is None and len(args) > 1:
                        position_embeddings = args[1]
                    
                    # 获取 Q 投影并应用归一化和 RoPE
                    config = module.config
                    num_heads = config.num_attention_heads
                    num_kv_heads = config.num_key_value_heads
                    head_dim = self.head_dim
                    
                    hidden_shape = (batch_size, seq_len, num_heads, head_dim)
                    query_states = module.q_proj(hidden_states).view(hidden_shape)
                    query_states = module.q_norm(query_states)
                    
                    # 应用 RoPE (如果有 position_embeddings)
                    if position_embeddings is not None:
                        cos, sin = position_embeddings
                        # 简化的 RoPE 应用（与模型一致的 unsqueeze_dim=2）
                        cos = cos.unsqueeze(2)
                        sin = sin.unsqueeze(2)
                        # rotate_half 和 RoPE 应用
                        x1 = query_states[..., : head_dim // 2]
                        x2 = query_states[..., head_dim // 2 :]
                        query_states = (query_states * cos) + (torch.cat((-x2, x1), dim=-1) * sin)
                    
                    # Reshape: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
                    query_states = query_states.transpose(1, 2)
                    
                    # 从 cache 获取完整的 K
                    full_key = None
                    if hasattr(past_key_value, 'key_cache') and len(past_key_value.key_cache) > layer_idx:
                        # DynamicCache
                        full_key = past_key_value.key_cache[layer_idx]
                    elif hasattr(past_key_value, 'shared_layers') and layer_idx in past_key_value.shared_layers:
                        # Gemma3n shared layers
                        full_key, _ = past_key_value.shared_layers[layer_idx]
                    elif hasattr(past_key_value, '__getitem__'):
                        try:
                            # 尝试 tuple/list 格式
                            if isinstance(past_key_value[layer_idx], tuple):
                                full_key = past_key_value[layer_idx][0]
                            else:
                                full_key = past_key_value[layer_idx]
                        except:
                            pass
                    
                    if full_key is None:
                        return output
                    
                    # full_key: [batch, num_kv_heads, kv_seq_len, head_dim]
                    kv_seq_len = full_key.shape[2]
                    
                    # 处理 GQA (grouped query attention)
                    num_key_value_groups = num_heads // num_kv_heads
                    if num_kv_heads != num_heads:
                        full_key = full_key.unsqueeze(2).expand(-1, -1, num_key_value_groups, -1, -1)
                        full_key = full_key.reshape(batch_size, num_heads, kv_seq_len, head_dim)
                    
                    # 计算 attention scores: Q @ K^T
                    # Gemma3n 使用 scaling=1.0 因为 Q 和 K 已经被 q_norm 和 k_norm 归一化了
                    attn_weights = torch.matmul(query_states, full_key.transpose(2, 3))
                    # attn_weights: [batch, num_heads, 1, kv_seq_len]
                    
                    # Softmax
                    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
                    attn_weights = attn_weights.squeeze(2)  # [batch, num_heads, kv_seq_len]
                    
                    # 保存注意力
                    last_token_attn = attn_weights[0].detach().cpu()  # [num_heads, kv_seq_len]
                    
                    # 开始新的生成步
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
                    
                    del last_token_attn, attn_weights, full_key
                    
                except Exception as e:
                    # 如果计算失败，静默跳过（可以打开下面的注释来调试）
                    # print(f"Hook error at layer {layer_idx}: {e}")
                    pass
                
                return output
            
            return hook
        
        # Gemma3n model structure: model.model.language_model.layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model'):
            base_model = self.model.model.language_model
            if hasattr(base_model, 'layers'):
                layers = base_model.layers
                self.num_layers = len(layers)
                for layer_idx, layer in enumerate(layers):
                    # Hook on self_attn module with with_kwargs=True
                    if hasattr(layer, 'self_attn'):
                        hook = layer.self_attn.register_forward_hook(
                            make_attn_hook(layer_idx), 
                            with_kwargs=True
                        )
                        self.hooks.append(hook)
                print(f"Registered attention hooks for {len(self.hooks)} self_attn modules")
            else:
                print("Warning: Could not find language_model.layers")
        else:
            print("Warning: Could not find model.model.language_model")
    
    def remove_hooks(self):
        """移除所有 hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _normalize_text(self, text: str) -> str:
        """标准化文本用于匹配：去除标点、空格，转小写"""
        import re
        # 去除所有标点符号和空格
        text = re.sub(r'[^\w]', '', text.lower())
        return text
    
    def _diagnose_generation_quality(
        self,
        generated_text: str,
        whisperx_words: List[str],
        token_ids: List[int]
    ) -> Dict:
        """诊断生成文本的质量和与WhisperX单词的匹配情况"""
        import re
        
        diagnosis = {
            'generated_text_length': len(generated_text),
            'generated_text_preview': generated_text[:100],
            'num_tokens': len(token_ids),
            'whisperx_word_count': len(whisperx_words),
            'issues': []
        }
        
        # 检查生成文本是否为空或过短
        if len(generated_text.strip()) == 0:
            diagnosis['issues'].append('empty_generation')
        elif len(generated_text.strip()) < 10:
            diagnosis['issues'].append('very_short_generation')
        
        # 检查是否有重复模式（可能是模型卡住了）
        if len(generated_text) > 20:
            # 检查是否有重复的短语
            words_in_text = generated_text.split()
            if len(words_in_text) > 5:
                # 检查前5个词是否重复出现
                first_5 = ' '.join(words_in_text[:5])
                if first_5 in generated_text[len(first_5):]:
                    diagnosis['issues'].append('repetitive_pattern')
        
        # 检查特殊字符或异常模式
        if re.search(r'[^\w\s\.,!?;:\-\'"]', generated_text):
            diagnosis['issues'].append('unusual_characters')
        
        # 尝试直接匹配：检查WhisperX单词有多少出现在生成文本中
        normalized_generated = self._normalize_text(generated_text)
        matched_words = []
        unmatched_words = []
        
        for word in whisperx_words:
            normalized_word = self._normalize_text(word)
            if normalized_word in normalized_generated:
                matched_words.append(word)
            else:
                unmatched_words.append(word)
        
        diagnosis['direct_matches'] = len(matched_words)
        diagnosis['direct_match_ratio'] = len(matched_words) / len(whisperx_words) if whisperx_words else 0.0
        diagnosis['unmatched_words_preview'] = unmatched_words[:10]
        
        if len(unmatched_words) > len(matched_words):
            diagnosis['issues'].append('low_direct_match_ratio')
        
        return diagnosis
    
    def align_tokens_to_words(
        self,
        token_ids: List[int],
        generated_text: str,
        word_alignments: List[WordAlignment],
        max_skip_threshold: Optional[int] = None,  # 如果为None，将根据模型大小自动设置
        debug: bool = False
    ) -> List[Tuple[int, str, Optional[WordAlignment]]]:
        """将生成的 token 对齐到单词
        
        返回: List[(token_id, token_text, word_alignment or None)]
        
        策略：
        1. 解码每个 token
        2. 累积 token 直到匹配到一个单词
        3. 将该单词的所有 token 都标记为属于该单词
        4. 如果累积token数超过阈值仍未匹配，跳过当前WhisperX单词
        
        改进：
        - 使用更鲁棒的文本标准化（去除所有非字母数字字符）
        - 支持模糊匹配（允许部分匹配）
        - 增加诊断信息
        - 根据模型大小自动调整 max_skip_threshold
        """
        # 根据模型大小自动设置 max_skip_threshold
        if max_skip_threshold is None:
            model_size = getattr(self, 'current_model_size', 'E4B')
            if model_size == "E4B":
                max_skip_threshold = 30  # E4B模型需要更大的阈值
            else:
                max_skip_threshold = 15  # E2B模型使用较小的阈值
            if debug:
                print(f"  [Debug] Using max_skip_threshold={max_skip_threshold} for {model_size} model")
        
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
        match_details = []  # 用于诊断
        
        # 预处理：标准化所有 WhisperX 单词
        normalized_words = []
        for wa in word_alignments:
            normalized_words.append(self._normalize_text(wa.word))
        
        for i, (token_id, token_text) in enumerate(zip(token_ids, token_texts)):
            accumulated_text += token_text
            
            # 检查是否完成了当前单词
            matched_word = None
            if current_word_idx < len(word_alignments):
                word = word_alignments[current_word_idx].word.strip()
                clean_word = normalized_words[current_word_idx]
                
                # 标准化累积文本
                clean_accumulated = self._normalize_text(accumulated_text)
                
                # 检查是否匹配（完全匹配或包含匹配）
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
                        temp_clean = self._normalize_text(temp_accumulated + token_txt)
                        
                        # 找到单词开始的 token
                        if len(temp_clean) <= match_start_pos:
                            word_token_start = j + 1
                        
                        temp_accumulated += token_txt
                        
                        # 找到单词结束的 token
                        temp_clean_after = self._normalize_text(temp_accumulated)
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
                        next_clean_word = normalized_words[current_word_idx + 1]
                        
                        # 如果累积文本包含下一个单词，跳过当前单词
                        if next_clean_word in clean_accumulated:
                            skipped_words.append(word_alignments[current_word_idx].word)
                            if debug:
                                print(f"  [Skip] Word '{word}' not found in generated text, skipping...")
                            current_word_idx += 1
                            # 继续尝试匹配新的当前单词（不重置accumulated_text）
                            continue
                    
                    # 如果也匹配不上下一个，或者已经是最后一个单词了
                    # 检查累积长度是否过长，如果是则强制跳过
                    if len(clean_accumulated) > len(clean_word) * 3:
                        skipped_words.append(word_alignments[current_word_idx].word)
                        if debug:
                            print(f"  [Skip] Word '{word}' timeout (accumulated {tokens_accumulated} tokens), forcing skip...")
                        current_word_idx += 1
                        accumulated_text = ""
                        word_start_token_idx = i + 1
            
            # 如果还没匹配到单词，先添加为 None
            if len(token_to_word) == i:
                token_to_word.append((token_id, token_text, None))
        
        # 输出跳过信息
        if skipped_words and debug:
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
        
        # 3. 准备输入 - 使用 Gemma3n 的 conversation 格式
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an automatic speech recognizer. Transcribe the English speech only."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_np},
                    {"type": "text", "text": "Please transcribe the audio."}
                ],
            },
        ]
        
        # 使用 Gemma3n processor
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # 准备模型输入
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
        
        # 4. 定位音频 tokens (Gemma3n uses config.audio_token_id)
        input_ids = inputs.input_ids[0]
        
        # Get audio_token_id from model config
        global _AUDIO_TOKEN_ID
        if _AUDIO_TOKEN_ID is None:
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'audio_token_id'):
                _AUDIO_TOKEN_ID = self.model.config.audio_token_id
            else:
                _AUDIO_TOKEN_ID = 262273  # Default Gemma3n audio_token_id
        
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
                else:
                    print("Warning: No audio tokens detected")
                    del inputs, audio_np
                    return
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
        
        # 调试信息：检查初始状态
        initial_seq_len = inputs.input_ids.shape[1]
        print(f"Initial sequence length: {initial_seq_len}")
        print(f"Max audio token position: {max(audio_token_indices)}")
        print(f"Number of hooks registered: {len(self.hooks)}")
        
        # 7. 生成并捕获注意力
        # 注意力通过 hook 手动从 Q/K 计算，兼容 sdpa/flash_attention
        print(f"Generating transcription...")
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                input_features=inputs.input_features if hasattr(inputs, 'input_features') else None,
                input_features_mask=inputs.input_features_mask if hasattr(inputs, 'input_features_mask') else None,
                max_new_tokens=128,
                return_dict_in_generate=True,
                use_cache=True,
                do_sample=False,
                temperature=0.0,
                top_p=1.0
            )
        
        # 8. 提取生成的 tokens
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]  # 去掉 prompt 部分
        generated_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
        print(f"Generated {len(generated_ids)} tokens")
        print(f"Captured {len(self.generation_attentions)} generation steps")
        print(f"is_generating flag: {self.is_generating}")
        print(f"prefill_length: {self.prefill_length}")
        if len(self.generation_attentions) > 0:
            print(f"  - First step has {len(self.generation_attentions[0])} layer-head pairs")
            print(f"  - Expected: {self.num_layers} layers × {self.num_heads} heads = {self.num_layers * self.num_heads} pairs")
        else:
            print(f"  - Warning: No attention captured! Check hook registration")
            print(f"  - Debug: is_generating={self.is_generating}, prefill_length={self.prefill_length}, max_audio_pos={max(self.current_audio_indices) if self.current_audio_indices else 'N/A'}")
        
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
        
        # 生成参数后缀用于文件名
        alpha_str = f"{self.alpha:.1f}".rstrip('0').rstrip('.')
        beta_str = f"{self.beta:.1f}".rstrip('0').rstrip('.')
        gamma_str = f"{self.gamma:.1f}".rstrip('0').rstrip('.')
        param_suffix = f"_topk{self.topk}_alpha{alpha_str}_beta{beta_str}_gamma{gamma_str}"
        
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
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model (if not specified, will use default based on --model_type)')
    parser.add_argument('--model_type', type=str, default='auto',
                       choices=['E2B', 'E4B', 'auto'],
                       help='Gemma model type: E2B (30 layers) or E4B (35 layers). If "auto", will detect from model config or path. Default: auto')
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
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Weight for hit ratio in importance score (default: 1.0)')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Weight for attention ratio in importance score (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Weight for coverage ratio in importance score (default: 1.0)')
    parser.add_argument('--skip_detailed_analysis', action='store_true',
                       help='Skip saving detailed analysis file (word_attention_analysis_*.json) to save disk space. You can still process attention_score using layer_head_attention_stats_*.json')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # 根据 model_type 设置默认 model_path（如果未指定）
    if args.model_path is None:
        if args.model_type == 'E2B':
            args.model_path = "/mnt/petrelfs/wenzichen/hf_models/google/gemma-3n-E2B-it"
        elif args.model_type == 'E4B':
            args.model_path = "/mnt/petrelfs/wenzichen/hf_models/google/gemma-3n-E4B-it"
        else:  # auto
            # 尝试从路径推断，或使用 E2B 作为默认值
            args.model_path = "/mnt/petrelfs/wenzichen/hf_models/google/gemma-3n-E2B-it"
    
    # 如果指定了 model_type 但路径中包含另一个类型，给出警告
    if args.model_type != 'auto':
        if args.model_type == 'E2B' and 'E4B' in args.model_path:
            print(f"Warning: --model_type is E2B but model_path contains 'E4B': {args.model_path}")
        elif args.model_type == 'E4B' and 'E2B' in args.model_path:
            print(f"Warning: --model_type is E4B but model_path contains 'E2B': {args.model_path}")
    
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
    processor = Gemma3nProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    if args.device == 'cuda':
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
        ).to(device)
    
    model = model.eval()
    
    # 检测模型类型（E2B 或 E4B）
    detected_model_type = args.model_type
    if args.model_type == 'auto' or args.model_type is None:
        # 从模型配置中检测层数
        num_layers = None
        if hasattr(model, 'config') and hasattr(model.config, 'text_config'):
            num_layers = model.config.text_config.num_hidden_layers
        elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            num_layers = model.model.language_model.config.num_hidden_layers
        
        if num_layers is not None:
            if num_layers == 30:
                detected_model_type = 'E2B'
            elif num_layers == 35:
                detected_model_type = 'E4B'
            else:
                # 尝试从路径推断
                if 'E2B' in args.model_path:
                    detected_model_type = 'E2B'
                elif 'E4B' in args.model_path:
                    detected_model_type = 'E4B'
                else:
                    detected_model_type = 'E2B'  # 默认
                    print(f"Warning: Could not determine model type from num_layers={num_layers}, defaulting to E2B")
        else:
            # 从路径推断
            if 'E2B' in args.model_path:
                detected_model_type = 'E2B'
            elif 'E4B' in args.model_path:
                detected_model_type = 'E4B'
            else:
                detected_model_type = 'E2B'  # 默认
                print(f"Warning: Could not determine model type, defaulting to E2B")
    
    print(f"Detected model type: {detected_model_type}")
    
    # 创建分析器
    analyzer = WordAttentionAnalyzer(
        model, processor, args.output_dir, 
        topk=args.topk,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        save_detailed_analysis=not args.skip_detailed_analysis
    )
    
    # 设置模型大小
    analyzer.current_model_size = detected_model_type
    
    # ========== 统计所有完整音频（不分片）的音频 tokens 总数 ==========
    print(f"\n{'='*80}")
    print(f"Tokenizing all complete audio samples (no chunking) to count total audio tokens...")
    print(f"{'='*80}")
    print(f"Note: Each audio file is processed as a complete unit (no splitting/chunking)")
    print(f"{'='*80}\n")
    
    audio_token_stats = []
    for idx, sample in enumerate(samples, 1):
        try:
            sample_id = sample['sample_id']
            audio_path = sample['audio_path']
            
            # 加载完整音频（不分片）
            audio_np = analyzer.load_audio(audio_path)
            if audio_np is None:
                continue
            
            # 统计完整音频 tokenize 后的音频 tokens 数量（从头到尾）
            num_audio_tokens, total_seq_len = analyzer.count_audio_tokens(audio_np)
            audio_duration = len(audio_np) / 16000
            
            audio_token_stats.append({
                'sample_id': sample_id,
                'num_audio_tokens': num_audio_tokens,
                'total_seq_len': total_seq_len,
                'audio_duration': audio_duration,
                'tokens_per_second': num_audio_tokens / audio_duration if audio_duration > 0 else 0
            })
            
            if audio_duration > 0:
                print(f"[{idx}/{len(samples)}] {sample_id}: {num_audio_tokens} audio tokens "
                      f"(complete audio, duration: {audio_duration:.2f}s, total seq_len: {total_seq_len}, "
                      f"tokens/s: {num_audio_tokens/audio_duration:.2f})")
            else:
                print(f"[{idx}/{len(samples)}] {sample_id}: {num_audio_tokens} audio tokens "
                      f"(complete audio, duration: {audio_duration:.2f}s, total seq_len: {total_seq_len})")
            
        except Exception as e:
            print(f"Error processing {sample.get('sample_id', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 输出统计摘要 - 强调所有音频从头到尾的总 tokens 数
    if audio_token_stats:
        total_audio_tokens = sum(s['num_audio_tokens'] for s in audio_token_stats)
        avg_audio_tokens = total_audio_tokens / len(audio_token_stats)
        min_audio_tokens = min(s['num_audio_tokens'] for s in audio_token_stats)
        max_audio_tokens = max(s['num_audio_tokens'] for s in audio_token_stats)
        total_duration = sum(s['audio_duration'] for s in audio_token_stats)
        avg_tokens_per_second = np.mean([s['tokens_per_second'] for s in audio_token_stats if s['tokens_per_second'] > 0])
        
        print(f"\n{'='*80}")
        print(f"Complete Audio Token Statistics (No Chunking):")
        print(f"{'='*80}")
        print(f"Total samples processed: {len(audio_token_stats)}")
        print(f"\n>>> TOTAL AUDIO TOKENS (all complete audios from start to end): {total_audio_tokens} <<<")
        print(f"    (This is the sum of all audio tokens from all complete audio files)")
        print(f"\nPer-sample statistics:")
        print(f"  - Average audio tokens per sample: {avg_audio_tokens:.2f}")
        print(f"  - Min audio tokens in a sample: {min_audio_tokens}")
        print(f"  - Max audio tokens in a sample: {max_audio_tokens}")
        print(f"\nAudio duration statistics:")
        print(f"  - Total audio duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
        print(f"  - Average tokens per second: {avg_tokens_per_second:.2f}")
        print(f"{'='*80}\n")
    else:
        print("Warning: No audio token statistics collected!")
    
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

# ========== Gemma E2B 模型 ==========
# 1. 使用 E2B 模型（自动检测，默认路径）
CUDA_VISIBLE_DEVICES=5 python word_attention_analysis.py \
    --model_type E2B \
    --num_samples 10

# 2. 使用 E2B 模型（指定路径）
CUDA_VISIBLE_DEVICES=5 python word_attention_analysis.py \
    --model_type E2B \
    --model_path /mnt/public/models/gemma-3n-E2B-it \
    --num_samples 10

# 3. 使用 E2B 模型处理所有样本
CUDA_VISIBLE_DEVICES=7 python word_attention_analysis.py \
    --model_type E2B \
    --num_samples 0 \
    --alpha 1.0 --beta 1.0 --gamma 1.0

# ========== Gemma E4B 模型 ==========
# 4. 使用 E4B 模型（自动检测，默认路径）
CUDA_VISIBLE_DEVICES=5 python word_attention_analysis.py \
    --model_type E4B \
    --num_samples 10

# 5. 使用 E4B 模型（指定路径）
CUDA_VISIBLE_DEVICES=5 python word_attention_analysis.py \
    --model_type E4B \
    --model_path /mnt/public/models/gemma-3n-E4B-it \
    --num_samples 10

# 6. 使用 E4B 模型处理所有样本
CUDA_VISIBLE_DEVICES=7 python word_attention_analysis.py \
    --model_type E4B \
    --num_samples 0 \
    --alpha 1.0 --beta 1.0 --gamma 1.0

# ========== 自动检测模型类型 ==========
# 7. 自动检测模型类型（从路径或配置）
CUDA_VISIBLE_DEVICES=5 python word_attention_analysis.py \
    --model_type auto \
    --model_path /mnt/petrelfs/wenzichen/hf_models/google/gemma-3n-E2B-it \
    --num_samples 10

# ========== 其他参数示例 ==========
# 8. 处理第100-200个样本（索引从0开始，左闭右开）
CUDA_VISIBLE_DEVICES=5 python word_attention_analysis.py \
    --model_type E2B \
    --start_idx 100 --end_idx 200

# 9. 处理从第500个开始的所有样本
python word_attention_analysis.py \
    --model_type E2B \
    --start_idx 500

# 10. 处理指定范围的样本（E4B模型）
CUDA_VISIBLE_DEVICES=5 python word_attention_analysis.py \
    --model_type E4B \
    --start_idx 1 \
    --end_idx 500 \
    --topk 32 \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/audio_head/word_attention_results_E4B_batch1

# 11. 完整参数示例（E2B）
CUDA_VISIBLE_DEVICES=5 python word_attention_analysis.py \
    --model_type E2B \
    --model_path /mnt/petrelfs/wenzichen/hf_models/google/gemma-3n-E2B-it \
    --audio_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/output_audio \
    --whisperx_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/whisperx_parquet_output \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/gemma/word_attention_results_E2B \
    --num_samples 10 \
    --score_threshold 0.95 \
    --topk 32 \
    --alpha 1.0 --beta 1.0 --gamma 1.0

# 12. 完整参数示例（E4B）
CUDA_VISIBLE_DEVICES=5 python word_attention_analysis.py \
    --model_type E4B \
    --model_path /mnt/petrelfs/wenzichen/hf_models/google/gemma-3n-E4B-it \
    --audio_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/output_audio \
    --whisperx_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/whisperx_parquet_output \
    --output_dir /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/gemma/word_attention_results_E4B \
    --num_samples 10 \
    --score_threshold 0.95 \
    --topk 32 \
    --alpha 1.0 --beta 1.0 --gamma 1.0

注意：
- --model_type: 指定模型类型 (E2B/E4B/auto)，默认 auto
  - E2B: 30层模型，默认路径 /mnt/petrelfs/wenzichen/hf_models/google/gemma-3n-E2B-it
  - E4B: 35层模型，默认路径 /mnt/petrelfs/wenzichen/hf_models/google/gemma-3n-E4B-it
  - auto: 自动从模型配置或路径检测
- --model_path: 如果未指定，会根据 --model_type 自动设置默认路径
- 样本按文件名排序
- start_idx 是从0开始的索引（包含）
- end_idx 是结束索引（不包含），类似Python切片
- 如果同时指定start_idx/end_idx和num_samples，优先使用start_idx/end_idx
- E2B 和 E4B 模型使用不同的 max_skip_threshold（E2B: 15, E4B: 30）
'''