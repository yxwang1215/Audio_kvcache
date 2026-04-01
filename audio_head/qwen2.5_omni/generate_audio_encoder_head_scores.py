#!/usr/bin/env python3
"""
计算 Audio Encoder Head 的重要性分数

该脚本通过分析 audio encoder 各层的 attention weights 来计算每个 head 的重要性分数。
支持多种重要性指标：
1. Attention Variance: attention weights 的方差（反映 head 的激活程度）
2. Attention Entropy: attention weights 的熵（反映 head 的信息量）
3. Attention Sparsity: attention weights 的稀疏度（反映 head 的选择性）
4. Feature Importance: 基于特征重要性的综合分数

输出格式与现有的 head_score 格式兼容。
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# 添加脚本所在目录到 Python 路径
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


class AudioEncoderHeadAnalyzer:
    """分析 Audio Encoder 各层 attention heads 的重要性"""
    
    def __init__(self, model, processor, device='cuda'):
        self.model = model
        self.processor = processor
        self.device = device
        
        # 获取 audio encoder
        self.audio_encoder = model.get_audio_encoder() if hasattr(model, 'get_audio_encoder') else model.audio_tower
        
        # 检测层数和头数
        self.num_layers = len(self.audio_encoder.layers)
        self.num_heads = self.audio_encoder.layers[0].self_attn.num_heads if len(self.audio_encoder.layers) > 0 else 16
        
        print(f"Audio Encoder: {self.num_layers} layers, {self.num_heads} heads per layer")
        
        # 存储每层的 attention weights
        self.attention_weights = defaultdict(list)  # {(layer_idx, head_idx): [weights_list]}
        self.hooks = []
        
    def register_hooks(self):
        """注册 hooks 来捕获每层的 attention weights"""
        
        def make_attn_hook(layer_idx):
            """为指定层创建 attention hook"""
            def hook(module, args, kwargs, output):
                # 尝试从 output 获取 attention weights
                # 对于 audio encoder，需要手动计算
                try:
                    # 获取 hidden_states
                    # audio encoder layer 的 forward 签名: forward(hidden_states, cu_seqlens, return_logits=False)
                    hidden_states = None
                    cu_seqlens = None
                    
                    # 尝试从 args 获取（位置参数）
                    if len(args) > 0:
                        hidden_states = args[0]
                    if len(args) > 1:
                        cu_seqlens = args[1]
                    
                    # 尝试从 kwargs 获取（关键字参数）
                    if hidden_states is None:
                        hidden_states = kwargs.get('hidden_states', None)
                    if cu_seqlens is None:
                        cu_seqlens = kwargs.get('cu_seqlens', None)
                    
                    if hidden_states is None or cu_seqlens is None:
                        return output
                    
                    # 确保 hidden_states 是 2D tensor [seq_len, embed_dim]
                    if hidden_states.dim() == 3:
                        # [batch, seq_len, embed_dim] -> [batch*seq_len, embed_dim]
                        batch_size, seq_len, embed_dim = hidden_states.shape
                        hidden_states = hidden_states.reshape(-1, embed_dim)
                    elif hidden_states.dim() != 2:
                        return output
                    
                    seq_length, embed_dim = hidden_states.shape
                    
                    # 手动计算 attention weights
                    with torch.no_grad():
                        # 获取 Q, K
                        query_states = module.q_proj(hidden_states)
                        key_states = module.k_proj(hidden_states)
                        
                        num_heads = module.num_heads
                        head_dim = embed_dim // num_heads
                        
                        # Reshape: [seq_len, num_heads, head_dim]
                        query_states = query_states.view(seq_length, num_heads, head_dim)
                        key_states = key_states.view(seq_length, num_heads, head_dim)
                        
                        # Transpose for attention: [num_heads, seq_len, head_dim]
                        query_states = query_states.transpose(0, 1)
                        key_states = key_states.transpose(0, 1)
                        
                        # 计算 attention scores: [num_heads, seq_len, seq_len]
                        attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / (head_dim ** 0.5)
                        
                        # 应用 attention mask（基于 cu_seqlens）
                        # cu_seqlens 是累积序列长度，例如 [0, 100, 200] 表示两个序列，长度分别为 100 和 100
                        if cu_seqlens is not None and len(cu_seqlens) > 1:
                            attention_mask = torch.full(
                                [seq_length, seq_length],
                                torch.finfo(attn_scores.dtype).min,
                                device=attn_scores.device,
                                dtype=attn_scores.dtype,
                            )
                            # 为每个序列设置有效的 attention 区域
                            for i in range(1, len(cu_seqlens)):
                                start = int(cu_seqlens[i-1].item() if torch.is_tensor(cu_seqlens[i-1]) else cu_seqlens[i-1])
                                end = int(cu_seqlens[i].item() if torch.is_tensor(cu_seqlens[i]) else cu_seqlens[i])
                                if start < seq_length and end <= seq_length:
                                    attention_mask[start:end, start:end] = 0
                            
                            # 应用 mask: [num_heads, seq_len, seq_len]
                            attn_scores = attn_scores + attention_mask.unsqueeze(0)
                        
                        # Softmax: [num_heads, seq_len, seq_len]
                        attn_weights = F.softmax(attn_scores, dim=-1)
                        
                        # 存储每个 head 的 attention weights
                        for head_idx in range(num_heads):
                            head_weights = attn_weights[head_idx].detach().cpu()  # [seq_len, seq_len]
                            self.attention_weights[(layer_idx, head_idx)].append(head_weights)
                
                except Exception as e:
                    # 如果计算失败，跳过（不打印错误以避免干扰）
                    pass
                
                return output
            
            return hook
        
        # 为每一层注册 hook
        for layer_idx, layer in enumerate(self.audio_encoder.layers):
            hook = layer.self_attn.register_forward_hook(make_attn_hook(layer_idx))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """移除所有 hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_head_importance_scores(self) -> Dict[str, List[float]]:
        """
        计算每个 head 的重要性分数
        
        返回格式: {"layer-head": [importance_score, variance_score, entropy_score, sparsity_score]}
        """
        head_scores = {}
        
        print(f"\nComputing head importance scores from {sum(len(v) for v in self.attention_weights.values())} attention samples...")
        
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                key = f"{layer_idx}-{head_idx}"
                hook_key = (layer_idx, head_idx)
                
                if hook_key not in self.attention_weights or len(self.attention_weights[hook_key]) == 0:
                    # 如果没有数据，使用零分数
                    head_scores[key] = [0.0, 0.0, 0.0, 0.0]
                    continue
                
                # 收集该 head 的所有 attention weights
                all_weights = self.attention_weights[hook_key]
                
                # 计算各种重要性指标
                variance_scores = []
                entropy_scores = []
                sparsity_scores = []
                
                for weights in all_weights:
                    # weights: [seq_len, seq_len]
                    # 计算每行的统计量（每个 token 作为 query 时的 attention 分布）
                    
                    # 1. Variance: attention weights 的方差（反映激活程度）
                    # 对每行计算方差，然后取平均
                    row_variances = weights.var(dim=-1)  # [seq_len]
                    variance_score = row_variances.mean().item()
                    variance_scores.append(variance_score)
                    
                    # 2. Entropy: attention weights 的熵（反映信息量）
                    # 对每行计算熵，然后取平均
                    # 添加小的 epsilon 避免 log(0)
                    eps = 1e-8
                    weights_safe = weights + eps
                    row_entropies = -(weights_safe * torch.log(weights_safe)).sum(dim=-1)  # [seq_len]
                    entropy_score = row_entropies.mean().item()
                    entropy_scores.append(entropy_score)
                    
                    # 3. Sparsity: attention weights 的稀疏度（反映选择性）
                    # 使用 L1 范数来衡量稀疏度（越小越稀疏）
                    # 或者使用 top-k 集中度（top-k weights 的占比）
                    k = max(1, weights.shape[-1] // 10)  # top 10%
                    topk_weights, _ = torch.topk(weights, k, dim=-1)
                    sparsity_score = (topk_weights.sum(dim=-1) / (weights.sum(dim=-1) + eps)).mean().item()
                    sparsity_scores.append(sparsity_score)
                
                # 计算平均分数
                avg_variance = np.mean(variance_scores) if variance_scores else 0.0
                avg_entropy = np.mean(entropy_scores) if entropy_scores else 0.0
                avg_sparsity = np.mean(sparsity_scores) if sparsity_scores else 0.0
                
                # 归一化分数到 [0, 1] 范围（使用 min-max normalization）
                # 注意：这里先计算原始分数，后续可以在所有 heads 上统一归一化
                
                # 综合重要性分数：加权组合
                # importance = w1 * variance + w2 * entropy + w3 * sparsity
                # 使用简单的平均（可以后续调整权重）
                importance_score = (avg_variance + avg_entropy + avg_sparsity) / 3.0
                
                head_scores[key] = [
                    float(importance_score),
                    float(avg_variance),
                    float(avg_entropy),
                    float(avg_sparsity)
                ]
        
        # 全局归一化（可选）：将所有分数归一化到 [0, 1]
        # 提取所有分数
        all_importance = [v[0] for v in head_scores.values()]
        all_variance = [v[1] for v in head_scores.values()]
        all_entropy = [v[2] for v in head_scores.values()]
        all_sparsity = [v[3] for v in head_scores.values()]
        
        # 计算 min-max
        if all_importance:
            imp_min, imp_max = min(all_importance), max(all_importance)
            var_min, var_max = min(all_variance), max(all_variance)
            ent_min, ent_max = min(all_entropy), max(all_entropy)
            spa_min, spa_max = min(all_sparsity), max(all_sparsity)
            
            # 归一化（避免除零）
            imp_range = imp_max - imp_min if imp_max > imp_min else 1.0
            var_range = var_max - var_min if var_max > var_min else 1.0
            ent_range = ent_max - ent_min if ent_max > ent_min else 1.0
            spa_range = spa_max - spa_min if spa_max > spa_min else 1.0
            
            # 重新归一化
            for key in head_scores:
                scores = head_scores[key]
                head_scores[key] = [
                    (scores[0] - imp_min) / imp_range,
                    (scores[1] - var_min) / var_range,
                    (scores[2] - ent_min) / ent_range,
                    (scores[3] - spa_min) / spa_range,
                ]
        
        return head_scores
    
    def analyze_audio(self, audio_path: str):
        """分析单个音频文件"""
        print(f"  Analyzing: {Path(audio_path).name}")
        
        # 加载音频
        try:
            # 使用 processor 处理音频
            audio_inputs = self.processor(
                audios=audio_path,
                return_tensors="pt",
                padding=True
            )
            
            # 移动到设备
            audio_inputs = {k: v.to(self.device) if torch.is_tensor(v) else v 
                           for k, v in audio_inputs.items()}
            
        except Exception as e:
            print(f"    ⚠ Error loading audio: {e}")
            return
        
        # 前向传播以触发 hooks
        try:
            with torch.no_grad():
                # 获取 audio features
                # 注意：这里需要调用 audio encoder 的 forward 方法来触发 hooks
                if hasattr(self.model, 'get_audio_features'):
                    # 使用模型的 get_audio_features 方法
                    audio_features, _, _ = self.model.get_audio_features(
                        input_features=audio_inputs['input_features'],
                        feature_attention_mask=audio_inputs.get('feature_attention_mask', None),
                        audio_feature_lengths=audio_inputs.get('feature_lens', None)
                    )
                else:
                    # 直接调用 audio_encoder
                    # 需要准备正确的输入格式
                    input_features = audio_inputs['input_features']
                    feature_lens = audio_inputs.get('feature_lens', None)
                    
                    # 计算 aftercnn_lens
                    if feature_lens is not None:
                        aftercnn_lens = self.audio_encoder._get_feat_extract_output_lengths(feature_lens)[1]
                    else:
                        aftercnn_lens = None
                    
                    audio_outputs = self.audio_encoder(
                        input_features=input_features,
                        feature_lens=feature_lens,
                        aftercnn_lens=aftercnn_lens
                    )
        except Exception as e:
            print(f"    ⚠ Error processing audio: {e}")
            import traceback
            traceback.print_exc()
            return
    
    def analyze_audio_batch(self, audio_paths: List[str]):
        """批量分析多个音频文件"""
        print(f"\n{'='*60}")
        print(f"Analyzing {len(audio_paths)} audio files")
        print(f"{'='*60}")
        
        # 注册 hooks
        self.register_hooks()
        
        try:
            # 分析每个音频
            for i, audio_path in enumerate(audio_paths):
                print(f"\n[{i+1}/{len(audio_paths)}] Processing: {Path(audio_path).name}")
                self.analyze_audio(audio_path)
        finally:
            # 移除 hooks
            self.remove_hooks()
        
        # 计算重要性分数
        head_scores = self.compute_head_importance_scores()
        
        return head_scores


def main():
    parser = argparse.ArgumentParser(
        description='Generate audio encoder head importance scores',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to Qwen2.5-Omni model'
    )
    parser.add_argument(
        '--audio_dir',
        type=str,
        default="/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/output_audio",
        help='Directory containing audio files to analyze'
    )
    parser.add_argument(
        '--whisperx_dir',
        type=str,
        default=None,
        help='Directory containing WhisperX JSON files (optional, for consistency with word_attention_analysis.py)'
    )
    parser.add_argument(
        '--audio_files',
        type=str,
        nargs='+',
        default=None,
        help='List of audio file paths to analyze'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Output JSON file path for head scores'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help='Start index for processing samples (inclusive)'
    )
    parser.add_argument(
        '--end_idx',
        type=int,
        default=-1,
        help='End index for processing samples (exclusive, -1 means no limit)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=0,
        help='Number of samples to process (0 or -1 means all samples, ignored if start_idx/end_idx specified)'
    )
    parser.add_argument(
        '--max_files',
        type=int,
        default=None,
        help='Maximum number of audio files to process (for quick testing, deprecated, use start_idx/end_idx or num_samples instead)'
    )
    
    args = parser.parse_args()
    
    # 确定要分析的音频文件
    audio_files = []
    
    if args.audio_files:
        # 方式1: 直接指定音频文件列表
        audio_files = args.audio_files
    elif args.audio_dir:
        # 方式2: 从目录读取（类似 word_attention_analysis.py）
        audio_dir = Path(args.audio_dir)
        if not audio_dir.exists():
            print(f"Error: Audio directory not found: {audio_dir}")
            return
        
        # 获取所有音频文件（优先支持 .flac，与 word_attention_analysis.py 一致）
        # 但也支持其他常见格式
        found_files = []
        # 首先查找 .flac 文件（主要格式）
        found_files.extend(list(audio_dir.glob("*.flac")))
        # 如果没找到 .flac，尝试其他格式
        if not found_files:
            audio_extensions = ['.wav', '.mp3', '.m4a', '.ogg']
            for ext in audio_extensions:
                found_files.extend(list(audio_dir.glob(f'*{ext}')))
                found_files.extend(list(audio_dir.glob(f'**/*{ext}')))
        
        # 去重并排序（按文件名，与 word_attention_analysis.py 一致）
        found_files = sorted(set(found_files), key=lambda x: x.name)
        all_audio_files = [str(f) for f in found_files]
        
        total_available = len(all_audio_files)
        print(f"Total audio files available: {total_available}")
        print(f"Files are sorted by filename")
        
        # 应用样本选择逻辑（与 word_attention_analysis.py 一致）
        # 优先级: start_idx/end_idx > num_samples > max_files
        if args.start_idx is not None and args.end_idx != -1:
            # 使用范围选择（优先级最高）
            start = args.start_idx
            end = args.end_idx if args.end_idx > 0 else None
            audio_files = all_audio_files[start:end]
            print(f"[Range Mode] Selected samples range: [{start}:{end if end else 'end'}] = {len(audio_files)} files")
        elif args.num_samples > 0:
            # 使用数量限制（从开头取）
            audio_files = all_audio_files[:args.num_samples]
            print(f"[Count Mode] Selected first {len(audio_files)} samples")
        elif args.max_files:
            # 兼容旧的 max_files 参数
            audio_files = all_audio_files[:args.max_files]
            print(f"[Max Files Mode] Selected first {len(audio_files)} samples")
        else:
            # 使用所有样本
            audio_files = all_audio_files
            print(f"[All Mode] Selected all {len(audio_files)} samples")
        
        # 如果指定了 whisperx_dir，可以验证 JSON 文件是否存在（可选）
        if args.whisperx_dir:
            whisperx_dir = Path(args.whisperx_dir)
            if whisperx_dir.exists():
                print(f"WhisperX directory specified: {whisperx_dir} (for reference)")
                # 可以在这里添加验证逻辑，但 audio encoder head scores 不需要 WhisperX JSON
    else:
        print("Error: Must provide either --audio_dir or --audio_files")
        return
    
    if not audio_files:
        print("Error: No audio files found")
        return
    
    print(f"\n{'='*60}")
    print(f"Audio Encoder Head Importance Score Generation")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    if args.audio_dir:
        print(f"Audio directory: {args.audio_dir}")
    if args.whisperx_dir:
        print(f"WhisperX directory: {args.whisperx_dir} (for reference)")
    print(f"Audio files: {len(audio_files)}")
    if args.start_idx is not None and args.end_idx != -1:
        print(f"Range: [{args.start_idx}:{args.end_idx}]")
    elif args.num_samples > 0:
        print(f"Count: first {args.num_samples} files")
    print(f"Output: {args.output_file}")
    print(f"Device: {args.device}")
    print(f"{'='*60}")
    
    # 加载模型
    print(f"\nLoading model...")
    try:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map=args.device
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 创建分析器
    analyzer = AudioEncoderHeadAnalyzer(model, processor, device=args.device)
    
    # 分析音频
    head_scores = analyzer.analyze_audio_batch(audio_files)
    
    # 保存结果
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(head_scores, f, indent=4)
    
    print(f"✓ Saved successfully!")
    print(f"\nStatistics:")
    print(f"  Total heads: {len(head_scores)}")
    if head_scores:
        non_zero_heads = sum(1 for v in head_scores.values() if any(x > 0 for x in v))
        print(f"  Non-zero heads: {non_zero_heads}")
        print(f"  Zero heads: {len(head_scores) - non_zero_heads}")
        
        # 计算统计信息
        all_importance = [v[0] for v in head_scores.values()]
        if all_importance:
            print(f"  Mean importance score: {np.mean(all_importance):.6f}")
            print(f"  Max importance score: {np.max(all_importance):.6f}")
            print(f"  Min importance score: {np.min(all_importance):.6f}")
    
    print(f"\n{'='*60}")
    print("Head score generation completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


'''
# 使用示例（参考 word_attention_analysis.py 的方式）

# 方式1: 使用默认路径（与 word_attention_analysis.py 一致）
python generate_audio_encoder_head_scores.py \
    --model_path /mnt/public/models/Qwen2.5-Omni-3B \
    --audio_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/output_audio \
    --whisperx_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/whisperx_parquet_output \
    --output_file ./head_score/audio_encoder_head_scores.json \
    --device cuda \
    --start_idx 0 \
    --end_idx 10

# 方式2: 使用数量限制
python generate_audio_encoder_head_scores.py \
    --model_path /mnt/public/models/Qwen2.5-Omni-3B \
    --audio_dir /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/librispeech/output_audio \
    --output_file ./head_score/audio_encoder_head_scores.json \
    --device cuda \
    --num_samples 10

# 方式3: 指定具体的音频文件列表
python generate_audio_encoder_head_scores.py \
    --model_path /mnt/public/models/Qwen2.5-Omni-3B \
    --audio_files /path/to/audio1.flac /path/to/audio2.flac \
    --output_file ./head_score/audio_encoder_head_scores.json \
    --device cuda

# 方式4: 使用脚本
bash scripts/run_audio_encoder_head_scores.sh
'''