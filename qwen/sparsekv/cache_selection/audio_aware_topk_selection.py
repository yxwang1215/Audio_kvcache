"""
基于音频时序特性的改进TopK选择算法

基于分析发现：
1. 音频边界（开始/结束）被显著过度选择（1.3-1.4倍偏差）
2. 选择具有很强的局部连续性（小间隔≤5比例100%）
3. 不同Layer关注不同的时序位置（早期层关注后半段，后期层关注前半段）
4. 选择密度高，几乎覆盖所有位置

改进策略：
1. 边界增强：给音频开始和结束部分额外的权重
2. 时序连续性约束：在选择时考虑局部连续性
3. Layer自适应：不同layer使用不同的选择策略
4. 平滑约束：保持时序平滑，避免选择过于分散
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class AudioAwareTopKSelector:
    """
    音频感知的TopK选择器
    
    结合音频的时序特性优化KV cache选择：
    - 边界增强：音频开始和结束部分的重要性
    - 时序连续性：保持选择的局部连续性
    - Layer自适应：不同layer关注不同时序位置
    """
    
    def __init__(
        self,
        audio_token_start: Optional[int] = None,
        audio_token_length: Optional[int] = None,
        boundary_boost: float = 1.3,  # 边界增强系数（基于分析：1.3-1.4倍）
        continuity_weight: float = 0.1,  # 连续性权重
        layer_position_bias: Optional[float] = None,  # Layer位置偏置（None=自动计算）
        smooth_window: int = 5,  # 平滑窗口大小
        min_continuity_length: int = 2,  # 最小连续片段长度
    ):
        self.audio_token_start = audio_token_start
        self.audio_token_length = audio_token_length
        self.boundary_boost = boundary_boost
        self.continuity_weight = continuity_weight
        self.layer_position_bias = layer_position_bias
        self.smooth_window = smooth_window
        self.min_continuity_length = min_continuity_length
        
    def apply_boundary_boost(
        self, 
        attn_scores: torch.Tensor, 
        seq_length: int,
        boundary_ratio: float = 0.1  # 边界比例（前10%和后10%）
    ) -> torch.Tensor:
        """
        对音频边界部分应用增强权重
        
        Args:
            attn_scores: [bsz, num_heads, seq_len] attention scores
            seq_length: 序列总长度
            boundary_ratio: 边界比例（默认0.1表示前10%和后10%）
            
        Returns:
            增强后的attention scores
        """
        if self.audio_token_start is None or self.audio_token_length is None:
            return attn_scores
        
        # 计算音频序列内的相对位置
        device = attn_scores.device
        # 使用实际的 attn_scores 序列长度，而不是传入的 seq_length
        actual_seq_len = attn_scores.shape[-1]
        seq_positions = torch.arange(actual_seq_len, device=device)
        
        # 判断哪些位置在audio token范围内
        in_audio = (seq_positions >= self.audio_token_start) & \
                   (seq_positions < self.audio_token_start + self.audio_token_length)
        
        if not in_audio.any():
            return attn_scores
        
        # 计算audio token内的相对位置
        audio_relative_positions = seq_positions - self.audio_token_start
        audio_relative_positions = torch.clamp(audio_relative_positions, 0, self.audio_token_length - 1)
        
        # 计算边界mask（前10%和后10%）
        boundary_start_threshold = self.audio_token_length * boundary_ratio
        boundary_end_threshold = self.audio_token_length * (1 - boundary_ratio)
        
        is_start_boundary = (audio_relative_positions < boundary_start_threshold) & in_audio
        is_end_boundary = (audio_relative_positions >= boundary_end_threshold) & in_audio
        
        # 创建边界增强mask
        boundary_boost_mask = torch.ones_like(seq_positions, dtype=attn_scores.dtype)
        boundary_boost_mask[is_start_boundary] = self.boundary_boost
        boundary_boost_mask[is_end_boundary] = self.boundary_boost
        
        # 应用边界增强（对audio token内的位置）
        # attn_scores shape: [bsz, num_heads, seq_len]
        boundary_boost_mask = boundary_boost_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
        boosted_scores = attn_scores * boundary_boost_mask
        
        return boosted_scores
    
    def apply_layer_position_bias(
        self,
        attn_scores: torch.Tensor,
        seq_length: int,
        layer_idx: int,
        num_layers: int
    ) -> torch.Tensor:
        """
        根据layer位置应用时序偏置
        
        基于分析：早期layer关注后半段，后期layer关注前半段
        
        Args:
            attn_scores: [bsz, num_heads, seq_len] attention scores
            seq_length: 序列总长度
            layer_idx: 当前layer索引
            num_layers: 总layer数
            
        Returns:
            应用偏置后的attention scores
        """
        if self.audio_token_start is None or self.audio_token_length is None:
            return attn_scores
        
        device = attn_scores.device
        # 使用实际的 attn_scores 序列长度
        actual_seq_len = attn_scores.shape[-1]
        
        # 计算layer的相对位置（0=早期，1=后期）
        layer_ratio = layer_idx / max(num_layers - 1, 1)
        
        # 早期layer（前1/3）：更关注后半段
        # 后期layer（后1/3）：更关注前半段
        if layer_ratio < 0.33:  # 早期layer
            # 对后半段增强
            position_bias = torch.linspace(0.9, 1.1, actual_seq_len, device=device)
        elif layer_ratio > 0.67:  # 后期layer
            # 对前半段增强
            position_bias = torch.linspace(1.1, 0.9, actual_seq_len, device=device)
        else:  # 中期layer
            # 相对均匀
            position_bias = torch.ones(actual_seq_len, device=device)
        
        # 只对audio token内的位置应用偏置
        seq_positions = torch.arange(actual_seq_len, device=device)
        in_audio = (seq_positions >= self.audio_token_start) & \
                   (seq_positions < self.audio_token_start + self.audio_token_length)
        
        # 创建偏置mask
        bias_mask = torch.ones(actual_seq_len, device=device)
        bias_mask[in_audio] = position_bias[in_audio]
        
        # 应用偏置
        bias_mask = bias_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
        biased_scores = attn_scores * bias_mask
        
        return biased_scores
    
    def apply_continuity_smoothing(
        self,
        attn_scores: torch.Tensor,
        seq_length: int
    ) -> torch.Tensor:
        """
        应用时序连续性平滑
        
        基于分析：选择具有很强的局部连续性，倾向于选择连续的音频片段
        
        Args:
            attn_scores: [bsz, num_heads, seq_len] attention scores
            seq_length: 序列总长度
            
        Returns:
            平滑后的attention scores
        """
        if self.audio_token_start is None or self.audio_token_length is None:
            return attn_scores
        
        # 使用滑动窗口平均来增强连续性
        # 这会让相邻位置的选择分数相互影响
        device = attn_scores.device
        # 使用实际的 attn_scores 序列长度
        actual_seq_len = attn_scores.shape[-1]
        
        # 对audio token内的位置应用平滑
        seq_positions = torch.arange(actual_seq_len, device=device)
        in_audio = (seq_positions >= self.audio_token_start) & \
                   (seq_positions < self.audio_token_start + self.audio_token_length)
        
        if not in_audio.any():
            return attn_scores
        
        # 使用1D卷积进行平滑（等价于滑动窗口平均）
        # attn_scores shape: [bsz, num_heads, seq_len]
        bsz, num_heads, seq_len = attn_scores.shape
        
        # 创建平滑kernel
        kernel = torch.ones(1, 1, self.smooth_window, device=device) / self.smooth_window
        kernel = kernel.to(attn_scores.dtype)
        
        # 应用平滑（使用conv1d，padding='same'保持长度）
        smoothed = F.conv1d(
            attn_scores.view(bsz * num_heads, 1, seq_len),
            kernel,
            padding=self.smooth_window // 2
        )
        smoothed = smoothed.view(bsz, num_heads, seq_len)
        
        # 混合原始分数和平滑分数
        # 对audio token内的位置使用平滑，其他位置保持原样
        audio_mask = in_audio.unsqueeze(0).unsqueeze(0).float()  # [1, 1, seq_len]
        final_scores = attn_scores * (1 - self.continuity_weight * audio_mask) + \
                      smoothed * (self.continuity_weight * audio_mask)
        
        return final_scores
    
    def select_with_continuity_constraint(
        self,
        attn_scores: torch.Tensor,
        k: int,
        seq_length: int
    ) -> torch.Tensor:
        """
        带连续性约束的TopK选择
        
        在选择时考虑局部连续性，避免选择过于分散
        
        Args:
            attn_scores: [bsz, num_heads, seq_len] attention scores
            k: 要选择的位置数
            seq_length: 序列总长度
            
        Returns:
            selected_indices: [bsz, num_heads, k] 选择的位置索引
        """
        if self.audio_token_start is None or self.audio_token_length is None:
            # 标准TopK选择
            return attn_scores.topk(k, dim=-1).indices
        
        device = attn_scores.device
        bsz, num_heads, seq_len = attn_scores.shape
        # 使用实际的序列长度
        actual_seq_len = seq_len
        
        # 判断哪些位置在audio token范围内
        seq_positions = torch.arange(actual_seq_len, device=device)
        in_audio = (seq_positions >= self.audio_token_start) & \
                   (seq_positions < self.audio_token_start + self.audio_token_length)
        
        if not in_audio.any():
            return attn_scores.topk(k, dim=-1).indices
        
        # 对audio token内的位置，使用连续性约束的选择
        # 策略：先选择高分位置，然后确保这些位置周围也有选择（连续性）
        
        # 1. 标准TopK选择
        topk_values, topk_indices = attn_scores.topk(k, dim=-1)  # [bsz, num_heads, k]
        
        # 2. 对audio token内的选择，检查连续性
        # 如果选择的间隔太大，尝试填充中间位置
        
        # 简化版本：直接使用TopK，但后续可以通过后处理增强连续性
        # 更复杂的版本可以在选择时动态调整
        
        return topk_indices
    
    def enhance_audio_aware_selection(
        self,
        attn_scores: torch.Tensor,
        seq_length: int,
        layer_idx: int,
        num_layers: int
    ) -> torch.Tensor:
        """
        完整的音频感知增强流程
        
        按顺序应用：
        1. 边界增强
        2. Layer位置偏置
        3. 连续性平滑
        
        Args:
            attn_scores: [bsz, num_heads, seq_len] attention scores
            seq_length: 序列总长度
            layer_idx: 当前layer索引
            num_layers: 总layer数
            
        Returns:
            增强后的attention scores
        """
        # 1. 边界增强
        enhanced_scores = self.apply_boundary_boost(attn_scores, seq_length)
        
        # 2. Layer位置偏置
        enhanced_scores = self.apply_layer_position_bias(
            enhanced_scores, seq_length, layer_idx, num_layers
        )
        
        # 3. 连续性平滑
        enhanced_scores = self.apply_continuity_smoothing(enhanced_scores, seq_length)
        
        return enhanced_scores


def create_audio_aware_selector(
    config,
    layer_idx: int,
    num_layers: int,
    boundary_boost: float = 1.3,
    continuity_weight: float = 0.1,
    smooth_window: int = 5
) -> AudioAwareTopKSelector:
    """
    创建音频感知的TopK选择器
    
    Args:
        config: 模型配置，包含audio_token_start和audio_token_length
        layer_idx: 当前layer索引
        num_layers: 总layer数
        boundary_boost: 边界增强系数
        continuity_weight: 连续性权重
        smooth_window: 平滑窗口大小
        
    Returns:
        AudioAwareTopKSelector实例
    """
    audio_token_start = getattr(config, 'audio_token_start', None)
    audio_token_length = getattr(config, 'audio_token_num', None)
    
    return AudioAwareTopKSelector(
        audio_token_start=audio_token_start,
        audio_token_length=audio_token_length,
        boundary_boost=boundary_boost,
        continuity_weight=continuity_weight,
        layer_position_bias=None,  # 自动计算
        smooth_window=smooth_window,
        min_continuity_length=2
    )


# 使用示例和集成指南
"""
集成到SparseMM的步骤：

1. 在AdaKVCluster或SparseMM的calcul_attn_sore方法中：
   
   def calcul_attn_sore(self, key_states, query_states):
       # ... 原有代码计算attn_weights_mean_pooling ...
       
       # 应用音频感知增强
       if self.config is not None:
           selector = create_audio_aware_selector(
               self.config,
               self.layer_idx,
               self.num_hidden_layers,
               boundary_boost=1.3,  # 可配置
               continuity_weight=0.1  # 可配置
           )
           seq_length = attn_weights_mean_pooling.shape[-1]
           attn_weights_mean_pooling = selector.enhance_audio_aware_selection(
               attn_weights_mean_pooling.unsqueeze(0),  # 添加batch维度
               seq_length,
               self.layer_idx,
               self.num_hidden_layers
           ).squeeze(0)  # 移除batch维度
       
       return attn_weights_mean_pooling

2. 在update_kv方法中，使用增强后的scores进行选择：
   
   attn_score = self.calcul_attn_sore(key_states, query_states)
   # attn_score已经是增强后的了

3. 配置参数（可通过环境变量或config）：
   - AUDIO_BOUNDARY_BOOST: 边界增强系数（默认1.3）
   - AUDIO_CONTINUITY_WEIGHT: 连续性权重（默认0.1）
   - AUDIO_SMOOTH_WINDOW: 平滑窗口大小（默认5）

4. 可选：添加连续性约束的后处理：
   
   def select_with_continuity(self, indices, k, min_gap=5):
       # 确保选择的indices之间间隔不超过min_gap
       # 如果间隔太大，填充中间位置
       ...
"""

