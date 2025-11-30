"""
集成示例：如何在SparseMM中使用音频感知TopK选择

这个文件展示了如何修改sparsemm_utils_timing.py来集成音频感知选择算法
"""

# ============================================================================
# 步骤1: 在sparsemm_utils_timing.py顶部添加导入
# ============================================================================

"""
import os
from audio_aware_topk_selection import create_audio_aware_selector
"""

# ============================================================================
# 步骤2: 修改AdaKVCluster.calcul_attn_sore方法
# ============================================================================

"""
原始代码（sparsemm_utils_timing.py line 433-467）:

    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        # ... mask和softmax ...
        attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)
        
        attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0],num_heads//self.num_key_value_groups,self.num_key_value_groups,-1)
        if self.gqa_func == 'max':
            attn_weights_mean = attn_weights_mean.max(dim=-2).values
        elif self.gqa_func == 'mean':
            attn_weights_mean = attn_weights_mean.mean(dim=-2)
        
        if self.pooling == 'avgpool':
            attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        elif self.pooling == 'maxpool':
            attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        return attn_weights_mean_pooling

修改后的代码:

    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        # ... mask和softmax ...
        attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)
        
        attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0],num_heads//self.num_key_value_groups,self.num_key_value_groups,-1)
        if self.gqa_func == 'max':
            attn_weights_mean = attn_weights_mean.max(dim=-2).values
        elif self.gqa_func == 'mean':
            attn_weights_mean = attn_weights_mean.mean(dim=-2)
        
        if self.pooling == 'avgpool':
            attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        elif self.pooling == 'maxpool':
            attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        
        # ===== 新增：应用音频感知增强 =====
        if hasattr(self, 'config') and self.config is not None:
            # 检查是否启用音频感知选择（可通过环境变量控制）
            use_audio_aware = os.environ.get("USE_AUDIO_AWARE_TOPK", "false").lower() == "true"
            
            if use_audio_aware:
                try:
                    from audio_aware_topk_selection import create_audio_aware_selector
                    
                    # 获取配置参数
                    boundary_boost = float(os.environ.get("AUDIO_BOUNDARY_BOOST", "1.3"))
                    continuity_weight = float(os.environ.get("AUDIO_CONTINUITY_WEIGHT", "0.1"))
                    
                    # 创建选择器
                    selector = create_audio_aware_selector(
                        self.config,
                        self.layer_idx if hasattr(self, 'layer_idx') else 0,
                        self.num_hidden_layers if hasattr(self, 'num_hidden_layers') else 36,
                        boundary_boost=boundary_boost,
                        continuity_weight=continuity_weight
                    )
                    
                    # 计算序列长度（需要考虑window_size）
                    # attn_weights_mean_pooling shape: [bsz, num_kv_heads, seq_len]
                    seq_length = attn_weights_mean_pooling.shape[-1] + self.window_size
                    
                    # 应用增强（需要添加batch维度）
                    if attn_weights_mean_pooling.dim() == 2:
                        # [num_kv_heads, seq_len] -> [1, num_kv_heads, seq_len]
                        attn_weights_mean_pooling = attn_weights_mean_pooling.unsqueeze(0)
                    
                    # 应用音频感知增强
                    attn_weights_mean_pooling = selector.enhance_audio_aware_selection(
                        attn_weights_mean_pooling,
                        seq_length,
                        self.layer_idx if hasattr(self, 'layer_idx') else 0,
                        self.num_hidden_layers if hasattr(self, 'num_hidden_layers') else 36
                    )
                    
                    # 移除batch维度（如果之前添加了）
                    if attn_weights_mean_pooling.dim() == 3 and attn_weights_mean_pooling.shape[0] == 1:
                        attn_weights_mean_pooling = attn_weights_mean_pooling.squeeze(0)
                        
                except Exception as e:
                    # 如果导入或使用失败，回退到原始方法
                    import warnings
                    warnings.warn(f"Audio-aware TopK selection failed: {e}, using original method")
        
        return attn_weights_mean_pooling
"""

# ============================================================================
# 步骤3: 修改SparseMM类，确保config被正确传递
# ============================================================================

"""
在SparseMM.__init__中，确保config被保存：

    def __init__(self, ..., config=None):
        # ... 原有代码 ...
        self.config = config  # 确保config被保存
"""

# ============================================================================
# 步骤4: 在LibriSpeech_qwen2.5_timing.py中添加环境变量支持
# ============================================================================

"""
在文件顶部添加：

ENV_AUDIO_BOUNDARY_BOOST = float(os.environ.get("AUDIO_BOUNDARY_BOOST", "1.3"))
ENV_AUDIO_CONTINUITY_WEIGHT = float(os.environ.get("AUDIO_CONTINUITY_WEIGHT", "0.1"))
ENV_AUDIO_SMOOTH_WINDOW = int(os.environ.get("AUDIO_SMOOTH_WINDOW", "5"))
USE_AUDIO_AWARE_TOPK = os.environ.get("USE_AUDIO_AWARE_TOPK", "false").lower() == "true"
"""

# ============================================================================
# 步骤5: 确保config包含audio token信息
# ============================================================================

"""
在LibriSpeech_qwen2.5_timing.py中，确保在调用model.enable_sparsemm_attention时，
config已经包含了audio_token_start和audio_token_length信息。

或者在SparseMM初始化时，从config中读取这些信息。
"""

# ============================================================================
# 完整的使用示例
# ============================================================================

"""
运行命令：

conda activate Qwen3_2 &&
CUDA_VISIBLE_DEVICES=0 \
USE_SPARSEKV=true \
USE_AUDIO_AWARE_TOPK=true \
AUDIO_BOUNDARY_BOOST=1.3 \
AUDIO_CONTINUITY_WEIGHT=0.1 \
AUDIO_SMOOTH_WINDOW=5 \
BUDGET=0.7 \
RATIO=0.5 \
WINDOW_SIZE=32 \
KERNEL_SIZE=5 \
POOLING=maxpool \
HEAD_SCORE_FILE=aggregated \
RESULTS_DIR=/path/to/results \
python LibriSpeech_qwen2.5_timing.py
"""

# ============================================================================
# 调试和验证
# ============================================================================

"""
1. 验证音频token信息是否正确传递：
   - 检查config.audio_token_start和config.audio_token_length
   - 在calcul_attn_sore中添加print语句验证

2. 验证增强是否生效：
   - 对比增强前后的attention scores分布
   - 检查边界位置的分数是否被增强

3. 性能监控：
   - 测量增强前后的计算时间
   - 检查内存使用情况

4. 结果验证：
   - 对比WER变化
   - 分析选择分布的变化
   - 检查边界选择比例
"""

