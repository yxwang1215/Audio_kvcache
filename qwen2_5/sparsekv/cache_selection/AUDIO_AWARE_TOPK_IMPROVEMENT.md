# 基于音频时序特性的TopK选择算法改进方案

## 一、问题分析

基于对`topk_selection_records_134686_0000.json`的分析，我们发现了以下关键特点：

### 1.1 音频边界的重要性
- **发现**：前10%和后10%的选择比例均超过13%（期望10%），偏差约1.3-1.4倍
- **高密度区域**：音频开始部分（0.0%-1.3%）和结束部分（98.7%-100.0%）被所有层频繁选择
- **意义**：音频边界包含重要的起始/结束信息、静音检测或特殊标记

### 1.2 时序连续性特征
- **发现**：小间隔(≤5)比例100%，选择非常密集
- **连续选择比例**：约5%，最大连续片段仅2 tokens
- **意义**：选择策略是分散但密集的模式，符合音频时序特性

### 1.3 Layer间时序模式变化
- **早期Layers（前1/3）**：更关注音频后半段（位置=3178.56），用于获取上下文
- **后期Layers（后1/3）**：更关注音频前半段（位置=2821.31），聚焦关键特征
- **意义**：不同层有不同的时序关注点，形成层次化特征提取

### 1.4 选择密度特征
- **平均密度**：每个位置被选择约20次（36层中约20层选择）
- **最大密度**：36（所有层都选择）
- **意义**：某些关键位置被所有层关注，可能是重要的声学特征点

## 二、改进策略

### 2.1 边界增强（Boundary Boost）
**原理**：给音频开始和结束部分额外的权重

```python
# 对前10%和后10%的位置应用1.3-1.4倍的权重增强
boundary_boost = 1.3  # 基于分析结果
if position in [0, audio_length * 0.1) or position in [audio_length * 0.9, audio_length):
    score *= boundary_boost
```

**实现位置**：`AudioAwareTopKSelector.apply_boundary_boost()`

### 2.2 Layer自适应时序偏置（Layer Position Bias）
**原理**：不同layer关注不同的时序位置

```python
# 早期layer：对后半段增强
if layer_ratio < 0.33:
    position_bias = linspace(0.9, 1.1, seq_length)  # 后半段增强

# 后期layer：对前半段增强
elif layer_ratio > 0.67:
    position_bias = linspace(1.1, 0.9, seq_length)  # 前半段增强

# 中期layer：相对均匀
else:
    position_bias = ones(seq_length)
```

**实现位置**：`AudioAwareTopKSelector.apply_layer_position_bias()`

### 2.3 时序连续性平滑（Continuity Smoothing）
**原理**：使用滑动窗口平滑，增强相邻位置的选择相关性

```python
# 使用滑动窗口平均（窗口大小=5）
smoothed_score = conv1d(score, kernel_size=5)
final_score = (1 - continuity_weight) * original_score + continuity_weight * smoothed_score
```

**实现位置**：`AudioAwareTopKSelector.apply_continuity_smoothing()`

### 2.4 连续性约束选择（Continuity Constraint）
**原理**：在选择时考虑局部连续性，避免选择过于分散

```python
# 策略：先选择高分位置，然后确保这些位置周围也有选择
# 如果选择的间隔太大，尝试填充中间位置
```

**实现位置**：`AudioAwareTopKSelector.select_with_continuity_constraint()`

## 三、集成方案

### 3.1 修改SparseMM的`calcul_attn_sore`方法

在`/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/SparseMM/sparsemm/sparsemm_utils_timing.py`中：

```python
def calcul_attn_sore(self, key_states, query_states):
    # ... 原有代码 ...
    
    # 在返回前应用音频感知增强
    if self.config is not None:
        from audio_aware_topk_selection import create_audio_aware_selector
        
        selector = create_audio_aware_selector(
            self.config,
            self.layer_idx,
            self.num_hidden_layers,
            boundary_boost=float(os.getenv('AUDIO_BOUNDARY_BOOST', '1.3')),
            continuity_weight=float(os.getenv('AUDIO_CONTINUITY_WEIGHT', '0.1'))
        )
        
        seq_length = attn_weights_mean_pooling.shape[-1]
        # 添加batch维度进行处理
        attn_weights_mean_pooling = attn_weights_mean_pooling.unsqueeze(0)
        attn_weights_mean_pooling = selector.enhance_audio_aware_selection(
            attn_weights_mean_pooling,
            seq_length,
            self.layer_idx,
            self.num_hidden_layers
        )
        attn_weights_mean_pooling = attn_weights_mean_pooling.squeeze(0)
    
    return attn_weights_mean_pooling
```

### 3.2 配置参数

在`LibriSpeech_qwen2.5_timing.py`中添加环境变量支持：

```python
# 音频感知TopK选择参数
ENV_AUDIO_BOUNDARY_BOOST = float(os.environ.get("AUDIO_BOUNDARY_BOOST", "1.3"))
ENV_AUDIO_CONTINUITY_WEIGHT = float(os.environ.get("AUDIO_CONTINUITY_WEIGHT", "0.1"))
ENV_AUDIO_SMOOTH_WINDOW = int(os.environ.get("AUDIO_SMOOTH_WINDOW", "5"))
```

### 3.3 使用示例

```bash
# 启用音频感知TopK选择
conda activate Qwen3_2 &&
CUDA_VISIBLE_DEVICES=0 \
USE_SPARSEKV=true \
BUDGET=0.7 \
RATIO=0.5 \
WINDOW_SIZE=32 \
KERNEL_SIZE=5 \
POOLING=maxpool \
HEAD_SCORE_FILE=aggregated \
AUDIO_BOUNDARY_BOOST=1.3 \
AUDIO_CONTINUITY_WEIGHT=0.1 \
AUDIO_SMOOTH_WINDOW=5 \
RESULTS_DIR=/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv/LibriSpeech_audio_aware \
python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv/LibriSpeech_qwen2.5_timing.py
```

## 四、实验建议

### 4.1 参数调优实验

#### 实验1：边界增强系数
- **参数**：`AUDIO_BOUNDARY_BOOST` = [1.0, 1.2, 1.3, 1.4, 1.5]
- **指标**：WER, 选择分布, 边界选择比例
- **预期**：1.3-1.4可能最优（基于分析结果）

#### 实验2：连续性权重
- **参数**：`AUDIO_CONTINUITY_WEIGHT` = [0.0, 0.05, 0.1, 0.15, 0.2]
- **指标**：WER, 选择连续性, 小间隔比例
- **预期**：0.1左右可能最优

#### 实验3：平滑窗口大小
- **参数**：`AUDIO_SMOOTH_WINDOW` = [3, 5, 7, 9]
- **指标**：WER, 选择平滑度
- **预期**：5-7可能最优（基于分析中的小间隔≤5）

### 4.2 消融实验

1. **仅边界增强**：只应用`apply_boundary_boost`
2. **仅Layer偏置**：只应用`apply_layer_position_bias`
3. **仅连续性平滑**：只应用`apply_continuity_smoothing`
4. **完整方案**：应用所有增强

### 4.3 对比实验

- **Baseline**：当前SparseMM实现
- **Audio-Aware**：应用音频感知增强
- **指标**：WER, 选择分布, 时序连续性, 边界选择比例

## 五、预期效果

### 5.1 选择质量提升
- **边界选择**：更准确地识别音频边界，提高边界信息保留
- **时序连续性**：保持更好的时序连续性，符合音频特性
- **Layer分工**：不同layer更明确地关注不同时序位置

### 5.2 性能影响
- **计算开销**：增加约5-10%的计算时间（主要是平滑操作）
- **内存开销**：几乎无额外内存开销
- **准确率**：预期WER降低0.5-1.0%（通过更好的选择策略）

### 5.3 可解释性
- **选择模式**：更符合音频处理的直觉
- **Layer行为**：不同layer的关注点更清晰
- **边界重要性**：明确体现音频边界的重要性

## 六、进一步优化方向

### 6.1 动态边界检测
- 不固定使用前10%/后10%，而是基于attention score动态检测边界
- 使用梯度或变化率来识别音频边界

### 6.2 音频特征感知
- 如果有音频特征（如能量、频谱），可以结合这些特征进行选择
- 对静音段、语音段使用不同的选择策略

### 6.3 自适应连续性约束
- 根据音频内容动态调整连续性权重
- 语音段：高连续性；静音段：低连续性

### 6.4 多尺度选择
- 结合不同时间尺度的选择（短时、中时、长时）
- 使用多分辨率attention score进行选择

## 七、实现检查清单

- [ ] 在`sparsemm_utils_timing.py`中集成`AudioAwareTopKSelector`
- [ ] 添加环境变量配置支持
- [ ] 确保`config.audio_token_start`和`config.audio_token_length`正确传递
- [ ] 添加开关控制（可选启用/禁用音频感知增强）
- [ ] 添加日志记录增强前后的选择分布
- [ ] 运行消融实验验证各组件效果
- [ ] 运行参数调优实验找到最优参数
- [ ] 对比实验验证整体效果

## 八、注意事项

1. **性能考虑**：平滑操作会增加计算开销，需要权衡效果和性能
2. **参数敏感性**：不同数据集可能需要不同的参数设置
3. **向后兼容**：确保在非音频任务中不会影响性能
4. **测试覆盖**：需要测试各种音频长度和边界情况

