# Decoded Token 与区域对齐原理说明

## 核心思想

这个分析工具的核心假设是：**当模型生成某个单词的 token 时，如果该 head 是"重要的"（对音频理解关键），那么它的注意力应该集中在对应单词的音频 token 区域上**。

## 完整流程

### 1. 数据准备阶段

#### 1.1 获取单词时间对齐（WhisperX）
```
WhisperX JSON → 单词列表（带时间戳）
例如: "hello" → start_time: 0.5s, end_time: 0.8s, score: 0.98
```

#### 1.2 时间戳映射到音频 Token 位置
```python
# 假设音频有 1000 个音频 token，总时长 10 秒
# "hello" 在 0.5s - 0.8s
# 映射到: audio_token_start = 50, audio_token_end = 80
```

**原理**：假设音频 token 在时间上均匀分布，通过比例映射：
- `start_ratio = start_time / audio_duration`
- `token_start = int(start_ratio * num_audio_tokens)`

### 2. Token 到单词的对齐（关键步骤）

#### 2.1 问题
模型生成的 token 序列和 WhisperX 的单词序列可能不完全一致：
- Token 化差异：一个单词可能被分成多个 token（如 "hello" → ["hello", "##llo"]）
- 生成顺序：模型可能跳过某些单词或生成额外内容
- 标点/空格：tokenizer 可能产生不同的 token 序列

#### 2.2 对齐策略（`align_tokens_to_words`）

**核心算法**：
```python
1. 解码每个生成的 token → 文本片段
2. 累积 token 文本，直到匹配到 WhisperX 中的单词
3. 将匹配的 token 标记为属于该单词
4. 如果累积太多 token 仍未匹配，跳过当前单词（容错机制）
```

**示例**：
```
生成的 tokens: ["The", " quick", " brown", " fox"]
WhisperX 单词: ["The", "quick", "brown", "fox"]

对齐过程：
- Token 0 "The" → 匹配单词 0 "The" ✓
- Token 1 " quick" → 累积 "The quick" → 匹配单词 1 "quick" ✓
- Token 2 " brown" → 累积 "The quick brown" → 匹配单词 2 "brown" ✓
- Token 3 " fox" → 累积 "The quick brown fox" → 匹配单词 3 "fox" ✓
```

**标准化处理**（提高鲁棒性）：
- 去除标点、空格、转小写
- 例如："Hello!" → "hello"，"The" → "the"

**容错机制**：
- 如果累积超过 `max_skip_threshold` 个 token 仍未匹配，尝试匹配下一个单词
- 如果下一个单词匹配成功，跳过当前单词（可能是 WhisperX 标注错误或模型生成差异）

### 3. 注意力分析阶段

#### 3.1 捕获注意力权重

在生成过程中，通过 hook 捕获每个 decode step 的注意力：
```python
# 对于每个生成步（生成一个 token）
for step_idx in range(num_generated_tokens):
    # 获取该步所有 layer/head 的注意力权重
    step_attention = {
        (layer_0, head_0): attention_tensor_0,  # shape: [seq_len]
        (layer_0, head_1): attention_tensor_1,
        ...
        (layer_L, head_H): attention_tensor_LH
    }
```

#### 3.2 计算 Top-K 注意力位置

对于每个 layer/head 的注意力向量 `attn`（shape: `[seq_len]`）：
```python
# 找到 top-k 个注意力最高的位置
topk_val = min(self.topk, attn.numel())  # 例如 topk=32
values, indices = attn.topk(topk_val)

# indices 是注意力最高的 k 个位置的索引
# 例如: indices = [50, 51, 52, 100, 101, ...]  # 这些是 prefill 阶段的 token 位置
```

#### 3.3 统计命中情况

对于生成单词 "hello" 的 token，检查 top-k 位置是否落在 "hello" 的音频 token 区域：

```python
# 目标单词的音频 token 范围
word_audio_indices = {50, 51, 52, ..., 80}  # "hello" 对应的音频 token

# 统计 top-k 中有多少落在目标区域
word_hits = sum(1 for idx in topk_indices if idx in word_audio_indices)
# 例如: topk_indices = [50, 51, 100, 101, ...]
#       word_hits = 2  (50, 51 落在目标区域)
```

**三个关键指标**：

1. **Hit Ratio (命中率)**: `r_hit = c_word / K`
   - `c_word`: top-k 中命中目标单词音频 token 的数量
   - `K`: top-k 值（如 32）
   - **含义**：top-k 中有多少比例关注了目标单词

2. **Attention Ratio (注意力比例)**: `r_attn = weight_word / weight_all_audio`
   - `weight_word`: 目标单词音频 token 的注意力权重和
   - `weight_all_audio`: 所有音频 token 的注意力权重和
   - **含义**：对目标单词的注意力占所有音频注意力的比例

3. **Coverage Ratio (覆盖率)**: `r_cov = c_word / |A(w_i)|`
   - `c_word`: top-k 中命中目标单词音频 token 的数量
   - `|A(w_i)|`: 目标单词的音频 token 总数
   - **含义**：目标单词的音频 token 有多少被 top-k 覆盖

### 4. Head 重要性评分

#### 4.1 聚合统计

对于每个 layer/head，收集所有高质量单词 token 的统计：
```python
# 例如：L10_H5 在所有单词 token 上的表现
layer_head_data['L10_H5'] = {
    'topk_hits_on_word': [2, 3, 1, 4, ...],  # 每个 token 的命中数
    'attn_weight_on_word': [0.15, 0.20, 0.10, ...],
    ...
}
```

#### 4.2 计算平均指标

```python
# 计算该 head 在所有单词 token 上的平均表现
S_hit = mean(hit_ratios)      # 平均命中率
S_attn = mean(attn_ratios)    # 平均注意力比例
S_cov = mean(coverage_ratios) # 平均覆盖率
```

#### 4.3 加权重要性分数

```python
importance_score = α·S_hit + β·S_attn + γ·S_cov

# 默认权重: α=1.0, β=1.0, γ=1.0
# 这意味着三个指标同等重要
```

**为什么这个公式有效？**

- **S_hit 高**：说明该 head 在 top-k 中经常关注目标单词的音频区域
- **S_attn 高**：说明该 head 对目标单词分配了较高的注意力权重
- **S_cov 高**：说明该 head 能够覆盖目标单词的大部分音频 token

**三者结合**：如果一个 head 同时满足这三个条件，说明它在生成单词时**确实在关注对应的音频区域**，因此是"重要的"。

## 为什么能找到重要的 Head？

### 核心假设验证

1. **假设**：重要的 head 在生成单词时，注意力应该集中在对应单词的音频 token 上
2. **验证方法**：
   - 使用 WhisperX（高精度 ASR）提供"正确答案"（单词及其时间戳）
   - 检查模型生成该单词时，head 的注意力是否真的关注了对应区域
3. **结果**：如果某个 head 的 `importance_score` 高，说明它确实在"做正确的事"

### 统计显著性

通过分析**大量样本**和**大量单词 token**，可以：
- 排除偶然性：单个样本的偶然匹配不能说明问题
- 发现模式：如果某个 head 在大多数情况下都能正确关注，说明它确实重要
- 量化重要性：通过平均和分布统计，量化每个 head 的重要性

### Top-K 的优势

使用 top-k 而非绝对值的好处：
- **更鲁棒**：不受注意力权重绝对值大小影响
- **更符合实际**：模型通常只关注 top-k 个位置
- **可解释性**：top-k 命中率更直观（"32 个位置中有几个命中目标"）

## 潜在局限性

1. **对齐误差**：
   - Token 到单词的对齐可能不完美（特别是标点、空格处理）
   - 时间戳到音频 token 的映射是线性假设，可能不够精确

2. **WhisperX 标注质量**：
   - 依赖 WhisperX 的准确性（虽然筛选了 score > 0.95）
   - 如果 WhisperX 标注错误，会影响分析结果

3. **模型生成质量**：
   - 如果模型生成质量差（与 WhisperX 差异大），对齐率会下降
   - 代码中有诊断机制来检测这种情况

4. **Top-K 选择**：
   - Top-k 值（如 32）是超参数，可能影响结果
   - 不同模型可能需要不同的 top-k 值

## 总结

这个方法的**核心创新**在于：
1. **使用外部对齐信息**（WhisperX）作为"正确答案"
2. **验证模型内部注意力**是否与正确答案一致
3. **量化每个 head 的一致性**，从而识别重要 head

这是一种**基于对齐验证的重要性分析方法**，而不是基于梯度或扰动的方法。

