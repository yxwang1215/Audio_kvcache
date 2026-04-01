# Audio Encoder Head Importance Scores

这个工具用于计算 Audio Encoder 各层 attention heads 的重要性分数。

## 功能说明

该工具通过分析 audio encoder 各层的 attention weights 来计算每个 head 的重要性分数，支持以下指标：

1. **Importance Score**: 综合重要性分数（加权组合）
2. **Variance Score**: attention weights 的方差（反映 head 的激活程度）
3. **Entropy Score**: attention weights 的熵（反映 head 的信息量）
4. **Sparsity Score**: attention weights 的稀疏度（反映 head 的选择性）

## 使用方法

### 方式1: 使用 Shell 脚本（推荐）

```bash
# 编辑脚本配置
vim scripts/run_audio_encoder_head_scores.sh

# 运行脚本
bash scripts/run_audio_encoder_head_scores.sh
```

在脚本中需要配置：
- `MODEL_PATH`: 模型路径
- `AUDIO_DIR`: 音频文件目录（或 `AUDIO_FILES` 指定具体文件）
- `OUTPUT_FILE`: 输出 JSON 文件路径
- `MAX_FILES`: 可选，限制处理的文件数量（用于快速测试）

### 方式2: 直接使用 Python 脚本

```bash
python generate_audio_encoder_head_scores.py \
    --model_path /mnt/public/models/Qwen2.5-Omni-7B \
    --audio_dir /path/to/audio/files \
    --output_file /path/to/output/audio_encoder_head_scores.json \
    --device cuda \
    --max_files 10
```

或者指定具体的音频文件：

```bash
python generate_audio_encoder_head_scores.py \
    --model_path /mnt/public/models/Qwen2.5-Omni-7B \
    --audio_files /path/to/audio1.wav /path/to/audio2.wav \
    --output_file /path/to/output/audio_encoder_head_scores.json \
    --device cuda
```

## 输出格式

输出的 JSON 文件格式如下：

```json
{
    "0-0": [0.123, 0.456, 0.789, 0.234],
    "0-1": [0.234, 0.567, 0.890, 0.345],
    ...
    "35-15": [0.345, 0.678, 0.901, 0.456]
}
```

其中：
- 键格式：`"layer-head"`（例如 `"0-0"` 表示第 0 层第 0 个 head）
- 值格式：`[importance_score, variance_score, entropy_score, sparsity_score]`
  - `importance_score`: 综合重要性分数（0-1）
  - `variance_score`: 方差分数（0-1）
  - `entropy_score`: 熵分数（0-1）
  - `sparsity_score`: 稀疏度分数（0-1）

## 重要性指标说明

### 1. Variance Score（方差分数）
- **定义**: attention weights 的方差
- **含义**: 反映 head 的激活程度
- **高值**: head 对不同位置的关注差异较大（选择性较强）
- **低值**: head 对所有位置的关注较为均匀

### 2. Entropy Score（熵分数）
- **定义**: attention weights 的熵（信息熵）
- **含义**: 反映 head 的信息量
- **高值**: attention 分布较为均匀，信息量高
- **低值**: attention 集中在少数位置，信息量低

### 3. Sparsity Score（稀疏度分数）
- **定义**: top-k attention weights 的集中度
- **含义**: 反映 head 的选择性
- **高值**: attention 集中在少数位置（选择性高）
- **低值**: attention 分布较为分散

### 4. Importance Score（综合重要性分数）
- **定义**: 上述三个指标的加权平均
- **公式**: `importance = (variance + entropy + sparsity) / 3`
- **用途**: 用于排序和选择重要的 heads

## 注意事项

1. **内存使用**: 处理大量音频文件时，attention weights 会占用较多内存。建议使用 `--max_files` 限制文件数量进行测试。

2. **计算时间**: 每个音频文件需要完整的前向传播，处理时间取决于：
   - 音频文件数量
   - 音频长度
   - 模型大小

3. **音频格式**: 支持常见的音频格式（.wav, .mp3, .flac, .m4a, .ogg）

4. **设备要求**: 建议使用 GPU（`--device cuda`）以加速计算

## 示例

### 快速测试（处理 5 个文件）

```bash
python generate_audio_encoder_head_scores.py \
    --model_path /mnt/public/models/Qwen2.5-Omni-7B \
    --audio_dir /path/to/audio/files \
    --output_file ./head_score/test_scores.json \
    --device cuda \
    --max_files 5
```

### 完整分析（处理所有文件）

```bash
python generate_audio_encoder_head_scores.py \
    --model_path /mnt/public/models/Qwen2.5-Omni-7B \
    --audio_dir /path/to/audio/files \
    --output_file ./head_score/full_scores.json \
    --device cuda
```

## 与现有 Head Scores 的兼容性

输出的 JSON 格式与现有的 `audio_head_scores_*.json` 格式兼容，可以直接用于：
- SparseKV 的 head selection
- 其他需要 head importance scores 的应用

## 故障排除

### 问题1: 没有捕获到 attention weights

**可能原因**:
- audio encoder 的 forward 方法签名不匹配
- hook 注册失败

**解决方法**:
- 检查模型版本和代码兼容性
- 查看错误日志中的异常信息

### 问题2: 内存不足

**解决方法**:
- 减少 `--max_files` 的数量
- 使用更小的模型（如 3B 而不是 7B）
- 分批处理音频文件

### 问题3: 音频加载失败

**解决方法**:
- 检查音频文件格式是否支持
- 确保音频文件路径正确
- 检查文件权限

## 相关文件

- `generate_audio_encoder_head_scores.py`: 主脚本
- `scripts/run_audio_encoder_head_scores.sh`: 运行脚本
- `generate_head_scores.py`: LLM head scores 生成（用于对比）
- `batch_generate_head_scores.py`: 批量生成 LLM head scores

