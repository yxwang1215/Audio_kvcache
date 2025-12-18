# Audio KV Cache Compression Methods

本项目实现了多种 KV Cache 压缩方法，用于优化音频理解任务中的内存使用和计算效率。

## 📋 目录

- [环境配置](#环境配置)
- [核心方法介绍](#核心方法介绍)
- [支持的方法](#支持的方法)
- [快速开始](#快速开始)
- [详细配置](#详细配置)
- [FFT 滤波增强](#fft-滤波增强)
- [目录结构](#目录结构)
- [常见问题](#常见问题)

## 🔧 环境配置

本项目使用 conda 环境管理。环境配置文件位于项目根目录下的 `environment.yml`。

### 创建环境

```bash
# 使用 environment.yml 创建 conda 环境
conda env create -f environment.yml

# 激活环境
conda activate Qwen3_2
```

### 环境说明

- **环境名称**: `Qwen3_2`
- **Python 版本**: 3.11.0
- **主要依赖**: 
  - PyTorch 及相关深度学习库
  - Transformers
  - 音频处理库（librosa, soundfile 等）
  - 其他依赖请参考 `environment.yml` 文件

详细依赖列表请查看项目根目录下的 `environment.yml` 文件。

### 编译 SparseMM 算子

本项目需要编译 SparseMM 的 CUDA 算子。请参考 SparseMM 仓库配置算子：

```bash
# 进入 SparseMM csrc 目录
cd /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/SparseMM/csrc

# 编译 CUDA 算子
make

# 或者使用 Python setup
pip install packaging torch
pip uninstall ninja && pip cache purge && pip install ninja --no-cache-dir
python setup.py build_ext --inplace
```

**注意**: 
- 如果遇到 CUDA 编译错误，请检查您的 GPU 虚拟架构（GPU Virtual Architecture）和 GPU 特性（GPU Feature）
- 可能需要根据您的 GPU 架构修改 `csrc/build.py` 中的编译标志
- 编译完成后，算子会自动被项目代码调用

## ⭐ 核心方法介绍

本项目实现了多种 KV Cache 压缩方法，其中**核心方法**包括：

### 1. SparseKV - 基于 Head Score 的自适应 KV Cache 压缩

**SparseKV** 是本项目的核心方法，专门针对音频理解任务设计。

#### 核心思想

SparseKV 通过**预计算的 Head Score** 来实现自适应的 KV Cache 分配：

- **Head Score 机制**: 为每个注意力头（attention head）计算重要性分数，分数高的头获得更多的 KV Cache 预算
- **自适应分配**: 根据 Head Score 动态分配每个头的 KV Cache 容量，重要头保留更多信息
- **音频感知**: 通过分析音频任务中的注意力模式，识别对音频理解最关键的注意力头

#### 工作原理

1. **预计算阶段**: 使用 `audio_head/` 目录下的工具分析模型在音频任务上的注意力模式，生成 Head Score 文件
2. **运行时阶段**: 
   - 根据 Head Score 为每个头分配不同的 KV Cache 容量
   - 使用注意力权重计算每个 token 的重要性
   - 基于重要性分数选择保留的 KV Cache

#### 优势

- ✅ **任务特化**: 针对音频理解任务优化，性能优于通用方法
- ✅ **自适应**: 根据 Head Score 自动调整，无需手动调参
- ✅ **高效**: 在保持性能的同时显著减少内存使用

### 2. FFT 频谱滤波 - SparseKV+FFT 的核心增强

**FFT 频谱滤波**是 SparseKV+FFT 方法中的核心增强技术，通过频域分析提升注意力权重计算的准确性。

#### 核心思想

FFT 滤波将注意力权重从时域转换到频域，利用频谱特性进行滤波和增强：

- **频域分析**: 使用 FFT 将注意力权重序列转换到频域
- **频谱滤波**: 通过低通/高通滤波器保留或抑制特定频率成分
- **时域重构**: 将滤波后的频谱转换回时域，得到平滑的注意力权重

#### 工作原理

1. **FFT 变换**: 将注意力权重序列进行 FFT 变换，得到频谱表示
2. **频谱滤波**: 
   - **低通滤波**: 保留低频成分，平滑注意力权重，适合捕捉长期依赖
   - **高通滤波**: 保留高频成分，突出局部变化
   - **自适应截止**: 根据序列特性动态调整截止频率
3. **IFFT 重构**: 将滤波后的频谱通过 IFFT 转换回时域
4. **混合策略**: 将原始注意力权重与滤波后的权重按比例混合

#### 优势

- ✅ **频谱感知**: 利用频域特性更好地理解注意力模式
- ✅ **平滑增强**: 通过滤波减少噪声，提升注意力权重的质量
- ✅ **音频适配**: 特别适合音频信号的频谱特性

#### 与 SparseKV 的结合

SparseKV+FFT 将两种技术结合：
- **SparseKV**: 提供基于 Head Score 的自适应 KV Cache 分配
- **FFT 滤波**: 在计算注意力权重时进行频谱增强，提升选择准确性

这种结合在音频理解任务上取得了最佳的性能表现。

## 🎯 支持的方法

| 方法 | 环境变量 | 描述 | 需要 Head Score | FFT 支持 |
|------|----------|------|----------------|---------|
| **SparseKV** | `KV_METHOD="sparsekv"` | 基于 head score 的自适应 KV cache 压缩 | ✅ | ✅ |
| **SnapKV** | `KV_METHOD="snapkv"` | 基于注意力权重的 KV cache 压缩 | ❌ | ✅ |
| **AdaKV** | `KV_METHOD="adakv"` | 自适应 KV cache 分配 | ❌ | ❌ |
| **PyramidKV** | `KV_METHOD="pyramidkv"` | 金字塔形 KV cache 压缩 | ❌ | ❌ |
| **MixKV** | `KV_METHOD="mixkv"` | SparseKV + MixKV 混合策略 KV cache 压缩 | ✅ | ❌ |
| **SnapKV_FFT** | `KV_METHOD="sparsekv"` + `HEAD_SCORE_FILE="uniform"` + `USE_FFT_FILTER="true"` | SparseKV (uniform) + FFT 频谱滤波 | ❌ | ✅ |
| **SparseKV_FFT** | `KV_METHOD="sparsekv"` + `USE_FFT_FILTER="true"` | SparseKV + FFT 频谱滤波 | ✅ | ✅ |

## 🚀 快速开始

### 1. SparseKV

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/path/to/model"
export SAMPLE_LIMIT=0

# SparseKV 配置
export KV_METHOD="sparsekv"
export BUDGET=0.4                          # 40% 压缩率
export RATIO=0.5                          
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"

# Head Score 配置（必需）
export HEAD_SCORE_PATH="/path/to/audio_head_scores.json"
export HEAD_SCORE_AGGREGATION="first"      # first/mean/max/min

# 运行脚本
source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/sparsekv_fft_dataset && \
python code/qwen3_kespeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir ./results/SparseKV/budget0.4
```

### 2. SnapKV

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/path/to/model"
export SAMPLE_LIMIT=0

# SnapKV 配置
export KV_METHOD="snapkv"
export BUDGET=0.4
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"

# 运行脚本
source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/sparsekv_fft_dataset && \
python code/qwen3_kespeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir ./results/SnapKV/budget0.4
```

### 3. AdaKV

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/path/to/model"
export SAMPLE_LIMIT=0

# AdaKV 配置
export KV_METHOD="adakv"
export BUDGET=0.4
export RATIO=0.2                           # floor_alpha 参数
export WINDOW_SIZE=32
export KERNEL_SIZE=7                       # AdaKV 默认使用 7
export POOLING="maxpool"

# 运行脚本
source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/sparsekv_fft_dataset && \
python code/qwen3_kespeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir ./results/AdaKV/budget0.4
```

### 4. PyramidKV

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/path/to/model"
export SAMPLE_LIMIT=0

# PyramidKV 配置
export KV_METHOD="pyramidkv"
export BUDGET=0.4
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="avgpool"                   # PyramidKV 默认使用 avgpool

# 运行脚本
source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/sparsekv_fft_dataset && \
python code/qwen3_kespeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir ./results/PyramidKV/budget0.4
```

### 5. MixKV (SparseKV+MixKV)

**注意**: MixKV 是 SparseKV + MixKV 的混合方法。

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/path/to/model"
export SAMPLE_LIMIT=0

# MixKV 配置（基于 SparseKV + MixKV）
export KV_METHOD="sparsekv"               # 基座方法：使用 SparseKV
export METHOD="mixsparsemm"               # 指定使用 MixedSparseMM（SparseKV 基座 + MixKV 评分）
export BUDGET=0.4
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"

# Head Score 配置（使用均匀分布，不需要 head score 文件）
export HEAD_SCORE_FILE="uniform"          # 使用均匀分布

# FFT 滤波配置（启用 FFT 增强）
export USE_FFT_FILTER="true"              # 启用 FFT 滤波
export FFT_FILTER_TYPE="lowpass"          # 滤波类型: lowpass/highpass/saliency
export FFT_CUTOFF_RATIO="0.7"             # 截止频率比例
export FFT_ALPHA="0.5"                    # 混合系数
export FFT_ADAPTIVE_CUTOFF="false"        # 自适应截止频率
export FFT_USE_SALIENCY="false"           # 是否使用显著性检测

# MixKV 特有配置（固定配置）
export SELECT_METHOD="headwisemixkv"      # 固定为 headwisemixkv（基于 head 的混合策略）

# 运行脚本
source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/sparsekv_fft_dataset && \
python code/qwen3_kespeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir ./results/MixKV/budget0.4
```

## 🔧 详细配置

### 通用环境变量

所有方法共享以下环境变量：

```bash
# 基本配置
export CUDA_VISIBLE_DEVICES=0              # GPU 设备
export MODEL_PATH="/path/to/model"         # 模型路径
export SAMPLE_LIMIT=0                      # 样本数量限制（0=全部）
                                           # 注意：只有 CoVoST2 en_zh 任务使用样本限制（SAMPLE_LIMIT=4898），其他任务默认使用全部样本

# KV Cache 压缩配置
export KV_METHOD="sparsekv"                # 方法选择: sparsekv/snapkv/adakv/pyramidkv/mixkv
export BUDGET=0.4                          # 压缩预算（<1.0 表示比例，>=1 表示绝对值）
export RATIO=0.5                           # 最小缓存比例
export WINDOW_SIZE=32                      # 窗口大小
export KERNEL_SIZE=5                       # 卷积核大小
export POOLING="maxpool"                   # 池化方法 (maxpool/avgpool)
```

### SparseKV 特有配置

```bash
# Head Score 配置（必需）
export HEAD_SCORE_PATH="/path/to/audio_head_scores.json"
export HEAD_SCORE_AGGREGATION="first"      # first/mean/max/min

# 或者使用均匀分布（不需要 head score 文件）
export HEAD_SCORE_FILE="uniform"
```

### MixKV 特有配置

**注意**: MixKV 是 SparseKV + MixKV 的混合方法，需要 Head Score 文件。

```bash
# 基座方法配置（必需）
export KV_METHOD="sparsekv"               # 基座方法：使用 SparseKV
export METHOD="mixsparsemm"               # 指定使用 MixedSparseMM（SparseKV 基座 + MixKV 评分）

# Head Score 配置（必需，来自 SparseKV 基座方法）
export HEAD_SCORE_PATH="/path/to/audio_head_scores.json"
export HEAD_SCORE_AGGREGATION="first"      # first/mean/max/min

# MixKV 选择方法（固定配置）
export SELECT_METHOD="headwisemixkv"      # 固定为 headwisemixkv（基于 head 的混合策略）
                                           # 注意：当前实现中此参数固定为 headwisemixkv
                                           # 其他可选值（当前未使用）:
                                           # - attn: 基于注意力权重
                                           # - keydiff: 基于 key 差异
                                           # - mixkv: 混合策略
                                           # - knorm: 基于 key 范数
                                           # - vnorm: 基于 value 范数
```

### Budget 参数说明

`BUDGET` 参数支持两种模式：

1. **比例模式** (`BUDGET < 1.0`): 按 prefill token 数量的比例计算
   ```bash
   export BUDGET=0.4  # 保留 40% 的 KV cache
   ```

2. **绝对值模式** (`BUDGET >= 1`): 固定 token 数量
   ```bash
   export BUDGET=4096  # 固定保留 4096 个 token
   ```

## 🎵 FFT 滤波增强

以下 KV Cache 方法支持 FFT 频谱滤波增强，通过设置 `USE_FFT_FILTER="true"` 来启用：
- SparseKV
- SnapKV

**注意**: AdaKV、MixKV 和 PyramidKV 目前不支持 FFT 滤波。

### SnapKV_FFT

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/path/to/model"
export SAMPLE_LIMIT=4898                  # CoVoST2 en_zh 任务使用样本数量限制（0=全部）

# SparseKV 基础配置（使用均匀分布，不需要 head score 文件）
export KV_METHOD="sparsekv"
export BUDGET=0.8
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export HEAD_SCORE_FILE="uniform"           # 使用均匀分布，不需要 head score 文件

# FFT 滤波配置
export USE_FFT_FILTER="true"               # 启用 FFT 滤波
export FFT_FILTER_TYPE="lowpass"           # 滤波类型: lowpass/highpass/saliency
export FFT_CUTOFF_RATIO="0.7"              # 截止频率比例
export FFT_ALPHA="0.5"                     # 混合系数
export FFT_ADAPTIVE_CUTOFF="false"         # 自适应截止频率
export FFT_USE_SALIENCY="false"            # 是否使用显著性检测

# 运行脚本
source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/sparsekv_fft_dataset && \
python code/qwen3_covost2.py \
    --direction en_zh \
    --model-path "${MODEL_PATH}" \
    --output-dir ./results/SnapKV_FFT/budget0.8
```

### SparseKV_FFT

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/path/to/model"
export SAMPLE_LIMIT=0

# SparseKV 基础配置
export KV_METHOD="sparsekv"
export BUDGET=0.8
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export HEAD_SCORE_PATH="/path/to/audio_head_scores.json"
export HEAD_SCORE_AGGREGATION="first"

# FFT 滤波配置
export USE_FFT_FILTER="true"
export FFT_FILTER_TYPE="lowpass"
export FFT_CUTOFF_RATIO="0.7"
export FFT_ALPHA="0.5"
export FFT_ADAPTIVE_CUTOFF="false"
export FFT_USE_SALIENCY="false"

# 运行脚本
source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/sparsekv_fft_dataset && \
python code/qwen3_kespeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir ./results/SparseKV_FFT/budget0.8
```

### FFT 滤波参数说明

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `USE_FFT_FILTER` | 是否启用 FFT 滤波 | `"false"` | `"true"` / `"false"` |
| `FFT_FILTER_TYPE` | 滤波类型 | `"lowpass"` | `"lowpass"` / `"highpass"` / `"saliency"` |
| `FFT_CUTOFF_RATIO` | 截止频率比例 | `"0.7"` | `0.0` - `1.0` |
| `FFT_ALPHA` | 混合系数 | `"0.5"` | `0.0` - `1.0` |
| `FFT_ADAPTIVE_CUTOFF` | 自适应截止频率 | `"false"` | `"true"` / `"false"` |
| `FFT_USE_SALIENCY` | 使用显著性检测 | `"false"` | `"true"` / `"false"` |

**注意**: FFT 滤波可以与支持的方法（SparseKV、SnapKV）结合使用，只需在对应方法的配置基础上添加 FFT 相关环境变量即可。AdaKV、MixKV 和 PyramidKV 不支持 FFT 滤波。

## 📁 目录结构

```
Audio_kvcache/
├── qwen2_5/                          # Qwen2.5-Omni 模型
│   └── sparsekv_fft_dataset/
│       ├── code/                     # 代码文件
│       └── scripts/                  # 运行脚本
│           ├── ASR/                  # 自动语音识别任务
│           ├── ST/                   # 语音翻译任务
│           └── KV_METHODS_README.md # 详细文档
├── qwen3/                            # Qwen3-Omni 模型
│   └── sparsekv_fft_dataset/
│       ├── code/
│       └── scripts/
│           ├── ASR/
│           ├── ST/
│           └── audio_qa/
├── gemma/                            # Gemma 模型
│   └── sparsekv_fft_dataset/
│       ├── code/
│       └── scripts/
├── audio_head/                       # Head Score 计算工具
│   ├── qwen2.5_omni/
│   ├── qwen3_omni/
│   └── gemma/
└── README.md                         # 本文档
```

## 📝 使用示例

### 示例 1: 运行 KeSpeech ASR 任务（SparseKV）

```bash
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/sparsekv_fft_dataset/scripts/ASR
bash run_kespeech.sh
```

### 示例 2: 运行 CoVoST2 语音翻译任务（SnapKV_FFT）

**注意**: **只有 CoVoST2 en_zh 任务使用了样本数量限制** `SAMPLE_LIMIT=4898`，其他所有任务默认使用全部样本（`SAMPLE_LIMIT=0`）。如需使用全部样本，请修改脚本中的 `SAMPLE_LIMIT=0`。

```bash
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/sparsekv_fft_dataset/scripts/ST
bash run_covost2_en_zh.sh
```

## 🔍 方法对比

| 方法 | 需要 Head Score | 计算复杂度 | 内存效率 | 推荐场景 |
|------|----------------|-----------|---------|---------|
| **SparseKV** | ✅ | 中 | 中 | 音频理解任务（推荐） |
| **SnapKV** | ❌ | 低 | 高 | 通用场景 |
| **AdaKV** | ❌ | 中 | 中 | 需要自适应分配 |
| **PyramidKV** | ❌ | 低 | 高 | 长序列处理 |
| **MixKV** | ✅ | 高 | 中 | SparseKV + MixKV 混合策略（需要基座方法参数） |
| **SnapKV_FFT** | ❌ | 中 | 高 | 音频任务 + 频谱增强 |
| **SparseKV_FFT** | ✅ | 中 | 中 | 音频任务 + 频谱增强（推荐） |

## 🐛 常见问题

### Q1: 如何生成 Head Score 文件？

A: 使用 `audio_head/` 目录下的工具：

```bash
cd audio_head/qwen3_omni
bash run_word_attention_analysis_E2B.sh
```

### Q2: FFT 滤波和普通方法的区别？

A: FFT 滤波使用频谱域分析来增强注意力权重计算，通常能提升音频任务的表现。可以通过设置 `USE_FFT_FILTER="true"` 来启用。

### Q3: 如何选择合适的方法？

A: 
- **音频理解任务**: 推荐使用 SparseKV 或 SparseKV_FFT
- **通用任务**: 推荐使用 SnapKV 或 SnapKV_FFT
- **长序列**: 推荐使用 PyramidKV
- **需要混合策略**: 使用 MixKV（SparseKV + MixKV 混合方法，需要 Head Score）

### Q4: Budget 参数如何设置？

A: 
- 从 0.8 开始尝试（保留 80% 的 KV cache）
- 根据任务表现和内存限制调整
- 音频任务通常可以使用 0.6-0.8 的 budget

### Q5: 如何验证方法是否生效？

A: 检查日志输出，应该看到类似信息：
```
[SnapKV] SnapKV attention enabled successfully
[SparseMM FFT] Layer 0: Using FFT spectral filtering with filter_type=lowpass, cutoff_ratio=0.7, alpha=0.5
```

## 📚 更多资源

- 详细的方法说明和参数配置，请参考各模型目录下的 `scripts/KV_METHODS_README.md`
- Head Score 计算工具使用说明，请参考 `audio_head/` 目录下的文档

## 📄 许可证

请参考项目根目录的 LICENSE 文件。

