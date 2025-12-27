# KV Cache Methods Usage Guide

本文档说明如何在 Qwen2.5-Omni 模型上使用不同的 KV Cache 压缩方法。

## 支持的方法

| 方法 | 环境变量 | 描述 | 需要 Head Score |
|------|----------|------|----------------|
| **SparseKV** | `KV_METHOD="sparsekv"` | 基于 head score 的自适应 KV cache 压缩 | ✅ |
| **SnapKV** | `KV_METHOD="snapkv"` | 基于注意力权重的 KV cache 压缩 | ❌ |
| **AdaKV** | `KV_METHOD="adakv"` | 自适应 KV cache 分配 | ❌ |
| **PyramidKV** | `KV_METHOD="pyramidkv"` | 金字塔形 KV cache 压缩 | ❌ |
| **MixKV** | `KV_METHOD="mixkv"` | 混合策略 KV cache 压缩 | ❌ |
| **KNorm** | `KV_METHOD="knorm"` | 基于 Key 向量 L2 范数的 KV cache 压缩 | ❌ |
| **H2O** | `KV_METHOD="h2o"` | Heavy-Hitter Oracle: 基于累积注意力分数的 KV cache 压缩 | ❌ |
| **None** | `KV_METHOD="none"` | 禁用 KV cache 压缩（Full Cache） | ❌ |

## 向后兼容性

为了保持向后兼容，系统同时支持新旧两种配置方式：

```bash
# 新方式（推荐）
export KV_METHOD="sparsekv"

# 旧方式（向后兼容）
export USE_SPARSEKV="true"   # 等价于 KV_METHOD="sparsekv"
export USE_SPARSEKV="false"  # 等价于 KV_METHOD="none"
```

**优先级**：`KV_METHOD` > `USE_SPARSEKV`

## 通用环境变量

所有方法共享以下环境变量：

```bash
# 基本配置
export CUDA_VISIBLE_DEVICES=0              # GPU 设备
export MODEL_PATH="/path/to/model"         # 模型路径
export SAMPLE_LIMIT=0                      # 样本数量限制（0=全部）

# KV Cache 压缩配置
export KV_METHOD="sparsekv"                # 方法选择
export BUDGET=0.4                          # 压缩预算（<1.0 表示比例，>=1 表示绝对值）
export RATIO=0.5                           # 最小缓存比例
export WINDOW_SIZE=32                      # 窗口大小
export KERNEL_SIZE=5                       # 卷积核大小
export POOLING="maxpool"                   # 池化方法 (maxpool/avgpool)

# FFT 滤波配置（可选）
export USE_FFT_FILTER="false"              # 是否使用 FFT 滤波
export FFT_FILTER_TYPE="lowpass"           # 滤波类型
export FFT_CUTOFF_RATIO="0.7"              # 截止频率比例
export FFT_ALPHA="0.5"                     # 混合系数
export FFT_ADAPTIVE_CUTOFF="false"         # 自适应截止
```

## 方法特定配置

### 1. SparseKV（推荐用于音频模型）

```bash
#!/bin/bash
export KV_METHOD="sparsekv"
export BUDGET=0.4                          # 40% 压缩率
export RATIO=0.5                           # 最小缓存比例
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"

# Head Score 配置（SparseKV 特有）
export HEAD_SCORE_PATH="/path/to/audio_head_scores.json"
export HEAD_SCORE_AGGREGATION="first"      # first/mean/max/min

# 优化选项
export USE_OPTIMIZED_SPARSEMM="true"       # 使用优化实现
```

**完整示例脚本** (`run_librispeech_sparsekv.sh`):

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/mnt/public/models/Qwen2.5-Omni-3B"
export SAMPLE_LIMIT=0

export KV_METHOD="sparsekv"
export BUDGET=0.4
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export HEAD_SCORE_PATH="/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/audio_head/qwen2.5_omni/head_score/qwen2_5_3B/audio_head_scores_topk24_alpha1_beta1_gamma1.json"
export HEAD_SCORE_AGGREGATION="first"
export USE_OPTIMIZED_SPARSEMM="true"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_dataset && \
python code/qwen2.5_librispeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir ./results/SparseKV/budget0.4
```

### 2. SnapKV

```bash
#!/bin/bash
export KV_METHOD="snapkv"
export BUDGET=0.4
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"

# SnapKV 不需要 head score 配置
export USE_OPTIMIZED_SPARSEMM="false"
```

**完整示例脚本** (`run_librispeech_snapkv.sh`):

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/mnt/public/models/Qwen2.5-Omni-3B"
export SAMPLE_LIMIT=0

export KV_METHOD="snapkv"
export BUDGET=0.4
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export USE_OPTIMIZED_SPARSEMM="false"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_dataset && \
python code/qwen2.5_librispeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir ./results/SnapKV/budget0.4
```

### 3. AdaKV

```bash
#!/bin/bash
export KV_METHOD="adakv"
export BUDGET=0.4
export RATIO=0.2                           # floor_alpha 参数
export WINDOW_SIZE=32
export KERNEL_SIZE=7                       # AdaKV 默认使用 7
export POOLING="maxpool"
export USE_OPTIMIZED_SPARSEMM="false"
```

**完整示例脚本** (`run_librispeech_adakv.sh`):

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/mnt/public/models/Qwen2.5-Omni-3B"
export SAMPLE_LIMIT=0

export KV_METHOD="adakv"
export BUDGET=0.4
export RATIO=0.2
export WINDOW_SIZE=32
export KERNEL_SIZE=7
export POOLING="maxpool"
export USE_OPTIMIZED_SPARSEMM="false"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_dataset && \
python code/qwen2.5_librispeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir ./results/AdaKV/budget0.4
```

### 4. PyramidKV

```bash
#!/bin/bash
export KV_METHOD="pyramidkv"
export BUDGET=0.4
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="avgpool"                   # PyramidKV 默认使用 avgpool
export USE_OPTIMIZED_SPARSEMM="false"
```

**完整示例脚本** (`run_librispeech_pyramidkv.sh`):

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/mnt/public/models/Qwen2.5-Omni-3B"
export SAMPLE_LIMIT=0

export KV_METHOD="pyramidkv"
export BUDGET=0.4
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="avgpool"
export USE_OPTIMIZED_SPARSEMM="false"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_dataset && \
python code/qwen2.5_librispeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir ./results/PyramidKV/budget0.4
```

### 5. MixKV

```bash
#!/bin/bash
export KV_METHOD="mixkv"
export BUDGET=0.4
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"

# MixKV 特有配置
export SELECT_METHOD="headwisemixkv"               # attn/keydiff/mixkv/headwisemixkv/knorm/vnorm
export SIMILARITY_SCORE_PATH="/path/to/similarity_scores.json"  # 可选
export USE_OPTIMIZED_SPARSEMM="false"
```

**完整示例脚本** (`run_librispeech_mixkv.sh`):

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/mnt/public/models/Qwen2.5-Omni-3B"
export SAMPLE_LIMIT=0

export KV_METHOD="mixkv"
export BUDGET=0.4
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export SELECT_METHOD="mixkv"
export USE_OPTIMIZED_SPARSEMM="false"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_dataset && \
python code/qwen2.5_librispeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir ./results/MixKV/budget0.4
```

### 6. KNorm

```bash
#!/bin/bash
export KV_METHOD="knorm"
export BUDGET=0.4
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"

# KNorm 不需要 head score 配置
export USE_OPTIMIZED_SPARSEMM="false"
```

**完整示例脚本** (`run_librispeech_knorm.sh`):

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/mnt/public/models/Qwen2.5-Omni-3B"
export SAMPLE_LIMIT=0

export KV_METHOD="knorm"
export BUDGET=0.4
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export USE_OPTIMIZED_SPARSEMM="false"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_dataset && \
python code/qwen2.5_librispeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir ./results/KNorm/budget0.4
```

**KNorm 方法说明**:
- KNorm (Key Norm) 使用 Key 向量的 L2 范数来选择重要的 KV 对
- 较小的 Key 范数表示该 token 更重要（因为会收到更多注意力）
- 使用标准的 `DynamicCache`（所有 head 压缩到相同长度）
- 不需要 Head Score 文件

### 7. H2O (Heavy-Hitter Oracle)

```bash
#!/bin/bash
export KV_METHOD="h2o"
export BUDGET=0.4
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="avgpool"

# H2O 特有配置
export H2O_HEAVY_RATIO=0.5              # Heavy hitter vs recent tokens 比例
export USE_OPTIMIZED_SPARSEMM="false"
```

**完整示例脚本** (`run_librispeech_h2o.sh`):

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/mnt/public/models/Qwen2.5-Omni-3B"
export SAMPLE_LIMIT=0

export KV_METHOD="h2o"
export BUDGET=0.4
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="avgpool"
export H2O_HEAVY_RATIO=0.5

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_dataset && \
python code/qwen2.5_librispeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir ./results/H2O/budget0.4
```

**H2O 方法说明**:
- H2O (Heavy-Hitter Oracle) 基于累积注意力分数识别"重击手"token
- 维护两类 token：heavy hitters（累积注意力分数最高的 token）和 recent tokens（滑动窗口）
- `H2O_HEAVY_RATIO` 控制 heavy hitters 与 recent tokens 的比例（默认 0.5）
- 基于论文：https://arxiv.org/abs/2306.14048
- 使用标准的 `DynamicCache`
- 不需要 Head Score 文件

### 8. Full Cache (无压缩)

```bash
#!/bin/bash
export KV_METHOD="none"
# 无需其他 KV cache 配置
```

**完整示例脚本** (`run_librispeech_full.sh`):

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/mnt/public/models/Qwen2.5-Omni-3B"
export SAMPLE_LIMIT=0

export KV_METHOD="none"

source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen2_5/sparsekv_fft_dataset && \
python code/qwen2.5_librispeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir ./results/FullCache
```

## 批量运行所有方法

使用 `run_all_kv_methods.sh` 脚本批量测试所有方法：

```bash
# 运行所有方法
./ASR/run_all_kv_methods.sh

# 运行指定方法
./ASR/run_all_kv_methods.sh sparsekv snapkv adakv
```

## Budget 参数说明

`BUDGET` 参数支持两种模式：

1. **比例模式** (`BUDGET < 1.0`): 按 prefill token 数量的比例计算
   ```bash
   export BUDGET=0.4  # 保留 40% 的 KV cache
   ```

2. **绝对值模式** (`BUDGET >= 1`): 固定 token 数量
   ```bash
   export BUDGET=4096  # 固定保留 4096 个 token
   ```

## 方法对比

| 方法 | 需要 Head Score | 计算复杂度 | 推荐场景 |
|------|----------------|-----------|---------|
| SparseKV | ✅ | 中 | 音频理解任务（推荐） |
| SnapKV | ❌ | 低 | 通用场景 |
| AdaKV | ❌ | 中 | 需要自适应分配 |
| PyramidKV | ❌ | 低 | 长序列处理 |
| MixKV | ❌ | 高 | 需要混合策略 |
| KNorm | ❌ | 低 | 基于 Key 范数的压缩 |
| H2O | ❌ | 中 | 基于累积注意力的压缩 |

## 注意事项

1. **Head Score 文件**: SparseKV 方法需要预计算的 head score 文件，其他方法不需要
2. **GQA 兼容性**: 所有方法都已支持 GQA（Grouped Query Attention）
3. **动态 Budget**: 使用比例模式时，Budget 会根据每个样本的 prefill token 数量动态调整
4. **内存使用**: 不同方法的内存使用有所不同，PyramidKV 和 SnapKV 内存效率较高

## 文件结构

```
scripts/
├── ASR/
│   ├── run_librispeech.sh          # SparseKV (默认)
│   ├── run_librispeech_snapkv.sh   # SnapKV
│   ├── run_librispeech_adakv.sh    # AdaKV
│   ├── run_librispeech_pyramidkv.sh # PyramidKV
│   ├── run_librispeech_mixkv.sh    # MixKV
│   ├── run_librispeech_knorm.sh    # KNorm
│   ├── run_librispeech_h2o.sh      # H2O
│   └── run_all_kv_methods.sh       # 批量运行
├── ST/
│   ├── run_covost2_en_zh.sh
│   ├── run_covost2_zh_en.sh
│   ├── run_covost2_en_zh_h2o.sh    # H2O
│   └── run_covost2_zh_en_h2o.sh    # H2O
├── audio_captioning/
│   └── ...
├── audio_qa/
│   └── ...
└── KV_METHODS_README.md            # 本文档
```
