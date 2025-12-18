# Audio KV Cache Compression Methods

This project implements multiple KV Cache compression methods for optimizing memory usage and computational efficiency in audio understanding tasks.

## 📋 Table of Contents

- [Environment Setup](#environment-setup)
- [Core Methods Introduction](#core-methods-introduction)
- [Supported Methods](#supported-methods)
- [Quick Start](#quick-start)
- [Detailed Configuration](#detailed-configuration)
- [FFT Filter Enhancement](#fft-filter-enhancement)
- [Directory Structure](#directory-structure)
- [Usage Examples](#usage-examples)
- [Method Comparison](#method-comparison)
- [FAQ](#faq)

## 🔧 Environment Setup

This project uses conda for environment management. The environment configuration file is located at `environment.yml` in the project root directory.

### Create Environment

```bash
# Create conda environment using environment.yml
conda env create -f environment.yml

# Activate environment
conda activate Qwen3_2
```

### Environment Details

- **Environment Name**: `Qwen3_2`
- **Python Version**: 3.11.0
- **Main Dependencies**: 
  - PyTorch and related deep learning libraries
  - Transformers
  - Audio processing libraries (librosa, soundfile, etc.)
  - Other dependencies please refer to `environment.yml` file

For detailed dependency list, please check the `environment.yml` file in the project root directory.

### Compile SparseMM Operators

This project requires compiling SparseMM CUDA operators. Please refer to the SparseMM repository to configure the operators:

```bash
# Navigate to SparseMM csrc directory
cd /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/SparseMM/csrc

# Compile CUDA operators
make

# Or use Python setup
pip install packaging torch
pip uninstall ninja && pip cache purge && pip install ninja --no-cache-dir
python setup.py build_ext --inplace
```

**Note**: 
- If you encounter CUDA compilation errors, please check your GPU Virtual Architecture and GPU Features
- You may need to modify the compilation flags in `csrc/build.py` according to your GPU architecture
- After compilation, the operators will be automatically called by the project code

## ⭐ Core Methods Introduction

This project implements multiple KV Cache compression methods, with the **core methods** including:

### 1. SparseKV - Head Score-based Adaptive KV Cache Compression

**SparseKV** is the core method of this project, specifically designed for audio understanding tasks.

#### Core Concept

SparseKV achieves adaptive KV Cache allocation through **pre-computed Head Scores**:

- **Head Score Mechanism**: Computes importance scores for each attention head, with higher-scoring heads receiving more KV Cache budget
- **Adaptive Allocation**: Dynamically allocates KV Cache capacity for each head based on Head Score, with important heads retaining more information
- **Audio-Aware**: Identifies the most critical attention heads for audio understanding by analyzing attention patterns in audio tasks

#### How It Works

1. **Pre-computation Phase**: Use tools in the `audio_head/` directory to analyze the model's attention patterns on audio tasks and generate Head Score files
2. **Runtime Phase**: 
   - Allocate different KV Cache capacities for each head based on Head Score
   - Compute importance of each token using attention weights
   - Select KV Cache to retain based on importance scores

#### Advantages

- ✅ **Task-Specific**: Optimized for audio understanding tasks, outperforming general methods
- ✅ **Adaptive**: Automatically adjusts based on Head Score, no manual parameter tuning required
- ✅ **Efficient**: Significantly reduces memory usage while maintaining performance

### 2. FFT Spectral Filtering - Core Enhancement in SparseKV+FFT

**FFT Spectral Filtering** is the core enhancement technique in the SparseKV+FFT method, improving the accuracy of attention weight computation through frequency domain analysis.

#### Core Concept

FFT filtering transforms attention weights from time domain to frequency domain, utilizing spectral characteristics for filtering and enhancement:

- **Frequency Domain Analysis**: Uses FFT to transform attention weight sequences to frequency domain
- **Spectral Filtering**: Preserves or suppresses specific frequency components through lowpass/highpass filters
- **Time Domain Reconstruction**: Converts filtered spectrum back to time domain to obtain smoothed attention weights

#### How It Works

1. **FFT Transform**: Perform FFT transform on attention weight sequences to obtain spectral representation
2. **Spectral Filtering**: 
   - **Lowpass Filtering**: Preserves low-frequency components, smooths attention weights, suitable for capturing long-term dependencies
   - **Highpass Filtering**: Preserves high-frequency components, highlights local changes
   - **Adaptive Cutoff**: Dynamically adjusts cutoff frequency based on sequence characteristics
3. **IFFT Reconstruction**: Convert filtered spectrum back to time domain through IFFT
4. **Mixing Strategy**: Mix original attention weights with filtered weights proportionally

#### Advantages

- ✅ **Spectrum-Aware**: Better understands attention patterns using frequency domain characteristics
- ✅ **Smoothing Enhancement**: Reduces noise through filtering, improving attention weight quality
- ✅ **Audio-Adapted**: Particularly suitable for spectral characteristics of audio signals

#### Integration with SparseKV

SparseKV+FFT combines both techniques:
- **SparseKV**: Provides Head Score-based adaptive KV Cache allocation
- **FFT Filtering**: Performs spectral enhancement when computing attention weights, improving selection accuracy

This combination achieves the best performance on audio understanding tasks.

## 🎯 Supported Methods

| Method | Environment Variable | Description | Requires Head Score | FFT Support |
|--------|---------------------|-------------|---------------------|------------|
| **SparseKV** | `KV_METHOD="sparsekv"` | Head score-based adaptive KV cache compression | ✅ | ✅ |
| **SnapKV** | `KV_METHOD="snapkv"` | Attention weight-based KV cache compression | ❌ | ✅ |
| **AdaKV** | `KV_METHOD="adakv"` | Adaptive KV cache allocation | ❌ | ❌ |
| **PyramidKV** | `KV_METHOD="pyramidkv"` | Pyramid-shaped KV cache compression | ❌ | ❌ |
| **MixKV** | `KV_METHOD="mixkv"` | SparseKV + MixKV hybrid strategy KV cache compression | ✅ | ❌ |
| **SnapKV_FFT** | `KV_METHOD="sparsekv"` + `HEAD_SCORE_FILE="uniform"` + `USE_FFT_FILTER="true"` | SparseKV (uniform) + FFT spectral filtering | ❌ | ✅ |
| **SparseKV_FFT** | `KV_METHOD="sparsekv"` + `USE_FFT_FILTER="true"` | SparseKV + FFT spectral filtering | ✅ | ✅ |

## 🚀 Quick Start

### 1. SparseKV

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/path/to/model"
export SAMPLE_LIMIT=0

# SparseKV configuration
export KV_METHOD="sparsekv"
export BUDGET=0.4                          # 40% compression rate
export RATIO=0.5                          
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"

# Head Score configuration (required)
export HEAD_SCORE_PATH="/path/to/audio_head_scores.json"
export HEAD_SCORE_AGGREGATION="first"      # first/mean/max/min

# Run script
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

# SnapKV configuration
export KV_METHOD="snapkv"
export BUDGET=0.4
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"

# Run script
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

# AdaKV configuration
export KV_METHOD="adakv"
export BUDGET=0.4
export RATIO=0.2                           # floor_alpha parameter
export WINDOW_SIZE=32
export KERNEL_SIZE=7                       # AdaKV defaults to 7
export POOLING="maxpool"

# Run script
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

# PyramidKV configuration
export KV_METHOD="pyramidkv"
export BUDGET=0.4
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="avgpool"                   # PyramidKV defaults to avgpool

# Run script
source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/sparsekv_fft_dataset && \
python code/qwen3_kespeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir ./results/PyramidKV/budget0.4
```

### 5. MixKV (SparseKV+MixKV)

**Note**: MixKV is a hybrid method combining SparseKV + MixKV.

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/path/to/model"
export SAMPLE_LIMIT=0

# MixKV configuration (based on SparseKV + MixKV)
export KV_METHOD="sparsekv"               # Base method: use SparseKV
export METHOD="mixsparsemm"               # Specify MixedSparseMM (SparseKV base + MixKV scoring)
export BUDGET=0.4
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"

# Head Score configuration (use uniform distribution, no head score file needed)
export HEAD_SCORE_FILE="uniform"          # Use uniform distribution

# FFT filter configuration (enable FFT enhancement)
export USE_FFT_FILTER="true"              # Enable FFT filtering
export FFT_FILTER_TYPE="lowpass"          # Filter type: lowpass/highpass/saliency
export FFT_CUTOFF_RATIO="0.7"             # Cutoff frequency ratio
export FFT_ALPHA="0.5"                    # Mixing coefficient
export FFT_ADAPTIVE_CUTOFF="false"        # Adaptive cutoff frequency
export FFT_USE_SALIENCY="false"           # Whether to use saliency detection

# MixKV-specific configuration (fixed)
export SELECT_METHOD="headwisemixkv"      # Fixed to headwisemixkv (head-based hybrid strategy)

# Run script
source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/sparsekv_fft_dataset && \
python code/qwen3_kespeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir ./results/MixKV/budget0.4
```

## 🔧 Detailed Configuration

### Common Environment Variables

All methods share the following environment variables:

```bash
# Basic configuration
export CUDA_VISIBLE_DEVICES=0              # GPU device
export MODEL_PATH="/path/to/model"         # Model path
export SAMPLE_LIMIT=0                      # Sample limit (0=all)
                                           # Note: Only CoVoST2 en_zh task uses sample limit (SAMPLE_LIMIT=4898), other tasks default to all samples

# KV Cache compression configuration
export KV_METHOD="sparsekv"                # Method selection: sparsekv/snapkv/adakv/pyramidkv/mixkv
export BUDGET=0.4                          # Compression budget (<1.0 means ratio, >=1 means absolute value)
export RATIO=0.5                           # Minimum cache ratio
export WINDOW_SIZE=32                      # Window size
export KERNEL_SIZE=5                       # Convolution kernel size
export POOLING="maxpool"                   # Pooling method (maxpool/avgpool)
```

### SparseKV-Specific Configuration

```bash
# Head Score configuration (required)
export HEAD_SCORE_PATH="/path/to/audio_head_scores.json"
export HEAD_SCORE_AGGREGATION="first"      # first/mean/max/min

# Or use uniform distribution (no head score file needed)
export HEAD_SCORE_FILE="uniform"
```

### MixKV-Specific Configuration

**Note**: MixKV is a hybrid method combining SparseKV + MixKV, requires Head Score file.

```bash
# Base method configuration (required)
export KV_METHOD="sparsekv"               # Base method: use SparseKV
export METHOD="mixsparsemm"               # Specify MixedSparseMM (SparseKV base + MixKV scoring)

# Head Score configuration (required, from SparseKV base method)
export HEAD_SCORE_PATH="/path/to/audio_head_scores.json"
export HEAD_SCORE_AGGREGATION="first"      # first/mean/max/min

# MixKV selection method (fixed configuration)
export SELECT_METHOD="headwisemixkv"      # Fixed to headwisemixkv (head-based hybrid strategy)
                                           # Note: This parameter is fixed to headwisemixkv in current implementation
                                           # Other optional values (currently unused):
                                           # - attn: Based on attention weights
                                           # - keydiff: Based on key differences
                                           # - mixkv: Hybrid strategy
                                           # - knorm: Based on key norm
                                           # - vnorm: Based on value norm
```

### Budget Parameter Explanation

The `BUDGET` parameter supports two modes:

1. **Ratio Mode** (`BUDGET < 1.0`): Calculated as a ratio of prefill token count
   ```bash
   export BUDGET=0.4  # Retain 40% of KV cache
   ```

2. **Absolute Value Mode** (`BUDGET >= 1`): Fixed token count
   ```bash
   export BUDGET=4096  # Fixed retention of 4096 tokens
   ```

## 🎵 FFT Filter Enhancement

The following KV Cache methods support FFT spectral filtering enhancement, enabled by setting `USE_FFT_FILTER="true"`:
- SparseKV
- SnapKV

**Note**: AdaKV, MixKV, and PyramidKV currently do not support FFT filtering.

### SnapKV_FFT

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/path/to/model"
export SAMPLE_LIMIT=4898                  # CoVoST2 en_zh task uses sample limit (0=all)

# SparseKV base configuration (use uniform distribution, no head score file needed)
export KV_METHOD="sparsekv"
export BUDGET=0.8
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export HEAD_SCORE_FILE="uniform"           # Use uniform distribution, no head score file needed

# FFT filter configuration
export USE_FFT_FILTER="true"               # Enable FFT filtering
export FFT_FILTER_TYPE="lowpass"           # Filter type: lowpass/highpass/saliency
export FFT_CUTOFF_RATIO="0.7"              # Cutoff frequency ratio
export FFT_ALPHA="0.5"                     # Mixing coefficient
export FFT_ADAPTIVE_CUTOFF="false"         # Adaptive cutoff frequency
export FFT_USE_SALIENCY="false"            # Whether to use saliency detection

# Run script
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

# SparseKV base configuration
export KV_METHOD="sparsekv"
export BUDGET=0.8
export RATIO=0.5
export WINDOW_SIZE=32
export KERNEL_SIZE=5
export POOLING="maxpool"
export HEAD_SCORE_PATH="/path/to/audio_head_scores.json"
export HEAD_SCORE_AGGREGATION="first"

# FFT filter configuration
export USE_FFT_FILTER="true"
export FFT_FILTER_TYPE="lowpass"
export FFT_CUTOFF_RATIO="0.7"
export FFT_ALPHA="0.5"
export FFT_ADAPTIVE_CUTOFF="false"
export FFT_USE_SALIENCY="false"

# Run script
source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh
conda activate Qwen3_2 && \
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/sparsekv_fft_dataset && \
python code/qwen3_kespeech.py \
    --model-path "${MODEL_PATH}" \
    --output-dir ./results/SparseKV_FFT/budget0.8
```

### FFT Filter Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `USE_FFT_FILTER` | Whether to enable FFT filtering | `"false"` | `"true"` / `"false"` |
| `FFT_FILTER_TYPE` | Filter type | `"lowpass"` | `"lowpass"` / `"highpass"` / `"saliency"` |
| `FFT_CUTOFF_RATIO` | Cutoff frequency ratio | `"0.7"` | `0.0` - `1.0` |
| `FFT_ALPHA` | Mixing coefficient | `"0.5"` | `0.0` - `1.0` |
| `FFT_ADAPTIVE_CUTOFF` | Adaptive cutoff frequency | `"false"` | `"true"` / `"false"` |
| `FFT_USE_SALIENCY` | Use saliency detection | `"false"` | `"true"` / `"false"` |

**Note**: FFT filtering can be combined with supported methods (SparseKV, SnapKV) by adding FFT-related environment variables to the corresponding method's configuration. AdaKV, MixKV, and PyramidKV do not support FFT filtering.

## 📁 Directory Structure

```
Audio_kvcache/
├── qwen2_5/                          # Qwen2.5-Omni model
│   └── sparsekv_fft_dataset/
│       ├── code/                     # Code files
│       └── scripts/                  # Run scripts
│           ├── ASR/                  # Automatic Speech Recognition tasks
│           ├── ST/                   # Speech Translation tasks
│           └── KV_METHODS_README.md # Detailed documentation
├── qwen3/                            # Qwen3-Omni model
│   └── sparsekv_fft_dataset/
│       ├── code/
│       └── scripts/
│           ├── ASR/
│           ├── ST/
│           └── audio_qa/
├── gemma/                            # Gemma model
│   └── sparsekv_fft_dataset/
│       ├── code/
│       └── scripts/
├── audio_head/                       # Head Score computation tools
│   ├── qwen2.5_omni/
│   ├── qwen3_omni/
│   └── gemma/
└── README.md                         # This document
```

## 📝 Usage Examples

### Example 1: Run KeSpeech ASR Task (SparseKV)

```bash
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/sparsekv_fft_dataset/scripts/ASR
bash run_kespeech.sh
```

### Example 2: Run CoVoST2 Speech Translation Task (SnapKV_FFT)

**Note**: **Only the CoVoST2 en_zh task uses sample limit** `SAMPLE_LIMIT=4898`, all other tasks default to using all samples (`SAMPLE_LIMIT=0`). To use all samples, modify `SAMPLE_LIMIT=0` in the script.

```bash
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen3/sparsekv_fft_dataset/scripts/ST
bash run_covost2_en_zh.sh
```

## 🔍 Method Comparison

| Method | Requires Head Score | Computational Complexity | Memory Efficiency | Recommended Scenario |
|--------|---------------------|-------------------------|-------------------|---------------------|
| **SparseKV** | ✅ | Medium | Medium | Audio understanding tasks (recommended) |
| **SnapKV** | ❌ | Low | High | General scenarios |
| **AdaKV** | ❌ | Medium | Medium | Requires adaptive allocation |
| **PyramidKV** | ❌ | Low | High | Long sequence processing |
| **MixKV** | ✅ | High | Medium | SparseKV + MixKV hybrid strategy (requires base method parameters) |
| **SnapKV_FFT** | ❌ | Medium | High | Audio tasks + spectral enhancement |
| **SparseKV_FFT** | ✅ | Medium | Medium | Audio tasks + spectral enhancement (recommended) |

## 🐛 FAQ

### Q1: How to generate Head Score files?

A: Use tools in the `audio_head/` directory:

```bash
cd audio_head/qwen3_omni
bash run_word_attention_analysis_E2B.sh
```

### Q2: What's the difference between FFT filtering and regular methods?

A: FFT filtering uses frequency domain analysis to enhance attention weight computation, typically improving performance on audio tasks. It can be enabled by setting `USE_FFT_FILTER="true"`.

### Q3: How to choose the appropriate method?

A: 
- **Audio understanding tasks**: Recommended to use SparseKV or SparseKV_FFT
- **General tasks**: Recommended to use SnapKV or SnapKV_FFT
- **Long sequences**: Recommended to use PyramidKV
- **Requires hybrid strategy**: Use MixKV (SparseKV + MixKV hybrid method, requires Head Score)

### Q4: How to set the Budget parameter?

A: 
- Start with 0.8 (retain 80% of KV cache)
- Adjust based on task performance and memory constraints
- Audio tasks typically can use budget of 0.6-0.8

### Q5: How to verify if the method is working?

A: Check the log output, you should see information like:
```
[SnapKV] SnapKV attention enabled successfully
[SparseMM FFT] Layer 0: Using FFT spectral filtering with filter_type=lowpass, cutoff_ratio=0.7, alpha=0.5
```

## 📚 Additional Resources

- For detailed method descriptions and parameter configurations, please refer to `scripts/KV_METHODS_README.md` in each model directory
- For Head Score computation tool usage, please refer to documentation in the `audio_head/` directory

## 📄 License

Please refer to the LICENSE file in the project root directory.
