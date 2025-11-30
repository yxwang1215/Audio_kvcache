# FFT TopK Selection Visualization

This module provides functionality to analyze and visualize the differences in topk selection before and after FFT filtering.

## Features

1. **Statistics Collection**: Automatically collects attention scores and topk indices before and after FFT filtering for each layer and head.

2. **Analysis Metrics**:
   - Jaccard similarity between before/after topk selections
   - Overlap ratio
   - Unique tokens in each selection
   - Budget analysis

3. **Visualizations** (ACL-style):
   - Layer-head comparison plots
   - Attention score comparison plots
   - Statistics summary plots

## Usage

### Running with FFT Statistics Collection

The statistics collection is automatically enabled when `USE_FFT_FILTER=true` is set. Run your script as usual:

```bash
conda activate Qwen3_2 &&
CUDA_VISIBLE_DEVICES=7 \
USE_SPARSEKV=true \
USE_FFT_FILTER=true \
FFT_FILTER_TYPE=saliency \
FFT_CUTOFF_RATIO=0.15 \
FFT_USE_SALIENCY=true \
FFT_ALPHA=0.6 \
FFT_ADAPTIVE_CUTOFF=true \
BUDGET=0.5 \
RATIO=0.5 \
WINDOW_SIZE=32 \
HEAD_SCORE_FILE=aggregated \
RESULTS_DIR=/path/to/results \
python LibriSpeech_qwen2.5.py
```

### Output Files

After running, you will find:

1. **Statistics JSON**: `{RESULTS_DIR}/fft_topk_stats.json`
   - Contains all collected statistics for analysis

2. **Visualization Directory**: `{RESULTS_DIR}/fft_visualizations/`
   - `statistics_summary.png`: Overall statistics distribution
   - `layer_head_comparison.png`: Comparison for each layer
   - `attention_comparison_layer{X}_head{Y}.png`: Detailed attention score plots

### Manual Visualization

You can also run the visualization script manually:

```bash
python visualize_fft_topk.py \
    --stats-file /path/to/fft_topk_stats.json \
    --output-dir /path/to/output \
    --max-samples 5
```

## Statistics Structure

The statistics JSON file contains data organized by layer and head:

```json
{
  "layer_0_head_0": {
    "before_fft_scores": [...],
    "after_fft_scores": [...],
    "before_fft_topk_indices": [...],
    "after_fft_topk_indices": [...],
    "head_budgets": [...],
    "sample_ids": [...]
  },
  ...
}
```

## Visualization Types

### 1. Statistics Summary
- Distribution of Jaccard similarities
- Distribution of overlap ratios
- Unique token count comparison
- Budget vs similarity analysis

### 2. Layer-Head Comparison
- Per-layer comparison of all heads
- Shows Jaccard similarity, overlap ratio, and unique token counts
- Helps identify which layers/heads are most affected by FFT

### 3. Attention Score Comparison
- Detailed plots for selected layer-head combinations
- Shows attention scores before/after FFT
- Highlights selected topk positions
- Visualizes selection differences

## Notes

- Statistics are collected only when `USE_FFT_FILTER=true`
- Each sample's statistics are tracked separately
- The visualization uses ACL-style formatting (Times New Roman, high DPI)
- Plots are saved in PNG format with 300 DPI

