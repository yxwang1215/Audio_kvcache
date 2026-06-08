# jV7P-Q2: SSS Motivation vs. Simpler Smoothers

Reviewer concern: high-frequency score noise is asserted, not shown; no comparison to uniform stride blending, Gaussian smoothing, or additive uniform prior.

## Strategy (two pillars)

| Pillar | What it shows | Cost |
|--------|----------------|------|
| **A. Spectral diagnostic** (offline) | Quantifies low/high-frequency energy and shows SSS selects a different token set than local/implicit smoothers | Minutes (existing `fft_topk_stats.json`) |
| **B. Smoother ablation** (E2E) | Same SparseKV + head scores + budget; only swap smoother | ~5× one ASR run |

## A. Run spectral diagnostic (no GPU model)

Uses cached scores from `qwen2_5/sparsekv_fft_visualization/`.

```bash
conda activate Qwen3_2
python rebuttal/analyze_sss_spectrum.py \
  --stats-json qwen2_5/sparsekv_fft_visualization/results/LibriSpeech_budget0.5_fft_saliency_cutoff0.15_alpha0.6/fft_topk_stats.json \
  --output-json rebuttal/sss_spectrum_diagnostic.json \
  --alpha 0.5 --cutoff 0.7 --kernel-size 5
```

Copy means into `rebuttal/tex/jv7p_q2_sss_motivation.tex` Table 1.

Current cached diagnostic (`max_samples_per_head=5`) gives: low-frequency energy 0.700, high-frequency residual 0.300, cutoff bin fraction 0.660. SSS/raw Jaccard is 0.436, and SSS vs. maxpool/Gaussian/stride Jaccard is about 0.42-0.44. This supports the narrower claim that SSS is not equivalent to local smoothing; the E2E ablation below is what validates whether the distinct ranking is beneficial.

## B. Run end-to-end smoother ablation

New env: `SCORE_SMOOTH_METHOD` in `qwen2_5/sparsekv_fft_dataset/model/score_smoothing.py`.

| Method | Env |
|--------|-----|
| No smoothing | `SCORE_SMOOTH_METHOD=none` |
| SSS (paper) | `SCORE_SMOOTH_METHOD=sss` + `USE_FFT_FILTER=true` |
| MaxPool | `SCORE_SMOOTH_METHOD=maxpool` |
| Gaussian | `SCORE_SMOOTH_METHOD=gaussian` |
| Uniform stride | `SCORE_SMOOTH_METHOD=uniform_stride` |
| Uniform prior | `SCORE_SMOOTH_METHOD=uniform_prior` |
| High-pass control | `SCORE_SMOOTH_METHOD=highpass` (diagnostic only) |

```bash
# Full ASR sweep used for the rebuttal table:
GPU_ID=5 SAMPLE_LIMIT=50 BUDGET=0.4 bash rebuttal/run_sss_smoother_ablation.sh

# Equivalent explicit task list:
GPU_ID=5 SAMPLE_LIMIT=50 BUDGET=0.4 \
  TASKS="librispeech kespeech multilingual-spanish multilingual-french multilingual-german" \
  bash rebuttal/run_sss_smoother_ablation.sh

# Quick smoke test
SAMPLE_LIMIT=20 METHODS="sss maxpool gaussian" GPU_ID=5 bash rebuttal/run_sss_smoother_ablation.sh
```

Aggregate:

```bash
python rebuttal/aggregate_smoother_ablation.py \
  --root /mnt/users/wangyuxuan-20250915/Audio/Results/Qwen2_5_Omni_3B/ablation_jv7p_q2_smoothers \
  --tasks librispeech kespeech multilingual-spanish multilingual-french multilingual-german
```

Fill Table 2 in `rebuttal/tex/jv7p_q2_sss_motivation.tex`.

## Optional figure

Re-use `qwen2_5/sparsekv_fft_visualization/analyze_topk_distribution.py` on the same stats JSON for before/after top-k heatmaps (already in repo).

## Add to rebuttal.tex

```latex
\input{jv7p_q2_sss_motivation.tex}
```

## Expected outcome

- If **SSS ≈ Gaussian/maxpool** on WER → be conservative: claim SSS is a frequency-domain implementation with global receptive field, but do not overstate that frequency filtering is uniquely responsible.
- If **SSS > simpler smoothers** → strong answer: frequency filtering is not replaceable by local smoothing.
- **High-pass worse** than unsmoothed → direct evidence HF components are harmful for eviction scores.
