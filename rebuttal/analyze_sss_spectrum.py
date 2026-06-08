#!/usr/bin/env python3
"""
jV7P-Q2 diagnostic: quantify high-frequency energy in KV score traces and
compare top-k stability under SSS vs. simpler smoothers (offline, no GPU model).

Usage:
  python rebuttal/analyze_sss_spectrum.py \\
    --stats-json qwen2_5/sparsekv_fft_visualization/results/.../fft_topk_stats.json \\
    --output-json rebuttal/sss_spectrum_diagnostic.json

Requires: torch, numpy (same env as inference).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "qwen2_5" / "sparsekv_fft_dataset" / "model"
sys.path.insert(0, str(MODEL_DIR))

from score_smoothing import apply_kv_score_smoothing  # noqa: E402


def _band_energy(scores_1d: np.ndarray, cutoff_ratio: float = 0.7) -> Dict[str, float]:
    """Energy below vs above adaptive cutoff (same rule as SSS, gamma=cutoff_ratio)."""
    x = torch.tensor(scores_1d, dtype=torch.float32)
    if x.numel() < 8:
        return {
            "low_energy_frac": 1.0,
            "high_energy_frac": 0.0,
            "seq_len": float(x.numel()),
        }
    mag = torch.abs(torch.fft.rfft(x)).numpy()
    energy = mag**2
    total = float(energy.sum()) + 1e-12
    cum = np.cumsum(energy)
    thr = total * cutoff_ratio
    cutoff_idx = int(np.searchsorted(cum, thr))
    cutoff_idx = min(max(cutoff_idx, 1), len(energy) - 1)
    low_e = float(energy[:cutoff_idx].sum()) / total
    high_e = float(energy[cutoff_idx:].sum()) / total
    return {
        "low_energy_frac": low_e,
        "high_energy_frac": high_e,
        "cutoff_bin_frac": cutoff_idx / len(energy),
        "seq_len": float(x.numel()),
    }


def _topk_indices(scores_1d: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores_1d))
    if k <= 0:
        return np.array([], dtype=np.int64)
    return np.argpartition(scores_1d, -k)[-k:]


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def _smooth_numpy(
    scores: np.ndarray,
    method: str,
    *,
    kernel_size: int = 5,
    alpha: float = 0.5,
    cutoff: float = 0.7,
) -> np.ndarray:
    """Apply one smoother to [seq_len] scores; returns numpy 1d."""
    os.environ["FFT_ALPHA"] = str(alpha)
    os.environ["FFT_CUTOFF_RATIO"] = str(cutoff)
    os.environ["FFT_ADAPTIVE_CUTOFF"] = "false"
    t = torch.tensor(scores, dtype=torch.float32).unsqueeze(0)  # [1, seq]
    if method == "none":
        out = t
    elif method == "highpass":
        os.environ["USE_FFT_FILTER"] = "true"
        out = apply_kv_score_smoothing(t, kernel_size=kernel_size, pooling="maxpool", method="highpass")
    else:
        if method in ("sss", "fft"):
            os.environ["USE_FFT_FILTER"] = "true"
        else:
            os.environ["USE_FFT_FILTER"] = "false"
        out = apply_kv_score_smoothing(t, kernel_size=kernel_size, pooling="maxpool", method=method)
    return out.squeeze(0).detach().cpu().numpy()


def analyze_stats(
    stats: Dict[str, Any],
    *,
    max_samples_per_head: int = 20,
    topk_frac: float = 0.1,
    alpha: float = 0.5,
    cutoff: float = 0.7,
    kernel_size: int = 5,
) -> Dict[str, Any]:
    methods = ["none", "maxpool", "gaussian", "uniform_stride", "uniform_prior", "sss", "highpass"]
    band_acc = defaultdict(list)
    jaccard_vs_none = defaultdict(list)
    jaccard_sss_vs_alt = defaultdict(list)
    n_traces = 0

    for _key, data in stats.items():
        before_list = data.get("before_fft_scores", [])
        budgets = data.get("head_budgets", [])
        for idx, scores_list in enumerate(before_list):
            if max_samples_per_head and idx >= max_samples_per_head:
                break
            scores = np.asarray(scores_list, dtype=np.float64)
            if scores.size < 16:
                continue
            n_traces += 1
            band = _band_energy(scores, cutoff_ratio=cutoff)
            for bk, bv in band.items():
                if bk != "seq_len":
                    band_acc[bk].append(bv)

            budget = int(budgets[idx]) if idx < len(budgets) and budgets[idx] else max(1, int(scores.size * topk_frac))
            k = max(1, min(budget, scores.size - 1))
            topk_raw = _topk_indices(scores, k)

            smoothed = {}
            for m in methods:
                smoothed[m] = _smooth_numpy(scores, m, kernel_size=kernel_size, alpha=alpha, cutoff=cutoff)

            for m in methods:
                if m == "none":
                    continue
                topk_m = _topk_indices(smoothed[m], k)
                jaccard_vs_none[m].append(_jaccard(topk_raw, topk_m))

            topk_sss = _topk_indices(smoothed["sss"], k)
            for m in ("maxpool", "gaussian", "uniform_stride", "uniform_prior"):
                topk_m = _topk_indices(smoothed[m], k)
                jaccard_sss_vs_alt[m].append(_jaccard(topk_sss, topk_m))

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    return {
        "n_score_traces": n_traces,
        "spectral_energy": {k: _mean(v) for k, v in band_acc.items()},
        "topk_jaccard_vs_unsmoothed": {k: _mean(v) for k, v in jaccard_vs_none.items()},
        "topk_jaccard_sss_vs_simpler": {k: _mean(v) for k, v in jaccard_sss_vs_alt.items()},
        "interpretation": {
            "high_freq_noise": (
                "The adaptive SSS cutoff captures the target low-frequency energy mass "
                "and leaves a measurable high-frequency residual for controlled ablation."
            ),
            "sss_vs_simpler": (
                "Low overlap between SSS-selected tokens and simpler smoothers indicates "
                "that frequency-aware filtering is not equivalent to local pooling, Gaussian blur, "
                "or an additive uniform prior. End-to-end ablations determine whether the distinct "
                "ranking is beneficial."
            ),
        },
        "settings": {
            "alpha": alpha,
            "cutoff": cutoff,
            "kernel_size": kernel_size,
            "max_samples_per_head": max_samples_per_head,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SSS spectral diagnostic (jV7P-Q2)")
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--max-samples-per-head", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--cutoff", type=float, default=0.7)
    parser.add_argument("--kernel-size", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading {args.stats_json} ...")
    with open(args.stats_json, encoding="utf-8") as f:
        stats = json.load(f)

    report = analyze_stats(
        stats,
        max_samples_per_head=args.max_samples_per_head,
        alpha=args.alpha,
        cutoff=args.cutoff,
        kernel_size=args.kernel_size,
    )

    print(json.dumps(report, indent=2))
    out = args.output_json or Path(__file__).parent / "sss_spectrum_diagnostic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
