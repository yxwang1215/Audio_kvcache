"""
Alternative KV importance score smoothers for jV7P-Q2 rebuttal ablations.

Set SCORE_SMOOTH_METHOD to one of:
  none | raw          - no smoothing before top-k
  sss | fft          - spectral low-pass (SSS; same as USE_FFT_FILTER=true)
  maxpool | avgpool  - local Conv1d pooling (USE_FFT_FILTER=false default)
  gaussian           - 1D Gaussian convolution along the score sequence
  uniform_stride     - block mean with stride, upsampled (uniform stride blending)
  uniform_prior      - additive / convex mix with uniform 1/L prior

Legacy: if SCORE_SMOOTH_METHOD is unset, USE_FFT_FILTER=true -> sss, else pooling.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F


def _resolve_method(explicit: Optional[str], pooling: str) -> str:
    if explicit:
        return explicit.strip().lower()
    if os.environ.get("USE_FFT_FILTER", "false").lower() == "true":
        return "sss"
    return pooling.lower()


def gaussian_score_smoothing(
    scores: torch.Tensor,
    sigma: float,
    alpha: float,
    kernel_size: int,
) -> torch.Tensor:
    """Depthwise 1D Gaussian blur; sigma defaults from kernel_size if <= 0."""
    if scores.dim() == 2:
        scores = scores.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    if sigma <= 0:
        sigma = max(float(kernel_size) / 4.0, 0.5)

    radius = max(1, int(math.ceil(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, device=scores.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (xs / sigma) ** 2)
    kernel = kernel / kernel.sum()
    weight = kernel.view(1, 1, -1).expand(scores.shape[1], 1, -1).contiguous()

    orig = scores
    x = scores.to(torch.float32)
    pad = radius
    smoothed = F.conv1d(x, weight, padding=pad, groups=scores.shape[1])
    smoothed = smoothed[..., : orig.shape[-1]]
    out = (1.0 - alpha) * orig + alpha * smoothed.to(orig.dtype)
    return out.squeeze(0) if squeeze else out


def uniform_stride_blend(
    scores: torch.Tensor,
    stride: int,
    alpha: float,
) -> torch.Tensor:
    """Average within each stride block and broadcast (uniform stride blending)."""
    if scores.dim() == 2:
        scores = scores.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    orig = scores
    b, h, seq_len = scores.shape
    stride = max(1, int(stride))
    pad_len = (stride - seq_len % stride) % stride
    if pad_len:
        pad = torch.zeros(b, h, pad_len, device=scores.device, dtype=scores.dtype)
        padded = torch.cat([scores, pad], dim=-1)
    else:
        padded = scores

    n_blocks = padded.shape[-1] // stride
    blocks = padded.view(b, h, n_blocks, stride).mean(dim=-1)
    upsampled = blocks.repeat_interleave(stride, dim=-1)[..., :seq_len]
    out = (1.0 - alpha) * orig + alpha * upsampled
    return out.squeeze(0) if squeeze else out


def uniform_prior_smoothing(
    scores: torch.Tensor,
    alpha: float,
    additive: bool = True,
) -> torch.Tensor:
    """
    Additive uniform prior: s' = s + alpha * (1/L)
    or convex mix: s' = (1-alpha)*s + alpha*(1/L) when SMOOTH_PRIOR_MODE=convex
    """
    if scores.dim() == 2:
        scores = scores.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    seq_len = scores.shape[-1]
    uniform = torch.full(
        (1, 1, seq_len), 1.0 / max(seq_len, 1), device=scores.device, dtype=scores.dtype
    )
    mode = os.environ.get("SMOOTH_PRIOR_MODE", "additive").lower()
    if mode == "convex":
        out = (1.0 - alpha) * scores + alpha * uniform
    elif additive:
        out = scores + alpha * uniform
    else:
        out = (1.0 - alpha) * scores + alpha * uniform
    return out.squeeze(0) if squeeze else out


def apply_kv_score_smoothing(
    attn_weights_mean: torch.Tensor,
    *,
    kernel_size: int,
    pooling: str = "maxpool",
    method: Optional[str] = None,
) -> torch.Tensor:
    """Dispatch score smoothing for SparseMM KV selection."""
    m = _resolve_method(method, pooling)

    alpha = float(os.environ.get("FFT_ALPHA", "0.5"))
    cutoff = float(os.environ.get("FFT_CUTOFF_RATIO", "0.3"))
    filter_type = os.environ.get("FFT_FILTER_TYPE", "lowpass")
    use_saliency = os.environ.get("FFT_USE_SALIENCY", "false").lower() == "true"
    adaptive = os.environ.get("FFT_ADAPTIVE_CUTOFF", "true").lower() == "true"
    sigma = float(os.environ.get("GAUSSIAN_SIGMA", "0"))
    stride = int(os.environ.get("STRIDE_BLEND", str(kernel_size)))

    if m in ("none", "raw", "identity"):
        return attn_weights_mean
    if m in ("sss", "fft", "spectral", "lowpass"):
        try:
            from .sparsemm_utils import spectral_attention_smoothing
        except ImportError:
            from sparsemm_utils import spectral_attention_smoothing

        return spectral_attention_smoothing(
            attn_weights_mean,
            filter_type=filter_type if m not in ("sss", "fft", "spectral") else "lowpass",
            cutoff_ratio=cutoff,
            alpha=alpha,
            use_saliency=use_saliency,
            adaptive_cutoff=adaptive,
        )
    if m == "highpass":
        try:
            from .sparsemm_utils import spectral_attention_smoothing
        except ImportError:
            from sparsemm_utils import spectral_attention_smoothing

        return spectral_attention_smoothing(
            attn_weights_mean,
            filter_type="highpass",
            cutoff_ratio=cutoff,
            alpha=alpha,
            use_saliency=False,
            adaptive_cutoff=adaptive,
        )
    if m == "maxpool":
        return F.max_pool1d(
            attn_weights_mean,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )
    if m == "avgpool":
        return F.avg_pool1d(
            attn_weights_mean,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )
    if m == "gaussian":
        return gaussian_score_smoothing(attn_weights_mean, sigma, alpha, kernel_size)
    if m in ("uniform_stride", "stride_blend", "stride"):
        return uniform_stride_blend(attn_weights_mean, stride, alpha)
    if m in ("uniform_prior", "uniform", "prior"):
        return uniform_prior_smoothing(attn_weights_mean, alpha)
    raise ValueError(
        f"Unknown SCORE_SMOOTH_METHOD={m!r}. "
        "Use none|sss|maxpool|avgpool|gaussian|uniform_stride|uniform_prior|highpass."
    )
