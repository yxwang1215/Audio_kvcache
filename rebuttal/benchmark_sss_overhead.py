#!/usr/bin/env python3
"""
Benchmark SSS (Spectral Score Smoothing) overhead vs. Conv1d pooling (w/o SSS).

Answers rebuttal concern VY3D-Q2: RFFT/IRFFT per head per decoding step.

Modes:
  1) micro  - isolate spectral_attention_smoothing vs maxpool1d (default)
  2) decode-sim - simulate T decode steps x L layers (score length ~ compressed cache)
  3) e2e    - run qwen2.5_librispeech.py twice (USE_FFT_FILTER true/false), compare timing JSON

Example:
  CUDA_VISIBLE_DEVICES=0 python benchmark_sss_overhead.py --mode micro
  CUDA_VISIBLE_DEVICES=0 python benchmark_sss_overhead.py --mode decode-sim --prefill-len 5000 --decode-tokens 128
  CUDA_VISIBLE_DEVICES=0 python benchmark_sss_overhead.py --mode e2e --sample-limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_MODEL_DIR = REPO_ROOT / "qwen2_5" / "sparsekv_fft_dataset" / "model"
sys.path.insert(0, str(ORIGINAL_MODEL_DIR))
from sparsemm_utils import spectral_attention_smoothing  # noqa: E402

# Qwen2.5-Omni-3B defaults (36 layers, GQA 16 -> 2 KV heads per layer)
DEFAULT_NUM_LAYERS = 36
DEFAULT_NUM_KV_HEADS = 2
DEFAULT_KERNEL_SIZE = 5
DEFAULT_CUTOFF = 0.7
DEFAULT_ALPHA = 0.5


@dataclass
class BenchRow:
    seq_len: int
    num_layers: int
    num_kv_heads: int
    with_sss_ms: float
    without_sss_ms: float
    overhead_ms: float
    overhead_pct: float
    per_layer_sss_us: float
    per_head_sss_us: float


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench_fn(fn: Callable[[], None], warmup: int, iters: int, device: torch.device) -> float:
    """Return average milliseconds per iteration."""
    for _ in range(warmup):
        fn()
    _cuda_sync()

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        _cuda_sync()
        return start.elapsed_time(end) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _cuda_sync()
    return (time.perf_counter() - t0) * 1000.0 / iters


def _make_scores(
    batch: int,
    num_kv_heads: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return torch.randn(batch, num_kv_heads, seq_len, device=device, dtype=dtype)


def _apply_sss(scores: torch.Tensor, cutoff: float, alpha: float) -> torch.Tensor:
    return spectral_attention_smoothing(
        scores,
        filter_type="lowpass",
        cutoff_ratio=cutoff,
        alpha=alpha,
        use_saliency=False,
        adaptive_cutoff=False,
    )


def _apply_maxpool(scores: torch.Tensor, kernel_size: int) -> torch.Tensor:
    # scores: [B, H, L] -> max_pool1d expects [B, C, L]
    return F.max_pool1d(
        scores,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        stride=1,
    )


def bench_single_layer(
    seq_len: int,
    num_kv_heads: int,
    device: torch.device,
    cutoff: float,
    alpha: float,
    kernel_size: int,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    scores = _make_scores(1, num_kv_heads, seq_len, device)

    def with_sss():
        _apply_sss(scores, cutoff, alpha)

    def without_sss():
        _apply_maxpool(scores, kernel_size)

    t_sss = _bench_fn(with_sss, warmup, iters, device)
    t_pool = _bench_fn(without_sss, warmup, iters, device)
    return t_sss, t_pool


def bench_all_layers_per_step(
    seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    device: torch.device,
    cutoff: float,
    alpha: float,
    kernel_size: int,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    """One decode-step worth of SSS: all layers, each with independent score tensor."""
    layers_scores = [_make_scores(1, num_kv_heads, seq_len, device) for _ in range(num_layers)]

    def with_sss():
        for s in layers_scores:
            _apply_sss(s, cutoff, alpha)

    def without_sss():
        for s in layers_scores:
            _apply_maxpool(s, kernel_size)

    t_sss = _bench_fn(with_sss, warmup, iters, device)
    t_pool = _bench_fn(without_sss, warmup, iters, device)
    return t_sss, t_pool


def run_micro(
    seq_lens: List[int],
    num_layers: int,
    num_kv_heads: int,
    device: torch.device,
    cutoff: float,
    alpha: float,
    kernel_size: int,
    warmup: int,
    iters: int,
) -> List[BenchRow]:
    rows: List[BenchRow] = []
    print("\n=== Micro-benchmark: original SSS implementation vs maxpool ===")
    print(f"Imported from: {ORIGINAL_MODEL_DIR / 'sparsemm_utils.py'}")
    print(f"{'seq_len':>8} {'w/ SSS(ms)':>12} {'w/o SSS(ms)':>12} {'overhead':>10} {'%':>8} {'us/head':>10}")
    print("-" * 72)

    for seq_len in seq_lens:
        t_sss, t_pool = bench_single_layer(
            seq_len, num_kv_heads, device, cutoff, alpha, kernel_size, warmup, iters
        )
        oh = t_sss - t_pool
        pct = 100.0 * oh / t_pool if t_pool > 0 else 0.0
        us_head = (t_sss * 1000.0) / max(num_kv_heads, 1)
        rows.append(
            BenchRow(
                seq_len=seq_len,
                num_layers=1,
                num_kv_heads=num_kv_heads,
                with_sss_ms=t_sss,
                without_sss_ms=t_pool,
                overhead_ms=oh,
                overhead_pct=pct,
                per_layer_sss_us=t_sss * 1000.0,
                per_head_sss_us=us_head,
            )
        )
        print(
            f"{seq_len:8d} {t_sss:12.4f} {t_pool:12.4f} {oh:10.4f} {pct:7.2f}% {us_head:10.2f}"
        )
    return rows


def run_decode_sim(
    prefill_len: int,
    retention: float,
    window: int,
    decode_tokens: int,
    num_layers: int,
    num_kv_heads: int,
    device: torch.device,
    cutoff: float,
    alpha: float,
    kernel_size: int,
    warmup: int,
    iters: int,
) -> dict:
    """
    Simulate decode: each step runs SSS on all layers.
    Score length grows during early decode then stabilizes near compressed budget.
    """
    budget = max(int(retention * prefill_len), window + 64)
    compressed_score_len = max(budget - window, 64)

    print("\n=== Decode simulation (all layers per decode step) ===")
    print(f"prefill_len={prefill_len}, retention={retention}, budget≈{budget}, "
          f"stable score_len≈{compressed_score_len}, decode_tokens={decode_tokens}, layers={num_layers}")

    # Per-step cost at stable cache length
    t_sss_step, t_pool_step = bench_all_layers_per_step(
        compressed_score_len,
        num_layers,
        num_kv_heads,
        device,
        cutoff,
        alpha,
        kernel_size,
        warmup,
        iters,
    )

    total_sss = t_sss_step * decode_tokens
    total_pool = t_pool_step * decode_tokens
    oh = total_sss - total_pool

    # Also report prefill-length score (worst case: first compression on full context)
    t_sss_prefill, t_pool_prefill = bench_all_layers_per_step(
        max(prefill_len - window, 64),
        num_layers,
        num_kv_heads,
        device,
        cutoff,
        alpha,
        kernel_size,
        max(1, warmup // 2),
        max(5, iters // 2),
    )

    result = {
        "prefill_len": prefill_len,
        "retention": retention,
        "window": window,
        "decode_tokens": decode_tokens,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "compressed_score_len": compressed_score_len,
        "per_decode_step_with_sss_ms": t_sss_step,
        "per_decode_step_without_sss_ms": t_pool_step,
        "per_decode_step_overhead_ms": t_sss_step - t_pool_step,
        "per_decode_step_overhead_pct": 100.0 * (t_sss_step - t_pool_step) / t_pool_step if t_pool_step else 0,
        "total_decode_sss_overhead_ms": oh,
        "prefill_all_layers_with_sss_ms": t_sss_prefill,
        "prefill_all_layers_without_sss_ms": t_pool_prefill,
        "sss_only_per_layer_per_step_us": (t_sss_step * 1000.0) / num_layers,
        "sss_only_per_head_per_step_us": (t_sss_step * 1000.0) / (num_layers * num_kv_heads),
    }

    print(f"Per decode step (L={num_layers}): w/ SSS {t_sss_step:.4f} ms, w/o {t_pool_step:.4f} ms, "
          f"overhead {result['per_decode_step_overhead_ms']:.4f} ms ({result['per_decode_step_overhead_pct']:.2f}%)")
    print(f"Total SSS overhead for {decode_tokens} decode tokens: {oh:.2f} ms")
    print(f"Prefill one-shot all layers (len={prefill_len - window}): w/ SSS {t_sss_prefill:.2f} ms, w/o {t_pool_prefill:.2f} ms")
    print(f"Per head per layer per step: {result['sss_only_per_head_per_step_us']:.2f} µs")

    return result


def _load_timing_summary(path: Path) -> dict:
    data = json.loads(path.read_text())
    if "global_summary" in data:
        return data["global_summary"]
    if "summary" in data:
        return data["summary"]
    return data


def _per_token_decode_ms(summary: dict) -> float:
    """Wall-clock ms per generated token from measured decode phase."""
    tps = summary.get("overall_tps", 0.0)
    if tps and tps > 0:
        return 1000.0 / tps
    total_tokens = summary.get("total_decode_tokens", 0)
    count = summary.get("count", 0)
    avg_decode = summary.get("avg_decode_time", 0.0)
    if total_tokens > 0 and count > 0:
        total_decode_s = avg_decode * count
        return 1000.0 * total_decode_s / total_tokens
    return 0.0


def build_sss_vs_decode_comparison(decode_sim: dict, e2e: dict) -> dict:
    """Compare isolated SSS operator cost against real end-to-end decode latency."""
    s0 = e2e.get("without_sss", {})
    s1 = e2e.get("with_sss", {})

    real_per_token_ms = _per_token_decode_ms(s0)
    real_per_token_with_sss_ms = _per_token_decode_ms(s1)

    isolated_overhead_ms = decode_sim.get("per_decode_step_overhead_ms", 0.0)
    isolated_sss_ms = decode_sim.get("per_decode_step_with_sss_ms", 0.0)
    isolated_pool_ms = decode_sim.get("per_decode_step_without_sss_ms", 0.0)

    e2e_extra_per_token_ms = 0.0
    if real_per_token_ms > 0 and real_per_token_with_sss_ms > 0:
        e2e_extra_per_token_ms = real_per_token_with_sss_ms - real_per_token_ms

    return {
        "real_decode_per_token_ms": real_per_token_ms,
        "real_decode_per_token_with_sss_ms": real_per_token_with_sss_ms,
        "isolated_sss_all_layers_per_step_ms": isolated_sss_ms,
        "isolated_maxpool_all_layers_per_step_ms": isolated_pool_ms,
        "isolated_sss_overhead_per_step_ms": isolated_overhead_ms,
        "isolated_overhead_fraction_of_real_decode": (
            isolated_overhead_ms / real_per_token_ms if real_per_token_ms > 0 else None
        ),
        "isolated_overhead_pct_of_real_decode": (
            100.0 * isolated_overhead_ms / real_per_token_ms if real_per_token_ms > 0 else None
        ),
        "e2e_extra_decode_per_token_ms": e2e_extra_per_token_ms,
        "e2e_extra_decode_pct_of_real_decode": (
            100.0 * e2e_extra_per_token_ms / real_per_token_ms if real_per_token_ms > 0 else None
        ),
        "e2e_decode_slowdown_pct": e2e.get("decode_slowdown_pct"),
        "e2e_tps_drop_pct": e2e.get("tps_drop_pct"),
        "interpretation": (
            "isolated_* uses a stress test (SSS on all layers every decode step); "
            "e2e_* is the measured wall-clock decode delta with identical ASR settings."
        ),
    }


def print_sss_vs_decode_comparison(comparison: dict, decode_sim: dict) -> None:
    print("\n=== SSS overhead vs. real decode (per generated token) ===")
    real_ms = comparison.get("real_decode_per_token_ms", 0.0)
    iso_oh = comparison.get("isolated_sss_overhead_per_step_ms", 0.0)
    e2e_extra = comparison.get("e2e_extra_decode_per_token_ms", 0.0)

    print(f"Real decode (E2E, w/o SSS):              {real_ms:.2f} ms/token  "
          f"(from CUDA-timed generate)")
    print(f"Real decode (E2E, w/ SSS):               "
          f"{comparison.get('real_decode_per_token_with_sss_ms', 0):.2f} ms/token")
    print(f"Isolated SSS stress (36 layers/step):     +{iso_oh:.2f} ms  "
          f"({comparison.get('isolated_overhead_pct_of_real_decode', 0):.1f}% of real decode)")
    print(f"E2E wall-clock SSS effect on decode:     +{e2e_extra:.2f} ms/token  "
          f"({comparison.get('e2e_extra_decode_pct_of_real_decode', 0):.1f}% of real decode)")
    print(f"E2E avg decode/sample slowdown:           "
          f"{comparison.get('e2e_decode_slowdown_pct', 0):.2f}%")
    print(f"Decode-sim score_len (stable):            {decode_sim.get('compressed_score_len')}")
    print(
        "\nUse isolated row as operator upper bound; use E2E row for actual ASR decode impact."
    )


def _run_e2e_inference(
    *,
    use_fft: bool,
    sample_limit: int,
    output_dir: Path,
    model_path: str,
    gpu: str,
    budget: float,
    conda_env: Optional[str],
    e2e_backend: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "with_sss" if use_fft else "without_sss"
    timing_file = output_dir / f"timing_{tag}.json"
    results_file = output_dir / f"results_{tag}.jsonl"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["MODEL_PATH"] = model_path
    env["SAMPLE_LIMIT"] = str(sample_limit)
    env["USE_FFT_FILTER"] = "true" if use_fft else "false"
    env["FFT_CUTOFF_RATIO"] = str(DEFAULT_CUTOFF)
    env["FFT_ALPHA"] = str(DEFAULT_ALPHA)
    env["FFT_ADAPTIVE_CUTOFF"] = "false"
    env["BUDGET"] = str(budget)
    env["KV_METHOD"] = "snapkv"
    env["WINDOW_SIZE"] = "32"
    env["KERNEL_SIZE"] = str(DEFAULT_KERNEL_SIZE)
    env["POOLING"] = "maxpool"
    env["HEAD_SCORE_FILE"] = "uniform"

    if e2e_backend == "kespeech":
        script = REPO_ROOT / "qwen2_5" / "sparsekv_fft_dataset" / "code" / "qwen2.5_kespeech.py"
        backend_label = "KeSpeech (CUDA prefill/decode timing)"
    else:
        script = REPO_ROOT / "qwen2_5" / "sparsekv_fft_dataset" / "code" / "qwen2.5_librispeech.py"
        backend_label = "LibriSpeech-long"
    print(f"    E2E backend: {backend_label}")
    cmd = [
        sys.executable,
        str(script),
        "--model-path",
        model_path,
        "--output-dir",
        str(output_dir),
        "--output-file",
        str(results_file),
        "--timing-output-file",
        str(timing_file),
    ]

    print(f"\n>>> E2E run ({tag}): USE_FFT_FILTER={env['USE_FFT_FILTER']}")
    if conda_env:
        shell_cmd = (
            f"source /mnt/users/wangyuxuan-20250915/miniconda/etc/profile.d/conda.sh && "
            f"conda activate {conda_env} && "
            + " ".join(cmd)
        )
        subprocess.run(["bash", "-lc", shell_cmd], env=env, check=True, cwd=str(script.parent.parent))
    else:
        subprocess.run(cmd, env=env, check=True, cwd=str(script.parent.parent))

    return timing_file


def run_e2e(
    sample_limit: int,
    output_dir: Path,
    model_path: str,
    gpu: str,
    budget: float,
    conda_env: Optional[str],
    e2e_backend: str,
) -> dict:
    without = _run_e2e_inference(
        use_fft=False,
        sample_limit=sample_limit,
        output_dir=output_dir,
        model_path=model_path,
        gpu=gpu,
        budget=budget,
        conda_env=conda_env,
        e2e_backend=e2e_backend,
    )
    with_sss = _run_e2e_inference(
        use_fft=True,
        sample_limit=sample_limit,
        output_dir=output_dir,
        model_path=model_path,
        gpu=gpu,
        budget=budget,
        conda_env=conda_env,
        e2e_backend=e2e_backend,
    )

    s0 = _load_timing_summary(without)
    s1 = _load_timing_summary(with_sss)

    avg_decode_0 = s0.get("avg_decode_time", 0.0)
    avg_decode_1 = s1.get("avg_decode_time", 0.0)
    avg_wall_0 = s0.get("avg_wall_time", 0.0)
    avg_wall_1 = s1.get("avg_wall_time", 0.0)
    avg_prefill_0 = s0.get("avg_prefill_time", 0.0)
    avg_prefill_1 = s1.get("avg_prefill_time", 0.0)
    tps_0 = s0.get("overall_tps", 0.0)
    tps_1 = s1.get("overall_tps", 0.0)
    per_token_0 = _per_token_decode_ms(s0)
    per_token_1 = _per_token_decode_ms(s1)

    result = {
        "e2e_backend": e2e_backend,
        "sample_limit": sample_limit,
        "without_sss": s0,
        "with_sss": s1,
        "avg_wall_time_without_sss_s": avg_wall_0,
        "avg_wall_time_with_sss_s": avg_wall_1,
        "wall_time_extra_s": avg_wall_1 - avg_wall_0,
        "wall_time_slowdown_pct": 100.0 * (avg_wall_1 - avg_wall_0) / avg_wall_0 if avg_wall_0 else 0,
        "avg_decode_time_without_sss_s": avg_decode_0,
        "avg_decode_time_with_sss_s": avg_decode_1,
        "decode_slowdown_pct": 100.0 * (avg_decode_1 - avg_decode_0) / avg_decode_0 if avg_decode_0 else 0,
        "avg_prefill_time_without_sss_s": avg_prefill_0,
        "avg_prefill_time_with_sss_s": avg_prefill_1,
        "overall_tps_without_sss": tps_0,
        "overall_tps_with_sss": tps_1,
        "tps_drop_pct": 100.0 * (tps_0 - tps_1) / tps_0 if tps_0 else 0,
        "per_token_decode_ms_without_sss": per_token_0,
        "per_token_decode_ms_with_sss": per_token_1,
        "per_token_decode_extra_ms": per_token_1 - per_token_0,
    }

    print(f"\n=== E2E timing comparison ({e2e_backend}) ===")
    print(f"Avg wall time:    w/o SSS {avg_wall_0:.4f}s  |  w/ SSS {avg_wall_1:.4f}s  "
          f"(+{result['wall_time_extra_s']:.4f}s, +{result['wall_time_slowdown_pct']:.2f}%)")
    print(f"Avg decode time:  w/o SSS {avg_decode_0:.4f}s  |  w/ SSS {avg_decode_1:.4f}s  "
          f"(+{result['decode_slowdown_pct']:.2f}%)")
    print(f"Per-token decode: w/o SSS {per_token_0:.2f} ms  |  w/ SSS {per_token_1:.2f} ms  "
          f"(+{result['per_token_decode_extra_ms']:.2f} ms)")
    print(f"Avg prefill time: w/o SSS {avg_prefill_0:.4f}s  |  w/ SSS {avg_prefill_1:.4f}s")
    print(f"Overall TPS:      w/o SSS {tps_0:.2f}  |  w/ SSS {tps_1:.2f}  "
          f"(-{result['tps_drop_pct']:.2f}%)")

    return result


def load_e2e_from_ref_timing(
  ref_without: Path,
  ref_with: Path,
) -> dict:
    """Build e2e dict from existing timing JSON files (skip inference)."""
    s0 = _load_timing_summary(ref_without)
    s1 = _load_timing_summary(ref_with)
    per_token_0 = _per_token_decode_ms(s0)
    per_token_1 = _per_token_decode_ms(s1)
    avg_decode_0 = s0.get("avg_decode_time", 0.0)
    avg_decode_1 = s1.get("avg_decode_time", 0.0)
    tps_0 = s0.get("overall_tps", 0.0)
    tps_1 = s1.get("overall_tps", 0.0)
    return {
        "e2e_backend": "ref_timing_json",
        "ref_without": str(ref_without),
        "ref_with": str(ref_with),
        "without_sss": s0,
        "with_sss": s1,
        "avg_decode_time_without_sss_s": avg_decode_0,
        "avg_decode_time_with_sss_s": avg_decode_1,
        "decode_slowdown_pct": 100.0 * (avg_decode_1 - avg_decode_0) / avg_decode_0 if avg_decode_0 else 0,
        "overall_tps_without_sss": tps_0,
        "overall_tps_with_sss": tps_1,
        "tps_drop_pct": 100.0 * (tps_0 - tps_1) / tps_0 if tps_0 else 0,
        "per_token_decode_ms_without_sss": per_token_0,
        "per_token_decode_ms_with_sss": per_token_1,
        "per_token_decode_extra_ms": per_token_1 - per_token_0,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark SSS overhead (VY3D-Q2)")
    parser.add_argument(
        "--mode",
        choices=["micro", "decode-sim", "all", "e2e", "compare"],
        default="compare",
        help="compare=micro+decode-sim+e2e+ratio; e2e=real ASR decode timing only",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1000, 3000, 5000, 8000, 12000])
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--num-kv-heads", type=int, default=DEFAULT_NUM_KV_HEADS)
    parser.add_argument("--prefill-len", type=int, default=5000, help="For decode-sim")
    parser.add_argument("--retention", type=float, default=0.4, help="KV retention ratio rho")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--kernel-size", type=int, default=DEFAULT_KERNEL_SIZE)
    parser.add_argument("--output-json", type=str, default=None)
    # e2e
    parser.add_argument("--sample-limit", type=int, default=10)
    parser.add_argument("--e2e-output-dir", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="/mnt/public/models/Qwen2.5-Omni-3B")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--budget", type=float, default=0.4)
    parser.add_argument("--conda-env", type=str, default=None, help="e.g. Qwen3_2")
    parser.add_argument(
        "--e2e-backend",
        choices=["kespeech", "librispeech"],
        default="kespeech",
        help="kespeech uses CUDA event prefill/decode split (recommended)",
    )
    parser.add_argument("--skip-e2e", action="store_true", help="Only micro/decode-sim, no model run")
    parser.add_argument(
        "--ref-timing-without",
        type=str,
        default=None,
        help="Existing timing JSON (USE_FFT_FILTER=false) to skip e2e baseline run",
    )
    parser.add_argument(
        "--ref-timing-with",
        type=str,
        default=None,
        help="Existing timing JSON (USE_FFT_FILTER=true) to skip e2e SSS run",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    out: dict = {"device": str(device), "config": vars(args)}

    run_micro_bench = args.mode in ("micro", "all", "compare")
    run_decode_sim_bench = args.mode in ("decode-sim", "all", "compare")
    run_e2e_bench = args.mode in ("e2e", "compare") and not args.skip_e2e

    if run_micro_bench:
        rows = run_micro(
            args.seq_lens,
            args.num_layers,
            args.num_kv_heads,
            device,
            args.cutoff,
            args.alpha,
            args.kernel_size,
            args.warmup,
            args.iters,
        )
        out["micro"] = [asdict(r) for r in rows]

    if run_decode_sim_bench:
        out["decode_sim"] = run_decode_sim(
            args.prefill_len,
            args.retention,
            args.window,
            args.decode_tokens,
            args.num_layers,
            args.num_kv_heads,
            device,
            args.cutoff,
            args.alpha,
            args.kernel_size,
            args.warmup,
            args.iters,
        )

    if run_e2e_bench:
        e2e_dir = Path(args.e2e_output_dir or (REPO_ROOT / "rebuttal" / "sss_timing_e2e"))
        out["e2e"] = run_e2e(
            args.sample_limit,
            e2e_dir,
            args.model_path,
            args.gpu,
            args.budget,
            args.conda_env,
            args.e2e_backend,
        )
    elif args.ref_timing_without and args.ref_timing_with:
        out["e2e"] = load_e2e_from_ref_timing(
            Path(args.ref_timing_without),
            Path(args.ref_timing_with),
        )
        print(f"\nLoaded E2E timing from:\n  w/o SSS: {args.ref_timing_without}\n  w/ SSS:  {args.ref_timing_with}")

    if "decode_sim" in out and "e2e" in out:
        out["comparison"] = build_sss_vs_decode_comparison(out["decode_sim"], out["e2e"])
        print_sss_vs_decode_comparison(out["comparison"], out["decode_sim"])

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"\nSaved: {args.output_json}")

    print("\n--- Rebuttal guidance ---")
    print(
        "1) comparison.e2e_* = SSS on/off inside real ASR decode (primary for wall-clock).\n"
        "2) comparison.isolated_* = operator stress test (all layers, every step).\n"
        "3) Do not report isolated overhead alone as full decode slowdown."
    )
    if "comparison" in out:
        c = out["comparison"]
        print(
            f"\nSummary: isolated SSS stress = +{c['isolated_sss_overhead_per_step_ms']:.1f} ms/step "
            f"({c.get('isolated_overhead_pct_of_real_decode', 0):.0f}% of {c['real_decode_per_token_ms']:.1f} ms real decode); "
            f"E2E decode delta = +{c.get('e2e_extra_decode_per_token_ms', 0):.2f} ms/token."
        )


if __name__ == "__main__":
    main()
