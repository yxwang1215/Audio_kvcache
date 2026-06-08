#!/usr/bin/env python3
"""
Generate random, shuffled, or inverse head-importance JSON files for VY3D-Q3 ablation.

Keeps the same key layout and per-head list length as a reference audio-aware JSON,
so only the *source* of head importance changes at inference time (SSS unchanged).

Usage:
  python generate_random_head_scores.py \\
    --reference /path/to/audio_head_scores_topk24_alpha1_beta1_gamma1.json \\
    --output-dir /path/to/ablation/random \\
    --mode random --seeds 42 43 44

  python generate_random_head_scores.py \\
    --reference ... --output-dir ... --mode shuffled --seeds 42

  python generate_random_head_scores.py \\
    --reference ... --output-dir ... --mode inverse
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_reference(path: Path) -> Dict[str, List[float]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got {type(data)}")
    return data


def infer_dims(reference: Dict[str, Any]) -> tuple[int, int, int]:
    """Return (num_layers, num_heads, samples_per_head)."""
    layers: set[int] = set()
    heads: set[int] = set()
    sample_len = 0
    for key, value in reference.items():
        if key.startswith("__"):
            continue
        layer_s, head_s = key.split("-", 1)
        layers.add(int(layer_s))
        heads.add(int(head_s))
        if isinstance(value, list):
            sample_len = max(sample_len, len(value))
    if not layers or not heads:
        raise ValueError("Could not infer layer/head dimensions from reference JSON")
    return max(layers) + 1, max(heads) + 1, sample_len or 1


def _random_scalar(rng: np.random.Generator) -> float:
    # Strictly positive so normalization in SparseMM is well-defined.
    return float(rng.uniform(1e-6, 1.0))


def generate_random_scores(
    reference: Dict[str, Any],
    seed: int,
    per_sample: bool = True,
) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    out: Dict[str, List[float]] = {}
    for key, value in reference.items():
        if key.startswith("__"):
            out[key] = value
            continue
        n = len(value) if isinstance(value, list) and value else 1
        if per_sample and n > 1:
            out[key] = [_random_scalar(rng) for _ in range(n)]
        else:
            scalar = _random_scalar(rng)
            out[key] = [scalar] * n
    return out


def generate_shuffled_scores(reference: Dict[str, Any], seed: int) -> Dict[str, List[float]]:
    """Permute head keys: same score multiset, breaks audio–head alignment."""
    py_rng = random.Random(seed)
    keys = [k for k in reference if not k.startswith("__")]
    values = [reference[k] for k in keys]
    py_rng.shuffle(values)
    out: Dict[str, Any] = {}
    for special in reference:
        if special.startswith("__"):
            out[special] = reference[special]
    for key, val in zip(keys, values):
        out[key] = val
    return out


def _aggregate_score(value: Any, aggregation: str) -> float:
    if isinstance(value, list):
        if not value:
            return 0.0
        if aggregation == "first":
            return float(value[0])
        if aggregation == "mean":
            return float(np.mean(value))
        if aggregation == "max":
            return float(np.max(value))
        if aggregation == "min":
            return float(np.min(value))
        raise ValueError(f"Unsupported inverse aggregation: {aggregation}")
    return float(value)


def generate_inverse_scores(reference: Dict[str, Any], aggregation: str = "first") -> Dict[str, List[float]]:
    """
    Reverse the audio-aware head allocation while preserving the original score distribution.

    The most important head receives the score list from the least important head, the second
    most important receives the second least important, etc. With HEAD_SCORE_AGGREGATION=first,
    this directly reverses the budget allocation used by SparseKV.
    """
    keys = [k for k in reference if not k.startswith("__")]
    ranked_high_to_low = sorted(
        keys,
        key=lambda k: _aggregate_score(reference[k], aggregation),
        reverse=True,
    )
    ranked_low_to_high_values = [
        reference[k] for k in sorted(keys, key=lambda k: _aggregate_score(reference[k], aggregation))
    ]

    out: Dict[str, Any] = {}
    for special in reference:
        if special.startswith("__"):
            out[special] = reference[special]
    for key, val in zip(ranked_high_to_low, ranked_low_to_high_values):
        out[key] = val
    return out


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Wrote {path} ({len(data)} keys)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random/shuffled head score JSON for audio-supervision ablation.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Audio-aware reference JSON (defines keys and list lengths).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated JSON files.",
    )
    parser.add_argument(
        "--mode",
        choices=("random", "shuffled", "inverse"),
        default="random",
        help="random: fresh scores; shuffled: permute heads; inverse: reverse importance ranking.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds (one output file per seed).",
    )
    parser.add_argument(
        "--per-sample",
        action="store_true",
        default=True,
        help="Randomize each entry in per-head score lists (default: on).",
    )
    parser.add_argument(
        "--single-scalar",
        action="store_true",
        help="Use one random scalar per head (replicated across list).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Optional: build full grid if reference is incomplete.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Optional: build full grid if reference is incomplete.",
    )
    parser.add_argument(
        "--samples-per-head",
        type=int,
        default=None,
        help="List length per head when building from --num-layers/--num-heads.",
    )
    parser.add_argument(
        "--inverse-aggregation",
        choices=("first", "mean", "max", "min"),
        default="first",
        help="Score aggregation used to rank heads for --mode inverse.",
    )
    args = parser.parse_args()

    reference = load_reference(args.reference)
    nl, nh, ns = infer_dims(reference)
    if args.num_layers is not None:
        nl = args.num_layers
    if args.num_heads is not None:
        nh = args.num_heads
    if args.samples_per_head is not None:
        ns = args.samples_per_head

    # Ensure complete layer-head grid for generation.
    grid_ref: Dict[str, List[float]] = {}
    for layer in range(nl):
        for head in range(nh):
            key = f"{layer}-{head}"
            if key in reference:
                val = reference[key]
                grid_ref[key] = val if isinstance(val, list) else [float(val)]
            else:
                grid_ref[key] = [0.0] * ns

    per_sample = args.per_sample and not args.single_scalar
    ref_stem = args.reference.stem

    mode_seeds = args.seeds if args.mode in ("random", "shuffled") else [None]
    for seed in mode_seeds:
        if args.mode == "random":
            assert seed is not None
            data = generate_random_scores(grid_ref, seed=seed, per_sample=per_sample)
            suffix = f"random_seed{seed}"
        elif args.mode == "shuffled":
            assert seed is not None
            data = generate_shuffled_scores(grid_ref, seed=seed)
            suffix = f"shuffled_seed{seed}"
        else:
            data = generate_inverse_scores(grid_ref, aggregation=args.inverse_aggregation)
            suffix = "inverse"

        out_path = args.output_dir / f"{ref_stem}_{suffix}.json"
        meta_path = args.output_dir / f"{ref_stem}_{suffix}_meta.json"
        write_json(data, out_path)
        meta = {
            "reference": str(args.reference.resolve()),
            "mode": args.mode,
            "seed": seed,
            "inverse_aggregation": args.inverse_aggregation if args.mode == "inverse" else None,
            "num_layers": nl,
            "num_heads": nh,
            "samples_per_head": ns,
            "per_sample_random": per_sample,
            "output": str(out_path.resolve()),
        }
        write_json(meta, meta_path)


if __name__ == "__main__":
    main()
