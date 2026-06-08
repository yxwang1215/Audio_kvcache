#!/usr/bin/env python3
"""Summarize jV7P-Q2 smoother ablation JSON results into a LaTeX-friendly table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _metric_from_result(data: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    for key in ("wer_mean", "wer", "cer", "accuracy", "bleu", "avg_accuracy"):
        if key in data and data[key] is not None:
            v = data[key]
            if isinstance(v, (int, float)):
                return key, float(v)
    summary = data.get("summary") or data.get("metrics") or data.get("current_statistics")
    if isinstance(summary, dict):
        for key in ("wer_mean", "current_avg_wer", "wer", "cer", "accuracy", "bleu"):
            if key in summary and summary[key] is not None:
                return key, float(summary[key])
    return "metric", None


def _word_accuracy(data: Dict[str, Any], metric: Optional[float]) -> Optional[float]:
    if isinstance(data.get("word_accuracy"), (int, float)):
        return float(data["word_accuracy"])
    summary = data.get("summary") or data.get("metrics") or data.get("current_statistics")
    if isinstance(summary, dict):
        for key in ("word_accuracy", "current_avg_accuracy"):
            if isinstance(summary.get(key), (int, float)):
                return float(summary[key])
    if metric is not None:
        return 100.0 - metric
    return None


def _long_decode_count(data: Dict[str, Any], threshold: int) -> Optional[int]:
    detailed = data.get("detailed_results")
    if not isinstance(detailed, list):
        return None
    count = 0
    for row in detailed:
        if isinstance(row, dict) and isinstance(row.get("decode_token_count"), (int, float)):
            count += int(row["decode_token_count"] >= threshold)
    return count


def collect(root: Path, task: str, long_decode_threshold: int) -> List[Dict[str, Any]]:
    task_dir = root / task
    if not task_dir.is_dir():
        raise FileNotFoundError(task_dir)
    rows: List[Dict[str, Any]] = []
    for smooth_dir in sorted(task_dir.glob("Smooth_*")):
        method = smooth_dir.name.replace("Smooth_", "")
        for budget_dir in sorted(smooth_dir.glob("budget*")):
            budget = budget_dir.name.replace("budget", "")
            for jf in budget_dir.glob("*results*.json"):
                try:
                    with open(jf, encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue
                if data.get("status") != "completed":
                    continue
                metric_name, metric = _metric_from_result(data)
                total_samples = data.get("total_samples") or data.get("valid_samples")
                if not isinstance(total_samples, int):
                    progress = data.get("progress")
                    if isinstance(progress, dict) and isinstance(progress.get("processed"), int):
                        total_samples = progress["processed"]
                long_decode_count = _long_decode_count(data, long_decode_threshold)
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "budget": budget,
                        "metric_name": metric_name,
                        "metric": metric,
                        "word_accuracy": _word_accuracy(data, metric),
                        "total_samples": total_samples,
                        "long_decode": long_decode_count,
                        "long_decode_threshold": long_decode_threshold,
                        "path": str(jf),
                    }
                )
                break
    return rows


def _ordered_unique(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--tasks", nargs="+", default=[])
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--long-decode-threshold", type=int, default=1000)
    args = parser.parse_args()

    tasks = _ordered_unique([*args.task, *args.tasks])
    if not tasks:
        raise SystemExit("Specify --task TASK or --tasks TASK [TASK ...].")

    rows: List[Dict[str, Any]] = []
    for task in tasks:
        rows.extend(collect(args.root, task, args.long_decode_threshold))

    print(f"{'Task':<24} {'Method':<16} {'Budget':<8} {'WER':<10} {'Acc':<10} {'Long':<10} Path")
    print("-" * 120)
    for r in rows:
        wer = f"{r['metric']:.2f}" if r["metric"] is not None else "n/a"
        acc = f"{r['word_accuracy']:.2f}" if r["word_accuracy"] is not None else "n/a"
        total = r["total_samples"] or "?"
        long_decode = r["long_decode"]
        long_text = "n/a" if long_decode is None else f"{long_decode}/{total}"
        print(
            f"{r['task']:<24} {r['method']:<16} {r['budget']:<8} "
            f"{wer:<10} {acc:<10} {long_text:<10} {r['path']}"
        )

    print("\nLaTeX rows:")
    print("% Dataset & Method & WER $\\downarrow$ & Word Acc. $\\uparrow$ & Long decode \\\\")
    for r in rows:
        wer = f"{r['metric']:.2f}" if r["metric"] is not None else "--"
        acc = f"{r['word_accuracy']:.2f}" if r["word_accuracy"] is not None else "--"
        total = r["total_samples"] or "?"
        long_decode = r["long_decode"]
        long_text = "--" if long_decode is None else f"{long_decode}/{total}"
        print(f"{r['task']} & {r['method']} & {wer} & {acc} & {long_text} \\\\")

    out = args.output_json or args.root / "smoother_ablation_summary_all_asr.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"tasks": tasks, "rows": rows}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
