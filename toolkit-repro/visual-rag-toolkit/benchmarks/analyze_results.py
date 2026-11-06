#!/usr/bin/env python3
"""
Analyze and compare benchmark results.

Usage:
    # Compare exhaustive vs two-stage
    python analyze_results.py --results results/

    # Compare multiple models
    python analyze_results.py --dirs results_colsmol/ results_colpali/
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


def load_all_results(results_dir: Path) -> Dict[str, Dict]:
    """Load all result files from directory."""
    results = {}
    for f in results_dir.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)

        # Key by dataset + method
        dataset = data.get("dataset", f.stem).split("/")[-1]
        method = "two_stage" if data.get("two_stage") else "exhaustive"
        key = f"{dataset}_{method}"

        results[key] = {
            "dataset": dataset,
            "method": method,
            "model": data.get("model", "unknown"),
            **data.get("metrics", {}),
        }
    return results


def compare_methods(results: Dict[str, Dict]) -> None:
    """Compare exhaustive vs two-stage on same datasets."""

    # Group by dataset
    datasets = {}
    for key, data in results.items():
        ds = data["dataset"].replace("_twostage", "")
        if ds not in datasets:
            datasets[ds] = {}
        datasets[ds][data["method"]] = data

    print("\n" + "=" * 80)
    print("EXHAUSTIVE vs TWO-STAGE COMPARISON")
    print("=" * 80)

    print(f"\n{'Dataset':<30} {'Method':<12} {'NDCG@10':>10} {'MRR@10':>10} {'Time(ms)':>10}")
    print("-" * 72)

    improvements = []
    speedups = []

    for dataset, methods in sorted(datasets.items()):
        for method in ["exhaustive", "two_stage"]:
            if method in methods:
                m = methods[method]
                time_ms = m.get("avg_search_time_ms", 0)
                print(
                    f"{dataset:<30} {method:<12} {m.get('ndcg@10', 0):>10.4f} {m.get('mrr@10', 0):>10.4f} {time_ms:>10.2f}"
                )

        # Calculate improvement
        if "exhaustive" in methods and "two_stage" in methods:
            ex = methods["exhaustive"]
            ts = methods["two_stage"]

            ndcg_diff = ts.get("ndcg@10", 0) - ex.get("ndcg@10", 0)
            improvements.append(ndcg_diff)

            ex_time = ex.get("avg_search_time_ms", 1)
            ts_time = ts.get("avg_search_time_ms", 1)
            if ts_time > 0:
                speedups.append(ex_time / ts_time)

        print()

    # Summary
    if improvements:
        print("-" * 72)
        print(f"Average NDCG@10 difference (two_stage - exhaustive): {np.mean(improvements):+.4f}")
        print(f"Retention rate: {100 * (1 + np.mean(improvements)):.1f}%")

    if speedups:
        print(f"Average speedup: {np.mean(speedups):.1f}x")


def analyze_stage1_recall(results: Dict[str, Dict]) -> None:
    """Analyze how well stage 1 preserves relevant documents."""
    print("\n" + "=" * 80)
    print("STAGE 1 RECALL ANALYSIS")
    print("=" * 80)
    print("\n(Stage 1 recall = how often relevant doc is in prefetch candidates)")
    print("This requires detailed results with stage1_rank info - run with --detailed")


def print_leaderboard(results: Dict[str, Dict]) -> None:
    """Print results in leaderboard format."""
    print("\n" + "=" * 80)
    print("LEADERBOARD FORMAT")
    print("=" * 80)

    # Best result per dataset
    best = {}
    for key, data in results.items():
        ds = data["dataset"].replace("_twostage", "")
        ndcg = data.get("ndcg@10", 0)
        if ds not in best or ndcg > best[ds].get("ndcg@10", 0):
            best[ds] = data

    # Compute average
    ndcg_scores = [d.get("ndcg@10", 0) for d in best.values()]
    avg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0

    print(f"\nModel: {list(results.values())[0].get('model', 'unknown')}")
    print(f"\n{'Dataset':<35} {'NDCG@10':>10}")
    print("-" * 45)

    for ds, data in sorted(best.items()):
        method_tag = " (2-stage)" if data.get("method") == "two_stage" else ""
        print(f"{ds + method_tag:<35} {data.get('ndcg@10', 0):>10.4f}")

    print("-" * 45)
    print(f"{'AVERAGE':<35} {avg:>10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--results", type=str, default="results", help="Results directory")
    parser.add_argument("--dirs", nargs="+", help="Multiple result directories to compare")
    parser.add_argument("--compare", action="store_true", help="Compare exhaustive vs two-stage")
    parser.add_argument("--leaderboard", action="store_true", help="Print in leaderboard format")

    args = parser.parse_args()

    if args.dirs:
        # Compare multiple directories
        all_results = {}
        for d in args.dirs:
            results = load_all_results(Path(d))
            for k, v in results.items():
                all_results[f"{d}_{k}"] = v
        results = all_results
    else:
        results = load_all_results(Path(args.results))

    if not results:
        print("❌ No results found")
        return

    print(f"📊 Loaded {len(results)} result files")

    if args.compare or not args.leaderboard:
        compare_methods(results)

    if args.leaderboard or not args.compare:
        print_leaderboard(results)


if __name__ == "__main__":
    main()
