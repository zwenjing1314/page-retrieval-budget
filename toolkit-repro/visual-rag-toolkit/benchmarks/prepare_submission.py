#!/usr/bin/env python3
"""
Prepare submission for ViDoRe leaderboard.

Reads evaluation results and formats them for HuggingFace submission.

Usage:
    python prepare_submission.py --results results/ --output submission.json
    python prepare_submission.py --results results/ --model-name "MyModel" --upload
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ViDoRe leaderboard expected datasets
VIDORE_DATASETS = {
    "docvqa_test_subsampled": "DocVQA",
    "infovqa_test_subsampled": "InfoVQA",
    "tabfquad_test_subsampled": "TabFQuAD",
    "tatdqa_test": "TAT-DQA",
    "arxivqa_test_subsampled": "ArXivQA",
    "shiftproject_test": "SHIFT",
}


def load_results(results_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load all result JSON files from directory."""
    results = {}

    for json_file in results_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)

        dataset = data.get("dataset", json_file.stem)
        dataset_short = dataset.split("/")[-1].replace("_twostage", "")

        results[dataset_short] = {
            "ndcg@5": data["metrics"].get("ndcg@5", 0),
            "ndcg@10": data["metrics"].get("ndcg@10", 0),
            "mrr@10": data["metrics"].get("mrr@10", 0),
            "recall@5": data["metrics"].get("recall@5", 0),
            "recall@10": data["metrics"].get("recall@10", 0),
            "two_stage": data.get("two_stage", False),
            "model": data.get("model", "unknown"),
        }

    return results


def format_submission(
    results: Dict[str, Dict],
    model_name: str,
    model_url: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Format results for ViDoRe leaderboard submission."""

    # Calculate average scores
    ndcg10_scores = [r["ndcg@10"] for r in results.values()]
    avg_ndcg10 = sum(ndcg10_scores) / len(ndcg10_scores) if ndcg10_scores else 0

    submission = {
        "model_name": model_name,
        "model_url": model_url or "",
        "description": description or "Visual RAG Toolkit submission",
        "submitted_at": datetime.now().isoformat(),
        "average_ndcg@10": avg_ndcg10,
        "results": {},
    }

    # Add per-dataset results
    for dataset_short, metrics in results.items():
        display_name = VIDORE_DATASETS.get(dataset_short, dataset_short)
        submission["results"][display_name] = {
            "ndcg@5": metrics["ndcg@5"],
            "ndcg@10": metrics["ndcg@10"],
            "mrr@10": metrics["mrr@10"],
        }

    return submission


def print_summary(results: Dict[str, Dict], submission: Dict[str, Any]):
    """Print summary table."""
    print("\n" + "=" * 70)
    print(f"MODEL: {submission['model_name']}")
    print("=" * 70)

    print(f"\n{'Dataset':<25} {'NDCG@5':>10} {'NDCG@10':>10} {'MRR@10':>10}")
    print("-" * 55)

    for dataset, metrics in results.items():
        display = VIDORE_DATASETS.get(dataset, dataset)[:24]
        print(
            f"{display:<25} {metrics['ndcg@5']:>10.4f} {metrics['ndcg@10']:>10.4f} {metrics['mrr@10']:>10.4f}"
        )

    print("-" * 55)
    print(f"{'AVERAGE':<25} {'':<10} {submission['average_ndcg@10']:>10.4f}")
    print("=" * 70)


def upload_to_huggingface(submission: Dict[str, Any], repo_id: str = "vidore/results"):
    """Upload submission to HuggingFace."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        return False

    api = HfApi()

    # Save to temp file
    temp_file = Path(f"/tmp/submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(temp_file, "w") as f:
        json.dump(submission, f, indent=2)

    try:
        api.upload_file(
            path_or_fileobj=str(temp_file),
            path_in_repo=f"submissions/{submission['model_name']}.json",
            repo_id=repo_id,
            repo_type="space",
        )
        print(f"✅ Uploaded to {repo_id}")
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Prepare ViDoRe submission")
    parser.add_argument(
        "--results", type=str, default="results", help="Directory with result JSON files"
    )
    parser.add_argument(
        "--output", type=str, default="submission.json", help="Output submission file"
    )
    parser.add_argument(
        "--model-name", type=str, default="visual-rag-toolkit", help="Model name for leaderboard"
    )
    parser.add_argument("--model-url", type=str, help="URL to model/paper")
    parser.add_argument("--description", type=str, help="Model description")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace")

    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return

    # Load results
    results = load_results(results_dir)
    if not results:
        print(f"❌ No result files found in {results_dir}")
        return

    print(f"📊 Found {len(results)} dataset results")

    # Format submission
    submission = format_submission(
        results,
        model_name=args.model_name,
        model_url=args.model_url,
        description=args.description,
    )

    # Print summary
    print_summary(results, submission)

    # Save
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"\n💾 Saved to: {output_path}")

    # Upload if requested
    if args.upload:
        upload_to_huggingface(submission)


if __name__ == "__main__":
    main()
