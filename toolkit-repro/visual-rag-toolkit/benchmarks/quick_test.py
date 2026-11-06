#!/usr/bin/env python3
"""
Quick Benchmark - Validate retrieval quality with ViDoRe data.

This script:
1. Downloads samples from ViDoRe (with ground truth relevance)
2. Embeds with ColSmol-500M
3. Tests retrieval strategies (exhaustive vs two-stage)
4. Computes METRICS: NDCG@K, MRR@K, Recall@K
5. Compares speed and quality

Usage:
    python quick_test.py --samples 100
    python quick_test.py --samples 500 --skip-exhaustive  # Faster
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to Python path (so we can import visual_rag)
# This allows running the script directly without pip install
_script_dir = Path(__file__).parent
_parent_dir = _script_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

# Visual RAG imports (now works without pip install)
from visual_rag.embedding import VisualEmbedder  # noqa: E402
from visual_rag.embedding.pooling import (  # noqa: E402
    compute_maxsim_score,
    tile_level_mean_pooling,
)

# Optional: datasets for ViDoRe
try:
    from datasets import load_dataset as hf_load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_vidore_sample(num_samples: int = 100) -> List[Dict]:
    """
    Load sample from ViDoRe DocVQA with ground truth.

    Each sample has a query and its relevant document (1:1 mapping).
    This allows computing retrieval metrics.
    """
    if not HAS_DATASETS:
        logger.error("Install datasets: pip install datasets")
        sys.exit(1)

    logger.info(f"📥 Loading {num_samples} samples from ViDoRe DocVQA...")

    ds = hf_load_dataset("vidore/docvqa_test_subsampled", split="test")

    samples = []
    for i, example in enumerate(ds):
        if i >= num_samples:
            break

        samples.append(
            {
                "id": i,
                "doc_id": f"doc_{i}",
                "query_id": f"q_{i}",
                "image": example.get("image", example.get("page_image")),
                "query": example.get("query", example.get("question", "")),
                # Ground truth: query i is relevant to doc i
                "relevant_doc": f"doc_{i}",
            }
        )

    logger.info(f"✅ Loaded {len(samples)} samples with ground truth")
    return samples


def embed_all(
    samples: List[Dict],
    model_name: str = "vidore/colSmol-500M",
) -> Dict[str, Any]:
    """Embed all documents and queries."""
    logger.info(f"\n🤖 Loading model: {model_name}")
    embedder = VisualEmbedder(model_name=model_name)

    images = [s["image"] for s in samples]
    queries = [s["query"] for s in samples if s["query"]]

    # Embed images
    logger.info(f"🎨 Embedding {len(images)} documents...")
    start_time = time.time()

    embeddings, token_infos = embedder.embed_images(images, batch_size=4, return_token_info=True)

    doc_embed_time = time.time() - start_time
    logger.info(f"   Time: {doc_embed_time:.2f}s ({doc_embed_time/len(images)*1000:.1f}ms/doc)")

    # Process embeddings: extract visual tokens + tile-level pooling
    doc_data = {}
    for i, (emb, token_info) in enumerate(zip(embeddings, token_infos)):
        if hasattr(emb, "cpu"):
            emb = emb.cpu()
        emb_np = emb.numpy() if hasattr(emb, "numpy") else np.array(emb)

        # Extract visual tokens only (filter special tokens)
        visual_indices = token_info["visual_token_indices"]
        visual_emb = emb_np[visual_indices].astype(np.float32)

        # Tile-level pooling
        n_rows = token_info.get("n_rows", 4)
        n_cols = token_info.get("n_cols", 3)
        num_tiles = n_rows * n_cols + 1 if n_rows and n_cols else 13

        tile_pooled = tile_level_mean_pooling(visual_emb, num_tiles, patches_per_tile=64)

        doc_data[f"doc_{i}"] = {
            "embedding": visual_emb,
            "pooled": tile_pooled,
            "num_visual_tokens": len(visual_indices),
            "num_tiles": tile_pooled.shape[0],
        }

    # Embed queries
    logger.info(f"🔍 Embedding {len(queries)} queries...")
    start_time = time.time()

    query_data = {}
    for i, query in enumerate(tqdm(queries, desc="Queries")):
        q_emb = embedder.embed_query(query)
        if hasattr(q_emb, "cpu"):
            q_emb = q_emb.cpu()
        q_np = q_emb.numpy() if hasattr(q_emb, "numpy") else np.array(q_emb)
        query_data[f"q_{i}"] = q_np.astype(np.float32)

    query_embed_time = time.time() - start_time

    return {
        "docs": doc_data,
        "queries": query_data,
        "samples": samples,
        "doc_embed_time": doc_embed_time,
        "query_embed_time": query_embed_time,
        "model": model_name,
    }


def search_exhaustive(query_emb: np.ndarray, docs: Dict, top_k: int = 10) -> List[Dict]:
    """Exhaustive MaxSim search over all documents."""
    scores = []
    for doc_id, doc in docs.items():
        score = compute_maxsim_score(query_emb, doc["embedding"])
        scores.append({"id": doc_id, "score": score})

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_k]


def search_two_stage(
    query_emb: np.ndarray,
    docs: Dict,
    prefetch_k: int = 20,
    top_k: int = 10,
) -> List[Dict]:
    """
    Two-stage retrieval with tile-level pooling.

    Stage 1: Fast prefetch using tile-pooled vectors
    Stage 2: Exact MaxSim reranking on candidates
    """
    # Stage 1: Tile-level pooled search
    query_pooled = query_emb.mean(axis=0)
    query_pooled = query_pooled / (np.linalg.norm(query_pooled) + 1e-8)

    stage1_scores = []
    for doc_id, doc in docs.items():
        doc_pooled = doc["pooled"]
        doc_norm = doc_pooled / (np.linalg.norm(doc_pooled, axis=1, keepdims=True) + 1e-8)
        tile_sims = np.dot(doc_norm, query_pooled)
        score = float(tile_sims.max())
        stage1_scores.append({"id": doc_id, "score": score})

    stage1_scores.sort(key=lambda x: x["score"], reverse=True)
    candidates = stage1_scores[:prefetch_k]

    # Stage 2: Exact MaxSim on candidates
    reranked = []
    for cand in candidates:
        doc_id = cand["id"]
        score = compute_maxsim_score(query_emb, docs[doc_id]["embedding"])
        reranked.append(
            {"id": doc_id, "score": score, "stage1_rank": stage1_scores.index(cand) + 1}
        )

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]


def compute_metrics(
    results: Dict[str, List[Dict]],
    samples: List[Dict],
    k_values: List[int] = [1, 3, 5, 7, 10],
) -> Dict[str, float]:
    """
    Compute retrieval metrics.

    Since ViDoRe has 1:1 query-doc mapping (1 relevant doc per query):
    - Recall@K (Hit Rate): Is the relevant doc in top-K? (0 or 1)
    - Precision@K: (# relevant in top-K) / K
    - MRR@K: 1/rank if found in top-K, else 0
    - NDCG@K: DCG / IDCG with binary relevance
    """
    metrics = {}

    # Also track per-query ranks for analysis
    all_ranks = []

    for k in k_values:
        recalls = []
        precisions = []
        mrrs = []
        ndcgs = []

        for sample in samples:
            query_id = sample["query_id"]
            relevant_doc = sample["relevant_doc"]

            if query_id not in results:
                continue

            ranking = results[query_id][:k]
            ranked_ids = [r["id"] for r in ranking]

            # Find rank of relevant doc (1-indexed, 0 if not found)
            rank = 0
            for i, doc_id in enumerate(ranked_ids):
                if doc_id == relevant_doc:
                    rank = i + 1
                    break

            # Recall@K (Hit Rate): 1 if found in top-K
            found = 1.0 if rank > 0 else 0.0
            recalls.append(found)

            # Precision@K: (# relevant found) / K
            # With 1 relevant doc: 1/K if found, 0 otherwise
            precision = found / k
            precisions.append(precision)

            # MRR@K: 1/rank if found
            mrr = 1.0 / rank if rank > 0 else 0.0
            mrrs.append(mrr)

            # NDCG@K (binary relevance)
            # DCG = 1/log2(rank+1) if found, 0 otherwise
            # IDCG = 1/log2(2) = 1 (best case: relevant at rank 1)
            dcg = 1.0 / np.log2(rank + 1) if rank > 0 else 0.0
            idcg = 1.0
            ndcg = dcg / idcg
            ndcgs.append(ndcg)

            # Track actual rank for analysis (only for k=10)
            if k == max(k_values):
                full_ranking = results[query_id]
                full_rank = 0
                for i, r in enumerate(full_ranking):
                    if r["id"] == relevant_doc:
                        full_rank = i + 1
                        break
                all_ranks.append(full_rank)

        metrics[f"Recall@{k}"] = np.mean(recalls)
        metrics[f"P@{k}"] = np.mean(precisions)
        metrics[f"MRR@{k}"] = np.mean(mrrs)
        metrics[f"NDCG@{k}"] = np.mean(ndcgs)

    # Add summary stats
    if all_ranks:
        found_ranks = [r for r in all_ranks if r > 0]
        metrics["avg_rank"] = np.mean(found_ranks) if found_ranks else float("inf")
        metrics["median_rank"] = np.median(found_ranks) if found_ranks else float("inf")
        metrics["not_found"] = sum(1 for r in all_ranks if r == 0)

    return metrics


def run_benchmark(
    data: Dict,
    skip_exhaustive: bool = False,
    prefetch_k: int = None,
    top_k: int = 10,
) -> Dict[str, Dict]:
    """Run retrieval benchmark with metrics."""
    docs = data["docs"]
    queries = data["queries"]
    samples = data["samples"]
    num_docs = len(docs)

    # Auto-set prefetch_k to be meaningful (default: 20, or 20% of docs if >100 docs)
    if prefetch_k is None:
        if num_docs <= 100:
            prefetch_k = 20  # Default: prefetch 20, rerank to top-10
        else:
            prefetch_k = max(20, min(100, int(num_docs * 0.2)))  # 20% for larger collections

    # Ensure prefetch_k < num_docs for meaningful two-stage comparison
    if prefetch_k >= num_docs:
        logger.warning(f"⚠️  prefetch_k={prefetch_k} >= num_docs={num_docs}")
        logger.warning("   Two-stage will fetch ALL docs (same as exhaustive)")
        logger.warning(f"   Use --samples > {prefetch_k * 3} for meaningful comparison")

    logger.info(f"📊 Benchmark config: {num_docs} docs, prefetch_k={prefetch_k}, top_k={top_k}")
    logger.info(f"   (Both methods return top-{top_k} results - realistic retrieval scenario)")

    results = {}

    # Two-stage retrieval (NOVEL)
    logger.info(
        f"\n🔬 Running Two-Stage retrieval (prefetch top-{prefetch_k}, rerank to top-{top_k})..."
    )
    two_stage_results = {}
    two_stage_times = []

    for sample in tqdm(samples, desc="Two-Stage"):
        query_id = sample["query_id"]
        query_emb = queries[query_id]

        start = time.time()
        ranking = search_two_stage(query_emb, docs, prefetch_k=prefetch_k, top_k=top_k)
        two_stage_times.append(time.time() - start)

        two_stage_results[query_id] = ranking

    two_stage_metrics = compute_metrics(two_stage_results, samples)
    two_stage_metrics["avg_time_ms"] = np.mean(two_stage_times) * 1000
    two_stage_metrics["prefetch_k"] = prefetch_k
    two_stage_metrics["top_k"] = top_k
    results["two_stage"] = two_stage_metrics

    # Exhaustive search (baseline)
    if not skip_exhaustive:
        logger.info(
            f"🔬 Running Exhaustive MaxSim (searches ALL {num_docs} docs, returns top-{top_k})..."
        )
        exhaustive_results = {}
        exhaustive_times = []

        for sample in tqdm(samples, desc="Exhaustive"):
            query_id = sample["query_id"]
            query_emb = queries[query_id]

            start = time.time()
            ranking = search_exhaustive(query_emb, docs, top_k=top_k)
            exhaustive_times.append(time.time() - start)

            exhaustive_results[query_id] = ranking

        exhaustive_metrics = compute_metrics(exhaustive_results, samples)
        exhaustive_metrics["avg_time_ms"] = np.mean(exhaustive_times) * 1000
        exhaustive_metrics["top_k"] = top_k
        results["exhaustive"] = exhaustive_metrics

    return results


def print_results(data: Dict, benchmark_results: Dict, show_precision: bool = False):
    """Print benchmark results."""
    print("\n" + "=" * 80)
    print("📊 BENCHMARK RESULTS")
    print("=" * 80)

    num_docs = len(data["docs"])
    print(f"\n🤖 Model: {data['model']}")
    print(f"📄 Documents: {num_docs}")
    print(f"🔍 Queries: {len(data['queries'])}")

    # Embedding stats
    sample_doc = list(data["docs"].values())[0]
    print("\n📏 Embedding (after visual token filtering):")
    print(f"   Visual tokens per doc: {sample_doc['num_visual_tokens']}")
    print(f"   Tile-pooled vectors: {sample_doc['num_tiles']}")

    if "two_stage" in benchmark_results:
        prefetch_k = benchmark_results["two_stage"].get("prefetch_k", "?")
        print(f"   Two-stage prefetch_k: {prefetch_k} (of {num_docs} docs)")

    # Method labels - clearer naming
    def get_label(method):
        if method == "two_stage":
            return "Pooled+Rerank"  # Tile-pooled prefetch + MaxSim rerank
        else:
            return "Full MaxSim"  # Exhaustive MaxSim on all docs

    # Recall / Hit Rate table
    print("\n🎯 RECALL (Hit Rate) @ K:")
    print(f"   {'Method':<20} {'@1':>8} {'@3':>8} {'@5':>8} {'@7':>8} {'@10':>8}")
    print(f"   {'-'*60}")

    for method, metrics in benchmark_results.items():
        print(
            f"   {get_label(method):<20} "
            f"{metrics.get('Recall@1', 0):>8.3f} "
            f"{metrics.get('Recall@3', 0):>8.3f} "
            f"{metrics.get('Recall@5', 0):>8.3f} "
            f"{metrics.get('Recall@7', 0):>8.3f} "
            f"{metrics.get('Recall@10', 0):>8.3f}"
        )

    # Precision table (optional)
    if show_precision:
        print("\n📐 PRECISION @ K:")
        print(f"   {'Method':<20} {'@1':>8} {'@3':>8} {'@5':>8} {'@7':>8} {'@10':>8}")
        print(f"   {'-'*60}")

        for method, metrics in benchmark_results.items():
            print(
                f"   {get_label(method):<20} "
                f"{metrics.get('P@1', 0):>8.3f} "
                f"{metrics.get('P@3', 0):>8.3f} "
                f"{metrics.get('P@5', 0):>8.3f} "
                f"{metrics.get('P@7', 0):>8.3f} "
                f"{metrics.get('P@10', 0):>8.3f}"
            )

    # NDCG table
    print("\n📈 NDCG @ K:")
    print(f"   {'Method':<20} {'@1':>8} {'@3':>8} {'@5':>8} {'@7':>8} {'@10':>8}")
    print(f"   {'-'*60}")

    for method, metrics in benchmark_results.items():
        print(
            f"   {get_label(method):<20} "
            f"{metrics.get('NDCG@1', 0):>8.3f} "
            f"{metrics.get('NDCG@3', 0):>8.3f} "
            f"{metrics.get('NDCG@5', 0):>8.3f} "
            f"{metrics.get('NDCG@7', 0):>8.3f} "
            f"{metrics.get('NDCG@10', 0):>8.3f}"
        )

    # MRR table
    print("\n🔍 MRR @ K:")
    print(f"   {'Method':<20} {'@1':>8} {'@3':>8} {'@5':>8} {'@7':>8} {'@10':>8}")
    print(f"   {'-'*60}")

    for method, metrics in benchmark_results.items():
        print(
            f"   {get_label(method):<20} "
            f"{metrics.get('MRR@1', 0):>8.3f} "
            f"{metrics.get('MRR@3', 0):>8.3f} "
            f"{metrics.get('MRR@5', 0):>8.3f} "
            f"{metrics.get('MRR@7', 0):>8.3f} "
            f"{metrics.get('MRR@10', 0):>8.3f}"
        )

    # Speed comparison
    top_k = benchmark_results.get("two_stage", benchmark_results.get("exhaustive", {})).get(
        "top_k", 10
    )
    print(f"\n⏱️  SPEED (both return top-{top_k} results):")
    print(f"   {'Method':<20} {'Time (ms)':>12} {'Docs searched':>15}")
    print(f"   {'-'*50}")

    for method, metrics in benchmark_results.items():
        if method == "two_stage":
            searched = metrics.get("prefetch_k", "?")
            label = f"{searched} (stage-1)"
        else:
            searched = num_docs
            label = f"{searched} (all)"
        print(f"   {get_label(method):<20} {metrics.get('avg_time_ms', 0):>12.2f} {label:>15}")

    # Comparison summary
    if "exhaustive" in benchmark_results and "two_stage" in benchmark_results:
        ex = benchmark_results["exhaustive"]
        ts = benchmark_results["two_stage"]

        print("\n💡 POOLED+RERANK vs FULL MAXSIM:")

        for k in [1, 5, 10]:
            ex_recall = ex.get(f"Recall@{k}", 0)
            ts_recall = ts.get(f"Recall@{k}", 0)
            if ex_recall > 0:
                retention = ts_recall / ex_recall * 100
                print(
                    f"   • Recall@{k} retention: {retention:.1f}% ({ts_recall:.3f} vs {ex_recall:.3f})"
                )

        speedup = ex["avg_time_ms"] / ts["avg_time_ms"] if ts["avg_time_ms"] > 0 else 0
        print(f"   • Speedup: {speedup:.1f}x")

        # Rank stats with explanation
        if "avg_rank" in ts:
            prefetch_k = ts.get("prefetch_k", "?")
            top_k = ts.get("top_k", 10)
            not_found = ts.get("not_found", 0)
            total = len(data["queries"])

            print("\n📊 POOLED+RERANK STATISTICS:")
            print("   Stage-1 (pooled prefetch):")
            print(f"      • Searches top-{prefetch_k} candidates using tile-pooled vectors")
            print(
                f"      • {total - not_found}/{total} queries ({100 - not_found/total*100:.1f}%) had relevant doc in prefetch"
            )
            print(
                f"      • {not_found}/{total} queries ({not_found/total*100:.1f}%) missed (relevant doc ranked >{prefetch_k})"
            )
            print("   Stage-2 (MaxSim reranking):")
            print("      • Reranks prefetch candidates with exact MaxSim")
            print(f"      • Returns final top-{top_k} results")
            if ts["avg_rank"] < float("inf"):
                print(f"      • Avg rank of relevant doc (when found): {ts['avg_rank']:.1f}")
                print(f"      • Median rank: {ts['median_rank']:.1f}")
            print(f"\n   💡 The {not_found/total*100:.1f}% miss rate is for stage-1 prefetch.")
            print(
                f"      Final Recall@{top_k} shows how many relevant docs ARE in top-{top_k} results."
            )

    print("\n" + "=" * 80)
    print("✅ Benchmark complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Quick benchmark for visual-rag-toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--samples", type=int, default=100, help="Number of samples (default: 100)")
    parser.add_argument(
        "--model",
        type=str,
        default="vidore/colSmol-500M",
        help="Model: vidore/colSmol-500M (default), vidore/colpali-v1.3",
    )
    parser.add_argument(
        "--prefetch-k",
        type=int,
        default=None,
        help="Stage 1 candidates for two-stage (default: 20 for <=100 docs, auto for larger)",
    )
    parser.add_argument(
        "--skip-exhaustive", action="store_true", help="Skip exhaustive baseline (faster)"
    )
    parser.add_argument(
        "--show-precision", action="store_true", help="Show Precision@K metrics (hidden by default)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10, realistic retrieval scenario)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🧪 VISUAL RAG TOOLKIT - RETRIEVAL BENCHMARK")
    print("=" * 70)

    # Load samples
    samples = load_vidore_sample(args.samples)

    if not samples:
        logger.error("No samples loaded!")
        sys.exit(1)

    # Embed all
    data = embed_all(samples, args.model)

    # Run benchmark
    benchmark_results = run_benchmark(
        data,
        skip_exhaustive=args.skip_exhaustive,
        prefetch_k=args.prefetch_k,
        top_k=args.top_k,
    )

    # Print results
    print_results(data, benchmark_results, show_precision=args.show_precision)


if __name__ == "__main__":
    main()
