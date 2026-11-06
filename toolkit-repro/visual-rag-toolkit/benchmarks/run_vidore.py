#!/usr/bin/env python3
"""
ViDoRe Benchmark Evaluation Script

Evaluates visual document retrieval on the ViDoRe benchmark datasets.

Usage:
    # Single dataset
    python run_vidore.py --dataset vidore/docvqa_test_subsampled

    # All datasets
    python run_vidore.py --all

    # With two-stage retrieval
    python run_vidore.py --dataset vidore/docvqa_test_subsampled --two-stage
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ViDoRe benchmark datasets
# Official leaderboard: https://huggingface.co/spaces/vidore/vidore-leaderboard
VIDORE_DATASETS = {
    # === RECOMMENDED FOR QUICK TESTING (smaller, faster) ===
    "docvqa": "vidore/docvqa_test_subsampled",  # ~500 queries, Document VQA
    "infovqa": "vidore/infovqa_test_subsampled",  # ~500 queries, Infographics
    "tabfquad": "vidore/tabfquad_test_subsampled",  # ~500 queries, Tables
    # === FULL EVALUATION ===
    "tatdqa": "vidore/tatdqa_test",  # ~1500 queries, Financial tables
    "arxivqa": "vidore/arxivqa_test_subsampled",  # ~500 queries, Scientific papers
    "shift": "vidore/shiftproject_test",  # ~500 queries, Sustainability reports
}

# Aliases for convenience
QUICK_DATASETS = ["docvqa", "infovqa"]  # Fast testing
ALL_DATASETS = list(VIDORE_DATASETS.keys())


def load_dataset(dataset_name: str) -> Dict[str, Any]:
    """Load a ViDoRe dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")

    logger.info(f"Loading dataset: {dataset_name}")

    # Load dataset
    ds = load_dataset(dataset_name, split="test")

    # Extract queries and documents
    # ViDoRe format: each example has query, image, and relevant doc info
    queries = []
    documents = []
    qrels = {}  # query_id -> {doc_id: relevance}

    for idx, example in enumerate(tqdm(ds, desc="Loading data")):
        query_id = f"q_{idx}"
        doc_id = f"d_{idx}"

        # Get query text
        query_text = example.get("query", example.get("question", ""))
        queries.append(
            {
                "id": query_id,
                "text": query_text,
            }
        )

        # Get document image
        image = example.get("image", example.get("page_image"))
        documents.append(
            {
                "id": doc_id,
                "image": image,
            }
        )

        # Relevance (self-document is relevant)
        qrels[query_id] = {doc_id: 1}

    logger.info(f"Loaded {len(queries)} queries and {len(documents)} documents")

    return {
        "queries": queries,
        "documents": documents,
        "qrels": qrels,
    }


def embed_documents(
    documents: List[Dict],
    embedder,
    batch_size: int = 4,
    return_pooled: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Embed all documents.

    Args:
        documents: List of {id, image} dicts
        embedder: VisualEmbedder instance
        batch_size: Batch size for embedding
        return_pooled: Also return tile-level pooled embeddings (for two-stage)

    Returns:
        doc_embeddings dict, and optionally pooled_embeddings dict
    """
    from visual_rag.embedding.pooling import tile_level_mean_pooling

    logger.info(f"Embedding {len(documents)} documents...")

    images = [doc["image"] for doc in documents]

    # Get embeddings with token info for proper pooling
    embeddings, token_infos = embedder.embed_images(
        images, batch_size=batch_size, return_token_info=True
    )

    doc_embeddings = {}
    pooled_embeddings = {} if return_pooled else None

    for doc, emb, token_info in zip(documents, embeddings, token_infos):
        if hasattr(emb, "numpy"):
            emb_np = emb.numpy()
        elif hasattr(emb, "cpu"):
            emb_np = emb.cpu().numpy()
        else:
            emb_np = np.array(emb)

        doc_embeddings[doc["id"]] = emb_np.astype(np.float32)

        # Compute tile-level pooling (NOVEL approach)
        if return_pooled:
            n_rows = token_info.get("n_rows", 4)
            n_cols = token_info.get("n_cols", 3)
            num_tiles = n_rows * n_cols + 1 if n_rows and n_cols else 13

            pooled = tile_level_mean_pooling(emb_np, num_tiles, patches_per_tile=64)
            pooled_embeddings[doc["id"]] = pooled.astype(np.float32)

    if return_pooled:
        return doc_embeddings, pooled_embeddings
    return doc_embeddings


def embed_queries(
    queries: List[Dict],
    embedder,
) -> Dict[str, np.ndarray]:
    """Embed all queries."""
    logger.info(f"Embedding {len(queries)} queries...")

    query_embeddings = {}
    for query in tqdm(queries, desc="Embedding queries"):
        emb = embedder.embed_query(query["text"])
        if hasattr(emb, "numpy"):
            emb = emb.numpy()
        elif hasattr(emb, "cpu"):
            emb = emb.cpu().numpy()
        query_embeddings[query["id"]] = np.array(emb, dtype=np.float32)

    return query_embeddings


def compute_maxsim(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """Compute ColBERT-style MaxSim score."""
    # Normalize
    query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8)
    doc_norm = doc_emb / (np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-8)

    # Compute similarity matrix
    sim_matrix = np.dot(query_norm, doc_norm.T)

    # MaxSim: max per query token, then sum
    max_sims = sim_matrix.max(axis=1)
    return float(max_sims.sum())


def search_exhaustive(
    query_emb: np.ndarray,
    doc_embeddings: Dict[str, np.ndarray],
    top_k: int = 10,
) -> List[Dict]:
    """Exhaustive MaxSim search over all documents."""
    scores = []
    for doc_id, doc_emb in doc_embeddings.items():
        score = compute_maxsim(query_emb, doc_emb)
        scores.append({"id": doc_id, "score": score})

    # Sort by score
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_k]


def search_two_stage(
    query_emb: np.ndarray,
    doc_embeddings: Dict[str, np.ndarray],
    pooled_embeddings: Dict[str, np.ndarray],
    prefetch_k: int = 100,
    top_k: int = 10,
) -> List[Dict]:
    """
    Two-stage retrieval: tile-level pooled prefetch + MaxSim rerank.

    Stage 1: Use tile-level pooled vectors for fast retrieval
             Each doc has [num_tiles, 128] pooled representation
             Compute MaxSim on pooled vectors (much faster)

    Stage 2: Exact MaxSim reranking on top candidates
             Use full multi-vector embeddings for precision
    """
    # Stage 1: Pooled MaxSim (fast approximation)
    # Query pooled: mean across query tokens → [128]
    query_pooled = query_emb.mean(axis=0)
    query_pooled = query_pooled / (np.linalg.norm(query_pooled) + 1e-8)

    stage1_scores = []
    for doc_id, doc_pooled in pooled_embeddings.items():
        # doc_pooled shape: [num_tiles, 128] from tile-level pooling
        # Compute similarity with each tile, take max (simplified MaxSim)
        doc_norm = doc_pooled / (np.linalg.norm(doc_pooled, axis=1, keepdims=True) + 1e-8)
        tile_sims = np.dot(doc_norm, query_pooled)
        score = float(tile_sims.max())  # Max tile similarity
        stage1_scores.append({"id": doc_id, "score": score})

    stage1_scores.sort(key=lambda x: x["score"], reverse=True)
    candidates = stage1_scores[:prefetch_k]

    # Stage 2: Exact MaxSim rerank on candidates
    reranked = []
    for cand in candidates:
        doc_id = cand["id"]
        doc_emb = doc_embeddings[doc_id]
        score = compute_maxsim(query_emb, doc_emb)
        reranked.append(
            {
                "id": doc_id,
                "score": score,
                "stage1_score": cand["score"],
                "stage1_rank": stage1_scores.index(cand) + 1,
            }
        )

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]


def compute_metrics(
    results: Dict[str, List[Dict]],
    qrels: Dict[str, Dict[str, int]],
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    ndcg_5 = []
    ndcg_10 = []
    mrr_10 = []
    recall_5 = []
    recall_10 = []

    for query_id, ranking in results.items():
        relevant = set(qrels.get(query_id, {}).keys())

        # MRR@10
        rr = 0.0
        for i, doc in enumerate(ranking[:10]):
            if doc["id"] in relevant:
                rr = 1.0 / (i + 1)
                break
        mrr_10.append(rr)

        # Recall@5, Recall@10
        retrieved_5 = set(d["id"] for d in ranking[:5])
        retrieved_10 = set(d["id"] for d in ranking[:10])

        if relevant:
            recall_5.append(len(retrieved_5 & relevant) / len(relevant))
            recall_10.append(len(retrieved_10 & relevant) / len(relevant))

        # NDCG@5, NDCG@10
        dcg_5 = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranking[:5]) if d["id"] in relevant)
        dcg_10 = sum(
            1.0 / np.log2(i + 2) for i, d in enumerate(ranking[:10]) if d["id"] in relevant
        )

        # Ideal DCG
        k_rel = min(len(relevant), 5)
        idcg_5 = sum(1.0 / np.log2(i + 2) for i in range(k_rel))
        k_rel = min(len(relevant), 10)
        idcg_10 = sum(1.0 / np.log2(i + 2) for i in range(k_rel))

        ndcg_5.append(dcg_5 / idcg_5 if idcg_5 > 0 else 0.0)
        ndcg_10.append(dcg_10 / idcg_10 if idcg_10 > 0 else 0.0)

    return {
        "ndcg@5": float(np.mean(ndcg_5)),
        "ndcg@10": float(np.mean(ndcg_10)),
        "mrr@10": float(np.mean(mrr_10)),
        "recall@5": float(np.mean(recall_5)),
        "recall@10": float(np.mean(recall_10)),
    }


def run_evaluation(
    dataset_name: str,
    model_name: str = "vidore/colSmol-500M",
    two_stage: bool = False,
    prefetch_k: int = 100,
    top_k: int = 10,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full evaluation on a dataset."""
    from visual_rag.embedding import VisualEmbedder

    logger.info("=" * 60)
    logger.info(f"Evaluating: {dataset_name}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Two-stage: {two_stage}")
    logger.info("=" * 60)

    # Load dataset
    data = load_dataset(dataset_name)

    # Initialize embedder
    embedder = VisualEmbedder(model_name=model_name)

    # Embed documents (with tile-level pooling if two-stage)
    start_time = time.time()
    if two_stage:
        doc_embeddings, pooled_embeddings = embed_documents(
            data["documents"], embedder, return_pooled=True
        )
        logger.info("Using tile-level pooling for two-stage retrieval")
    else:
        doc_embeddings = embed_documents(data["documents"], embedder)
        pooled_embeddings = None
    embed_time = time.time() - start_time
    logger.info(f"Document embedding time: {embed_time:.2f}s")

    # Embed queries
    query_embeddings = embed_queries(data["queries"], embedder)

    # Run search
    logger.info("Running search...")
    results = {}
    search_times = []

    for query in tqdm(data["queries"], desc="Searching"):
        query_id = query["id"]
        query_emb = query_embeddings[query_id]

        start = time.time()
        if two_stage:
            ranking = search_two_stage(
                query_emb, doc_embeddings, pooled_embeddings, prefetch_k=prefetch_k, top_k=top_k
            )
        else:
            ranking = search_exhaustive(query_emb, doc_embeddings, top_k=top_k)
        search_times.append(time.time() - start)

        results[query_id] = ranking

    avg_search_time = np.mean(search_times)
    logger.info(f"Average search time: {avg_search_time * 1000:.2f}ms")

    # Compute metrics
    metrics = compute_metrics(results, data["qrels"])
    metrics["avg_search_time_ms"] = avg_search_time * 1000
    metrics["embed_time_s"] = embed_time

    logger.info("\nResults:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        dataset_short = dataset_name.split("/")[-1]
        suffix = "_twostage" if two_stage else ""
        result_file = output_path / f"{dataset_short}{suffix}.json"

        with open(result_file, "w") as f:
            json.dump(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "two_stage": two_stage,
                    "metrics": metrics,
                },
                f,
                indent=2,
            )

        logger.info(f"Saved results to: {result_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="ViDoRe Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets:
  {', '.join(VIDORE_DATASETS.keys())}

Examples:
  # Quick test on DocVQA
  python run_vidore.py --dataset docvqa

  # Quick test with two-stage (your novel approach)
  python run_vidore.py --dataset docvqa --two-stage

  # Run on recommended quick datasets
  python run_vidore.py --quick

  # Full evaluation on all datasets
  python run_vidore.py --all

  # Compare exhaustive vs two-stage
  python run_vidore.py --dataset docvqa
  python run_vidore.py --dataset docvqa --two-stage
  python analyze_results.py --results results/ --compare
""",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(VIDORE_DATASETS.keys()),
        help=f"Dataset to evaluate: {', '.join(VIDORE_DATASETS.keys())}",
    )
    parser.add_argument(
        "--quick", action="store_true", help=f"Run on quick datasets: {QUICK_DATASETS}"
    )
    parser.add_argument("--all", action="store_true", help="Evaluate on all ViDoRe datasets")
    parser.add_argument(
        "--model",
        type=str,
        default="vidore/colSmol-500M",
        help="Model: vidore/colSmol-500M (default), vidore/colpali-v1.3, vidore/colqwen2-v1.0",
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Use two-stage retrieval (tile-level pooled prefetch + MaxSim rerank)",
    )
    parser.add_argument(
        "--prefetch-k", type=int, default=100, help="Stage 1 candidates (default: 100)"
    )
    parser.add_argument("--top-k", type=int, default=10, help="Final results (default: 10)")
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory (default: results)"
    )

    args = parser.parse_args()

    # Determine which datasets to run
    if args.all:
        dataset_keys = ALL_DATASETS
    elif args.quick:
        dataset_keys = QUICK_DATASETS
    elif args.dataset:
        dataset_keys = [args.dataset]
    else:
        parser.error("Specify --dataset, --quick, or --all")

    # Convert keys to full HuggingFace paths
    datasets = [VIDORE_DATASETS[k] for k in dataset_keys]
    logger.info(f"Running on {len(datasets)} dataset(s): {dataset_keys}")

    all_results = {}
    for dataset in datasets:
        try:
            metrics = run_evaluation(
                dataset_name=dataset,
                model_name=args.model,
                two_stage=args.two_stage,
                prefetch_k=args.prefetch_k,
                top_k=args.top_k,
                output_dir=args.output_dir,
            )
            all_results[dataset] = metrics
        except Exception as e:
            logger.error(f"Failed on {dataset}: {e}")
            continue

    # Summary
    if len(all_results) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)

        avg_ndcg10 = np.mean([m["ndcg@10"] for m in all_results.values()])
        avg_mrr10 = np.mean([m["mrr@10"] for m in all_results.values()])

        logger.info(f"Average NDCG@10: {avg_ndcg10:.4f}")
        logger.info(f"Average MRR@10: {avg_mrr10:.4f}")


if __name__ == "__main__":
    main()
