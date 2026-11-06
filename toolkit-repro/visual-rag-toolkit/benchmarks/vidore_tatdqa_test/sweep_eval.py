import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from benchmarks.vidore_tatdqa_test.dataset_loader import load_vidore_dataset_auto
from benchmarks.vidore_tatdqa_test.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from visual_rag import VisualEmbedder
from visual_rag.retrieval import MultiVectorRetriever

logger = logging.getLogger(__name__)


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if Path(".env").exists():
        load_dotenv(".env")


def _torch_dtype_to_str(dtype) -> str:
    if dtype is None:
        return "auto"
    s = str(dtype)
    return s.replace("torch.", "")


def _parse_torch_dtype(dtype_str: str):
    if dtype_str == "auto":
        return None
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_str]


def _infer_collection_vector_dtype(*, client, collection_name: str) -> Optional[str]:
    try:
        info = client.get_collection(collection_name)
    except Exception:
        return None
    vectors = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
    if not vectors:
        return None

    initial = None
    if isinstance(vectors, dict):
        initial = vectors.get("initial")
    else:
        try:
            initial = vectors.get("initial")
        except Exception:
            initial = None

    dt = getattr(initial, "datatype", None) if initial is not None else None
    if dt is None:
        return None

    s = str(dt).lower()
    if "float16" in s:
        return "float16"
    if "float32" in s:
        return "float32"
    return None


def _evaluate(
    *,
    retriever: MultiVectorRetriever,
    queries: List,
    qrels: Dict[str, Dict[str, int]],
    top_k: int,
    mode: str,
    stage1_mode: str,
    prefetch_k: int,
    max_queries: int = 0,
    precomputed_query_embeddings: Optional[List[np.ndarray]] = None,
) -> Dict[str, float]:
    ndcg10: List[float] = []
    mrr10: List[float] = []
    recall10: List[float] = []
    recall5: List[float] = []
    latencies_ms: List[float] = []

    if max_queries and max_queries > 0:
        queries = queries[:max_queries]

    iterator = queries
    try:
        from tqdm import tqdm

        iterator = tqdm(queries, desc=f"🔎 Evaluating ({mode})", unit="q")
    except ImportError:
        pass

    for idx, q in enumerate(iterator):
        start = time.time()
        if precomputed_query_embeddings is None:
            results = retriever.search(
                query=q.text,
                top_k=top_k,
                mode=mode,
                prefetch_k=prefetch_k,
                stage1_mode=stage1_mode,
            )
        else:
            query_embedding = precomputed_query_embeddings[idx]
            if mode == "single_full":
                results = retriever._single_stage.search(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    strategy="multi_vector",
                )
            elif mode == "two_stage":
                results = retriever._two_stage.search(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    prefetch_k=prefetch_k,
                    stage1_mode=stage1_mode,
                )
            else:
                raise ValueError(f"Unsupported mode for precomputed embeddings: {mode}")
        latencies_ms.append((time.time() - start) * 1000.0)

        ranking = [str(r["id"]) for r in results]
        rels = qrels.get(q.query_id, {})

        ndcg10.append(ndcg_at_k(ranking, rels, k=10))
        mrr10.append(mrr_at_k(ranking, rels, k=10))
        recall5.append(recall_at_k(ranking, rels, k=5))
        recall10.append(recall_at_k(ranking, rels, k=10))

    return {
        "ndcg@10": float(np.mean(ndcg10)),
        "mrr@10": float(np.mean(mrr10)),
        "recall@5": float(np.mean(recall5)),
        "recall@10": float(np.mean(recall10)),
        "avg_latency_ms": float(np.mean(latencies_ms)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="vidore/tatdqa_test")
    parser.add_argument("--collection", type=str, default="vidore_tatdqa_test")
    parser.add_argument("--model", type=str, default="vidore/colSmol-500M")
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype for model weights (default: auto; inferred from collection vector dtype when possible).",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--mode", type=str, default="two_stage", choices=["single_full", "two_stage"]
    )
    parser.add_argument(
        "--stage1-mode",
        type=str,
        default="tokens_vs_standard_pooling",
        choices=[
            "pooled_query_vs_standard_pooling",
            "tokens_vs_standard_pooling",
            "pooled_query_vs_experimental_pooling",
            "tokens_vs_experimental_pooling",
            "pooled_query_vs_global",
            # Backwards-compatible aliases
            "pooled_query_vs_tiles",
            "tokens_vs_tiles",
            "pooled_query_vs_experimental",
            "tokens_vs_experimental",
        ],
    )
    parser.add_argument(
        "--prefetch-ks",
        type=str,
        default="20,50,100,200,400",
        help="Comma-separated list of prefetch_k values (only used for mode=two_stage).",
    )
    parser.add_argument("--prefer-grpc", action="store_true")
    parser.add_argument("--out-dir", type=str, default="results/sweeps")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Limit number of queries for a quick smoke test (0 = all).",
    )
    parser.add_argument(
        "--sample-queries",
        type=int,
        default=0,
        help="Sample N queries for sweeps (0 = disable).",
    )
    parser.add_argument(
        "--sample-strategy",
        type=str,
        default="head",
        choices=["head", "random"],
        help="How to sample queries when --sample-queries is set.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for --sample-strategy random.",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=32,
        help="Batch size for embedding queries. Set 0 to disable pre-embedding and embed per query.",
    )
    args = parser.parse_args()

    _maybe_load_dotenv()

    if not os.getenv("QDRANT_URL"):
        raise ValueError("QDRANT_URL not set. Add it to .env or export it in your shell.")

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Collection: {args.collection}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Mode: {args.mode}")
    if args.mode == "two_stage":
        logger.info(f"Stage1 mode: {args.stage1_mode}")
        logger.info(f"Prefetch ks: {args.prefetch_ks}")
    if args.max_queries:
        logger.info(f"Max queries (smoke test): {args.max_queries}")

    from qdrant_client import QdrantClient

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=args.prefer_grpc,
        check_compatibility=False,
        timeout=120,
    )

    requested_torch_dtype = args.torch_dtype
    if requested_torch_dtype == "auto":
        vdt = _infer_collection_vector_dtype(client=client, collection_name=args.collection)
        if vdt == "float16":
            requested_torch_dtype = "float16"
        elif vdt == "float32":
            requested_torch_dtype = "float32"

    torch_dtype = _parse_torch_dtype(requested_torch_dtype)

    corpus, queries, qrels, protocol = load_vidore_dataset_auto(args.dataset)
    del corpus
    logger.info(f"Loaded protocol={protocol}, queries={len(queries)}")

    if args.max_queries and args.max_queries > 0:
        queries = queries[: args.max_queries]
    if args.sample_queries and args.sample_queries > 0:
        if args.sample_strategy == "head":
            queries = queries[: args.sample_queries]
        else:
            rng = np.random.default_rng(int(args.sample_seed))
            n = min(int(args.sample_queries), len(queries))
            idxs = rng.choice(len(queries), size=n, replace=False).tolist()
            queries = [queries[i] for i in idxs]
    logger.info(f"Eval queries: {len(queries)}")

    embedder = VisualEmbedder(model_name=args.model, torch_dtype=torch_dtype)
    logger.info(f"Effective torch dtype: {_torch_dtype_to_str(embedder.torch_dtype)}")

    retriever = MultiVectorRetriever(
        collection_name=args.collection,
        embedder=embedder,
        prefer_grpc=args.prefer_grpc,
        qdrant_client=client,
    )

    precomputed_query_embeddings: Optional[List[np.ndarray]] = None
    if args.query_batch_size and args.query_batch_size > 0:
        texts = [q.text for q in queries]
        logger.info(f"Pre-embedding {len(texts)} queries (batch={args.query_batch_size})...")
        q_tensors = embedder.embed_queries(
            texts, batch_size=args.query_batch_size, show_progress=True
        )
        precomputed_query_embeddings = [t.detach().cpu().float().numpy() for t in q_tensors]
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    out_dir = Path(args.out_dir) / args.collection
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "single_full":
        metrics = _evaluate(
            retriever=retriever,
            queries=queries,
            qrels=qrels,
            top_k=args.top_k,
            mode="single_full",
            stage1_mode=args.stage1_mode,
            prefetch_k=0,
            max_queries=args.max_queries,
            precomputed_query_embeddings=precomputed_query_embeddings,
        )
        out_path = out_dir / f"{protocol}__single_full__top{args.top_k}.json"
        with open(out_path, "w") as f:
            json.dump(
                {
                    "dataset": args.dataset,
                    "protocol": protocol,
                    "collection": args.collection,
                    "model": args.model,
                    "torch_dtype": _torch_dtype_to_str(embedder.torch_dtype),
                    "mode": "single_full",
                    "top_k": args.top_k,
                    "max_queries": args.max_queries,
                    "sample_queries": args.sample_queries,
                    "sample_strategy": args.sample_strategy if args.sample_queries else None,
                    "sample_seed": (
                        args.sample_seed
                        if args.sample_queries and args.sample_strategy == "random"
                        else None
                    ),
                    "metrics": metrics,
                },
                f,
                indent=2,
            )
        print(out_path)
        print(json.dumps(metrics, indent=2))
        return

    prefetch_ks = [int(x.strip()) for x in args.prefetch_ks.split(",") if x.strip()]
    for k in prefetch_ks:
        metrics = _evaluate(
            retriever=retriever,
            queries=queries,
            qrels=qrels,
            top_k=args.top_k,
            mode="two_stage",
            stage1_mode=args.stage1_mode,
            prefetch_k=k,
            max_queries=args.max_queries,
            precomputed_query_embeddings=precomputed_query_embeddings,
        )
        out_path = (
            out_dir
            / f"{protocol}__two_stage__{args.stage1_mode}__prefetch{k}__top{args.top_k}.json"
        )
        with open(out_path, "w") as f:
            json.dump(
                {
                    "dataset": args.dataset,
                    "protocol": protocol,
                    "collection": args.collection,
                    "model": args.model,
                    "mode": "two_stage",
                    "stage1_mode": args.stage1_mode,
                    "prefetch_k": k,
                    "top_k": args.top_k,
                    "torch_dtype": _torch_dtype_to_str(embedder.torch_dtype),
                    "max_queries": args.max_queries,
                    "sample_queries": args.sample_queries,
                    "sample_strategy": args.sample_strategy if args.sample_queries else None,
                    "sample_seed": (
                        args.sample_seed
                        if args.sample_queries and args.sample_strategy == "random"
                        else None
                    ),
                    "metrics": metrics,
                },
                f,
                indent=2,
            )
        print(out_path)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
