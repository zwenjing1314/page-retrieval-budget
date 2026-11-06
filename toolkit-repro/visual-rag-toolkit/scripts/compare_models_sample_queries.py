"""
Compare retrieval quality across two model+collection pairs on the same dataset queries.

This is a read-only diagnostic:
- Loads BEIR dataset (queries + qrels)
- Remaps qrels doc_ids -> Qdrant point IDs for each collection
- Runs retrieval for a sample of queries
- Computes simple hit-rate statistics + per-query best-rank
- Writes a JSON report under results/model_compare/

Example:
  python scripts/compare_models_sample_queries.py \\
    --dataset vidore/esg_reports_v2 \\
    --top-k 100 \\
    --max-queries 50
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qdrant_client.http import models as qm

from benchmarks.vidore_tatdqa_test.dataset_loader import load_vidore_beir_dataset
from visual_rag import VisualEmbedder
from visual_rag.retrieval import MultiVectorRetriever


def _stable_uuid(text: str) -> str:
    hex_str = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"


def _union_point_id(*, dataset_name: str, source_doc_id: str, union_namespace: str) -> str:
    return _stable_uuid(f"{union_namespace}::{dataset_name}::{source_doc_id}")


def _get_qdrant_env() -> Tuple[str, Optional[str]]:
    url = os.getenv("QDRANT_URL") or os.getenv("DEST_QDRANT_URL") or os.getenv("SIGIR_QDRANT_URL")
    if not url:
        raise SystemExit("QDRANT_URL not set")
    key = (
        os.getenv("QDRANT_API_KEY")
        or os.getenv("DEST_QDRANT_API_KEY")
        or os.getenv("SIGIR_QDRANT_KEY")
    )
    return str(url), (str(key) if key else None)


@dataclass(frozen=True)
class RunSpec:
    name: str
    model: str
    collection: str
    torch_dtype: str
    output_dtype: str


SPECS: List[RunSpec] = [
    RunSpec(
        name="colqwen2.5_fp16_collection",
        model="vidore/colqwen2.5-v0.2",
        collection="vidore_beir_v2_3ds__colqwen25_v0_2__nocrop__union__fp16",
        torch_dtype="float16",
        output_dtype="float16",
    ),
    RunSpec(
        name="colpali1.3_collection",
        model="vidore/colpali-v1.3",
        collection="vidore_beir_v2_3ds__colpali_v1_3__nocrop__union",
        torch_dtype="float16",
        output_dtype="float16",
    ),
]


def _parse_dtype(s: str):
    if s == "float16":
        import torch

        return torch.float16
    if s == "float32":
        import torch

        return torch.float32
    if s == "bfloat16":
        import torch

        return torch.bfloat16
    return None


def _np_dtype(s: str):
    return np.float16 if s == "float16" else np.float32


def _build_remapped_qrels(
    *, corpus, qrels, dataset_name: str, collection: str
) -> Dict[str, Dict[str, int]]:
    # corpus doc_id values are stable_uuid(source_doc_id)
    id_map: Dict[str, str] = {}
    for doc in corpus:
        source_doc_id = str((doc.payload or {}).get("source_doc_id") or doc.doc_id)
        id_map[str(doc.doc_id)] = _union_point_id(
            dataset_name=str(dataset_name),
            source_doc_id=str(source_doc_id),
            union_namespace=str(collection),
        )

    remapped: Dict[str, Dict[str, int]] = {}
    for qid, rels in (qrels or {}).items():
        out: Dict[str, int] = {}
        for did, score in (rels or {}).items():
            mapped = id_map.get(str(did))
            if mapped:
                out[str(mapped)] = int(score)
        if out:
            remapped[str(qid)] = out
    return remapped


def _rank_stats_for_query(
    *, ranking: List[str], qrels: Dict[str, int], top_k: int
) -> Dict[str, Any]:
    relset = {did for did, s in (qrels or {}).items() if int(s) > 0}
    best_rank = None
    for i, did in enumerate(ranking[:top_k]):
        if str(did) in relset:
            best_rank = i + 1
            break
    return {
        "num_relevant": int(len(relset)),
        "best_rank": int(best_rank) if best_rank is not None else None,
        "hit@1": bool(best_rank == 1),
        "hit@5": bool(best_rank is not None and best_rank <= 5),
        "hit@10": bool(best_rank is not None and best_rank <= 10),
        "hit@100": bool(best_rank is not None and best_rank <= 100),
    }


def _run_one(
    *,
    spec: RunSpec,
    dataset_name: str,
    corpus,
    queries,
    qrels,
    top_k: int,
    max_queries: int,
    prefer_grpc: bool,
    timeout: int,
) -> Dict[str, Any]:
    url, key = _get_qdrant_env()

    remapped_qrels = _build_remapped_qrels(
        corpus=corpus, qrels=qrels, dataset_name=dataset_name, collection=spec.collection
    )
    # Keep only queries with at least one positive relevant doc
    kept = [
        q for q in queries if any(v > 0 for v in remapped_qrels.get(str(q.query_id), {}).values())
    ]
    kept = kept[: int(max_queries)] if int(max_queries) > 0 else kept

    flt = qm.Filter(
        must=[qm.FieldCondition(key="dataset", match=qm.MatchValue(value=str(dataset_name)))]
    )

    embedder = VisualEmbedder(
        model_name=str(spec.model),
        torch_dtype=_parse_dtype(spec.torch_dtype),
        output_dtype=_np_dtype(spec.output_dtype),
    )
    retriever = MultiVectorRetriever(
        collection_name=str(spec.collection),
        model_name=str(spec.model),
        embedder=embedder,
        qdrant_url=url,
        qdrant_api_key=key,
        prefer_grpc=bool(prefer_grpc),
        request_timeout=int(timeout),
    )

    per_query: Dict[str, Any] = {}
    hits1 = hits5 = hits10 = hits100 = 0
    best_ranks: List[int] = []
    for q in kept:
        qid = str(q.query_id)
        rels = remapped_qrels.get(qid, {})
        res = retriever.search(q.text, top_k=int(top_k), mode="single_full", filter_obj=flt)
        ranking = [str(r["id"]) for r in (res or [])]
        st = _rank_stats_for_query(ranking=ranking, qrels=rels, top_k=int(top_k))
        per_query[qid] = {
            "text": str(q.text),
            "stats": st,
            "top10": ranking[:10],
        }
        hits1 += 1 if st["hit@1"] else 0
        hits5 += 1 if st["hit@5"] else 0
        hits10 += 1 if st["hit@10"] else 0
        hits100 += 1 if st["hit@100"] else 0
        if st["best_rank"] is not None:
            best_ranks.append(int(st["best_rank"]))

    n = max(len(kept), 1)
    summary = {
        "queries_eval": int(len(kept)),
        "hit_rate@1": float(hits1 / n),
        "hit_rate@5": float(hits5 / n),
        "hit_rate@10": float(hits10 / n),
        "hit_rate@100": float(hits100 / n),
        "median_best_rank": float(np.median(best_ranks)) if best_ranks else None,
        "mean_best_rank": float(np.mean(best_ranks)) if best_ranks else None,
    }

    # Best-effort release memory
    try:
        import torch

        del retriever
        del embedder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    return {
        "spec": spec.__dict__,
        "summary": summary,
        "per_query": per_query,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="vidore/esg_reports_v2")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--max-queries", type=int, default=50)
    ap.add_argument("--prefer-grpc", action="store_true", default=True)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", default="auto")
    args = ap.parse_args()

    dataset_name = str(args.dataset)
    corpus, queries, qrels = load_vidore_beir_dataset(dataset_name)

    out_dir = Path("results") / "model_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        args.out
        if str(args.out) != "auto"
        else f"compare__{dataset_name.replace('/', '_')}__top{int(args.top_k)}__q{int(args.max_queries)}.json"
    )

    out: Dict[str, Any] = {
        "dataset": dataset_name,
        "top_k": int(args.top_k),
        "max_queries": int(args.max_queries),
        "runs": {},
    }
    for spec in SPECS:
        out["runs"][spec.name] = _run_one(
            spec=spec,
            dataset_name=dataset_name,
            corpus=corpus,
            queries=queries,
            qrels=qrels,
            top_k=int(args.top_k),
            max_queries=int(args.max_queries),
            prefer_grpc=bool(args.prefer_grpc),
            timeout=int(args.timeout),
        )

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Wrote: {out_path}")
    print(json.dumps({k: v["summary"] for k, v in out["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()
