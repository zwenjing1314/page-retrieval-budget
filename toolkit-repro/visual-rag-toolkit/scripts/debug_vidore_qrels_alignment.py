"""
Debug ViDoRe-v2 evaluation mismatches between qrels and Qdrant point IDs.

This script helps answer:
- Are relevant docs (from qrels) actually present in Qdrant?
- Are we mapping qrels doc IDs to the correct Qdrant point IDs?
- Does per_dataset filtering actually reduce the search space?
- If docs exist, at what rank do they appear for single_full retrieval?

Typical use:
  python scripts/debug_vidore_qrels_alignment.py \\
    --dataset vidore/esg_reports_v2 \\
    --collection vidore_beir_v2_3ds__colqwen25_v0_2__nocrop__union__fp32 \\
    --model vidore/colqwen2.5-v0.2 \\
    --max-queries 5 \\
    --top-k 200 \\
    --no-prefer-grpc
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from benchmarks.vidore_tatdqa_test.dataset_loader import load_vidore_beir_dataset
from visual_rag import VisualEmbedder
from visual_rag.retrieval import MultiVectorRetriever


def _stable_uuid(text: str) -> str:
    import hashlib

    hex_str = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"


def _union_point_id(*, dataset_name: str, source_doc_id: str, union_namespace: str) -> str:
    return _stable_uuid(f"{union_namespace}::{dataset_name}::{source_doc_id}")


def _infer_qdrant_conn(prefer_grpc: bool, timeout: int) -> QdrantClient:
    url = os.getenv("QDRANT_URL") or os.getenv("DEST_QDRANT_URL") or os.getenv("SIGIR_QDRANT_URL")
    if not url:
        raise SystemExit("QDRANT_URL not set")
    key = (
        os.getenv("QDRANT_API_KEY")
        or os.getenv("DEST_QDRANT_API_KEY")
        or os.getenv("SIGIR_QDRANT_KEY")
    )
    return QdrantClient(
        url=url,
        api_key=key,
        prefer_grpc=bool(prefer_grpc),
        timeout=int(timeout),
        check_compatibility=False,
    )


def _count_by_dataset(client: QdrantClient, collection: str, dataset: str) -> Tuple[int, int]:
    # exact counts can be slow; we keep it exact for correctness.
    all_cnt = client.count(collection_name=collection, exact=True).count
    ds_cnt = client.count(
        collection_name=collection,
        count_filter=qm.Filter(
            must=[qm.FieldCondition(key="dataset", match=qm.MatchValue(value=str(dataset)))]
        ),
        exact=True,
    ).count
    return int(ds_cnt), int(all_cnt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--max-queries", type=int, default=5)
    ap.add_argument("--prefer-grpc", action="store_true", default=False)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument(
        "--torch-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"]
    )
    args = ap.parse_args()

    corpus, queries, qrels = load_vidore_beir_dataset(str(args.dataset))
    print(
        f"Loaded dataset={args.dataset}: corpus={len(corpus)} queries={len(queries)} qrels_q={len(qrels)}"
    )

    # Build mapping exactly like the benchmark does.
    id_map: Dict[str, str] = {}
    for doc in corpus:
        src = str((doc.payload or {}).get("source_doc_id") or doc.doc_id)
        id_map[str(doc.doc_id)] = _union_point_id(
            dataset_name=str(args.dataset),
            source_doc_id=str(src),
            union_namespace=str(args.collection),
        )

    remapped_qrels: Dict[str, Dict[str, int]] = {}
    for qid, rels in qrels.items():
        out: Dict[str, int] = {}
        for did, score in rels.items():
            if int(score) <= 0:
                continue
            mapped = id_map.get(str(did))
            if mapped:
                out[str(mapped)] = int(score)
        if out:
            remapped_qrels[str(qid)] = out

    # Connectivity + counts
    client = _infer_qdrant_conn(bool(args.prefer_grpc), int(args.timeout))
    ds_cnt, all_cnt = _count_by_dataset(client, str(args.collection), str(args.dataset))
    print(f"Qdrant counts: dataset={ds_cnt} / all={all_cnt} (collection={args.collection})")

    # Pick queries that still have qrels after remap
    kept = [q for q in queries if str(q.query_id) in remapped_qrels]
    kept = kept[: int(args.max_queries)]
    print(f"Queries kept after qrels remap: {len(kept)} (showing up to {args.max_queries})")

    # Build retriever with the exact same embedder/retrieval path.
    td = None
    if str(args.torch_dtype) != "auto":
        import torch

        td = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
            str(args.torch_dtype)
        ]
    embedder = VisualEmbedder(model_name=str(args.model), torch_dtype=td)
    retriever = MultiVectorRetriever(
        collection_name=str(args.collection),
        model_name=str(args.model),
        embedder=embedder,
        qdrant_client=client,
        prefer_grpc=bool(args.prefer_grpc),
        request_timeout=int(args.timeout),
    )

    flt = qm.Filter(
        must=[qm.FieldCondition(key="dataset", match=qm.MatchValue(value=str(args.dataset)))]
    )

    for i, q in enumerate(kept):
        qid = str(q.query_id)
        rels = remapped_qrels.get(qid, {})
        # Only positive qrels are truly relevant.
        rel_ids = [rid for rid, s in (rels or {}).items() if int(s) > 0]
        print("\n" + "-" * 90)
        print(f"Q{i}: {qid}  text={q.text[:120]!r}")
        print(f"  relevant_ids(remapped)={len(rel_ids)}  sample={rel_ids[:3]}")

        # Check if relevant IDs exist in Qdrant at all
        exists = 0
        try:
            recs = client.retrieve(
                collection_name=str(args.collection),
                ids=rel_ids[:20],
                with_payload=False,
                with_vectors=False,
                timeout=int(args.timeout),
            )
            exists = len(recs)
        except Exception:
            exists = 0
        print(f"  relevant_ids_exist_in_qdrant(sample<=20): {exists}")

        # Search per_dataset filter
        res = retriever.search(q.text, top_k=int(args.top_k), mode="single_full", filter_obj=flt)
        ranked = [str(r["id"]) for r in res]
        # Find best rank of any relevant doc
        best_rank = None
        for rid in rel_ids:
            if rid in ranked:
                rnk = ranked.index(rid) + 1
                best_rank = rnk if best_rank is None else min(best_rank, rnk)
        print(f"  best_rank_in_top{args.top_k} (per_dataset filter): {best_rank}")
        print(f"  top10 ids: {ranked[:10]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
