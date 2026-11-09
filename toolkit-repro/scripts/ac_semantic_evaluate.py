"""Evaluate the cached hierarchical semantic-pooling representation.

This script is deliberately cache-only: it never loads the model, encodes pages
or queries, or writes to an existing Qdrant collection.  Use --query-limit 2
for the first smoke check, then omit it for the fixed 232-query evaluation.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pyarrow.parquet as pq
from importlib.metadata import version
from qdrant_client import QdrantClient, models

from prb_cache import CacheStore, digest, json_bytes
from benchmarks.vidore_tatdqa_test.metrics import ndcg_at_k, recall_at_k
from visual_rag.retrieval.multi_vector import MultiVectorRetriever


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/encoded-economics-v1"
H_RECORD = ROOT / "records/ac-semantic-full-cache-20261109-124408-721631"
A_RECORD = ROOT / "records/p2-formal-a-20261107-012250-612a76"
FIELD = "semantic_hierarchical_30"
DEFAULT_COLLECTION = "prb_ac_h_full_176185423f13"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_bytes(json_bytes(value))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def metrics(ranking, rels):
    return {
        **{f"NDCG@{k}": ndcg_at_k(ranking, rels, k) for k in (5, 10)},
        **{f"Recall@{k}": recall_at_k(ranking, rels, k) for k in (5, 10, 100)},
    }


def local_reference(query, h_vectors):
    q = query.astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    scores = {}
    for corpus_id, pooled in h_vectors.items():
        p = pooled.astype(np.float32)
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        scores[corpus_id] = float((q @ p.T).max(axis=1).sum(dtype=np.float32))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--query-limit", type=int, choices=(2, 232), default=2)
    args = parser.parse_args()

    record = ROOT / "records" / (
        "ac-semantic-eval-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        + "-" + uuid.uuid4().hex[:6]
    )
    record.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "STARTED",
        "collection": args.collection,
        "field": FIELD,
        "query_limit": args.query_limit,
        "scope": "Cached H representation; no model/encoding/pooling",
        "model_loads": 0,
        "encoding_calls": 0,
        "pooling_calls": 0,
        "search_calls": 0,
    }
    save(record / "report.json", report)
    print("Record:", record, flush=True)
    print("Collection:", args.collection, flush=True)

    client = None
    try:
        h_report = load(H_RECORD / "report.json")
        require(h_report["status"] == "PASS_FULL_CACHE" and h_report["completed_pages"] == 452,
                "Full H cache is not accepted")
        profile = load(CACHE / "profile.json")
        store = CacheStore(CACHE, profile, create=False)
        require(store.manifest()["complete"], "Shared cache is incomplete")

        a_report = load(A_RECORD / "report.json")
        require(a_report["status"] == "PASS" and a_report["counts"] == {
            "pages": 452, "queries": 232, "qrels": 3628
        }, "A evidence is not accepted")
        mapping = load(A_RECORD / "page-id-mapping.json")
        by_point = {str(x["point_id"]): x for x in mapping}
        by_corpus = {str(x["corpus_id"]): x for x in mapping}
        require(len(by_point) == len(by_corpus) == 452, "Invalid page mapping")

        h_vectors = {}
        for item in sorted(load(H_RECORD / "manifest.json"), key=lambda x: int(x["id"])):
            corpus_id = str(item["id"])
            with np.load(H_RECORD / "pages" / corpus_id / "vectors.npz", allow_pickle=False) as saved:
                pooled = saved["pooled"].copy()
            require(pooled.shape == (34, 128) and pooled.dtype == np.float16,
                    f"Invalid H vector for page {corpus_id}")
            h_vectors[corpus_id] = pooled
        require(len(h_vectors) == 452, "H vector count mismatch")

        data = ROOT / ("data/economics_reports_v2-" + profile["dataset_revision"])
        query_rows = pq.read_table(
            data / "queries/test-00000-of-00001.parquet",
            columns=["query-id", "query", "language"],
        ).to_pylist()
        official = {str(row["query-id"]): row for row in query_rows}
        require(len(official) == 232, "Query count mismatch")
        queries = {}
        for qid, row in official.items():
            arrays, meta = store.read("queries", qid)
            require(meta["identity"]["text"] == row["query"], f"Query text mismatch: {qid}")
            queries[qid] = arrays["vectors"]
        qrels = load(A_RECORD / "qrels.json")
        require(set(qrels) == set(queries), "Qrels/query IDs mismatch")

        client = QdrantClient(url="http://127.0.0.1:16333", trust_env=False, timeout=300)
        require(client.info().version == "1.14.1", "Unexpected Qdrant server version")
        info = client.get_collection(args.collection)
        require(info.status.value == "green", f"Collection status: {info.status}")
        require(client.count(args.collection, exact=True).count == 452, "Collection point count mismatch")
        require(set(info.config.params.vectors) == {FIELD}, "Stored field mismatch")
        vector = info.config.params.vectors[FIELD]
        require(vector.size == 128 and vector.distance == models.Distance.COSINE and
                vector.datatype == models.Datatype.FLOAT16 and
                vector.multivector_config.comparator == models.MultiVectorComparator.MAX_SIM,
                "H vector configuration mismatch")
        require(info.config.hnsw_config.m == 0 and info.config.hnsw_config.full_scan_threshold == 10,
                "H HNSW configuration mismatch")

        class CheckedClient:
            def __init__(self, actual):
                self.actual = actual
                self.calls = 0

            def __getattr__(self, name):
                return getattr(self.actual, name)

            def query_points(self, *args, **kwargs):
                require(kwargs.get("collection_name") == args_collection, "Wrong collection")
                require(kwargs.get("using") == FIELD, "Wrong H field")
                require(kwargs.get("limit") == 100, "Wrong top_k")
                require(kwargs.get("prefetch") is None and kwargs.get("query_filter") is None,
                        "Unexpected prefetch/filter")
                require(kwargs.get("with_payload") is True and kwargs.get("with_vectors") is False,
                        "Unexpected response fields")
                require(kwargs["search_params"].exact is True, "Search must be exact")
                self.calls += 1
                return self.actual.query_points(*args, **kwargs)

        args_collection = args.collection
        checked = CheckedClient(client)
        retriever = MultiVectorRetriever(
            collection_name=args.collection,
            qdrant_client=checked,
            embedder=object(),
            experimental_vector_name=FIELD,
            request_timeout=300,
            max_retries=1,
        )

        order = ["0", "20"] + [qid for qid in sorted(queries, key=int) if qid not in {"0", "20"}]
        order = order[:args.query_limit]
        rows = []
        rankings = {}
        for number, qid in enumerate(order, 1):
            query = queries[qid]
            started = time.perf_counter()
            hits = retriever.search_embedded(query_embedding=query, top_k=100,
                                             mode="single_experimental_tokens")
            seconds = time.perf_counter() - started
            require(len(hits) == 100, f"Unexpected hit count for query {qid}")
            ranking = [str(hit["id"]) for hit in hits]
            require(len(set(ranking)) == 100, f"Duplicate hit IDs for query {qid}")
            require(all(float(a["score_final"]) >= float(b["score_final"])
                        for a, b in zip(hits, hits[1:])), f"Unsorted hits for query {qid}")
            for hit in hits:
                point = by_point[str(hit["id"])]
                require(hit["payload"]["corpus_id"] == point["corpus_id"], "Payload mismatch")

            if qid in {"0", "20"}:
                refs = local_reference(query, h_vectors)
                q_tokens = len(query)
                max_error = max(
                    abs(float(hit["score_final"]) - refs[by_point[str(hit["id"])] ["corpus_id"]]) / q_tokens
                    for hit in hits
                )
                require(max_error <= 1e-3, f"Local H score mismatch for query {qid}: {max_error}")
                save(record / f"diagnostic-{qid}.json", {
                    "query_id": qid, "max_score_error_per_token": max_error,
                    "local_reference_scores": refs,
                })
                print(f"H query {qid} local score check: PASS", flush=True)

            # qrels are keyed by the stable point IDs, matching A/B/C.
            values = metrics(ranking, qrels[qid])
            rows.append({"query_id": qid, "query_tokens": len(query),
                         "search_call_seconds": seconds, **values})
            rankings[qid] = ranking
            if number <= 2 or number % 20 == 0 or number == len(order):
                print(f"Evaluated H: {number}/{len(order)}", flush=True)

        report.update({
            "status": "PASS",
            "completed_queries": len(rows),
            "search_calls": checked.calls,
            "counts": {"pages": 452, "queries": 232, "qrels": 3628},
            "aggregate_metrics": {
                name: math.fsum(row[name] for row in rows) / len(rows)
                for name in ("NDCG@5", "NDCG@10", "Recall@5", "Recall@10", "Recall@100")
            },
            "source_h_report_sha256": digest(H_RECORD / "report.json"),
            "source_a_report_sha256": digest(A_RECORD / "report.json"),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "collection_info": info.model_dump(mode="json"),
            "query_evaluation": "smoke_2" if args.query_limit == 2 else "full_232",
        })
        with (record / "per-query.tsv").open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        save(record / "rankings.json", rankings)
        save(record / "report.json", report)
        print("Metrics:", report["aggregate_metrics"], flush=True)
        print("Evaluation queries:", len(rows), flush=True)
        print("Model loads: 0; encoding calls: 0; pooling calls: 0; search calls:", checked.calls, flush=True)
        print("H semantic evaluation:", "SMOKE PASS" if args.query_limit == 2 else "PASS", flush=True)
        print("Record:", record, flush=True)
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
