import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient

from prb_index import build_collection, FIELD_SETS
from visual_rag.embedding.pooling import (
    colpali_row_mean_pooling,
    colpali_experimental_pooling_from_rows,
)
from visual_rag.retrieval.multi_vector import MultiVectorRetriever

root = Path(__file__).resolve().parents[1]
sources = list((root / "records").glob("p1-model-entry-*/report.json"))
assert sources, "Missing model-entry report"
source_report = max(sources, key=lambda p: p.stat().st_mtime)
source = source_report.parent
metadata = json.loads(source_report.read_text(encoding="utf-8"))
assert metadata["status"] == "PASS", "Latest model-entry check did not pass"

cache = source / "entry-vectors.npz"
assert hashlib.sha256(cache.read_bytes()).hexdigest() == metadata["cache_sha256"]
with np.load(cache, allow_pickle=False) as saved:
    page = saved["page_vectors"].copy()
    queries = [
        saved["query0_single"].copy(),
        saved["query1_single"].copy(),
    ]

record = root / "records" / (
    "p1-abc-fields-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    + "-" + uuid.uuid4().hex[:6]
)
record.mkdir(parents=True, exist_ok=False)
report = {
    "status": "STARTED",
    "source": str(source),
    "tolerance_per_query_token": 1e-3,
    "collections": {},
    "queries": [],
    "scope": "One cached real page plus two synthetic variants; diagnostic only",
}


def save():
    (record / "report.json").write_bytes(
        json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
    )


def reference(q, p):
    q = q.astype(np.float32)
    p = p.astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return float((q @ p.T).max(axis=1).sum(dtype=np.float32))


class NoEncoding:
    def embed_query(self, *args, **kwargs):
        raise AssertionError("This test must use cached queries")


class TracedClient:
    def __init__(self, client):
        self.client = client
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.client, name)

    def query_points(self, *args, **kwargs):
        params = kwargs.get("search_params")
        assert params is not None and params.exact is True
        prefetch = kwargs.get("prefetch")
        if prefetch is not None:
            entries = prefetch if isinstance(prefetch, list) else [prefetch]
            for entry in entries:
                assert entry.params is not None and entry.params.exact is True
        self.calls.append({
            "using": kwargs.get("using"),
            "has_prefetch": prefetch is not None,
        })
        return self.client.query_points(*args, **kwargs)


variants = [
    ("real-page-63", page, "real"),
    ("synthetic-negative", -page, "synthetic"),
    ("synthetic-dimension-roll", np.roll(page, 17, axis=1), "synthetic"),
]
points = []
for label, vectors, kind in variants:
    rows = colpali_row_mean_pooling(vectors, output_dtype=np.float16)
    pooled = colpali_experimental_pooling_from_rows(
        rows, window_size=3, output_dtype=np.float16
    )
    points.append({
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, "prb-p1-" + label)),
        "payload": {
            "kind": kind,
            "label": label,
            "source_corpus_id": 63,
            "source_doc_id": metadata["page"]["doc_id"],
            "diagnostic": True,
        },
        "vectors": {
            "initial": np.ascontiguousarray(vectors, dtype=np.float16),
            "experimental_pooling_3": pooled,
        },
    })

client = None
save()
print("Record:", record, flush=True)

try:
    client = QdrantClient(url="http://127.0.0.1:16333", timeout=60)
    assert client.info().version == "1.14.1"
    traced = TracedClient(client)
    retrievers = {}

    for mode in ("A", "B", "C"):
        name = "prb_p1_fields_" + mode.lower() + "_" + uuid.uuid4().hex[:10]
        report["collections"][mode] = name
        save()
        info = build_collection(client, name, mode, points)
        (record / (mode + "-collection.json")).write_bytes(
            info.model_dump_json(indent=2).encode("utf-8")
        )
        retrievers[mode] = MultiVectorRetriever(
            collection_name=name,
            qdrant_client=traced,
            embedder=NoEncoding(),
            experimental_vector_name="experimental_pooling_3",
            max_retries=1,
        )
        print(mode, "stored fields:", list(FIELD_SETS[mode]), "PASS", flush=True)

    expected_ids = {p["id"] for p in points}

    for query_number, q in enumerate(queries):
        results = {
            "A": retrievers["A"].search_embedded(
                query_embedding=q, top_k=3, mode="single_full"
            ),
            "B": retrievers["B"].search_embedded(
                query_embedding=q, top_k=3, mode="two_stage",
                prefetch_k=3,
                stage1_mode="tokens_vs_experimental_pooling",
            ),
            "C": retrievers["C"].search_embedded(
                query_embedding=q, top_k=3,
                mode="single_experimental_tokens",
            ),
        }
        item = {
            "query_id": metadata["queries"][query_number]["query_id"],
            "results": results,
            "max_error_per_token": {},
            "near_ties": {},
        }
        report["queries"].append(item)
        bound = report["tolerance_per_query_token"] * len(q)

        for mode, hits in results.items():
            assert len(hits) == 3
            assert {str(h["id"]) for h in hits} == expected_ids
            field = "experimental_pooling_3" if mode == "C" else "initial"
            refs = {
                p["id"]: reference(q, p["vectors"][field])
                for p in points
            }
            maximum = 0.0
            for hit in hits:
                error = abs(hit["score_final"] - refs[str(hit["id"])]) / len(q)
                maximum = max(maximum, error)
                assert error <= report["tolerance_per_query_token"], (
                    mode, hit["id"], error
                )

            # Allow near ties, but reject a reversal outside the error bound.
            near_ties = []
            for i, left in enumerate(hits):
                for right in hits[i + 1:]:
                    a, b = str(left["id"]), str(right["id"])
                    assert refs[a] + 2 * bound >= refs[b], "Ranking inversion"
                    if abs(refs[a] - refs[b]) <= 2 * bound:
                        near_ties.append([a, b])
            item["near_ties"][mode] = near_ties
            item["max_error_per_token"][mode] = maximum

        a_scores = {str(h["id"]): h["score_final"] for h in results["A"]}
        b_scores = {str(h["id"]): h["score_final"] for h in results["B"]}
        for key in expected_ids:
            assert abs(a_scores[key] - b_scores[key]) / len(q) <= 1e-3
        print("Query", item["query_id"], "A/B/C score and rank checks: PASS")
        save()

    assert len(traced.calls) == 6
    assert sum(c["has_prefetch"] for c in traced.calls) == 2
    report["actual_query_calls"] = traced.calls
    report["module_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), root / "scripts/prb_index.py"]
    }
    report["status"] = "PASS"
    save()
    print("Author retrieval modes and exact parameters: PASS")
    print("A/B equality with all candidates prefetched: PASS")
    print("C pooled-only stored-field verification: PASS")
    print("P1 A/B/C field and retrieval acceptance: PASS")

except Exception as exc:
    report["status"] = "FAILED"
    report["error"] = repr(exc)
    save()
    raise

finally:
    if client is not None:
        client.close()