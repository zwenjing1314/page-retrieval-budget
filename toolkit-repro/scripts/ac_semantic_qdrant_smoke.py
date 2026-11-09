"""Three-page Qdrant smoke test for the full H cache.

Creates one new isolated collection and retains it for evidence. No model,
encoding, pooling, parameter search, or changes to A/B/C collections.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
from datetime import datetime
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/encoded-economics-v1"
H_RECORD = ROOT / "records/ac-semantic-full-cache-20261109-124408-721631"
A_RECORD = ROOT / "records/p2-formal-a-20261107-012250-612a76"
FIELD = "semantic_hierarchical_30"
PAGE_IDS = ("60", "304", "63")
QUERY_ID = "169"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    sys.dont_write_bytecode = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    record = ROOT / "records" / ("ac-semantic-qdrant-smoke-" + datetime.now().strftime("%Y%m%d-%H%M%S")
                                 + "-" + uuid.uuid4().hex[:6])
    record.mkdir(parents=True, exist_ok=False)
    collection = "prb_ac_h_smoke_" + uuid.uuid4().hex[:12]
    print("Record:", record, flush=True)
    print("Collection:", collection, flush=True)
    report = {"status": "STARTED", "collection": collection, "field": FIELD,
              "pages": list(PAGE_IDS), "query_id": QUERY_ID, "model_loads": 0,
              "encoding_calls": 0, "pooling_calls": 0, "qdrant_calls": 0,
              "scope": "three-page H storage and MaxSim smoke test"}
    client = None
    try:
        import numpy as np
        from qdrant_client import QdrantClient, models
        from prb_cache import CacheStore

        profile = load(CACHE / "profile.json")
        store = CacheStore(CACHE, profile, create=False)
        h_report = load(H_RECORD / "report.json")
        require(h_report["status"] == "PASS_FULL_CACHE" and h_report["completed_pages"] == 452,
                "Full H cache not accepted")
        require(load(A_RECORD / "report.json")["status"] == "PASS", "A record not accepted")
        mapping = {row["corpus_id"]: row for row in load(A_RECORD / "page-id-mapping.json")}
        query_arrays, query_meta = store.read("queries", QUERY_ID)
        query = query_arrays["vectors"]
        require(query.shape == (18, 128), "Unexpected query cache shape")
        points = []
        expected_scores = {}
        for pid in PAGE_IDS:
            arrays, meta = store.read("pages", pid)
            with np.load(H_RECORD / "pages" / pid / "vectors.npz", allow_pickle=False) as saved:
                h = saved["pooled"].copy()
            require(h.shape == (34, 128) and h.dtype == np.float16, f"Invalid H page {pid}")
            q = query.astype(np.float32)
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            p = h.astype(np.float32)
            p /= np.linalg.norm(p, axis=1, keepdims=True)
            expected_scores[pid] = float((q @ p.T).max(axis=1).sum(dtype=np.float32))
            points.append(models.PointStruct(
                id=mapping[pid]["point_id"],
                vector={FIELD: h.tolist()},
                payload={"corpus_id": pid, "doc_id": meta["identity"]["doc_id"],
                         "representation": "hierarchical_pool_factor_30", "diagnostic": True},
            ))
        client = QdrantClient(url="http://127.0.0.1:16333", trust_env=False, timeout=60)
        before = client.get_collections()
        require(collection not in {item.name for item in before.collections}, "Collection collision")
        client.create_collection(
            collection_name=collection,
            vectors_config={FIELD: models.VectorParams(
                size=128, distance=models.Distance.COSINE, datatype=models.Datatype.FLOAT16,
                on_disk=False,
                multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
            )},
            hnsw_config=models.HnswConfigDiff(m=0, full_scan_threshold=10, on_disk=False),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
            on_disk_payload=False,
        )
        for point in points:
            client.upsert(collection_name=collection, wait=True, points=[point])
        info = client.get_collection(collection)
        require(info.status.value == "green", f"Collection status: {info.status}")
        require(client.count(collection, exact=True).count == 3, "Point count mismatch")
        require(set(info.config.params.vectors) == {FIELD}, "Stored field mismatch")
        vector = info.config.params.vectors[FIELD]
        require(vector.size == 128 and vector.datatype == models.Datatype.FLOAT16 and
                vector.multivector_config.comparator == models.MultiVectorComparator.MAX_SIM,
                "Vector config mismatch")
        require(info.config.hnsw_config.m == 0 and info.config.hnsw_config.full_scan_threshold == 10,
                "HNSW config mismatch")
        require(info.config.optimizer_config.indexing_threshold == 0 and
                info.config.params.on_disk_payload is False, "Storage config mismatch")
        stored, _ = client.scroll(collection_name=collection, limit=3, with_vectors=True, with_payload=True)
        require(len(stored) == 3, "Readback count mismatch")
        for point in stored:
            rows = np.asarray(point.vector[FIELD])
            require(rows.shape == (34, 128) and np.isfinite(rows).all(), "Readback vector mismatch")
        response = client.query_points(collection_name=collection, query=query.tolist(), using=FIELD,
                                       limit=3, search_params=models.SearchParams(exact=True),
                                       with_payload=True, with_vectors=False)
        hits = response.points
        require(len(hits) == 3, "Search count mismatch")
        actual_order = [str(hit.payload["corpus_id"]) for hit in hits]
        expected_order = [pid for pid, _ in sorted(expected_scores.items(), key=lambda item: (-item[1], int(item[0])))]
        require(actual_order == expected_order, f"Qdrant order differs: {actual_order} vs {expected_order}")
        report.update({"status": "PASS_SMOKE", "collection_info": info.model_dump(mode="json"),
                       "query_metadata": query_meta, "expected_local_scores": expected_scores,
                       "expected_order": expected_order, "qdrant_order": actual_order,
                       "source_h_report_sha256": digest(H_RECORD / "report.json"),
                       "cache_profile_sha256": store.profile_hash, "script_sha256": digest(Path(__file__)),
                       "retained_collection": True})
        save(record / "collection.json", info.model_dump(mode="json"))
        save(record / "report.json.tmp", report)
        (record / "report.json.tmp").replace(record / "report.json")
        print("Server collection: green; points: 3; stored field:", FIELD, flush=True)
        print("H vector readback: 34x128 FP16 PASS", flush=True)
        print("Expected order:", expected_order, flush=True)
        print("Qdrant order:", actual_order, flush=True)
        print("Model loads: 0; encoding calls: 0; pooling calls: 0", flush=True)
        print("H Qdrant smoke: PASS; collection retained for review", flush=True)
    except Exception as error:
        report.update({"status": "FAIL", "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        if client is not None:
            client.close()
        if not (record / "report.json").exists():
            save(record / "report.json", report)


if __name__ == "__main__":
    main()
