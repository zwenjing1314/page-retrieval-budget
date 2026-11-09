"""Build and read back the full independent H collection.

Storage-only stage: no model loading, page/query encoding, retrieval metrics,
parameter search, or changes to the existing A/B/C/smoke collections.
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

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/encoded-economics-v1"
H_RECORD = ROOT / "records/ac-semantic-full-cache-20261109-124408-721631"
A_RECORD = ROOT / "records/p2-formal-a-20261107-012250-612a76"
FIELD = "semantic_hierarchical_30"


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
    record = ROOT / "records" / ("ac-semantic-build-" + datetime.now().strftime("%Y%m%d-%H%M%S")
                                 + "-" + uuid.uuid4().hex[:6])
    record.mkdir(parents=True, exist_ok=False)
    collection = "prb_ac_h_full_" + uuid.uuid4().hex[:12]
    print("Record:", record, flush=True)
    print("Collection:", collection, flush=True)
    report = {"status": "STARTED", "collection": collection, "field": FIELD,
              "expected_points": 452, "model_loads": 0, "encoding_calls": 0,
              "query_encoding_calls": 0, "pooling_calls": 0, "search_calls": 0,
              "qdrant_calls": 0, "scope": "full H storage and readback only",
              "retained_collection": True}
    client = None
    uploaded = 0
    try:
        import numpy as np
        from qdrant_client import QdrantClient, models
        from prb_cache import CacheStore

        h_report = load(H_RECORD / "report.json")
        require(h_report["status"] == "PASS_FULL_CACHE" and h_report["completed_pages"] == 452,
                "Full H cache is not accepted")
        profile = load(CACHE / "profile.json")
        store = CacheStore(CACHE, profile, create=False)
        require(store.manifest()["complete"], "Frozen source cache is incomplete")
        mapping = load(A_RECORD / "page-id-mapping.json")
        require(len(mapping) == 452 and len({x["corpus_id"] for x in mapping}) == 452,
                "Invalid page mapping")
        by_page = {str(x["corpus_id"]): x for x in mapping}
        client = QdrantClient(url="http://127.0.0.1:16333", trust_env=False, timeout=120)
        require(collection not in {item.name for item in client.get_collections().collections},
                "Collection name collision")
        started = time.perf_counter()
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
        h_manifest = load(H_RECORD / "manifest.json")
        require(len(h_manifest) == 452, "H manifest is incomplete")
        for index, item in enumerate(sorted(h_manifest, key=lambda x: int(x["id"])), 1):
            pid = str(item["id"])
            arrays, meta = store.read("pages", pid)
            with np.load(H_RECORD / "pages" / pid / "vectors.npz", allow_pickle=False) as saved:
                pooled = saved["pooled"].copy()
            require(pooled.shape == (34, 128) and pooled.dtype == np.float16,
                    f"Invalid H representation for page {pid}")
            require(item["pooled_sha256"] == digest(H_RECORD / "pages" / pid / "vectors.npz"),
                    f"H artifact hash changed for page {pid}")
            point_id = by_page[pid]["point_id"]
            client.upsert(
                collection_name=collection,
                wait=True,
                points=[models.PointStruct(
                    id=point_id,
                    vector={FIELD: pooled.tolist()},
                    payload={"corpus_id": pid, "doc_id": meta["identity"]["doc_id"],
                             "representation": "hierarchical_pool_factor_30", "diagnostic": False},
                )],
            )
            uploaded = index
            if index == 1 or index % 50 == 0 or index == 452:
                print(f"Uploaded: {index}/452", flush=True)
        upload_seconds = time.perf_counter() - started
        info = client.get_collection(collection)
        require(info.status.value == "green", f"Collection status: {info.status}")
        require(client.count(collection, exact=True).count == 452, "Point count mismatch")
        require(set(info.config.params.vectors) == {FIELD}, "Stored field mismatch")
        vector = info.config.params.vectors[FIELD]
        require(vector.size == 128 and vector.distance == models.Distance.COSINE and
                vector.datatype == models.Datatype.FLOAT16 and vector.on_disk is False and
                vector.multivector_config.comparator == models.MultiVectorComparator.MAX_SIM,
                "Vector configuration mismatch")
        require(info.config.hnsw_config.m == 0 and info.config.hnsw_config.full_scan_threshold == 10 and
                info.config.hnsw_config.on_disk is False, "HNSW configuration mismatch")
        require(info.config.optimizer_config.indexing_threshold == 0 and
                info.config.params.on_disk_payload is False, "Storage configuration mismatch")

        readback_started = time.perf_counter()
        seen = set()
        offset = None
        batch_count = 0
        while True:
            batch, offset = client.scroll(collection_name=collection, offset=offset, limit=32,
                                          with_vectors=True, with_payload=True)
            batch_count += 1
            for point in batch:
                point_id = str(point.id)
                require(point_id not in seen, "Duplicate point in readback")
                seen.add(point_id)
                require(set(point.vector) == {FIELD}, "Readback field mismatch")
                rows = np.asarray(point.vector[FIELD])
                require(rows.shape == (34, 128) and np.isfinite(rows).all(),
                        "Readback vector shape/value mismatch")
                payload = point.payload
                require(str(payload["corpus_id"]) in by_page and
                        payload["doc_id"] == by_page[str(payload["corpus_id"])] ["doc_id"],
                        "Readback payload mismatch")
            if offset is None:
                break
        readback_seconds = time.perf_counter() - readback_started
        require(len(seen) == 452, f"Readback IDs: {len(seen)}/452")
        report.update({"status": "PASS_FULL_BUILD", "uploaded_points": uploaded,
                       "readback_points": len(seen), "readback_batches": batch_count,
                       "upload_wall_seconds": upload_seconds, "readback_wall_seconds": readback_seconds,
                       "collection_info": info.model_dump(mode="json"),
                       "source_h_report_sha256": digest(H_RECORD / "report.json"),
                       "source_h_manifest_sha256": digest(H_RECORD / "manifest.json"),
                       "cache_profile_sha256": store.profile_hash, "script_sha256": digest(Path(__file__)),
                       "query_evaluation": "not run"})
        save(record / "collection.json", info.model_dump(mode="json"))
        save(record / "report.json", report)
        print("Server collection: green; points: 452; stored field:", FIELD, flush=True)
        print("Full H vector readback: 452/452, each 34x128 FP16 PASS", flush=True)
        print("Upload seconds:", round(upload_seconds, 3), "Readback seconds:", round(readback_seconds, 3), flush=True)
        print("Model loads: 0; encoding calls: 0; pooling calls: 0; search calls: 0", flush=True)
        print("Full H Qdrant build: PASS; collection retained; evaluation not run", flush=True)
    except Exception as error:
        report.update({"status": "FAIL", "uploaded_points": uploaded,
                       "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        if client is not None:
            client.close()
        if not (record / "report.json").exists():
            save(record / "report.json", report)


if __name__ == "__main__":
    main()
