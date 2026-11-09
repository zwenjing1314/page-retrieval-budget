"""Generate a separate full H cache from the frozen page cache.

H uses colpali-engine 0.3.10 HierarchicalTokenPooler(pool_factor=30).
No model loading, page re-encoding, query work, Qdrant access or parameter search.
Each output page is written independently under a new records directory.
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
PILOT = ROOT / "records/ac-semantic-pilot-20261108-160234-1a6cd4"
LOCAL = ROOT / "records/ac-semantic-local-20261109-123825-d92e7e"
POOL_FACTOR = 30


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
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    record = ROOT / "records" / ("ac-semantic-full-cache-" + datetime.now().strftime("%Y%m%d-%H%M%S")
                                 + "-" + uuid.uuid4().hex[:6])
    (record / "pages").mkdir(parents=True, exist_ok=False)
    print("Record:", record, flush=True)
    report = {"status": "STARTED", "pool_factor": POOL_FACTOR,
              "model_loads": 0, "encoding_calls": 0, "query_encoding_calls": 0,
              "qdrant_calls": 0, "source_cache": str(CACHE),
              "output_scope": "full 452-page H representation only",
              "fairness": "same frozen 1024x128 effective page vectors as A/C; FP16; no query changes"}
    outputs = []
    try:
        import numpy as np
        import torch
        from colpali_engine.compression import HierarchicalTokenPooler
        from prb_cache import CacheStore

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(version("colpali-engine") == "0.3.10", "Expected colpali-engine 0.3.10")
        require(load(PILOT / "report.json")["status"] == "PASS_DIAGNOSTIC", "Pilot not accepted")
        require(load(LOCAL / "report.json")["status"] == "PASS_DIAGNOSTIC", "Local comparison not accepted")
        profile = load(CACHE / "profile.json")
        store = CacheStore(CACHE, profile, create=False)
        manifest = store.manifest()
        require(manifest["complete"] and len(manifest["completed"]["pages"]) == 452,
                "Frozen page cache is incomplete")
        source = Path(__import__("inspect").getfile(HierarchicalTokenPooler))
        report.update({"cache_profile_sha256": store.profile_hash,
                       "pooler_source": str(source), "pooler_source_sha256": digest(source),
                       "versions": {name: version(name) for name in
                                    ("torch", "colpali-engine", "numpy", "scipy")},
                       "script_sha256": digest(Path(__file__)), "expected_pages": 452})
        pooler = HierarchicalTokenPooler()
        started_all = time.perf_counter()
        for index, item in enumerate(sorted(manifest["completed"]["pages"], key=lambda x: int(x["id"])), 1):
            pid = str(item["id"])
            arrays, meta = store.read("pages", pid)
            full = arrays["vectors"]
            require(full.shape == (1024, 128) and full.dtype == np.float16,
                    f"Unexpected source shape for page {pid}")
            source_hash = digest(CACHE / "pages" / pid / "vectors.npz")
            started = time.perf_counter()
            result = pooler.pool_embeddings([torch.from_numpy(full.copy())],
                                             pool_factor=POOL_FACTOR, num_workers=1,
                                             return_dict=True)
            elapsed = time.perf_counter() - started
            pooled = result.pooled_embeddings[0].detach().cpu().numpy().copy()
            require(pooled.shape == (34, 128) and pooled.dtype == np.float16,
                    f"H output shape/dtype failed for page {pid}")
            norms = np.linalg.norm(pooled.astype(np.float32), axis=1)
            require(np.isfinite(pooled).all() and np.all(norms > 0), f"H invalid values for page {pid}")
            require(float(np.max(np.abs(norms - 1))) <= 0.005, f"H norm failed for page {pid}")
            membership = result.cluster_id_to_indices[0]
            labels = np.full(1024, -1, dtype=np.int32)
            visits = np.zeros(1024, dtype=np.int32)
            sizes = []
            for _, indices_tuple in sorted(membership.items()):
                indices = indices_tuple[0].cpu().numpy().astype(np.int64)
                if not len(indices):
                    continue
                require(np.all((indices >= 0) & (indices < 1024)), f"Invalid cluster for page {pid}")
                visits[indices] += 1
                labels[indices] = len(sizes)
                sizes.append(int(len(indices)))
            require(np.all(visits == 1) and np.all(labels >= 0) and len(sizes) == 34,
                    f"Partition failed for page {pid}")
            destination = record / "pages" / pid
            destination.mkdir()
            with (destination / "vectors.npz").open("xb") as output:
                np.savez(output, pooled=pooled, token_cluster=labels,
                         cluster_sizes=np.asarray(sizes, dtype=np.int32))
            output_meta = {"id": pid, "corpus_id": pid, "doc_id": meta["identity"]["doc_id"],
                           "source_vectors_sha256": source_hash, "source_shape": list(full.shape),
                           "pooled_shape": list(pooled.shape), "pooled_dtype": str(pooled.dtype),
                           "cluster_sizes": sizes, "token_coverage_exact_once": True,
                           "max_unit_norm_error": float(np.max(np.abs(norms - 1))),
                           "pool_factor": POOL_FACTOR, "pooling_seconds": elapsed,
                           "pooled_sha256": digest(destination / "vectors.npz")}
            save(destination / "metadata.json", output_meta)
            outputs.append(output_meta)
            if index == 1 or index % 25 == 0 or index == 452:
                print(f"Pages: {index}/452; latest={pid}; seconds={elapsed:.4f}; "
                      f"norm_error={output_meta['max_unit_norm_error']:.6g}", flush=True)
        elapsed_all = time.perf_counter() - started_all
        report.update({"status": "PASS_FULL_CACHE", "completed_pages": len(outputs),
                       "total_pooling_seconds": elapsed_all,
                       "sum_per_page_pooling_seconds": sum(x["pooling_seconds"] for x in outputs),
                       "mean_pooling_seconds": sum(x["pooling_seconds"] for x in outputs) / len(outputs),
                       "min_pooling_seconds": min(x["pooling_seconds"] for x in outputs),
                       "max_pooling_seconds": max(x["pooling_seconds"] for x in outputs),
                       "all_shapes_34x128": all(x["pooled_shape"] == [34, 128] for x in outputs),
                       "all_exact_budget_bytes": all(x["pooled_shape"] == [34, 128] for x in outputs),
                       "all_coverage_exact_once": all(x["token_coverage_exact_once"] for x in outputs),
                       "full_cache_directory": str(record / "pages")})
        save(record / "manifest.json", outputs)
        print("Completed pages: 452/452", flush=True)
        print("All H shapes 34x128:", report["all_shapes_34x128"], flush=True)
        print("Full H cache: PASS; no model/encoding/Qdrant", flush=True)
        print("STOP: do not build Qdrant or evaluate until this record is reviewed", flush=True)
    except Exception as error:
        report.update({"status": "FAIL", "completed_pages": len(outputs),
                       "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        save(record / "report.json", report)


if __name__ == "__main__":
    main()
