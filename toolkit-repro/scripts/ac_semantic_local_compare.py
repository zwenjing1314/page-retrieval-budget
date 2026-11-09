"""Three-page local A/C/H comparison for query 169.

H is the existing colpali-engine HierarchicalTokenPooler with the fixed
pool_factor=30. Diagnostic only: no model, encoding, Qdrant, parameter search,
or full-corpus run.
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
LABELS = ROOT / "records/ac-token-labels-20261109-122808-eb4cf9"
PAGE_IDS = ("60", "304", "63")
QUERY_ID = "169"
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


def normalized(value):
    import numpy as np
    array = np.asarray(value, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    require(np.all(norms > 0) and np.isfinite(array).all(), "Invalid vector matrix")
    return array / norms


def maxsim(query, page):
    import numpy as np
    matrix = normalized(query) @ normalized(page).T
    maxima = matrix.max(axis=1)
    return float(maxima.sum(dtype=np.float32)), [float(x) for x in maxima]


def main():
    sys.dont_write_bytecode = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    record = ROOT / "records" / ("ac-semantic-local-" + datetime.now().strftime("%Y%m%d-%H%M%S")
                                 + "-" + uuid.uuid4().hex[:6])
    record.mkdir(parents=True, exist_ok=False)
    print("Record:", record, flush=True)
    report = {"status": "STARTED", "query_id": QUERY_ID, "pages": list(PAGE_IDS),
              "pool_factor": POOL_FACTOR, "model_loads": 0, "encoding_calls": 0,
              "qdrant_calls": 0, "scope": "three-page local diagnostic",
              "interpretation_limit": "not a corpus retrieval result or quality estimate"}
    try:
        import numpy as np
        import torch
        from colpali_engine.compression import HierarchicalTokenPooler
        from prb_cache import CacheStore

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(version("colpali-engine") == "0.3.10", "Expected colpali-engine 0.3.10")
        require(load(PILOT / "report.json")["status"] == "PASS_DIAGNOSTIC", "Pilot not accepted")
        require(load(LABELS / "report.json")["status"] == "PASS_DIAGNOSTIC", "Token labels not accepted")
        profile = load(CACHE / "profile.json")
        store = CacheStore(CACHE, profile, create=False)
        query_arrays, query_meta = store.read("queries", QUERY_ID)
        query = query_arrays["vectors"]
        require(query.shape == (18, 128), "Unexpected query shape")
        pooler = HierarchicalTokenPooler()
        rows = []
        for pid in PAGE_IDS:
            arrays, meta = store.read("pages", pid)
            full, pooled = arrays["vectors"], arrays["pooled"]
            require(full.shape == (1024, 128) and pooled.shape == (34, 128),
                    "Frozen cache representation changed")
            started = time.perf_counter()
            output = pooler.pool_embeddings([torch.from_numpy(full.copy())],
                                            pool_factor=POOL_FACTOR, num_workers=1,
                                            return_dict=True)
            h = output.pooled_embeddings[0].detach().cpu().numpy().copy()
            elapsed = time.perf_counter() - started
            require(h.shape == (34, 128) and h.dtype == np.float16, "H output rule failed")
            a_score, a_tokens = maxsim(query, full)
            c_score, c_tokens = maxsim(query, pooled)
            h_score, h_tokens = maxsim(query, h)
            rows.append({"corpus_id": pid, "doc_id": meta["identity"]["doc_id"],
                         "A_score": a_score, "C_score": c_score, "H_score": h_score,
                         "A_token_maxima": a_tokens, "C_token_maxima": c_tokens,
                         "H_token_maxima": h_tokens, "H_shape": list(h.shape),
                         "H_pooling_seconds": elapsed,
                         "H_vector_bytes": int(h.nbytes), "C_vector_bytes": int(pooled.nbytes)})
        for key in ("A_score", "C_score", "H_score"):
            ranked = sorted(rows, key=lambda row: (-row[key], int(row["corpus_id"])))
            for rank, row in enumerate(ranked, 1):
                row[key.replace("_score", "_rank_three_pages")] = rank
        report.update({"status": "PASS_DIAGNOSTIC", "cache_profile_sha256": store.profile_hash,
                       "query_identity": query_meta["identity"], "rows": rows,
                       "script_sha256": digest(Path(__file__)),
                       "warning": "A/C/H absolute scores are not comparable across representations; "
                                  "only within-representation page ordering is descriptive."})
        save(record / "local-compare.json", rows)
        for row in rows:
            print("Page", row["corpus_id"], "A/C/H:", round(row["A_score"], 6),
                  round(row["C_score"], 6), round(row["H_score"], 6),
                  "three-page ranks A/C/H:", row["A_rank_three_pages"],
                  row["C_rank_three_pages"], row["H_rank_three_pages"],
                  "H build seconds:", round(row["H_pooling_seconds"], 4), flush=True)
        print("Model loads: 0; encoding calls: 0; Qdrant calls: 0", flush=True)
        print("A/C/H three-page local comparison: PASS; no corpus retrieval", flush=True)
    except Exception as error:
        report.update({"status": "FAIL", "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        save(record / "report.json", report)


if __name__ == "__main__":
    main()
