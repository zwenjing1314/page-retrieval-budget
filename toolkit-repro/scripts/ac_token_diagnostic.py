"""Cache-only token contribution diagnostic for the saved query-169 example.

This does not load a model, encode anything, call Qdrant, or change old records.
It compares full cached page vectors (A representation) with pooled cached page
vectors (C representation) for pages 60 and 304.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
import uuid
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/encoded-economics-v1"
PILOT = ROOT / "records/ac-semantic-pilot-20261108-160234-1a6cd4"
CASE = ROOT / "records/ac-case-view-20261109-115612-0de5b2"
QUERY_ID = "169"
PAGE_IDS = ("60", "304")


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


def normalize(value):
    import numpy as np
    array = np.asarray(value, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    require(np.all(norms > 0) and np.isfinite(array).all(), "Invalid vector norm")
    return array / norms


def score_breakdown(query, page):
    import numpy as np
    q = normalize(query)
    p = normalize(page)
    matrix = q @ p.T
    maxima = matrix.max(axis=1)
    argmax = matrix.argmax(axis=1)
    return {
        "score": float(maxima.sum(dtype=np.float32)),
        "query_token_maxima": [float(x) for x in maxima],
        "query_token_argmax_page_token": [int(x) for x in argmax],
        "matrix_shape": list(matrix.shape),
        "top_query_contributions": [
            {"query_token": int(i), "max_cosine": float(maxima[i]),
             "page_token": int(argmax[i])}
            for i in np.argsort(-maxima, kind="stable")[:min(8, len(maxima))]
        ],
    }


def main():
    sys.dont_write_bytecode = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    record = ROOT / "records" / ("ac-token-diagnostic-" + datetime.now().strftime("%Y%m%d-%H%M%S")
                                 + "-" + uuid.uuid4().hex[:6])
    record.mkdir(parents=True, exist_ok=False)
    print("Record:", record, flush=True)
    report = {
        "status": "STARTED", "query_id": QUERY_ID, "pages": list(PAGE_IDS),
        "model_loads": 0, "encoding_calls": 0, "pooling_calls": 0, "qdrant_calls": 0,
        "scope": "cached query/page MaxSim contribution comparison",
        "interpretation_limit": "diagnostic association; no causal claim about pooling",
    }
    try:
        import numpy as np
        from prb_cache import CacheStore, digest as cache_digest

        profile = load(CACHE / "profile.json")
        store = CacheStore(CACHE, profile, create=False)
        require(store.manifest()["complete"], "Cache is incomplete")
        require(load(PILOT / "report.json")["status"] == "PASS_DIAGNOSTIC", "Pilot not accepted")
        require(load(CASE / "report.json")["status"] == "PASS_EXPORT", "Case export not accepted")
        for relative, expected in profile["encoding_code_sha256"].items():
            require(cache_digest(ROOT / relative.replace("\\", "/")) == expected,
                    "Frozen encoding source changed")
        query_arrays, query_meta = store.read("queries", QUERY_ID)
        query = query_arrays["vectors"]
        require(query.shape[1] == 128 and len(query) > 0, "Invalid query cache")
        page_mapping = {row["corpus_id"]: row for row in load(
            ROOT / "records/p2-formal-a-20261107-012250-612a76/page-id-mapping.json")}
        rankings = {}
        for line in (ROOT / "records/p2-formal-a-20261107-012250-612a76/rankings.jsonl").read_text(
                encoding="utf-8").splitlines():
            row = json.loads(line)
            if str(row["query_id"]) == QUERY_ID:
                rankings["A"] = row["hits"]
        for line in (ROOT / "records/p2-formal-c-20261107-014210-68a2ce/rankings.jsonl").read_text(
                encoding="utf-8").splitlines():
            row = json.loads(line)
            if str(row["query_id"]) == QUERY_ID:
                rankings["C"] = row["hits"]
        require(set(rankings) == {"A", "C"}, "Saved rankings missing")
        results = []
        for pid in PAGE_IDS:
            arrays, meta = store.read("pages", pid)
            full, pooled = arrays["vectors"], arrays["pooled"]
            require(full.shape == (1024, 128) and pooled.shape == (34, 128),
                    "Frozen page representation shape changed")
            a = score_breakdown(query, full)
            c = score_breakdown(query, pooled)
            point_id = page_mapping[pid]["point_id"]
            ranks = {group: next(hit["rank"] for hit in rankings[group]
                                 if str(hit["point_id"]) == point_id)
                     for group in ("A", "C")}
            deltas = [{"query_token": i, "A_max_cosine": a["query_token_maxima"][i],
                       "C_max_cosine": c["query_token_maxima"][i],
                       "C_minus_A": c["query_token_maxima"][i] - a["query_token_maxima"][i],
                       "A_page_token": a["query_token_argmax_page_token"][i],
                       "C_page_token": c["query_token_argmax_page_token"][i]}
                      for i in range(len(query))]
            results.append({
                "corpus_id": pid, "doc_id": meta["identity"]["doc_id"], "ranks": ranks,
                "full_shape": list(full.shape), "pooled_shape": list(pooled.shape),
                "A_full_score_recomputed": a["score"], "C_pooled_score_recomputed": c["score"],
                "A_breakdown": a, "C_breakdown": c, "token_deltas": deltas,
                "A_ranked_score": next(float(hit["score"]) for hit in rankings["A"]
                                        if str(hit["point_id"]) == point_id),
                "C_ranked_score": next(float(hit["score"]) for hit in rankings["C"]
                                        if str(hit["point_id"]) == point_id),
                "source_sha256": {"vectors.npz": digest(CACHE / "pages" / pid / "vectors.npz"),
                                  "metadata.json": digest(CACHE / "pages" / pid / "metadata.json")},
            })
        report.update({"status": "PASS_DIAGNOSTIC", "cache_profile_sha256": store.profile_hash,
                       "query_cache_metadata": query_meta, "query_shape": list(query.shape),
                       "results": results, "script_sha256": digest(Path(__file__)),
                       "summary": "A and C scores are recomputed from normalized cached vectors; "
                                  "ranking scores are retained for audit. C-minus-A per-token values are descriptive."})
        save(record / "token-diagnostic.json", results)
        print("Query:", QUERY_ID, "shape:", query.shape, flush=True)
        for item in results:
            print("Page", item["corpus_id"], "ranks", item["ranks"],
                  "recomputed A/C", round(item["A_full_score_recomputed"], 6),
                  round(item["C_pooled_score_recomputed"], 6), flush=True)
            print("  top A query contributions:", item["A_breakdown"]["top_query_contributions"], flush=True)
            print("  top C query contributions:", item["C_breakdown"]["top_query_contributions"], flush=True)
        print("Model loads: 0; encoding calls: 0; pooling calls: 0; Qdrant calls: 0", flush=True)
        print("Token contribution diagnostic: PASS; no new retrieval", flush=True)
    except Exception as error:
        report.update({"status": "FAIL", "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        save(record / "report.json", report)


if __name__ == "__main__":
    main()
