"""User-run, cache-only A/C analysis and a three-page official pooler pilot.

No model loading, encoding, network access, Qdrant access or parameter search.
Only creates a new records/ac-semantic-pilot-* directory. Existing inputs are read-only.
"""

import csv
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
import warnings
from datetime import datetime
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/encoded-economics-v1"
A_RECORD = ROOT / "records/p2-formal-a-20261107-012250-612a76"
C_RECORD = ROOT / "records/p2-formal-c-20261107-014210-68a2ce"
PAGE_IDS = ("0", "1", "63")
POOL_FACTOR = 30
TARGET = 34
METRICS = ("NDCG@5", "NDCG@10", "Recall@5", "Recall@10", "Recall@100")
EPS = 1e-12


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)


def paired_inputs(digest, profile_hash):
    accepted = {}
    names = ("per-query.tsv", "rankings.jsonl", "qrels.json", "page-id-mapping.json")
    for group, directory in (("A", A_RECORD), ("C", C_RECORD)):
        report = load_json(directory / "report.json")
        require(report["status"] == "PASS", f"Unaccepted {group} record")
        require(report["group"] == group, "Record group mismatch")
        require(report["counts"] == {"pages": 452, "queries": 232, "qrels": 3628},
                "Unexpected benchmark counts")
        require(report["completed_queries"] == 232, "Incomplete evaluation")
        require(report["cache_profile_sha256"] == profile_hash, "Cache identity mismatch")
        for name in names:
            require(digest(directory / name) == report["artifact_sha256"][name],
                    f"Historical artifact changed: {directory / name}")
        with (directory / "per-query.tsv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file, delimiter="\t"))
        ranking_rows = [json.loads(line) for line in
                        (directory / "rankings.jsonl").read_text(encoding="utf-8").splitlines()
                        if line.strip()]
        per_query = {str(row["query_id"]): row for row in rows}
        rankings = {str(row["query_id"]): row["hits"] for row in ranking_rows}
        require(len(rows) == len(per_query) == len(ranking_rows) == len(rankings) == 232,
                "Duplicate or missing query IDs")
        accepted[group] = {"report": report, "per_query": per_query, "rankings": rankings}
    require(load_json(A_RECORD / "qrels.json") == load_json(C_RECORD / "qrels.json"),
            "A/C qrels differ")
    require(load_json(A_RECORD / "page-id-mapping.json") ==
            load_json(C_RECORD / "page-id-mapping.json"), "A/C page mapping differs")
    return accepted


def analyze_ac(record, accepted, profile, pq):
    qrels = load_json(A_RECORD / "qrels.json")
    mapping = load_json(A_RECORD / "page-id-mapping.json")
    by_point = {str(row["point_id"]): row for row in mapping}
    require(len(by_point) == 452, "Invalid point mapping")
    data = ROOT / "data" / ("economics_reports_v2-" + profile["dataset_revision"])
    query_file = data / "queries/test-00000-of-00001.parquet"
    import hashlib
    manifest = load_json(data / "download-manifest.json")
    require(hashlib.sha256((data / "download-manifest.json").read_bytes()).hexdigest() ==
            profile["dataset_download_manifest_sha256"], "Dataset manifest changed")
    entry = next(item for item in manifest["files"] if item["path"] ==
                 "queries/test-00000-of-00001.parquet")
    require(hashlib.sha256(query_file.read_bytes()).hexdigest() == entry["sha256"],
            "Official query table changed")
    official = {str(row["query-id"]): row for row in
                pq.read_table(query_file, columns=["query-id", "query", "language"]).to_pylist()}
    expected = set(profile["expected_query_ids"])
    for group in accepted.values():
        require(set(group["per_query"]) == set(group["rankings"]) == expected,
                "A/C query coverage mismatch")
    require(set(official) == set(qrels) == expected, "Official query/label mismatch")
    rows = []
    all_moves = {}
    for qid in sorted(expected, key=int):
        relevant = {pid for pid, grade in qrels[qid].items() if grade > 0}
        require(relevant and relevant <= set(by_point), "Invalid relevant page set")
        row = {"query_id": qid, "language": official[qid]["language"],
               "query": official[qid]["query"], "relevant_pages": len(relevant)}
        ids = {}
        for group in ("A", "C"):
            hits = accepted[group]["rankings"][qid]
            ids[group] = [str(hit["point_id"]) for hit in hits]
            require(len(ids[group]) == len(set(ids[group])) == 100, "Invalid top100")
            require(set(ids[group]) <= set(by_point), "Unknown result page")
            require([hit["rank"] for hit in hits] == list(range(1, 101)), "Rank order mismatch")
            for hit in hits:
                require(str(hit["corpus_id"]) == by_point[str(hit["point_id"])]["corpus_id"],
                        "Ranking corpus ID mismatch")
            for metric in METRICS:
                k = int(metric.split("@")[1])
                binary = [int(pid in relevant) for pid in ids[group][:k]]
                if metric.startswith("Recall"):
                    calculated = sum(binary) / len(relevant)
                else:
                    ideal = math.fsum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
                    calculated = math.fsum(value / math.log2(i + 2)
                                           for i, value in enumerate(binary)) / ideal
                value = float(accepted[group]["per_query"][qid][metric])
                require(abs(value - calculated) <= EPS, f"Historical metric mismatch: {group}/{qid}")
                row[group + "_" + metric] = value
        for metric in METRICS:
            row["C_minus_A_" + metric] = row["C_" + metric] - row["A_" + metric]
        moves = {}
        for k in (10, 100):
            a, c = set(ids["A"][:k]), set(ids["C"][:k])
            row[f"top{k}_overlap_count"] = len(a & c)
            for direction, changed in (("A_only", (a - c) & relevant),
                                       ("C_only", (c - a) & relevant)):
                pages = sorted((by_point[pid]["corpus_id"] for pid in changed), key=int)
                key = f"relevant_{direction}_top{k}"
                row[key] = ",".join(pages)
                moves[key] = pages
        rows.append(row)
        all_moves[qid] = moves
    with (record / "ac-per-query.tsv").open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    summary = {"queries": len(rows), "delta_definition": "C minus A", "metrics": {}}
    for metric in METRICS:
        values = [row["C_minus_A_" + metric] for row in rows]
        item = {"C_lower": sum(x < -EPS for x in values),
                "equal": sum(abs(x) <= EPS for x in values),
                "C_higher": sum(x > EPS for x in values)}
        for group in ("A", "C"):
            item[group + "_mean"] = math.fsum(row[group + "_" + metric] for row in rows) / len(rows)
            require(abs(item[group + "_mean"] - accepted[group]["report"]["aggregate_metrics"][metric])
                    <= EPS, "Aggregate mean mismatch")
        item["mean_C_minus_A"] = math.fsum(values) / len(values)
        summary["metrics"][metric] = item
    save_json(record / "ac-summary.json", summary)
    primary = "C_minus_A_NDCG@10"
    ordered = sorted(rows, key=lambda row: (row[primary], int(row["query_id"])))
    candidates = {
        "largest_NDCG10_drop": next((r for r in ordered if r[primary] < -EPS), None),
        "equal_NDCG10_smallest_id": next((r for r in sorted(rows, key=lambda r: int(r["query_id"]))
                                         if abs(r[primary]) <= EPS), None),
        "largest_NDCG10_gain": next((r for r in sorted(rows, key=lambda r: (-r[primary], int(r["query_id"])))
                                    if r[primary] > EPS), None),
    }
    cases = {}
    for label, row in candidates.items():
        if row is None:
            cases[label] = None
            continue
        qid = row["query_id"]
        page_details = []
        ranks = {g: {str(h["point_id"]): h["rank"] for h in accepted[g]["rankings"][qid]}
                 for g in ("A", "C")}
        for pid, grade in sorted(qrels[qid].items(), key=lambda pair: int(by_point[pair[0]]["corpus_id"])):
            if grade > 0:
                page_details.append({**by_point[pid], "grade": grade,
                                     "A_rank_top100": ranks["A"].get(pid),
                                     "C_rank_top100": ranks["C"].get(pid)})
        cases[label] = {"paired_row": row, "relevant_pages": page_details,
                        "null_rank_means": "not in stored top100; does not mean absent from corpus",
                        "A_top10": accepted["A"]["rankings"][qid][:10],
                        "C_top10": accepted["C"]["rankings"][qid][:10]}
    save_json(record / "ac-cases.json", cases)
    print("A/C paired queries:", len(rows), "PASS", flush=True)
    print("NDCG@10 C-lower/equal/C-higher:", summary["metrics"]["NDCG@10"], flush=True)
    return summary


def main():
    # These settings affect this diagnostic process only, not the saved Conda environment.
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    record = ROOT / "records" / ("ac-semantic-pilot-" + datetime.now().strftime("%Y%m%d-%H%M%S")
                                 + "-" + uuid.uuid4().hex[:6])
    record.mkdir(parents=True, exist_ok=False)
    print("Record:", record, flush=True)
    report = {"status": "STARTED", "scope": "paired historical analysis + 3-page existing-pooler diagnostic",
              "pages": list(PAGE_IDS), "pool_factor": POOL_FACTOR, "target_max_vectors": TARGET,
              "model_loads": 0, "encoding_calls": 0, "qdrant_calls": 0,
              "method": "colpali-engine 0.3.10 HierarchicalTokenPooler; unchanged official implementation",
              "device": "cpu", "num_workers": 1, "numeric_threads": 1,
              "primary_pair_metric": "NDCG@10", "requires_user_review_before_any_next_stage": True,
              "cost_boundary": "one pool_embeddings call per page; excludes imports, cache read, validation and writes",
              "C_construction_seconds": None,
              "cost_limit": "C read time is not C construction time; no speed ratio or full-system resource claim"}
    page_reports = []
    try:
        import numpy as np
        import pyarrow.parquet as pq
        import torch
        from colpali_engine.compression import HierarchicalTokenPooler
        from prb_cache import CacheStore, digest

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(version("colpali-engine") == "0.3.10", "Expected frozen colpali-engine 0.3.10")
        report["versions"] = {name: version(name) for name in
                              ("torch", "colpali-engine", "numpy", "scipy", "pyarrow", "transformers")}
        profile = load_json(CACHE / "profile.json")
        for name, expected in profile["versions"].items():
            require(version(name) == expected, f"Frozen environment changed: {name}")
        for relative, expected in profile["encoding_code_sha256"].items():
            require(digest(ROOT / relative.replace("\\", "/")) == expected,
                    f"Frozen encoding source changed: {relative}")
        store = CacheStore(CACHE, profile, create=False)
        report["cache_profile_sha256"] = store.profile_hash
        report["script_sha256"] = digest(Path(__file__))
        source = Path(inspect.getfile(HierarchicalTokenPooler))
        base_source = Path(inspect.getfile(HierarchicalTokenPooler.__bases__[0]))
        report["pooler_source"] = {"path": str(source), "sha256": digest(source),
                                   "base_path": str(base_source), "base_sha256": digest(base_source)}
        report["implementation_caveat"] = (
            "Official code passes square 1-X@X.T to Ward linkage as a 2D observation matrix. "
            "This pilot preserves that behavior; fcluster maxclust is an upper bound, not exact K."
        )
        accepted = paired_inputs(digest, store.profile_hash)
        report["historical_inputs"] = {g: {"directory": str(d), "report_sha256": digest(d / "report.json")}
                                        for g, d in (("A", A_RECORD), ("C", C_RECORD))}
        report["ac_summary"] = analyze_ac(record, accepted, profile, pq)
        pooler = HierarchicalTokenPooler()
        (record / "pages").mkdir()

        def norm_stats(array):
            require(array.ndim == 2 and array.shape[1] == 128 and np.isfinite(array).all(),
                    "Invalid vector matrix")
            norms = np.linalg.norm(array.astype(np.float32), axis=1)
            require(np.all(norms > 0), "Zero-norm vector")
            return {"min": float(norms.min()), "max": float(norms.max()),
                    "max_unit_error": float(np.max(np.abs(norms - 1)))}

        for pid in PAGE_IDS:
            files = [CACHE / "pages" / pid / name for name in ("vectors.npz", "metadata.json")]
            before = {str(path): digest(path) for path in files}
            started = time.perf_counter()
            arrays, meta = store.read("pages", pid)
            read_seconds = time.perf_counter() - started
            full, c_vectors = arrays["vectors"], arrays["pooled"]
            require(full.shape == (1024, 128) and full.dtype == np.float16, "Full vector rules changed")
            require(c_vectors.shape == (34, 128) and c_vectors.dtype == np.float16, "C vector rules changed")
            require(max(len(full) // POOL_FACTOR, 1) == TARGET, "Budget derivation changed")
            tensor = torch.from_numpy(full.copy())
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                started = time.perf_counter()
                output = pooler.pool_embeddings([tensor], pool_factor=POOL_FACTOR,
                                                num_workers=1, return_dict=True)
                pooling_seconds = time.perf_counter() - started
            h_vectors = output.pooled_embeddings[0].detach().cpu().numpy().copy()
            require(h_vectors.dtype == np.float16 and 1 <= len(h_vectors) <= TARGET,
                    "Unexpected official pooler output")
            require(norm_stats(h_vectors)["max_unit_error"] <= 0.005, "Cluster output norm check failed")
            membership = output.cluster_id_to_indices[0]
            labels = np.full(len(full), -1, dtype=np.int32)
            visits = np.zeros(len(full), dtype=np.int32)
            centroids, sizes = [], []
            for _, indices_tuple in sorted(membership.items()):
                indices = indices_tuple[0].cpu().numpy().astype(np.int64)
                if not len(indices):
                    continue
                require(np.all((indices >= 0) & (indices < len(full))), "Invalid cluster member")
                visits[indices] += 1
                labels[indices] = len(sizes)
                sizes.append(int(len(indices)))
                centroids.append(full[indices].astype(np.float32).mean(axis=0))
            require(np.all(visits == 1) and np.all(labels >= 0), "Tokens not partitioned exactly once")
            require(len(sizes) == len(h_vectors) and sum(sizes) == 1024, "Cluster counts mismatch")
            means = np.stack(centroids)
            mean_norms = np.linalg.norm(means, axis=1, keepdims=True)
            require(np.all(mean_norms > 0), "Zero-norm cluster mean")
            reference = (means / mean_norms).astype(np.float16)
            max_error = float(np.max(np.abs(reference.astype(np.float32) - h_vectors.astype(np.float32))))
            require(max_error <= 1e-3, "Independent mean + normalization check failed")
            # Diagnostic FP32 views implement the same cosine rule for every representation.
            cosine_norms = {}
            for name, vectors in (("full", full), ("C", c_vectors), ("H", h_vectors)):
                view = vectors.astype(np.float32)
                view /= np.linalg.norm(view, axis=1, keepdims=True)
                cosine_norms[name] = norm_stats(view)
                require(cosine_norms[name]["max_unit_error"] <= 1e-5, "Cosine view normalization failed")
            destination = record / "pages" / (pid + ".npz")
            started = time.perf_counter()
            with destination.open("xb") as output_file:
                np.savez(output_file, pooled=h_vectors, token_cluster=labels,
                         cluster_sizes=np.asarray(sizes, dtype=np.int32))
            with np.load(destination, allow_pickle=False) as saved:
                require(np.array_equal(saved["pooled"], h_vectors), "Saved vectors differ")
                require(np.array_equal(saved["token_cluster"], labels), "Saved membership differs")
            write_seconds = time.perf_counter() - started
            require(before == {str(path): digest(path) for path in files}, "Source cache changed")
            item = {"corpus_id": pid, "doc_id": meta["identity"]["doc_id"],
                    "source_sha256": before, "full_shape": list(full.shape),
                    "C_shape": list(c_vectors.shape), "H_shape": list(h_vectors.shape),
                    "C_norms": norm_stats(c_vectors), "H_norms": norm_stats(h_vectors),
                    "full_norms": norm_stats(full), "cosine_view_norms": cosine_norms,
                    "cluster_sizes": sizes, "token_coverage_exact_once": True,
                    "independent_reconstruction_max_error": max_error,
                    "C_vector_bytes": int(c_vectors.nbytes), "H_vector_bytes": int(h_vectors.nbytes),
                    "equal_vector_byte_budget": h_vectors.nbytes == c_vectors.nbytes,
                    "within_vector_byte_budget": h_vectors.nbytes <= c_vectors.nbytes,
                    "cache_read_seconds": read_seconds, "official_pooler_seconds": pooling_seconds,
                    "new_npz_write_and_readback_seconds": write_seconds,
                    "new_npz_bytes": destination.stat().st_size, "new_npz_sha256": digest(destination),
                    "warnings": [{"category": warning.category.__name__, "message": str(warning.message)}
                                 for warning in caught]}
            page_reports.append(item)
            print(f"Page {pid}: full={full.shape}, C={c_vectors.shape}, H={h_vectors.shape}; "
                  f"H unit-error={item['H_norms']['max_unit_error']:.6g}; "
                  f"build={pooling_seconds:.4f}s; bytes C/H={c_vectors.nbytes}/{h_vectors.nbytes}; "
                  f"warnings={len(caught)}", flush=True)
            for warning in item["warnings"]:
                print("  WARNING:", warning["category"], warning["message"], flush=True)
        save_json(record / "page-audit.json", page_reports)
        report["page_audit"] = page_reports
        report["all_pages_equal_vector_byte_budget"] = all(x["equal_vector_byte_budget"] for x in page_reports)
        report["warnings_count"] = sum(len(x["warnings"]) for x in page_reports)
        report["status"] = "PASS_DIAGNOSTIC"
        print("Pooler numerical/partition checks: PASS", flush=True)
        print("All sampled pages exactly match 34-vector budget:",
              report["all_pages_equal_vector_byte_budget"], flush=True)
        print("Model loads: 0; encoding calls: 0; Qdrant calls: 0", flush=True)
        print("STOP: diagnostic only; wait for review before full pooling or retrieval.", flush=True)
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
        report["completed_page_audit"] = page_reports
        raise
    finally:
        save_json(record / "report.json", report)


if __name__ == "__main__":
    main()
