import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/encoded-economics-v1"
A_RECORD = ROOT / "records/p2-formal-a-20261107-012250-612a76"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import csv
import json
import math
import subprocess
import threading
import time
import uuid
from datetime import datetime
from importlib.metadata import version

import numpy as np
import psutil
import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

from prb_cache import CacheStore, digest, json_bytes
from prb_index import build_collection
from benchmarks.vidore_tatdqa_test.metrics import (
    ndcg_at_k, recall_at_k,
)
from visual_rag.retrieval.multi_vector import MultiVectorRetriever


record = ROOT / "records" / (
    "p2-formal-b-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    + "-" + uuid.uuid4().hex[:6]
)
record.mkdir(parents=True, exist_ok=False)

report = {
    "status": "STARTED",
    "group": "B",
    "collection": "prb_p2_b_" + uuid.uuid4().hex[:12],
    "source_a_record": str(A_RECORD),
    "retrieval_mode": "two_stage",
    "stage1_mode": "tokens_vs_experimental_pooling",
    "stored_fields": ["initial", "experimental_pooling_3"],
    "prefetch_k": 256,
    "top_k": 100,
    "exact": True,
    "concurrency": 1,
    "completed_queries": 0,
    "score_tolerance_per_query_token": 1e-3,
    "model_encoding_calls": 0,
    "candidate_note": (
        "Candidates independently replayed with the same pooled query. "
        "Not an internal server prefetch trace. Exact boundary-score ties "
        "are recorded, including all tied IDs."
    ),
    "timing_note": (
        "B call timing excludes candidate replay and correctness checks. "
        "This is not the P3 warmed three-round benchmark."
    ),
    "resource_note": (
        "Python RSS sampled every 0.1 seconds after imports; lower bound. "
        "Excludes Qdrant/Docker memory."
    ),
    "storage_note": (
        "Vector bytes are theoretical. Physical storage remains pending."
    ),
}

client = None
started = time.perf_counter()
stop = threading.Event()
process = psutil.Process()
ram = {"peak_rss_bytes": process.memory_info().rss, "samples": 0}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save():
    (record / "report.json").write_bytes(json_bytes(report))


def write_json(name, value):
    (record / name).write_bytes(json_bytes(value))


def append_json(name, value):
    with (record / name).open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False) + "\n")


def sample_ram():
    while not stop.is_set():
        ram["peak_rss_bytes"] = max(
            ram["peak_rss_bytes"], process.memory_info().rss
        )
        ram["samples"] += 1
        stop.wait(0.1)


class NoEncoding:
    def embed_query(self, *args, **kwargs):
        raise AssertionError("Cached queries only")


class CheckedClient:
    def __init__(self, actual):
        self.actual = actual
        self.uploaded = 0
        self.search_calls = 0

    def __getattr__(self, name):
        return getattr(self.actual, name)

    def upsert(self, *args, **kwargs):
        assert kwargs.get("wait") is True
        result = self.actual.upsert(*args, **kwargs)
        self.uploaded += len(kwargs["points"])
        if self.uploaded == 1 or self.uploaded % 50 == 0:
            print(f"Uploaded: {self.uploaded}/452", flush=True)
        return result

    def query_points(self, *args, **kwargs):
        assert kwargs["collection_name"] == report["collection"]
        assert kwargs["using"] == "initial"
        assert kwargs["limit"] == 100
        assert kwargs.get("query_filter") is None
        assert kwargs.get("with_payload") is True
        assert kwargs.get("with_vectors", False) is False
        assert kwargs["search_params"].exact is True

        prefetch = kwargs["prefetch"]
        assert isinstance(prefetch, list) and len(prefetch) == 1
        stage1 = prefetch[0]
        assert stage1.using == "experimental_pooling_3"
        assert stage1.limit == 256
        assert stage1.params.exact is True
        assert stage1.filter is None
        assert np.array_equal(
            np.asarray(stage1.query, dtype=np.float32),
            np.asarray(kwargs["query"], dtype=np.float32),
        )
        self.search_calls += 1
        return self.actual.query_points(*args, **kwargs)


def verify_hits(hits):
    assert len(hits) == 100
    ids = [str(hit["id"]) for hit in hits]
    assert len(set(ids)) == 100
    assert set(ids) <= set(expected)
    scores = [float(hit["score_final"]) for hit in hits]
    assert all(math.isfinite(x) for x in scores)
    assert all(a >= b for a, b in zip(scores, scores[1:]))
    for hit in hits:
        assert hit["payload"] == expected[str(hit["id"])]["payload"]
    return ids


def metrics(ranking, rels):
    relevant = {pid for pid, grade in rels.items() if grade > 0}
    assert relevant
    result = {}

    for k in (5, 10):
        dcg = math.fsum(
            1 / math.log2(i + 2)
            for i, pid in enumerate(ranking[:k])
            if pid in relevant
        )
        ideal = math.fsum(
            1 / math.log2(i + 2)
            for i in range(min(k, len(relevant)))
        )
        value = dcg / ideal
        assert abs(value - ndcg_at_k(ranking, rels, k)) <= 1e-12
        result[f"NDCG@{k}"] = value

    for k in (5, 10, 100):
        value = len(set(ranking[:k]) & relevant) / len(relevant)
        assert abs(value - recall_at_k(ranking, rels, k)) <= 1e-12
        result[f"Recall@{k}"] = value

    assert all(0 <= x <= 1 for x in result.values())
    return result


def pooled_reference(query, vectors):
    q = query.astype(np.float32)
    p = vectors.astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return float((q @ p.T).max(axis=1).sum(dtype=np.float32))


save()
sampler = threading.Thread(target=sample_ram, daemon=True)
sampler.start()
print("Record:", record, flush=True)

try:
    report["stage"] = "input_acceptance"
    save()

    a_report = load(A_RECORD / "report.json")
    assert a_report["status"] == "PASS"
    assert a_report["group"] == "A"
    assert a_report["completed_queries"] == 232
    assert a_report["counts"] == {
        "pages": 452, "queries": 232, "qrels": 3628
    }
    report["source_a_report_sha256"] = digest(A_RECORD / "report.json")

    # Check accepted A artifacts and code before reusing them.
    for name, expected_hash in a_report["artifact_sha256"].items():
        assert digest(A_RECORD / name) == expected_hash, name
    for relative, expected_hash in a_report["code_sha256"].items():
        path = ROOT / relative.replace("\\", "/")
        assert digest(path) == expected_hash, (
            "Source changed since A acceptance", path
        )
    for name, expected_version in a_report["versions"].items():
        assert version(name) == expected_version

    profile = load(CACHE / "profile.json")
    assert profile["dataset_revision"] == (
        "76fe40166ba07b1bf50457f5c6057cacdd045f10"
    )
    for relative, expected_hash in profile["encoding_code_sha256"].items():
        assert digest(ROOT / relative.replace("\\", "/")) == expected_hash
    for name, expected_version in profile["versions"].items():
        assert version(name) == expected_version

    store = CacheStore(CACHE, profile, create=False)
    assert store.profile_hash == a_report["cache_profile_sha256"]
    manifest = store.manifest()
    assert manifest["complete"]
    assert not manifest["pending_directories"]
    assert manifest == load(A_RECORD / "cache-manifest.json")
    write_json("cache-manifest.json", manifest)

    data = ROOT / (
        "data/economics_reports_v2-" + profile["dataset_revision"]
    )
    assert digest(data / "download-manifest.json") == (
        profile["dataset_download_manifest_sha256"]
    )

    mapping = load(A_RECORD / "page-id-mapping.json")
    assert len(mapping) == 452
    assert {x["corpus_id"] for x in mapping} == set(
        profile["expected_page_ids"]
    )
    points = []
    for item in sorted(mapping, key=lambda x: int(x["corpus_id"])):
        pid = item["corpus_id"]
        arrays, meta = store.read("pages", pid)
        assert meta["identity"]["doc_id"] == item["doc_id"]
        points.append({
            "id": item["point_id"],
            "payload": {
                "corpus_id": pid,
                "doc_id": item["doc_id"],
                "diagnostic": False,
            },
            "vectors": {
                "initial": arrays["vectors"],
                "experimental_pooling_3": arrays["pooled"],
            },
        })
    expected = {str(p["id"]): p for p in points}
    assert len(expected) == 452
    write_json("page-id-mapping.json", mapping)

    query_rows = pq.read_table(
        data / "queries/test-00000-of-00001.parquet",
        columns=["query-id", "query", "language"],
    ).to_pylist()
    official_queries = {str(x["query-id"]): x for x in query_rows}
    assert len(query_rows) == len(official_queries) == 232
    assert set(official_queries) == set(profile["expected_query_ids"])

    queries = {}
    for qid, row in official_queries.items():
        arrays, meta = store.read("queries", qid)
        assert meta["identity"]["text"] == row["query"]
        assert meta["identity"]["language"] == row["language"]
        queries[qid] = arrays["vectors"]

    qrels = load(A_RECORD / "qrels.json")
    assert set(qrels) == set(queries)
    assert sum(map(len, qrels.values())) == 3628
    assert all(set(rels) <= set(expected) for rels in qrels.values())
    assert all(grade == 2 for rels in qrels.values() for grade in rels.values())
    write_json("qrels.json", qrels)

    with (A_RECORD / "rankings.jsonl").open(encoding="utf-8") as source:
        a_rows = [json.loads(line) for line in source if line.strip()]
    a_rankings = {x["query_id"]: x["hits"] for x in a_rows}
    assert len(a_rows) == len(a_rankings) == 232
    assert set(a_rankings) == set(queries)
    for hits in a_rankings.values():
        assert len(hits) == 100
        assert len({h["point_id"] for h in hits}) == 100
        assert {h["point_id"] for h in hits} <= set(expected)

    a_diagnostics = {
        x["query_id"]: x for x in load(A_RECORD / "diagnostics.json")
    }
    assert {"0", "20"} <= set(a_diagnostics)

    code_paths = [
        Path(__file__),
        ROOT / "scripts/prb_index.py",
        ROOT / "scripts/prb_cache.py",
        ROOT / "visual-rag-toolkit/visual_rag/retrieval/multi_vector.py",
        ROOT / "visual-rag-toolkit/visual_rag/retrieval/two_stage.py",
        ROOT / "visual-rag-toolkit/benchmarks/vidore_tatdqa_test/metrics.py",
    ]
    report["code_sha256"] = {
        str(path.relative_to(ROOT)): digest(path)
        for path in code_paths
    }
    report["cache_profile_sha256"] = store.profile_hash
    report["versions"] = {
        name: version(name) for name in a_report["versions"]
    }
    report["counts"] = {"pages": 452, "queries": 232, "qrels": 3628}
    report["outer_repository_commit"] = subprocess.run(
        ["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    patch = subprocess.run(
        ["git", "-C", str(ROOT.parent), "diff", "HEAD", "--",
         "toolkit-repro/scripts", "toolkit-repro/visual-rag-toolkit"],
        check=True, capture_output=True,
    ).stdout
    (record / "working-changes.patch").write_bytes(patch)

    print("A evidence and shared full cache: PASS", flush=True)

    client = QdrantClient(
        url="http://127.0.0.1:16333",
        timeout=300,
        trust_env=False,
    )
    assert client.info().version == "1.14.1"
    report["qdrant_server_version"] = client.info().version
    checked = CheckedClient(client)

    report["stage"] = "build_and_readback"
    save()
    t = time.perf_counter()
    info = build_collection(
        checked, report["collection"], "B", points
    )
    report["build_upload_and_readback_seconds"] = time.perf_counter() - t
    assert checked.uploaded == 452
    assert info.config.quantization_config is None
    assert all(
        vector.quantization_config is None
        for vector in info.config.params.vectors.values()
    )
    (record / "collection-before.json").write_bytes(
        info.model_dump_json(indent=2).encode("utf-8")
    )
    report["index_id_set_matches_official"] = True
    report["actual_stored_fields"] = list(info.config.params.vectors)
    report["theoretical_vector_bytes"] = 452 * (1024 + 34) * 128 * 2
    print("B fields: initial + experimental_pooling_3 PASS", flush=True)
    print("B full ID/config/payload readback: PASS", flush=True)

    retriever = MultiVectorRetriever(
        collection_name=report["collection"],
        qdrant_client=checked,
        embedder=NoEncoding(),
        experimental_vector_name="experimental_pooling_3",
        request_timeout=300,
        max_retries=1,
    )

    # Check both accepted diagnostic queries before the remaining queries.
    order = ["0", "20"] + [
        qid for qid in sorted(queries, key=int)
        if qid not in {"0", "20"}
    ]
    rows = []

    for number, qid in enumerate(order, 1):
        report["current_query"] = qid
        report["stage"] = (
            "diagnostic_queries" if number <= 2 else "all_query_evaluation"
        )
        save()
        query = queries[qid]
        tokens = len(query)
        bound = report["score_tolerance_per_query_token"] * tokens

        # Author's native server-side Prefetch -> full-vector rerank.
        t = time.perf_counter()
        hits = retriever.search_embedded(
            query_embedding=query,
            top_k=100,
            mode="two_stage",
            prefetch_k=256,
            stage1_mode="tokens_vs_experimental_pooling",
        )
        search_seconds = time.perf_counter() - t
        ranking = verify_hits(hits)

        # Separate replay for candidate evidence, outside B call timing.
        t = time.perf_counter()
        pooled = client.query_points(
            collection_name=report["collection"],
            query=query.astype(np.float32).tolist(),
            using="experimental_pooling_3",
            limit=452,
            search_params=models.SearchParams(exact=True),
            with_payload=False,
            with_vectors=False,
            timeout=300,
        ).points
        replay_seconds = time.perf_counter() - t
        pooled_ids = [str(x.id) for x in pooled]
        pooled_scores = [float(x.score) for x in pooled]
        assert len(pooled_ids) == len(set(pooled_ids)) == 452
        assert set(pooled_ids) == set(expected)
        assert all(math.isfinite(s) for s in pooled_scores)
        assert all(a >= b for a, b in zip(pooled_scores, pooled_scores[1:]))

        cutoff = pooled_scores[255]
        replay_candidates = set(pooled_ids[:256])
        strict_candidates = {
            pid for pid, score in zip(pooled_ids, pooled_scores)
            if score > cutoff
        }
        boundary_ties = {
            pid for pid, score in zip(pooled_ids, pooled_scores)
            if score == cutoff
        }
        possible_candidates = strict_candidates | boundary_ties
        ambiguous = pooled_scores[255] == pooled_scores[256]

        assert set(ranking) <= possible_candidates, (
            "B result outside possible top256 candidates", qid
        )
        if not ambiguous:
            assert set(ranking) <= replay_candidates

        # Full-vector scores must agree with A for every shared result.
        a_hits = a_rankings[qid]
        a_scores = {h["point_id"]: float(h["score"]) for h in a_hits}
        b_scores = {str(h["id"]): float(h["score_final"]) for h in hits}
        common = set(a_scores) & set(b_scores)
        common_error = max(
            (abs(a_scores[pid] - b_scores[pid]) / tokens for pid in common),
            default=0.0,
        )
        assert common_error <= report["score_tolerance_per_query_token"]

        # A high-scoring point definitely retained by stage1 cannot vanish.
        for pid in set(a_scores) & strict_candidates:
            if pid not in b_scores:
                assert a_scores[pid] <= min(b_scores.values()) + 2 * bound, (
                    "Retained high-score A point missing after rerank", qid, pid
                )

        diagnostic = None
        if qid in {"0", "20"}:
            refs = a_diagnostics[qid]["all_452_fp32_reference_scores"]
            assert set(refs) == set(expected)
            full_error = max(
                abs(b_scores[pid] - refs[pid]) / tokens
                for pid in ranking
            )
            assert full_error <= report["score_tolerance_per_query_token"]

            minimum_previous = float("inf")
            for pid in ranking:
                assert minimum_previous + 2 * bound >= refs[pid]
                minimum_previous = min(minimum_previous, refs[pid])

            definite_omitted = strict_candidates - set(ranking)
            if definite_omitted:
                assert max(refs[pid] for pid in definite_omitted) <= (
                    min(refs[pid] for pid in ranking) + 2 * bound
                )

            pooled_error = max(
                abs(score - pooled_reference(
                    query, expected[pid]["vectors"]["experimental_pooling_3"]
                )) / tokens
                for pid, score in zip(pooled_ids, pooled_scores)
            )
            assert pooled_error <= report["score_tolerance_per_query_token"]
            diagnostic = {
                "query_id": qid,
                "full_score_max_error_per_token": full_error,
                "pooled_score_max_error_per_token": pooled_error,
            }
            append_json("diagnostics.jsonl", diagnostic)
            print(
                f"B query {qid} pooled/full scores and rerank: PASS",
                flush=True,
            )

        relevant = set(qrels[qid])
        coverage = len(relevant & replay_candidates) / len(relevant)
        upper = len(relevant & possible_candidates) / len(relevant)
        values = metrics(ranking, qrels[qid])
        assert values["Recall@100"] <= upper + 1e-12

        a_metrics = metrics([h["point_id"] for h in a_hits], qrels[qid])
        row = {
            "query_id": qid,
            "query_tokens": tokens,
            "relevant_pages": len(relevant),
            "search_call_seconds": search_seconds,
            "candidate_replay_seconds": replay_seconds,
            "stage1_recall256_replay": coverage,
            "stage1_recall256_upper_with_boundary_ties": upper,
            "stage1_boundary_ambiguous": ambiguous,
            "a_top100_in_replay256": len(set(a_scores) & replay_candidates),
            "a_b_common_top100": len(common),
            "a_b_common_max_score_error_per_token": common_error,
            **values,
            **{
                "delta_A_" + name: value - a_metrics[name]
                for name, value in values.items()
            },
        }
        rows.append(row)

        append_json("rankings.jsonl", {
            "query_id": qid,
            "hits": [
                {
                    "rank": rank,
                    "point_id": str(hit["id"]),
                    "corpus_id": hit["payload"]["corpus_id"],
                    "doc_id": hit["payload"]["doc_id"],
                    "score": float(hit["score_final"]),
                }
                for rank, hit in enumerate(hits, 1)
            ],
        })
        append_json("prefetch-candidates.jsonl", {
            "query_id": qid,
            "provenance": "Independent exact pooled-search replay",
            "cutoff_score": cutoff,
            "boundary_ambiguous": ambiguous,
            "boundary_tied_point_ids": sorted(boundary_ties),
            "stage1_recall256_replay": coverage,
            "stage1_recall256_upper_with_boundary_ties": upper,
            "candidates": [
                {
                    "rank": rank,
                    "point_id": pid,
                    "corpus_id": expected[pid]["payload"]["corpus_id"],
                    "score": score,
                }
                for rank, (pid, score) in enumerate(
                    zip(pooled_ids[:256], pooled_scores[:256]), 1
                )
            ],
        })

        report["completed_queries"] = number
        if number <= 2 or number % 20 == 0 or number == 232:
            save()
            print(f"Evaluated B: {number}/232", flush=True)

    assert len(rows) == 232
    assert {x["query_id"] for x in rows} == set(qrels)
    assert checked.search_calls == 232

    with (record / "per-query.tsv").open(
        "x", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)

    names = ["NDCG@5", "NDCG@10", "Recall@5", "Recall@10", "Recall@100"]
    aggregate = {
        name: math.fsum(row[name] for row in rows) / 232
        for name in names
    }
    a_aggregate = {
        name: math.fsum(
            metrics([h["point_id"] for h in a_rankings[qid]], qrels[qid])[name]
            for qid in queries
        ) / 232
        for name in names
    }
    for name in names:
        assert abs(a_aggregate[name] - a_report["aggregate_metrics"][name]) <= 1e-12

    comparison = {
        "A": a_aggregate,
        "B": aggregate,
        "B_minus_A": {
            name: aggregate[name] - a_aggregate[name] for name in names
        },
        "stage1_recall256_replay_macro": math.fsum(
            row["stage1_recall256_replay"] for row in rows
        ) / 232,
        "boundary_ambiguous_query_count": sum(
            row["stage1_boundary_ambiguous"] for row in rows
        ),
    }
    write_json("metrics.json", {
        "group": "B", "query_count": 232, "values": aggregate
    })
    write_json("comparison-with-a.json", comparison)

    final_info = client.get_collection(report["collection"])
    assert final_info.config == info.config
    assert final_info.indexed_vectors_count == 0
    assert client.count(report["collection"], exact=True).count == 452
    (record / "collection-after.json").write_bytes(
        final_info.model_dump_json(indent=2).encode("utf-8")
    )
    for relative, expected_hash in report["code_sha256"].items():
        assert digest(ROOT / relative) == expected_hash

    report["aggregate_metrics"] = aggregate
    report["comparison_with_a"] = comparison
    report["actual_two_stage_query_calls"] = checked.search_calls
    report["candidate_replay_calls"] = 232
    report["evaluation_search_call_seconds_total"] = math.fsum(
        row["search_call_seconds"] for row in rows
    )
    report["candidate_replay_seconds_total"] = math.fsum(
        row["candidate_replay_seconds"] for row in rows
    )
    report["artifact_sha256"] = {
        path.name: digest(path)
        for path in record.iterdir()
        if path.is_file() and path.name != "report.json"
    }
    report["stage"] = "finished"
    report["status"] = "PASS"

except BaseException as exc:
    report["status"] = (
        "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
    )
    report["error"] = repr(exc)
    raise

finally:
    stop.set()
    sampler.join(timeout=2)
    if client is not None:
        client.close()
    report["ram_sampling"] = ram
    report["script_work_wall_seconds"] = time.perf_counter() - started
    save()

print("Collection:", report["collection"])
print("Counts:", report["counts"])
print("Metrics:", report["aggregate_metrics"])
print("B minus A:", report["comparison_with_a"]["B_minus_A"])
print(
    "Stage1 Recall@256 replay:",
    report["comparison_with_a"]["stage1_recall256_replay_macro"],
)
print(
    "Boundary-ambiguous queries:",
    report["comparison_with_a"]["boundary_ambiguous_query_count"],
)
print("Evaluation queries:", report["completed_queries"])
print("Model encoding calls:", report["model_encoding_calls"])
print("Record:", record)
print("P2 formal B evaluation: PASS")