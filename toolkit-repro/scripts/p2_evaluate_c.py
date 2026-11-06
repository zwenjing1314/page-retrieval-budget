import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/encoded-economics-v1"
A_RECORD = ROOT / "records/p2-formal-a-20261107-012250-612a76"
B_RECORD = ROOT / "records/p2-formal-b-20261107-013537-34e55b"
FIELD = "experimental_pooling_3"

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
from qdrant_client import QdrantClient

from prb_cache import CacheStore, digest, json_bytes
from prb_index import build_collection
from benchmarks.vidore_tatdqa_test.metrics import (
    ndcg_at_k, recall_at_k,
)
from visual_rag.retrieval.multi_vector import MultiVectorRetriever


record = ROOT / "records" / (
    "p2-formal-c-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    + "-" + uuid.uuid4().hex[:6]
)
record.mkdir(parents=True, exist_ok=False)
report = {
    "status": "STARTED",
    "group": "C",
    "collection": "prb_p2_c_" + uuid.uuid4().hex[:12],
    "source_a_record": str(A_RECORD),
    "source_b_record": str(B_RECORD),
    "retrieval_mode": "single_experimental_tokens",
    "stored_fields": [FIELD],
    "top_k": 100,
    "exact": True,
    "concurrency": 1,
    "completed_queries": 0,
    "model_encoding_calls": 0,
    "score_tolerance_per_query_token": 1e-3,
    "vector_readback_component_tolerance": 1e-3,
    "scope": "Pooled-only engineering extension; no new method",
    "timing_note": (
        "One cached-query evaluation pass; excludes correctness checks. "
        "Not the P3 warmed three-round benchmark."
    ),
    "resource_note": (
        "Python RSS sampled every 0.1 seconds after imports; lower bound. "
        "Excludes Qdrant/Docker memory."
    ),
    "storage_note": (
        "C online collection stores pooled vectors only. "
        "Research cache still retains full vectors. "
        "Physical collection storage measurement remains pending."
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
        assert all(set(p.vector) == {FIELD} for p in kwargs["points"])
        result = self.actual.upsert(*args, **kwargs)
        self.uploaded += len(kwargs["points"])
        if self.uploaded == 1 or self.uploaded % 50 == 0:
            print(f"Uploaded: {self.uploaded}/452", flush=True)
        return result

    def query_points(self, *args, **kwargs):
        assert kwargs["collection_name"] == report["collection"]
        assert kwargs["using"] == FIELD
        assert kwargs["limit"] == 100
        assert kwargs.get("query_filter") is None
        assert kwargs.get("prefetch") is None
        assert kwargs["search_params"].exact is True
        assert kwargs["with_payload"] is True
        assert kwargs["with_vectors"] is False
        assert np.asarray(kwargs["query"]).shape == current_query.shape
        self.search_calls += 1
        return self.actual.query_points(*args, **kwargs)


def metrics(ranking, rels):
    relevant = {pid for pid, grade in rels.items() if grade > 0}
    assert relevant
    values = {}
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
        values[f"NDCG@{k}"] = dcg / ideal
        assert abs(
            values[f"NDCG@{k}"] - ndcg_at_k(ranking, rels, k)
        ) <= 1e-12
    for k in (5, 10, 100):
        value = len(set(ranking[:k]) & relevant) / len(relevant)
        assert abs(value - recall_at_k(ranking, rels, k)) <= 1e-12
        values[f"Recall@{k}"] = value
    assert all(0 <= value <= 1 for value in values.values())
    return values


def reference_all(query):
    q = query.astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    refs = {}
    for point in points:
        p = point["vectors"][FIELD].astype(np.float32)
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        refs[str(point["id"])] = float(
            (q @ p.T).max(axis=1).sum(dtype=np.float32)
        )
    return refs


save()
sampler = threading.Thread(target=sample_ram, daemon=True)
sampler.start()
print("Record:", record, flush=True)

try:
    report["stage"] = "input_acceptance"
    save()
    accepted = {}

    for group, source in (("A", A_RECORD), ("B", B_RECORD)):
        previous = load(source / "report.json")
        assert previous["status"] == "PASS"
        assert previous["group"] == group
        assert previous["completed_queries"] == 232
        assert previous["counts"] == {
            "pages": 452, "queries": 232, "qrels": 3628
        }
        for name, expected_hash in previous["artifact_sha256"].items():
            assert digest(source / name) == expected_hash, (group, name)
        for relative, expected_hash in previous["code_sha256"].items():
            assert digest(ROOT / relative.replace("\\", "/")) == expected_hash
        for name, expected_version in previous["versions"].items():
            assert version(name) == expected_version
        accepted[group] = previous

    report["source_report_sha256"] = {
        "A": digest(A_RECORD / "report.json"),
        "B": digest(B_RECORD / "report.json"),
    }

    profile = load(CACHE / "profile.json")
    for relative, expected_hash in profile["encoding_code_sha256"].items():
        assert digest(ROOT / relative.replace("\\", "/")) == expected_hash
    for name, expected_version in profile["versions"].items():
        assert version(name) == expected_version

    store = CacheStore(CACHE, profile, create=False)
    assert store.profile_hash == accepted["A"]["cache_profile_sha256"]
    assert store.profile_hash == accepted["B"]["cache_profile_sha256"]
    manifest = store.manifest()
    assert manifest["complete"] and not manifest["pending_directories"]
    assert manifest == load(A_RECORD / "cache-manifest.json")
    assert manifest == load(B_RECORD / "cache-manifest.json")
    write_json("cache-manifest.json", manifest)

    mapping = load(A_RECORD / "page-id-mapping.json")
    assert mapping == load(B_RECORD / "page-id-mapping.json")
    assert len(mapping) == 452
    assert {x["corpus_id"] for x in mapping} == set(
        profile["expected_page_ids"]
    )

    points = []
    for item in sorted(mapping, key=lambda x: int(x["corpus_id"])):
        arrays, meta = store.read("pages", item["corpus_id"])
        assert meta["identity"]["doc_id"] == item["doc_id"]
        # Deliberately pass no "initial" vector to the index builder.
        points.append({
            "id": item["point_id"],
            "payload": {
                "corpus_id": item["corpus_id"],
                "doc_id": item["doc_id"],
                "diagnostic": False,
            },
            "vectors": {FIELD: arrays["pooled"]},
        })
    expected = {str(p["id"]): p for p in points}
    assert len(expected) == 452
    write_json("page-id-mapping.json", mapping)

    data = ROOT / (
        "data/economics_reports_v2-" + profile["dataset_revision"]
    )
    assert digest(data / "download-manifest.json") == (
        profile["dataset_download_manifest_sha256"]
    )
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
    assert qrels == load(B_RECORD / "qrels.json")
    assert set(qrels) == set(queries)
    assert sum(map(len, qrels.values())) == 3628
    assert all(set(rels) <= set(expected) for rels in qrels.values())
    assert all(grade == 2 for rels in qrels.values() for grade in rels.values())
    write_json("qrels.json", qrels)

    with (B_RECORD / "prefetch-candidates.jsonl").open(
        encoding="utf-8"
    ) as source:
        candidate_rows = [json.loads(line) for line in source if line.strip()]
    b_candidates = {x["query_id"]: x for x in candidate_rows}
    assert len(candidate_rows) == len(b_candidates) == 232
    assert set(b_candidates) == set(queries)

    code_paths = [
        Path(__file__),
        ROOT / "scripts/prb_index.py",
        ROOT / "scripts/prb_cache.py",
        ROOT / "visual-rag-toolkit/visual_rag/retrieval/multi_vector.py",
        ROOT / "visual-rag-toolkit/visual_rag/retrieval/single_stage.py",
        ROOT / "visual-rag-toolkit/benchmarks/vidore_tatdqa_test/metrics.py",
    ]
    report["code_sha256"] = {
        str(path.relative_to(ROOT)): digest(path) for path in code_paths
    }
    report["cache_profile_sha256"] = store.profile_hash
    report["versions"] = {
        name: version(name) for name in accepted["A"]["versions"]
    }
    report["counts"] = {"pages": 452, "queries": 232, "qrels": 3628}
    report["outer_repository_commit"] = subprocess.run(
        ["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    (record / "working-changes.patch").write_bytes(subprocess.run(
        ["git", "-C", str(ROOT.parent), "diff", "HEAD", "--",
         "toolkit-repro/scripts", "toolkit-repro/visual-rag-toolkit"],
        check=True, capture_output=True,
    ).stdout)
    print("A/B evidence and shared full cache: PASS", flush=True)

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
    info = build_collection(checked, report["collection"], "C", points)
    assert checked.uploaded == 452
    assert info.config.quantization_config is None
    assert info.config.params.vectors[FIELD].quantization_config is None
    (record / "collection-before.json").write_bytes(
        info.model_dump_json(indent=2).encode("utf-8")
    )

    # Audit all 452 points, including actual stored vector contents.
    seen = set()
    audit = []
    maximum_error = 0.0
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=report["collection"],
            offset=offset,
            limit=16,
            with_vectors=True,
            with_payload=True,
        )
        for point in batch:
            pid = str(point.id)
            assert pid in expected and pid not in seen
            seen.add(pid)
            assert set(point.vector) == {FIELD}
            assert point.payload == expected[pid]["payload"]

            actual = np.asarray(point.vector[FIELD], dtype=np.float32)
            ref = expected[pid]["vectors"][FIELD].astype(np.float32)
            ref /= np.linalg.norm(ref, axis=1, keepdims=True)
            assert actual.shape == (34, 128)
            assert np.isfinite(actual).all()
            error = float(np.max(np.abs(actual - ref)))
            assert error <= report["vector_readback_component_tolerance"]
            maximum_error = max(maximum_error, error)
            audit.append({
                "point_id": pid,
                "corpus_id": point.payload["corpus_id"],
                "actual_fields": list(point.vector),
                "actual_vector_shape": list(actual.shape),
                "max_abs_error_from_normalized_cache": error,
            })
        if offset is None:
            break
    assert seen == set(expected)
    write_json("online-field-audit.json", {
        "status": "PASS",
        "points": audit,
        "points_with_initial_field": 0,
        "total_pooled_vectors": 452 * 34,
        "max_vector_readback_component_error": maximum_error,
    })
    report["index_prepare_and_readback_seconds"] = time.perf_counter() - t
    report["actual_stored_fields"] = list(info.config.params.vectors)
    report["index_id_set_matches_official"] = True
    report["points_with_initial_field"] = 0
    report["theoretical_vector_bytes"] = 452 * 34 * 128 * 2

    print("C stored fields: ['experimental_pooling_3'] PASS", flush=True)
    print("All 452 points: no initial field; pooled readback PASS", flush=True)

    retriever = MultiVectorRetriever(
        collection_name=report["collection"],
        qdrant_client=checked,
        embedder=NoEncoding(),
        experimental_vector_name=FIELD,
        request_timeout=300,
        max_retries=1,
    )

    order = ["0", "20"] + [
        qid for qid in sorted(queries, key=int)
        if qid not in {"0", "20"}
    ]
    rows = []

    for number, qid in enumerate(order, 1):
        report["stage"] = (
            "diagnostic_queries" if number <= 2 else "all_query_evaluation"
        )
        report["current_query"] = qid
        save()
        current_query = queries[qid]
        tokens = len(current_query)
        bound = report["score_tolerance_per_query_token"] * tokens

        t = time.perf_counter()
        hits = retriever.search_embedded(
            query_embedding=current_query,
            top_k=100,
            mode="single_experimental_tokens",
        )
        seconds = time.perf_counter() - t

        ranking = [str(h["id"]) for h in hits]
        scores = {str(h["id"]): float(h["score_final"]) for h in hits}
        assert len(hits) == len(set(ranking)) == 100
        assert set(ranking) <= set(expected)
        assert all(math.isfinite(score) for score in scores.values())
        assert all(
            scores[a] >= scores[b] for a, b in zip(ranking, ranking[1:])
        )
        for hit in hits:
            assert hit["payload"] == expected[str(hit["id"])]["payload"]

        # Compare C against B's independently replayed pooled search.
        evidence = b_candidates[qid]
        candidates = evidence["candidates"]
        assert len(candidates) == 256
        replay_scores = {
            x["point_id"]: float(x["score"]) for x in candidates
        }
        assert len(replay_scores) == 256
        boundary_ids = set(evidence["boundary_tied_point_ids"])
        assert set(ranking) <= set(replay_scores) | boundary_ids

        max_replay_error = max(
            abs(
                scores[pid] - replay_scores.get(pid, evidence["cutoff_score"])
            ) / tokens
            for pid in ranking
        )
        assert max_replay_error <= report["score_tolerance_per_query_token"]

        # Do not require identical order inside the registered tie tolerance.
        omitted = set(replay_scores) - set(ranking)
        assert max(replay_scores[pid] for pid in omitted) <= (
            min(scores.values()) + 2 * bound
        )

        if qid in {"0", "20"}:
            refs = reference_all(current_query)
            full_error = max(
                abs(scores[pid] - refs[pid]) / tokens for pid in ranking
            )
            assert full_error <= report["score_tolerance_per_query_token"]

            minimum_previous = float("inf")
            for pid in ranking:
                assert minimum_previous + 2 * bound >= refs[pid]
                minimum_previous = min(minimum_previous, refs[pid])

            excluded = set(refs) - set(ranking)
            boundary_gap = (
                max(refs[pid] for pid in excluded)
                - min(refs[pid] for pid in ranking)
            )
            assert boundary_gap <= 2 * bound

            append_json("diagnostics.jsonl", {
                "query_id": qid,
                "max_score_error_per_token": full_error,
                "top100_boundary_gap": boundary_gap,
                "all_452_fp32_reference_scores": refs,
            })
            print(f"C query {qid} full-corpus score/top100: PASS", flush=True)

        values = metrics(ranking, qrels[qid])
        rows.append({
            "query_id": qid,
            "query_tokens": tokens,
            "relevant_pages": len(qrels[qid]),
            "search_call_seconds": seconds,
            "max_score_error_per_token_vs_b_pooled_replay": max_replay_error,
            "top100_overlap_with_b_pooled_replay": len(
                set(ranking) & {x["point_id"] for x in candidates[:100]}
            ),
            **values,
        })
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
        report["completed_queries"] = number
        if number <= 2 or number % 20 == 0 or number == 232:
            save()
            print(f"Evaluated C: {number}/232", flush=True)

    assert len(rows) == 232
    assert {row["query_id"] for row in rows} == set(qrels)
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
        name: math.fsum(row[name] for row in rows) / 232 for name in names
    }
    comparison = {
        "A": accepted["A"]["aggregate_metrics"],
        "B": accepted["B"]["aggregate_metrics"],
        "C": aggregate,
        "C_minus_A": {
            name: aggregate[name] - accepted["A"]["aggregate_metrics"][name]
            for name in names
        },
        "C_minus_B": {
            name: aggregate[name] - accepted["B"]["aggregate_metrics"][name]
            for name in names
        },
    }
    write_json("metrics.json", {
        "group": "C", "query_count": 232, "values": aggregate
    })
    write_json("comparison-abc.json", comparison)

    final_info = client.get_collection(report["collection"])
    assert final_info.config == info.config
    assert set(final_info.config.params.vectors) == {FIELD}
    assert final_info.indexed_vectors_count == 0
    assert client.count(report["collection"], exact=True).count == 452
    (record / "collection-after.json").write_bytes(
        final_info.model_dump_json(indent=2).encode("utf-8")
    )
    for relative, expected_hash in report["code_sha256"].items():
        assert digest(ROOT / relative) == expected_hash

    report["aggregate_metrics"] = aggregate
    report["comparison_abc"] = comparison
    report["actual_query_calls"] = checked.search_calls
    report["evaluation_search_call_seconds_total"] = math.fsum(
        row["search_call_seconds"] for row in rows
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
print("C minus A:", report["comparison_abc"]["C_minus_A"])
print("C minus B:", report["comparison_abc"]["C_minus_B"])
print("Points with initial field:", report["points_with_initial_field"])
print("Evaluation queries:", report["completed_queries"])
print("Model encoding calls:", report["model_encoding_calls"])
print("Record:", record)
print("P2 formal C evaluation: PASS")