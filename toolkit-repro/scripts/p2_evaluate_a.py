import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/encoded-economics-v1"
DATA = ROOT / (
    "data/economics_reports_v2-"
    "76fe40166ba07b1bf50457f5c6057cacdd045f10"
)
SOURCE = ROOT / "records/p2-full-cache-20261107-005355-ff459c"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
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
from prb_resume_a import resume_collection_a
from benchmarks.vidore_tatdqa_test.dataset_loader import (
    _normalize_qrels,
    _stable_uuid,
)
from benchmarks.vidore_tatdqa_test.metrics import (
    ndcg_at_k,
    recall_at_k,
)
from visual_rag.retrieval.multi_vector import MultiVectorRetriever


parser = argparse.ArgumentParser()
parser.add_argument("--resume-record", type=Path)
args = parser.parse_args()
resume_source = None
resume_report = None

if args.resume_record is not None:
    resume_source = args.resume_record.resolve()
    assert resume_source.parent == (ROOT / "records").resolve()
    resume_report = json.loads(
        (resume_source / "report.json").read_text(encoding="utf-8")
    )
    assert resume_report["status"] in ("FAILED", "INTERRUPTED")
    assert resume_report["group"] == "A"
    assert resume_report["stage"] == "build_and_readback"
    assert resume_report["completed_queries"] == 0
    assert resume_report["stored_fields"] == ["initial"]
    assert resume_report["retrieval_mode"] == "single_full"

record = ROOT / "records" / (
    "p2-formal-a-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    + "-" + uuid.uuid4().hex[:6]
)
record.mkdir(parents=True, exist_ok=False)

report = {
    "status": "STARTED",
    "group": "A",
    "source_full_cache_record": str(SOURCE),
    "collection": (
        resume_report["collection"] if resume_report
        else "prb_p2_a_" + uuid.uuid4().hex[:12]
    ),
    "scope": "Economics full 452 pages and 232 queries",
    "retrieval_mode": "single_full",
    "stored_fields": ["initial"],
    "top_k": 100,
    "exact": True,
    "concurrency": 1,
    "score_tolerance_per_query_token": 1e-3,
    "completed_queries": 0,
    "resource_note": (
        "Python process RSS sampled every 0.1 seconds; lower bound. "
        "Does not measure Qdrant/Docker memory."
    ),
    "timing_note": (
        "One evaluation pass, cached queries, no model encoding. "
        "Not the P3 warmed three-round benchmark."
    ),
    "storage_note": (
        "Theoretical vector bytes only at this stage. "
        "Physical collection storage acceptance remains pending."
    ),
}

started = time.perf_counter()
stop = threading.Event()
process = psutil.Process()
ram = {"peak_rss_bytes": process.memory_info().rss, "samples": 0}
client = None


def save():
    (record / "report.json").write_bytes(json_bytes(report))


def sample_ram():
    while not stop.is_set():
        ram["peak_rss_bytes"] = max(
            ram["peak_rss_bytes"], process.memory_info().rss
        )
        ram["samples"] += 1
        stop.wait(0.1)


def write_json(name, value):
    (record / name).write_bytes(json_bytes(value))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class NoEncoding:
    def embed_query(self, *args, **kwargs):
        raise AssertionError("Use cached query vectors only")


class CheckedClient:
    def __init__(self, actual):
        self.actual = actual
        self.uploads = 0
        self.search_calls = 0

    def __getattr__(self, name):
        return getattr(self.actual, name)

    def upsert(self, *args, **kwargs):
        assert kwargs.get("wait") is True
        result = self.actual.upsert(*args, **kwargs)
        self.uploads += len(kwargs["points"])
        if self.uploads == 1 or self.uploads % 50 == 0:
            print(f"Uploaded this run: {self.uploads}", flush=True)
        return result

    def query_points(self, *args, **kwargs):
        assert kwargs["collection_name"] == report["collection"]
        assert kwargs.get("using") == "initial"
        assert kwargs.get("limit") == 100
        assert kwargs.get("query_filter") is None
        assert kwargs.get("prefetch") is None
        assert kwargs.get("with_payload") is True
        assert kwargs.get("with_vectors") is False
        params = kwargs.get("search_params")
        assert params is not None and params.exact is True
        self.search_calls += 1
        return self.actual.query_points(*args, **kwargs)


def checked_hits(hits):
    assert len(hits) == 100, ("Unexpected result count", len(hits))
    ids = [str(h["id"]) for h in hits]
    assert len(set(ids)) == 100
    assert set(ids) <= set(expected_points)
    scores = [float(h["score_final"]) for h in hits]
    assert all(math.isfinite(s) for s in scores)
    assert all(a >= b for a, b in zip(scores, scores[1:]))

    for hit in hits:
        key = str(hit["id"])
        assert hit["payload"] == expected_points[key]["payload"]
    return ids


def search(qid):
    t = time.perf_counter()
    hits = retriever.search_embedded(
        query_embedding=queries[qid],
        top_k=100,
        mode="single_full",
    )
    seconds = time.perf_counter() - t
    checked_hits(hits)
    return hits, seconds


def full_reference(q):
    """Independent FP32 cosine MaxSim over every official page."""
    q = q.astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    refs = {}

    for start in range(0, len(points), 16):
        block = points[start:start + 16]
        p = np.stack([
            point["vectors"]["initial"] for point in block
        ]).astype(np.float32)
        p /= np.linalg.norm(p, axis=2, keepdims=True)
        similarities = np.matmul(
            q[None, :, :], p.transpose(0, 2, 1)
        )
        values = similarities.max(axis=2).sum(
            axis=1, dtype=np.float32
        )
        for point, value in zip(block, values):
            refs[str(point["id"])] = float(value)
    return refs


def metrics_for(ranking, rels):
    """Independent binary form; all official positive grades are 2."""
    relevant = {pid for pid, score in rels.items() if score > 0}
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
        value = dcg / ideal
        assert abs(value - ndcg_at_k(ranking, rels, k)) <= 1e-12
        values[f"NDCG@{k}"] = value

    for k in (5, 10, 100):
        value = len(set(ranking[:k]) & relevant) / len(relevant)
        assert abs(value - recall_at_k(ranking, rels, k)) <= 1e-12
        values[f"Recall@{k}"] = value

    assert all(0 <= value <= 1 for value in values.values())
    return values


save()
sampler = threading.Thread(target=sample_ram, daemon=True)
sampler.start()
print("Record:", record, flush=True)

try:
    report["stage"] = "input_acceptance"
    save()

    accepted = read_json(SOURCE / "report.json")
    assert accepted["status"] == "PASS"
    assert accepted["cache_complete"] is True
    assert accepted["counts"] == {
        "pages": 452, "queries": 232, "qrels": 3628
    }
    report["source_report_sha256"] = digest(SOURCE / "report.json")

    profile = read_json(CACHE / "profile.json")
    assert profile["dataset_revision"] == (
        "76fe40166ba07b1bf50457f5c6057cacdd045f10"
    )
    assert profile["dtype"] == "float16"
    assert profile["crop"] is False
    assert digest(DATA / "download-manifest.json") == (
        profile["dataset_download_manifest_sha256"]
    )
    for relative, expected in profile["encoding_code_sha256"].items():
        path = ROOT / Path(relative.replace("\\", "/"))
        assert digest(path) == expected, f"Encoding source changed: {path}"
    for name, expected in profile["versions"].items():
        assert version(name) == expected, f"Version changed: {name}"

    store = CacheStore(CACHE, profile, create=False)
    manifest = store.manifest()
    assert manifest["complete"]
    assert not manifest["pending_directories"]
    assert manifest == read_json(SOURCE / "cache-manifest.json")
    write_json("cache-manifest.json", manifest)

    page_rows = pq.read_table(
        DATA / "corpus/test-00000-of-00001.parquet",
        columns=["corpus-id", "doc-id"],
    ).to_pylist()
    query_rows = pq.read_table(
        DATA / "queries/test-00000-of-00001.parquet",
        columns=["query-id", "query", "language"],
    ).to_pylist()

    pages = {str(row["corpus-id"]): row for row in page_rows}
    official_queries = {
        str(row["query-id"]): row for row in query_rows
    }
    assert len(page_rows) == len(pages) == 452
    assert len(query_rows) == len(official_queries) == 232
    assert set(pages) == set(profile["expected_page_ids"])
    assert set(official_queries) == set(profile["expected_query_ids"])
    assert {
        x["id"] for x in manifest["completed"]["pages"]
    } == set(pages)
    assert {
        x["id"] for x in manifest["completed"]["queries"]
    } == set(official_queries)

    points = []
    for pid in sorted(pages, key=int):
        arrays, meta = store.read("pages", pid)
        assert meta["identity"]["doc_id"] == pages[pid]["doc-id"]
        points.append({
            "id": _stable_uuid(pid),
            "payload": {
                "corpus_id": pid,
                "doc_id": pages[pid]["doc-id"],
                "diagnostic": False,
            },
            "vectors": {"initial": arrays["vectors"]},
        })

    expected_points = {str(p["id"]): p for p in points}
    assert len(expected_points) == 452

    queries = {}
    for qid in sorted(official_queries, key=int):
        arrays, meta = store.read("queries", qid)
        row = official_queries[qid]
        assert meta["identity"]["text"] == row["query"]
        assert meta["identity"]["language"] == row["language"]
        queries[qid] = arrays["vectors"]

    raw_qrels = pq.read_table(
        DATA / "qrels/test-00000-of-00001.parquet"
    ).to_pylist()
    qrels = _normalize_qrels(raw_qrels)
    independent_qrels = {}
    for row in raw_qrels:
        assert int(row["score"]) == 2
        independent_qrels.setdefault(str(row["query-id"]), {})[
            _stable_uuid(str(row["corpus-id"]))
        ] = 2

    assert qrels == independent_qrels
    assert len(raw_qrels) == sum(map(len, qrels.values())) == 3628
    assert set(qrels) == set(queries)
    assert all(set(rels) <= set(expected_points) for rels in qrels.values())
    assert qrels["0"][_stable_uuid("63")] == 2

    write_json("qrels.json", qrels)
    write_json("page-id-mapping.json", [
        {
            "point_id": p["id"],
            "corpus_id": p["payload"]["corpus_id"],
            "doc_id": p["payload"]["doc_id"],
        }
        for p in points
    ])
    report["counts"] = {"pages": 452, "queries": 232, "qrels": 3628}
    report["cache_profile_sha256"] = store.profile_hash

    sources = [
        Path(__file__),
        ROOT / "scripts/prb_cache.py",
        ROOT / "scripts/prb_index.py",
        ROOT / "scripts/prb_resume_a.py",
        ROOT / "visual-rag-toolkit/visual_rag/retrieval/multi_vector.py",
        ROOT / "visual-rag-toolkit/visual_rag/retrieval/single_stage.py",
        ROOT / "visual-rag-toolkit/benchmarks/vidore_tatdqa_test/metrics.py",
        ROOT / "visual-rag-toolkit/benchmarks/vidore_tatdqa_test/dataset_loader.py",
    ]
    report["code_sha256"] = {
        str(p.relative_to(ROOT)): digest(p) for p in sources
    }
    report["versions"] = {
        name: version(name)
        for name in ("numpy", "qdrant-client", "pydantic", "pyarrow")
    }
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

    print("Inputs: 452 pages / 232 queries / 3628 qrels PASS", flush=True)

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
    if resume_report is not None:
        assert resume_report["cache_profile_sha256"] == store.profile_hash
        assert resume_report["counts"] == report["counts"]
        assert resume_report["source_report_sha256"] == (
            report["source_report_sha256"]
        )
        assert resume_report["versions"] == report["versions"]
        for filename in ("qrels.json", "page-id-mapping.json"):
            assert digest(resume_source / filename) == digest(record / filename)
        for relative, expected in resume_report["code_sha256"].items():
            normalized = relative.replace("\\", "/")
            if normalized != "scripts/p2_evaluate_a.py":
                assert digest(ROOT / normalized) == expected, (
                    "Source changed since failed run", relative
                )

        report["resume_source_record"] = str(resume_source)
        report["resume_source_report_sha256"] = digest(
            resume_source / "report.json"
        )
        report["connection_trust_env"] = False
        report["index_preparation_note"] = (
            "This run verifies existing points and uploads missing points. "
            "Earlier upload work is recorded in the original failed run."
        )
        save()
        info, recovery = resume_collection_a(
            checked, report["collection"], points, record
        )
        report["resume"] = recovery
    else:
        info = build_collection(
            checked, report["collection"], "A", points
        )
    report["index_prepare_and_readback_seconds"] = time.perf_counter() - t
    (record / "collection-before.json").write_bytes(
        info.model_dump_json(indent=2).encode("utf-8")
    )
    assert (
        checked.uploads
        + report.get("resume", {}).get("reused_points", 0)
    ) == 452
    report["index_id_set_matches_official"] = True
    report["actual_stored_fields"] = list(info.config.params.vectors)
    report["theoretical_vector_bytes"] = 452 * 1024 * 128 * 2
    print("A stored fields: ['initial'] PASS", flush=True)
    print("A full ID/config/payload readback: PASS", flush=True)

    retriever = MultiVectorRetriever(
        collection_name=report["collection"],
        qdrant_client=checked,
        embedder=NoEncoding(),
        request_timeout=300,
        max_retries=1,
    )

    report["stage"] = "full_corpus_diagnostics"
    save()
    diagnostics = []
    for qid in ("0", "20"):
        hits, seconds = search(qid)
        ranking = checked_hits(hits)
        refs = full_reference(queries[qid])
        tokens = len(queries[qid])
        bound = report["score_tolerance_per_query_token"] * tokens
        errors = [
            abs(float(hit["score_final"]) - refs[str(hit["id"])]) / tokens
            for hit in hits
        ]
        assert max(errors) <= report["score_tolerance_per_query_token"]

        # Reject ranking inversions outside the predetermined tie allowance.
        lowest_previous = float("inf")
        for pid in ranking:
            assert lowest_previous + 2 * bound >= refs[pid]
            lowest_previous = min(lowest_previous, refs[pid])

        excluded = set(refs) - set(ranking)
        boundary_gap = (
            max(refs[pid] for pid in excluded)
            - min(refs[pid] for pid in ranking)
        )
        assert boundary_gap <= 2 * bound, "Top100 membership mismatch"

        diagnostics.append({
            "query_id": qid,
            "query_tokens": tokens,
            "max_score_error_per_token": max(errors),
            "top100_boundary_gap": boundary_gap,
            "allowed_rank_tie_score_difference": 2 * bound,
            "search_call_seconds": seconds,
            "hits": hits,
            "all_452_fp32_reference_scores": refs,
        })
        write_json("diagnostics.json", diagnostics)
        print(f"Full-corpus query {qid} scoring/top100: PASS", flush=True)

    report["stage"] = "all_query_evaluation"
    save()
    per_query = []
    metric_names = [
        "NDCG@5", "NDCG@10",
        "Recall@5", "Recall@10", "Recall@100",
    ]

    with (record / "rankings.jsonl").open(
        "x", encoding="utf-8", newline="\n"
    ) as output:
        for number, qid in enumerate(sorted(queries, key=int), 1):
            report["current_query"] = qid
            hits, seconds = search(qid)
            ranking = checked_hits(hits)
            values = metrics_for(ranking, qrels[qid])
            per_query.append({
                "query_id": qid,
                "relevant_pages": len(qrels[qid]),
                "query_tokens": len(queries[qid]),
                "search_call_seconds": seconds,
                **values,
            })

            # Keep database ranking order, including ties.
            # Do not remove candidates whose numeric ID equals the query ID.
            saved_hits = [
                {
                    "rank": rank,
                    "point_id": str(hit["id"]),
                    "corpus_id": hit["payload"]["corpus_id"],
                    "doc_id": hit["payload"]["doc_id"],
                    "score": float(hit["score_final"]),
                }
                for rank, hit in enumerate(hits, 1)
            ]
            output.write(json.dumps({
                "query_id": qid,
                "hits": saved_hits,
            }, ensure_ascii=False) + "\n")
            output.flush()

            report["completed_queries"] = number
            if number == 1 or number % 20 == 0 or number == 232:
                save()
                print(f"Evaluated: {number}/232", flush=True)

    assert len(per_query) == 232
    assert {x["query_id"] for x in per_query} == set(qrels)
    assert checked.search_calls == 234  # 2 diagnostics + 232 formal queries

    with (record / "per-query.tsv").open(
        "x", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(
            output, fieldnames=list(per_query[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(per_query)

    aggregate = {
        name: math.fsum(row[name] for row in per_query) / 232
        for name in metric_names
    }
    write_json("metrics.json", {
        "group": "A",
        "query_count": 232,
        "aggregation": "Macro mean over every official query",
        "values": aggregate,
    })

    final_info = client.get_collection(report["collection"])
    assert final_info.config == info.config
    assert final_info.indexed_vectors_count == 0
    assert client.count(report["collection"], exact=True).count == 452
    (record / "collection-after.json").write_bytes(
        final_info.model_dump_json(indent=2).encode("utf-8")
    )

    for relative, expected in report["code_sha256"].items():
        assert digest(ROOT / relative) == expected, "Source changed during run"

    report["aggregate_metrics"] = aggregate
    report["evaluation_search_call_seconds_total"] = math.fsum(
        row["search_call_seconds"] for row in per_query
    )
    report["actual_query_calls"] = checked.search_calls
    report["model_encoding_calls"] = 0
    report["artifact_sha256"] = {
        p.name: digest(p)
        for p in record.iterdir()
        if p.is_file() and p.name != "report.json"
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
print("Evaluation queries:", report["completed_queries"])
print("Model encoding calls:", report["model_encoding_calls"])
print("Sampled Python peak RSS GiB:", ram["peak_rss_bytes"] / 1024**3)
print("Record:", record)
print("P2 formal A evaluation: PASS")