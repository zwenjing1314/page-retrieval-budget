import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/encoded-economics-v1"
SOURCES = {
    "A": ROOT / "records/p2-formal-a-20261107-012250-612a76",
    "B": ROOT / "records/p2-formal-b-20261107-013537-34e55b",
    "C": ROOT / "records/p2-formal-c-20261107-014210-68a2ce",
}
FIELDS = {
    "A": {"initial"},
    "B": {"initial", "experimental_pooling_3"},
    "C": {"experimental_pooling_3"},
}

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import csv
import json
import math
import platform
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from importlib.metadata import version

import numpy as np
import psutil
from qdrant_client import QdrantClient

from prb_cache import CacheStore, digest, json_bytes
from visual_rag.retrieval.multi_vector import MultiVectorRetriever


record = ROOT / "records" / (
    "p3-search-timing-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    + "-" + uuid.uuid4().hex[:6]
)
record.mkdir(parents=True, exist_ok=False)

report = {
    "status": "STARTED",
    "concurrency": 1,
    "queries_per_block": 232,
    "warmup": "One complete 232-query pass per group, order A/B/C",
    "measured_orders": [
        ["A", "B", "C"],
        ["B", "C", "A"],
        ["C", "A", "B"],
    ],
    "timing_boundary": (
        "Cached query array through author search_embedded return. "
        "Includes query conversion, HTTP serialization, localhost transport, "
        "Qdrant search and response parsing. Excludes query encoding, "
        "candidate replay, result validation, artifact writing and metrics."
    ),
    "block_wall_boundary": (
        "Continuous 232-query loop, including per-call timers and in-memory "
        "result accumulation; validation and file writing follow the loop."
    ),
    "qps_definition": "232 / block_wall_seconds",
    "latency_definition": "Per search_embedded call; percentile method=linear",
    "resource_note": (
        "Python RSS sampled every 0.1 seconds after imports; lower bound. "
        "Docker stats are before/after snapshots, not peak memory."
    ),
    "effect_metrics_recomputed": False,
    "model_encoding_calls": 0,
    "completed_measured_blocks": 0,
}
started = time.perf_counter()
stop = threading.Event()
process = psutil.Process()
ram = {"peak_rss_bytes": process.memory_info().rss, "samples": 0}
client = None


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


def command(args):
    result = subprocess.run(
        args,
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    return result.stdout.strip()


def docker_snapshot(label):
    state = json.loads(command([
        "docker", "inspect", "--format", "{{json .State}}",
        "prb-qdrant-1141",
    ]))
    assert state["Running"] is True
    assert state["OOMKilled"] is False
    result = {
        "state": state,
        "stats": command([
            "docker", "stats", "--no-stream",
            "--format", "{{json .}}", "prb-qdrant-1141",
        ]),
    }
    write_json("docker-" + label + ".json", result)
    return result


class NoEncoding:
    def embed_query(self, *args, **kwargs):
        raise AssertionError("Timing must use cached queries")


class CheckedClient:
    def __init__(self, actual, group):
        self.actual = actual
        self.group = group
        self.calls = 0
        self.validate = True

    def __getattr__(self, name):
        return getattr(self.actual, name)

    def query_points(self, *args, **kwargs):
        # Validate actual request parameters during the complete warmup.
        # Measured rounds retain only dispatch and a call counter.
        if self.validate:
            assert kwargs["collection_name"] == collections[self.group]
            assert kwargs["limit"] == 100
            assert kwargs.get("query_filter") is None
            assert kwargs["search_params"].exact is True
            assert kwargs["with_payload"] is True
            assert kwargs.get("with_vectors", False) is False

            if self.group == "B":
                assert kwargs["using"] == "initial"
                prefetch = kwargs["prefetch"]
                assert isinstance(prefetch, list) and len(prefetch) == 1
                assert prefetch[0].using == "experimental_pooling_3"
                assert prefetch[0].limit == 256
                assert prefetch[0].params.exact is True
                assert prefetch[0].filter is None
                assert np.array_equal(
                    np.asarray(prefetch[0].query, dtype=np.float32),
                    np.asarray(kwargs["query"], dtype=np.float32),
                )
            else:
                expected_field = (
                    "initial" if self.group == "A"
                    else "experimental_pooling_3"
                )
                assert kwargs["using"] == expected_field
                assert kwargs.get("prefetch") is None

        self.calls += 1
        return self.actual.query_points(*args, **kwargs)


def collection_audit(group, label):
    name = collections[group]
    info = client.get_collection(name)
    actual = json.loads(info.model_dump_json())
    baseline = load(SOURCES[group] / "collection-after.json")
    assert actual["config"] == baseline["config"], (
        "Collection configuration changed", group
    )
    assert set(info.config.params.vectors) == FIELDS[group]
    assert info.status.value == "green", (group, info.status)
    assert info.indexed_vectors_count == 0
    assert client.count(name, exact=True).count == 452

    seen = set()
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=name,
            offset=offset,
            limit=128,
            with_payload=True,
            with_vectors=False,
        )
        for point in batch:
            pid = str(point.id)
            assert pid in expected_payloads and pid not in seen
            seen.add(pid)
            assert point.payload == expected_payloads[pid]
        if offset is None:
            break
    assert seen == set(expected_payloads)
    write_json(f"{group}-collection-{label}.json", actual)


def search(group, query):
    retriever = retrievers[group]
    if group == "A":
        return retriever.search_embedded(
            query_embedding=query, top_k=100, mode="single_full"
        )
    if group == "B":
        return retriever.search_embedded(
            query_embedding=query,
            top_k=100,
            mode="two_stage",
            prefetch_k=256,
            stage1_mode="tokens_vs_experimental_pooling",
        )
    return retriever.search_embedded(
        query_embedding=query,
        top_k=100,
        mode="single_experimental_tokens",
    )


def run_block(group, phase, round_number):
    report["current"] = {
        "phase": phase, "round": round_number, "group": group
    }
    save()
    print(
        f"Starting {phase} round={round_number} group={group}",
        flush=True,
    )

    output = []
    block_started = time.perf_counter()
    for qid in query_order:
        query = queries[qid]
        t = time.perf_counter()
        hits = search(group, query)
        elapsed = time.perf_counter() - t
        output.append((qid, elapsed, hits))
    block_wall = time.perf_counter() - block_started

    # All validation and writing below are outside the timed block.
    durations = []
    changed_rankings = 0
    maximum_common_score_error = 0.0
    for qid, elapsed, hits in output:
        assert elapsed > 0 and math.isfinite(elapsed)
        durations.append(elapsed)
        ids = [str(hit["id"]) for hit in hits]
        assert len(hits) == len(set(ids)) == 100
        assert set(ids) <= set(expected_payloads)
        scores = {
            str(hit["id"]): float(hit["score_final"]) for hit in hits
        }
        assert all(math.isfinite(value) for value in scores.values())
        assert all(scores[a] >= scores[b] for a, b in zip(ids, ids[1:]))
        for hit in hits:
            assert hit["payload"] == expected_payloads[str(hit["id"])]

        old_hits = baseline_rankings[group][qid]
        old_ids = [hit["point_id"] for hit in old_hits]
        old_scores = {
            hit["point_id"]: float(hit["score"]) for hit in old_hits
        }
        common = set(scores) & set(old_scores)
        error = max(
            (
                abs(scores[pid] - old_scores[pid]) / len(queries[qid])
                for pid in common
            ),
            default=0.0,
        )
        assert error <= 1e-3, (
            "Shared-point score changed", group, qid, error
        )
        maximum_common_score_error = max(
            maximum_common_score_error, error
        )
        changed_rankings += ids != old_ids

        append_json("query-times.jsonl", {
            "phase": phase,
            "round": round_number,
            "group": group,
            "query_id": qid,
            "seconds": elapsed,
        })
        if phase == "measured":
            append_json("timing-rankings.jsonl", {
                "round": round_number,
                "group": group,
                "query_id": qid,
                "hits": [
                    {
                        "rank": rank,
                        "point_id": pid,
                        "score": scores[pid],
                    }
                    for rank, pid in enumerate(ids, 1)
                ],
            })

    assert len(durations) == 232
    summary = {
        "phase": phase,
        "round": round_number,
        "group": group,
        "queries": 232,
        "block_wall_seconds": block_wall,
        "search_call_seconds_sum": math.fsum(durations),
        "qps": 232 / block_wall,
        "p50_ms": float(np.percentile(
            durations, 50, method="linear"
        ) * 1000),
        "p95_ms": float(np.percentile(
            durations, 95, method="linear"
        ) * 1000),
        "rankings_changed_vs_p2": changed_rankings,
        "max_common_score_error_per_token": maximum_common_score_error,
    }
    append_json("blocks.jsonl", summary)
    print(
        f"{phase} round={round_number} {group}: "
        f"{block_wall:.3f}s; QPS={summary['qps']:.3f}; "
        f"p50={summary['p50_ms']:.3f}ms; "
        f"p95={summary['p95_ms']:.3f}ms",
        flush=True,
    )
    return summary, durations


save()
sampler = threading.Thread(target=sample_ram, daemon=True)
sampler.start()
print("Record:", record, flush=True)

try:
    assert Path(sys.prefix).name.lower() == "page-retrieval-budget"
    reports = {}
    collections = {}
    baseline_rankings = {}
    profile = load(CACHE / "profile.json")
    store = CacheStore(CACHE, profile, create=False)

    for group, source in SOURCES.items():
        previous = load(source / "report.json")
        assert previous["status"] == "PASS"
        assert previous["group"] == group
        assert previous["completed_queries"] == 232
        assert previous["counts"] == {
            "pages": 452, "queries": 232, "qrels": 3628
        }
        assert previous["cache_profile_sha256"] == store.profile_hash
        for name, expected_hash in previous["artifact_sha256"].items():
            assert digest(source / name) == expected_hash, (group, name)
        for relative, expected_hash in previous["code_sha256"].items():
            assert digest(ROOT / relative.replace("\\", "/")) == expected_hash
        for name, expected_version in previous["versions"].items():
            assert version(name) == expected_version

        with (source / "rankings.jsonl").open(encoding="utf-8") as saved:
            rank_rows = [
                json.loads(line) for line in saved if line.strip()
            ]
        baseline_rankings[group] = {
            row["query_id"]: row["hits"] for row in rank_rows
        }
        assert len(rank_rows) == len(baseline_rankings[group]) == 232
        reports[group] = previous
        collections[group] = previous["collection"]

    assert reports["C"]["points_with_initial_field"] == 0
    for name, expected_version in profile["versions"].items():
        assert version(name) == expected_version

    manifest = store.manifest()
    assert manifest["complete"] and not manifest["pending_directories"]
    for source in SOURCES.values():
        assert manifest == load(source / "cache-manifest.json")

    query_order = sorted(profile["expected_query_ids"], key=int)
    assert len(query_order) == len(set(query_order)) == 232
    queries = {
        qid: store.read("queries", qid)[0]["vectors"]
        for qid in query_order
    }
    for group in SOURCES:
        assert set(baseline_rankings[group]) == set(query_order)

    mapping = load(SOURCES["A"] / "page-id-mapping.json")
    for source in SOURCES.values():
        assert mapping == load(source / "page-id-mapping.json")
    expected_payloads = {
        item["point_id"]: {
            "corpus_id": item["corpus_id"],
            "doc_id": item["doc_id"],
            "diagnostic": False,
        }
        for item in mapping
    }
    assert len(expected_payloads) == 452

    report["collections"] = collections
    report["cache_profile_sha256"] = store.profile_hash
    report["source_records"] = {
        group: {
            "path": str(source),
            "report_sha256": digest(source / "report.json"),
        }
        for group, source in SOURCES.items()
    }
    report["hardware"] = {
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "host_ram_bytes": psutil.virtual_memory().total,
        "python": sys.version,
        "python_executable": sys.executable,
        "docker_runtime": command([
            "docker", "info", "--format",
            "{{.NCPU}} CPUs; {{.MemTotal}} bytes; {{.KernelVersion}}",
        ]),
        "gpu_and_driver": command([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]),
    }
    code_paths = [
        Path(__file__),
        ROOT / "scripts/prb_cache.py",
        ROOT / "visual-rag-toolkit/visual_rag/retrieval/multi_vector.py",
        ROOT / "visual-rag-toolkit/visual_rag/retrieval/single_stage.py",
        ROOT / "visual-rag-toolkit/visual_rag/retrieval/two_stage.py",
    ]
    report["code_sha256"] = {
        str(path.relative_to(ROOT)): digest(path)
        for path in code_paths
    }
    report["versions"] = {
        name: version(name) for name in reports["A"]["versions"]
    }
    report["outer_repository_commit"] = command([
        "git", "-C", str(ROOT.parent), "rev-parse", "HEAD"
    ])
    write_json("query-order.json", query_order)

    client = QdrantClient(
        url="http://127.0.0.1:16333",
        timeout=300,
        trust_env=False,
    )
    assert client.info().version == "1.14.1"
    report["qdrant_server_version"] = client.info().version

    for group in ("A", "B", "C"):
        collection_audit(group, "before")
    before = docker_snapshot("before")

    wrappers = {
        group: CheckedClient(client, group) for group in SOURCES
    }
    retrievers = {
        group: MultiVectorRetriever(
            collection_name=collections[group],
            qdrant_client=wrappers[group],
            embedder=NoEncoding(),
            experimental_vector_name="experimental_pooling_3",
            request_timeout=300,
            max_retries=1,
        )
        for group in SOURCES
    }
    print("Frozen A/B/C inputs, live IDs and configurations: PASS", flush=True)

    warmup_summaries = []
    for group in ("A", "B", "C"):
        summary, _ = run_block(group, "warmup", 0)
        warmup_summaries.append(summary)
        assert wrappers[group].calls == 232
        wrappers[group].validate = False
    write_json("warmup-summary.json", warmup_summaries)
    print("Complete-query warmup: PASS", flush=True)

    measured = []
    all_durations = {group: [] for group in SOURCES}
    for round_number, order in enumerate(report["measured_orders"], 1):
        for group in order:
            summary, durations = run_block(
                group, "measured", round_number
            )
            measured.append(summary)
            all_durations[group].extend(durations)
            report["completed_measured_blocks"] = len(measured)
            save()

    assert len(measured) == 9
    for group in SOURCES:
        assert wrappers[group].calls == 232 * 4
        assert len(all_durations[group]) == 232 * 3

    with (record / "round-summary.tsv").open(
        "x", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(
            output, fieldnames=list(measured[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(measured)

    combined = {}
    for group in ("A", "B", "C"):
        blocks = [row for row in measured if row["group"] == group]
        seconds = math.fsum(row["block_wall_seconds"] for row in blocks)
        durations = all_durations[group]
        combined[group] = {
            "rounds": 3,
            "queries": 696,
            "total_block_wall_seconds": seconds,
            "qps": 696 / seconds,
            "p50_ms": float(np.percentile(
                durations, 50, method="linear"
            ) * 1000),
            "p95_ms": float(np.percentile(
                durations, 95, method="linear"
            ) * 1000),
            "rankings_changed_vs_p2_across_rounds": sum(
                row["rankings_changed_vs_p2"] for row in blocks
            ),
        }

    after = docker_snapshot("after")
    assert before["state"]["StartedAt"] == after["state"]["StartedAt"], (
        "Container restarted during timing"
    )
    for group in ("A", "B", "C"):
        collection_audit(group, "after")
    for relative, expected_hash in report["code_sha256"].items():
        assert digest(ROOT / relative) == expected_hash
    for group, source in SOURCES.items():
        assert digest(source / "report.json") == (
            report["source_records"][group]["report_sha256"]
        )

    comparison = {
        "combined_three_rounds": combined,
        "B_over_A_qps_ratio": combined["B"]["qps"] / combined["A"]["qps"],
        "C_over_A_qps_ratio": combined["C"]["qps"] / combined["A"]["qps"],
        "note": (
            "All three measured rounds included; no best-round selection. "
            "Warmup excluded. Cached-query search, not query-encoding E2E."
        ),
    }
    write_json("timing-summary.json", comparison)
    report["timing_summary"] = comparison
    report["query_calls_including_warmup"] = {
        group: wrappers[group].calls for group in SOURCES
    }
    report["artifact_sha256"] = {
        path.name: digest(path)
        for path in record.iterdir()
        if path.is_file() and path.name != "report.json"
    }
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

print("Combined three measured rounds:")
for group, values in report["timing_summary"]["combined_three_rounds"].items():
    print(group, values)
print("B/A QPS ratio:", report["timing_summary"]["B_over_A_qps_ratio"])
print("C/A QPS ratio:", report["timing_summary"]["C_over_A_qps_ratio"])
print("Measured blocks:", report["completed_measured_blocks"])
print("Model encoding calls:", report["model_encoding_calls"])
print("Record:", record)
print("P3 cached-search timing: PASS")