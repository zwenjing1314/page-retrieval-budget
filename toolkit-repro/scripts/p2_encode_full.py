import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/encoded-economics-v1"
DATA = ROOT / "data/economics_reports_v2-76fe40166ba07b1bf50457f5c6057cacdd045f10"
ADAPTER = ROOT / "models/colpali-v1.3-b5c6dd62125326e6f0b540c1f7b36e901cdc0a11"
BASE = ROOT / "models/colpaligemma-3b-pt-448-base-30ab955d073de4a91dc5a288e8c97226647e3e5a"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PRB_COLPALI_BASE"] = str(BASE)
os.environ["PRB_ECONOMICS_SNAPSHOT"] = str(DATA)
os.environ["VISUALRAG_SORT_QUERIES_BY_LENGTH"] = "0"

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime
from importlib.metadata import version

import numpy as np
import psutil
import pyarrow.parquet as pq
from PIL import Image

from prb_cache import CacheStore, digest, json_bytes

parser = argparse.ArgumentParser()
parser.add_argument("--verify-only", action="store_true")
args = parser.parse_args()

record = ROOT / "records" / (
    "p2-full-cache-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    + "-" + uuid.uuid4().hex[:6]
)
record.mkdir(parents=True, exist_ok=False)
report = {
    "status": "STARTED",
    "verify_only": args.verify_only,
    "model_loaded": False,
    "new_pages": 0,
    "new_queries": 0,
    "timing_totals_seconds": {},
    "resource_note": (
        "RSS sampled every 0.1 seconds; sampled peak is a lower bound. "
        "CUDA allocated/reserved peaks describe this Python process. "
        "This run supplements existing caches; it is not all-data total time."
    ),
}
started = time.perf_counter()
stop = threading.Event()
process = psutil.Process()
ram = {"peak_rss_bytes": process.memory_info().rss, "samples": 0}
torch = None
hook = None


def save():
    (record / "report.json").write_bytes(json_bytes(report))


def sample_ram():
    while not stop.is_set():
        ram["peak_rss_bytes"] = max(
            ram["peak_rss_bytes"], process.memory_info().rss
        )
        ram["samples"] += 1
        stop.wait(0.1)


def event(value):
    with (record / "items.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")


def finish_item(kind, item_id, arrays, identity, timings):
    t = time.perf_counter()
    store.put(kind, item_id, arrays, identity)
    timings["cache_write_and_readback"] = time.perf_counter() - t
    for key, value in timings.items():
        report["timing_totals_seconds"][key] = (
            report["timing_totals_seconds"].get(key, 0) + value
        )
    report["new_pages" if kind == "pages" else "new_queries"] += 1
    event({
        "status": "PASS",
        "kind": kind,
        "id": item_id,
        "timings_seconds": timings,
        "vector_shape": list(arrays["vectors"].shape),
    })
    save()


save()
sampler = threading.Thread(target=sample_ram, daemon=True)
sampler.start()
print("Record:", record, flush=True)

try:
    profile = json.loads(
        (CACHE / "profile.json").read_text(encoding="utf-8")
    )
    assert profile["dataset_revision"] == (
        "76fe40166ba07b1bf50457f5c6057cacdd045f10"
    )
    assert profile["model"]["adapter_revision"] == (
        "b5c6dd62125326e6f0b540c1f7b36e901cdc0a11"
    )
    assert profile["model"]["base_revision"] == (
        "30ab955d073de4a91dc5a288e8c97226647e3e5a"
    )
    assert profile["dtype"] == "float16"
    assert profile["attention"] == "sdpa" and profile["crop"] is False
    assert digest(DATA / "download-manifest.json") == (
        profile["dataset_download_manifest_sha256"]
    )
    for relative, expected in profile["encoding_code_sha256"].items():
        path = ROOT / Path(relative.replace("\\", "/"))
        assert digest(path) == expected, f"Encoding source changed: {path}"
    for name, expected in profile["versions"].items():
        assert version(name) == expected, f"Version changed: {name}"

    page_rows = pq.read_table(
        DATA / "corpus/test-00000-of-00001.parquet",
        columns=["corpus-id", "doc-id"],
    ).to_pylist()
    query_rows = pq.read_table(
        DATA / "queries/test-00000-of-00001.parquet",
        columns=["query-id", "query", "language"],
    ).to_pylist()
    pages = {str(row["corpus-id"]): row for row in page_rows}
    queries = {str(row["query-id"]): row for row in query_rows}
    positions = {str(row["corpus-id"]): i for i, row in enumerate(page_rows)}
    assert len(page_rows) == len(pages) == 452
    assert len(query_rows) == len(queries) == 232
    assert set(profile["expected_page_ids"]) == set(pages)
    assert set(profile["expected_query_ids"]) == set(queries)

    store = CacheStore(CACHE, profile, create=False)
    before = store.manifest()
    assert not before["pending_directories"], (
        "Pending writes found; preserve them and report before continuing"
    )
    (record / "cache-before.json").write_bytes(json_bytes(before))
    completed_pages = {x["id"] for x in before["completed"]["pages"]}
    completed_queries = {x["id"] for x in before["completed"]["queries"]}

    # Verify the official identities of reused entries.
    for kind, completed, official, identity_key in [
        ("pages", completed_pages, pages, "doc_id"),
        ("queries", completed_queries, queries, "text"),
    ]:
        for item_id in completed:
            metadata = json.loads(
                (store.path(kind, item_id) / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )["identity"]
            value = (
                official[item_id]["doc-id"]
                if kind == "pages" else official[item_id]["query"]
            )
            assert metadata[identity_key] == value

    missing_pages = [
        x for x in profile["expected_page_ids"] if x not in completed_pages
    ]
    missing_queries = [
        x for x in profile["expected_query_ids"] if x not in completed_queries
    ]
    report["reused_counts"] = {
        "pages": len(completed_pages), "queries": len(completed_queries)
    }
    report["requested_new_counts"] = {
        "pages": len(missing_pages), "queries": len(missing_queries)
    }
    report["available_ram_before_bytes"] = psutil.virtual_memory().available
    report["disk_free_before_bytes"] = shutil.disk_usage(ROOT).free
    save()
    print("Reused:", report["reused_counts"], flush=True)
    print("Missing:", report["requested_new_counts"], flush=True)

    if args.verify_only:
        assert not missing_pages and not missing_queries, "Cache is incomplete"

    if missing_pages or missing_queries:
        import torch
        from visual_rag.embedding.visual_embedder import VisualEmbedder
        from visual_rag.embedding.pooling import (
            colpali_row_mean_pooling,
            colpali_experimental_pooling_from_rows,
        )

        assert torch.cuda.is_available()
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        embedder = VisualEmbedder(
            model_name=str(ADAPTER),
            device="cuda:0",
            torch_dtype=torch.float16,
            output_dtype=np.float16,
            batch_size=1,
            processor_speed="slow",
            filter_special_tokens=True,
        )
        assert embedder.filter_special_tokens
        model = embedder.model
        processor = embedder.processor
        torch.cuda.synchronize()
        report["model_load_seconds"] = time.perf_counter() - t
        report["model_loaded"] = True
        report["gpu"] = torch.cuda.get_device_name(0)

        assert not model.training
        assert not any(p.requires_grad for p in model.parameters())
        assert model.active_adapters() == ["default"]
        assert all(p.device.type == "cuda" for p in model.parameters())
        assert all(
            p.dtype == torch.float16
            for p in model.parameters() if p.is_floating_point()
        )
        assert processor.tokenizer.is_fast
        assert processor.image_token_id == profile["image_token_id"]
        assert processor.image_processor.to_dict() == (
            profile["image_processor"]["image_config"]
        )

        # Capture actual model inputs, avoiding a second preprocessing pass.
        captured = {}

        def capture_inputs(module, positional, kwargs):
            captured["calls"] = captured.get("calls", 0) + 1
            assert kwargs["input_ids"].shape[0] == 1
            captured["input_ids"] = (
                kwargs["input_ids"][0].detach().cpu().numpy().copy()
            )
            captured["attention_mask"] = (
                kwargs["attention_mask"][0].detach().cpu().numpy().copy()
            )

        hook = model.register_forward_pre_hook(capture_inputs, with_kwargs=True)

        if missing_pages:
            table = pq.read_table(
                DATA / "corpus/test-00000-of-00001.parquet"
            )

        for number, pid in enumerate(missing_pages, 1):
            report["current"] = {"stage": "page_encoding", "id": pid}
            save()
            timings = {}
            t = time.perf_counter()
            row = table.slice(positions[pid], 1).to_pylist()[0]
            assert str(row["corpus-id"]) == pid
            with Image.open(io.BytesIO(row["image"]["bytes"])) as original:
                image = original.convert("RGB")
                image.load()
            timings["image_decode"] = time.perf_counter() - t

            captured.clear()
            t = time.perf_counter()
            with torch.inference_mode():
                full, info = embedder.embed_images(
                    [image], batch_size=1,
                    return_token_info=True, show_progress=False,
                )
            torch.cuda.synchronize()
            timings["page_embed_images"] = time.perf_counter() - t
            assert captured["calls"] == 1, "Unexpected retry or extra forward"
            vectors = embedder.extract_visual_embedding(full[0], info[0])
            ids = captured["input_ids"]
            attention = captured["attention_mask"]
            keep = ids == processor.image_token_id
            assert np.array_equal(
                np.flatnonzero(keep), info[0]["visual_token_indices"]
            )

            t = time.perf_counter()
            rows = colpali_row_mean_pooling(
                vectors, output_dtype=np.float16
            )
            pooled = colpali_experimental_pooling_from_rows(
                rows, window_size=3, output_dtype=np.float16
            )
            timings["page_pooling"] = time.perf_counter() - t
            finish_item("pages", pid, {
                "vectors": vectors,
                "raw": full[0].numpy(),
                "input_ids": ids,
                "attention_mask": attention,
                "keep_mask": keep,
                "rows": rows,
                "pooled": pooled,
            }, {
                "corpus_id": pid,
                "doc_id": row["doc-id"],
                "image_size": list(image.size),
                "image_bytes_sha256": hashlib.sha256(
                    row["image"]["bytes"]
                ).hexdigest(),
                "source_record": str(record),
                "source": "Toolkit batch=1; actual forward inputs captured",
            }, timings)

            if number == 1 or number % 20 == 0 or number == len(missing_pages):
                print(
                    f"New pages: {number}/{len(missing_pages)}; "
                    f"total cached: {len(completed_pages) + number}/452",
                    flush=True,
                )

        for number, qid in enumerate(missing_queries, 1):
            report["current"] = {"stage": "query_encoding", "id": qid}
            save()
            row = queries[qid]
            captured.clear()
            t = time.perf_counter()
            with torch.inference_mode():
                vectors = (
                    embedder.embed_query(row["query"])
                    .detach().cpu().numpy().copy()
                )
            torch.cuda.synchronize()
            timings = {"query_embed_query": time.perf_counter() - t}
            assert captured["calls"] == 1
            ids = captured["input_ids"]
            attention = captured["attention_mask"]
            keep = attention.astype(bool)
            if (ids[keep] >= 4).any():
                keep &= ids >= 4

            finish_item("queries", qid, {
                "vectors": vectors,
                "input_ids": ids,
                "attention_mask": attention,
                "keep_mask": keep,
            }, {
                "query_id": qid,
                "text": row["query"],
                "language": row["language"],
                "source_record": str(record),
                "source": "Toolkit embed_query; actual forward inputs captured",
            }, timings)

            if number == 1 or number % 20 == 0 or number == len(missing_queries):
                print(
                    f"New queries: {number}/{len(missing_queries)}; "
                    f"total cached: {len(completed_queries) + number}/232",
                    flush=True,
                )

    report["current"] = {"stage": "full_cache_acceptance"}
    save()
    final = store.manifest()
    assert not final["pending_directories"]
    final_pages = {x["id"] for x in final["completed"]["pages"]}
    final_queries = {x["id"] for x in final["completed"]["queries"]}
    assert final_pages == set(pages)
    assert final_queries == set(queries)
    assert final["complete"]

    # Verify every cached identity against the official tables.
    for kind, ids, official in [
        ("pages", final_pages, pages), ("queries", final_queries, queries)
    ]:
        for item_id in ids:
            metadata = json.loads(
                (store.path(kind, item_id) / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )["identity"]
            if kind == "pages":
                assert metadata["doc_id"] == official[item_id]["doc-id"]
            else:
                assert metadata["text"] == official[item_id]["query"]
                assert metadata["language"] == official[item_id]["language"]

    from benchmarks.vidore_tatdqa_test.dataset_loader import (
        _normalize_qrels, _stable_uuid,
    )
    raw_qrels = pq.read_table(
        DATA / "qrels/test-00000-of-00001.parquet"
    ).to_pylist()
    qrels = _normalize_qrels(raw_qrels)
    expected_qrels = {}
    for row in raw_qrels:
        expected_qrels.setdefault(str(row["query-id"]), {})[
            _stable_uuid(str(row["corpus-id"]))
        ] = int(row["score"])
    assert qrels == expected_qrels
    assert len(raw_qrels) == sum(map(len, qrels.values())) == 3628
    assert set(qrels) == final_queries
    index_page_ids = {_stable_uuid(pid) for pid in final_pages}
    assert all(set(rels) <= index_page_ids for rels in qrels.values())
    assert qrels["0"][_stable_uuid("63")] == 2

    (record / "cache-manifest.json").write_bytes(json_bytes(final))
    mappings = [
        {
            "corpus_id": pid,
            "point_id": _stable_uuid(pid),
            "doc_id": pages[pid]["doc-id"],
        }
        for pid in sorted(pages, key=int)
    ]
    (record / "page-id-mapping.json").write_bytes(json_bytes(mappings))

    report["counts"] = {"pages": 452, "queries": 232, "qrels": 3628}
    report["cache_complete"] = True
    report["cache_npz_bytes"] = sum(
        (store.path(kind, entry["id"]) / "vectors.npz").stat().st_size
        for kind in ("pages", "queries")
        for entry in final["completed"][kind]
    )
    report["cache_storage_note"] = (
        "Research NPZ files only; includes raw/full vectors. "
        "Not online index storage and not total project disk usage."
    )
    report["timing_boundaries"] = {
        "page_embed_images": (
            "Toolkit call wall time; includes preprocessing and cleanup"
        ),
        "query_embed_query": "Toolkit call through CPU output copy",
        "page_pooling": "Author row mean plus legacy window=3",
        "cache_write_and_readback": "Atomic item write plus integrity readback",
        "script_work_wall_seconds": "After imports through driver finalization",
    }
    report["code_sha256"] = {
        str(p.relative_to(ROOT)): digest(p)
        for p in [
            Path(__file__),
            ROOT / "scripts/prb_cache.py",
            ROOT / "visual-rag-toolkit/visual_rag/embedding/visual_embedder.py",
            ROOT / "visual-rag-toolkit/visual_rag/embedding/pooling.py",
        ]
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
    report["status"] = "PASS"

except BaseException as exc:
    report["status"] = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
    report["error"] = repr(exc)
    event({
        "status": report["status"],
        "current": report.get("current"),
        "error": repr(exc),
    })
    raise

finally:
    if hook is not None:
        hook.remove()
    stop.set()
    sampler.join(timeout=2)
    report["ram_sampling"] = ram
    report["ram_sampling_interval_seconds"] = 0.1
    report["script_work_wall_seconds"] = time.perf_counter() - started
    if torch is not None and report["model_loaded"]:
        try:
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        except Exception as exc:
            report["cuda_statistics_error"] = repr(exc)
    save()

print("Counts:", report["counts"])
print("Newly encoded:", report["new_pages"], "pages;",
      report["new_queries"], "queries")
print("Cache complete:", report["cache_complete"])
print("This-run timing totals:", report["timing_totals_seconds"])
print("Sampled peak RSS GiB:", ram["peak_rss_bytes"] / 1024**3)
if "peak_allocated_bytes" in report:
    print("PyTorch peak allocated GiB:",
          report["peak_allocated_bytes"] / 1024**3)
print("Record:", record)
print("P2 full encoding cache: PASS")