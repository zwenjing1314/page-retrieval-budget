import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "models/colpali-v1.3-b5c6dd62125326e6f0b540c1f7b36e901cdc0a11"
BASE = ROOT / "models/colpaligemma-3b-pt-448-base-30ab955d073de4a91dc5a288e8c97226647e3e5a"
DATA = ROOT / "data/economics_reports_v2-76fe40166ba07b1bf50457f5c6057cacdd045f10"
ENTRY = ROOT / "records/p1-model-entry-20261107-000621-660a64"
OLD = ROOT / "records/p1-single-page-20261106-231636-dd7c3e"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PRB_COLPALI_BASE"] = str(BASE)
os.environ["PRB_ECONOMICS_SNAPSHOT"] = str(DATA)

import argparse
import io
import json
import time
import uuid
from datetime import datetime
from importlib.metadata import version

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from prb_cache import CacheStore, digest, json_bytes


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pool(vectors):
    from visual_rag.embedding.pooling import (
        colpali_row_mean_pooling,
        colpali_experimental_pooling_from_rows,
    )
    rows = colpali_row_mean_pooling(vectors, output_dtype=np.float16)
    pooled = colpali_experimental_pooling_from_rows(
        rows, window_size=3, output_dtype=np.float16
    )
    return rows, pooled


parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["prepare", "resume-check", "retrieve"])
parser.add_argument("--pages", type=int, default=3)
args = parser.parse_args()
assert 3 <= args.pages <= 8, "Diagnostic page count must be 3..8"

record = ROOT / "records" / (
    "p1-real-cache-" + args.action + "-"
    + datetime.now().strftime("%Y%m%d-%H%M%S")
    + "-" + uuid.uuid4().hex[:6]
)
record.mkdir(parents=True, exist_ok=False)
report = {"status": "STARTED", "action": args.action, "model_loaded": False}


def save():
    (record / "report.json").write_bytes(json_bytes(report))


save()
print("Record:", record, flush=True)

try:
    entry_meta = read_json(ENTRY / "report.json")
    old_meta = read_json(OLD / "report.json")
    assert entry_meta["status"] == old_meta["status"] == "PASS"
    assert digest(ENTRY / "entry-vectors.npz") == entry_meta["cache_sha256"]
    assert digest(OLD / "single-page-cache.npz") == old_meta["cache_sha256"]

    with np.load(ENTRY / "entry-vectors.npz", allow_pickle=False) as saved:
        seed = {k: saved[k].copy() for k in saved.files}
    with np.load(OLD / "single-page-cache.npz", allow_pickle=False) as saved:
        old = {k: saved[k].copy() for k in saved.files}

    page_rows = pq.read_table(
        DATA / "corpus/test-00000-of-00001.parquet",
        columns=["corpus-id", "doc-id"],
    ).to_pylist()
    query_rows = pq.read_table(
        DATA / "queries/test-00000-of-00001.parquet",
        columns=["query-id", "query", "language"],
    ).to_pylist()
    page_map = {str(r["corpus-id"]): r for r in page_rows}
    query_map = {str(r["query-id"]): r for r in query_rows}
    assert len(page_map) == 452 and len(query_map) == 232

    page_ids = ["63"] + [
        str(x) for x in sorted(int(k) for k in page_map if k != "63")[:args.pages - 1]
    ]
    query_ids = [str(q["query_id"]) for q in entry_meta["queries"]]
    assert len(query_ids) == 2 and "0" in query_ids

    image_ids = np.unique(
        old["page_input_ids"].reshape(-1)[old["page_visual_mask"]]
    )
    assert len(image_ids) == 1
    encoding_files = [
        ROOT / "visual-rag-toolkit/visual_rag/embedding/visual_embedder.py",
        ROOT / "visual-rag-toolkit/visual_rag/embedding/pooling.py",
    ]
    for path in encoding_files:
        key = str(path.relative_to(ROOT))
        assert digest(path) == entry_meta["code_sha256"][key], (
            "Encoding code changed since accepted model-entry check"
        )

    profile = {
        "layout_version": 1,
        "model": {
            "adapter_repo": "vidore/colpali-v1.3",
            "adapter_revision": "b5c6dd62125326e6f0b540c1f7b36e901cdc0a11",
            "base_repo": "vidore/colpaligemma-3b-pt-448-base",
            "base_revision": "30ab955d073de4a91dc5a288e8c97226647e3e5a",
        },
        "dataset_revision": "76fe40166ba07b1bf50457f5c6057cacdd045f10",
        "dataset_download_manifest_sha256": digest(DATA / "download-manifest.json"),
        "expected_page_ids": sorted(page_map, key=int),
        "expected_query_ids": sorted(query_map, key=int),
        "image_token_id": int(image_ids[0]),
        "image_grid": [32, 32],
        "image_processor": entry_meta["processor"],
        "attention": "sdpa",
        "dtype": "float16",
        "crop": False,
        "query_filter": "attention_mask; then input_ids>=4 if any survive",
        "normalization": "ColPali L2; row/pool outputs not renormalized",
        "pooling": "author legacy window=3; 32 rows -> 34 vectors",
        "encoding_code_sha256": {
            str(p.relative_to(ROOT)): digest(p) for p in encoding_files
        },
        "versions": {
            k: version(k)
            for k in ["torch", "transformers", "peft", "colpali-engine"]
        },
    }
    store = CacheStore(
        ROOT / "cache/encoded-economics-v1",
        profile,
        create=args.action == "prepare",
    )

    if args.action == "prepare":
        imported = []
        if not store.exists("pages", "63"):
            rows, pooled = pool(seed["page_vectors"])
            assert np.array_equal(
                seed["page_visual_indices"],
                np.flatnonzero(old["page_visual_mask"]),
            )
            store.put("pages", "63", {
                "vectors": seed["page_vectors"],
                "raw": seed["page_raw"],
                "input_ids": old["page_input_ids"].reshape(-1),
                "attention_mask": old["page_attention_mask"].reshape(-1),
                "keep_mask": old["page_visual_mask"],
                "rows": rows,
                "pooled": pooled,
            }, {
                "corpus_id": "63",
                "doc_id": page_map["63"]["doc-id"],
                "source_record": str(ENTRY),
                "mask_source": str(OLD),
                "image_size": entry_meta["page"]["image_size"],
            })
            imported.append("page:63")

        for i, qid in enumerate(query_ids):
            if store.exists("queries", qid):
                continue
            official = query_map[qid]
            assert entry_meta["queries"][i]["text"] == official["query"]
            ids = seed[f"query{i}_input_ids"].reshape(-1)
            attention = seed[f"query{i}_attention_mask"].reshape(-1)
            keep = attention.astype(bool)
            if (ids[keep] >= 4).any():
                keep &= ids >= 4
            store.put("queries", qid, {
                "vectors": seed[f"query{i}_single"],
                "input_ids": ids,
                "attention_mask": attention,
                "keep_mask": keep,
            }, {
                "query_id": qid,
                "text": official["query"],
                "language": official["language"],
                "source_record": str(ENTRY),
            })
            imported.append("query:" + qid)

        # Validate existing entries before deciding what needs encoding.
        missing = []
        for pid in page_ids:
            if store.exists("pages", pid):
                store.read("pages", pid)
            else:
                missing.append(pid)

        report["imported"] = imported
        report["new_page_ids"] = missing
        report["new_page_seconds"] = {}
        save()
        print("Imported:", imported)
        print("Pages needing encoding:", missing, flush=True)

        if missing:
            import torch
            from visual_rag.embedding.visual_embedder import VisualEmbedder

            assert torch.cuda.is_available()
            torch.manual_seed(0)
            torch.cuda.reset_peak_memory_stats()
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
            processor = embedder.processor
            report["model_loaded"] = True
            assert processor.image_token_id == profile["image_token_id"]
            assert processor.image_processor.to_dict() == (
                profile["image_processor"]["image_config"]
            )
            table = pq.read_table(DATA / "corpus/test-00000-of-00001.parquet")

            for pid in missing:
                report["current_page_id"] = pid
                save()
                selected = table.filter(
                    pc.equal(table["corpus-id"], int(pid))
                ).to_pylist()
                assert len(selected) == 1
                row = selected[0]
                from PIL import Image
                with Image.open(io.BytesIO(row["image"]["bytes"])) as original:
                    image = original.convert("RGB")
                    image.load()

                # Preserve input IDs/masks as well as the effective vectors.
                processed = processor.process_images([image])
                started = time.perf_counter()
                with torch.inference_mode():
                    full, info = embedder.embed_images(
                        [image], batch_size=1,
                        return_token_info=True, show_progress=False,
                    )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                vectors = embedder.extract_visual_embedding(full[0], info[0])
                ids = processed["input_ids"][0].numpy()
                attention = processed["attention_mask"][0].numpy()
                keep = ids == processor.image_token_id
                assert np.array_equal(
                    np.flatnonzero(keep),
                    info[0]["visual_token_indices"],
                )
                rows, pooled = pool(vectors)
                store.put("pages", pid, {
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
                    "image_bytes_sha256": __import__("hashlib").sha256(
                        row["image"]["bytes"]
                    ).hexdigest(),
                    "source": "Toolkit batch=1",
                })
                report["new_page_seconds"][pid] = elapsed
                print("Cached page:", pid, "seconds:", elapsed, flush=True)
                save()

            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()

    # Common cache acceptance; no model calls here.
    pages = {}
    for pid in page_ids:
        arrays, meta = store.read("pages", pid)
        assert meta["identity"]["doc_id"] == page_map[pid]["doc-id"]
        rows, pooled = pool(arrays["vectors"])
        assert np.array_equal(rows, arrays["rows"])
        assert np.array_equal(pooled, arrays["pooled"])
        pages[pid] = (arrays, meta)

    queries = {}
    for qid in query_ids:
        arrays, meta = store.read("queries", qid)
        assert meta["identity"]["text"] == query_map[qid]["query"]
        queries[qid] = arrays["vectors"]

    manifest = store.manifest()
    assert not manifest["pending_directories"], (
        "Pending writes found; preserve them and report before continuing"
    )
    (record / "cache-manifest.json").write_bytes(json_bytes(manifest))
    report["diagnostic_page_ids"] = page_ids
    report["diagnostic_query_ids"] = query_ids
    report["cache_complete_for_full_dataset"] = manifest["complete"]
    print("Cache valid: pages", page_ids, "queries", query_ids, flush=True)

    if args.action == "resume-check":
        assert report["model_loaded"] is False
        print("Resume check: 0 model loads; 0 new encodings; PASS")

    if args.action == "retrieve":
        from qdrant_client import QdrantClient
        from prb_index import build_collection
        from visual_rag.retrieval.multi_vector import MultiVectorRetriever
        from benchmarks.vidore_tatdqa_test.dataset_loader import _stable_uuid

        class NoEncoding:
            def embed_query(self, *a, **k):
                raise AssertionError("Cached retrieval must not encode queries")

        def reference(q, p):
            q = q.astype(np.float32)
            p = p.astype(np.float32)
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            p /= np.linalg.norm(p, axis=1, keepdims=True)
            return float((q @ p.T).max(axis=1).sum(dtype=np.float32))

        points = []
        for pid, (arrays, meta) in pages.items():
            points.append({
                "id": _stable_uuid(pid),
                "payload": {
                    "corpus_id": pid,
                    "doc_id": meta["identity"]["doc_id"],
                    "diagnostic": True,
                },
                "vectors": {
                    "initial": arrays["vectors"],
                    "experimental_pooling_3": arrays["pooled"],
                },
            })

        client = QdrantClient(url="http://127.0.0.1:16333", timeout=60)
        report["collections"] = {}
        report["retrieval"] = []
        try:
            assert client.info().version == "1.14.1"
            retrievers = {}
            for mode in ("A", "B", "C"):
                name = "prb_p1_real_" + mode.lower() + "_" + uuid.uuid4().hex[:10]
                report["collections"][mode] = name
                save()
                info = build_collection(client, name, mode, points)
                (record / (mode + "-collection.json")).write_bytes(
                    info.model_dump_json(indent=2).encode("utf-8")
                )
                retrievers[mode] = MultiVectorRetriever(
                    collection_name=name,
                    qdrant_client=client,
                    embedder=NoEncoding(),
                    experimental_vector_name="experimental_pooling_3",
                    max_retries=1,
                )

            expected_ids = {p["id"] for p in points}
            for qid, q in queries.items():
                outputs = {
                    "A": retrievers["A"].search_embedded(
                        query_embedding=q, top_k=3, mode="single_full"
                    ),
                    "B": retrievers["B"].search_embedded(
                        query_embedding=q, top_k=3, mode="two_stage",
                        prefetch_k=3,
                        stage1_mode="tokens_vs_experimental_pooling",
                    ),
                    "C": retrievers["C"].search_embedded(
                        query_embedding=q, top_k=3,
                        mode="single_experimental_tokens",
                    ),
                }
                item = {"query_id": qid, "results": outputs, "near_ties": {}}
                report["retrieval"].append(item)
                for mode, hits in outputs.items():
                    assert len(hits) == 3
                    assert {str(h["id"]) for h in hits} == expected_ids
                    field = "experimental_pooling_3" if mode == "C" else "initial"
                    refs = {
                        p["id"]: reference(q, p["vectors"][field])
                        for p in points
                    }
                    bound = 1e-3 * len(q)
                    for hit in hits:
                        assert abs(
                            hit["score_final"] - refs[str(hit["id"])]
                        ) <= bound
                    near = []
                    for i, a in enumerate(hits):
                        for b in hits[i + 1:]:
                            x, y = str(a["id"]), str(b["id"])
                            assert refs[x] + 2 * bound >= refs[y]
                            if abs(refs[x] - refs[y]) <= 2 * bound:
                                near.append([x, y])
                    item["near_ties"][mode] = near

                a = {str(h["id"]): h["score_final"] for h in outputs["A"]}
                b = {str(h["id"]): h["score_final"] for h in outputs["B"]}
                assert all(abs(a[k] - b[k]) / len(q) <= 1e-3 for k in a)
                print("Real query", qid, "A/B/C diagnostic: PASS")
                save()
        finally:
            client.close()

    report["status"] = "PASS"
    save()
    print("Full-dataset cache complete:", manifest["complete"])
    print("P1 real-cache", args.action, "PASS")

except Exception as exc:
    report["status"] = "FAILED"
    report["error"] = repr(exc)
    save()
    raise