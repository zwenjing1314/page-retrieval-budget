import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
adapter = root / (
    "models/colpali-v1.3-"
    "b5c6dd62125326e6f0b540c1f7b36e901cdc0a11"
)
base = root / (
    "models/colpaligemma-3b-pt-448-base-"
    "30ab955d073de4a91dc5a288e8c97226647e3e5a"
)
data = root / (
    "data/economics_reports_v2-"
    "76fe40166ba07b1bf50457f5c6057cacdd045f10"
)

# Configure this process before importing model libraries.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PRB_COLPALI_BASE"] = str(base)
os.environ["PRB_ECONOMICS_SNAPSHOT"] = str(data)
os.environ["VISUALRAG_SORT_QUERIES_BY_LENGTH"] = "0"

import hashlib
import io
import json
import time
import uuid
from datetime import datetime
from importlib.metadata import version

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from PIL import Image

from visual_rag.embedding.visual_embedder import VisualEmbedder

old_record = root / "records/p1-single-page-20261106-231636-dd7c3e"
record = root / "records" / (
    "p1-model-entry-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    + "-" + uuid.uuid4().hex[:6]
)
record.mkdir(parents=True, exist_ok=False)

limits = {
    "unit_norm_max_error": 0.005,
    "vector_max_absolute_error": 0.001,
    "score_absolute_error_per_query_token": 0.001,
}
report = {
    "status": "STARTED",
    "limits": limits,
    "adapter": str(adapter),
    "base": str(base),
    "data": str(data),
    "previous_diagnostic": str(old_record),
    "forward_calls": [],
    "comparisons": [],
    "resource_note": (
        "Diagnostic timings, not benchmark measurements. "
        "CUDA allocated/reserved peaks exclude other processes."
    ),
}


def save():
    (record / "report.json").write_bytes(
        json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
    )


def check_vectors(name, array):
    assert array.dtype == np.float16, (name, array.dtype)
    assert array.ndim == 2 and array.shape[1] == 128, name
    assert np.isfinite(array).all(), name
    error = float(
        np.max(np.abs(np.linalg.norm(array.astype(np.float32), axis=1) - 1))
    )
    report.setdefault("vectors", {})[name] = {
        "shape": list(array.shape),
        "unit_norm_max_error": error,
    }
    assert error <= limits["unit_norm_max_error"], (name, error)


def compare(name, a, b):
    assert a.shape == b.shape, (name, a.shape, b.shape)
    error = float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))
    report["comparisons"].append({
        "name": name, "max_absolute_error": error,
    })
    print(name, "max error:", error, flush=True)
    assert error <= limits["vector_max_absolute_error"], (name, error)


def maxsim(q, p):
    q = q.astype(np.float32)
    p = p.astype(np.float32)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    return float((q @ p.T).max(axis=1).sum(dtype=np.float32))


def selected_ids(batch, i):
    ids = batch["input_ids"][i].cpu()
    keep = batch["attention_mask"][i].cpu().bool()
    if (ids[keep] >= 4).any():
        keep = keep & (ids >= 4)
    return ids[keep].numpy()


hook = None
save()
print("Record:", record, flush=True)

try:
    old_meta = json.loads(
        (old_record / "report.json").read_text(encoding="utf-8")
    )
    assert old_meta["status"] == "PASS"
    old_cache = old_record / "single-page-cache.npz"
    assert hashlib.sha256(old_cache.read_bytes()).hexdigest() == (
        old_meta["cache_sha256"]
    )
    with np.load(old_cache, allow_pickle=False) as saved:
        old_page = saved["page_vectors"].copy()
        old_query = saved["query_vectors"].copy()

    assert torch.cuda.is_available()
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()

    # Use the actual patched lazy-loading entry; no private object injection.
    embedder = VisualEmbedder(
        model_name=str(adapter),
        device="cuda:0",
        torch_dtype=torch.float16,
        output_dtype=np.float16,
        batch_size=1,
        processor_speed="slow",
        filter_special_tokens=True,
    )
    assert embedder.filter_special_tokens, (
        "Special-token filtering was overridden by an environment variable"
    )

    started = time.perf_counter()
    model = embedder.model
    processor = embedder.processor
    torch.cuda.synchronize()
    report["load_seconds"] = time.perf_counter() - started

    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
    assert model.active_adapters() == ["default"]
    assert all(p.device.type == "cuda" for p in model.parameters())
    assert all(
        p.dtype == torch.float16
        for p in model.parameters() if p.is_floating_point()
    )
    assert processor.tokenizer.is_fast
    assert type(processor.image_processor).__name__ == "SiglipImageProcessor"
    assert processor.query_prefix == "Query: "
    assert processor.query_augmentation_token == processor.tokenizer.pad_token
    report["processor"] = {
        "tokenizer": type(processor.tokenizer).__name__,
        "image_processor": type(processor.image_processor).__name__,
        "image_config": processor.image_processor.to_dict(),
    }
    print("Patched local model loading: PASS", flush=True)

    rows = pq.read_table(
        data / "queries/test-00000-of-00001.parquet"
    ).to_pylist()
    q0 = next(r for r in rows if r["query-id"] == 0)
    first_inputs = processor.process_queries([q0["query"]])
    first_length = first_inputs["input_ids"].shape[1]

    # Choose a real second query with a different unpadded length.
    other = None
    for row in sorted(rows, key=lambda r: len(r["query"])):
        if row["query-id"] == 0:
            continue
        inputs = processor.process_queries([row["query"]])
        if inputs["input_ids"].shape[1] != first_length:
            other = row
            break
    assert other is not None
    chosen = [q0, other]
    texts = [row["query"] for row in chosen]
    single_inputs = [processor.process_queries([text]) for text in texts]
    batch_inputs = processor.process_queries(texts)

    lengths = [int(x["input_ids"].shape[1]) for x in single_inputs]
    assert lengths[0] != lengths[1]
    assert int((batch_inputs["attention_mask"] == 0).sum()) > 0

    for i in range(2):
        assert np.array_equal(
            selected_ids(single_inputs[i], 0),
            selected_ids(batch_inputs, i),
        ), "Batching changed selected query token IDs"

    # Record real forward calls to detect silent per-query fallback.
    phase = "setup"

    def before_forward(module, args, kwargs):
        ids = kwargs.get("input_ids")
        report["forward_calls"].append({
            "phase": phase,
            "input_shape": list(ids.shape),
            "has_pixels": "pixel_values" in kwargs,
        })

    hook = model.register_forward_pre_hook(before_forward, with_kwargs=True)

    phase = "single_queries"
    started = time.perf_counter()
    with torch.inference_mode():
        singles = [
            embedder.embed_query(text).detach().cpu().numpy()
            for text in texts
        ]
    torch.cuda.synchronize()
    report["single_query_seconds"] = time.perf_counter() - started

    phase = "batch_queries"
    started = time.perf_counter()
    with torch.inference_mode():
        batches = [
            x.detach().cpu().numpy()
            for x in embedder.embed_queries(
                texts, batch_size=2, show_progress=False
            )
        ]
    torch.cuda.synchronize()
    report["batch_query_seconds"] = time.perf_counter() - started

    batch_calls = [
        c for c in report["forward_calls"] if c["phase"] == "batch_queries"
    ]
    assert len(batch_calls) == 1, "Batch used retries or single-query fallback"
    assert batch_calls[0]["input_shape"][0] == 2
    assert len(batches) == 2

    corpus = pq.read_table(data / "corpus/test-00000-of-00001.parquet")
    selected = corpus.filter(
        pc.equal(corpus["corpus-id"], 63)
    ).to_pylist()
    assert len(selected) == 1
    page = selected[0]
    with Image.open(io.BytesIO(page["image"]["bytes"])) as original:
        image = original.convert("RGB")
        image.load()

    phase = "page"
    started = time.perf_counter()
    with torch.inference_mode():
        full, token_info = embedder.embed_images(
            [image], batch_size=1,
            return_token_info=True, show_progress=False,
        )
    torch.cuda.synchronize()
    report["page_seconds"] = time.perf_counter() - started
    assert token_info[0]["num_visual_tokens"] == 1024
    page_vectors = embedder.extract_visual_embedding(full[0], token_info[0])
    assert page_vectors.shape == (1024, 128)

    # Save outputs before numerical assertions, including a failing batch.
    cache = record / "entry-vectors.npz"
    np.savez(
        cache,
        page_vectors=page_vectors,
        page_raw=full[0].numpy(),
        page_visual_indices=np.asarray(
            token_info[0]["visual_token_indices"], dtype=np.int64
        ),
        query0_single=singles[0],
        query1_single=singles[1],
        query0_batch=batches[0],
        query1_batch=batches[1],
        batch_input_ids=batch_inputs["input_ids"].numpy(),
        batch_attention_mask=batch_inputs["attention_mask"].numpy(),
        query0_input_ids=single_inputs[0]["input_ids"].numpy(),
        query0_attention_mask=single_inputs[0]["attention_mask"].numpy(),
        query1_input_ids=single_inputs[1]["input_ids"].numpy(),
        query1_attention_mask=single_inputs[1]["attention_mask"].numpy(),
    )

    report["queries"] = [
        {
            "query_id": row["query-id"],
            "text": row["query"],
            "tokens_before_filter": lengths[i],
            "tokens_after_filter": int(singles[i].shape[0]),
        }
        for i, row in enumerate(chosen)
    ]
    report["page"] = {
        "corpus_id": 63,
        "doc_id": page["doc-id"],
        "image_size": list(image.size),
        "token_info": token_info[0],
    }

    check_vectors("page", page_vectors)
    for i in range(2):
        check_vectors(f"query{i}_single", singles[i])
        check_vectors(f"query{i}_batch", batches[i])
        assert len(singles[i]) == len(selected_ids(single_inputs[i], 0))

    compare("Page entry vs old diagnostic", page_vectors, old_page)
    compare("Query 0 entry vs old diagnostic", singles[0], old_query)

    for i in range(2):
        compare(f"Query {i} single vs batch", singles[i], batches[i])
        single_score = maxsim(singles[i], page_vectors)
        batch_score = maxsim(batches[i], page_vectors)
        error = abs(single_score - batch_score) / len(singles[i])
        report.setdefault("score_checks", []).append({
            "query_id": chosen[i]["query-id"],
            "single": single_score,
            "batch": batch_score,
            "absolute_error_per_query_token": error,
        })
        assert error <= limits["score_absolute_error_per_query_token"]
        print(f"Query {i} score error/token:", error, flush=True)

    source_files = [
        Path(__file__),
        root / "visual-rag-toolkit/visual_rag/embedding/visual_embedder.py",
        root / "visual-rag-toolkit/visual_rag/embedding/pooling.py",
        root / "visual-rag-toolkit/benchmarks/vidore_tatdqa_test/dataset_loader.py",
        root / "visual-rag-toolkit/benchmarks/vidore_beir_qdrant/run_qdrant_beir.py",
    ]
    report["code_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_files
    }
    report["versions"] = {
        name: version(name)
        for name in [
            "torch", "transformers", "peft", "colpali-engine", "qdrant-client"
        ]
    }
    report["cache_sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    report["status"] = "PASS"
    save()

    print("Query IDs:", [r["query-id"] for r in chosen])
    print("Unpadded query lengths:", lengths)
    print("Actual batch forward calls:", batch_calls)
    print("Page shape:", page_vectors.shape)
    print("Peak allocated GiB:",
          report["peak_allocated_bytes"] / 1024**3)
    print("Variable-length token-mask check: PASS")
    print("Single vs batch query check: PASS")
    print("P1 model entry acceptance: PASS")

except Exception as exc:
    report["status"] = "FAILED"
    report["error"] = repr(exc)
    save()
    raise

finally:
    if hook is not None:
        hook.remove()