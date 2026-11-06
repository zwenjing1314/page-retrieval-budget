import os

# Set before importing Hugging Face libraries. No remote fallback.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import hashlib
import io
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor
from visual_rag.embedding.visual_embedder import VisualEmbedder
from visual_rag.embedding.pooling import (
    colpali_row_mean_pooling,
    colpali_experimental_pooling_from_rows,
)
from transformers import AutoTokenizer, SiglipImageProcessor

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
record = root / "records" / (
    "p1-single-page-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    + "-" + uuid.uuid4().hex[:6]
)
record.mkdir(parents=True, exist_ok=False)

# Register thresholds before examining outputs.
limits = {
    "unit_norm_absolute_error": 0.005,
    "repeat_max_element_absolute_error": 0.001,
}
report = {
    "status": "STARTED",
    "adapter": str(adapter),
    "base": str(base),
    "data": str(data),
    "limits": limits,
    "image_processor_class": "SiglipImageProcessor",
    "tokenizer_use_fast": True,
    "processor_loading": "explicit local image processor and tokenizer",
    "attention_implementation": "sdpa",
    "batch_size": 1,
    "note": (
        "Diagnostic explicit local base+adapter loading. "
        "Toolkit model loader is not yet patched or accepted."
    ),
}


def save_report():
    (record / "report.json").write_bytes(
        json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
    )


save_report()
print("Record:", record, flush=True)

try:
    for directory in (adapter, base, data):
        assert (directory / "download-manifest.json").is_file(), directory

    for directory in (adapter, base):
        manifest = json.loads(
            (directory / "download-manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["revision"] in directory.name

    assert torch.cuda.is_available()
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    free, total = torch.cuda.mem_get_info()
    print("GPU free/total GiB:", free / 1024**3, total / 1024**3, flush=True)

    queries = pq.read_table(
        data / "queries/test-00000-of-00001.parquet"
    ).to_pylist()
    query = next(r for r in queries if r["query-id"] == 0)

    corpus = pq.read_table(
        data / "corpus/test-00000-of-00001.parquet"
    )
    selected = corpus.filter(
        __import__("pyarrow").compute.equal(corpus["corpus-id"], 63)
    ).to_pylist()
    assert len(selected) == 1
    page = selected[0]

    with Image.open(io.BytesIO(page["image"]["bytes"])) as original:
        image = original.convert("RGB")
        image.load()

        tokenizer = AutoTokenizer.from_pretrained(
        str(adapter),
        local_files_only=True,
        use_fast=True,
    )
    assert tokenizer.is_fast

    image_processor = SiglipImageProcessor.from_pretrained(
        str(adapter),
        local_files_only=True,
    )
    processor = ColPaliProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
    )
    print("Tokenizer:", type(tokenizer).__name__, flush=True)
    print("Image processor:", type(image_processor).__name__, flush=True)
    assert processor.query_prefix == "Query: "
    assert processor.query_augmentation_token == processor.tokenizer.pad_token

    doc_inputs = processor.process_images([image])
    query_inputs = processor.process_queries([query["query"]])
    image_mask = (
        doc_inputs["input_ids"][0] == processor.image_token_id
    )
    assert image_mask.sum().item() == 1024

    query_attn = query_inputs["attention_mask"][0].bool()
    query_mask = query_attn.clone()
    valid_ids = query_inputs["input_ids"][0][query_attn]
    if (valid_ids >= 4).any():
        query_mask = query_attn & (query_inputs["input_ids"][0] >= 4)
    assert query_mask.any()

    print("Processor offline check: PASS", flush=True)
    print("Loading fixed local base and adapter...", flush=True)
    started = time.perf_counter()

    model = ColPali.from_pretrained(
        str(base),
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.load_adapter(
        str(adapter),
        adapter_name="default",
        is_trainable=False,
        adapter_kwargs={"local_files_only": True},
    )
    model.set_adapter("default")
    model.eval()
    model.requires_grad_(False)
    torch.cuda.synchronize()
    report["model_load_seconds"] = time.perf_counter() - started
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
    assert all(p.device.type == "cuda" for p in model.parameters())
    assert all(
        p.dtype == torch.float16
        for p in model.parameters() if p.is_floating_point()
    )

    print("Active adapters:", model.active_adapters(), flush=True)

    # Seed only this diagnostic instance with explicitly loaded objects.
    # The original Toolkit loading function remains unchanged.
    embedder = VisualEmbedder(
        model_name=str(adapter),
        device="cuda:0",
        torch_dtype=torch.float16,
        batch_size=1,
        processor_speed="slow",
    )
    embedder._model = model
    embedder._processor = processor
    embedder._image_token_id = processor.image_token_id

    def forward_cpu(inputs):
        batch = inputs.to("cuda:0")
        with torch.inference_mode():
            result = model(**batch).detach().cpu()
        torch.cuda.synchronize()
        return result[0]

    started = time.perf_counter()
    full_page = forward_cpu(doc_inputs)
    page_vectors = full_page[image_mask]
    full_query = forward_cpu(query_inputs)
    query_vectors = full_query[query_mask]
    report["first_page_and_query_seconds"] = time.perf_counter() - started

    assert tuple(page_vectors.shape) == (1024, 128)
    assert query_vectors.ndim == 2 and query_vectors.shape[1] == 128

    def check_vectors(name, vectors):
        assert vectors.dtype == torch.float16
        assert torch.isfinite(vectors).all().item(), name
        norms = vectors.float().norm(dim=-1)
        error = (norms - 1).abs().max().item()
        assert error <= limits["unit_norm_absolute_error"], (name, error)
        return {
            "shape": list(vectors.shape),
            "dtype": str(vectors.dtype),
            "norm_min": norms.min().item(),
            "norm_max": norms.max().item(),
            "unit_norm_max_error": error,
        }

    report["page_vectors"] = check_vectors("page", page_vectors)
    report["query_vectors"] = check_vectors("query", query_vectors)

    # Repeat image once; use Toolkit's own query method for repeat.
    page_repeat = forward_cpu(doc_inputs)[image_mask]
    with torch.inference_mode():
        query_repeat = embedder.embed_query(query["query"]).detach().cpu()

    assert query_repeat.shape == query_vectors.shape
    page_error = (page_vectors.float() - page_repeat.float()).abs().max().item()
    query_error = (query_vectors.float() - query_repeat.float()).abs().max().item()
    report["repeat_max_element_error"] = {
        "page": page_error, "query": query_error,
    }
    assert page_error <= limits["repeat_max_element_absolute_error"]
    assert query_error <= limits["repeat_max_element_absolute_error"]

    rows = colpali_row_mean_pooling(
        page_vectors, grid_size=32, output_dtype=np.float16
    )
    pooled = colpali_experimental_pooling_from_rows(
        rows, window_size=3, output_dtype=np.float16
    )
    assert rows.shape == (32, 128)
    assert pooled.shape == (34, 128)
    assert np.isfinite(pooled).all()

    # Save raw outputs and masks; this is a diagnostic cache, not full corpus.
    cache = record / "single-page-cache.npz"
    np.savez(
        cache,
        page_raw=full_page.numpy(),
        page_input_ids=doc_inputs["input_ids"].cpu().numpy(),
        page_attention_mask=doc_inputs["attention_mask"].cpu().numpy(),
        page_visual_mask=image_mask.numpy(),
        page_vectors=page_vectors.numpy(),
        query_raw=full_query.numpy(),
        query_input_ids=query_inputs["input_ids"].cpu().numpy(),
        query_attention_mask=query_inputs["attention_mask"].cpu().numpy(),
        query_keep_mask=query_mask.numpy(),
        query_vectors=query_vectors.numpy(),
        row_vectors=rows,
        pooled_vectors=pooled,
    )

    # Re-pool exclusively from saved vectors; no further model call.
    with np.load(cache, allow_pickle=False) as saved:
        recovered_rows = colpali_row_mean_pooling(
            saved["page_vectors"], grid_size=32, output_dtype=np.float16
        )
        recovered_pool = colpali_experimental_pooling_from_rows(
            recovered_rows, window_size=3, output_dtype=np.float16
        )
        assert np.array_equal(recovered_pool, saved["pooled_vectors"])

    report.update({
        "query_id": query["query-id"],
        "query_text": query["query"],
        "corpus_id": page["corpus-id"],
        "doc_id": page["doc-id"],
        "image_size": list(image.size),
        "image_grid": [32, 32],
        "query_tokens_before_filter": int(full_query.shape[0]),
        "query_tokens_attention_valid": int(query_attn.sum()),
        "query_tokens_after_filter": int(query_mask.sum()),
        "page_raw_shape": list(full_page.shape),
        "pooled_shape": list(pooled.shape),
        "normalization": "ColPali forward L2; pooling not renormalized here",
        "query_filter": "attention_mask then input_ids>=4 if any survive",
        "image_filter": "input_ids == processor.image_token_id",
        "processor_image_config": processor.image_processor.to_dict(),
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "status": "PASS",
    })
    save_report()

    print("Page:", report["page_vectors"], flush=True)
    print("Query:", report["query_vectors"], flush=True)
    print("Repeat errors:", report["repeat_max_element_error"], flush=True)
    print("Pooling:", pooled.shape, flush=True)
    print("Query tokens before/after:",
          full_query.shape[0], query_vectors.shape[0], flush=True)
    print("Peak allocated GiB:",
          report["peak_allocated_bytes"] / 1024**3, flush=True)
    print("Cache-only pooling: PASS", flush=True)
    print("P1 single-page diagnostic: PASS", flush=True)

except Exception as exc:
    report["status"] = "FAILED"
    report["error"] = repr(exc)
    save_report()
    raise