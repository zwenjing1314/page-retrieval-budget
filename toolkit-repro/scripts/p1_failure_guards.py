import copy
import json
import os
import uuid
import torch
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmarks.vidore_beir_qdrant import run_qdrant_beir as runner

root = Path(__file__).resolve().parents[1]
record = root / "records" / (
    "p1-failure-guards-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    + "-" + uuid.uuid4().hex[:6]
)
record.mkdir(parents=True, exist_ok=False)
report = {"status": "STARTED", "cases": []}


def save():
    (record / "report.json").write_bytes(
        json.dumps(report, indent=2).encode("utf-8")
    )


save()
print("Record:", record, flush=True)


class FakeIndexer:
    def __init__(self):
        self.uploads = 0

    def create_collection(self, **kwargs):
        pass

    def create_payload_indexes(self, **kwargs):
        pass

    def upload_batch(self, points, **kwargs):
        self.uploads += 1
        return 0


class FakeEmbedder:
    def __init__(self, failure):
        self.failure = failure
        self.encodes = 0
        self.torch_dtype = torch.float16

    def embed_images(self, images, **kwargs):
        self.encodes += 1
        if self.failure == "encoding":
            raise ValueError("Injected encoding failure")
        return (
            [np.ones((1024, 128), dtype=np.float16)],
            [{"visual_token_indices": list(range(1024))}],
        )

    def mean_pool_visual_embedding(self, *args, **kwargs):
        if self.failure == "pooling":
            raise ValueError("Injected pooling failure")
        return np.ones((32, 128), dtype=np.float16)

    def experimental_pool_visual_embedding(self, *args, **kwargs):
        return np.ones((34, 128), dtype=np.float16)

    def global_pool_from_mean_pool(self, rows):
        return rows.mean(axis=0)


def exercise(stage):
    fake_indexer = FakeIndexer()
    fake_embedder = FakeEmbedder(stage)
    failure_log = record / (stage + ".jsonl")
    page = SimpleNamespace(
        doc_id="synthetic-page",
        image=Image.new("RGB", (8, 8), "white"),
        payload={"source_doc_id": "0", "doc-id": "synthetic-document"},
    )

    kwargs = dict(
        dataset_name="vidore/economics_reports_v2",
        corpus=[page],
        embedder=fake_embedder,
        collection_name="synthetic-no-database",
        prefer_grpc=False,
        qdrant_vector_dtype="float16",
        recreate=False,
        indexing_threshold=0,
        batch_size=1,
        upload_batch_size=1,
        upload_workers=0,
        upsert_wait=True,
        max_corpus_docs=0,
        sample_corpus_docs=0,
        sample_corpus_strategy="first",
        sample_seed=0,
        payload_indexes=[],
        union_namespace="synthetic",
        model_name="vidore/colpali-v1.3",
        resume=False,
        qdrant_timeout=30,
        full_scan_threshold=10,
        crop_empty=False,
        crop_empty_percentage_to_remove=0.0,
        crop_empty_remove_page_number=False,
        crop_empty_preserve_border_px=0,
        crop_empty_uniform_std_threshold=0.0,
        max_mean_pool_vectors=32,
        pooling_windows=[3],
        experimental_pooling_kernel="legacy",
        colsmol_experimental_2d=False,
        no_cloudinary=True,
        cloudinary_folder="unused",
        retry_failures=False,
        only_failures=False,
    )

    # Replace only external dependencies. Execute the real indexing function.
    with (
        patch.dict(os.environ, {"QDRANT_URL": "http://unused.invalid"}),
        patch.object(runner, "QdrantIndexer", return_value=fake_indexer),
        patch.object(runner, "_failed_log_path", return_value=failure_log),
        patch.object(runner, "_load_failed_union_ids", return_value=set()),
    ):
        try:
            runner._index_beir_corpus(**kwargs)
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected indexing to stop")

    expected = {
        "encoding": "Encoding failed;",
        "pooling": "Pooling failed;",
        "upload": "Incomplete upload:",
    }[stage]
    assert expected in message, message
    assert fake_embedder.encodes == 1, "Unexpected retry or continued encoding"
    assert fake_indexer.uploads == (1 if stage == "upload" else 0)
    assert failure_log.is_file()
    entries = [
        json.loads(line)
        for line in failure_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(item.get("stage") == stage for item in entries)
    report["cases"].append({
        "stage": stage,
        "status": "PASS",
        "error": message,
        "encoding_calls": fake_embedder.encodes,
        "upload_calls": fake_indexer.uploads,
        "log": str(failure_log),
    })
    print(stage + " stop-and-log check: PASS", flush=True)


try:
    for stage in ["encoding", "pooling", "upload"]:
        exercise(stage)

    labels = {"0": {"page0": 2}}
    before = copy.deepcopy(labels)
    for failed in [{"page0"}, {"unlabelled-negative-page"}]:
        try:
            runner._remove_failed_from_qrels(labels, failed)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Expected evaluation to stop")
        assert labels == before

    report["label_preservation"] = "PASS"
    report["status"] = "PASS"
    save()
    print("Label preservation: PASS")
    print("P1 failure guards: PASS")

except Exception as exc:
    report["status"] = "FAILED"
    report["error"] = repr(exc)
    save()
    raise