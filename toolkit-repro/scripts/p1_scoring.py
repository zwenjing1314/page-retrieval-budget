import hashlib
import json
import math
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from qdrant_client import QdrantClient, models

from visual_rag.embedding.pooling import compute_maxsim_score
from benchmarks.vidore_tatdqa_test.metrics import ndcg_at_k, recall_at_k

root = Path(__file__).resolve().parents[1]
source = root / "records/p1-single-page-20261106-231636-dd7c3e"
record = root / "records" / (
    "p1-scoring-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    + "-" + uuid.uuid4().hex[:6]
)
record.mkdir(parents=True, exist_ok=False)
tolerance = 1e-3

report = {
    "status": "STARTED",
    "source_record": str(source),
    "absolute_score_error_per_query_token": tolerance,
    "collections": [],
    "comparisons": [],
}


def save():
    (record / "report.json").write_bytes(
        json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
    )


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    assert np.all(norms > 0), "Zero vector is not valid here"
    return x / norms


def reference(q, p):
    # CPU FP32, no padding: max over page vectors, sum over query vectors.
    return float((unit(q) @ unit(p).T).max(axis=1).sum(dtype=np.float32))


def half_reference(q, p):
    # Normalize in FP32, cast to FP16, FP16 CUDA dot product.
    # Sum token scores in FP32 to make the reduction boundary explicit.
    a = torch.tensor(unit(q), device="cuda", dtype=torch.float16)
    b = torch.tensor(unit(p), device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        return float((a @ b.T).float().max(dim=1).values.sum().cpu())


def compare(label, actual, expected, tokens):
    error = abs(actual - expected) / tokens
    report["comparisons"].append({
        "label": label, "actual": actual, "reference": expected,
        "query_tokens": tokens, "absolute_error_per_token": error,
    })
    print(label, "score=", actual, "error/token=", error, flush=True)
    assert error <= tolerance, (label, error)


def make_collection(client, suffix, fields):
    name = "prb_p1_" + suffix + "_" + uuid.uuid4().hex[:10]
    report["collections"].append(name)
    save()
    client.create_collection(
        collection_name=name,
        vectors_config={
            field: models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                datatype=models.Datatype.FLOAT16,
                on_disk=False,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            )
            for field in fields
        },
        hnsw_config=models.HnswConfigDiff(
            m=0, full_scan_threshold=10, on_disk=False
        ),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        on_disk_payload=False,
    )
    info = client.get_collection(name)
    assert set(info.config.params.vectors) == set(fields)
    for v in info.config.params.vectors.values():
        assert v.datatype == models.Datatype.FLOAT16
        assert v.on_disk is False
        assert v.distance == models.Distance.COSINE
        assert v.multivector_config.comparator == models.MultiVectorComparator.MAX_SIM
    assert info.config.hnsw_config.m == 0
    assert info.config.hnsw_config.full_scan_threshold == 10
    assert info.config.optimizer_config.indexing_threshold == 0
    return name


def query(client, name, q, using, limit, prefetch=None):
    kwargs = {}
    if prefetch is not None:
        kwargs["prefetch"] = prefetch
    return client.query_points(
        collection_name=name,
        query=np.asarray(q, dtype=np.float16).tolist(),
        using=using,
        limit=limit,
        search_params=models.SearchParams(exact=True),
        **kwargs,
    ).points


save()
print("Record:", record, flush=True)
client = None

try:
    metadata = json.loads((source / "report.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "PASS"
    cache = source / "single-page-cache.npz"
    assert hashlib.sha256(cache.read_bytes()).hexdigest() == metadata["cache_sha256"]

    with np.load(cache, allow_pickle=False) as saved:
        q = saved["query_vectors"].copy()
        p = saved["page_vectors"].copy()
        pooled = saved["pooled_vectors"].copy()

    assert q.shape == (18, 128)
    assert p.shape == (1024, 128)
    assert pooled.shape == (34, 128)
    assert torch.cuda.is_available()

    # FP32 Toolkit score versus independent FP32 definition.
    for label, vectors in [("full", p), ("pooled", pooled)]:
        ref = reference(q, vectors)
        toolkit = compute_maxsim_score(
            q.astype(np.float32), vectors.astype(np.float32), normalize=True
        )
        compare("Toolkit FP32 " + label, toolkit, ref, len(q))
        compare("FP16 CUDA " + label, half_reference(q, vectors), ref, len(q))

    # Variable-length, all-negative synthetic examples.
    e0 = np.zeros(128, dtype=np.float32)
    e1 = np.zeros(128, dtype=np.float32)
    e0[0] = 1
    e1[1] = 1
    sq = e0[None, :]
    pages = [
        np.stack([-e0]),
        np.stack([-0.8 * e0 + 0.6 * e1, -e0]),
        np.stack([-0.6 * e0 + 0.8 * e1]),
    ]
    expected = [-1.0, -0.8, -0.6]
    for i, page in enumerate(pages):
        assert math.isclose(reference(sq, page), expected[i], abs_tol=1e-6)
        compare("Synthetic FP16 " + str(i),
                half_reference(sq, page), expected[i], 1)

    padded = np.zeros((3, 2, 128), dtype=np.float32)
    valid = np.zeros((3, 2), dtype=bool)
    for i, page in enumerate(pages):
        padded[i, :len(page)] = page
        valid[i, :len(page)] = True

    similarities = np.einsum("qd,bpd->bqp", sq, padded)
    masked = np.where(valid[:, None, :], similarities, -np.inf)
    masked_scores = masked.max(axis=2).sum(axis=1)
    assert np.allclose(masked_scores, expected, atol=1e-6)
    wrong_scores = similarities.max(axis=2).sum(axis=1)
    assert wrong_scores[0] == 0 and wrong_scores[2] == 0
    report["padding_example"] = {
        "correct": masked_scores.tolist(),
        "incorrect_unmasked": wrong_scores.tolist(),
    }
    print("Variable-length and all-negative MaxSim: PASS", flush=True)

    client = QdrantClient(url="http://127.0.0.1:16333", timeout=60)
    assert client.info().version == "1.14.1"

    # Synthetic diagnostic: pooled field intentionally copies full vectors.
    # This tests API/rerank behavior, not the scientific pooling method.
    synthetic = make_collection(client, "synthetic", ["full", "pooled"])
    client.upsert(
        synthetic, wait=True,
        points=[
            models.PointStruct(
                id=i,
                vector={
                    "full": page.astype(np.float16).tolist(),
                    "pooled": page.astype(np.float16).tolist(),
                },
                payload={"kind": "synthetic", "source_id": str(i)},
            )
            for i, page in enumerate(pages)
        ],
    )
    assert client.count(synthetic, exact=True).count == 3
    a = query(client, synthetic, sq, "full", 3)
    b = query(
        client, synthetic, sq, "full", 3,
        prefetch=models.Prefetch(
            query=sq.tolist(), using="pooled", limit=3,
            params=models.SearchParams(exact=True),
        ),
    )
    assert [hit.id for hit in a] == [2, 1, 0]
    assert [hit.id for hit in b] == [hit.id for hit in a]
    for hits, tag in [(a, "A"), (b, "B")]:
        for hit in hits:
            compare("Synthetic Qdrant " + tag + " id=" + str(hit.id),
                    hit.score, expected[hit.id], 1)
    print("Full-candidate prefetch/rerank equality: PASS", flush=True)

    # Actual cached page: two-field diagnostic and pooled-only diagnostic.
    real = make_collection(client, "cached_page", ["full", "pooled"])
    compressed = make_collection(client, "pooled_only", ["pooled"])

    payload = {
        "corpus_id": metadata["corpus_id"],
        "doc_id": metadata["doc_id"],
        "kind": "single-page diagnostic",
    }
    client.upsert(real, wait=True, points=[
        models.PointStruct(
            id=63,
            vector={"full": p.tolist(), "pooled": pooled.tolist()},
            payload=payload,
        )
    ])
    client.upsert(compressed, wait=True, points=[
        models.PointStruct(id=63, vector={"pooled": pooled.tolist()}, payload=payload)
    ])
    for name in [real, compressed]:
        assert client.count(name, exact=True).count == 1

    full_hit = query(client, real, q, "full", 1)[0]
    pooled_hit = query(client, compressed, q, "pooled", 1)[0]
    assert full_hit.id == pooled_hit.id == 63
    compare("Cached full Qdrant", full_hit.score, reference(q, p), len(q))
    compare("Cached pooled Qdrant", pooled_hit.score,
            reference(q, pooled), len(q))

    retrieved = client.retrieve(
        compressed, ids=[63], with_vectors=True, with_payload=True
    )
    assert len(retrieved) == 1
    assert set(retrieved[0].vector) == {"pooled"}
    assert np.asarray(retrieved[0].vector["pooled"]).shape == (34, 128)
    assert retrieved[0].payload == payload
    print("Pooled-only stored-field check: PASS", flush=True)

    # Hand example: two equal-grade relevant pages, one retrieved at rank 2.
    ranking = ["1", "0", "2"]
    rels = {"0": 2, "2": 2}
    hand_ndcg = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    toolkit_ndcg = ndcg_at_k(ranking, rels, 2)
    assert math.isclose(toolkit_ndcg, hand_ndcg, abs_tol=1e-12)
    assert recall_at_k(ranking, rels, 2) == 0.5
    assert ndcg_at_k([], rels, 2) == 0
    assert recall_at_k([], rels, 2) == 0
    assert ndcg_at_k(["0"], {"0": 2}, 1) == 1
    assert recall_at_k(["0"], {"0": 2}, 1) == 1
    report["metric_example"] = {
        "ndcg_at_2": toolkit_ndcg, "recall_at_2": 0.5,
        "note": "ID 0 remains a valid candidate; no identical-ID exclusion",
    }
    print("Hand-calculated metrics: PASS", flush=True)

    for name in report["collections"]:
        info = client.get_collection(name)
        (record / (name + ".json")).write_bytes(
            info.model_dump_json(indent=2).encode("utf-8")
        )

    report["script_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    report["status"] = "PASS"
    save()
    print("P1 scoring diagnostic: PASS", flush=True)

except Exception as exc:
    report["status"] = "FAILED"
    report["error"] = repr(exc)
    save()
    raise

finally:
    if client is not None:
        client.close()