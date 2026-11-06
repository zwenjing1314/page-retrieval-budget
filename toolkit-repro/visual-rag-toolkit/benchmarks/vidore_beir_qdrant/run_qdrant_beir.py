import argparse
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmarks.vidore_tatdqa_test.dataset_loader import load_vidore_beir_dataset
from benchmarks.vidore_tatdqa_test.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from visual_rag import VisualEmbedder
from visual_rag.indexing.cloudinary_uploader import CloudinaryUploader
from visual_rag.indexing.qdrant_indexer import QdrantIndexer
from visual_rag.retrieval import MultiVectorRetriever


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if Path(".env").exists():
        load_dotenv(".env")


def _torch_dtype_to_str(dtype) -> str:
    if dtype is None:
        return "auto"
    s = str(dtype)
    return s.replace("torch.", "")


def _parse_torch_dtype(dtype_str: str):
    if dtype_str == "auto":
        return None
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_str]


def _stable_uuid(text: str) -> str:
    import hashlib

    hex_str = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"


def _sample_list(items: List[Any], *, k: int, strategy: str, seed: int) -> List[Any]:
    if not k or k <= 0:
        return items
    if k >= len(items):
        return items
    if strategy == "first":
        return items[:k]
    if strategy == "random":
        import random

        rng = random.Random(int(seed))
        indices = rng.sample(range(len(items)), k)
        return [items[i] for i in indices]
    raise ValueError("sample strategy must be 'first' or 'random'")


def _parse_payload_indexes(values: List[str]) -> List[Dict[str, str]]:
    indexes: List[Dict[str, str]] = []
    for raw in values or []:
        if ":" not in raw:
            raise ValueError("payload index must be in field:type format")
        field, type_str = raw.split(":", 1)
        field = field.strip()
        type_str = type_str.strip()
        if not field or not type_str:
            raise ValueError("payload index must be in field:type format")
        indexes.append({"field": field, "type": type_str})
    return indexes


def _union_point_id(
    *, dataset_name: str, source_doc_id: str, union_namespace: Optional[str]
) -> str:
    ns = f"{union_namespace}::{dataset_name}" if union_namespace else dataset_name
    return _stable_uuid(f"{ns}::{source_doc_id}")


def _filter_qrels(
    qrels: Dict[str, Dict[str, int]], query_ids: List[str]
) -> Dict[str, Dict[str, int]]:
    keep = set(query_ids)
    return {qid: rels for qid, rels in qrels.items() if qid in keep}


def _failed_log_path(*, collection_name: str, dataset_name: str) -> Path:
    dir_name = _safe_filename(collection_name)
    return Path("results") / dir_name / f"index_failures__{_safe_filename(dataset_name)}.jsonl"


def _resolve_output_path(raw_output: str, *, collection_name: str) -> Path:
    """
    Default behavior:
    - If --output is a bare filename, write it to results/{collection_name}/{filename}
    - If --output points into the legacy results/reports/, rewrite into results/{collection_name}/
    - If --output includes any other directory (relative or absolute), respect it
    """
    p = Path(str(raw_output))
    dir_name = _safe_filename(collection_name)

    if str(p).startswith("results/reports/"):
        return Path("results") / dir_name / p.name
    if p.is_absolute():
        return p
    if p.parent == Path("."):
        return Path("results") / dir_name / p.name
    return p


def _default_output_filename(*, args, datasets: List[str]) -> str:
    model_tag = _safe_filename(str(args.model).split("/")[-1])
    scope_tag = _safe_filename(str(args.evaluation_scope))
    mode_tag = _safe_filename(str(args.mode))
    topk_tag = f"top{int(args.top_k)}"
    ds_tag = f"{len(datasets)}ds"

    parts = [model_tag, mode_tag]
    if str(args.mode) == "two_stage":
        parts.append(_safe_filename(str(args.stage1_mode)))
        # Disambiguate which experimental named vector was used.
        # Without this, `--experimental-pooling-k 3` overwrites the default `experimental_pooling` run.
        if str(args.stage1_mode) in (
            "pooled_query_vs_experimental_pooling",
            "tokens_vs_experimental_pooling",
            "pooled_query_vs_experimental",
            "tokens_vs_experimental",
        ):
            if getattr(args, "experimental_pooling_technique", None):
                parts.append(f"exptech{_safe_filename(str(args.experimental_pooling_technique))}")
            if getattr(args, "experimental_pooling_k", None) is not None:
                parts.append(f"expk{int(args.experimental_pooling_k)}")
        parts.append(f"pk{int(args.prefetch_k)}")
    if str(args.mode) in ("single_experimental_tokens", "single_experimental_pooled"):
        if getattr(args, "experimental_pooling_technique", None):
            parts.append(f"exptech{_safe_filename(str(args.experimental_pooling_technique))}")
        if getattr(args, "experimental_pooling_k", None) is not None:
            parts.append(f"expk{int(args.experimental_pooling_k)}")
    if str(args.mode) == "three_stage":
        parts.append("tokens_vs_global")
        parts.append(f"s1k{int(args.stage1_k)}")
        parts.append("tokens_vs_experimental_pooling")
        parts.append(f"s2k{int(args.stage2_k)}")
    parts.extend([topk_tag, scope_tag, ds_tag])

    if bool(args.crop_empty):
        pct = int(round(float(args.crop_empty_percentage_to_remove) * 100))
        parts.append(f"crop{pct}")

    name = "__".join([p for p in parts if p])
    return f"{name}.json"


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_failed_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    with path.open("r") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            for key in ("union_doc_id", "doc_id"):
                v = obj.get(key)
                if v:
                    ids.add(str(v))
    return ids


def _load_failed_union_ids(
    path: Path,
    *,
    dataset_name: str,
    union_namespace: Optional[str],
) -> set:
    """
    Load a set of union_doc_id values usable against Qdrant point IDs.

    Older logs may contain union_doc_id computed without union_namespace.
    We always recompute the current union_doc_id from source_doc_id to make retries consistent.
    """
    if not path.exists():
        return set()
    out = set()
    with path.open("r") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            src = obj.get("source_doc_id")
            if src:
                out.add(
                    _union_point_id(
                        dataset_name=str(dataset_name),
                        source_doc_id=str(src),
                        union_namespace=union_namespace,
                    )
                )
            u = obj.get("union_doc_id")
            if u:
                out.add(str(u))
    return out


def _remove_failed_from_qrels(
    qrels: Dict[str, Dict[str, int]], failed_ids: set
) -> Tuple[Dict[str, Dict[str, int]], int]:
    removed = 0
    if not failed_ids:
        return qrels, 0
    out: Dict[str, Dict[str, int]] = {}
    for qid, rels in (qrels or {}).items():
        new_rels: Dict[str, int] = {}
        for did, score in (rels or {}).items():
            if str(did) in failed_ids:
                removed += 1
                continue
            new_rels[str(did)] = int(score)
        out[str(qid)] = new_rels
    return out, removed


def _filter_failed_ids_to_missing(
    *,
    qdrant_client,
    collection_name: str,
    failed_ids: set,
    timeout: int,
    batch_size: int = 128,
) -> set:
    """
    Failure logs are append-only and can contain historical IDs that may have been
    successfully retried later. To avoid poisoning evaluation, keep only IDs that
    are *still missing* in Qdrant.
    """
    failed_ids = set(str(x) for x in (failed_ids or set()) if x)
    if not failed_ids:
        return set()

    missing = set()
    ids_list = list(failed_ids)
    for i in range(0, len(ids_list), int(batch_size)):
        chunk = ids_list[i : i + int(batch_size)]
        try:
            recs = qdrant_client.retrieve(
                collection_name=str(collection_name),
                ids=chunk,
                with_payload=False,
                with_vectors=False,
                timeout=int(timeout),
            )
            present = set(str(r.id) for r in (recs or []))
            for cid in chunk:
                if str(cid) not in present:
                    missing.add(str(cid))
        except Exception:
            # If retrieve fails (e.g. transient network), be conservative: treat as missing.
            missing.update(str(cid) for cid in chunk)
    return missing


def _evaluate(
    *,
    queries,
    qrels: Dict[str, Dict[str, int]],
    retriever: MultiVectorRetriever,
    embedder: VisualEmbedder,
    top_k: int,
    prefetch_k: int,
    mode: str,
    stage1_mode: str,
    stage1_k: int,
    stage2_k: int,
    max_queries: int,
    drop_empty_queries: bool,
    filter_obj=None,
) -> Dict[str, float]:
    eval_started_at = time.time()
    if drop_empty_queries:
        queries = [q for q in queries if any(v > 0 for v in qrels.get(q.query_id, {}).values())]
    if max_queries and max_queries > 0:
        queries = queries[:max_queries]
    if not queries:
        return {
            "ndcg@1": 0.0,
            "ndcg@5": 0.0,
            "ndcg@10": 0.0,
            "ndcg@100": 0.0,
            "mrr@1": 0.0,
            "mrr@5": 0.0,
            "mrr@10": 0.0,
            "mrr@100": 0.0,
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "recall@100": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "eval_wall_time_s": 0.0,
            "eval_search_time_s": 0.0,
            "qps": 0.0,
            "num_queries_eval": 0,
        }

    ndcg1: List[float] = []
    ndcg5: List[float] = []
    ndcg10: List[float] = []
    ndcg100: List[float] = []
    mrr1: List[float] = []
    mrr5: List[float] = []
    mrr10: List[float] = []
    mrr100: List[float] = []
    recall1: List[float] = []
    recall5: List[float] = []
    recall10: List[float] = []
    recall100: List[float] = []
    latencies_ms: List[float] = []

    retrieve_k = max(100, top_k)

    query_texts = [q.text for q in queries]
    embed_started_at = time.time()
    print(f"📝 Embedding {len(query_texts)} queries…")
    sys.stdout.flush()
    # Always try to show a progress bar during query embedding.
    # If the installed VisualEmbedder version doesn't support show_progress, fall back gracefully.
    try:
        query_embeddings = embedder.embed_queries(
            query_texts,
            batch_size=getattr(embedder, "batch_size", None),
            show_progress=True,
        )
    except TypeError:
        query_embeddings = embedder.embed_queries(
            query_texts,
            batch_size=getattr(embedder, "batch_size", None),
        )
    embed_s = float(max(time.time() - embed_started_at, 0.0))
    print(f"✅ Embedded queries in {embed_s:.2f}s")
    sys.stdout.flush()

    iterator = queries
    try:
        from tqdm import tqdm

        iterator = tqdm(queries, desc="Searching", unit="q")
    except ImportError:
        pass

    for q, qemb in zip(iterator, query_embeddings):
        start = time.time()
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and isinstance(qemb, torch.Tensor):
            # Keep evaluation stable across dtypes/devices:
            # - numpy doesn't support bfloat16
            # - float16 queries can cause large quality drops on some backends
            qemb_np = qemb.detach().float().cpu().numpy()
        else:
            qemb_np = qemb.numpy()

        results = retriever.search_embedded(
            query_embedding=qemb_np,
            top_k=retrieve_k,
            mode=mode,
            prefetch_k=prefetch_k,
            stage1_mode=stage1_mode,
            stage1_k=int(stage1_k),
            stage2_k=int(stage2_k),
            filter_obj=filter_obj,
        )
        latencies_ms.append((time.time() - start) * 1000.0)

        ranking = [str(r["id"]) for r in results]
        rels = qrels.get(q.query_id, {})

        ndcg1.append(ndcg_at_k(ranking, rels, k=1))
        ndcg5.append(ndcg_at_k(ranking, rels, k=5))
        ndcg10.append(ndcg_at_k(ranking, rels, k=10))
        ndcg100.append(ndcg_at_k(ranking, rels, k=100))
        mrr1.append(mrr_at_k(ranking, rels, k=1))
        mrr5.append(mrr_at_k(ranking, rels, k=5))
        mrr10.append(mrr_at_k(ranking, rels, k=10))
        mrr100.append(mrr_at_k(ranking, rels, k=100))
        recall1.append(recall_at_k(ranking, rels, k=1))
        recall5.append(recall_at_k(ranking, rels, k=5))
        recall10.append(recall_at_k(ranking, rels, k=10))
        recall100.append(recall_at_k(ranking, rels, k=100))

    eval_wall_time_s = float(max(time.time() - eval_started_at, 0.0))
    eval_search_time_s = float(np.sum(latencies_ms) / 1000.0) if latencies_ms else 0.0
    qps = float(len(queries) / eval_wall_time_s) if eval_wall_time_s > 0 else 0.0
    return {
        "ndcg@1": float(np.mean(ndcg1)),
        "ndcg@5": float(np.mean(ndcg5)),
        "ndcg@10": float(np.mean(ndcg10)),
        "ndcg@100": float(np.mean(ndcg100)),
        "mrr@1": float(np.mean(mrr1)),
        "mrr@5": float(np.mean(mrr5)),
        "mrr@10": float(np.mean(mrr10)),
        "mrr@100": float(np.mean(mrr100)),
        "recall@1": float(np.mean(recall1)),
        "recall@5": float(np.mean(recall5)),
        "recall@10": float(np.mean(recall10)),
        "recall@100": float(np.mean(recall100)),
        "avg_latency_ms": float(np.mean(latencies_ms)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "eval_wall_time_s": eval_wall_time_s,
        "eval_search_time_s": eval_search_time_s,
        "qps": qps,
        "num_queries_eval": int(len(queries)),
    }


def _detect_collection_vector_dtype(*, client, collection_name: str) -> Optional[str]:
    """
    Best-effort detection of the stored vector datatype for a Qdrant collection.

    Returns:
        "float16", "float32", or None if unavailable.
    """
    try:
        info = client.get_collection(str(collection_name))
    except Exception:
        return None

    try:
        vectors = info.config.params.vectors or {}
    except Exception:
        vectors = {}

    vp = None
    if isinstance(vectors, dict):
        vp = vectors.get("initial") or (next(iter(vectors.values())) if vectors else None)
    if vp is None:
        return None

    dt = getattr(vp, "datatype", None)
    if dt is None:
        return None

    s = str(dt).lower()
    if "float16" in s:
        return "float16"
    if "float32" in s:
        return "float32"
    return None


def _collection_vector_names(*, client, collection_name: str) -> set[str]:
    try:
        info = client.get_collection(str(collection_name))
    except Exception:
        return set()
    try:
        vectors = info.config.params.vectors or {}
    except Exception:
        vectors = {}
    if isinstance(vectors, dict):
        return set(str(k) for k in vectors.keys())
    return set()


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


def _safe_filename(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def _index_beir_corpus(
    *,
    dataset_name: str,
    corpus,
    embedder: VisualEmbedder,
    collection_name: str,
    prefer_grpc: bool,
    qdrant_vector_dtype: str,
    recreate: bool,
    indexing_threshold: int,
    batch_size: int,
    upload_batch_size: int,
    upload_workers: int,
    upsert_wait: bool,
    max_corpus_docs: int,
    sample_corpus_docs: int,
    sample_corpus_strategy: str,
    sample_seed: int,
    payload_indexes: List[Dict[str, str]],
    union_namespace: Optional[str],
    model_name: str,
    resume: bool,
    qdrant_timeout: int,
    full_scan_threshold: int,
    crop_empty: bool,
    crop_empty_percentage_to_remove: float,
    crop_empty_remove_page_number: bool,
    crop_empty_preserve_border_px: int,
    crop_empty_uniform_std_threshold: float,
    max_mean_pool_vectors: Optional[int],
    pooling_windows: Optional[List[int]],
    experimental_pooling_kernel: str,
    colsmol_experimental_2d: bool,
    no_cloudinary: bool,
    cloudinary_folder: str,
    retry_failures: bool,
    only_failures: bool,
) -> None:
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise ValueError("QDRANT_URL not set")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    indexer = QdrantIndexer(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        prefer_grpc=prefer_grpc,
        vector_datatype=qdrant_vector_dtype,
        timeout=int(qdrant_timeout),
    )

    model_lower = (model_name or "").lower()
    is_colqwen25 = "colqwen2.5" in model_lower or "colqwen2_5" in model_lower
    is_colsmol = "colsmol" in model_lower
    kernel_arg = str(experimental_pooling_kernel or "auto").lower().strip()
    if is_colqwen25:
        # ColQwen2.5: store technique variants (k is fixed at 3)
        if pooling_windows:
            raise ValueError(
                "ColQwen2.5 does not support --pooling-windows; it stores gaussian+triangular variants."
            )
        if kernel_arg not in ("auto", "gaussian", "triangular"):
            raise ValueError(
                "ColQwen2.5 experimental pooling kernel is fixed to gaussian+triangular variants (k=3)."
            )
        kernel = "gaussian"
        ks_norm = [3]
        experimental_vector_names = [
            "experimental_pooling_gaussian",
            "experimental_pooling_triangular",
        ]
    else:
        # ColPali-style experimental pooling supports different window sizes
        if kernel_arg == "auto":
            kernel = "legacy"
        else:
            kernel = kernel_arg

        default_k = 3
        ks = pooling_windows if pooling_windows else [default_k]
        seen_ks = set()
        ks_norm = []
        for k in ks:
            try:
                ki = int(k)
            except Exception:
                continue
            if ki <= 0:
                continue
            if ki in seen_ks:
                continue
            seen_ks.add(ki)
            ks_norm.append(ki)
        if not ks_norm:
            ks_norm = [default_k]
        experimental_vector_names = [f"experimental_pooling_{int(k)}" for k in ks_norm]
    if is_colsmol and bool(colsmol_experimental_2d):
        experimental_vector_names.append("experimental_pooling_2d")

    indexer.create_collection(
        force_recreate=recreate,
        indexing_threshold=indexing_threshold,
        full_scan_threshold=int(full_scan_threshold),
        experimental_vector_names=experimental_vector_names,
    )
    indexer.create_payload_indexes(fields=payload_indexes)

    cloudinary_uploader: Optional[CloudinaryUploader] = None
    if not bool(no_cloudinary):
        try:
            cloudinary_uploader = CloudinaryUploader(folder=str(cloudinary_folder))
        except Exception:
            cloudinary_uploader = None

    failure_log = _failed_log_path(collection_name=collection_name, dataset_name=dataset_name)
    failed_ids = _load_failed_union_ids(
        failure_log, dataset_name=dataset_name, union_namespace=union_namespace
    )
    previously_failed_ids = set(failed_ids)

    existing_ids = set()
    if resume:
        offset = None
        while True:
            points, next_offset = indexer.client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            for p in points:
                existing_ids.add(str(p.id))
            if not next_offset or not points:
                break
            offset = next_offset
        if not bool(retry_failures) and not bool(only_failures):
            existing_ids |= failed_ids

    target_ids = None
    if bool(only_failures):
        target_ids = set(str(x) for x in failed_ids)
        if not target_ids:
            print(f"No failed ids found for dataset={dataset_name}; nothing to retry.")
            return

    if sample_corpus_docs and sample_corpus_docs > 0:
        corpus = _sample_list(
            list(corpus),
            k=int(sample_corpus_docs),
            strategy=str(sample_corpus_strategy),
            seed=int(sample_seed),
        )
    elif max_corpus_docs and max_corpus_docs > 0:
        corpus = corpus[:max_corpus_docs]
    total_docs = len(corpus)
    points_buffer: List[Dict[str, Any]] = []

    def _safe_public_id(s: str) -> str:
        out = _safe_filename(str(s))
        return out[:180] if len(out) > 180 else out

    def _ensure_pil(img):
        try:
            from PIL import Image
        except Exception:
            return img
        if img is None:
            return None
        if isinstance(img, Image.Image):
            return img
        try:
            return img.convert("RGB")
        except Exception:
            return img

    def _resized_for_display(img):
        from PIL import Image

        if img is None or not isinstance(img, Image.Image):
            return None
        out = img.copy()
        out.thumbnail((1024, 1024), Image.BICUBIC)
        return out

    uploaded_docs = 0
    skipped_docs = 0
    start_time = time.time()
    last_tick_time = start_time
    last_tick_docs = 0
    last_s_per_doc = 0.0

    pbar = None
    try:
        from tqdm import tqdm

        pbar = tqdm(total=total_docs, desc="📦 Indexing corpus", unit="doc")
    except ImportError:
        pass

    import threading
    from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor
    from concurrent.futures import wait as futures_wait

    stop_event = threading.Event()
    executor = (
        ThreadPoolExecutor(max_workers=int(upload_workers))
        if upload_workers and upload_workers > 0
        else None
    )
    futures = []

    def _upload(points: List[Dict[str, Any]]) -> int:
        uploaded = int(
            indexer.upload_batch(
                points, delay_between_batches=0.0, wait=upsert_wait, stop_event=stop_event
            )
            or 0
        )
        if uploaded <= 0 and points:
            for p in points:
                pid = str(p.get("id") or "")
                if pid and pid not in failed_ids:
                    _append_jsonl(
                        failure_log,
                        {
                            "dataset": dataset_name,
                            "collection": collection_name,
                            "model": model_name,
                            "source_doc_id": str(
                                (p.get("metadata") or {}).get("source_doc_id") or ""
                            ),
                            "doc_id": str((p.get("metadata") or {}).get("doc_id") or ""),
                            "union_doc_id": pid,
                            "error": "Qdrant upsert failed (all retries exhausted)",
                        },
                    )
                    failed_ids.add(pid)
        return uploaded

    def _drain(block: bool) -> None:
        nonlocal uploaded_docs
        if not futures:
            return
        done, _ = futures_wait(futures, return_when=FIRST_EXCEPTION, timeout=None if block else 0)
        for d in list(done):
            futures.remove(d)
            uploaded_docs += int(d.result() or 0)

    try:
        for start in range(0, total_docs, batch_size):
            batch = corpus[start : start + batch_size]
            batch_total = len(batch)
            if target_ids is not None:
                filtered = []
                for d in batch:
                    source_doc_id = str((d.payload or {}).get("source_doc_id") or d.doc_id)
                    union_doc_id = _union_point_id(
                        dataset_name=dataset_name,
                        source_doc_id=source_doc_id,
                        union_namespace=union_namespace,
                    )
                    if union_doc_id in existing_ids:
                        skipped_docs += 1
                        continue
                    if union_doc_id in target_ids:
                        filtered.append(d)
                    else:
                        skipped_docs += 1
                batch = filtered
            elif existing_ids:
                filtered = []
                for d in batch:
                    source_doc_id = str((d.payload or {}).get("source_doc_id") or d.doc_id)
                    union_doc_id = _union_point_id(
                        dataset_name=dataset_name,
                        source_doc_id=source_doc_id,
                        union_namespace=union_namespace,
                    )
                    if union_doc_id in existing_ids:
                        skipped_docs += 1
                    else:
                        filtered.append(d)
                batch = filtered

            if not batch:
                if pbar is not None:
                    pbar.update(batch_total)
                    now = time.time()
                    done_docs = int(pbar.n)
                    elapsed = max(now - start_time, 1e-9)
                    avg_s_per_doc = elapsed / max(done_docs, 1)
                    delta_docs = done_docs - last_tick_docs
                    delta_t = max(now - last_tick_time, 1e-9)
                    if delta_docs > 0:
                        last_s_per_doc = delta_t / delta_docs
                        last_tick_time = now
                        last_tick_docs = done_docs
                    pbar.set_postfix(
                        {
                            "avg_s/doc": f"{avg_s_per_doc:.2f}",
                            "last_s/doc": f"{last_s_per_doc:.2f}",
                            "upl": uploaded_docs,
                            "skip": skipped_docs,
                        }
                    )
                continue

            if crop_empty:
                from visual_rag.preprocessing.crop_empty import CropEmptyConfig
                from visual_rag.preprocessing.crop_empty import crop_empty as _crop_empty

                crop_cfg = CropEmptyConfig(
                    percentage_to_remove=float(crop_empty_percentage_to_remove),
                    remove_page_number=bool(crop_empty_remove_page_number),
                    preserve_border_px=int(crop_empty_preserve_border_px),
                    uniform_rowcol_std_threshold=float(crop_empty_uniform_std_threshold),
                )
                crop_metas = []
                images = []
                original_images = []
                for d in batch:
                    original_img = _ensure_pil(d.image)
                    original_images.append(original_img)
                    cropped, meta = _crop_empty(original_img, config=crop_cfg)
                    images.append(cropped)
                    crop_metas.append(meta)
            else:
                original_images = [_ensure_pil(d.image) for d in batch]
                images = original_images
                crop_metas = [None for _ in batch]
            try:
                embeddings, token_infos = embedder.embed_images(
                    images,
                    batch_size=batch_size,
                    return_token_info=True,
                    show_progress=False,
                )
            except Exception:
                # Retry per-doc to isolate flaky backend / corrupted sample issues.
                embeddings = []
                token_infos = []
                for doc_i, img_i, crop_meta_i in zip(batch, images, crop_metas):
                    try:
                        e1, t1 = embedder.embed_images(
                            [img_i],
                            batch_size=1,
                            return_token_info=True,
                            show_progress=False,
                        )
                        embeddings.append(e1[0])
                        token_infos.append(t1[0])
                    except Exception as e_single:
                        source_doc_id_i = str(
                            (doc_i.payload or {}).get("source_doc_id") or doc_i.doc_id
                        )
                        union_doc_id_i = _union_point_id(
                            dataset_name=dataset_name,
                            source_doc_id=source_doc_id_i,
                            union_namespace=union_namespace,
                        )
                        if str(union_doc_id_i) not in failed_ids:
                            _append_jsonl(
                                failure_log,
                                {
                                    "dataset": dataset_name,
                                    "collection": collection_name,
                                    "model": model_name,
                                    "source_doc_id": str(source_doc_id_i),
                                    "doc_id": str(getattr(doc_i, "doc_id", "")),
                                    "union_doc_id": str(union_doc_id_i),
                                    "error": str(e_single),
                                },
                            )
                            failed_ids.add(str(union_doc_id_i))
                        existing_ids.add(str(union_doc_id_i))
                        skipped_docs += 1
                if pbar is not None:
                    pbar.update(batch_total)
                continue

            for doc, emb, token_info, crop_meta, original_img, embed_img in zip(
                batch, embeddings, token_infos, crop_metas, original_images, images
            ):
                source_doc_id = str((doc.payload or {}).get("source_doc_id") or doc.doc_id)
                union_doc_id = _union_point_id(
                    dataset_name=dataset_name,
                    source_doc_id=source_doc_id,
                    union_namespace=union_namespace,
                )

                try:
                    emb_np = (
                        emb.cpu().float().numpy()
                        if hasattr(emb, "cpu")
                        else np.array(emb, dtype=np.float32)
                    )
                    visual_indices = token_info.get("visual_token_indices") or list(
                        range(emb_np.shape[0])
                    )
                    visual_embedding = emb_np[visual_indices].astype(np.float32)

                    tv = max_mean_pool_vectors
                    if tv is not None:
                        try:
                            tv_i = int(tv)
                            tv = None if tv_i <= 0 else tv_i
                        except Exception:
                            tv = 32
                    tile_pooled = embedder.mean_pool_visual_embedding(
                        visual_embedding, token_info, target_vectors=tv
                    )

                    experimental_pooled_by_name: Dict[str, Any] = {}
                    if is_colqwen25:
                        exp_gaussian = embedder.experimental_pool_visual_embedding(
                            visual_embedding,
                            token_info,
                            target_vectors=tv,
                            mean_pool=tile_pooled,
                            window_size=3,
                            kernel="gaussian",
                        )
                        exp_triangular = embedder.experimental_pool_visual_embedding(
                            visual_embedding,
                            token_info,
                            target_vectors=tv,
                            mean_pool=tile_pooled,
                            window_size=3,
                            kernel="triangular",
                        )
                        experimental_pooled_by_name["experimental_pooling"] = exp_gaussian
                        experimental_pooled_by_name["experimental_pooling_gaussian"] = exp_gaussian
                        experimental_pooled_by_name["experimental_pooling_triangular"] = (
                            exp_triangular
                        )
                    else:
                        canonical_k = int(ks_norm[0]) if ks_norm else 3
                        for k in ks_norm:
                            exp = embedder.experimental_pool_visual_embedding(
                                visual_embedding,
                                token_info,
                                target_vectors=tv,
                                mean_pool=tile_pooled,
                                window_size=int(k),
                                kernel=kernel,
                            )
                            experimental_pooled_by_name[f"experimental_pooling_{int(k)}"] = exp
                            if int(k) == int(canonical_k):
                                experimental_pooled_by_name["experimental_pooling"] = exp

                    if is_colsmol and bool(colsmol_experimental_2d):
                        try:
                            from visual_rag.embedding.pooling import (
                                colsmol_tile_4n_pooling_from_tiles,
                            )

                            n_rows = (token_info or {}).get("n_rows")
                            n_cols = (token_info or {}).get("n_cols")
                            if n_rows and n_cols:
                                exp2d = colsmol_tile_4n_pooling_from_tiles(
                                    tile_pooled,
                                    n_rows=int(n_rows),
                                    n_cols=int(n_cols),
                                    has_global=True,
                                    include_self=True,
                                    output_dtype=embedder.output_dtype,
                                )
                                experimental_pooled_by_name["experimental_pooling_2d"] = exp2d
                        except Exception:
                            pass

                    global_pooled = embedder.global_pool_from_mean_pool(tile_pooled)

                    # Log whenever ColQwen2.5 adaptive mean pooling actually downsamples rows.
                    if is_colqwen25:
                        grid_h_eff = (token_info or {}).get("grid_h_eff")
                        if grid_h_eff is not None:
                            try:
                                h_eff = int(grid_h_eff)
                                out_rows = int(getattr(tile_pooled, "shape", [0])[0])
                            except Exception:
                                h_eff = 0
                                out_rows = 0
                            if h_eff > 0 and out_rows > 0 and out_rows < h_eff:
                                tv_disp = (
                                    int(max_mean_pool_vectors)
                                    if max_mean_pool_vectors is not None
                                    else None
                                )
                                msg = (
                                    "Downsampled ColQwen mean-pool rows for "
                                    f"union_doc_id={union_doc_id} (source_doc_id={source_doc_id}): "
                                    f"grid_h_eff={h_eff} -> {out_rows} "
                                    f"(--max-mean-pool-vectors={tv_disp})"
                                )
                                if pbar is not None:
                                    try:
                                        pbar.write(msg)
                                    except Exception:
                                        print(msg)
                                else:
                                    print(msg)
                except Exception as e_single:
                    if str(union_doc_id) not in failed_ids:
                        _append_jsonl(
                            failure_log,
                            {
                                "dataset": dataset_name,
                                "collection": collection_name,
                                "model": model_name,
                                "source_doc_id": str(source_doc_id),
                                "doc_id": str(getattr(doc, "doc_id", "")),
                                "union_doc_id": str(union_doc_id),
                                "error": str(e_single),
                            },
                        )
                        failed_ids.add(str(union_doc_id))
                    existing_ids.add(str(union_doc_id))
                    skipped_docs += 1
                    continue

                num_tiles = int(tile_pooled.shape[0])
                patches_per_tile = (
                    int(visual_embedding.shape[0] // max(num_tiles, 1)) if num_tiles else 0
                )

                resized_img = _resized_for_display(embed_img) or embed_img
                original_url = ""
                cropped_url = ""
                resized_url = ""
                if (
                    cloudinary_uploader is not None
                    and original_img is not None
                    and resized_img is not None
                ):
                    base_public_id = _safe_public_id(f"{dataset_name}__{union_doc_id}")
                    try:
                        if crop_empty:
                            o_url, c_url, r_url = (
                                cloudinary_uploader.upload_original_cropped_and_resized(
                                    original_img,
                                    (
                                        embed_img
                                        if embed_img is not None and embed_img is not original_img
                                        else None
                                    ),
                                    resized_img,
                                    base_public_id,
                                )
                            )
                            original_url = o_url or ""
                            cropped_url = c_url or ""
                            resized_url = r_url or ""
                        else:
                            o_url, r_url = cloudinary_uploader.upload_original_and_resized(
                                original_img,
                                resized_img,
                                base_public_id,
                            )
                            original_url = o_url or ""
                            resized_url = r_url or ""
                    except Exception:
                        pass

                payload = {
                    "dataset": dataset_name,
                    "doc_id": doc.doc_id,
                    "union_doc_id": union_doc_id,
                    "page": resized_url or original_url or "",
                    "original_url": original_url,
                    "cropped_url": cropped_url if crop_empty else "",
                    "resized_url": resized_url,
                    "original_width": int(original_img.width) if original_img is not None else None,
                    "original_height": (
                        int(original_img.height) if original_img is not None else None
                    ),
                    "cropped_width": (
                        int(embed_img.width) if (crop_empty and embed_img is not None) else None
                    ),
                    "cropped_height": (
                        int(embed_img.height) if (crop_empty and embed_img is not None) else None
                    ),
                    "resized_width": int(resized_img.width) if resized_img is not None else None,
                    "resized_height": int(resized_img.height) if resized_img is not None else None,
                    "num_tiles": int(num_tiles),
                    "patches_per_tile": int(patches_per_tile),
                    "torch_dtype": _torch_dtype_to_str(embedder.torch_dtype),
                    "model_name": model_name,
                    "crop_empty_enabled": bool(crop_empty),
                    "crop_empty_crop_box": (
                        (crop_meta or {}).get("crop_box") if crop_empty else None
                    ),
                    "crop_empty_remove_page_number": (
                        bool(crop_empty_remove_page_number) if crop_empty else None
                    ),
                    "crop_empty_percentage_to_remove": (
                        float(crop_empty_percentage_to_remove) if crop_empty else None
                    ),
                    "index_recovery_previously_failed": bool(union_doc_id in previously_failed_ids),
                    "index_recovery_mode": (
                        "only_failures"
                        if bool(only_failures)
                        else ("retry_failures" if bool(retry_failures) else None)
                    ),
                    "index_recovery_pooling_inferred_tiles": bool(
                        (token_info or {}).get("num_tiles") is None
                        and (token_info or {}).get("n_rows") is None
                        and (token_info or {}).get("n_cols") is None
                    ),
                    "index_recovery_num_visual_tokens": int(visual_embedding.shape[0]),
                    "experimental_pooling_windows": ks_norm,
                    "experimental_pooling_default_window": int(ks_norm[0]) if ks_norm else None,
                    "experimental_pooling_kernel": str(kernel),
                    "colsmol_experimental_2d": (
                        bool(colsmol_experimental_2d) if is_colsmol else None
                    ),
                    "max_mean_pool_vectors": (
                        int(max_mean_pool_vectors) if max_mean_pool_vectors is not None else None
                    ),
                    **(doc.payload or {}),
                }

                points_buffer.append(
                    {
                        "id": union_doc_id,
                        "visual_embedding": visual_embedding,
                        "tile_pooled_embedding": tile_pooled,
                        "experimental_pooled_embedding": experimental_pooled_by_name,
                        "global_pooled_embedding": global_pooled,
                        "metadata": payload,
                    }
                )

                if len(points_buffer) >= upload_batch_size:
                    chunk = points_buffer
                    points_buffer = []
                    if executor is None:
                        uploaded_docs += int(_upload(chunk) or 0)
                    else:
                        futures.append(executor.submit(_upload, chunk))
                        _drain(block=len(futures) >= int(upload_workers) * 2)

            if pbar is not None:
                pbar.update(len(batch))
                now = time.time()
                done_docs = int(pbar.n)
                elapsed = max(now - start_time, 1e-9)
                avg_s_per_doc = elapsed / max(done_docs, 1)
                delta_docs = done_docs - last_tick_docs
                delta_t = max(now - last_tick_time, 1e-9)
                if delta_docs > 0:
                    last_s_per_doc = delta_t / delta_docs
                    last_tick_time = now
                    last_tick_docs = done_docs
                pbar.set_postfix(
                    {
                        "avg_s/doc": f"{avg_s_per_doc:.2f}",
                        "last_s/doc": f"{last_s_per_doc:.2f}",
                        "upl": uploaded_docs,
                        "skip": skipped_docs,
                    }
                )

            if executor is not None:
                _drain(block=False)
    except KeyboardInterrupt:
        stop_event.set()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        raise

    if points_buffer:
        if executor is None:
            uploaded_docs += int(_upload(points_buffer) or 0)
        else:
            futures.append(executor.submit(_upload, points_buffer))

    if executor is not None:
        _drain(block=True)
        executor.shutdown(wait=True)

    if pbar is not None:
        pbar.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--collection", type=str, required=True)
    parser.add_argument("--model", type=str, default="vidore/colSmol-500M")
    parser.add_argument(
        "--processor-speed",
        type=str,
        default="fast",
        choices=["fast", "slow", "auto"],
        help="Processor implementation: fast (default, with fallback to slow), slow, or auto.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--qdrant-vector-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32"],
    )
    grpc_group = parser.add_mutually_exclusive_group()
    grpc_group.add_argument("--prefer-grpc", dest="prefer_grpc", action="store_true", default=True)
    grpc_group.add_argument("--no-prefer-grpc", dest="prefer_grpc", action="store_false")
    parser.add_argument("--index", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--indexing-threshold", type=int, default=0)
    parser.add_argument("--full-scan-threshold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--upload-batch-size", type=int, default=8)
    parser.add_argument("--upload-workers", type=int, default=0)
    parser.add_argument("--upsert-wait", action="store_true")
    parser.add_argument("--max-corpus-docs", type=int, default=0)
    parser.add_argument("--sample-corpus-docs", type=int, default=0)
    parser.add_argument(
        "--sample-corpus-strategy", type=str, default="first", choices=["first", "random"]
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--sample-queries", type=int, default=0)
    parser.add_argument(
        "--sample-query-strategy", type=str, default="first", choices=["first", "random"]
    )
    parser.add_argument("--sample-query-seed", type=int, default=0)
    parser.add_argument("--index-from-queries", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--qdrant-timeout", type=int, default=120)
    parser.add_argument("--qdrant-retries", type=int, default=3)
    parser.add_argument("--qdrant-retry-sleep", type=float, default=0.5)
    parser.add_argument("--crop-empty", action="store_true", default=False)
    parser.add_argument("--crop-empty-percentage-to-remove", type=float, default=0.9)
    parser.add_argument("--crop-empty-remove-page-number", action="store_true", default=False)
    parser.add_argument("--crop-empty-preserve-border-px", type=int, default=1)
    parser.add_argument("--crop-empty-uniform-std-threshold", type=float, default=0.0)
    parser.add_argument(
        "--max-mean-pool-vectors",
        type=int,
        default=32,
        help=(
            "Cap ColQwen2.5 adaptive row-mean pooling to at most this many vectors. "
            "Default: 32 (legacy behavior). "
            "If <= 0, treated as no cap (use all effective rows)."
        ),
    )
    parser.add_argument(
        "--pooling-windows",
        "--pooling_windows",
        type=int,
        nargs="+",
        default=None,
        help=(
            "ColPali only: experimental pooling window size(s). Provide one int to override the default window, "
            "or multiple ints to index/store multiple experimental vectors. "
            "When multiple are provided, vectors are stored as 'experimental_pooling_{k}' and "
            "the canonical 'experimental_pooling' aliases the first provided k."
        ),
    )
    parser.add_argument(
        "--experimental-pooling-kernel",
        "--experimental_pooling_kernel",
        type=str,
        default="auto",
        choices=["auto", "legacy", "uniform", "triangular", "gaussian"],
        help=(
            "Experimental pooling kernel. "
            "'legacy' uses the historical ColPali conv-style pooling (N->N+2r; default for ColPali). "
            "'gaussian'/'triangular'/'uniform' use weighted same-length smoothing (N->N). "
            "Ignored for ColQwen2.5 indexing (which stores gaussian+triangular variants with k=3)."
        ),
    )
    parser.add_argument(
        "--colsmol-experimental-2d",
        "--colsmol_experimental_2d",
        action="store_true",
        default=False,
        help="For ColSmol indexing, also store 2D 4-neighborhood experimental pooling as 'experimental_pooling_2d'.",
    )
    payload_group = parser.add_mutually_exclusive_group()
    payload_group.add_argument("--index-common-metadata", action="store_true", default=True)
    payload_group.add_argument(
        "--no-index-common-metadata", dest="index_common_metadata", action="store_false"
    )
    parser.add_argument("--payload-index", action="append", default=[])
    cloud_group = parser.add_mutually_exclusive_group()
    cloud_group.add_argument(
        "--cloudinary",
        dest="no_cloudinary",
        action="store_false",
        default=True,
        help="Enable Cloudinary uploads during indexing (default: disabled).",
    )
    cloud_group.add_argument(
        "--no-cloudinary",
        dest="no_cloudinary",
        action="store_true",
        help="Disable Cloudinary uploads during indexing (default).",
    )
    parser.add_argument(
        "--cloudinary-folder",
        type=str,
        default="vidore-beir",
        help="Cloudinary base folder for uploads (default: vidore-beir).",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        default=False,
        help="On --resume, retry documents listed in index_failures__<collection>__<dataset>.jsonl (default: skip them).",
    )
    parser.add_argument(
        "--only-failures",
        action="store_true",
        default=False,
        help="Index only documents listed in index_failures__<collection>__<dataset>.jsonl.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Retrieve top-k results (default: 100 to calculate metrics at all cutoffs)",
    )
    parser.add_argument(
        "--prefetch-k",
        type=int,
        default=200,
        help="Prefetch candidates for two-stage (default: 200)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        default=False,
        help="If set, run indexing only and skip evaluation.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single_full",
        choices=[
            "single_full",
            "single_tiles",
            "single_global",
            "single_experimental_tokens",
            "single_experimental_pooled",
            "two_stage",
            "three_stage",
        ],
    )
    parser.add_argument(
        "--stage1-mode",
        type=str,
        default="tokens_vs_standard_pooling",
        choices=[
            # New naming (preferred)
            "pooled_query_vs_standard_pooling",
            "tokens_vs_standard_pooling",
            "pooled_query_vs_experimental_pooling",
            "tokens_vs_experimental_pooling",
            "pooled_query_vs_global",
            # Backwards-compatible aliases (deprecated)
            "pooled_query_vs_tiles",
            "tokens_vs_tiles",
            "pooled_query_vs_experimental",
            "tokens_vs_experimental",
        ],
        help=(
            "Two-stage stage1 prefetch mode. "
            "standard_pooling uses Qdrant named vector 'mean_pooling'. "
            "experimental_pooling uses Qdrant named vector 'experimental_pooling'. "
            "global uses Qdrant named vector 'global_pooling'."
        ),
    )
    parser.add_argument(
        "--experimental-pooling-k",
        "--experimental_pooling_k",
        type=int,
        default=None,
        help=(
            "ColPali only: when using an experimental stage1-mode, select which indexed experimental vector to use "
            "(Qdrant named vector: 'experimental_pooling_{k}'). If omitted, uses 'experimental_pooling'."
        ),
    )
    parser.add_argument(
        "--experimental-pooling-technique",
        "--experimental_pooling_technique",
        type=str,
        default=None,
        choices=["gaussian", "triangular"],
        help=(
            "ColQwen only: select which experimental pooling technique to use for stage-1/single-stage experimental "
            "modes. Maps to Qdrant named vectors: 'experimental_pooling_gaussian' or 'experimental_pooling_triangular'. "
            "If omitted, uses 'experimental_pooling' (Gaussian alias)."
        ),
    )
    parser.add_argument(
        "--stage1-k", type=int, default=1000, help="Three-stage stage1 top_k (default: 1000)"
    )
    parser.add_argument(
        "--stage2-k", type=int, default=300, help="Three-stage stage2 top_k (default: 300)"
    )
    parser.add_argument("--max-queries", type=int, default=0)
    drop_group = parser.add_mutually_exclusive_group()
    drop_group.add_argument(
        "--drop-empty-queries", dest="drop_empty_queries", action="store_true", default=True
    )
    drop_group.add_argument(
        "--no-drop-empty-queries", dest="drop_empty_queries", action="store_false"
    )
    parser.add_argument(
        "--evaluation-scope",
        type=str,
        default="union",
        choices=["union", "per_dataset"],
        help="Evaluation scope: 'union' searches over the whole collection (cross-dataset distractors). "
        "'per_dataset' applies a Qdrant filter so each dataset's queries search only its own subset (leaderboard-comparable).",
    )
    cont_group = parser.add_mutually_exclusive_group()
    cont_group.add_argument(
        "--continue-on-error",
        dest="continue_on_error",
        action="store_true",
        default=False,
        help="Continue evaluating remaining datasets if one dataset fails (default: false).",
    )
    cont_group.add_argument(
        "--no-continue-on-error",
        dest="continue_on_error",
        action="store_false",
        help="Stop the run immediately on the first dataset evaluation failure (default).",
    )
    parser.add_argument("--output", type=str, default="auto")
    parser.add_argument(
        "--ensure-in-ram",
        dest="ensure_in_ram",
        action="store_true",
        default=False,
        help="Best-effort: patch collection config so vectors/indexes are stored in RAM (on_disk=false).",
    )

    args = parser.parse_args()

    # Backwards-compatible stage1_mode mapping (deprecated names)
    stage1_map = {
        "pooled_query_vs_tiles": "pooled_query_vs_standard_pooling",
        "tokens_vs_tiles": "tokens_vs_standard_pooling",
        "pooled_query_vs_experimental": "pooled_query_vs_experimental_pooling",
        "tokens_vs_experimental": "tokens_vs_experimental_pooling",
    }
    if str(args.stage1_mode) in stage1_map:
        old = str(args.stage1_mode)
        new = stage1_map[old]
        warnings.warn(
            f"--stage1-mode {old} is deprecated; use {new} instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        args.stage1_mode = new

    _maybe_load_dotenv()

    if args.recreate:
        args.index = True

    if (
        args.sample_corpus_docs
        and int(args.sample_corpus_docs) > 0
        and args.max_corpus_docs
        and int(args.max_corpus_docs) > 0
    ):
        raise ValueError("Use only one of --sample-corpus-docs or --max-corpus-docs (not both).")
    if args.sample_queries and int(args.sample_queries) > 0 and args.index_from_queries:
        if (args.sample_corpus_docs and int(args.sample_corpus_docs) > 0) or (
            args.max_corpus_docs and int(args.max_corpus_docs) > 0
        ):
            raise ValueError(
                "Use --index-from-queries with --sample-queries only (do not combine with corpus sampling)."
            )

    if args.upsert_wait:
        print("Qdrant upserts wait for completion (wait=True).")
    else:
        print("Qdrant upserts are async (wait=False).")
    print(
        f"Qdrant request timeout: {int(args.qdrant_timeout)}s, retries: {int(args.qdrant_retries)}."
    )

    datasets: List[str] = []
    if args.datasets:
        datasets = list(args.datasets)
    elif args.dataset:
        datasets = [args.dataset]
    else:
        raise ValueError("Provide --dataset (single) or --datasets (one or more)")

    if str(args.output).strip().lower() == "auto":
        args.output = _default_output_filename(args=args, datasets=datasets)

    loaded: List[Tuple[str, Any, Any, Dict[str, Dict[str, int]]]] = []
    for ds_name in datasets:
        corpus, queries, qrels = load_vidore_beir_dataset(ds_name)
        loaded.append((ds_name, corpus, queries, qrels))

    # Resolve the dtype used for query embeddings:
    # - If user sets float16/float32 explicitly, respect it.
    # - If auto: try to detect from the existing Qdrant collection (prevents silent score drops),
    #            otherwise fall back to float16 (preserves legacy default for new collections).
    effective_qdrant_vector_dtype = str(args.qdrant_vector_dtype)
    if effective_qdrant_vector_dtype == "auto":
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        detected = None
        if qdrant_url:
            try:
                from qdrant_client import QdrantClient

                client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    prefer_grpc=bool(args.prefer_grpc),
                    timeout=int(args.qdrant_timeout),
                    check_compatibility=False,
                )
                detected = _detect_collection_vector_dtype(
                    client=client, collection_name=str(args.collection)
                )
            except Exception:
                detected = None
        effective_qdrant_vector_dtype = detected or "float16"
        if detected:
            print(f"🔎 Detected Qdrant vector dtype for collection: {detected}")
        else:
            print("🔎 Could not detect Qdrant vector dtype; defaulting to float16")
        sys.stdout.flush()

    output_dtype = np.float16 if effective_qdrant_vector_dtype == "float16" else np.float32
    embedder = VisualEmbedder(
        model_name=args.model,
        batch_size=args.batch_size,
        torch_dtype=_parse_torch_dtype(args.torch_dtype),
        output_dtype=output_dtype,
        processor_speed=str(args.processor_speed),
    )

    selected: List[Tuple[str, Any, Any, Dict[str, Dict[str, int]]]] = []
    for ds_name, corpus, queries, qrels in loaded:
        if args.sample_queries and int(args.sample_queries) > 0:
            queries = _sample_list(
                list(queries),
                k=int(args.sample_queries),
                strategy=str(args.sample_query_strategy),
                seed=int(args.sample_query_seed),
            )
            qrels = _filter_qrels(qrels, [q.query_id for q in queries])

        if args.index_from_queries and args.sample_queries and int(args.sample_queries) > 0:
            rel_doc_ids = set()
            for q in queries:
                for did, score in qrels.get(q.query_id, {}).items():
                    if score > 0:
                        rel_doc_ids.add(str(did))
            corpus = [d for d in corpus if str(d.doc_id) in rel_doc_ids]
        else:
            if args.sample_corpus_docs and int(args.sample_corpus_docs) > 0:
                corpus = _sample_list(
                    list(corpus),
                    k=int(args.sample_corpus_docs),
                    strategy=str(args.sample_corpus_strategy),
                    seed=int(args.sample_seed),
                )
            elif args.max_corpus_docs and int(args.max_corpus_docs) > 0:
                corpus = corpus[: int(args.max_corpus_docs)]

        selected.append((ds_name, corpus, queries, qrels))

    payload_indexes: List[Dict[str, str]] = []
    if args.index:
        payload_indexes = _parse_payload_indexes(args.payload_index)
        if args.index_common_metadata:
            payload_indexes.extend(
                [
                    {"field": "dataset", "type": "keyword"},
                    {"field": "source_doc_id", "type": "keyword"},
                    {"field": "doc_id", "type": "keyword"},
                    {"field": "filename", "type": "keyword"},
                    {"field": "page", "type": "keyword"},
                    {"field": "page_number", "type": "integer"},
                    {"field": "total_pages", "type": "integer"},
                    {"field": "has_text", "type": "bool"},
                    {"field": "text", "type": "text"},
                    {"field": "original_url", "type": "keyword"},
                    {"field": "resized_url", "type": "keyword"},
                    {"field": "original_width", "type": "integer"},
                    {"field": "original_height", "type": "integer"},
                    {"field": "resized_width", "type": "integer"},
                    {"field": "resized_height", "type": "integer"},
                    {"field": "num_tiles", "type": "integer"},
                    {"field": "tile_rows", "type": "integer"},
                    {"field": "tile_cols", "type": "integer"},
                    {"field": "patches_per_tile", "type": "integer"},
                    {"field": "num_visual_tokens", "type": "integer"},
                    {"field": "processor_version", "type": "keyword"},
                    {"field": "year", "type": "integer"},
                    {"field": "source", "type": "keyword"},
                ]
            )
            # Keep schema minimal: only add crop-related indexes when cropping is enabled.
            if bool(args.crop_empty):
                payload_indexes.extend(
                    [
                        {"field": "crop_empty_enabled", "type": "bool"},
                        {"field": "crop_empty_remove_page_number", "type": "bool"},
                        {"field": "crop_empty_percentage_to_remove", "type": "float"},
                        {"field": "cropped_url", "type": "keyword"},
                        {"field": "cropped_width", "type": "integer"},
                        {"field": "cropped_height", "type": "integer"},
                    ]
                )
        # If we are recreating the collection, clear historical failure logs so they don't
        # remove valid qrels during evaluation.
        if bool(args.recreate):
            for ds_name, _corpus, _queries, _qrels in selected:
                p = _failed_log_path(collection_name=args.collection, dataset_name=ds_name)
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
        for i, (ds_name, corpus, queries, _qrels) in enumerate(selected):
            print(f"Indexing {ds_name}: corpus_docs={len(corpus)} queries={len(queries)}")
            _index_beir_corpus(
                dataset_name=ds_name,
                corpus=corpus,
                embedder=embedder,
                collection_name=args.collection,
                prefer_grpc=args.prefer_grpc,
                qdrant_vector_dtype=effective_qdrant_vector_dtype,
                recreate=bool(args.recreate and i == 0),
                indexing_threshold=args.indexing_threshold,
                batch_size=args.batch_size,
                upload_batch_size=args.upload_batch_size,
                upload_workers=args.upload_workers,
                upsert_wait=bool(args.upsert_wait),
                max_corpus_docs=0,
                sample_corpus_docs=0,
                sample_corpus_strategy=str(args.sample_corpus_strategy),
                sample_seed=int(args.sample_seed),
                payload_indexes=payload_indexes,
                union_namespace=args.collection,
                model_name=args.model,
                resume=bool(args.resume),
                qdrant_timeout=int(args.qdrant_timeout),
                full_scan_threshold=int(args.full_scan_threshold),
                crop_empty=bool(args.crop_empty),
                crop_empty_percentage_to_remove=float(args.crop_empty_percentage_to_remove),
                crop_empty_remove_page_number=bool(args.crop_empty_remove_page_number),
                crop_empty_preserve_border_px=int(args.crop_empty_preserve_border_px),
                crop_empty_uniform_std_threshold=float(args.crop_empty_uniform_std_threshold),
                max_mean_pool_vectors=args.max_mean_pool_vectors,
                pooling_windows=args.pooling_windows,
                experimental_pooling_kernel=str(args.experimental_pooling_kernel),
                colsmol_experimental_2d=bool(args.colsmol_experimental_2d),
                no_cloudinary=bool(args.no_cloudinary),
                cloudinary_folder=str(args.cloudinary_folder),
                retry_failures=bool(args.retry_failures),
                only_failures=bool(args.only_failures),
            )

    out_path = _resolve_output_path(args.output, collection_name=str(args.collection))

    if bool(args.no_eval):
        dataset_index_failures: Dict[str, Dict[str, Any]] = {}
        dataset_counts: Dict[str, Dict[str, int]] = {}
        for ds_name, corpus, queries, _qrels in selected:
            dataset_counts[ds_name] = {
                "corpus_docs": int(len(corpus)),
                "queries": int(len(queries)),
                "queries_eval": 0,
            }
            failed_path = _failed_log_path(collection_name=args.collection, dataset_name=ds_name)
            failed_ids = _load_failed_union_ids(
                failed_path, dataset_name=ds_name, union_namespace=args.collection
            )
            dataset_index_failures[ds_name] = {
                "failed_log_path": str(failed_path),
                "failed_ids_count": int(len(failed_ids)),
                "qrels_removed": None,
            }
        _write_json_atomic(
            out_path,
            {
                "command": " ".join(sys.argv),
                "dataset": datasets[0] if len(datasets) == 1 else None,
                "datasets": datasets,
                "protocol": "beir",
                "collection": args.collection,
                "model": args.model,
                "torch_dtype": _torch_dtype_to_str(embedder.torch_dtype),
                "qdrant_vector_dtype": effective_qdrant_vector_dtype,
                "mode": args.mode,
                "stage1_mode": args.stage1_mode if args.mode == "two_stage" else None,
                "prefetch_k": args.prefetch_k if args.mode == "two_stage" else None,
                "top_k": args.top_k,
                "evaluation_scope": str(args.evaluation_scope),
                "dataset_counts": dataset_counts,
                "dataset_errors": {},
                "dataset_index_failures": dataset_index_failures,
                "qdrant_timeout": int(args.qdrant_timeout),
                "qdrant_retries": int(args.qdrant_retries),
                "qdrant_retry_sleep": float(args.qdrant_retry_sleep),
                "full_scan_threshold": int(args.full_scan_threshold),
                "no_eval": True,
                "eval_wall_time_s": None,
                "metrics": None,
                "metrics_by_dataset": {},
            },
        )
        print(f"Wrote index-only report: {out_path}")
        return

    if bool(args.ensure_in_ram):
        try:
            from visual_rag.qdrant_admin import QdrantAdmin

            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            admin = QdrantAdmin(
                url=qdrant_url,
                api_key=qdrant_api_key,
                prefer_grpc=bool(args.prefer_grpc),
                timeout=int(args.qdrant_timeout),
            )
            print(f"🧠 Ensuring collection in RAM (config): {args.collection}")
            sys.stdout.flush()
            info_after = admin.ensure_collection_all_in_ram(
                collection_name=str(args.collection),
                timeout=int(args.qdrant_timeout),
            )
            # Verify by printing current on_disk flags (what the UI reads)
            try:
                vectors = (((info_after or {}).get("config") or {}).get("params") or {}).get(
                    "vectors"
                ) or {}
                if isinstance(vectors, dict):
                    vec_flags = {}
                    for name, cfg in vectors.items():
                        if not isinstance(cfg, dict):
                            continue
                        vec_flags[str(name)] = (
                            bool(cfg.get("on_disk")) if cfg.get("on_disk") is not None else None
                        )
                else:
                    vec_flags = {}
                hnsw_on_disk = (
                    ((info_after or {}).get("config") or {}).get("hnsw_config") or {}
                ).get("on_disk")
                on_disk_payload = (
                    ((info_after or {}).get("config") or {}).get("params") or {}
                ).get("on_disk_payload")
            except Exception:
                vec_flags = {}
                hnsw_on_disk = None
                on_disk_payload = None

            print("✅ ensure-in-ram config applied. Verified config:")
            if hnsw_on_disk is not None:
                print(f"   collection.hnsw_config.on_disk={hnsw_on_disk}")
            if on_disk_payload is not None:
                print(f"   collection.params.on_disk_payload={on_disk_payload}")
            if vec_flags:
                for n in sorted(vec_flags.keys()):
                    print(f"   vector[{n}].on_disk={vec_flags[n]}")
            sys.stdout.flush()
        except Exception as e:
            print(f"⚠️ ensure-in-ram failed: {type(e).__name__}: {e}")
            sys.stdout.flush()

    def _is_colqwen_model(model_name: str) -> bool:
        return "colqwen" in str(model_name).lower()

    exp_vector_name = "experimental_pooling"
    uses_experimental_vector = (
        str(args.stage1_mode)
        in ("pooled_query_vs_experimental_pooling", "tokens_vs_experimental_pooling")
        and str(args.mode) in ("two_stage", "three_stage")
    ) or str(args.mode) in ("single_experimental_tokens", "single_experimental_pooled")

    if (
        getattr(args, "experimental_pooling_technique", None)
        and getattr(args, "experimental_pooling_k", None) is not None
    ):
        raise ValueError(
            "Use only one of --experimental-pooling-technique or --experimental-pooling-k (not both)."
        )

    if uses_experimental_vector and getattr(args, "experimental_pooling_technique", None):
        if not _is_colqwen_model(args.model):
            raise ValueError(
                "--experimental-pooling-technique is only supported for ColQwen models."
            )
        exp_vector_name = (
            f"experimental_pooling_{str(args.experimental_pooling_technique).strip().lower()}"
        )

    if uses_experimental_vector and getattr(args, "experimental_pooling_k", None) is not None:
        if _is_colqwen_model(args.model):
            raise ValueError(
                "--experimental-pooling-k is intended for ColPali (experimental_pooling_{k}), not ColQwen."
            )
        exp_vector_name = f"experimental_pooling_{int(args.experimental_pooling_k)}"

    retriever = MultiVectorRetriever(
        collection_name=args.collection,
        embedder=embedder,
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=(os.getenv("QDRANT_API_KEY")),
        prefer_grpc=args.prefer_grpc,
        request_timeout=int(args.qdrant_timeout),
        max_retries=int(args.qdrant_retries),
        retry_sleep=float(args.qdrant_retry_sleep),
        experimental_vector_name=exp_vector_name,
    )

    if str(args.mode) in ("single_experimental_tokens", "single_experimental_pooled") or (
        str(args.stage1_mode)
        in ("pooled_query_vs_experimental_pooling", "tokens_vs_experimental_pooling")
        and str(args.mode) in ("two_stage", "three_stage")
    ):
        existing_vectors = _collection_vector_names(
            client=retriever.client, collection_name=str(args.collection)
        )
        if exp_vector_name not in existing_vectors:
            candidates = sorted(
                [v for v in existing_vectors if str(v).startswith("experimental_pooling")]
            )
            raise ValueError(
                f"Requested experimental vector '{exp_vector_name}' is not present in the collection. "
                f"Available experimental vectors: {candidates or '[]'}. "
                "Re-index with --pooling-windows (and --recreate) to add it."
            )

    metrics_by_dataset: Dict[str, Dict[str, float]] = {}
    dataset_errors: Dict[str, str] = {}
    dataset_counts: Dict[str, Dict[str, int]] = {}
    dataset_index_failures: Dict[str, Dict[str, Any]] = {}
    eval_started_at = time.time()

    def _build_run_record() -> Dict[str, Any]:
        single_dataset = datasets[0] if len(datasets) == 1 else None
        single_metrics = metrics_by_dataset.get(single_dataset) if single_dataset else None
        run_cmd = " ".join(sys.argv)
        return {
            "command": run_cmd,
            "dataset": single_dataset,
            "datasets": datasets,
            "protocol": "beir",
            "collection": args.collection,
            "model": args.model,
            "torch_dtype": _torch_dtype_to_str(embedder.torch_dtype),
            "qdrant_vector_dtype": effective_qdrant_vector_dtype,
            "mode": args.mode,
            "stage1_mode": args.stage1_mode if args.mode == "two_stage" else None,
            "prefetch_k": args.prefetch_k if args.mode == "two_stage" else None,
            "stage1_k": int(args.stage1_k) if args.mode == "three_stage" else None,
            "stage2_k": int(args.stage2_k) if args.mode == "three_stage" else None,
            "top_k": args.top_k,
            "max_queries": args.max_queries,
            "max_corpus_docs": int(args.max_corpus_docs),
            "sample_corpus_docs": int(args.sample_corpus_docs),
            "sample_corpus_strategy": str(args.sample_corpus_strategy),
            "sample_seed": int(args.sample_seed),
            "sample_queries": int(args.sample_queries),
            "sample_query_strategy": str(args.sample_query_strategy),
            "sample_query_seed": int(args.sample_query_seed),
            "index_from_queries": bool(args.index_from_queries),
            "drop_empty_queries": bool(args.drop_empty_queries),
            "evaluation_scope": str(args.evaluation_scope),
            "payload_indexes": payload_indexes,
            "dataset_counts": dataset_counts,
            "dataset_errors": dataset_errors,
            "dataset_index_failures": dataset_index_failures,
            "qdrant_timeout": int(args.qdrant_timeout),
            "qdrant_retries": int(args.qdrant_retries),
            "qdrant_retry_sleep": float(args.qdrant_retry_sleep),
            "full_scan_threshold": int(args.full_scan_threshold),
            "max_mean_pool_vectors": (
                int(args.max_mean_pool_vectors) if args.max_mean_pool_vectors is not None else None
            ),
            "pooling_windows": args.pooling_windows,
            "experimental_pooling_k": args.experimental_pooling_k,
            "experimental_pooling_technique": getattr(args, "experimental_pooling_technique", None),
            "eval_wall_time_s": float(max(time.time() - eval_started_at, 0.0)),
            "metrics": single_metrics,
            "metrics_by_dataset": metrics_by_dataset,
        }

    for ds_name, corpus, queries, qrels in selected:
        print(
            f"Evaluating dataset={ds_name} "
            f"(corpus_docs={len(corpus)}, queries={len(queries)}) "
            f"scope={args.evaluation_scope} "
            f"mode={args.mode}"
            + (
                f", stage1_mode={args.stage1_mode}, prefetch_k={int(args.prefetch_k)}"
                if args.mode == "two_stage"
                else ""
            )
            + (
                f", stage1_k={int(args.stage1_k)}, stage2_k={int(args.stage2_k)}"
                if args.mode == "three_stage"
                else ""
            )
            + f", top_k={int(args.top_k)}"
        )
        sys.stdout.flush()

        dataset_counts[ds_name] = {
            "corpus_docs": int(len(corpus)),
            "queries": int(len(queries)),
            "queries_eval": 0,
        }
        id_map: Dict[str, str] = {}
        for doc in corpus:
            source_doc_id = str((doc.payload or {}).get("source_doc_id") or doc.doc_id)
            id_map[str(doc.doc_id)] = _union_point_id(
                dataset_name=ds_name,
                source_doc_id=source_doc_id,
                union_namespace=args.collection,
            )

        remapped_qrels: Dict[str, Dict[str, int]] = {}
        for qid, rels in qrels.items():
            out_rels: Dict[str, int] = {}
            for did, score in rels.items():
                mapped = id_map.get(str(did))
                if mapped:
                    out_rels[mapped] = int(score)
            if out_rels:
                remapped_qrels[qid] = out_rels

        failed_path = _failed_log_path(collection_name=args.collection, dataset_name=ds_name)
        failed_ids_all = _load_failed_union_ids(
            failed_path, dataset_name=ds_name, union_namespace=args.collection
        )
        # Only remove failed IDs that are actually missing in the current collection.
        failed_ids_missing = _filter_failed_ids_to_missing(
            qdrant_client=retriever.client,
            collection_name=str(args.collection),
            failed_ids=failed_ids_all,
            timeout=int(args.qdrant_timeout),
        )
        remapped_qrels, removed_rels = _remove_failed_from_qrels(remapped_qrels, failed_ids_missing)
        dataset_index_failures[ds_name] = {
            "failed_log_path": str(failed_path),
            "failed_ids_count": int(len(failed_ids_all)),
            "failed_ids_missing_count": int(len(failed_ids_missing)),
            "qrels_removed": int(removed_rels),
        }

        filter_obj = None
        if args.evaluation_scope == "per_dataset":
            from qdrant_client.http import models as qmodels

            filter_obj = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="dataset", match=qmodels.MatchValue(value=str(ds_name))
                    )
                ]
            )

        try:
            metrics_by_dataset[ds_name] = _evaluate(
                queries=queries,
                qrels=remapped_qrels,
                retriever=retriever,
                embedder=embedder,
                top_k=args.top_k,
                prefetch_k=args.prefetch_k,
                mode=args.mode,
                stage1_mode=args.stage1_mode,
                stage1_k=int(args.stage1_k),
                stage2_k=int(args.stage2_k),
                max_queries=int(args.max_queries),
                drop_empty_queries=bool(args.drop_empty_queries),
                filter_obj=filter_obj,
            )
            dataset_counts[ds_name]["queries_eval"] = int(
                metrics_by_dataset[ds_name].get("num_queries_eval", 0)
            )
            ds_only_out = {
                **_build_run_record(),
                "dataset": str(ds_name),
                "datasets": [str(ds_name)],
                "metrics": metrics_by_dataset[ds_name],
                "metrics_by_dataset": {str(ds_name): metrics_by_dataset[ds_name]},
            }
            per_ds_path = out_path.with_name(
                f"{out_path.stem}__{_safe_filename(ds_name)}{out_path.suffix}"
            )
            _write_json_atomic(per_ds_path, ds_only_out)
            print(f"Wrote dataset report: {per_ds_path}")
            print(json.dumps({str(ds_name): metrics_by_dataset[ds_name]}, indent=2))
            sys.stdout.flush()
        except Exception as e:
            dataset_errors[ds_name] = f"{type(e).__name__}: {e}"
            print(f"❌ Evaluation failed for dataset={ds_name}: {dataset_errors[ds_name]}")
            # Common cause for per_dataset: missing payload index for 'dataset'
            msg = str(e)
            if (
                'Index required but not found for "dataset"' in msg
                or "Index required but not found for 'dataset'" in msg
            ):
                print(
                    "   Hint: Qdrant requires a payload index for key 'dataset' to use per-dataset filtering.\n"
                    "   Fix: create payload index (keyword) on the collection, or rerun indexing with metadata indexes enabled."
                )
            sys.stdout.flush()
            if not bool(args.continue_on_error):
                _write_json_atomic(out_path, _build_run_record())
                raise
        finally:
            _write_json_atomic(out_path, _build_run_record())

    single_dataset = datasets[0] if len(datasets) == 1 else None
    single_metrics = metrics_by_dataset.get(single_dataset) if single_dataset else None
    if len(datasets) == 1 and single_metrics is not None:
        print(json.dumps(single_metrics, indent=2))
    else:
        print(json.dumps(metrics_by_dataset, indent=2))


if __name__ == "__main__":
    main()
