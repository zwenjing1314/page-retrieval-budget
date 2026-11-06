import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from benchmarks.vidore_tatdqa_test.dataset_loader import (
    load_vidore_dataset_auto,
    paired_doc_id,
    paired_payload,
)
from benchmarks.vidore_tatdqa_test.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from visual_rag import VisualEmbedder
from visual_rag.embedding.pooling import tile_level_mean_pooling
from visual_rag.indexing.qdrant_indexer import QdrantIndexer
from visual_rag.retrieval import MultiVectorRetriever


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


def _paired_collate(batch):
    idxs = [b[0] for b in batch]
    images = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    return idxs, images, metas


class _PairedHFDataset:
    def __init__(self, *, dataset_name: str, split: str, total_docs: int, image_col: str):
        self.dataset_name = dataset_name
        self.split = split
        self.total_docs = int(total_docs)
        self.image_col = image_col
        self._ds = None

    def __len__(self) -> int:
        return self.total_docs

    def _ensure_loaded(self):
        if self._ds is not None:
            return
        from datasets import load_dataset

        self._ds = load_dataset(self.dataset_name, split=self.split)

    def __getitem__(self, idx: int):
        self._ensure_loaded()
        row = self._ds[int(idx)]
        image = row[self.image_col]
        meta = {k: v for k, v in row.items() if k != self.image_col}
        return int(idx), image, meta


def _ensure_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing env var: {name}")
    return value


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if Path(".env").exists():
        load_dotenv(".env")


def _index_corpus(
    *,
    dataset_name: str,
    collection_name: str,
    corpus: List[Any],
    embedder: VisualEmbedder,
    qdrant_url: str,
    qdrant_api_key: Optional[str],
    prefer_grpc: bool,
    qdrant_vector_dtype: str,
    recreate: bool,
    batch_size: int,
    upload_batch_size: int,
    upload_workers: int,
    upsert_wait: bool,
    indexing_threshold: int,
    full_scan_threshold: int,
) -> None:
    indexer = QdrantIndexer(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        prefer_grpc=prefer_grpc,
        vector_datatype=qdrant_vector_dtype,
    )
    indexer.create_collection(
        force_recreate=recreate,
        indexing_threshold=indexing_threshold,
        full_scan_threshold=int(full_scan_threshold),
    )
    indexer.create_payload_indexes(
        fields=[
            {"field": "dataset", "type": "keyword"},
            {"field": "doc_id", "type": "keyword"},
            {"field": "torch_dtype", "type": "keyword"},
            {"field": "source", "type": "keyword"},
            {"field": "image_filename", "type": "keyword"},
            {"field": "page", "type": "keyword"},
            {"field": "source_doc_id", "type": "keyword"},
        ]
    )

    total_docs = len(corpus)
    embedded_docs = 0
    enqueued_docs = 0
    uploaded_docs = 0
    start_time = time.time()
    last_tick_time = start_time
    last_tick_docs = 0

    pbar = None
    try:
        from tqdm import tqdm

        pbar = tqdm(total=total_docs, desc="📦 Indexing", unit="doc")
    except ImportError:
        pass

    def _upload(points: List[Dict[str, Any]]) -> int:
        return indexer.upload_batch(
            points, delay_between_batches=0.0, wait=upsert_wait, stop_event=stop_event
        )

    executor = None
    futures = []
    import threading

    stop_event = threading.Event()
    if upload_workers and upload_workers > 0:
        from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor
        from concurrent.futures import wait as futures_wait

        executor = ThreadPoolExecutor(max_workers=upload_workers)

        def _drain(block: bool = False) -> None:
            nonlocal uploaded_docs
            nonlocal last_tick_time
            nonlocal last_tick_docs
            if not futures:
                return
            if block:
                done, _ = futures_wait(futures, return_when=FIRST_EXCEPTION)
            else:
                done, _ = futures_wait(futures, timeout=0, return_when=FIRST_EXCEPTION)
            for d in list(done):
                futures.remove(d)
                uploaded_docs += int(d.result() or 0)
            if pbar is not None:
                now = time.time()
                done_docs = int(pbar.n)
                elapsed = max(now - start_time, 1e-9)
                avg_s_per_doc = elapsed / max(done_docs, 1)
                delta_docs = done_docs - last_tick_docs
                delta_t = max(now - last_tick_time, 1e-9)
                last_s_per_doc = delta_t / max(delta_docs, 1)
                last_tick_time = now
                last_tick_docs = done_docs
                pbar.set_postfix(
                    {
                        "avg_s/doc": f"{avg_s_per_doc:.2f}",
                        "last_s/doc": f"{last_s_per_doc:.2f}",
                        "buffer": len(points_buffer),
                        "enq": enqueued_docs,
                        "upl": uploaded_docs,
                        "pending": len(futures),
                    }
                )

    points_buffer: List[Dict[str, Any]] = []
    try:
        for start in range(0, len(corpus), batch_size):
            batch = corpus[start : start + batch_size]
            images = [d.image for d in batch]
            embeddings, token_infos = embedder.embed_images(
                images,
                batch_size=batch_size,
                return_token_info=True,
                show_progress=False,
            )
            embedded_docs += len(batch)
            if pbar is not None:
                pbar.update(len(batch))
                now = time.time()
                done_docs = int(pbar.n)
                elapsed = max(now - start_time, 1e-9)
                avg_s_per_doc = elapsed / max(done_docs, 1)
                delta_docs = done_docs - last_tick_docs
                delta_t = max(now - last_tick_time, 1e-9)
                last_s_per_doc = delta_t / max(delta_docs, 1)
                last_tick_time = now
                last_tick_docs = done_docs
                pbar.set_postfix(
                    {
                        "avg_s/doc": f"{avg_s_per_doc:.2f}",
                        "last_s/doc": f"{last_s_per_doc:.2f}",
                        "buffer": len(points_buffer),
                        "enq": enqueued_docs,
                        "upl": uploaded_docs,
                        "pending": len(futures),
                    }
                )

            for doc, emb, token_info in zip(batch, embeddings, token_infos):
                if doc.image is None:
                    raise ValueError(
                        "CorpusDoc.image is None. For paired datasets, use _index_paired_dataset()."
                    )
                emb_np = (
                    emb.cpu().float().numpy()
                    if hasattr(emb, "cpu")
                    else np.array(emb, dtype=np.float32)
                )
                visual_indices = token_info.get("visual_token_indices") or list(
                    range(emb_np.shape[0])
                )
                visual_embedding = emb_np[visual_indices].astype(np.float32)

                n_rows = token_info.get("n_rows")
                n_cols = token_info.get("n_cols")
                if n_rows and n_cols:
                    num_tiles = int(n_rows) * int(n_cols) + 1
                else:
                    num_tiles = 13

                tile_pooled = tile_level_mean_pooling(
                    visual_embedding, num_tiles=num_tiles, patches_per_tile=64
                )
                global_pooled = tile_pooled.mean(axis=0).astype(np.float32)

                payload = {
                    "dataset": dataset_name,
                    "doc_id": doc.doc_id,
                    "torch_dtype": _torch_dtype_to_str(embedder.torch_dtype),
                    **(doc.payload or {}),
                }

                points_buffer.append(
                    {
                        "id": doc.doc_id,
                        "visual_embedding": visual_embedding,
                        "tile_pooled_embedding": tile_pooled,
                        "global_pooled_embedding": global_pooled,
                        "metadata": payload,
                    }
                )

                if len(points_buffer) >= upload_batch_size:
                    chunk = points_buffer
                    points_buffer = []
                    enqueued_docs += len(chunk)
                    if executor is None:
                        uploaded_docs += int(_upload(chunk) or 0)
                    else:
                        futures.append(executor.submit(_upload, chunk))
                        _drain(block=len(futures) >= upload_workers * 2)
                    if pbar is not None:
                        pbar.set_postfix(
                            {
                                "avg_s/doc": f"{avg_s_per_doc:.2f}",
                                "last_s/doc": f"{last_s_per_doc:.2f}",
                                "buffer": len(points_buffer),
                                "enq": enqueued_docs,
                                "upl": uploaded_docs,
                                "pending": len(futures),
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
        enqueued_docs += len(points_buffer)
        if executor is None:
            uploaded_docs += int(_upload(points_buffer) or 0)
        else:
            futures.append(executor.submit(_upload, points_buffer))

    if executor is not None:
        _drain(block=True)
        executor.shutdown(wait=True)

    if pbar is not None:
        pbar.set_postfix(
            {
                "avg_s/doc": f"{(max(time.time() - start_time, 1e-9) / max(int(pbar.n), 1)):.2f}",
                "last_s/doc": "n/a",
                "buffer": 0,
                "enq": enqueued_docs,
                "upl": uploaded_docs,
                "pending": 0,
            }
        )
        pbar.close()


def _index_paired_dataset(
    *,
    dataset_name: str,
    collection_name: str,
    total_docs: int,
    embedder: VisualEmbedder,
    qdrant_url: str,
    qdrant_api_key: Optional[str],
    prefer_grpc: bool,
    qdrant_vector_dtype: str,
    recreate: bool,
    batch_size: int,
    upload_batch_size: int,
    upload_workers: int,
    upsert_wait: bool,
    loader_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
    use_dataloader: bool,
    indexing_threshold: int,
    full_scan_threshold: int,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("datasets is required. Install with: pip install datasets") from e

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as e:
        raise ImportError(
            "torch is required. Install with: pip install visual-rag-toolkit[embedding]"
        ) from e

    ds0 = load_dataset(dataset_name, split="test")
    cols = set(ds0.column_names)
    image_col = "image" if "image" in cols else "page_image"
    del ds0

    indexer = QdrantIndexer(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        prefer_grpc=prefer_grpc,
        vector_datatype=qdrant_vector_dtype,
    )
    indexer.create_collection(
        force_recreate=recreate,
        indexing_threshold=indexing_threshold,
        full_scan_threshold=int(full_scan_threshold),
    )
    indexer.create_payload_indexes(
        fields=[
            {"field": "dataset", "type": "keyword"},
            {"field": "doc_id", "type": "keyword"},
            {"field": "torch_dtype", "type": "keyword"},
            {"field": "source", "type": "keyword"},
            {"field": "image_filename", "type": "keyword"},
            {"field": "page", "type": "keyword"},
            {"field": "source_doc_id", "type": "keyword"},
        ]
    )

    enqueued_docs = 0
    uploaded_docs = 0
    start_time = time.time()
    last_tick_time = start_time
    last_tick_docs = 0

    pbar = None
    try:
        from tqdm import tqdm

        pbar = tqdm(total=total_docs, desc="📦 Indexing", unit="doc")
    except ImportError:
        pass

    def _upload(points: List[Dict[str, Any]]) -> int:
        return indexer.upload_batch(
            points, delay_between_batches=0.0, wait=upsert_wait, stop_event=stop_event
        )

    executor = None
    futures = []
    import threading

    stop_event = threading.Event()
    if upload_workers and upload_workers > 0:
        from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor
        from concurrent.futures import wait as futures_wait

        executor = ThreadPoolExecutor(max_workers=upload_workers)

        def _drain(block: bool = False) -> None:
            nonlocal uploaded_docs
            nonlocal last_tick_time
            nonlocal last_tick_docs
            if not futures:
                return
            if block:
                done, _ = futures_wait(futures, return_when=FIRST_EXCEPTION)
            else:
                done, _ = futures_wait(futures, timeout=0, return_when=FIRST_EXCEPTION)
            for d in list(done):
                futures.remove(d)
                uploaded_docs += int(d.result() or 0)
            if pbar is not None:
                now = time.time()
                done_docs = int(pbar.n)
                elapsed = max(now - start_time, 1e-9)
                avg_s_per_doc = elapsed / max(done_docs, 1)
                delta_docs = done_docs - last_tick_docs
                delta_t = max(now - last_tick_time, 1e-9)
                last_s_per_doc = delta_t / max(delta_docs, 1)
                last_tick_time = now
                last_tick_docs = done_docs
                pbar.set_postfix(
                    {
                        "avg_s/doc": f"{avg_s_per_doc:.2f}",
                        "last_s/doc": f"{last_s_per_doc:.2f}",
                        "buffer": len(points_buffer),
                        "enq": enqueued_docs,
                        "upl": uploaded_docs,
                        "pending": len(futures),
                    }
                )

    points_buffer: List[Dict[str, Any]] = []
    try:
        if use_dataloader or (loader_workers and loader_workers > 0):
            dl_kwargs = {"batch_size": batch_size, "shuffle": False, "collate_fn": _paired_collate}
            if loader_workers and loader_workers > 0:
                dl_kwargs["num_workers"] = int(loader_workers)
                dl_kwargs["prefetch_factor"] = int(prefetch_factor)
                dl_kwargs["persistent_workers"] = bool(persistent_workers)
                dl_kwargs["pin_memory"] = bool(pin_memory and torch.cuda.is_available())

            data_loader = DataLoader(
                _PairedHFDataset(
                    dataset_name=dataset_name,
                    split="test",
                    total_docs=total_docs,
                    image_col=image_col,
                ),
                **dl_kwargs,
            )
            iterable = ((idxs, images, metas) for (idxs, images, metas) in data_loader)
        else:
            ds = load_dataset(dataset_name, split="test")

            def _iter_batches():
                for start in range(0, total_docs, batch_size):
                    batch = ds[start : start + batch_size]
                    images = batch[image_col]
                    metas = [
                        {k: batch[k][i] for k in batch.keys() if k != image_col}
                        for i in range(len(images))
                    ]
                    idxs = list(range(start, start + len(images)))
                    yield idxs, images, metas

            iterable = _iter_batches()

        for idxs, images, metas in iterable:
            embeddings, token_infos = embedder.embed_images(
                images,
                batch_size=batch_size,
                return_token_info=True,
                show_progress=False,
            )
            if pbar is not None:
                pbar.update(len(images))
                now = time.time()
                done_docs = int(pbar.n)
                elapsed = max(now - start_time, 1e-9)
                avg_s_per_doc = elapsed / max(done_docs, 1)
                delta_docs = done_docs - last_tick_docs
                delta_t = max(now - last_tick_time, 1e-9)
                last_s_per_doc = delta_t / max(delta_docs, 1)
                last_tick_time = now
                last_tick_docs = done_docs
                pbar.set_postfix(
                    {
                        "avg_s/doc": f"{avg_s_per_doc:.2f}",
                        "last_s/doc": f"{last_s_per_doc:.2f}",
                        "buffer": len(points_buffer),
                        "enq": enqueued_docs,
                        "upl": uploaded_docs,
                        "pending": len(futures),
                    }
                )

            for idx, meta, emb, token_info in zip(idxs, metas, embeddings, token_infos):
                doc_id = paired_doc_id(meta, int(idx))
                payload = {
                    "dataset": dataset_name,
                    "doc_id": doc_id,
                    "torch_dtype": _torch_dtype_to_str(embedder.torch_dtype),
                    **paired_payload(meta, int(idx)),
                }

                emb_np = (
                    emb.cpu().float().numpy()
                    if hasattr(emb, "cpu")
                    else np.array(emb, dtype=np.float32)
                )
                visual_indices = token_info.get("visual_token_indices") or list(
                    range(emb_np.shape[0])
                )
                visual_embedding = emb_np[visual_indices].astype(np.float32)

                n_rows = token_info.get("n_rows")
                n_cols = token_info.get("n_cols")
                num_tiles = int(n_rows) * int(n_cols) + 1 if n_rows and n_cols else 13

                tile_pooled = tile_level_mean_pooling(
                    visual_embedding, num_tiles=num_tiles, patches_per_tile=64
                )
                global_pooled = tile_pooled.mean(axis=0).astype(np.float32)

                points_buffer.append(
                    {
                        "id": doc_id,
                        "visual_embedding": visual_embedding,
                        "tile_pooled_embedding": tile_pooled,
                        "global_pooled_embedding": global_pooled,
                        "metadata": payload,
                    }
                )

                if len(points_buffer) >= upload_batch_size:
                    chunk = points_buffer
                    points_buffer = []
                    enqueued_docs += len(chunk)
                    if executor is None:
                        uploaded_docs += int(_upload(chunk) or 0)
                    else:
                        futures.append(executor.submit(_upload, chunk))
                        _drain(block=len(futures) >= upload_workers * 2)
            if executor is not None:
                _drain(block=False)
    except KeyboardInterrupt:
        stop_event.set()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        raise

    if points_buffer:
        enqueued_docs += len(points_buffer)
        if executor is None:
            uploaded_docs += int(_upload(points_buffer) or 0)
        else:
            futures.append(executor.submit(_upload, points_buffer))

    if executor is not None:
        _drain(block=True)
        executor.shutdown(wait=True)

    if pbar is not None:
        pbar.set_postfix(
            {
                "avg_s/doc": f"{(max(time.time() - start_time, 1e-9) / max(int(pbar.n), 1)):.2f}",
                "last_s/doc": "n/a",
                "buffer": 0,
                "enq": enqueued_docs,
                "upl": uploaded_docs,
                "pending": 0,
            }
        )
        pbar.close()


def _evaluate(
    *,
    queries: List[Any],
    qrels: Dict[str, Dict[str, int]],
    retriever: MultiVectorRetriever,
    top_k: int,
    prefetch_k: int,
    mode: str,
    stage1_mode: str,
) -> Dict[str, float]:
    ndcg10: List[float] = []
    mrr10: List[float] = []
    recall10: List[float] = []
    recall5: List[float] = []
    latencies_ms: List[float] = []

    for q in queries:
        start = time.time()
        results = retriever.search(
            query=q.text,
            top_k=top_k,
            mode=mode,
            prefetch_k=prefetch_k,
            stage1_mode=stage1_mode,
        )
        latencies_ms.append((time.time() - start) * 1000.0)

        ranking = [str(r["id"]) for r in results]
        rels = qrels.get(q.query_id, {})

        ndcg10.append(ndcg_at_k(ranking, rels, k=10))
        mrr10.append(mrr_at_k(ranking, rels, k=10))
        recall5.append(recall_at_k(ranking, rels, k=5))
        recall10.append(recall_at_k(ranking, rels, k=10))

    return {
        "ndcg@10": float(np.mean(ndcg10)),
        "mrr@10": float(np.mean(mrr10)),
        "recall@5": float(np.mean(recall5)),
        "recall@10": float(np.mean(recall10)),
        "avg_latency_ms": float(np.mean(latencies_ms)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="vidore/tatdqa_test")
    parser.add_argument("--collection", type=str, default="vidore_tatdqa_test")
    parser.add_argument("--model", type=str, default="vidore/colSmol-500M")
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype for model weights (default: auto; CUDA->bfloat16, else float32).",
    )
    parser.add_argument(
        "--qdrant-vector-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Datatype for vectors stored in Qdrant (default: float16).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--upload-batch-size", type=int, default=None)
    parser.add_argument("--upload-workers", type=int, default=0)
    wait_group = parser.add_mutually_exclusive_group()
    wait_group.add_argument(
        "--upsert-wait",
        action="store_true",
        help="Wait for Qdrant upserts to complete before continuing (default: false).",
    )
    wait_group.add_argument(
        "--no-upsert-wait",
        action="store_true",
        help="Deprecated (default is already no-wait). Kept for backwards compatibility.",
    )
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--use-dataloader",
        action="store_true",
        help="Use torch DataLoader even with --loader-workers 0 (default: false).",
    )
    grpc_group = parser.add_mutually_exclusive_group()
    grpc_group.add_argument("--prefer-grpc", dest="prefer_grpc", action="store_true", default=True)
    grpc_group.add_argument("--no-prefer-grpc", dest="prefer_grpc", action="store_false")
    parser.add_argument(
        "--index", action="store_true", help="Index corpus into Qdrant before evaluating"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection (implies --index)",
    )
    parser.add_argument(
        "--indexing-threshold",
        type=int,
        default=0,
        help="Qdrant optimizer indexing threshold (0 = always build indexes).",
    )
    parser.add_argument(
        "--full-scan-threshold",
        type=int,
        default=0,
        help="Qdrant HNSW full_scan_threshold (0 = always use HNSW).",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--prefetch-k", type=int, default=200)
    parser.add_argument(
        "--mode",
        type=str,
        default="single_full",
        choices=["single_full", "single_tiles", "single_global", "two_stage"],
    )
    parser.add_argument(
        "--stage1-mode",
        type=str,
        default="tokens_vs_standard_pooling",
        choices=[
            "pooled_query_vs_standard_pooling",
            "tokens_vs_standard_pooling",
            "pooled_query_vs_experimental_pooling",
            "tokens_vs_experimental_pooling",
            "pooled_query_vs_global",
            # Backwards-compatible aliases
            "pooled_query_vs_tiles",
            "tokens_vs_tiles",
            "pooled_query_vs_experimental",
            "tokens_vs_experimental",
        ],
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
            "ColQwen only: choose experimental pooling named vector for experimental stage-1. "
            "Maps to: 'experimental_pooling_gaussian' or 'experimental_pooling_triangular'. "
            "If omitted, uses 'experimental_pooling' (Gaussian alias)."
        ),
    )
    parser.add_argument("--output", type=str, default="results/qdrant_vidore_tatdqa_test.json")

    args = parser.parse_args()

    _maybe_load_dotenv()

    qdrant_url = _ensure_env("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    upload_batch_size = args.upload_batch_size or args.batch_size
    upsert_wait = bool(args.upsert_wait)

    if upsert_wait:
        print("Qdrant upserts wait for completion (wait=True).")
    else:
        print("Qdrant upserts are async (wait=False).")

    corpus, queries, qrels, protocol = load_vidore_dataset_auto(args.dataset)

    embedder = VisualEmbedder(
        model_name=args.model,
        batch_size=args.batch_size,
        torch_dtype=_parse_torch_dtype(args.torch_dtype),
    )

    if args.recreate:
        args.index = True

    if args.index:
        if protocol == "paired":
            _index_paired_dataset(
                dataset_name=args.dataset,
                collection_name=args.collection,
                total_docs=len(corpus),
                embedder=embedder,
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                prefer_grpc=args.prefer_grpc,
                qdrant_vector_dtype=args.qdrant_vector_dtype,
                recreate=args.recreate,
                batch_size=args.batch_size,
                upload_batch_size=upload_batch_size,
                upload_workers=args.upload_workers,
                upsert_wait=upsert_wait,
                loader_workers=args.loader_workers,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers,
                pin_memory=args.pin_memory,
                use_dataloader=args.use_dataloader,
                indexing_threshold=args.indexing_threshold,
                full_scan_threshold=args.full_scan_threshold,
            )
        else:
            _index_corpus(
                dataset_name=args.dataset,
                collection_name=args.collection,
                corpus=corpus,
                embedder=embedder,
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                prefer_grpc=args.prefer_grpc,
                qdrant_vector_dtype=args.qdrant_vector_dtype,
                recreate=args.recreate,
                batch_size=args.batch_size,
                upload_batch_size=upload_batch_size,
                upload_workers=args.upload_workers,
                upsert_wait=upsert_wait,
                indexing_threshold=args.indexing_threshold,
                full_scan_threshold=args.full_scan_threshold,
            )

    def _is_colqwen_model(model_name: str) -> bool:
        return "colqwen" in str(model_name).lower()

    exp_vector_name = "experimental_pooling"
    uses_experimental = str(args.stage1_mode) in (
        "pooled_query_vs_experimental_pooling",
        "tokens_vs_experimental_pooling",
    ) and str(args.mode) in ("two_stage", "three_stage")

    if (
        getattr(args, "experimental_pooling_technique", None)
        and getattr(args, "experimental_pooling_k", None) is not None
    ):
        raise ValueError(
            "Use only one of --experimental-pooling-technique or --experimental-pooling-k."
        )

    if uses_experimental and getattr(args, "experimental_pooling_technique", None):
        if not _is_colqwen_model(args.model):
            raise ValueError(
                "--experimental-pooling-technique is only supported for ColQwen models."
            )
        exp_vector_name = (
            f"experimental_pooling_{str(args.experimental_pooling_technique).strip().lower()}"
        )

    if uses_experimental and getattr(args, "experimental_pooling_k", None) is not None:
        if _is_colqwen_model(args.model):
            raise ValueError(
                "--experimental-pooling-k is intended for ColPali (experimental_pooling_{k}), not ColQwen."
            )
        exp_vector_name = f"experimental_pooling_{int(args.experimental_pooling_k)}"

    retriever = MultiVectorRetriever(
        collection_name=args.collection,
        embedder=embedder,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        prefer_grpc=args.prefer_grpc,
        experimental_vector_name=exp_vector_name,
    )

    metrics = _evaluate(
        queries=queries,
        qrels=qrels,
        retriever=retriever,
        top_k=args.top_k,
        prefetch_k=args.prefetch_k,
        mode=args.mode,
        stage1_mode=args.stage1_mode,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "protocol": protocol,
                "collection": args.collection,
                "model": args.model,
                "torch_dtype": _torch_dtype_to_str(embedder.torch_dtype),
                "qdrant_vector_dtype": args.qdrant_vector_dtype,
                "mode": args.mode,
                "stage1_mode": args.stage1_mode if args.mode == "two_stage" else None,
                "prefetch_k": args.prefetch_k if args.mode == "two_stage" else None,
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
