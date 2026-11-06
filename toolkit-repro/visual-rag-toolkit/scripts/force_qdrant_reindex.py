import argparse
import os
import time
from pathlib import Path


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if Path(".env").exists():
        load_dotenv(".env")


def _indexed_total(indexed_vectors_count) -> int:
    if indexed_vectors_count is None:
        return 0
    if isinstance(indexed_vectors_count, dict):
        try:
            return int(sum(int(v) for v in indexed_vectors_count.values()))
        except Exception:
            return 0
    try:
        return int(indexed_vectors_count)
    except Exception:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str, required=True)
    parser.add_argument("--prefer-grpc", action="store_true")
    parser.add_argument("--url", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--indexing-threshold", type=int, default=0)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construct", type=int, default=100)
    parser.add_argument("--full-scan-threshold", type=int, default=10000)
    parser.add_argument(
        "--on-disk",
        action="store_true",
        help="Store HNSW index on disk (recommended for large vectors).",
    )
    parser.add_argument("--max-indexing-threads", type=int, default=0)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--poll-sec", type=float, default=2.0)
    args = parser.parse_args()

    _maybe_load_dotenv()

    qdrant_url = args.url or os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise ValueError("QDRANT_URL not set")
    qdrant_api_key = args.api_key or os.getenv("QDRANT_API_KEY")

    from qdrant_client import QdrantClient
    from qdrant_client.http import models

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=args.prefer_grpc,
        check_compatibility=False,
        timeout=120,
    )

    hnsw = models.HnswConfigDiff(
        m=int(args.m),
        ef_construct=int(args.ef_construct),
        full_scan_threshold=int(args.full_scan_threshold),
        on_disk=bool(args.on_disk),
        max_indexing_threads=int(args.max_indexing_threads),
    )

    vectors_config = {
        "initial": models.VectorParamsDiff(hnsw_config=hnsw, on_disk=True),
        "mean_pooling": models.VectorParamsDiff(hnsw_config=hnsw),
        "global_pooling": models.VectorParamsDiff(hnsw_config=hnsw),
    }

    client.update_collection(
        collection_name=args.collection,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=int(args.indexing_threshold)
        ),
        hnsw_config=hnsw,
        vectors_config=vectors_config,
    )

    info = client.get_collection(args.collection)
    print(
        f"Triggered reindex update for '{args.collection}'. "
        f"points={info.points_count}, indexed_vectors={info.indexed_vectors_count}, "
        f"status={getattr(getattr(info.status, 'value', None), 'value', getattr(info, 'status', None))}"
    )

    if not args.wait:
        return

    start = time.time()
    while True:
        info = client.get_collection(args.collection)
        indexed_total = _indexed_total(info.indexed_vectors_count)
        total = int(info.points_count or 0)
        print(
            f"poll: points={info.points_count}, indexed_vectors={info.indexed_vectors_count}, "
            f"segments={getattr(info, 'segments_count', None)}"
        )
        if total > 0 and indexed_total >= total:
            return
        if time.time() - start > args.timeout_sec:
            return
        time.sleep(max(0.1, float(args.poll_sec)))


if __name__ == "__main__":
    main()
