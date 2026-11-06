import argparse
import json
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


def _as_jsonable(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_jsonable(v) for v in obj]
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return {k: _as_jsonable(v) for k, v in obj.__dict__.items()}
        except Exception:
            pass
    return str(obj)


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


def _snapshot_info(info) -> dict:
    status = getattr(info, "status", None)
    if hasattr(status, "value"):
        status = status.value
    optimizer_status = getattr(info, "optimizer_status", None)
    if hasattr(optimizer_status, "value"):
        optimizer_status = optimizer_status.value
    return {
        "status": _as_jsonable(status),
        "optimizer_status": _as_jsonable(optimizer_status),
        "points_count": _as_jsonable(getattr(info, "points_count", None)),
        "indexed_vectors_count": _as_jsonable(getattr(info, "indexed_vectors_count", None)),
        "segments_count": _as_jsonable(getattr(info, "segments_count", None)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str, required=True)
    parser.add_argument("--indexing-threshold", type=int, default=0)
    parser.add_argument("--prefer-grpc", action="store_true")
    parser.add_argument("--url", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--poll-sec", type=int, default=2)
    parser.add_argument("--dump-json", type=str, default="")
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
        timeout=60,
    )

    client.update_collection(
        collection_name=args.collection,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=int(args.indexing_threshold)
        ),
    )

    info = client.get_collection(args.collection)
    snap = _snapshot_info(info)
    print(
        f"Updated optimizers.indexing_threshold={args.indexing_threshold} for collection='{args.collection}'. "
        f"points={snap['points_count']}, indexed_vectors={snap['indexed_vectors_count']}, "
        f"status={snap['status']}, optimizer_status={snap['optimizer_status']}, segments={snap['segments_count']}"
    )
    if args.dump_json:
        out_path = Path(args.dump_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {"event": "after_update", "collection": args.collection, "snapshot": snap},
                f,
                indent=2,
            )

    if not args.wait:
        return

    start = time.time()
    while True:
        info = client.get_collection(args.collection)
        snap = _snapshot_info(info)
        indexed_total = _indexed_total(snap["indexed_vectors_count"])
        total = int(snap["points_count"] or 0)
        if indexed_total >= total and total > 0:
            print(
                f"Indexing complete: indexed_vectors={snap['indexed_vectors_count']}, points={snap['points_count']}"
            )
            if args.dump_json:
                out_path = Path(args.dump_json)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(
                        {"event": "complete", "collection": args.collection, "snapshot": snap},
                        f,
                        indent=2,
                    )
            return
        if time.time() - start > args.timeout_sec:
            print(
                f"Timeout waiting for indexing: indexed_vectors={snap['indexed_vectors_count']}, "
                f"points={snap['points_count']}, status={snap['status']}, optimizer_status={snap['optimizer_status']}"
            )
            if args.dump_json:
                out_path = Path(args.dump_json)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(
                        {"event": "timeout", "collection": args.collection, "snapshot": snap},
                        f,
                        indent=2,
                    )
            return
        print(
            f"Indexing in progress: indexed_vectors={snap['indexed_vectors_count']}, "
            f"points={snap['points_count']}, status={snap['status']}, optimizer_status={snap['optimizer_status']}, "
            f"segments={snap['segments_count']}"
        )
        time.sleep(max(0.1, float(args.poll_sec)))


if __name__ == "__main__":
    main()
