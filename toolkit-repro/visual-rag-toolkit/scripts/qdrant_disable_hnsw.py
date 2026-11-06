"""
Disable dense (HNSW) indexing for a Qdrant collection (make indexed_vectors_count go to 0).

Qdrant supports disabling HNSW by setting `hnsw_config.m = 0`.
For an already-indexed collection, Qdrant may run a reconstruction/optimization pass.
Once that finishes, `indexed_vectors_count` should become 0.

Ref: "Optimizing Memory for Bulk Uploads" (Qdrant, Feb 2025) recommends `m=0` to disable HNSW.

Usage:
  python scripts/qdrant_disable_hnsw.py --collection "my_collection" --wait
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Optional


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    if Path(".env").exists():
        load_dotenv(".env")


def _get_env(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def _as_jsonable(obj: Any):
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


def _snapshot(client, collection: str) -> dict:
    info = client.get_collection(collection)
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
    parser.add_argument("--collection", required=True)
    parser.add_argument("--prefer-grpc", action="store_true")
    parser.add_argument("--url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-sec", type=float, default=5.0)
    parser.add_argument("--timeout-sec", type=float, default=1800.0)
    parser.add_argument("--dump-json", default="", help="Optional path to dump snapshots JSON")
    args = parser.parse_args()

    _maybe_load_dotenv()

    qdrant_url = args.url or _get_env("QDRANT_URL")
    if not qdrant_url:
        raise SystemExit("QDRANT_URL not set (or pass --url)")
    qdrant_api_key = args.api_key or _get_env("QDRANT_API_KEY")

    from qdrant_client import QdrantClient
    from qdrant_client.http import models

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=bool(args.prefer_grpc),
        check_compatibility=False,
        timeout=float(args.timeout),
    )

    before = _snapshot(client, args.collection)
    print(
        f"Before: points={before['points_count']} indexed_vectors={before['indexed_vectors_count']} "
        f"status={before['status']} optimizer={before['optimizer_status']} segments={before['segments_count']}"
    )

    # Disable HNSW for dense vectors
    client.update_collection(
        collection_name=args.collection,
        hnsw_config=models.HnswConfigDiff(m=0),
    )
    after = _snapshot(client, args.collection)
    print(
        f"After update(m=0): points={after['points_count']} indexed_vectors={after['indexed_vectors_count']} "
        f"status={after['status']} optimizer={after['optimizer_status']} segments={after['segments_count']}"
    )

    if args.dump_json:
        out_path = Path(args.dump_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "collection": args.collection,
                    "before": before,
                    "after_update": after,
                },
                f,
                indent=2,
            )

    if not args.wait:
        return

    start = time.time()
    while True:
        snap = _snapshot(client, args.collection)
        indexed = _indexed_total(snap["indexed_vectors_count"])
        if indexed == 0:
            print(
                f"✅ Done: indexed_vectors_count is 0. points={snap['points_count']} "
                f"status={snap['status']} optimizer={snap['optimizer_status']}"
            )
            if args.dump_json:
                out_path = Path(args.dump_json)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(
                        {
                            "collection": args.collection,
                            "before": before,
                            "after_update": after,
                            "complete": snap,
                        },
                        f,
                        indent=2,
                    )
            return
        if time.time() - start > float(args.timeout_sec):
            print(
                f"⏳ Timeout waiting for indexed_vectors_count=0. indexed_vectors={snap['indexed_vectors_count']}, "
                f"points={snap['points_count']}, status={snap['status']}, optimizer={snap['optimizer_status']}"
            )
            return
        print(
            f"… waiting: indexed_vectors={snap['indexed_vectors_count']} points={snap['points_count']} "
            f"status={snap['status']} optimizer={snap['optimizer_status']} segments={snap['segments_count']}"
        )
        time.sleep(max(0.2, float(args.poll_sec)))


if __name__ == "__main__":
    main()
