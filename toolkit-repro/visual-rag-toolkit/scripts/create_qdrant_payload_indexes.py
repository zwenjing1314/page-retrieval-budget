import argparse
import os
from pathlib import Path
from typing import Dict, List


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if Path(".env").exists():
        load_dotenv(".env")


def _infer_type(values: List[object]) -> str:
    for v in values:
        if isinstance(v, bool):
            return "bool"
    for v in values:
        if isinstance(v, int) and not isinstance(v, bool):
            return "integer"
    for v in values:
        if isinstance(v, float):
            return "float"
    return "keyword"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str, required=True)
    parser.add_argument("--prefer-grpc", action="store_true")
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated list of payload fields to index (optional).",
    )
    args = parser.parse_args()

    _maybe_load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise ValueError("QDRANT_URL not set")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    from qdrant_client import QdrantClient

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=args.prefer_grpc,
        check_compatibility=False,
        timeout=120,
    )

    points, _ = client.scroll(
        collection_name=args.collection,
        limit=int(args.sample),
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        raise ValueError(f"No points found in collection '{args.collection}'")

    only = [s.strip() for s in args.only.split(",") if s.strip()]
    only_set = set(only) if only else None

    values_by_key: Dict[str, List[object]] = {}
    for p in points:
        payload = p.payload or {}
        if not isinstance(payload, dict):
            continue
        for k, v in payload.items():
            if only_set is not None and k not in only_set:
                continue
            if isinstance(v, dict) or isinstance(v, list):
                continue
            values_by_key.setdefault(k, []).append(v)

    from visual_rag.indexing.qdrant_indexer import QdrantIndexer

    indexer = QdrantIndexer(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=args.collection,
        prefer_grpc=args.prefer_grpc,
    )

    fields = [{"field": k, "type": _infer_type(vs)} for k, vs in sorted(values_by_key.items())]
    if not fields:
        raise ValueError("No indexable payload fields found (all were nested or empty?)")

    indexer.create_payload_indexes(fields=fields)
    print(
        f"Created/ensured {len(fields)} payload indexes on '{args.collection}': {[f['field'] for f in fields]}"
    )


if __name__ == "__main__":
    main()
