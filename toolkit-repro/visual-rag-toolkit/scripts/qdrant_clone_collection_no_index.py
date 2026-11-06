"""
Clone an existing Qdrant collection into a new collection with indexing disabled.

Why: Qdrant doesn't provide an in-place "de-index" for already-built HNSW.
This script clones points (payload + vectors) into a fresh collection created
with a very large `indexing_threshold`, so `indexed_vectors_count` stays 0.

Usage:
  python scripts/qdrant_clone_collection_no_index.py \
    --source vidore_beir_v2_... \
    --dest vidore_beir_v2_...__noindex \
    --embedding-dim 128 \
    --vector-dtype float32 \
    --indexing-threshold 1000000000 \
    --prefer-grpc
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except Exception:
    DOTENV_AVAILABLE = False

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


def _get_env(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def _require(value: Optional[str], *, name: str) -> str:
    if not value:
        raise SystemExit(
            f"Missing {name}. Set it in env (preferred) or pass the corresponding flag."
        )
    return value


def _chunks(seq: Sequence[Any], n: int) -> List[Sequence[Any]]:
    return [seq[i : i + n] for i in range(0, len(seq), n)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Existing source collection name")
    parser.add_argument("--dest", required=True, help="Destination collection name")
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Qdrant URL (or set QDRANT_URL env var)",
    )
    parser.add_argument(
        "--qdrant-api-key",
        default=None,
        help="Qdrant API key (or set QDRANT_API_KEY env var)",
    )
    parser.add_argument(
        "--prefer-grpc",
        action="store_true",
        help="Use gRPC transport (recommended for large vectors)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout seconds",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Vector dimension (typically 128 for ColPali/ColQwen)",
    )
    parser.add_argument(
        "--vector-dtype",
        choices=["float16", "float32"],
        default="float32",
        help="Vector datatype for destination collection",
    )
    parser.add_argument(
        "--indexing-threshold",
        type=int,
        default=1_000_000_000,
        help="Large value prevents HNSW building for small collections",
    )
    # Note: some qdrant-client versions don't support full_scan_threshold in OptimizersConfigDiff.
    parser.add_argument(
        "--recreate-dest",
        action="store_true",
        help="Delete destination collection if it already exists",
    )
    parser.add_argument(
        "--scroll-limit",
        type=int,
        default=256,
        help="How many points to fetch per scroll call",
    )
    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=64,
        help="How many points per upsert call",
    )

    args = parser.parse_args()

    if DOTENV_AVAILABLE:
        load_dotenv()

    url = args.qdrant_url or _get_env("QDRANT_URL")
    api_key = args.qdrant_api_key or _get_env("QDRANT_API_KEY")
    url = _require(url, name="QDRANT_URL/--qdrant-url")
    api_key = _require(api_key, name="QDRANT_API_KEY/--qdrant-api-key")

    client = QdrantClient(
        url=url,
        api_key=api_key,
        prefer_grpc=bool(args.prefer_grpc),
        timeout=float(args.timeout),
    )

    # Verify source exists
    src_info = client.get_collection(args.source)
    print(f"✅ Source collection found: {args.source}")
    print(
        f"   points_count≈{src_info.points_count} indexed_vectors_count={src_info.indexed_vectors_count}"
    )

    # Create/recreate destination with the same named vectors layout
    if args.recreate_dest:
        try:
            client.delete_collection(args.dest)
            print(f"🗑️ Deleted existing destination: {args.dest}")
        except Exception as e:
            print(f"⚠️ Could not delete dest (may not exist): {e}")

    # Use same vector names as toolkit expects
    datatype = qm.Datatype.FLOAT16 if args.vector_dtype == "float16" else qm.Datatype.FLOAT32
    multivector_config = qm.MultiVectorConfig(comparator=qm.MultiVectorComparator.MAX_SIM)
    vectors_config: Dict[str, qm.VectorParams] = {
        "initial": qm.VectorParams(
            size=int(args.embedding_dim),
            distance=qm.Distance.COSINE,
            on_disk=True,
            multivector_config=multivector_config,
            datatype=datatype,
        ),
        "mean_pooling": qm.VectorParams(
            size=int(args.embedding_dim),
            distance=qm.Distance.COSINE,
            on_disk=False,
            multivector_config=multivector_config,
            datatype=datatype,
        ),
        "experimental_pooling": qm.VectorParams(
            size=int(args.embedding_dim),
            distance=qm.Distance.COSINE,
            on_disk=False,
            multivector_config=multivector_config,
            datatype=datatype,
        ),
        "global_pooling": qm.VectorParams(
            size=int(args.embedding_dim),
            distance=qm.Distance.COSINE,
            on_disk=False,
            datatype=datatype,
        ),
    }

    try:
        client.create_collection(
            collection_name=args.dest,
            vectors_config=vectors_config,
            optimizers_config=qm.OptimizersConfigDiff(
                indexing_threshold=int(args.indexing_threshold),
            ),
        )
        # Keep filename index for skip_existing (cheap and helpful)
        try:
            client.create_payload_index(
                collection_name=args.dest,
                field_name="filename",
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass
        print(f"✅ Created destination collection: {args.dest}")
    except Exception as e:
        # If it already exists, proceed (unless user expected recreate)
        print(f"ℹ️ Destination create skipped/failed (may already exist): {e}")

    # Clone points
    next_offset = None
    total = 0

    while True:
        points, next_offset = client.scroll(
            collection_name=args.source,
            limit=int(args.scroll_limit),
            with_payload=True,
            with_vectors=True,
            offset=next_offset,
        )
        if not points:
            break

        # Upsert in batches
        for batch in _chunks(points, int(args.upsert_batch_size)):
            upsert_points: List[qm.PointStruct] = []
            for p in batch:
                # p.vector may be dict (named vectors) or list (single). We expect dict.
                vectors = p.vector
                payload = p.payload or {}
                upsert_points.append(
                    qm.PointStruct(
                        id=p.id,
                        vector=vectors,
                        payload=payload,
                    )
                )

            client.upsert(
                collection_name=args.dest,
                points=upsert_points,
                wait=True,
            )
            total += len(upsert_points)
            if total % 500 == 0:
                print(f"… cloned {total} points")

    dst_info = client.get_collection(args.dest)
    exact = client.count(collection_name=args.dest, exact=True)
    print("✅ Clone complete")
    print(
        f"   dest.points_count≈{dst_info.points_count} dest.indexed_vectors_count={dst_info.indexed_vectors_count}"
    )
    print(f"   dest.count(exact)= {exact.count}")


if __name__ == "__main__":
    main()
