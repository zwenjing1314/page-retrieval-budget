"""
Rebuild a Qdrant collection so `indexed_vectors_count` becomes 0 (no ANN/HNSW built).

Important:
- Qdrant does NOT support "unbuilding" an existing HNSW index in-place.
- The only reliable way to get indexed_vectors_count back to 0 is to:
  1) copy points to a temporary collection,
  2) delete + recreate the original collection with a very large indexing_threshold,
  3) copy points back,
  4) delete the temporary collection.

This script keeps the *final* collection name unchanged.

Usage:
  python scripts/qdrant_rebuild_collection_no_index.py \
    --collection "my_collection" \
    --embedding-dim 128 \
    --vector-dtype float32 \
    --indexing-threshold 1000000000
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
        raise SystemExit(f"Missing {name}. Provide flag or set env var.")
    return value


def _chunks(seq: Sequence[Any], n: int) -> List[Sequence[Any]]:
    return [seq[i : i + n] for i in range(0, len(seq), n)]


def _vectors_config(
    *,
    embedding_dim: int,
    vector_dtype: str,
    experimental_vector_names: Optional[List[str]] = None,
) -> Dict[str, qm.VectorParams]:
    datatype = qm.Datatype.FLOAT16 if vector_dtype == "float16" else qm.Datatype.FLOAT32
    multivector_config = qm.MultiVectorConfig(comparator=qm.MultiVectorComparator.MAX_SIM)
    vectors = {
        "initial": qm.VectorParams(
            size=int(embedding_dim),
            distance=qm.Distance.COSINE,
            on_disk=True,
            multivector_config=multivector_config,
            datatype=datatype,
        ),
        "mean_pooling": qm.VectorParams(
            size=int(embedding_dim),
            distance=qm.Distance.COSINE,
            on_disk=False,
            multivector_config=multivector_config,
            datatype=datatype,
        ),
        "experimental_pooling": qm.VectorParams(
            size=int(embedding_dim),
            distance=qm.Distance.COSINE,
            on_disk=False,
            multivector_config=multivector_config,
            datatype=datatype,
        ),
        "global_pooling": qm.VectorParams(
            size=int(embedding_dim),
            distance=qm.Distance.COSINE,
            on_disk=False,
            datatype=datatype,
        ),
    }
    if experimental_vector_names:
        for n in experimental_vector_names:
            s = str(n).strip()
            if not s:
                continue
            if s in vectors:
                continue
            vectors[s] = qm.VectorParams(
                size=int(embedding_dim),
                distance=qm.Distance.COSINE,
                on_disk=False,
                multivector_config=multivector_config,
                datatype=datatype,
            )
    return vectors


def _scroll_points(
    client: QdrantClient,
    *,
    collection: str,
    limit: int,
    offset: Any,
) -> Tuple[List[Any], Any]:
    # qdrant-client returns (points, next_offset)
    return client.scroll(
        collection_name=collection,
        limit=int(limit),
        with_payload=True,
        with_vectors=True,
        offset=offset,
    )


def _clone(
    client: QdrantClient,
    *,
    source: str,
    dest: str,
    embedding_dim: int,
    vector_dtype: str,
    indexing_threshold: int,
    recreate_dest: bool,
    scroll_limit: int,
    upsert_batch_size: int,
    experimental_vector_names: Optional[List[str]],
) -> int:
    if recreate_dest:
        try:
            client.delete_collection(dest)
        except Exception:
            pass

    # Create destination collection
    client.create_collection(
        collection_name=dest,
        vectors_config=_vectors_config(
            embedding_dim=embedding_dim,
            vector_dtype=vector_dtype,
            experimental_vector_names=experimental_vector_names,
        ),
        optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=int(indexing_threshold)),
    )
    # Keep filename payload index (cheap; useful for skip_existing)
    try:
        client.create_payload_index(
            collection_name=dest,
            field_name="filename",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass

    total = 0
    next_offset = None

    while True:
        points, next_offset = _scroll_points(
            client,
            collection=source,
            limit=scroll_limit,
            offset=next_offset,
        )
        if not points:
            break

        for batch in _chunks(points, int(upsert_batch_size)):
            upsert_points: List[qm.PointStruct] = []
            for p in batch:
                upsert_points.append(
                    qm.PointStruct(
                        id=p.id,
                        vector=p.vector,
                        payload=p.payload or {},
                    )
                )
            client.upsert(collection_name=dest, points=upsert_points, wait=True)
            total += len(upsert_points)
            if total % 500 == 0:
                print(f"… copied {total} points to {dest}")

        if next_offset is None:
            break

    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection", required=True, help="Collection to rebuild (final name stays same)"
    )
    parser.add_argument("--qdrant-url", default=None, help="Override QDRANT_URL")
    parser.add_argument("--qdrant-api-key", default=None, help="Override QDRANT_API_KEY")
    parser.add_argument("--prefer-grpc", action="store_true", help="Use gRPC transport")
    parser.add_argument("--timeout", type=float, default=300.0, help="Client timeout seconds")
    parser.add_argument(
        "--embedding-dim", type=int, default=128, help="Embedding dim (typically 128)"
    )
    parser.add_argument("--vector-dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument(
        "--indexing-threshold",
        type=int,
        default=1_000_000_000,
        help="Very large value keeps indexed_vectors_count at 0",
    )
    parser.add_argument("--scroll-limit", type=int, default=256)
    parser.add_argument("--upsert-batch-size", type=int, default=64)
    parser.add_argument(
        "--pooling-windows",
        "--pooling_windows",
        type=int,
        nargs="+",
        default=None,
        help=(
            "If provided, include additional experimental named vectors "
            "('experimental_pooling_{k}') in the rebuilt collection schema."
        ),
    )
    parser.add_argument(
        "--add-colqwen-techniques",
        dest="add_colqwen_techniques",
        action="store_true",
        default=False,
        help=(
            "Include ColQwen experimental technique named vectors in the rebuilt schema: "
            "'experimental_pooling_gaussian' and 'experimental_pooling_triangular'."
        ),
    )
    parser.add_argument(
        "--keep-temp", action="store_true", help="Do not delete temp collection at the end"
    )
    args = parser.parse_args()

    ks = args.pooling_windows or []
    seen = set()
    ks_norm: List[int] = []
    for k in ks:
        try:
            ki = int(k)
        except Exception:
            continue
        if ki <= 0 or ki in seen:
            continue
        seen.add(ki)
        ks_norm.append(ki)
    experimental_vector_names = [f"experimental_pooling_{k}" for k in ks_norm]
    if bool(getattr(args, "add_colqwen_techniques", False)):
        experimental_vector_names.extend(
            ["experimental_pooling_gaussian", "experimental_pooling_triangular"]
        )

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
        check_compatibility=False,
    )

    info = client.get_collection(args.collection)
    print(f"✅ Found collection: {args.collection}")
    print(f"   points_count≈{info.points_count} indexed_vectors_count={info.indexed_vectors_count}")

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    temp = f"{args.collection}__tmp_rebuild_noindex__{stamp}"
    print(f"🧪 Temp collection: {temp}")

    print("➡️ Step 1/4: Copying points to temp…")
    copied1 = _clone(
        client,
        source=args.collection,
        dest=temp,
        embedding_dim=int(args.embedding_dim),
        vector_dtype=str(args.vector_dtype),
        indexing_threshold=int(args.indexing_threshold),
        recreate_dest=True,
        scroll_limit=int(args.scroll_limit),
        upsert_batch_size=int(args.upsert_batch_size),
        experimental_vector_names=experimental_vector_names,
    )
    temp_info = client.get_collection(temp)
    print(
        f"✅ Temp ready: points_count≈{temp_info.points_count} indexed_vectors_count={temp_info.indexed_vectors_count}"
    )

    print("➡️ Step 2/4: Deleting original collection…")
    client.delete_collection(args.collection)
    time.sleep(1.0)

    print("➡️ Step 3/4: Recreating original with indexing disabled…")
    client.create_collection(
        collection_name=args.collection,
        vectors_config=_vectors_config(
            embedding_dim=int(args.embedding_dim),
            vector_dtype=str(args.vector_dtype),
            experimental_vector_names=experimental_vector_names,
        ),
        optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=int(args.indexing_threshold)),
    )
    try:
        client.create_payload_index(
            collection_name=args.collection,
            field_name="filename",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass

    print("➡️ Step 4/4: Copying points back to original…")
    copied2 = _clone(
        client,
        source=temp,
        dest=args.collection,
        embedding_dim=int(args.embedding_dim),
        vector_dtype=str(args.vector_dtype),
        indexing_threshold=int(args.indexing_threshold),
        recreate_dest=False,
        scroll_limit=int(args.scroll_limit),
        upsert_batch_size=int(args.upsert_batch_size),
        experimental_vector_names=experimental_vector_names,
    )

    final_info = client.get_collection(args.collection)
    exact = client.count(collection_name=args.collection, exact=True)
    print("✅ Rebuild complete")
    print(f"   copied_to_temp={copied1} copied_back={copied2}")
    print(
        f"   final.points_count≈{final_info.points_count} final.count(exact)={exact.count} "
        f"final.indexed_vectors_count={final_info.indexed_vectors_count}"
    )

    if not args.keep_temp:
        print("🧹 Deleting temp collection…")
        client.delete_collection(temp)
        print("✅ Temp deleted")
    else:
        print("ℹ️ Temp kept:", temp)


if __name__ == "__main__":
    main()
