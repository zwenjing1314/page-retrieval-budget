"""
Qdrant collection debugging / inspection helpers.

This script is intentionally lightweight and safe:
- Read-only operations (count/scroll/retrieve)
- Prints exact vs approximate counts (Qdrant UI often shows approximate)
- Can verify that IDs listed in index_failures__*.jsonl logs are actually present

Examples:

  # Inspect counts + vector sanity (REST)
  python scripts/qdrant_debug_collection.py \\
    --collection vidore_beir_v2_3ds__colqwen25_v0_2__nocrop__union__fp32__grpc

  # Same, but via gRPC
  python scripts/qdrant_debug_collection.py \\
    --collection vidore_beir_v2_3ds__colqwen25_v0_2__nocrop__union__fp32__grpc \\
    --prefer-grpc

  # Count per dataset (exact)
  python scripts/qdrant_debug_collection.py \\
    --collection <COLLECTION> \\
    --datasets vidore/esg_reports_v2 vidore/biomedical_lectures_v2 vidore/economics_reports_v2

  # Verify that any IDs in index_failures logs are present in Qdrant
  python scripts/qdrant_debug_collection.py \\
    --collection <COLLECTION> \\
    --check-failures

Environment:
  export QDRANT_URL=...
  export QDRANT_API_KEY=...  # optional

Or create a .env in repo root with the same variables.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    for p in (Path(".env"), Path("..") / ".env"):
        if p.exists():
            load_dotenv(p)


def _chunks(xs: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def inspect_collection(*, client: QdrantClient, collection: str, sample_points: int) -> None:
    info = client.get_collection(collection)
    print("collection:", collection)
    print("status:", info.status)
    print("optimizer_status:", info.optimizer_status)
    print("indexed_vectors_count:", info.indexed_vectors_count)
    print()

    approx = client.count(collection_name=collection, exact=False).count
    exact = client.count(collection_name=collection, exact=True).count
    print("info.points_count (approx):", info.points_count)
    print("count(exact=False):", approx)
    print("count(exact=True): ", exact)

    if sample_points > 0:
        points, _ = client.scroll(
            collection_name=collection,
            limit=int(sample_points),
            with_payload=True,
            with_vectors=True,
        )

        print("\nSample points vector sanity:")
        for p in points:
            vecs = p.vector or {}

            def _len(v):
                try:
                    return len(v)
                except Exception:
                    return None

            lengths = {
                k: _len(v)
                for k, v in vecs.items()
                if k in {"initial", "mean_pooling", "experimental_pooling", "global_pooling"}
            }
            print("id:", p.id)
            print("  dataset:", (p.payload or {}).get("dataset"))
            print("  vector_keys:", sorted(list(vecs.keys())))
            print("  lengths:", lengths)


def count_per_dataset(*, client: QdrantClient, collection: str, datasets: list[str]) -> None:
    if not datasets:
        return
    print("\nper-dataset exact counts:")
    total = 0
    for ds in datasets:
        c = client.count(
            collection_name=collection,
            count_filter=qm.Filter(
                must=[
                    qm.FieldCondition(key="dataset", match=qm.MatchValue(value=str(ds))),
                ]
            ),
            exact=True,
        ).count
        print(f"- {ds}: {c}")
        total += int(c)
    print("sum_datasets_exact:", total)


def dataset_distribution_scroll(*, client: QdrantClient, collection: str, limit: int) -> None:
    values: Counter[str] = Counter()
    offset = None
    seen = 0
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=min(int(limit), 2048),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            ds = (p.payload or {}).get("dataset")
            values[str(ds)] += 1
        seen += len(points)
        offset = next_offset
        if next_offset is None or (limit and seen >= int(limit)):
            break

    print("\nscroll distribution (dataset field):")
    print("scrolled_points:", seen)
    for k, v in values.most_common(20):
        print(" ", k, v)


def check_failure_logs_present(
    *,
    client: QdrantClient,
    collection: str,
    results_dir: Path,
    retrieve_batch: int,
) -> None:
    base = results_dir / collection
    if not base.exists():
        raise SystemExit(f"results dir not found: {base}")

    log_paths = sorted(base.glob("index_failures__*.jsonl"))
    if not log_paths:
        print("\nNo failure logs found under:", base)
        return

    failed_ids: set[str] = set()
    for p in log_paths:
        for line in p.read_text().splitlines():
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            u = obj.get("union_doc_id")
            if u:
                failed_ids.add(str(u))

    print("\nfailure logs:")
    for p in log_paths:
        print(" -", p)
    print("failed_ids_in_logs:", len(failed_ids))

    missing: list[str] = []
    ids = list(failed_ids)
    for chunk in _chunks(ids, int(retrieve_batch)):
        pts = client.retrieve(
            collection_name=collection,
            ids=chunk,
            with_payload=False,
            with_vectors=False,
        )
        present = set(str(p.id) for p in pts)
        for pid in chunk:
            if pid not in present:
                missing.append(pid)

    print("failed_ids_missing_in_qdrant:", len(missing))
    if missing:
        print("sample_missing_ids:", missing[:10])


def main() -> None:
    p = argparse.ArgumentParser(description="Qdrant collection debug utilities")
    p.add_argument("--collection", required=True)
    p.add_argument("--prefer-grpc", action="store_true", default=False)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--datasets", nargs="*", default=[])
    p.add_argument("--sample-points", type=int, default=5)
    p.add_argument("--scroll-limit", type=int, default=0, help="0 = no full scroll distribution")
    p.add_argument("--check-failures", action="store_true", default=False)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--retrieve-batch", type=int, default=64)
    args = p.parse_args()

    _maybe_load_dotenv()
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_API_KEY")
    if not url:
        raise SystemExit("QDRANT_URL not set")

    client = QdrantClient(
        url=url,
        api_key=key,
        prefer_grpc=bool(args.prefer_grpc),
        timeout=int(args.timeout),
    )

    inspect_collection(
        client=client, collection=str(args.collection), sample_points=int(args.sample_points)
    )
    count_per_dataset(client=client, collection=str(args.collection), datasets=list(args.datasets))

    if int(args.scroll_limit) > 0:
        dataset_distribution_scroll(
            client=client, collection=str(args.collection), limit=int(args.scroll_limit)
        )

    if bool(args.check_failures):
        check_failure_logs_present(
            client=client,
            collection=str(args.collection),
            results_dir=Path(str(args.results_dir)),
            retrieve_batch=int(args.retrieve_batch),
        )


if __name__ == "__main__":
    main()
