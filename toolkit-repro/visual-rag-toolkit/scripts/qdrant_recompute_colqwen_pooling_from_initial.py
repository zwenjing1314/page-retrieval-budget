"""
Recompute ColQwen2.5 pooled vectors from already-indexed `initial` vectors.

Why:
- Your collection already contains high-quality `initial` multi-vectors (single_full works well),
  but two-stage prefetch using `experimental_pooling` is poor.
- We can fix that WITHOUT re-indexing images by recomputing:
  - mean_pooling  (32×dim)  from `initial` (H×W×dim)
  - experimental_pooling (36×dim) from mean_pooling with window=5
  - global_pooling (dim)

How we infer (H, W):
- For each point we know `num_tokens=len(initial)` and the stored resized image aspect ratio.
- We factor `num_tokens` and pick the factor pair (h, w) whose w/h best matches width/height.

Usage:
  python scripts/qdrant_recompute_colqwen_pooling_from_initial.py \
    --collection "vidore_beir_v2_3ds__colqwen25_v0_2__nocrop__union__fp32__grpc" \
    --dataset "vidore/esg_reports_v2" \
    --limit 0
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except Exception:
    DOTENV_AVAILABLE = False

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from visual_rag.embedding.pooling import (
    adaptive_row_mean_pooling_from_grid,
    weighted_row_smoothing_same_length,
)


def _maybe_load_dotenv() -> None:
    if not DOTENV_AVAILABLE:
        return
    if Path(".env").exists():
        load_dotenv(".env")


def _stable_uuid(text: str) -> str:
    import hashlib

    hex_str = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"


def _infer_grid(num_tokens: int, *, width: Optional[int], height: Optional[int]) -> Tuple[int, int]:
    """
    Infer (grid_h, grid_w) such that grid_h*grid_w=num_tokens.

    Picks factor pair closest to the observed aspect ratio (width/height).
    """
    n = int(num_tokens)
    if n <= 0:
        raise ValueError("num_tokens must be > 0")

    # Fallback aspect if missing
    if width and height and int(width) > 0 and int(height) > 0:
        aspect = float(width) / float(height)
    else:
        aspect = 1.0

    best = None
    best_score = float("inf")

    # Enumerate factors up to sqrt(n)
    r = int(math.isqrt(n))
    for h in range(1, r + 1):
        if n % h != 0:
            continue
        w = n // h

        # Consider both orientations
        for hh, ww in ((h, w), (w, h)):
            if hh <= 0 or ww <= 0:
                continue
            cand = float(ww) / float(hh)
            # log-space ratio distance is symmetric and scale-invariant
            score = abs(math.log(max(cand, 1e-9) / max(aspect, 1e-9)))
            if score < best_score:
                best_score = score
                best = (int(hh), int(ww))

    if best is None:
        # Should never happen
        g = int(round(math.sqrt(n)))
        return g, max(1, n // max(1, g))
    return best


def _chunks(xs: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _has_none_nested(v: Any) -> bool:
    try:
        if not isinstance(v, list):
            return True
        if not v:
            return True
        if not isinstance(v[0], list):
            return True
        for row in v:
            if not isinstance(row, list):
                return True
            for x in row:
                if x is None:
                    return True
        return False
    except Exception:
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--dataset", required=True, help="payload['dataset'] value to filter on")
    ap.add_argument("--url", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--scroll-limit", type=int, default=128)
    ap.add_argument("--update-batch", type=int, default=64)
    ap.add_argument(
        "--retrieve-batch", type=int, default=16, help="Batch size for retrieve() calls"
    )
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit")
    ap.add_argument("--sleep-sec", type=float, default=0.0)
    ap.add_argument(
        "--max-mean-pool-vectors",
        type=int,
        default=32,
        help=(
            "Cap adaptive mean pooling rows to at most this many vectors. "
            "Default: 32. If <= 0, no cap is applied (use all effective rows)."
        ),
    )
    ap.add_argument(
        "--pooling-windows",
        "--pooling_windows",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Deprecated (ColQwen now uses technique variants). Ignored. "
            "This script always writes: experimental_pooling (Gaussian alias), "
            "experimental_pooling_gaussian and experimental_pooling_triangular (both k=3)."
        ),
    )
    args = ap.parse_args()

    _maybe_load_dotenv()

    url = args.url or os.getenv("QDRANT_URL") or ""
    if not url:
        raise SystemExit("QDRANT_URL not set (or pass --url)")
    api_key = args.api_key or os.getenv("QDRANT_API_KEY") or None

    client = QdrantClient(
        url=url,
        api_key=api_key,
        prefer_grpc=False,  # avoid DNS issues for 6334 in some envs
        timeout=float(args.timeout),
        check_compatibility=False,
    )

    # Required named vectors for ColQwen experimental variants
    exp_names = [
        "experimental_pooling",
        "experimental_pooling_gaussian",
        "experimental_pooling_triangular",
    ]
    try:
        info = client.get_collection(str(args.collection))
        vectors = info.config.params.vectors or {}
        existing = set(str(k) for k in vectors.keys()) if isinstance(vectors, dict) else set()
    except Exception:
        existing = set()
    missing = set(["mean_pooling", "global_pooling"] + exp_names) - existing if existing else set()
    if missing:
        raise SystemExit(
            f"Collection is missing required named vectors: {sorted(missing)}. "
            "Recreate/rebuild the collection schema to include them before recomputing."
        )

    flt = qm.Filter(
        must=[qm.FieldCondition(key="dataset", match=qm.MatchValue(value=str(args.dataset)))]
    )

    updated = 0
    scanned = 0
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=str(args.collection),
            scroll_filter=flt,
            limit=int(args.scroll_limit),
            offset=next_offset,
            with_payload=True,
            with_vectors=False,  # retrieve vectors per-point to avoid whole-batch parse failures
        )
        if not points:
            break

        pv_batch: List[qm.PointVectors] = []
        ids: List[Any] = [p.id for p in points]
        payload_by_id: Dict[Any, Dict[str, Any]] = {p.id: (p.payload or {}) for p in points}

        # Retrieve initial vectors in batches for speed; fallback to per-id retrieve on failure.
        records_by_id: Dict[Any, Any] = {}
        for id_chunk in _chunks(ids, int(args.retrieve_batch)):
            if not id_chunk:
                continue
            try:
                recs = client.retrieve(
                    collection_name=str(args.collection),
                    ids=id_chunk,
                    with_payload=False,
                    with_vectors=["initial"],
                    timeout=int(args.timeout),
                )
                for r in recs:
                    records_by_id[r.id] = r
            except Exception:
                # fallback: per-id
                for pid in id_chunk:
                    try:
                        recs = client.retrieve(
                            collection_name=str(args.collection),
                            ids=[pid],
                            with_payload=False,
                            with_vectors=["initial"],
                            timeout=int(args.timeout),
                        )
                        if recs:
                            records_by_id[recs[0].id] = recs[0]
                    except Exception:
                        continue

        for pid in ids:
            scanned += 1
            if args.limit and scanned > int(args.limit):
                break

            # Retrieve vectors for this point. Some points in this collection may contain placeholder
            # vectors with nulls from recovery attempts; retrieving per-id lets us skip them safely.
            rec = records_by_id.get(pid)
            if rec is None:
                continue
            vec = (rec.vector or {}).get("initial")
            if _has_none_nested(vec):
                continue

            emb = np.asarray(vec, dtype=np.float32)  # [num_tokens, dim]
            num_tokens = int(emb.shape[0])

            payload = payload_by_id.get(pid) or {}
            w = (
                payload.get("resized_width")
                or payload.get("cropped_width")
                or payload.get("original_width")
            )
            h = (
                payload.get("resized_height")
                or payload.get("cropped_height")
                or payload.get("original_height")
            )
            try:
                w_i = int(w) if w is not None else None
                h_i = int(h) if h is not None else None
            except Exception:
                w_i, h_i = None, None

            grid_h, grid_w = _infer_grid(num_tokens, width=w_i, height=h_i)
            if grid_h * grid_w != num_tokens:
                # safety: if factor inference failed, skip
                continue

            mean_pool = adaptive_row_mean_pooling_from_grid(
                emb,
                grid_h=int(grid_h),
                grid_w=int(grid_w),
                target_rows=(
                    int(grid_h)
                    if int(args.max_mean_pool_vectors) <= 0
                    else min(int(args.max_mean_pool_vectors), int(grid_h))
                ),
                output_dtype=np.float32,
            )
            exp_gaussian = weighted_row_smoothing_same_length(
                mean_pool, window_size=3, kernel="gaussian", output_dtype=np.float32
            )
            exp_triangular = weighted_row_smoothing_same_length(
                mean_pool, window_size=3, kernel="triangular", output_dtype=np.float32
            )
            glob = mean_pool.mean(axis=0).astype(np.float32)

            pv_batch.append(
                qm.PointVectors(
                    id=pid,
                    vector={
                        "mean_pooling": mean_pool.tolist(),
                        "global_pooling": glob.tolist(),
                        "experimental_pooling": exp_gaussian.tolist(),
                        "experimental_pooling_gaussian": exp_gaussian.tolist(),
                        "experimental_pooling_triangular": exp_triangular.tolist(),
                    },
                )
            )

            if len(pv_batch) >= int(args.update_batch):
                client.update_vectors(
                    collection_name=str(args.collection),
                    points=pv_batch,
                    wait=True,
                )
                updated += len(pv_batch)
                print(f"✅ updated vectors: {updated} (scanned={scanned})", flush=True)
                pv_batch = []
                if float(args.sleep_sec) > 0:
                    time.sleep(float(args.sleep_sec))

        if pv_batch:
            client.update_vectors(
                collection_name=str(args.collection),
                points=pv_batch,
                wait=True,
            )
            updated += len(pv_batch)
            print(f"✅ updated vectors: {updated} (scanned={scanned})", flush=True)

        if args.limit and scanned >= int(args.limit):
            break
        if next_offset is None:
            break

    print(f"Done. scanned={scanned}, updated={updated}")


if __name__ == "__main__":
    main()
