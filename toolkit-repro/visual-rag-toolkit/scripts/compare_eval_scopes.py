"""
Run the same benchmark twice (union vs per_dataset) and print full reports + deltas.

This is designed to answer: "How much do distractors (union scope) hurt vs per-dataset filtering?"

It runs:
  python -m benchmarks.vidore_beir_qdrant.run_qdrant_beir ... --evaluation-scope union
  python -m benchmarks.vidore_beir_qdrant.run_qdrant_beir ... --evaluation-scope per_dataset

Then prints, per dataset:
  - full metrics dict for union
  - full metrics dict for per_dataset
  - delta = per_dataset - union (for numeric metrics)

Usage example:
  python scripts/compare_eval_scopes.py \\
    --datasets vidore/esg_reports_v2 vidore/biomedical_lectures_v2 vidore/economics_reports_v2 \\
    --collection vidore_beir_v2_3ds__colpali_v1_3__nocrop__union \\
    --model vidore/colpali-v1.3 \\
    --mode single_full \\
    --top-k 100
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _as_number(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return float(int(x))
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _delta_metrics(per_ds: Dict[str, Any], union: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = set(per_ds.keys()) | set(union.keys())
    for k in sorted(keys):
        a = _as_number(per_ds.get(k))
        b = _as_number(union.get(k))
        if a is not None and b is not None:
            out[k] = a - b
        else:
            # keep non-numerics as a tuple when present in either
            if k in per_ds or k in union:
                out[k] = {"per_dataset": per_ds.get(k), "union": union.get(k)}
    return out


def _load_metrics_by_dataset(path: Path) -> Dict[str, Dict[str, Any]]:
    obj = json.loads(path.read_text())
    mbd = obj.get("metrics_by_dataset") or {}
    if not isinstance(mbd, dict):
        return {}
    # ensure nested dicts
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in mbd.items():
        if isinstance(v, dict):
            out[str(k)] = v
    return out


def _run_once(
    *,
    datasets: List[str],
    collection: str,
    model: str,
    mode: str,
    top_k: int,
    stage1_mode: Optional[str],
    prefetch_k: Optional[int],
    stage1_k: Optional[int],
    stage2_k: Optional[int],
    torch_dtype: str,
    qdrant_vector_dtype: str,
    prefer_grpc: bool,
    max_queries: int,
    evaluation_scope: str,
    qdrant_timeout: int,
    qdrant_retries: int,
    qdrant_retry_sleep: float,
    extra_args: List[str],
    out_path: Path,
) -> None:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "benchmarks.vidore_beir_qdrant.run_qdrant_beir",
        "--datasets",
        *datasets,
        "--collection",
        collection,
        "--model",
        model,
        "--mode",
        mode,
        "--top-k",
        str(int(top_k)),
        "--evaluation-scope",
        str(evaluation_scope),
        "--torch-dtype",
        torch_dtype,
        "--qdrant-vector-dtype",
        qdrant_vector_dtype,
        "--qdrant-timeout",
        str(int(qdrant_timeout)),
        "--qdrant-retries",
        str(int(qdrant_retries)),
        "--qdrant-retry-sleep",
        str(float(qdrant_retry_sleep)),
        "--max-queries",
        str(int(max_queries)),
        "--output",
        str(out_path),
    ]

    if not prefer_grpc:
        cmd.append("--no-prefer-grpc")
    else:
        cmd.append("--prefer-grpc")

    if str(mode) == "two_stage":
        if stage1_mode:
            cmd += ["--stage1-mode", str(stage1_mode)]
        if prefetch_k is not None:
            cmd += ["--prefetch-k", str(int(prefetch_k))]
    if str(mode) == "three_stage":
        if stage1_k is not None:
            cmd += ["--stage1-k", str(int(stage1_k))]
        if stage2_k is not None:
            cmd += ["--stage2-k", str(int(stage2_k))]

    cmd += list(extra_args or [])

    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_XET", "1")  # avoid xet crashes in some environments

    print("\n" + "=" * 90)
    print(f"RUN scope={evaluation_scope}")
    print(" ".join(cmd))
    print("=" * 90)
    sys.stdout.flush()

    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--mode", default="single_full", choices=["single_full", "two_stage", "three_stage"]
    )
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument(
        "--stage1-mode",
        default="",
        help="two_stage stage1 mode (e.g. tokens_vs_experimental_pooling or tokens_vs_standard_pooling)",
    )
    ap.add_argument("--prefetch-k", type=int, default=256)
    ap.add_argument("--stage1-k", type=int, default=1000)
    ap.add_argument("--stage2-k", type=int, default=300)
    ap.add_argument(
        "--torch-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"]
    )
    ap.add_argument("--qdrant-vector-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--prefer-grpc", action="store_true", default=False)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--qdrant-timeout", type=int, default=120)
    ap.add_argument("--qdrant-retries", type=int, default=3)
    ap.add_argument("--qdrant-retry-sleep", type=float, default=0.5)
    ap.add_argument(
        "--out-dir",
        default="results/scope_comparisons",
        help="Directory to write the two raw JSON reports + the merged report",
    )
    ap.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Pass-through extra args to run_qdrant_beir (repeatable), e.g. --extra-arg --crop-empty",
    )
    args = ap.parse_args()

    datasets = [str(x) for x in args.datasets]
    stage1_mode = str(args.stage1_mode).strip() or None

    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _now_tag()

    base = f"{tag}__{Path(args.collection).name}__{Path(args.model).name}__{args.mode}"
    union_path = out_dir / f"{base}__scope_union.json"
    per_path = out_dir / f"{base}__scope_per_dataset.json"
    merged_path = out_dir / f"{base}__scope_compare.json"

    _run_once(
        datasets=datasets,
        collection=str(args.collection),
        model=str(args.model),
        mode=str(args.mode),
        top_k=int(args.top_k),
        stage1_mode=stage1_mode,
        prefetch_k=int(args.prefetch_k),
        stage1_k=int(args.stage1_k),
        stage2_k=int(args.stage2_k),
        torch_dtype=str(args.torch_dtype),
        qdrant_vector_dtype=str(args.qdrant_vector_dtype),
        prefer_grpc=bool(args.prefer_grpc),
        max_queries=int(args.max_queries),
        evaluation_scope="union",
        qdrant_timeout=int(args.qdrant_timeout),
        qdrant_retries=int(args.qdrant_retries),
        qdrant_retry_sleep=float(args.qdrant_retry_sleep),
        extra_args=list(args.extra_arg or []),
        out_path=union_path,
    )
    _run_once(
        datasets=datasets,
        collection=str(args.collection),
        model=str(args.model),
        mode=str(args.mode),
        top_k=int(args.top_k),
        stage1_mode=stage1_mode,
        prefetch_k=int(args.prefetch_k),
        stage1_k=int(args.stage1_k),
        stage2_k=int(args.stage2_k),
        torch_dtype=str(args.torch_dtype),
        qdrant_vector_dtype=str(args.qdrant_vector_dtype),
        prefer_grpc=bool(args.prefer_grpc),
        max_queries=int(args.max_queries),
        evaluation_scope="per_dataset",
        qdrant_timeout=int(args.qdrant_timeout),
        qdrant_retries=int(args.qdrant_retries),
        qdrant_retry_sleep=float(args.qdrant_retry_sleep),
        extra_args=list(args.extra_arg or []),
        out_path=per_path,
    )

    union_mbd = _load_metrics_by_dataset(union_path)
    per_mbd = _load_metrics_by_dataset(per_path)

    all_ds = sorted(set(union_mbd.keys()) | set(per_mbd.keys()))
    comparison: Dict[str, Any] = {
        "meta": {
            "datasets": datasets,
            "collection": str(args.collection),
            "model": str(args.model),
            "mode": str(args.mode),
            "top_k": int(args.top_k),
            "stage1_mode": stage1_mode,
            "prefetch_k": int(args.prefetch_k) if str(args.mode) == "two_stage" else None,
            "stage1_k": int(args.stage1_k) if str(args.mode) == "three_stage" else None,
            "stage2_k": int(args.stage2_k) if str(args.mode) == "three_stage" else None,
            "torch_dtype": str(args.torch_dtype),
            "qdrant_vector_dtype": str(args.qdrant_vector_dtype),
            "prefer_grpc": bool(args.prefer_grpc),
            "max_queries": int(args.max_queries),
            "union_report": str(union_path),
            "per_dataset_report": str(per_path),
        },
        "by_dataset": {},
    }

    print("\n" + "#" * 90)
    print("SCOPE COMPARISON (per_dataset − union)")
    print("#" * 90)
    for ds in all_ds:
        u = union_mbd.get(ds, {})
        p = per_mbd.get(ds, {})
        d = _delta_metrics(p, u)
        comparison["by_dataset"][ds] = {
            "union": u,
            "per_dataset": p,
            "delta": d,
        }
        print("\n" + "-" * 90)
        print(ds)
        print("-" * 90)
        print("UNION:")
        print(json.dumps(u, indent=2, sort_keys=True))
        print("\nPER_DATASET:")
        print(json.dumps(p, indent=2, sort_keys=True))
        print("\nDELTA (per_dataset - union):")
        print(json.dumps(d, indent=2, sort_keys=True))
        sys.stdout.flush()

    merged_path.write_text(json.dumps(comparison, indent=2, sort_keys=True))
    print("\nWrote merged comparison:", merged_path)


if __name__ == "__main__":
    main()
