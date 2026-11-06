import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _stats(xs: List[int]) -> Dict[str, Any]:
    import numpy as np

    if not xs:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    arr = np.array(xs, dtype=np.int64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": int(arr.min()),
        "max": int(arr.max()),
    }


def _count_tokens(emb) -> int:
    try:
        import torch

        if isinstance(emb, torch.Tensor):
            return int(emb.shape[0])
    except Exception:
        pass
    try:
        return int(emb.shape[0])
    except Exception:
        return int(len(emb))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--model", type=str, default="vidore/colSmol-500M")
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--processor-speed", type=str, default="fast", choices=["fast", "slow", "auto"]
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-filter-special-tokens", action="store_true", default=False)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    datasets: List[str] = []
    if args.datasets:
        datasets = list(args.datasets)
    elif args.dataset:
        datasets = [args.dataset]
    else:
        raise SystemExit("Provide --dataset or --datasets")

    from benchmarks.vidore_beir_qdrant.run_qdrant_beir import _maybe_load_dotenv, _parse_torch_dtype
    from benchmarks.vidore_tatdqa_test.dataset_loader import load_vidore_beir_dataset
    from visual_rag.embedding.visual_embedder import VisualEmbedder

    _maybe_load_dotenv()

    embedder = VisualEmbedder(
        model_name=str(args.model),
        torch_dtype=_parse_torch_dtype(str(args.torch_dtype)),
        batch_size=int(args.batch_size),
        processor_speed=str(args.processor_speed),
    )
    filter_special = not bool(args.no_filter_special_tokens)

    out: Dict[str, Any] = {
        "model": str(args.model),
        "torch_dtype": str(args.torch_dtype),
        "processor_speed": str(args.processor_speed),
        "filter_special_tokens": bool(filter_special),
        "max_queries": int(args.max_queries),
        "datasets": {},
    }

    for ds in datasets:
        _, queries, _ = load_vidore_beir_dataset(ds)
        qs = [q.text for q in queries]
        if int(args.max_queries) and int(args.max_queries) > 0:
            qs = qs[: int(args.max_queries)]
        embs = embedder.embed_queries(
            qs,
            batch_size=int(args.batch_size),
            filter_special_tokens=bool(filter_special),
            show_progress=True,
        )
        token_counts = [_count_tokens(e) for e in embs]
        out["datasets"][str(ds)] = {
            "num_queries": int(len(qs)),
            "token_count": _stats(token_counts),
        }

    text = json.dumps(out, indent=2)
    if args.output:
        p = Path(str(args.output))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print(str(p))
    else:
        print(text)


if __name__ == "__main__":
    main()
