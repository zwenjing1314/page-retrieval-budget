import argparse
import json
import os
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str, required=True)
    parser.add_argument("--prefer-grpc", action="store_true")
    parser.add_argument("--url", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    _maybe_load_dotenv()

    qdrant_url = args.url or os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise ValueError("QDRANT_URL not set")
    qdrant_api_key = args.api_key or os.getenv("QDRANT_API_KEY")

    from qdrant_client import QdrantClient

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=args.prefer_grpc,
        check_compatibility=False,
        timeout=120,
    )

    info = client.get_collection(args.collection)
    payload_schema = getattr(info, "payload_schema", None)
    snap = {
        "collection": args.collection,
        "points_count": _as_jsonable(getattr(info, "points_count", None)),
        "indexed_vectors_count": _as_jsonable(getattr(info, "indexed_vectors_count", None)),
        "segments_count": _as_jsonable(getattr(info, "segments_count", None)),
        "status": _as_jsonable(
            getattr(getattr(info, "status", None), "value", getattr(info, "status", None))
        ),
        "optimizer_status": _as_jsonable(
            getattr(
                getattr(info, "optimizer_status", None),
                "value",
                getattr(info, "optimizer_status", None),
            )
        ),
        "config": _as_jsonable(getattr(info, "config", None)),
        "payload_schema": _as_jsonable(payload_schema),
    }

    print(json.dumps(snap, indent=2)[:10000])
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(snap, f, indent=2)


if __name__ == "__main__":
    main()
