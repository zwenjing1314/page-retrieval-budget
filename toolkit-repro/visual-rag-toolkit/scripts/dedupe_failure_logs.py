import argparse
import json
import os
import tempfile
from pathlib import Path


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            yield json.loads(s)


def _key(obj: dict) -> str:
    for k in ("union_doc_id", "id", "doc_id", "source_doc_id"):
        v = obj.get(k)
        if v:
            return str(v)
    return json.dumps(obj, sort_keys=True)


def _write_atomic(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for ln in lines:
                f.write(ln)
                f.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass


def dedupe_jsonl(path: Path) -> dict:
    last_by_key: dict[str, dict] = {}
    order: list[str] = []
    for obj in _iter_jsonl(path):
        k = _key(obj)
        if k not in last_by_key:
            order.append(k)
        last_by_key[k] = obj

    out_lines = [json.dumps(last_by_key[k], ensure_ascii=False) for k in order]
    _write_atomic(path, out_lines)
    return {"path": str(path), "unique": len(out_lines)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=str, nargs="+", required=True)
    args = parser.parse_args()

    for p in args.paths:
        path = Path(p)
        res = dedupe_jsonl(path)
        print(f"{res['path']}: unique={res['unique']}")


if __name__ == "__main__":
    main()
