from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    image: Any
    payload: Dict[str, Any]


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _stable_uuid(text: str) -> str:
    hex_str = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"


def paired_source_doc_id(row: Mapping[str, Any], idx: int) -> str:
    source_doc_id = _as_str(row.get("_id"))
    if source_doc_id:
        return source_doc_id
    image_filename = _as_str(row.get("image_filename"))
    page = _as_str(row.get("page"))
    return f"{image_filename}::page={page}::idx={idx}"


def paired_doc_id(row: Mapping[str, Any], idx: int) -> str:
    return _stable_uuid(paired_source_doc_id(row, idx))


def paired_payload(row: Mapping[str, Any], idx: int) -> Dict[str, Any]:
    return {
        "source": _as_str(row.get("source")),
        "image_filename": _as_str(row.get("image_filename")),
        "page": _as_str(row.get("page")),
        "source_doc_id": paired_source_doc_id(row, idx),
    }


def _normalize_qrels(qrels_rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    for row in qrels_rows:
        qid = _as_str(row.get("query-id") or row.get("query_id") or row.get("qid"))
        did = _as_str(
            row.get("corpus-id") or row.get("corpus_id") or row.get("doc_id") or row.get("did")
        )
        score = row.get("score") or row.get("relevance") or row.get("label") or 0
        try:
            score_int = int(score)
        except Exception:
            score_int = 0
        if not qid or not did:
            continue
        # Keep qrels compact and correct: score<=0 is non-relevant.
        if score_int <= 0:
            continue
        qrels.setdefault(qid, {})[_stable_uuid(did)] = score_int
    return qrels


def _expect_fields(obj: Any, required: List[str], context: str) -> None:
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(
            f"{context}: missing required field(s): {missing}. Available: {list(obj.keys())}"
        )


def _extract_beir_splits(ds: Any):
    if isinstance(ds, Mapping) and all(k in ds for k in ("corpus", "queries", "qrels")):
        return ds["corpus"], ds["queries"], ds["qrels"]
    if isinstance(ds, Mapping) and "test" in ds:
        test_split = ds["test"]
        if hasattr(test_split, "column_names"):
            cols = set(test_split.column_names)
            if all(k in cols for k in ("corpus", "queries", "qrels")):
                row = test_split[0]
                return row["corpus"], row["queries"], row["qrels"]
    return None


def _first_split(ds: Any):
    if isinstance(ds, Mapping):
        if "test" in ds:
            return ds["test"]
        return ds[next(iter(ds.keys()))]
    return ds


def _get_config_names(dataset_name: str) -> List[str]:
    from datasets import load_dataset_builder

    try:
        builder = load_dataset_builder(dataset_name)
        return list(getattr(builder, "builder_configs", {}).keys())
    except Exception:
        return []


def _normalize_dataset_alias(name: str) -> str:
    s = str(name or "").strip().lower()
    s = re.sub(r"[\s\\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_/]+", "", s)
    s = re.sub(r"_+", "_", s)
    return s


def _resolve_vidore_dataset_name(dataset_name: str) -> str:
    raw = str(dataset_name or "").strip()
    norm = _normalize_dataset_alias(raw)
    if not norm:
        return raw

    aliases = {
        "economics_macro_multilingual": "vidore/economics_reports_v2",
        "economics_macro_multilingual_v2": "vidore/economics_reports_v2",
        "economics_macro_multilingual_eng": "vidore/economics_reports_eng_v2",
        "economics_macro_multilingual_eng_v2": "vidore/economics_reports_eng_v2",
        "economics_reports": "vidore/economics_reports_v2",
        "economics_reports_v2": "vidore/economics_reports_v2",
        "economics_reports_eng": "vidore/economics_reports_eng_v2",
        "economics_reports_eng_v2": "vidore/economics_reports_eng_v2",
    }
    if norm in aliases:
        return aliases[norm]

    candidates: List[str] = []
    if "/" in norm:
        candidates.append(norm)
        repo = norm.rsplit("/", 1)[-1]
        if repo.endswith("_v2"):
            candidates.append(norm[: -(len("_v2"))])
        else:
            candidates.append(f"{norm}_v2")
    elif norm.startswith("vidore/"):
        candidates.append(norm)
        if not norm.endswith("_v2"):
            candidates.append(f"{norm}_v2")
    else:
        if norm.endswith("_v2"):
            candidates.append(f"vidore/{norm}")
        else:
            candidates.append(f"vidore/{norm}_v2")
            candidates.append(f"vidore/{norm}")

    return candidates[0] if candidates else raw


def _load_dataset_with_beir_config(dataset_name: str, config_names: List[str]):
    from datasets import load_dataset

    preferred = [n for n in config_names if "beir" in n.lower()]
    for name in preferred:
        try:
            ds = load_dataset(dataset_name, name=name)
        except Exception:
            continue
        if _extract_beir_splits(ds) is not None:
            return ds
    return None


def _load_beir_from_separate_configs(dataset_name: str, config_names: List[str]):
    from datasets import load_dataset

    def _pick(names: List[str]) -> Optional[str]:
        if not config_names:
            return names[0] if names else None
        for name in names:
            if name in config_names:
                return name
        return None

    corpus_name = _pick(["corpus", "docs"])
    queries_name = _pick(["queries"])
    qrels_name = _pick(["qrels"])
    if not corpus_name or not queries_name or not qrels_name:
        return None

    try:
        corpus_ds = load_dataset(dataset_name, name=corpus_name)
        queries_ds = load_dataset(dataset_name, name=queries_name)
        qrels_ds = load_dataset(dataset_name, name=qrels_name)
    except Exception:
        return None

    return _first_split(corpus_ds), _first_split(queries_ds), _first_split(qrels_ds)


def load_vidore_beir_dataset(
    dataset_name: str,
) -> Tuple[List[CorpusDoc], List[Query], Dict[str, Dict[str, int]]]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("datasets is required. Install with: pip install datasets") from e
    resolved = _resolve_vidore_dataset_name(dataset_name)
    candidates = []
    for cand in [resolved, dataset_name]:
        cand = str(cand or "").strip()
        if not cand:
            continue
        if cand not in candidates:
            candidates.append(cand)
        if "/" in cand:
            repo = cand.rsplit("/", 1)[-1]
            if repo.endswith("_v2"):
                alt = cand[: -(len("_v2"))]
                if alt not in candidates:
                    candidates.append(alt)
            else:
                alt = f"{cand}_v2"
                if alt not in candidates:
                    candidates.append(alt)

    last_err: Optional[Exception] = None
    extracted = None
    used_configs: List[str] = []
    for name_try in candidates:
        config_names = _get_config_names(name_try)
        used_configs = config_names
        ds = None
        if not config_names or "default" in config_names:
            try:
                ds = load_dataset(name_try)
            except Exception as e:
                last_err = e
                ds = None

        extracted = _extract_beir_splits(ds) if ds is not None else None
        if extracted is None:
            ds_beir = _load_dataset_with_beir_config(name_try, config_names)
            if ds_beir is not None:
                extracted = _extract_beir_splits(ds_beir)
            if extracted is None:
                extracted = _load_beir_from_separate_configs(name_try, config_names)
        if extracted is not None:
            break

    if extracted is None:
        if last_err is not None and not used_configs:
            raise ValueError(
                "Could not load dataset (check the dataset id and HF access). "
                f"Tried: {candidates}."
            ) from last_err
        raise ValueError(
            "Dataset does not look like BEIR/ViDoRe-v2 format. "
            f"Tried: {candidates}. Available configs: {used_configs or 'unknown'}"
        )

    corpus_split, queries_split, qrels_split = extracted

    corpus_docs: List[CorpusDoc] = []
    for row in corpus_split:
        if "corpus-id" in row:
            source_doc_id = _as_str(row["corpus-id"])
        elif "_id" in row:
            source_doc_id = _as_str(row["_id"])
        elif "doc-id" in row:
            source_doc_id = _as_str(row["doc-id"])
        else:
            _expect_fields(row, ["_id"], context="corpus row")
            source_doc_id = _as_str(row["_id"])
        doc_id = _stable_uuid(source_doc_id)
        image = row.get("image") or row.get("page_image") or row.get("document") or row.get("img")
        if image is None:
            raise ValueError(
                "corpus row: missing image field (tried image/page_image/document/img)"
            )
        payload = {
            **{
                k: v
                for k, v in row.items()
                if k != "image" and k != "page_image" and k != "document" and k != "img"
            },
            "source_doc_id": source_doc_id,
        }
        corpus_docs.append(CorpusDoc(doc_id=doc_id, image=image, payload=payload))

    queries: List[Query] = []
    for row in queries_split:
        if "_id" in row:
            qid = _as_str(row["_id"])
        elif "query-id" in row:
            qid = _as_str(row["query-id"])
        elif "query_id" in row:
            qid = _as_str(row["query_id"])
        else:
            _expect_fields(row, ["_id"], context="queries row")
            qid = _as_str(row["_id"])
        text = _as_str(row.get("text") or row.get("query") or row.get("question"))
        if not text:
            raise ValueError("queries row: missing text field (tried text/query/question)")
        queries.append(Query(query_id=qid, text=text))

    qrels = _normalize_qrels(qrels_split)
    if not qrels:
        raise ValueError("qrels split parsed to empty mapping; expected non-empty qrels")

    return corpus_docs, queries, qrels


def load_vidore_paired_dataset(
    dataset_name: str,
) -> Tuple[List[CorpusDoc], List[Query], Dict[str, Dict[str, int]]]:
    """
    Load ViDoRe v1-style paired QA datasets.

    Expected shape:
    - single split (usually "test")
    - each row has at least: query + image (+ optional metadata like page/image_filename/source)

    This protocol is "paired": each query is relevant to its paired page image.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("datasets is required. Install with: pip install datasets") from e

    ds = load_dataset(dataset_name, split="test")
    cols = set(ds.column_names)
    if "query" not in cols and "question" not in cols:
        raise ValueError(f"paired dataset: missing query/question column. Got: {sorted(cols)}")
    if "image" not in cols and "page_image" not in cols:
        raise ValueError(f"paired dataset: missing image/page_image column. Got: {sorted(cols)}")

    corpus_docs: List[CorpusDoc] = []
    queries: List[Query] = []
    qrels: Dict[str, Dict[str, int]] = {}

    image_cols = [c for c in ("image", "page_image") if c in cols]
    ds_meta = ds.remove_columns(image_cols) if image_cols else ds

    for idx, row in enumerate(ds_meta):
        query_text = _as_str(row.get("query") or row.get("question"))
        doc_id = paired_doc_id(row, idx)
        query_id = _as_str(row.get("query_id") or row.get("_query_id")) or f"q_{idx}"
        payload = paired_payload(row, idx)

        corpus_docs.append(CorpusDoc(doc_id=doc_id, image=None, payload=payload))
        queries.append(Query(query_id=query_id, text=query_text))
        qrels[query_id] = {doc_id: 1}

    return corpus_docs, queries, qrels


def load_vidore_dataset_auto(
    dataset_name: str,
) -> Tuple[List[CorpusDoc], List[Query], Dict[str, Dict[str, int]], str]:
    """
    Auto-detect ViDoRe dataset format.
    Returns: (corpus, queries, qrels, protocol)
    protocol in {"beir", "paired"}.
    """
    try:
        corpus, queries, qrels = load_vidore_beir_dataset(dataset_name)
        return corpus, queries, qrels, "beir"
    except ValueError:
        corpus, queries, qrels = load_vidore_paired_dataset(dataset_name)
        return corpus, queries, qrels, "paired"
