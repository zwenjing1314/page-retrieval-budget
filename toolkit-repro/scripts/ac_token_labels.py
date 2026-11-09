"""Decode cached query-token positions for the page-169 diagnostic.

Tokenizer-only, offline and read-only: no model, encoding, pooling or Qdrant.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import uuid
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/encoded-economics-v1"
DIAGNOSTIC = ROOT / "records/ac-token-diagnostic-20261109-121943-4b79ce"
BASE = ROOT / "models/colpaligemma-3b-pt-448-base-30ab955d073de4a91dc5a288e8c97226647e3e5a"
QUERY_ID = "169"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    sys.dont_write_bytecode = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    record = ROOT / "records" / ("ac-token-labels-" + datetime.now().strftime("%Y%m%d-%H%M%S")
                                 + "-" + uuid.uuid4().hex[:6])
    record.mkdir(parents=True, exist_ok=False)
    print("Record:", record, flush=True)
    report = {"status": "STARTED", "query_id": QUERY_ID,
              "model_loads": 0, "encoding_calls": 0, "pooling_calls": 0,
              "qdrant_calls": 0, "scope": "decode cached token positions only"}
    try:
        import numpy as np
        from transformers import AutoTokenizer

        diagnostic_report = load(DIAGNOSTIC / "report.json")
        require(diagnostic_report["status"] == "PASS_DIAGNOSTIC", "Token diagnostic not accepted")
        profile = load(CACHE / "profile.json")
        query_file = CACHE / "queries" / QUERY_ID / "vectors.npz"
        metadata_file = CACHE / "queries" / QUERY_ID / "metadata.json"
        require(query_file.exists() and metadata_file.exists(), "Query cache missing")
        metadata = load(metadata_file)
        require(metadata["identity"]["query_id"] == QUERY_ID, "Query identity mismatch")
        with np.load(query_file, allow_pickle=False) as arrays:
            input_ids = arrays["input_ids"].reshape(-1).astype(np.int64)
            attention = arrays["attention_mask"].reshape(-1).astype(bool)
            keep = arrays["keep_mask"].reshape(-1).astype(bool)
        require(len(input_ids) == len(attention) == len(keep), "Token arrays disagree")
        tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True, local_files_only=True)
        decoded = []
        for index, (token_id, active, selected) in enumerate(zip(input_ids, attention, keep)):
            token_text = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
            decoded.append({"sequence_position": index, "input_id": int(token_id),
                            "attention": bool(active), "selected_visual_query_token": bool(selected),
                            "decoded": token_text})
        selected_positions = [item["sequence_position"] for item in decoded if item["selected_visual_query_token"]]
        require(len(selected_positions) == 18, "Expected 18 selected query tokens")
        token_rows = []
        diagnostic = load(DIAGNOSTIC / "token-diagnostic.json")
        for page in diagnostic:
            for delta in page["token_deltas"]:
                query_token = int(delta["query_token"])
                sequence_position = selected_positions[query_token]
                token_rows.append({"page": page["corpus_id"], **delta,
                                   "sequence_position": sequence_position,
                                   "input_id": decoded[sequence_position]["input_id"],
                                   "decoded": decoded[sequence_position]["decoded"]})
        report.update({"status": "PASS_DIAGNOSTIC", "cache_profile_sha256":
                       hashlib.sha256(json.dumps(profile, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                       "tokenizer": "AutoTokenizer fast from frozen local base",
                       "tokenizer_path": str(BASE), "query_text": metadata["identity"]["text"],
                       "selected_token_count": len(selected_positions), "all_sequence_tokens": decoded,
                       "selected_sequence_positions": selected_positions, "rows": token_rows,
                       "script_sha256": digest(Path(__file__))})
        save(record / "token-labels.json", report)
        print("Query:", QUERY_ID, metadata["identity"]["text"], flush=True)
        for page in ("60", "304"):
            print("PAGE", page, flush=True)
            rows = [row for row in token_rows if row["page"] == page]
            rows.sort(key=lambda row: float(row["C_minus_A"]))
            for row in rows:
                print(f"  qtoken={row['query_token']:2d} pos={row['sequence_position']:2d} "
                      f"id={row['input_id']} text={row['decoded']!r} "
                      f"A={row['A_max_cosine']:.4f} C={row['C_max_cosine']:.4f} "
                      f"delta={row['C_minus_A']:.4f}", flush=True)
        print("Model loads: 0; encoding calls: 0; pooling calls: 0; Qdrant calls: 0", flush=True)
        print("Token label diagnostic: PASS; no new retrieval", flush=True)
    except Exception as error:
        report.update({"status": "FAIL", "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        save(record / "report.json", report)


if __name__ == "__main__":
    main()
