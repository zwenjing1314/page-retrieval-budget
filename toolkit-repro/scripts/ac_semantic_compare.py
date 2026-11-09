"""Compare the completed H evaluation against the frozen A/B/C evaluations."""

import csv
import json
import math
import uuid
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "records"
A = RECORDS / "p2-formal-a-20261107-012250-612a76"
B = RECORDS / "p2-formal-b-20261107-013537-34e55b"
C = RECORDS / "p2-formal-c-20261107-014210-68a2ce"
H = RECORDS / "ac-semantic-eval-20261109-172244-5e9662"
METRICS = ("NDCG@5", "NDCG@10", "Recall@5", "Recall@10", "Recall@100")


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    return {row["query_id"]: row for row in rows}


def numeric(rows):
    return {
        qid: {name: float(row[name]) for name in METRICS}
        for qid, row in rows.items()
    }


def read_rankings(path):
    if path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(qid): [str(pid) for pid in ranking]
                for qid, ranking in raw.items()}
    result = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                result[str(row["query_id"])] = [str(hit["point_id"])
                                                 for hit in row["hits"]]
    return result


def compare_metric(h, base, name):
    values = [h[qid][name] - base[qid][name] for qid in h]
    return {
        "mean_delta": math.fsum(values) / len(values),
        "higher": sum(value > 1e-12 for value in values),
        "equal": sum(abs(value) <= 1e-12 for value in values),
        "lower": sum(value < -1e-12 for value in values),
        "min_delta": min(values),
        "max_delta": max(values),
    }


def ranking_changes(h, base):
    overlap = []
    changed_top10 = 0
    changed_top100 = 0
    for qid in h:
        h10, b10 = set(h[qid][:10]), set(base[qid][:10])
        h100, b100 = set(h[qid][:100]), set(base[qid][:100])
        overlap.append({
            "query_id": qid,
            "top10_overlap": len(h10 & b10),
            "top100_overlap": len(h100 & b100),
        })
        changed_top10 += h10 != b10
        changed_top100 += h100 != b100
    return {
        "queries_with_changed_top10_set": changed_top10,
        "queries_with_changed_top100_set": changed_top100,
        "mean_top10_overlap": math.fsum(x["top10_overlap"] for x in overlap) / len(overlap),
        "mean_top100_overlap": math.fsum(x["top100_overlap"] for x in overlap) / len(overlap),
    }


def main():
    paths = {"A": A, "B": B, "C": C, "H": H}
    rows = {group: read_tsv(path / "per-query.tsv") for group, path in paths.items()}
    values = {group: numeric(data) for group, data in rows.items()}
    query_ids = set(values["H"])
    if any(set(data) != query_ids for data in values.values()):
        raise RuntimeError("Per-query IDs differ across A/B/C/H")
    if len(query_ids) != 232:
        raise RuntimeError(f"Expected 232 queries, got {len(query_ids)}")

    rankings = {
        "A": read_rankings(A / "rankings.jsonl"),
        "B": read_rankings(B / "rankings.jsonl"),
        "C": read_rankings(C / "rankings.jsonl"),
        "H": read_rankings(H / "rankings.json"),
    }
    if any(set(data) != query_ids for data in rankings.values()):
        raise RuntimeError("Ranking query IDs differ across A/B/C/H")

    record = RECORDS / (
        "ac-semantic-compare-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        + "-" + uuid.uuid4().hex[:6]
    )
    record.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "PASS",
        "query_count": len(query_ids),
        "sources": {group: str(path) for group, path in paths.items()},
        "aggregate_metrics": {
            group: {name: math.fsum(values[group][qid][name] for qid in query_ids) / len(query_ids)
                    for name in METRICS}
            for group in paths
        },
        "metric_comparisons": {
            base: {name: compare_metric(values["H"], values[base], name) for name in METRICS}
            for base in ("A", "B", "C")
        },
        "ranking_comparisons": {
            base: ranking_changes(rankings["H"], rankings[base])
            for base in ("A", "B", "C")
        },
        "scope": "Read-only comparison of frozen per-query outputs; no model/Qdrant calls",
    }
    (record / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for base in ("A", "B", "C"):
        rows_out = []
        for qid in sorted(query_ids, key=int):
            rows_out.append({
                "query_id": qid,
                **{f"H_minus_{base}_{name}": values["H"][qid][name] - values[base][qid][name]
                   for name in METRICS},
                f"H_{base}_top10_overlap": len(set(rankings["H"][qid][:10]) & set(rankings[base][qid][:10])),
                f"H_{base}_top100_overlap": len(set(rankings["H"][qid][:100]) & set(rankings[base][qid][:100])),
            })
        with (record / f"per-query-H-vs-{base}.tsv").open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows_out[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows_out)

    for group in ("A", "B", "C"):
        print(f"H vs {group}:", report["metric_comparisons"][group], flush=True)
        print(f"H/{group} ranking overlap:", report["ranking_comparisons"][group], flush=True)
    print("Compared queries:", len(query_ids), flush=True)
    print("Record:", record, flush=True)
    print("A/B/C/H per-query comparison: PASS", flush=True)


if __name__ == "__main__":
    main()
