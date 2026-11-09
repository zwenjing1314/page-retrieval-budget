"""Summarize frozen A/B/C/H metrics by the official query language field."""

import csv
import json
import math
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "records"
DATA = ROOT / "data/economics_reports_v2-76fe40166ba07b1bf50457f5c6057cacdd045f10"
SOURCES = {
    "A": RECORDS / "p2-formal-a-20261107-012250-612a76",
    "B": RECORDS / "p2-formal-b-20261107-013537-34e55b",
    "C": RECORDS / "p2-formal-c-20261107-014210-68a2ce",
    "H": RECORDS / "ac-semantic-eval-20261109-172244-5e9662",
}
METRICS = ("NDCG@5", "NDCG@10", "Recall@5", "Recall@10", "Recall@100")


def read_metrics(path):
    with path.open(encoding="utf-8", newline="") as source:
        return {row["query_id"]: row for row in csv.DictReader(source, delimiter="\t")}


def mean(rows, field):
    return math.fsum(float(row[field]) for row in rows) / len(rows)


def main():
    query_rows = pq.read_table(
        DATA / "queries/test-00000-of-00001.parquet",
        columns=["query-id", "language"],
    ).to_pylist()
    languages = {str(row["query-id"]): str(row["language"]).lower() for row in query_rows}
    if len(languages) != 232:
        raise RuntimeError(f"Expected 232 official queries; got {len(languages)}")

    data = {group: read_metrics(path / "per-query.tsv") for group, path in SOURCES.items()}
    query_ids = set(languages)
    if any(set(rows) != query_ids for rows in data.values()):
        raise RuntimeError("Query IDs do not match across language labels and metric records")

    groups = defaultdict(list)
    for qid, language in languages.items():
        groups[language].append(qid)

    result = {}
    for language, qids in sorted(groups.items()):
        rows = {group: [data[group][qid] for qid in qids] for group in SOURCES}
        result[language] = {
            "query_count": len(qids),
            "metrics": {
                group: {metric: mean(rows[group], metric) for metric in METRICS}
                for group in SOURCES
            },
            "H_minus_A": {
                metric: mean(rows["H"], metric) - mean(rows["A"], metric)
                for metric in METRICS
            },
            "H_minus_C": {
                metric: mean(rows["H"], metric) - mean(rows["C"], metric)
                for metric in METRICS
            },
            "H_vs_A_NDCG10": {
                "higher": sum(float(data["H"][qid]["NDCG@10"]) - float(data["A"][qid]["NDCG@10"]) > 1e-12 for qid in qids),
                "equal": sum(abs(float(data["H"][qid]["NDCG@10"]) - float(data["A"][qid]["NDCG@10"])) <= 1e-12 for qid in qids),
                "lower": sum(float(data["H"][qid]["NDCG@10"]) - float(data["A"][qid]["NDCG@10"]) < -1e-12 for qid in qids),
            },
            "H_vs_C_NDCG10": {
                "higher": sum(float(data["H"][qid]["NDCG@10"]) - float(data["C"][qid]["NDCG@10"]) > 1e-12 for qid in qids),
                "equal": sum(abs(float(data["H"][qid]["NDCG@10"]) - float(data["C"][qid]["NDCG@10"])) <= 1e-12 for qid in qids),
                "lower": sum(float(data["H"][qid]["NDCG@10"]) - float(data["C"][qid]["NDCG@10"]) < -1e-12 for qid in qids),
            },
        }

    record = RECORDS / (
        "ac-semantic-language-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        + "-" + uuid.uuid4().hex[:6]
    )
    record.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "PASS",
        "total_queries": len(query_ids),
        "language_counts": {language: len(qids) for language, qids in sorted(groups.items())},
        "by_language": result,
        "scope": "Post-hoc language grouping of frozen A/B/C/H outputs; no model or Qdrant calls",
        "interpretation_limit": "Small language groups are descriptive only and do not establish a causal language effect.",
    }
    (record / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (record / "by-language.tsv").open("w", encoding="utf-8", newline="") as output:
        fields = ["language", "query_count"]
        for group in SOURCES:
            fields.extend(f"{group}_{metric}" for metric in METRICS)
        fields.extend(f"H_minus_{group}_{metric}" for group in ("A", "C") for metric in METRICS)
        writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for language, info in result.items():
            row = {"language": language, "query_count": info["query_count"]}
            for group in SOURCES:
                row.update({f"{group}_{metric}": info["metrics"][group][metric] for metric in METRICS})
            for base in ("A", "C"):
                row.update({f"H_minus_{base}_{metric}": info[f"H_minus_{base}"][metric] for metric in METRICS})
            writer.writerow(row)

    for language, info in result.items():
        print(language, "n=", info["query_count"],
              "H-A NDCG@10=", round(info["H_minus_A"]["NDCG@10"], 6),
              "H-C NDCG@10=", round(info["H_minus_C"]["NDCG@10"], 6),
              "H/A higher-equal-lower=", info["H_vs_A_NDCG10"], flush=True)
    print("Languages:", len(result), "queries:", len(query_ids), flush=True)
    print("Record:", record, flush=True)
    print("Language-stratified comparison: PASS; descriptive only", flush=True)


if __name__ == "__main__":
    main()
