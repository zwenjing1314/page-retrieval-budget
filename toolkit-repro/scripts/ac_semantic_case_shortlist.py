"""Create a read-only shortlist of representative H/A/C query cases."""

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "records"
DATA = ROOT / "data/economics_reports_v2-76fe40166ba07b1bf50457f5c6057cacdd045f10"
A = RECORDS / "p2-formal-a-20261107-012250-612a76"
B = RECORDS / "p2-formal-b-20261107-013537-34e55b"
C = RECORDS / "p2-formal-c-20261107-014210-68a2ce"
H = RECORDS / "ac-semantic-eval-20261109-172244-5e9662"


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as source:
        return {row["query_id"]: row for row in csv.DictReader(source, delimiter="\t")}


def read_rankings(path):
    result = {}
    if path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(qid): [str(pid) for pid in ranking] for qid, ranking in raw.items()}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                result[str(row["query_id"])] = [str(hit["point_id"]) for hit in row["hits"]]
    return result


def metric(rows, qid, name):
    return float(rows[qid][name])


def best_positive_rank(ranking, relevant):
    positions = [index + 1 for index, point_id in enumerate(ranking) if point_id in relevant]
    return min(positions) if positions else None


def top_positive_count(ranking, relevant, k):
    return len(set(ranking[:k]) & relevant)


def case_row(qid, query_text, qrels, metrics, rankings, mapping):
    relevant = {str(point_id) for point_id, grade in qrels[qid].items() if grade > 0}
    return {
        "query_id": qid,
        "query": query_text,
        "relevant_pages": len(relevant),
        "ndcg10": {group: metric(metrics[group], qid, "NDCG@10") for group in ("A", "B", "C", "H")},
        "recall10": {group: metric(metrics[group], qid, "Recall@10") for group in ("A", "B", "C", "H")},
        "h_minus_a_ndcg10": metric(metrics["H"], qid, "NDCG@10") - metric(metrics["A"], qid, "NDCG@10"),
        "h_minus_c_ndcg10": metric(metrics["H"], qid, "NDCG@10") - metric(metrics["C"], qid, "NDCG@10"),
        "top10_overlap_with_a": len(set(rankings["H"][qid][:10]) & set(rankings["A"][qid][:10])),
        "top10_overlap_with_c": len(set(rankings["H"][qid][:10]) & set(rankings["C"][qid][:10])),
        "best_positive_rank": {
            group: best_positive_rank(rankings[group][qid], relevant)
            for group in ("A", "B", "C", "H")
        },
        "positive_count_at_10": {
            group: top_positive_count(rankings[group][qid], relevant, 10)
            for group in ("A", "B", "C", "H")
        },
        "h_top10": [
            {"point_id": point_id, "corpus_id": mapping[point_id]["corpus_id"]}
            for point_id in rankings["H"][qid][:10]
        ],
        "a_top10": [
            {"point_id": point_id, "corpus_id": mapping[point_id]["corpus_id"]}
            for point_id in rankings["A"][qid][:10]
        ],
        "c_top10": [
            {"point_id": point_id, "corpus_id": mapping[point_id]["corpus_id"]}
            for point_id in rankings["C"][qid][:10]
        ],
    }


def main():
    metric_rows = {
        "A": read_tsv(A / "per-query.tsv"),
        "B": read_tsv(B / "per-query.tsv"),
        "C": read_tsv(C / "per-query.tsv"),
        "H": read_tsv(H / "per-query.tsv"),
    }
    rankings = {
        "A": read_rankings(A / "rankings.jsonl"),
        "B": read_rankings(B / "rankings.jsonl"),
        "C": read_rankings(C / "rankings.jsonl"),
        "H": read_rankings(H / "rankings.json"),
    }
    query_ids = set(metric_rows["H"])
    if len(query_ids) != 232 or any(set(rows) != query_ids for rows in metric_rows.values()):
        raise RuntimeError("Expected identical 232-query metric files")
    if any(set(rows) != query_ids for rows in rankings.values()):
        raise RuntimeError("Expected identical 232-query ranking files")

    mapping_rows = json.loads((A / "page-id-mapping.json").read_text(encoding="utf-8"))
    mapping = {str(row["point_id"]): row for row in mapping_rows}
    qrels = json.loads((A / "qrels.json").read_text(encoding="utf-8"))
    query_rows = pq.read_table(DATA / "queries/test-00000-of-00001.parquet",
                               columns=["query-id", "query"]).to_pylist()
    query_text = {str(row["query-id"]): row["query"] for row in query_rows}

    rows = {
        qid: case_row(qid, query_text[qid], qrels, metric_rows, rankings, mapping)
        for qid in sorted(query_ids, key=int)
    }
    all_cases = list(rows.values())
    shortlist = {
        "h_recovers_over_c": sorted(
            [row for row in all_cases if row["h_minus_c_ndcg10"] > 0 and row["h_minus_a_ndcg10"] < 0],
            key=lambda row: row["h_minus_c_ndcg10"], reverse=True,
        )[:10],
        "h_regresses_from_a": sorted(
            all_cases, key=lambda row: row["h_minus_a_ndcg10"]
        )[:10],
        "h_worse_than_c": sorted(
            [row for row in all_cases if row["h_minus_c_ndcg10"] < 0],
            key=lambda row: row["h_minus_c_ndcg10"],
        )[:10],
        "h_close_to_a": sorted(
            all_cases, key=lambda row: abs(row["h_minus_a_ndcg10"])
        )[:10],
    }
    record = RECORDS / (
        "ac-semantic-cases-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        + "-" + uuid.uuid4().hex[:6]
    )
    record.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "PASS",
        "query_count": len(all_cases),
        "scope": "Read-only case selection from frozen A/B/C/H records",
        "categories": {name: len(items) for name, items in shortlist.items()},
    }
    (record / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (record / "shortlist.json").write_text(json.dumps(shortlist, ensure_ascii=False, indent=2), encoding="utf-8")
    (record / "all-cases.json").write_text(json.dumps(all_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, items in shortlist.items():
        print(name + ":", [item["query_id"] for item in items], flush=True)
    print("Compared queries:", len(all_cases), flush=True)
    print("Record:", record, flush=True)
    print("A/B/C/H case shortlist: PASS", flush=True)


if __name__ == "__main__":
    main()
