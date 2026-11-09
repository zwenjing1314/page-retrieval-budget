"""Export four representative H/A/C cases for human inspection.

The selected cases come from the frozen comparison shortlist. This is a
read-only export: it decodes original parquet images and reads saved rankings;
it does not load the model, encode, pool, or query Qdrant.
"""

import base64
import csv
import html
import io
import json
import uuid
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "records"
DATA = ROOT / "data/economics_reports_v2-76fe40166ba07b1bf50457f5c6057cacdd045f10"
A = RECORDS / "p2-formal-a-20261107-012250-612a76"
B = RECORDS / "p2-formal-b-20261107-013537-34e55b"
C = RECORDS / "p2-formal-c-20261107-014210-68a2ce"
H = RECORDS / "ac-semantic-eval-20261109-172244-5e9662"
CASES = [("98", "H recovers over C"), ("111", "H regresses from A"),
         ("102", "H worse than C"), ("24", "H close to A")]


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


def metric_table(qid, metrics):
    names = ("NDCG@5", "NDCG@10", "Recall@5", "Recall@10", "Recall@100")
    rows = []
    for group in ("A", "B", "C", "H"):
        rows.append("<tr><th>" + group + "</th>" + "".join(
            "<td>" + format(float(metrics[group][qid][name]), ".6f") + "</td>"
            for name in names) + "</tr>")
    header = "<tr><th>组</th>" + "".join("<th>" + name + "</th>" for name in names) + "</tr>"
    return "<table><thead>" + header + "</thead><tbody>" + "".join(rows) + "</tbody></table>"


def ranking_table(qid, rankings, mapping, qrels):
    positive = {str(pid) for pid, grade in qrels[qid].items() if grade > 0}
    body = []
    for group in ("A", "C", "H"):
        for rank, point_id in enumerate(rankings[group][qid][:10], 1):
            row = mapping[point_id]
            marker = "相关" if point_id in positive else "非正标注"
            body.append("<tr><td>" + group + "</td><td>" + str(rank) + "</td><td>" +
                        html.escape(str(row["corpus_id"])) + "</td><td>" + marker + "</td></tr>")
    return ("<table><thead><tr><th>组</th><th>排名</th><th>corpus-id</th>"
            "<th>qrels</th></tr></thead><tbody>" + "".join(body) + "</tbody></table>")


def main():
    metrics = {group: read_tsv(path / "per-query.tsv") for group, path in
               {"A": A, "B": B, "C": C, "H": H}.items()}
    rankings = {"A": read_rankings(A / "rankings.jsonl"),
                "C": read_rankings(C / "rankings.jsonl"),
                "H": read_rankings(H / "rankings.json")}
    mapping = {str(row["point_id"]): row for row in
               json.loads((A / "page-id-mapping.json").read_text(encoding="utf-8"))}
    qrels = json.loads((A / "qrels.json").read_text(encoding="utf-8"))
    query_rows = pq.read_table(DATA / "queries/test-00000-of-00001.parquet",
                               columns=["query-id", "query", "language"]).to_pylist()
    query_text = {str(row["query-id"]): row for row in query_rows}
    page_ids = set()
    for qid, _ in CASES:
        for group in ("A", "C", "H"):
            page_ids.update(mapping[point_id]["corpus_id"] for point_id in rankings[group][qid][:10])
    pages = pq.read_table(DATA / "corpus/test-00000-of-00001.parquet",
                          columns=["corpus-id", "doc-id", "image"],
                          filters=[("corpus-id", "in", [int(pid) for pid in page_ids])]).to_pylist()
    page_images = {}
    for page in pages:
        with Image.open(io.BytesIO(page["image"]["bytes"])) as image:
            image.load()
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            page_images[str(page["corpus-id"])] = (page["doc-id"], base64.b64encode(buffer.getvalue()).decode("ascii"))

    sections = []
    exported = []
    for qid, category in CASES:
        query = query_text[qid]
        relevant = {str(pid) for pid, grade in qrels[qid].items() if grade > 0}
        ndcg_a = float(metrics["A"][qid]["NDCG@10"])
        ndcg_h = float(metrics["H"][qid]["NDCG@10"])
        ndcg_c = float(metrics["C"][qid]["NDCG@10"])
        cards = []
        shown = set()
        for group in ("A", "C", "H"):
            for point_id in rankings[group][qid][:10]:
                corpus_id = mapping[point_id]["corpus_id"]
                if corpus_id in shown or corpus_id not in page_images:
                    continue
                shown.add(corpus_id)
                doc_id, encoded = page_images[corpus_id]
                cards.append("<figure><figcaption>" + group + "；corpus-id " + html.escape(corpus_id) +
                             "；" + ("qrels 正相关" if point_id in relevant else "无正相关标注") +
                             "；" + html.escape(doc_id) + "</figcaption><img src='data:image/png;base64," +
                             encoded + "'></figure>")
        sections.append("<section><h2>题目 " + qid + "：" + html.escape(category) + "</h2>"+
                        "<p>" + html.escape(query["query"]) + "（" + html.escape(query["language"]) + "）</p>"+
                        "<p>NDCG@10：A=" + format(ndcg_a, ".6f") + "；C=" + format(ndcg_c, ".6f") +
                        "；H=" + format(ndcg_h, ".6f") + "</p>" + metric_table(qid, metrics) +
                        ranking_table(qid, rankings, mapping, qrels) + "<div class='grid'>" + "".join(cards) + "</div></section>")
        exported.append({"query_id": qid, "category": category, "query": query,
                         "page_count_exported": len(shown), "h_minus_a_ndcg10": ndcg_h - ndcg_a,
                         "h_minus_c_ndcg10": ndcg_h - ndcg_c})

    record = RECORDS / ("ac-semantic-case-batch-" + datetime.now().strftime("%Y%m%d-%H%M%S") +
                        "-" + uuid.uuid4().hex[:6])
    record.mkdir(parents=True, exist_ok=False)
    document = (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>H/A/C 代表案例</title>"
        "<style>body{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:24px;color:#172033}"
        "table{border-collapse:collapse;margin:12px 0}th,td{border:1px solid #ccd3df;padding:5px 8px}"
        ".grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}figure{margin:0}img{max-width:100%;height:auto;border:1px solid #ccd3df}"
        "section{border-top:3px solid #172033;margin-top:32px;padding-top:16px}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>"
        "<h1>Economics：H/A/C 四类代表案例</h1>" + "".join(sections) + "</html>"
    )
    (record / "cases.html").write_text(document, encoding="utf-8")
    (record / "cases.json").write_text(json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Selected cases:", [qid for qid, _ in CASES], flush=True)
    print("Exported pages:", len(page_images), flush=True)
    print("Open in browser:", record / "cases.html", flush=True)
    print("H/A/C representative case export: PASS; no model/Qdrant calls", flush=True)


if __name__ == "__main__":
    main()
