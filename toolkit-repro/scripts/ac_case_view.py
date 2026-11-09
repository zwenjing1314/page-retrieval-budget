"""Export a saved A/C example for human reading; no encoding or new retrieval.

Reads query 169 and original pages 60/304. Creates a fresh evidence directory
with original-resolution PNGs, a self-contained local HTML view and source hashes.
"""

import csv
import hashlib
from html import escape
import io
import json
import math
import os
from pathlib import Path
import sys
import traceback
import uuid
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "records/ac-semantic-pilot-20261108-160234-1a6cd4"
SOURCES = {
    "A": ROOT / "records/p2-formal-a-20261107-012250-612a76",
    "C": ROOT / "records/p2-formal-c-20261107-014210-68a2ce",
}
QUERY_ID = "169"
PAGE_IDS = ("60", "304")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    sys.dont_write_bytecode = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    record = ROOT / "records" / ("ac-case-view-" + datetime.now().strftime("%Y%m%d-%H%M%S")
                                 + "-" + uuid.uuid4().hex[:6])
    record.mkdir(parents=True, exist_ok=False)
    print("Record:", record, flush=True)
    report = {"status": "STARTED", "query_id": QUERY_ID, "corpus_ids": list(PAGE_IDS),
              "selection": "Largest saved NDCG@10 drop; an explanatory case, not an unbiased effect estimate",
              "model_loads": 0, "encoding_calls": 0, "pooling_calls": 0, "qdrant_calls": 0,
              "operations": "read accepted records; decode two original images; export a local reading view"}
    try:
        import pyarrow.parquet as pq
        from PIL import Image

        require(load(PILOT / "report.json")["status"] == "PASS_DIAGNOSTIC", "Pilot not accepted")
        selected = load(PILOT / "ac-cases.json")["largest_NDCG10_drop"]
        require(selected["paired_row"]["query_id"] == QUERY_ID, "Unexpected example selection")
        historical, rankings, per_query = {}, {}, {}
        for group, directory in SOURCES.items():
            old = load(directory / "report.json")
            require(old["status"] == "PASS" and old["completed_queries"] == 232,
                    "Incomplete historical record")
            for name in ("rankings.jsonl", "per-query.tsv", "qrels.json", "page-id-mapping.json"):
                require(digest(directory / name) == old["artifact_sha256"][name],
                        f"Historical artifact changed: {group}/{name}")
            matches = [json.loads(line) for line in (directory / "rankings.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
            matches = [row for row in matches if str(row["query_id"]) == QUERY_ID]
            require(len(matches) == 1, "Query ranking missing or duplicated")
            rankings[group] = matches[0]["hits"]
            with (directory / "per-query.tsv").open(encoding="utf-8", newline="") as source:
                matches = [row for row in csv.DictReader(source, delimiter="\t")
                           if str(row["query_id"]) == QUERY_ID]
            require(len(matches) == 1, "Query metric row missing or duplicated")
            per_query[group] = matches[0]
            historical[group] = {"directory": str(directory), "report_sha256": digest(directory / "report.json")}
        require(load(SOURCES["A"] / "qrels.json") == load(SOURCES["C"] / "qrels.json"), "A/C labels differ")
        mapping = load(SOURCES["A"] / "page-id-mapping.json")
        require(mapping == load(SOURCES["C"] / "page-id-mapping.json"), "A/C ID maps differ")
        by_page = {row["corpus_id"]: row for row in mapping}
        qrels = load(SOURCES["A"] / "qrels.json")[QUERY_ID]
        positive = {pid for pid, grade in qrels.items() if grade > 0}
        require(positive == {by_page["60"]["point_id"]}, "Expected one positive page for this example")
        profile_path = ROOT / "cache/encoded-economics-v1/profile.json"
        profile = load(profile_path)
        data = ROOT / "data" / ("economics_reports_v2-" + profile["dataset_revision"])
        manifest_path = data / "download-manifest.json"
        require(digest(manifest_path) == profile["dataset_download_manifest_sha256"], "Dataset identity changed")
        manifest = load(manifest_path)
        for name in ("corpus", "queries"):
            relative = name + "/test-00000-of-00001.parquet"
            expected = next(row["sha256"] for row in manifest["files"] if row["path"] == relative)
            require(digest(data / relative) == expected, f"Original {name} file changed")
        query_rows = pq.read_table(data / "queries/test-00000-of-00001.parquet",
                                   columns=["query-id", "query", "language"],
                                   filters=[("query-id", "=", int(QUERY_ID))]).to_pylist()
        require(len(query_rows) == 1, "Official query missing")
        query = query_rows[0]
        require(query["query"] == selected["paired_row"]["query"], "Query text differs")
        pages = pq.read_table(data / "corpus/test-00000-of-00001.parquet",
                             columns=["corpus-id", "doc-id", "image"],
                             filters=[("corpus-id", "in", [int(pid) for pid in PAGE_IDS])]).to_pylist()
        pages = {str(page["corpus-id"]): page for page in pages}
        require(set(pages) == set(PAGE_IDS), "Original page missing")
        exported, panels = [], []
        for pid in PAGE_IDS:
            page = pages[pid]
            require(page["doc-id"] == by_page[pid]["doc_id"], "Page document mapping differs")
            image_bytes = page["image"]["bytes"]
            output_path = record / ("page-" + pid + ".png")
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
                size = list(image.size)
                with output_path.open("xb") as output:
                    image.convert("RGB").save(output, format="PNG")
            grade = qrels.get(by_page[pid]["point_id"])
            label = ("标注相关（score=" + str(grade) + "）" if grade is not None and grade > 0
                     else "qrels 中无正相关标注；不等于人工确认绝对无关")
            ranks = {g: next((hit["rank"] for hit in rankings[g] if str(hit["corpus_id"]) == pid), None)
                     for g in ("A", "C")}
            exported.append({"corpus_id": pid, "doc_id": page["doc-id"], "qrels_score": grade,
                             "ranks_top100": ranks, "image_size": size,
                             "original_image_bytes_sha256": hashlib.sha256(image_bytes).hexdigest(),
                             "png_sha256": digest(output_path), "png": output_path.name})
            panels.append("<section><h2>corpus-id " + pid + "</h2><p>" + escape(page["doc-id"]) +
                          "</p><p>" + escape(label) + "</p><p>A 排名：" + str(ranks["A"]) +
                          "；C 排名：" + str(ranks["C"]) + "</p><a href='" + output_path.name +
                          "'><img src='" + output_path.name + "' alt='original page " + pid +
                          "'></a><p>点击图片查看原始尺寸。corpus-id 不是 PDF 印刷页码。</p></section>")
        rank_tables = []
        for group in ("A", "C"):
            body = "".join("<tr><td>" + str(hit["rank"]) + "</td><td>" + str(hit["corpus_id"]) +
                           "</td><td>" + escape(hit["doc_id"]) + "</td><td>" +
                           ("是" if hit["point_id"] in positive else "未列为正相关") + "</td></tr>"
                           for hit in rankings[group][:10])
            rank_tables.append("<section><h2>" + group + " 原有 top10</h2><table><tr><th>排名</th>"
                               "<th>corpus-id</th><th>报告</th><th>正相关标注</th></tr>" + body + "</table></section>")
        rank_c = exported[0]["ranks_top100"]["C"]
        require(exported[0]["ranks_top100"]["A"] == 1 and rank_c == 7, "Unexpected example ranks")
        require(abs(float(per_query["C"]["NDCG@10"]) - 1 / math.log2(rank_c + 1)) <= 1e-12,
                "Example NDCG explanation disagrees with saved result")
        html = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>A/C query 169 页面对照</title><style>
body{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:24px;color:#172033;line-height:1.7}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}section{min-width:0}
img{max-width:100%;height:auto;border:1px solid #ccd3df}table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ccd3df;padding:6px;text-align:left}.note{background:#eef3f8;padding:16px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>
<h1>A/C 第 169 题：先区分排名退化与召回损失</h1>"""
        html += "<p>原问题（" + escape(query["language"]) + "）：" + escape(query["query"]) + "</p>"
        html += "<p>辅助中文翻译：在商业实践中实现数字化采用有哪些挑战？</p>"
        html += """<div class="note">该题唯一标注相关页是 corpus-id 60：A 第 1 名，C 第 7 名。
因此 Recall@10 两组都为 1；NDCG@10 从 1 降至 1/log2(8)=1/3；Recall@5 从 1 降至 0。
这是前排排序退化，不能称作 top10 或 top100 召回失败。</div>
<p>本页面使用已保存排名，不调用模型、池化器或数据库。分数绝对值不跨 A/C 直接比较。</p>
<div class="grid">""" + "".join(panels) + "</div><div class='grid'>" + "".join(rank_tables) + "</div>"
        html += """<h2>本步只回答三个问题</h2><ol>
<li>在页面 60 找到可能回应“数字化采用挑战”的文字或图表；找不到就如实记录不确定。</li>
<li>页面 304 中有哪些与问题主题相近的内容？记录可见文字，不把 qrels 中无正标注视为绝对无关。</li>
<li>关键内容在页面上集中于某一段、某些行，还是分散在多处？这只是后续诊断线索。</li>
</ol><p>当前证据没有证明行池化损失了哪个语义，也没有证明聚类池化能恢复正确排名。
这道题按最大 NDCG@10 降幅选作教学案例，不能用来估计全体效果或挑选参数。</p></html>"""
        with (record / "case.html").open("x", encoding="utf-8") as output:
            output.write(html)
        report.update({"status": "PASS_EXPORT", "query": query, "historical_records": historical,
                       "pilot_cases_sha256": digest(PILOT / "ac-cases.json"),
                       "profile_sha256": digest(profile_path), "script_sha256": digest(Path(__file__)),
                       "dataset_manifest_sha256": digest(manifest_path), "pages": exported,
                       "html_sha256": digest(record / "case.html"),
                       "next_action": "human page reading; no automatic experiment continuation"})
        print("Query:", QUERY_ID, query["query"], flush=True)
        for page in exported:
            print("Page:", page["corpus_id"], "ranks:", page["ranks_top100"],
                  "qrels score:", page["qrels_score"], flush=True)
        print("Open in browser:", record / "case.html", flush=True)
        print("Case export: PASS; no new retrieval, encoding or pooling", flush=True)
    except Exception as error:
        report.update({"status": "FAIL", "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        save(record / "report.json", report)


if __name__ == "__main__":
    main()
