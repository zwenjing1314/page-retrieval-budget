from pathlib import Path
import io

import pyarrow.parquet as pq
from PIL import Image

root = Path(
    r"C:\Users\WenJing\Documents\WorkTransfer\page-retrieval-budget"
    r"\toolkit-repro\data"
    r"\economics_reports_v2-76fe40166ba07b1bf50457f5c6057cacdd045f10"
)

def read_table(name):
    return pq.read_table(root / name / "test-00000-of-00001.parquet")

# 查看四张表的行数和字段
for name in ("docs", "corpus", "queries", "qrels"):
    table = read_table(name)
    print(name, "rows:", table.num_rows, "columns:", table.column_names)

# docs：第一份报告的标识和摘要
docs = read_table("docs")
print("\nFirst document:")
print(docs.slice(0, 1).to_pylist()[0])

# queries：第一个问题，只展示检索需要的字段
queries = read_table("queries")
query = queries.slice(0, 1).to_pylist()[0]
print("\nFirst query:")
print({key: query[key] for key in ("query-id", "query", "language")})

# corpus：将第一张页面的图片字节解码为 PIL 图像
corpus = read_table("corpus")
page = corpus.slice(0, 1).to_pylist()[0]
with Image.open(io.BytesIO(page["image"]["bytes"])) as image:
    image.load()
    print("\nFirst page:")
    print("corpus-id:", page["corpus-id"], "doc-id:", page["doc-id"])
    print("image size:", image.size, "mode:", image.mode)

# 查看第一个问题对应的部分相关性标注
qrels = read_table("qrels").to_pylist()
matches = [row for row in qrels if row["query-id"] == query["query-id"]]
print("\nJudgments for this query:", len(matches))
for row in matches[:5]:
    print({key: row[key] for key in ("query-id", "corpus-id", "score")})