import ast
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parents[1]
directory = root / "visual-rag-toolkit/visual_rag/retrieval"
single = directory / "single_stage.py"
two = directory / "two_stage.py"
originals = {p: p.read_bytes() for p in (single, two)}
texts = {
    p: raw.decode("utf-8-sig").replace("\r\n", "\n")
    for p, raw in originals.items()
}

text = texts[single]
assert "SearchParams(exact=True)" not in text, "Patch may already exist"
assert text.count("import numpy as np\n") == 1
text = text.replace(
    "import numpy as np\n",
    "import numpy as np\nfrom qdrant_client import models as qdrant_models\n",
    1,
)
old = "            query_filter=filter_obj,\n            limit=top_k,"
new = (
    "            query_filter=filter_obj,\n"
    "            search_params=qdrant_models.SearchParams(exact=True),\n"
    "            limit=top_k,"
)
assert text.count(old) == 1
texts[single] = text.replace(old, new, 1)

text = texts[two]
old = "                        limit=prefetch_k,\n                    )"
new = (
    "                        limit=prefetch_k,\n"
    "                        params=qdrant_models.SearchParams(exact=True),\n"
    "                    )"
)
assert text.count(old) == 1, "Source differs or patch already exists"
texts[two] = text.replace(old, new, 1)

for p, text in texts.items():
    ast.parse(text, filename=str(p))

record = root / "records" / (
    "p1-exact-search-patch-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    + "-" + uuid.uuid4().hex[:6]
)
record.mkdir(parents=True, exist_ok=False)
for p, raw in originals.items():
    (record / (p.stem + ".before.py")).write_bytes(raw)
for p, text in texts.items():
    p.write_bytes(text.encode("utf-8"))

result = subprocess.run(
    [
        "git", "-C", str(root.parent), "diff", "--",
        "toolkit-repro/visual-rag-toolkit/visual_rag/retrieval/single_stage.py",
        "toolkit-repro/visual-rag-toolkit/visual_rag/retrieval/two_stage.py",
    ],
    check=True, capture_output=True,
)
(record / "changes.patch").write_bytes(result.stdout)
print("Patch record:", record)
print("Exact-search patch applied; runtime verification pending")