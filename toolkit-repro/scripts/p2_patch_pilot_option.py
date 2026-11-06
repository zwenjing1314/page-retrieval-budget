import ast
import uuid
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "scripts/p1_real_cache.py"
raw = path.read_bytes()
text = raw.decode("utf-8-sig").replace("\r\n", "\n")
assert 'parser.add_argument("--pages"' not in text, "Already patched"

old = "args = parser.parse_args()\n"
new = (
    'parser.add_argument("--pages", type=int, default=3)\n'
    "args = parser.parse_args()\n"
    'assert 3 <= args.pages <= 8, "Diagnostic page count must be 3..8"\n'
)
assert text.count(old) == 1
text = text.replace(old, new, 1)

old = 'sorted(int(k) for k in page_map if k != "63")[:2]'
new = 'sorted(int(k) for k in page_map if k != "63")[:args.pages - 1]'
assert text.count(old) == 1
text = text.replace(old, new, 1)
ast.parse(text)

record = root / "records" / (
    "p2-pilot-option-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    + "-" + uuid.uuid4().hex[:6]
)
record.mkdir(parents=True, exist_ok=False)
(record / "p1_real_cache.before.py").write_bytes(raw)
path.write_bytes(text.encode("utf-8"))
print("Backup:", record)
print("Diagnostic page-count option: APPLIED")