import hashlib
import json
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parents[1]
records = root / "records"

checks = {
    "zero_id": "p1-zero-id-*/result.json",
    "single_page": "p1-single-page-*/report.json",
    "scoring_recovery": "p1-scoring-*/recovery-*/report.json",
    "full_loader": "p1-loader-acceptance-*/report.json",
    "failure_guards": "p1-failure-guards-*/report.json",
    "model_entry": "p1-model-entry-*/report.json",
    "abc_fields": "p1-abc-fields-*/report.json",
    "real_retrieval": "p1-real-cache-retrieve-*/report.json",
    "cache_resume": "p1-real-cache-resume-check-*/report.json",
}
evidence = {}
for name, pattern in checks.items():
    matches = list(records.glob(pattern))
    assert matches, f"Missing evidence: {name}"
    path = max(matches, key=lambda p: p.stat().st_mtime)
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["status"] == "PASS", f"Latest evidence failed: {path}"
    evidence[name] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    print(name, "PASS")

output = records / (
    "p1-closeout-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    + "-" + uuid.uuid4().hex[:6]
)
output.mkdir(parents=True, exist_ok=False)

commit = subprocess.run(
    ["git", "-C", str(root.parent), "rev-parse", "HEAD"],
    check=True, capture_output=True, text=True,
).stdout.strip()
diff = subprocess.run(
    ["git", "-C", str(root.parent), "diff", "HEAD", "--",
     "toolkit-repro/scripts", "toolkit-repro/visual-rag-toolkit"],
    check=True, capture_output=True,
).stdout
(output / "working-changes.patch").write_bytes(diff)
(output / "README.before-P1-closeout.md").write_bytes(
    (root.parent / "README.md").read_bytes()
)

report = {
    "status": "PASS",
    "stage": "P1",
    "scope": "Correctness and engineering diagnostics only",
    "outer_repository_commit": commit,
    "uncommitted_changes": "working-changes.patch",
    "evidence": evidence,
    "not_completed": [
        "Full 452-page / 232-query encoding cache",
        "Formal A/B/C evaluation and full-index ID equality",
        "Formal timing, RAM and physical storage acceptance",
    ],
}
(output / "report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)

lines = [
    "# P1 验收记录",
    "",
    "结论：P1 正确性与工程诊断通过，可以进入 P2。",
    "本结论不代表正式全库评价或最终科研验收通过。",
    "",
]
for name, item in evidence.items():
    lines.append(f"- {name}：[原始记录]({Path(item['path']).as_posix()})")
lines += [
    "",
    "历史失败记录保留。评分诊断使用原失败记录对应的 recovery 报告。",
    "正式 A/B/C 指标、全库索引一致性及实际存储验收仍待完成。",
]
(output / "P1-checklist.md").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)
print("P1 closeout:", output)
print("P1 evidence closeout: PASS")