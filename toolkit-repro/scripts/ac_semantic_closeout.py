"""Write a reproducible closeout report for the fixed H semantic-pooling study."""

import json
import uuid
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "records"
A = RECORDS / "p2-formal-a-20261107-012250-612a76"
B = RECORDS / "p2-formal-b-20261107-013537-34e55b"
C = RECORDS / "p2-formal-c-20261107-014210-68a2ce"
H = RECORDS / "ac-semantic-eval-20261109-172244-5e9662"
COMPARE = RECORDS / "ac-semantic-compare-20261109-173153-a862b5"
CASES = RECORDS / "ac-semantic-cases-20261109-174521-e3a981"
LANGUAGE = RECORDS / "ac-semantic-language-20261109-215411-e164f7"
PAIRED_LANGUAGE = RECORDS / "ac-semantic-language-paired-20261109-235341-0d8470"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def latest_batch():
    paths = sorted(RECORDS.glob("ac-semantic-case-batch-*/cases.json"))
    return paths[-1].parent if paths else None


def main():
    h = load(H / "report.json")
    comparison = load(COMPARE / "report.json")
    cases = load(CASES / "report.json")
    language = load(LANGUAGE / "report.json")
    paired_language = load(PAIRED_LANGUAGE / "report.json")
    if paired_language["status"] != "PASS":
        raise RuntimeError("Paired language analysis is not accepted")
    if paired_language["design"]["underlying_intents"] != 58 or not paired_language["design"]["qrels_identical_for_all_58_aligned_intents"]:
        raise RuntimeError("Paired language design is not the expected 58-intent aligned design")
    batch = latest_batch()

    record = RECORDS / (
        "ac-semantic-closeout-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        + "-" + uuid.uuid4().hex[:6]
    )
    record.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "PASS",
        "scope": "Economics 452 pages / 232 queries; fixed ColPali-v1.3 and cached inputs",
        "representation": "Existing HierarchicalTokenPooler, pool_factor=30, 34x128 FP16 per page",
        "model_or_data_changes": "none",
        "new_parameter_search": False,
        "formal_h_metrics": h["aggregate_metrics"],
        "frozen_comparison_metrics": comparison["aggregate_metrics"],
        "h_minus_a": {name: comparison["metric_comparisons"]["A"][name]["mean_delta"]
                      for name in comparison["metric_comparisons"]["A"]},
        "h_minus_b": {name: comparison["metric_comparisons"]["B"][name]["mean_delta"]
                      for name in comparison["metric_comparisons"]["B"]},
        "h_minus_c": {name: comparison["metric_comparisons"]["C"][name]["mean_delta"]
                      for name in comparison["metric_comparisons"]["C"]},
        "ranking_comparisons": comparison["ranking_comparisons"],
        "case_shortlist_record": str(CASES),
        "case_batch_record": str(batch) if batch else None,
        "language_record": str(LANGUAGE),
        "language_counts": language["language_counts"],
        "language_interpretation": language["interpretation_limit"],
        "paired_language_record": str(PAIRED_LANGUAGE),
        "paired_language_design": paired_language["design"],
        "paired_language_results": paired_language["per_language"],
        "paired_language_interactions_vs_english": paired_language["paired_language_interactions_vs_english"],
        "evidence_records": {
            "A": str(A), "B": str(B), "C": str(C), "H": str(H),
            "comparison": str(COMPARE),
        },
        "conclusion": (
            "At a fixed approximately-34-vector page budget, semantic hierarchical pooling "
            "improves over simple pooled-only C on the Economics subset, but remains below "
            "full-vector A and author pooled-prefetch B. The effect is query-dependent; "
            "the four language blocks are 58 aligned intents rather than 232 independent "
            "language samples, and paired language interaction intervals include zero."
        ),
        "next_boundary": (
            "Do not claim cross-dataset ViDoRe-v2 generalization. Any extension requires a "
            "separate dataset, cache, collection, and resource record."
        ),
    }
    (record / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = [
        "# Economics 语义聚类池化对照阶段性收口",
        "",
        "## 固定范围",
        "",
        "- 数据：Economics，452 页、232 个问题、3628 条 qrels。",
        "- 模型和输入：固定 ColPali-v1.3、已有页面/查询缓存，不重新编码。",
        "- H 表示：已有 HierarchicalTokenPooler，`pool_factor=30`，每页 `34x128 float16`。",
        "- 没有训练、参数搜索或跨数据集扩展。",
        "",
        "## 全库指标",
        "",
        "| 组 | NDCG@5 | NDCG@10 | Recall@5 | Recall@10 | Recall@100 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group, values in comparison["aggregate_metrics"].items():
        markdown.append("| " + group + " | " + " | ".join(f"{values[name]:.6f}" for name in ("NDCG@5", "NDCG@10", "Recall@5", "Recall@10", "Recall@100")) + " |")
    markdown.extend([
        "",
        "## 结论",
        "",
        "H 相对 C 有整体提升，但仍低于完整向量 A 和作者池化预筛 B。H 的 Top-10 变化具有题目依赖性；部分题目恢复了 C 丢失的正页面，部分题目比 C 更差。四种语言各 58 条，但它们是 58 个底层意图的四种语言版本；因此以意图为单位进行配对分析。相对 English 的 H-A NDCG@10 配对交互区间均包含 0，当前证据不支持语言造成差异。",
        "",
        "## 证据记录",
        "",
        f"- H 完整评价：`{H}`",
        f"- A/B/C/H 逐题对照：`{COMPARE}`",
        f"- 四类案例筛选：`{CASES}`",
        f"- 语言分组：`{LANGUAGE}`",
        f"- 语言配对意图分析：`{PAIRED_LANGUAGE}`",
        f"- 批量案例页面：`{batch}`" if batch else "- 批量案例页面：未找到",
        "",
        "## 范围边界",
        "",
        "当前结果只支持 Economics 子集上的固定输入对照，不支持直接宣称对整个 ViDoRe v2 泛化。跨数据集验证应另建数据缓存、Qdrant collection 和资源记录。",
    ])
    (record / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print("H metrics:", h["aggregate_metrics"], flush=True)
    print("Language groups:", language["language_counts"], flush=True)
    print("Record:", record, flush=True)
    print("Semantic pooling closeout: PASS", flush=True)


if __name__ == "__main__":
    main()
