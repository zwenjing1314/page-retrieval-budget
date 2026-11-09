# Economics 语义池化对照交接材料

交接日期：2026-11-10  
工作目录：`C:/Users/WenJing/Documents/WorkTransfer/page-retrieval-budget/toolkit-repro`

## 给主研究窗口的结论

在固定 ColPali-v1.3、Economics 452 页/232 查询、已有缓存和约 34 个页面向量的预算下：

1. 论文复现主线 A/B/C 已通过；A/B 是论文方法流程，C 是存储压缩工程扩展。
2. 已有 `HierarchicalTokenPooler(pool_factor=30)` 的 H 对照已完成缓存、建库、读回和 232 题评价。
3. H 指标为 NDCG@10 `0.450247`、Recall@10 `0.368596`、Recall@100 `0.870817`。
4. H 相对 C 分别提升 NDCG@10 `0.062291`、Recall@10 `0.058118`、Recall@100 `0.039241`；相对 A 仍下降 NDCG@10 `0.049985`、Recall@10 `0.033683`、Recall@100 `0.029442`。
5. H 是题目依赖的工程对照，不是论文新方法，也不能称为 ViDoRe v2 全集结论。

## 已冻结的关键对象

```text
Conda 环境：page-retrieval-budget
Qdrant：127.0.0.1:16333，Server 1.14.1
模型：models/colpali-v1.3-b5c6dd62125326e6f0b540c1f7b36e901cdc0a11
数据：data/economics_reports_v2-76fe40166ba07b1bf50457f5c6057cacdd045f10
共享缓存：cache/encoded-economics-v1
H collection：prb_ac_h_full_176185423f13
```

## 推荐先读的记录

1. `records/ac-semantic-closeout-20261109-235958-dc5b5a/report.md`
2. `records/ac-semantic-eval-20261109-172244-5e9662/report.json`
3. `records/ac-semantic-compare-20261109-173153-a862b5/report.json`
4. `records/ac-semantic-language-paired-20261109-235341-0d8470/report.json`
5. `records/ac-semantic-case-batch-20261109-212136-13f914/cases.html`
6. `docs/research/Economics语义池化对照阶段报告.md`

## 可复核脚本

```text
scripts/ac_semantic_full_cache.py
scripts/ac_semantic_build_collection.py
scripts/ac_semantic_evaluate.py
scripts/ac_semantic_compare.py
scripts/ac_semantic_case_shortlist.py
scripts/ac_semantic_case_batch.py
scripts/ac_semantic_language_paired.py
scripts/ac_semantic_closeout.py
```

这些脚本中的记录路径固定指向本次冻结实验；若要复用到新数据集，应复制为新的、独立命名的脚本或参数化脚本，并新建缓存和 collection。

## 重要解释限制

- 四种语言各 58 条，但它们是 58 个底层意图的翻译版本；不能把 232 条当作 232 个独立语言样本。
- 语言配对 bootstrap 的有效重采样单位是 58 个意图；相对 English 的语言交互区间包含 0。
- H 的绝对分数不能与 A/C 的绝对分数跨表示直接解释；应使用固定 qrels 下的排名指标和逐题差值。
- “qrels 中无正标注”不等于人工确认页面绝对无关。
- H 的理论向量字节数不等于 Qdrant 实际磁盘占用；存储审计包含历史 collection 和运行时文件。

## 当前建议

将本阶段作为 Economics 子集的完整分析和工程对照收口，先交回主窗口复核。暂不扩展其他 ViDoRe v2 数据集、暂不训练、暂不搜索参数、暂不修改旧 layoutpatch 或已有 A/B/C collection。任何跨数据集验证应作为新阶段单独立项并重新记录所有输入、资源和版本。
