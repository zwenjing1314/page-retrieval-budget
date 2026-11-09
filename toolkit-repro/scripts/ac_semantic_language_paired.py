"""Paired, intent-cluster bootstrap for the four translated query blocks.

The 232 Economics queries contain 58 aligned intents in four languages. This
script verifies alignment by identical qrels and resamples intent clusters,
not individual language-query rows. Results remain descriptive for this set.
"""

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "records"
DATA = ROOT / "data/economics_reports_v2-76fe40166ba07b1bf50457f5c6057cacdd045f10"
A = RECORDS / "p2-formal-a-20261107-012250-612a76"
C = RECORDS / "p2-formal-c-20261107-014210-68a2ce"
H = RECORDS / "ac-semantic-eval-20261109-172244-5e9662"
LANGUAGES = ("english", "french", "spanish", "german")
METRICS = ("NDCG@10", "Recall@10", "Recall@100")
BOOTSTRAPS = 20000
SEED = 20261109


def read_metrics(path):
    with path.open(encoding="utf-8", newline="") as source:
        return {row["query_id"]: row for row in csv.DictReader(source, delimiter="\t")}


def percentile_interval(samples):
    low, high = np.percentile(samples, [2.5, 97.5])
    return [float(low), float(high)]


def main():
    query_rows = pq.read_table(
        DATA / "queries/test-00000-of-00001.parquet",
        columns=["query-id", "language"],
    ).to_pylist()
    grouped = {language: [] for language in LANGUAGES}
    for row in query_rows:
        language = str(row["language"]).lower()
        if language not in grouped:
            raise RuntimeError(f"Unexpected language: {language}")
        grouped[language].append(str(row["query-id"]))
    for language in LANGUAGES:
        grouped[language].sort(key=int)
        if len(grouped[language]) != 58:
            raise RuntimeError(f"Expected 58 {language} queries, got {len(grouped[language])}")

    qrels_a = json.loads((A / "qrels.json").read_text(encoding="utf-8"))
    qrels_c = json.loads((C / "qrels.json").read_text(encoding="utf-8"))
    if qrels_a != qrels_c:
        raise RuntimeError("A and C qrels differ")
    aligned_qids = list(zip(*(grouped[language] for language in LANGUAGES)))
    for intent_index, qids in enumerate(aligned_qids):
        sets = [qrels_a[qid] for qid in qids]
        if any(item != sets[0] for item in sets[1:]):
            raise RuntimeError(f"qrels mismatch for aligned intent index {intent_index}: {qids}")

    rows = {
        "A": read_metrics(A / "per-query.tsv"),
        "C": read_metrics(C / "per-query.tsv"),
        "H": read_metrics(H / "per-query.tsv"),
    }
    expected = {qid for qids in grouped.values() for qid in qids}
    if any(set(group_rows) != expected for group_rows in rows.values()):
        raise RuntimeError("Evaluation query IDs do not align with official queries")

    # Arrays are ordered [intent, language, metric]. Each bootstrap draw samples
    # the same intent indices for every language, preserving the paired design.
    deltas = {base: np.empty((58, 4, len(METRICS)), dtype=np.float64) for base in ("A", "C")}
    for intent_index, qids in enumerate(aligned_qids):
        for language_index, qid in enumerate(qids):
            for metric_index, metric in enumerate(METRICS):
                h = float(rows["H"][qid][metric])
                for base in ("A", "C"):
                    deltas[base][intent_index, language_index, metric_index] = h - float(rows[base][qid][metric])

    rng = np.random.default_rng(SEED)
    sampled = rng.integers(0, 58, size=(BOOTSTRAPS, 58))
    bootstrap_means = {
        base: deltas[base][sampled].mean(axis=1)
        for base in ("A", "C")
    }

    result = {
        "status": "PASS",
        "design": {
            "underlying_intents": 58,
            "language_versions_per_intent": 4,
            "query_rows": 232,
            "qrels_identical_for_all_58_aligned_intents": True,
            "bootstrap_resampling_unit": "aligned underlying intent; all four language versions sampled together",
            "bootstrap_replicates": BOOTSTRAPS,
            "seed": SEED,
        },
        "language_order": list(LANGUAGES),
        "per_language": {},
        "paired_language_interactions_vs_english": {},
        "interpretation_limit": (
            "Percentile intervals summarize uncertainty over these 58 aligned intents. "
            "They do not establish causal language effects or generalization to other datasets."
        ),
    }

    for base in ("A", "C"):
        for language_index, language in enumerate(LANGUAGES):
            result["per_language"].setdefault(language, {})
            for metric_index, metric in enumerate(METRICS):
                samples = bootstrap_means[base][:, language_index, metric_index]
                result["per_language"][language][f"H_minus_{base}_{metric}"] = {
                    "mean": float(deltas[base][:, language_index, metric_index].mean()),
                    "intent_cluster_bootstrap_95pct_CI": percentile_interval(samples),
                }

        for language_index, language in enumerate(LANGUAGES[1:], start=1):
            result["paired_language_interactions_vs_english"].setdefault(language, {})
            for metric_index, metric in enumerate(METRICS):
                paired = deltas[base][:, language_index, metric_index] - deltas[base][:, 0, metric_index]
                bootstrap_paired = (
                    bootstrap_means[base][:, language_index, metric_index]
                    - bootstrap_means[base][:, 0, metric_index]
                )
                result["paired_language_interactions_vs_english"][language][f"H_minus_{base}_{metric}"] = {
                    "mean_paired_difference": float(paired.mean()),
                    "intent_cluster_bootstrap_95pct_CI": percentile_interval(bootstrap_paired),
                }

    record = RECORDS / (
        "ac-semantic-language-paired-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        + "-" + uuid.uuid4().hex[:6]
    )
    record.mkdir(parents=True, exist_ok=False)
    (record / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    for language in LANGUAGES:
        ah = result["per_language"][language]["H_minus_A_NDCG@10"]
        ch = result["per_language"][language]["H_minus_C_NDCG@10"]
        print(f"{language}: H-A NDCG@10={ah['mean']:.6f} CI={ah['intent_cluster_bootstrap_95pct_CI']}; "
              f"H-C={ch['mean']:.6f} CI={ch['intent_cluster_bootstrap_95pct_CI']}", flush=True)
    print("Aligned intent clusters:", len(aligned_qids), "with 4 language versions each", flush=True)
    print("Language interaction contrasts vs English:", flush=True)
    for language, effects in result["paired_language_interactions_vs_english"].items():
        item = effects["H_minus_A_NDCG@10"]
        print(f"{language} vs English, H-A NDCG@10 paired difference="
              f"{item['mean_paired_difference']:.6f} CI={item['intent_cluster_bootstrap_95pct_CI']}", flush=True)
    print("Record:", record, flush=True)
    print("Paired intent-cluster analysis: PASS; no model/Qdrant calls", flush=True)


if __name__ == "__main__":
    main()
