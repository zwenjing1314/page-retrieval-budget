from __future__ import annotations

from typing import Dict, List


def _dcg(relevances: List[float]) -> float:
    import math

    score = 0.0
    for i, rel in enumerate(relevances):
        if rel <= 0:
            continue
        score += (2.0**rel - 1.0) / math.log2(i + 2)
    return score


def ndcg_at_k(ranking: List[str], qrels: Dict[str, int], k: int) -> float:
    rels = [float(qrels.get(doc_id, 0)) for doc_id in ranking[:k]]
    dcg = _dcg(rels)
    ideal_rels = sorted((float(v) for v in qrels.values()), reverse=True)[:k]
    idcg = _dcg(ideal_rels)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def mrr_at_k(ranking: List[str], qrels: Dict[str, int], k: int) -> float:
    for i, doc_id in enumerate(ranking[:k]):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranking: List[str], qrels: Dict[str, int], k: int) -> float:
    relevant = {doc_id for doc_id, rel in qrels.items() if rel > 0}
    if not relevant:
        return 0.0
    retrieved = set(ranking[:k])
    return len(retrieved & relevant) / len(relevant)
