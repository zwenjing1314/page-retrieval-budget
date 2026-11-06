"""
Retrieval module - Search and retrieval strategies.

Components:
- TwoStageRetriever: Pooled prefetch → MaxSim reranking (our novel contribution)
- SingleStageRetriever: Direct multi-vector or pooled search
"""

from visual_rag.retrieval.multi_vector import MultiVectorRetriever
from visual_rag.retrieval.single_stage import SingleStageRetriever
from visual_rag.retrieval.three_stage import ThreeStageRetriever
from visual_rag.retrieval.two_stage import TwoStageRetriever

__all__ = [
    "TwoStageRetriever",
    "SingleStageRetriever",
    "MultiVectorRetriever",
    "ThreeStageRetriever",
]
