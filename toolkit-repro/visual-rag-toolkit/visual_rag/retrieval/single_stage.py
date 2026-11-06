"""
Single-Stage Retrieval for Visual Document Search.

Provides direct search without the two-stage complexity.
Use when:
- Collection is small (<10K documents)
- Latency is not critical
- Maximum accuracy is required
"""

import logging
from typing import Any, Dict, List, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SingleStageRetriever:
    """
    Single-stage visual document retrieval using native Qdrant search.

    Supports strategies:
    - multi_vector: Native MaxSim on full embeddings (using="initial")
    - tiles_maxsim: Native MaxSim between query tokens and tile vectors (using="mean_pooling")
    - pooled_tile: Pooled query vs tile vectors (using="mean_pooling")
    - pooled_global: Pooled query vs global pooled doc vector (using="global_pooling")
    - experimental_maxsim: Native MaxSim between query tokens and experimental vectors (using="experimental_pooling[_k]")
    - pooled_experimental: Pooled query vs experimental vectors (using="experimental_pooling[_k]")

    Args:
        qdrant_client: Connected Qdrant client
        collection_name: Name of the Qdrant collection
        request_timeout: Timeout for Qdrant requests (seconds)
        max_retries: Number of retry attempts on failure
        retry_sleep: Sleep time between retries (seconds)

    Example:
        >>> retriever = SingleStageRetriever(client, "my_collection")
        >>> results = retriever.search(query, top_k=10)
    """

    def __init__(
        self,
        qdrant_client,
        collection_name: str,
        experimental_vector_name: str = "experimental_pooling",
        request_timeout: int = 120,
        max_retries: int = 3,
        retry_sleep: float = 1.0,
    ):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.experimental_vector_name = str(experimental_vector_name)
        self.request_timeout = int(request_timeout)
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep

    def search(
        self,
        query_embedding: Union[torch.Tensor, np.ndarray],
        top_k: int = 10,
        strategy: str = "multi_vector",
        filter_obj=None,
    ) -> List[Dict[str, Any]]:
        """
        Single-stage search with configurable strategy.

        Args:
            query_embedding: Query embeddings [num_tokens, dim]
            top_k: Number of results
            strategy: "multi_vector", "tiles_maxsim", "pooled_tile", or "pooled_global"
            filter_obj: Qdrant filter

        Returns:
            List of results with scores and metadata
        """
        query_np = self._to_numpy(query_embedding)

        if strategy == "multi_vector":
            # Native multi-vector MaxSim
            vector_name = "initial"
            query_vector = query_np.tolist()
            logger.debug(f"🎯 Multi-vector search on '{vector_name}'")

        elif strategy == "tiles_maxsim":
            # Native multi-vector MaxSim against tile vectors
            vector_name = "mean_pooling"
            query_vector = query_np.tolist()
            logger.debug(f"🎯 Tile MaxSim search on '{vector_name}'")

        elif strategy == "pooled_tile":
            # Tile-level pooled
            vector_name = "mean_pooling"
            query_pooled = query_np.mean(axis=0)
            query_vector = query_pooled.tolist()
            logger.debug(f"🔍 Tile-pooled search on '{vector_name}'")

        elif strategy == "pooled_global":
            # Global pooled vector (single vector)
            vector_name = "global_pooling"
            query_pooled = query_np.mean(axis=0)
            query_vector = query_pooled.tolist()
            logger.debug(f"🔍 Global-pooled search on '{vector_name}'")

        elif strategy == "experimental_maxsim":
            # Native multi-vector MaxSim against experimental pooled vectors
            vector_name = self.experimental_vector_name
            query_vector = query_np.tolist()
            logger.debug(f"🎯 Experimental MaxSim search on '{vector_name}'")

        elif strategy == "pooled_experimental":
            # Pooled query vs experimental pooled vectors
            vector_name = self.experimental_vector_name
            query_pooled = query_np.mean(axis=0)
            query_vector = query_pooled.tolist()
            logger.debug(f"🔍 Experimental pooled search on '{vector_name}'")

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=vector_name,
            query_filter=filter_obj,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            timeout=self.request_timeout,
        ).points

        return [
            {
                "id": r.id,
                "score": r.score,
                "score_final": r.score,
                "payload": r.payload,
            }
            for r in results
        ]

    def _to_numpy(self, embedding: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert embedding to numpy array."""
        if isinstance(embedding, torch.Tensor):
            if embedding.dtype == torch.bfloat16:
                return embedding.cpu().float().numpy()
            return embedding.cpu().float().numpy()  # .float() for BFloat16 compatibility
        return np.array(embedding, dtype=np.float32)
