import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ThreeStageRetriever:
    def __init__(
        self,
        qdrant_client,
        collection_name: str,
        *,
        full_vector_name: str = "initial",
        experimental_vector_name: str = "experimental_pooling",
        global_vector_name: str = "global_pooling",
        request_timeout: int = 120,
        max_retries: int = 3,
        retry_sleep: float = 0.5,
    ):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.full_vector_name = full_vector_name
        self.experimental_vector_name = experimental_vector_name
        self.global_vector_name = global_vector_name
        self.request_timeout = int(request_timeout)
        self.max_retries = int(max_retries)
        self.retry_sleep = float(retry_sleep)

        self._global_is_multivector: Optional[bool] = None
        self._experimental_is_multivector: Optional[bool] = None

    def _retry_call(self, fn):
        import time

        last_err = None
        for attempt in range(self.max_retries):
            try:
                return fn()
            except Exception as e:
                last_err = e
                if attempt >= self.max_retries - 1:
                    break
                time.sleep(self.retry_sleep * (2**attempt))
        if last_err is not None:
            raise last_err

    def _to_numpy(self, embedding: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(embedding, torch.Tensor):
            if embedding.dtype == torch.bfloat16:
                return embedding.cpu().float().numpy()
            return embedding.cpu().float().numpy()  # .float() for BFloat16 compatibility
        return np.array(embedding, dtype=np.float32)

    def _infer_vector_is_multivector(self, vector_name: str) -> bool:
        info = self.client.get_collection(self.collection_name)
        cfg = getattr(info, "config", None)
        params = getattr(cfg, "params", None) if cfg is not None else None
        vectors = getattr(params, "vectors", None) if params is not None else None
        v = None
        try:
            if isinstance(vectors, dict):
                v = vectors.get(vector_name)
            else:
                v = vectors[vector_name]
        except Exception:
            v = None
        mv = getattr(v, "multivector_config", None) if v is not None else None
        if mv is None and isinstance(v, dict):
            mv = v.get("multivector_config")
        return mv is not None

    def _and_filter(self, base_filter, ids: List[Any]):
        from qdrant_client.http import models as m

        has_id = m.HasIdCondition(has_id=list(ids))
        if base_filter is None:
            return m.Filter(must=[has_id])
        return m.Filter(must=[base_filter, has_id])

    def search_server_side(
        self,
        *,
        query_embedding: Union[torch.Tensor, np.ndarray],
        top_k: int = 100,
        stage1_k: int = 1000,
        stage2_k: int = 300,
        filter_obj=None,
    ) -> List[Dict[str, Any]]:
        from qdrant_client.http import models as m

        query_np = self._to_numpy(query_embedding)

        stage1_query = query_np.mean(axis=0).tolist()
        stage2_query = query_np.tolist()
        stage3_query = query_np.tolist()

        logger.info(f"Stage 1: global prefetch {int(stage1_k)}")

        def _do_stage1():
            return self.client.query_points(
                collection_name=self.collection_name,
                query=stage1_query,
                using=self.global_vector_name,
                limit=int(stage1_k),
                query_filter=filter_obj,
                with_payload=False,
                with_vectors=False,
                timeout=self.request_timeout,
            ).points

        s1 = self._retry_call(_do_stage1)
        if not s1:
            return []
        s1_ids = [p.id for p in s1]
        s1_score = {str(p.id): float(p.score) for p in s1}

        logger.info(f"Stage 2: experimental prefetch {int(stage2_k)} (restricted to stage1)")

        stage2_filter = self._and_filter(filter_obj, s1_ids)

        def _do_stage2():
            return self.client.query_points(
                collection_name=self.collection_name,
                query=stage2_query,
                using=self.experimental_vector_name,
                limit=int(min(int(stage2_k), len(s1_ids))),
                query_filter=stage2_filter,
                with_payload=False,
                with_vectors=False,
                timeout=self.request_timeout,
            ).points

        s2 = self._retry_call(_do_stage2)
        if not s2:
            return []
        s2_ids = [p.id for p in s2]
        s2_score = {str(p.id): float(p.score) for p in s2}

        logger.info(f"Stage 3: exact rerank on initial to top {int(top_k)} (restricted to stage2)")

        stage3_filter = self._and_filter(filter_obj, s2_ids)

        def _do_stage3():
            return self.client.query_points(
                collection_name=self.collection_name,
                query=stage3_query,
                using=self.full_vector_name,
                limit=int(top_k),
                query_filter=stage3_filter,
                with_payload=True,
                with_vectors=False,
                search_params=m.SearchParams(exact=True),
                timeout=self.request_timeout,
            ).points

        s3 = self._retry_call(_do_stage3)
        out = []
        for p in s3:
            pid = str(p.id)
            out.append(
                {
                    "id": p.id,
                    "score_stage1": s1_score.get(pid),
                    "score_stage2": s2_score.get(pid),
                    "score_stage3": float(p.score),
                    "score_final": float(p.score),
                    "payload": p.payload,
                }
            )
        return out
