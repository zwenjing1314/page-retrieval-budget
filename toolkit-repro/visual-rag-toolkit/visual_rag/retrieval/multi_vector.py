import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
import torch

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    load_dotenv = None

try:
    from qdrant_client import QdrantClient

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None

from visual_rag.embedding.visual_embedder import VisualEmbedder
from visual_rag.retrieval.single_stage import SingleStageRetriever
from visual_rag.retrieval.three_stage import ThreeStageRetriever
from visual_rag.retrieval.two_stage import TwoStageRetriever


class MultiVectorRetriever:
    @staticmethod
    def _maybe_load_dotenv() -> None:
        if not DOTENV_AVAILABLE:
            return
        if os.path.exists(".env"):
            load_dotenv(".env")

    def __init__(
        self,
        collection_name: str,
        model_name: str = "vidore/colSmol-500M",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        request_timeout: int = 120,
        max_retries: int = 3,
        retry_sleep: float = 0.5,
        qdrant_client=None,
        embedder: Optional[VisualEmbedder] = None,
        experimental_vector_name: str = "experimental_pooling",
    ):
        if qdrant_client is None:
            self._maybe_load_dotenv()
            if not QDRANT_AVAILABLE:
                raise ImportError(
                    "Qdrant client not installed. Install with: pip install visual-rag-toolkit[qdrant]"
                )

            qdrant_url = (
                qdrant_url or os.getenv("QDRANT_URL") or os.getenv("SIGIR_QDRANT_URL")  # legacy
            )
            if not qdrant_url:
                raise ValueError("QDRANT_URL is required (pass qdrant_url or set env var).")

            qdrant_api_key = (
                qdrant_api_key
                or os.getenv("QDRANT_API_KEY")
                or os.getenv("SIGIR_QDRANT_KEY")  # legacy
            )

            grpc_port = None
            if prefer_grpc:
                try:
                    parsed = urlparse(qdrant_url)
                    port = parsed.port
                    if port == 6333:
                        grpc_port = 6334
                except Exception:
                    pass

            def _make_client(use_grpc: bool):
                return QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    timeout=request_timeout,
                    prefer_grpc=bool(use_grpc),
                    grpc_port=grpc_port,
                    check_compatibility=False,
                )

            client = _make_client(prefer_grpc)
            if prefer_grpc:
                try:
                    _ = client.get_collections()
                except Exception as e:
                    msg = str(e)
                    if (
                        "StatusCode.PERMISSION_DENIED" in msg
                        or "http2 header with status: 403" in msg
                    ):
                        client = _make_client(False)
                    else:
                        raise
            qdrant_client = client

        self.client = qdrant_client
        self.collection_name = collection_name

        self.embedder = embedder or VisualEmbedder(model_name=model_name)

        self._two_stage = TwoStageRetriever(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            experimental_vector_name=str(experimental_vector_name),
            request_timeout=request_timeout,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )
        self._three_stage = ThreeStageRetriever(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            experimental_vector_name=str(experimental_vector_name),
            request_timeout=request_timeout,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )
        self._single_stage = SingleStageRetriever(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            experimental_vector_name=str(experimental_vector_name),
            request_timeout=request_timeout,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )

    def build_filter(
        self,
        year: Optional[Any] = None,
        source: Optional[str] = None,
        district: Optional[str] = None,
        filename: Optional[str] = None,
        has_text: Optional[bool] = None,
    ):
        return self._two_stage.build_filter(
            year=year,
            source=source,
            district=district,
            filename=filename,
            has_text=has_text,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "single_full",
        prefetch_k: Optional[int] = None,
        stage1_mode: str = "pooled_query_vs_standard_pooling",
        filter_obj=None,
        return_embeddings: bool = False,
    ) -> List[Dict[str, Any]]:
        q = self.embedder.embed_query(query)
        if isinstance(q, torch.Tensor):
            # .float() converts BFloat16 to Float32 (numpy doesn't support BFloat16)
            query_embedding = q.detach().cpu().float().numpy()
        else:
            query_embedding = np.asarray(q, dtype=np.float32)

        return self.search_embedded(
            query_embedding=query_embedding,
            top_k=top_k,
            mode=mode,
            prefetch_k=prefetch_k,
            stage1_mode=stage1_mode,
            filter_obj=filter_obj,
            return_embeddings=return_embeddings,
        )

    def search_embedded(
        self,
        *,
        query_embedding,
        top_k: int = 10,
        mode: str = "single_full",
        prefetch_k: Optional[int] = None,
        stage1_mode: str = "pooled_query_vs_standard_pooling",
        stage1_k: Optional[int] = None,
        stage2_k: Optional[int] = None,
        filter_obj=None,
        return_embeddings: bool = False,
    ) -> List[Dict[str, Any]]:
        if mode == "single_full":
            return self._single_stage.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_obj=filter_obj,
                strategy="multi_vector",
            )
        elif mode in ("single_tiles", "single_pooled"):
            # single_tiles: MaxSim on pooled vectors (multi-vector)
            # single_pooled: backward-compatible alias (pooled query vs pooled vectors)
            return self._single_stage.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_obj=filter_obj,
                strategy=("tiles_maxsim" if mode == "single_tiles" else "pooled_tile"),
            )
        elif mode == "single_global":
            return self._single_stage.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_obj=filter_obj,
                strategy="pooled_global",
            )
        elif mode == "single_experimental_tokens":
            return self._single_stage.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_obj=filter_obj,
                strategy="experimental_maxsim",
            )
        elif mode == "single_experimental_pooled":
            return self._single_stage.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_obj=filter_obj,
                strategy="pooled_experimental",
            )
        elif mode == "two_stage":
            return self._two_stage.search_server_side(
                query_embedding=query_embedding,
                top_k=top_k,
                prefetch_k=prefetch_k,
                filter_obj=filter_obj,
                stage1_mode=stage1_mode,
            )
        elif mode == "three_stage":
            return self._three_stage.search_server_side(
                query_embedding=query_embedding,
                top_k=top_k,
                stage1_k=stage1_k,
                stage2_k=stage2_k,
                filter_obj=filter_obj,
                stage1_mode=stage1_mode,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
