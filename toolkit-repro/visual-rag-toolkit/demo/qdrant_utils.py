"""Qdrant connection and utility functions."""

import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from visual_rag.retrieval import MultiVectorRetriever


def get_qdrant_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Get Qdrant credentials from session state or environment variables.

    Priority: session_state > QDRANT_URL/QDRANT_API_KEY
    """
    url = st.session_state.get("qdrant_url_input") or os.getenv("QDRANT_URL")
    api_key = st.session_state.get("qdrant_key_input") or os.getenv("QDRANT_API_KEY")
    return url, api_key


def init_qdrant_client_with_creds(url: str, api_key: str):
    try:
        from qdrant_client import QdrantClient

        if not url:
            return None, "QDRANT_URL not configured"
        client = QdrantClient(url=url, api_key=api_key, timeout=60)
        client.get_collections()
        return client, None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner="Connecting to Qdrant...")
def init_qdrant_client():
    url, api_key = get_qdrant_credentials()
    return init_qdrant_client_with_creds(url, api_key)


@st.cache_resource(show_spinner="Loading embedding model...")
def init_embedder(model_name: str):
    try:
        from visual_rag import VisualEmbedder

        embedder = VisualEmbedder(model_name=model_name)
        embedder._load_model()
        return embedder, None
    except Exception as e:
        return None, f"{e}\n\n{traceback.format_exc()}"


@st.cache_data(ttl=300, show_spinner="Fetching collections...")
def get_collections(_url: str, _api_key: str) -> List[str]:
    client, err = init_qdrant_client_with_creds(_url, _api_key)
    if client is None:
        return []
    try:
        collections = client.get_collections().collections
        return sorted([c.name for c in collections])
    except Exception:
        return []


@st.cache_data(ttl=120, show_spinner="Loading collection stats...")
def get_collection_stats(collection_name: str) -> Dict[str, Any]:
    url, api_key = get_qdrant_credentials()
    client, err = init_qdrant_client_with_creds(url, api_key)
    if client is None:
        return {"error": err}
    try:
        info = client.get_collection(collection_name)
        vectors_config = getattr(
            getattr(getattr(info, "config", None), "params", None), "vectors", None
        )
        vector_info = {}
        if vectors_config is not None:
            if hasattr(vectors_config, "items"):
                for name, cfg in vectors_config.items():
                    size = getattr(cfg, "size", None)
                    multivec = getattr(cfg, "multivector_config", None)
                    on_disk = getattr(cfg, "on_disk", None)
                    datatype = str(getattr(cfg, "datatype", "Float32")).replace("Datatype.", "")
                    quantization = getattr(cfg, "quantization_config", None)
                    num_vectors = 1
                    if multivec is not None:
                        comparator = getattr(multivec, "comparator", None)
                        num_vectors = "N" if comparator else 1
                    vector_info[name] = {
                        "size": size,
                        "num_vectors": num_vectors,
                        "is_multivector": multivec is not None,
                        "on_disk": on_disk,
                        "datatype": datatype,
                        "quantization": quantization is not None,
                    }
            elif hasattr(vectors_config, "size"):
                on_disk = getattr(vectors_config, "on_disk", None)
                datatype = str(getattr(vectors_config, "datatype", "Float32")).replace(
                    "Datatype.", ""
                )
                multivec = getattr(vectors_config, "multivector_config", None)
                vector_info["default"] = {
                    "size": getattr(vectors_config, "size", None),
                    "num_vectors": "N" if multivec else 1,
                    "is_multivector": multivec is not None,
                    "on_disk": on_disk,
                    "datatype": datatype,
                }
        return {
            "points_count": getattr(info, "points_count", 0),
            "vectors_count": getattr(info, "vectors_count", getattr(info, "points_count", 0)),
            "status": str(getattr(info, "status", "unknown")),
            "vector_info": vector_info,
            "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
        }
    except Exception as e:
        return {"error": f"{e}\n\n{traceback.format_exc()}"}


@st.cache_data(ttl=60)
def sample_points_cached(
    collection_name: str, n: int, seed: int, _url: str, _api_key: str
) -> List[Dict[str, Any]]:
    client, err = init_qdrant_client_with_creds(_url, _api_key)
    if client is None:
        return []
    try:
        import random

        rng = random.Random(seed)
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=min(n * 10, 100),
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return []
        sampled = rng.sample(points, min(n, len(points)))
        results = []
        for p in sampled:
            payload = dict(p.payload) if p.payload else {}
            results.append(
                {
                    "id": str(p.id),
                    "payload": payload,
                }
            )
        return results
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_vector_sizes(collection_name: str, _url: str, _api_key: str) -> Dict[str, int]:
    client, err = init_qdrant_client_with_creds(_url, _api_key)
    if client is None:
        return {}
    try:
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=1,
            with_payload=False,
            with_vectors=True,
        )
        if not points:
            return {}
        vectors = points[0].vector
        sizes = {}
        if isinstance(vectors, dict):
            for name, vec in vectors.items():
                if isinstance(vec, list):
                    if vec and isinstance(vec[0], list):
                        sizes[name] = len(vec)
                    else:
                        sizes[name] = 1
                else:
                    sizes[name] = 1
        return sizes
    except Exception:
        return {}


def search_collection(
    collection_name: str,
    query: str,
    top_k: int = 10,
    mode: str = "single_full",
    prefetch_k: int = 256,
    stage1_mode: str = "pooled_query_vs_standard_pooling",
    stage1_k: int = 1000,
    stage2_k: int = 300,
    model_name: str = "vidore/colSmol-500M",
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        url, api_key = get_qdrant_credentials()

        print(f"[SEARCH] ---- query: {query[:80]!r} ----")
        print(f"[SEARCH] collection={collection_name}, model={model_name}")
        print(f"[SEARCH] mode={mode}, top_k={top_k}")
        if mode == "two_stage":
            print(f"[SEARCH]   stage1_mode={stage1_mode}, prefetch_k={prefetch_k}")
        elif mode == "three_stage":
            print(f"[SEARCH]   stage1_k={stage1_k}, stage2_k={stage2_k}")

        t_start = time.perf_counter()

        retriever = MultiVectorRetriever(
            collection_name=collection_name,
            model_name=model_name,
            qdrant_url=url,
            qdrant_api_key=api_key,
        )
        t_retriever = time.perf_counter()
        print(f"[SEARCH] retriever ready in {t_retriever - t_start:.2f}s")

        if mode == "three_stage":
            q_emb = retriever.embedder.embed_query(query)
            if hasattr(q_emb, "cpu"):
                q_emb = q_emb.cpu().numpy()
            t_embed = time.perf_counter()
            print(f"[SEARCH] query embedded in {t_embed - t_retriever:.2f}s (shape={q_emb.shape})")
            results = retriever.search_embedded(
                query_embedding=q_emb,
                top_k=top_k,
                mode=mode,
                stage1_k=stage1_k,
                stage2_k=stage2_k,
            )
        else:
            results = retriever.search(
                query=query,
                top_k=top_k,
                mode=mode,
                prefetch_k=prefetch_k,
                stage1_mode=stage1_mode,
            )

        t_done = time.perf_counter()
        print(f"[SEARCH] search done in {t_done - t_retriever:.2f}s, total={t_done - t_start:.2f}s")
        print(f"[SEARCH] returned {len(results)} results")
        if results:
            scores = [r.get("score_final", r.get("score_stage1", 0)) for r in results[:3]]
            print(f"[SEARCH] top-3 scores: {[f'{s:.4f}' for s in scores]}")
        return results, None
    except Exception as e:
        return [], f"{e}\n\n{traceback.format_exc()}"
