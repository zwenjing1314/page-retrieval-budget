import json
import logging
import os
import tempfile
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.CRITICAL
)
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

os.environ.setdefault("STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION", "false")
os.environ.setdefault("STREAMLIT_SERVER_ENABLE_CORS", "false")
os.environ.setdefault("STREAMLIT_SERVER_MAX_UPLOAD_SIZE", "500")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

try:
    from benchmarks.vidore_tatdqa_test.dataset_loader import load_vidore_beir_dataset
    from benchmarks.vidore_tatdqa_test.metrics import mrr_at_k, ndcg_at_k, recall_at_k
    from visual_rag import VisualEmbedder
    from visual_rag.indexing import QdrantIndexer
    from visual_rag.retrieval import MultiVectorRetriever

    VISUAL_RAG_AVAILABLE = True
except ImportError:
    VISUAL_RAG_AVAILABLE = False

load_dotenv(Path(__file__).parent / ".env")
if (Path(__file__).parent.parent / ".env").exists():
    load_dotenv(Path(__file__).parent.parent / ".env")

st.set_page_config(
    page_title="Visual RAG Toolkit - Demo",
    page_icon="🔬",
    layout="wide",
)

AVAILABLE_MODELS = [
    "vidore/colpali-v1.3",
    "vidore/colSmol-500M",
]

BENCHMARK_DATASETS = [
    "vidore/esg_reports_v2",
    "vidore/biomedical_lectures_v2",
    "vidore/economics_reports_v2",
]

DATASET_STATS = {
    "vidore/esg_reports_v2": {"docs": 1538, "queries": 228},
    "vidore/biomedical_lectures_v2": {"docs": 1016, "queries": 640},
    "vidore/economics_reports_v2": {"docs": 452, "queries": 232},
}

RETRIEVAL_MODES = [
    "single_full",
    "single_tiles",
    "single_global",
    "two_stage",
    "three_stage",
]

STAGE1_MODES = [
    "tokens_vs_standard_pooling",
    "tokens_vs_experimental_pooling",
    "pooled_query_vs_standard_pooling",
    "pooled_query_vs_experimental_pooling",
    "pooled_query_vs_global",
]


def get_qdrant_credentials():
    url = (
        st.session_state.get("qdrant_url_input")
        or os.getenv("SIGIR_QDRANT_URL")
        or os.getenv("DEST_QDRANT_URL")
        or os.getenv("QDRANT_URL")
    )
    api_key = st.session_state.get("qdrant_key_input") or (
        os.getenv("SIGIR_QDRANT_KEY")
        or os.getenv("SIGIR_QDRANT_API_KEY")
        or os.getenv("DEST_QDRANT_API_KEY")
        or os.getenv("QDRANT_API_KEY")
    )
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

        return VisualEmbedder(model_name=model_name), None
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
    stage1_mode: str = "tokens_vs_standard_pooling",
    stage1_k: int = 1000,
    stage2_k: int = 300,
    model_name: str = "vidore/colSmol-500M",
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        from visual_rag.retrieval import MultiVectorRetriever

        retriever = MultiVectorRetriever(
            collection_name=collection_name,
            model_name=model_name,
        )
        if mode == "three_stage":
            q_emb = retriever.embedder.embed_query(query)
            if hasattr(q_emb, "cpu"):
                q_emb = q_emb.cpu().numpy()
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
        return results, None
    except Exception as e:
        return [], f"{e}\n\n{traceback.format_exc()}"


def load_results_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def get_available_results() -> List[Path]:
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        return []
    results = []
    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            for f in subdir.glob("*.json"):
                if "index_failures" not in f.name:
                    results.append(f)
    return sorted(results, key=lambda x: x.stat().st_mtime, reverse=True)


def find_main_result_file(collection: str, mode: str) -> Optional[Path]:
    results = get_available_results()
    for r in results:
        if collection in str(r) and mode in r.name:
            if "__vidore_" not in r.name:
                return r
    return results[0] if results else None


def build_index_command(config: Dict[str, Any]) -> str:
    cmd_parts = ["python -m benchmarks.vidore_beir_qdrant.run_qdrant_beir"]
    cmd_parts.append(f"--datasets {' '.join(config['datasets'])}")
    cmd_parts.append(f"--collection {config['collection']}")
    cmd_parts.append(f"--model {config['model']}")
    cmd_parts.append("--index")
    if config.get("recreate"):
        cmd_parts.append("--recreate")
    if config.get("resume"):
        cmd_parts.append("--resume")
    if config.get("prefer_grpc"):
        cmd_parts.append("--prefer-grpc")
    else:
        cmd_parts.append("--no-prefer-grpc")
    cmd_parts.append(f"--torch-dtype {config.get('torch_dtype', 'float16')}")
    cmd_parts.append(f"--qdrant-vector-dtype {config.get('qdrant_vector_dtype', 'float16')}")
    cmd_parts.append(f"--batch-size {config.get('batch_size', 4)}")
    cmd_parts.append(f"--upload-batch-size {config.get('upload_batch_size', 8)}")
    cmd_parts.append(f"--qdrant-timeout {config.get('qdrant_timeout', 180)}")
    cmd_parts.append(f"--qdrant-retries {config.get('qdrant_retries', 5)}")
    if config.get("crop_empty"):
        cmd_parts.append("--crop-empty")
        cmd_parts.append(f"--crop-empty-percentage-to-remove {config.get('crop_percentage', 0.99)}")
    if config.get("no_cloudinary"):
        cmd_parts.append("--no-cloudinary")
    cmd_parts.append("--no-eval")
    return " \\\n  ".join(cmd_parts)


def build_eval_command(config: Dict[str, Any]) -> str:
    cmd_parts = ["python -m benchmarks.vidore_beir_qdrant.run_qdrant_beir"]
    cmd_parts.append(f"--datasets {' '.join(config['datasets'])}")
    cmd_parts.append(f"--collection {config['collection']}")
    cmd_parts.append(f"--model {config['model']}")
    cmd_parts.append(f"--mode {config['mode']}")
    if config["mode"] == "two_stage":
        cmd_parts.append(f"--stage1-mode {config.get('stage1_mode', 'tokens_vs_standard_pooling')}")
        cmd_parts.append(f"--prefetch-k {config.get('prefetch_k', 256)}")
    elif config["mode"] == "three_stage":
        cmd_parts.append(f"--stage1-k {config.get('stage1_k', 1000)}")
        cmd_parts.append(f"--stage2-k {config.get('stage2_k', 300)}")
    cmd_parts.append(f"--top-k {config.get('top_k', 100)}")
    cmd_parts.append(f"--evaluation-scope {config.get('evaluation_scope', 'union')}")
    if config.get("prefer_grpc"):
        cmd_parts.append("--prefer-grpc")
    else:
        cmd_parts.append("--no-prefer-grpc")
    cmd_parts.append(f"--torch-dtype {config.get('torch_dtype', 'float16')}")
    cmd_parts.append(f"--qdrant-vector-dtype {config.get('qdrant_vector_dtype', 'float16')}")
    cmd_parts.append(f"--qdrant-timeout {config.get('qdrant_timeout', 180)}")
    if config.get("result_prefix"):
        cmd_parts.append(f"--output {config['result_prefix']}")
    return " \\\n  ".join(cmd_parts)


def generate_python_eval_code(config: Dict[str, Any]) -> str:
    datasets_str = ", ".join([f'"{ds}"' for ds in config.get("datasets", [])])
    mode = config.get("mode", "single_full")
    model = config.get("model", "vidore/colpali-v1.3")
    collection = config.get("collection", "")
    top_k = config.get("top_k", 100)
    scope = config.get("evaluation_scope", "union")
    prefer_grpc = config.get("prefer_grpc", True)

    code_lines = [
        "import os",
        "from qdrant_client import QdrantClient",
        "from visual_rag import VisualEmbedder",
        "from visual_rag.retrieval import MultiVectorRetriever",
        "",
        "# Configuration",
        f'COLLECTION = "{collection}"',
        f'MODEL = "{model}"',
        f"TOP_K = {top_k}",
        f"DATASETS = [{datasets_str}]",
        "",
        "# Initialize clients",
        "client = QdrantClient(",
        '    url=os.getenv("QDRANT_URL"),',
        '    api_key=os.getenv("QDRANT_API_KEY"),',
        f"    prefer_grpc={prefer_grpc},",
        ")",
        "",
        "embedder = VisualEmbedder(",
        "    model_name=MODEL,",
        f'    torch_dtype="{config.get("torch_dtype", "float16")}",',
        ")",
        "",
        "# Initialize retriever",
        "retriever = MultiVectorRetriever(",
        "    client=client,",
        "    collection_name=COLLECTION,",
        "    embedder=embedder,",
        ")",
        "",
    ]

    if mode == "single_full":
        code_lines.extend(
            [
                "# Single-stage full retrieval",
                "def search(query: str):",
                "    query_embedding = embedder.embed_query(query)",
                "    return retriever.search_single_stage(",
                "        query_embedding=query_embedding,",
                f"        limit={top_k},",
                '        vector_name="initial",',
                "    )",
            ]
        )
    elif mode == "single_tiles":
        code_lines.extend(
            [
                "# Single-stage tiles retrieval",
                "def search(query: str):",
                "    query_embedding = embedder.embed_query(query)",
                "    return retriever.search_single_stage(",
                "        query_embedding=query_embedding,",
                f"        limit={top_k},",
                '        vector_name="mean_pooling",',
                "    )",
            ]
        )
    elif mode == "single_global":
        code_lines.extend(
            [
                "# Single-stage global retrieval",
                "def search(query: str):",
                "    query_embedding = embedder.embed_query(query)",
                "    return retriever.search_single_stage(",
                "        query_embedding=query_embedding,",
                f"        limit={top_k},",
                '        vector_name="global_pooling",',
                "    )",
            ]
        )
    elif mode == "two_stage":
        prefetch_k = config.get("prefetch_k", 256)
        stage1_mode = config.get("stage1_mode", "tokens_vs_standard_pooling")
        code_lines.extend(
            [
                "# Two-stage retrieval",
                "from visual_rag.retrieval import TwoStageRetriever",
                "",
                "two_stage = TwoStageRetriever(",
                "    client=client,",
                "    collection_name=COLLECTION,",
                "    embedder=embedder,",
                ")",
                "",
                "def search(query: str):",
                "    query_embedding = embedder.embed_query(query)",
                "    return two_stage.search(",
                "        query_embedding=query_embedding,",
                f"        prefetch_limit={prefetch_k},",
                f"        limit={top_k},",
                f'        stage1_mode="{stage1_mode}",',
                "    )",
            ]
        )
    elif mode == "three_stage":
        stage1_k = config.get("stage1_k", 1000)
        stage2_k = config.get("stage2_k", 300)
        code_lines.extend(
            [
                "# Three-stage retrieval",
                "from visual_rag.retrieval import ThreeStageRetriever",
                "",
                "three_stage = ThreeStageRetriever(",
                "    client=client,",
                "    collection_name=COLLECTION,",
                "    embedder=embedder,",
                ")",
                "",
                "def search(query: str):",
                "    query_embedding = embedder.embed_query(query)",
                "    return three_stage.search(",
                "        query_embedding=query_embedding,",
                f"        stage1_limit={stage1_k},",
                f"        stage2_limit={stage2_k},",
                f"        limit={top_k},",
                "    )",
            ]
        )

    if scope == "per_dataset":
        code_lines.extend(
            [
                "",
                "# Per-dataset filtering",
                "from qdrant_client.models import Filter, FieldCondition, MatchValue",
                "",
                'def search_dataset(query: str, dataset: str = "vidore/esg_reports_v2"):',
                "    query_embedding = embedder.embed_query(query)",
                "    dataset_filter = Filter(",
                "        must=[FieldCondition(",
                '            key="dataset",',
                "            match=MatchValue(value=dataset),",
                "        )]",
                "    )",
                "    # Add filter to your search call",
            ]
        )

    code_lines.extend(
        [
            "",
            "# Example usage",
            'results = search("What is the company revenue?")',
            "for r in results:",
            "    print(f\"Score: {r.score:.4f}, Doc: {r.payload.get('doc_id')}\")",
        ]
    )

    return "\n".join(code_lines)


def run_pythonic_evaluation(config: Dict[str, Any], progress_callback=None) -> Dict[str, Any]:
    if not VISUAL_RAG_AVAILABLE:
        raise ImportError("visual_rag package not available")

    url, api_key = get_qdrant_credentials()
    if not url:
        raise ValueError("QDRANT_URL not configured")

    datasets = config.get("datasets", [])
    collection = config["collection"]
    model = config.get("model", "vidore/colpali-v1.3")
    mode = config.get("mode", "single_full")
    top_k = config.get("top_k", 100)
    prefetch_k = config.get("prefetch_k", 256)
    stage1_mode = config.get("stage1_mode", "tokens_vs_standard_pooling")
    stage1_k = config.get("stage1_k", 1000)
    stage2_k = config.get("stage2_k", 300)
    evaluation_scope = config.get("evaluation_scope", "union")
    prefer_grpc = config.get("prefer_grpc", True)
    torch_dtype = config.get("torch_dtype", "float16")

    output_lines = []

    def log(msg):
        output_lines.append(msg)
        if progress_callback:
            progress_callback("\n".join(output_lines), None)

    log(f"[Pythonic Eval] Initializing embedder: {model}")
    embedder = VisualEmbedder(model_name=model, torch_dtype=torch_dtype)

    log(f"[Pythonic Eval] Connecting to Qdrant collection: {collection}")
    retriever = MultiVectorRetriever(
        collection_name=collection,
        model_name=model,
        qdrant_url=url,
        qdrant_api_key=api_key,
        prefer_grpc=prefer_grpc,
        embedder=embedder,
    )

    all_queries = []
    all_qrels: Dict[str, Dict[str, int]] = {}
    dataset_queries: Dict[str, List] = {}
    dataset_qrels: Dict[str, Dict[str, Dict[str, int]]] = {}

    for ds_name in datasets:
        log(f"[Pythonic Eval] Loading dataset: {ds_name}")
        corpus, queries, qrels = load_vidore_beir_dataset(ds_name)
        dataset_queries[ds_name] = queries
        dataset_qrels[ds_name] = qrels
        all_queries.extend(queries)
        for qid, rels in qrels.items():
            all_qrels[qid] = rels
        log(f"  → {len(corpus)} docs, {len(queries)} queries")

    def evaluate_queries(queries, qrels, filter_obj=None):
        if not queries:
            return {"ndcg@10": 0.0, "recall@10": 0.0, "mrr@10": 0.0, "num_queries": 0}

        ndcg10_vals = []
        recall10_vals = []
        mrr10_vals = []
        latencies = []

        query_texts = [q.text for q in queries]
        log(f"[Pythonic Eval] Embedding {len(query_texts)} queries...")
        query_embeddings = embedder.embed_queries(query_texts, show_progress=False)

        for i, (q, qemb) in enumerate(zip(queries, query_embeddings)):
            start = time.time()

            try:
                import torch

                if isinstance(qemb, torch.Tensor):
                    qemb_np = qemb.detach().cpu().numpy()
                else:
                    qemb_np = qemb.numpy()
            except ImportError:
                qemb_np = qemb.numpy()

            results = retriever.search_embedded(
                query_embedding=qemb_np,
                top_k=max(100, top_k),
                mode=mode,
                prefetch_k=prefetch_k,
                stage1_mode=stage1_mode,
                stage1_k=stage1_k,
                stage2_k=stage2_k,
                filter_obj=filter_obj,
            )
            latencies.append((time.time() - start) * 1000)

            ranking = [str(r["id"]) for r in results]
            rels = qrels.get(q.query_id, {})

            ndcg10_vals.append(ndcg_at_k(ranking, rels, k=10))
            recall10_vals.append(recall_at_k(ranking, rels, k=10))
            mrr10_vals.append(mrr_at_k(ranking, rels, k=10))

            if (i + 1) % 50 == 0:
                log(f"  → Processed {i+1}/{len(queries)} queries")
                if progress_callback:
                    progress_callback("\n".join(output_lines), (i + 1) / len(queries))

        return {
            "ndcg@10": float(np.mean(ndcg10_vals)),
            "recall@10": float(np.mean(recall10_vals)),
            "mrr@10": float(np.mean(mrr10_vals)),
            "avg_latency_ms": float(np.mean(latencies)),
            "num_queries": len(queries),
        }

    results = {}

    if evaluation_scope == "union":
        log(f"\n[Pythonic Eval] Evaluating UNION ({len(all_queries)} queries)...")
        union_metrics = evaluate_queries(all_queries, all_qrels)
        results["union"] = union_metrics
        log(f"  → NDCG@10: {union_metrics['ndcg@10']:.4f}")
        log(f"  → Recall@10: {union_metrics['recall@10']:.4f}")
        log(f"  → MRR@10: {union_metrics['mrr@10']:.4f}")
    else:
        for ds_name in datasets:
            log(f"\n[Pythonic Eval] Evaluating {ds_name}...")
            queries = dataset_queries[ds_name]
            qrels = dataset_qrels[ds_name]
            metrics = evaluate_queries(queries, qrels)
            results[ds_name] = metrics
            log(f"  → NDCG@10: {metrics['ndcg@10']:.4f}")
            log(f"  → Recall@10: {metrics['recall@10']:.4f}")

    log("\n" + "=" * 50)
    log("[Pythonic Eval] COMPLETE!")

    final_output = {
        "config": {
            "collection": collection,
            "model": model,
            "mode": mode,
            "datasets": datasets,
            "evaluation_scope": evaluation_scope,
        },
        "results": results,
    }

    return {"output": "\n".join(output_lines), "metrics": final_output}


def run_pythonic_indexing(config: Dict[str, Any], progress_callback=None) -> Dict[str, Any]:
    if not VISUAL_RAG_AVAILABLE:
        raise ImportError("visual_rag package not available")

    url, api_key = get_qdrant_credentials()
    if not url:
        raise ValueError("QDRANT_URL not configured")

    datasets = config.get("datasets", [])
    collection = config["collection"]
    model = config.get("model", "vidore/colpali-v1.3")
    recreate = config.get("recreate", False)
    batch_size = config.get("batch_size", 4)
    torch_dtype = config.get("torch_dtype", "float16")
    qdrant_vector_dtype = config.get("qdrant_vector_dtype", "float16")
    prefer_grpc = config.get("prefer_grpc", True)

    output_lines = []

    def log(msg):
        output_lines.append(msg)
        if progress_callback:
            progress_callback("\n".join(output_lines), None)

    log(f"[Pythonic Index] Initializing embedder: {model}")
    embedder = VisualEmbedder(model_name=model, torch_dtype=torch_dtype)

    log("[Pythonic Index] Connecting to Qdrant...")
    indexer = QdrantIndexer(
        url=url,
        api_key=api_key,
        collection_name=collection,
        prefer_grpc=prefer_grpc,
        vector_datatype=qdrant_vector_dtype,
    )

    log(f"[Pythonic Index] Creating collection: {collection}")
    indexer.create_collection(force_recreate=recreate)

    payload_fields = [
        {"field": "dataset", "type": "keyword"},
        {"field": "doc_id", "type": "keyword"},
        {"field": "source_doc_id", "type": "keyword"},
    ]
    indexer.create_payload_indexes(fields=payload_fields)

    total_uploaded = 0

    for ds_name in datasets:
        log(f"\n[Pythonic Index] Loading dataset: {ds_name}")
        corpus, queries, qrels = load_vidore_beir_dataset(ds_name)
        log(f"  → {len(corpus)} documents to index")

        for i in range(0, len(corpus), batch_size):
            batch = corpus[i : i + batch_size]

            images = []
            for doc in batch:
                img = doc.image if hasattr(doc, "image") else doc.get("image")
                if img is not None:
                    images.append(img)

            if not images:
                continue

            log(
                f"  → Embedding batch {i//batch_size + 1}/{(len(corpus) + batch_size - 1)//batch_size}..."
            )
            embeddings = embedder.embed_images(images)

            points = []
            for j, (doc, emb) in enumerate(zip(batch, embeddings)):
                doc_id = doc.doc_id if hasattr(doc, "doc_id") else doc.get("doc_id", str(i + j))

                if hasattr(emb, "cpu"):
                    emb_np = emb.cpu().numpy()
                else:
                    emb_np = np.array(emb)

                tile_pooled = emb_np.reshape(-1, 4, emb_np.shape[-1]).mean(axis=1)
                global_pooled = emb_np.mean(axis=0)

                points.append(
                    {
                        "id": f"{ds_name}_{doc_id}".replace("/", "_"),
                        "visual_embedding": emb_np,
                        "tile_pooled_embedding": tile_pooled,
                        "experimental_pooled_embedding": tile_pooled,
                        "global_pooled_embedding": global_pooled,
                        "metadata": {
                            "dataset": ds_name,
                            "doc_id": doc_id,
                            "source_doc_id": doc_id,
                        },
                    }
                )

            uploaded = indexer.upload_batch(points)
            total_uploaded += uploaded

            if progress_callback:
                prog = (i + len(batch)) / len(corpus)
                progress_callback("\n".join(output_lines), prog)

        log(f"  → Finished {ds_name}: {total_uploaded} points uploaded")

    log("\n" + "=" * 50)
    log(f"[Pythonic Index] COMPLETE! Total: {total_uploaded} points")

    return {"output": "\n".join(output_lines), "total_uploaded": total_uploaded}


def render_header():
    st.markdown(
        """
    <div style="text-align: center; padding: 10px 0 15px 0;">
        <h1 style="
            font-family: 'Georgia', serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: #1a1a2e;
            letter-spacing: 3px;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        ">
            🔬 Visual RAG Toolkit
        </h1>
        <p style="
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 0.95rem;
            color: #666;
            margin-top: 5px;
            letter-spacing: 1px;
        ">
            SIGIR 2026 Demo - Multi-Vector Visual Document Retrieval
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.subheader("🔑 Qdrant Credentials")

        env_url = (
            os.getenv("SIGIR_QDRANT_URL")
            or os.getenv("DEST_QDRANT_URL")
            or os.getenv("QDRANT_URL")
            or ""
        )
        env_key = (
            os.getenv("SIGIR_QDRANT_KEY")
            or os.getenv("SIGIR_QDRANT_API_KEY")
            or os.getenv("DEST_QDRANT_API_KEY")
            or os.getenv("QDRANT_API_KEY")
            or ""
        )

        qdrant_url = st.text_input(
            "Qdrant URL",
            value=st.session_state.get("qdrant_url_input", env_url),
            key="qdrant_url_widget",
            placeholder="https://xxx.cloud.qdrant.io:6333",
        )
        qdrant_key = st.text_input(
            "API Key",
            value=st.session_state.get("qdrant_key_input", env_key),
            key="qdrant_key_widget",
            type="password",
        )

        if qdrant_url != st.session_state.get(
            "qdrant_url_input"
        ) or qdrant_key != st.session_state.get("qdrant_key_input"):
            st.session_state["qdrant_url_input"] = qdrant_url
            st.session_state["qdrant_key_input"] = qdrant_key
            get_collections.clear()
            get_collection_stats.clear()
            sample_points_cached.clear()

        st.divider()

        st.subheader("📡 Status")
        url, api_key = get_qdrant_credentials()
        client, err = init_qdrant_client_with_creds(url, api_key)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if client:
                st.success("Qdrant ✓", icon="✅")
            else:
                st.error("Qdrant ✗", icon="❌")
        with col_s2:
            cloudinary_ok = all(
                [os.getenv("CLOUDINARY_CLOUD_NAME"), os.getenv("CLOUDINARY_API_KEY")]
            )
            if cloudinary_ok:
                st.success("Cloudinary ✓", icon="✅")
            else:
                st.warning("Cloudinary ✗", icon="⚠️")

        st.divider()

        with st.expander("📦 Collection", expanded=True):
            collections = get_collections(url, api_key)
            if collections:
                prev_collection = st.session_state.get("active_collection")
                selected = st.selectbox(
                    "Select Collection",
                    options=collections,
                    key="sidebar_collection",
                    label_visibility="collapsed",
                )
                if selected:
                    if selected != prev_collection:
                        st.session_state["model_loaded"] = False
                        st.session_state["loaded_model_key"] = None
                    st.session_state["active_collection"] = selected
                    stats = get_collection_stats(selected)
                    if "error" not in stats:
                        col1, col2 = st.columns(2)
                        col1.metric("Points", f"{stats.get('points_count', 0):,}")
                        status_raw = (
                            stats.get("status", "unknown").replace("CollectionStatus.", "").lower()
                        )
                        status_icon = (
                            "🟢"
                            if status_raw == "green"
                            else "🟡" if status_raw == "yellow" else "🔴"
                        )
                        col2.metric("Status", status_icon)

                        points = stats.get("points_count", 0)
                        indexed = stats.get("indexed_vectors_count", 0) or 0
                        is_indexed = indexed >= points and points > 0
                        col3, col4 = st.columns(2)
                        col3.metric("Indexed", f"{indexed:,}")
                        col4.metric("HNSW", "✅" if is_indexed else "⏳")

                        vector_info = stats.get("vector_info", {})
                        if vector_info:
                            st.markdown("---")
                            st.markdown("**🔢 Vectors**")
                            vec_sizes = get_vector_sizes(selected, url, api_key)
                            rows = []
                            sorted_names = sorted(vector_info.keys(), key=lambda x: len(x))
                            for vname in sorted_names:
                                vinfo = vector_info[vname]
                                dim = vinfo.get("size", "?")
                                num_vec = vec_sizes.get(vname, vinfo.get("num_vectors", 1))
                                dtype = vinfo.get("datatype", "?").upper()
                                on_disk = vinfo.get("on_disk", False)
                                disk_icon = "💾" if on_disk else "🧠"
                                dim_str = f"{num_vec}×{dim}"
                                rows.append(
                                    f"<tr><td style='text-align:left;padding-right:12px;'><code>{vname}</code></td><td style='text-align:right;'>{dim_str}, {dtype}, {disk_icon}</td></tr>"
                                )
                            table_html = f"<table style='width:100%;font-size:0.85em;'>{''.join(rows)}</table>"
                            st.markdown(table_html, unsafe_allow_html=True)
                    else:
                        st.error("Error loading stats")
            else:
                st.info("No collections")

        with st.expander("⚙️ Admin", expanded=False):
            active = st.session_state.get("active_collection")
            if active and client:
                stats = get_collection_stats(active)
                vector_info = stats.get("vector_info", {})
                if vector_info:
                    st.markdown("**Change Storage**")
                    vector_names = sorted(vector_info.keys())
                    sel_vec = st.selectbox("Vector", vector_names, key="admin_vec")
                    if sel_vec:
                        current_on_disk = vector_info.get(sel_vec, {}).get("on_disk", False)
                        current_in_ram = not current_on_disk
                        st.caption(f"Current: {'🧠 RAM' if current_in_ram else '💾 Disk'}")
                        target_in_ram = st.toggle(
                            "Move to RAM", value=current_in_ram, key=f"admin_ram_{sel_vec}"
                        )
                        if target_in_ram != current_in_ram:
                            if st.button("💾 Apply Change", key="admin_apply"):
                                try:
                                    from qdrant_client.models import VectorParamsDiff

                                    client.update_collection(
                                        collection_name=active,
                                        vectors_config={
                                            sel_vec: VectorParamsDiff(on_disk=not target_in_ram)
                                        },
                                    )
                                    get_collection_stats.clear()
                                    st.success(f"Updated {sel_vec}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed: {e}")
                        else:
                            st.caption("Toggle to change storage location")
                else:
                    st.info("No vectors")
            else:
                st.info("Select a collection")

        st.divider()

        if st.button("🔄 Refresh", type="secondary", use_container_width=True):
            get_collections.clear()
            get_collection_stats.clear()
            sample_points_cached.clear()
            st.rerun()


def render_upload_tab():
    if "upload_success" in st.session_state:
        msg = st.session_state.pop("upload_success")
        st.toast(f"✅ {msg}", icon="🎉")
        st.balloons()

    st.subheader("📤 PDF Upload & Processing")

    col_upload, col_config = st.columns([3, 2])

    with col_config:
        st.markdown("##### Configuration")

        c1, c2 = st.columns(2)
        with c1:
            model_name = st.selectbox("Model", AVAILABLE_MODELS, index=1, key="upload_model")
        with c2:
            collection_name = st.text_input(
                "Collection", value="my_collection", key="upload_collection_input"
            )

        c3, c4 = st.columns(2)
        with c3:
            crop_empty = st.toggle("Crop Margins", value=True, key="upload_crop")
        with c4:
            use_cloudinary = st.toggle("Cloudinary", value=True, key="upload_cloudinary")

        if crop_empty:
            crop_pct = st.slider("Crop %", 0.5, 0.99, 0.9, 0.01, key="upload_crop_pct")
        else:
            crop_pct = 0.9

    with col_upload:
        uploaded_files = st.file_uploader(
            "Select PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
        )

        if uploaded_files:
            st.success(f"**{len(uploaded_files)} file(s) selected**")

            if st.button("🚀 Process PDFs", type="primary", key="process_btn"):
                process_pdfs(
                    uploaded_files,
                    model_name,
                    collection_name,
                    crop_empty,
                    crop_pct,
                    use_cloudinary,
                )

    if st.session_state.get("last_upload_result"):
        st.divider()
        render_upload_results()


def process_pdfs(uploaded_files, model_name, collection_name, crop_empty, crop_pct, use_cloudinary):
    logs = []
    log_container = st.empty()
    progress = st.progress(0)
    status = st.empty()

    def log(msg):
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        log_container.code("\n".join(logs[-30:]), language="text")

    try:
        log(f"Starting: {len(uploaded_files)} files, model={model_name.split('/')[-1]}")

        from visual_rag import VisualEmbedder
        from visual_rag.indexing import CloudinaryUploader, ProcessingPipeline, QdrantIndexer

        log("Loading model...")
        embedder = VisualEmbedder(model_name=model_name)

        url, api_key = get_qdrant_credentials()
        log("Connecting to Qdrant...")
        indexer = QdrantIndexer(url=url, api_key=api_key, collection_name=collection_name)
        indexer.create_collection(force_recreate=False)

        cloudinary_uploader = None
        if use_cloudinary:
            try:
                cloudinary_uploader = CloudinaryUploader()
                log("Cloudinary ready")
            except Exception as e:
                log(f"Cloudinary failed: {e}")

        pipeline = ProcessingPipeline(
            embedder=embedder,
            indexer=indexer,
            cloudinary_uploader=cloudinary_uploader,
            crop_empty=crop_empty,
            crop_empty_percentage_to_remove=crop_pct,
        )

        total_uploaded, total_skipped, total_failed = 0, 0, 0
        file_results = []

        for i, f in enumerate(uploaded_files):
            status.text(f"Processing: {f.name}")
            log(f"[{i+1}/{len(uploaded_files)}] {f.name}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.getvalue())
                tmp_path = Path(tmp.name)

            try:
                result = pipeline.process_pdf(tmp_path)
                total_uploaded += result.get("uploaded", 0)
                total_skipped += result.get("skipped", 0)
                file_results.append(
                    {
                        "file": f.name,
                        "uploaded": result.get("uploaded", 0),
                        "skipped": result.get("skipped", 0),
                    }
                )
                log(f"  ✓ uploaded={result.get('uploaded', 0)}, skipped={result.get('skipped', 0)}")
            except Exception as e:
                total_failed += 1
                log(f"  ✗ Error: {e}")
            finally:
                os.unlink(tmp_path)

            progress.progress((i + 1) / len(uploaded_files))

        st.session_state["last_upload_result"] = {
            "total_uploaded": total_uploaded,
            "total_skipped": total_skipped,
            "total_failed": total_failed,
            "file_results": file_results,
            "collection": collection_name,
        }

        get_collection_stats.clear()
        sample_points_cached.clear()

        if total_uploaded > 0:
            st.session_state["upload_success"] = f"Uploaded {total_uploaded} pages"
            st.rerun()

    except Exception as e:
        log(f"ERROR: {e}")
        st.error(f"Processing error: {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())


def render_upload_results():
    result = st.session_state.get("last_upload_result", {})
    if not result:
        return

    st.subheader("📊 Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Uploaded", result.get("total_uploaded", 0))
    c2.metric("Skipped", result.get("total_skipped", 0))
    c3.metric("Failed", result.get("total_failed", 0))


def render_playground_tab():
    st.subheader("🎮 Playground")

    active_collection = st.session_state.get("active_collection")
    url, api_key = get_qdrant_credentials()

    if not active_collection:
        collections = get_collections(url, api_key)
        if collections:
            active_collection = collections[0]

    if not active_collection:
        st.warning("No collection available. Upload documents or select a collection.")
        return

    points_for_model = sample_points_cached(active_collection, 1, 0, url, api_key)
    model_name = None
    if points_for_model:
        model_name = points_for_model[0].get("payload", {}).get("model_name")
    if not model_name:
        model_name = AVAILABLE_MODELS[1]

    model_short = model_name.split("/")[-1] if model_name else "unknown"
    cache_key = f"{active_collection}_{model_name}"

    if st.session_state.get("loaded_model_key") != cache_key:
        st.session_state["model_loaded"] = False

    col_info, col_model = st.columns([2, 1])
    with col_info:
        st.info(f"**Collection:** `{active_collection}`")
    with col_model:
        if not st.session_state.get("model_loaded"):
            with st.spinner(f"Loading {model_short}..."):
                try:
                    from visual_rag.retrieval import MultiVectorRetriever

                    _ = MultiVectorRetriever(
                        collection_name=active_collection, model_name=model_name
                    )
                    st.session_state["model_loaded"] = True
                    st.session_state["loaded_model_key"] = cache_key
                    st.session_state["loaded_model_name"] = model_name
                except Exception:
                    st.warning(f"Failed: {model_short}")

        if st.session_state.get("model_loaded"):
            st.markdown(
                f"✅ Found <span style='color:#e74c3c;font-weight:bold;'>{model_short}</span> model",
                unsafe_allow_html=True,
            )

    with st.expander("📦 Sample Points Explorer", expanded=True):
        render_sample_explorer(active_collection, url, api_key)

    st.divider()

    st.subheader("🔍 RAG Query")
    render_rag_query_interface(active_collection, model_name)


def render_document_details(pt: dict, p: dict, score: float = None, rel_pct: float = None):
    doc_id = p.get("doc_id") or p.get("union_doc_id") or p.get("source_doc_id") or "?"
    corpus_id = p.get("corpus-id") or p.get("source_doc_id") or "?"
    dataset = p.get("dataset") or p.get("source") or "N/A"
    model = (p.get("model_name") or p.get("model") or "N/A").split("/")[-1]
    doc_name = p.get("doc-id") or p.get("filename") or "Unknown"

    num_tiles = p.get("num_tiles") or "?"
    visual_tokens = p.get("index_recovery_num_visual_tokens") or p.get("num_visual_tokens") or "?"
    patches_per_tile = p.get("patches_per_tile") or "?"
    torch_dtype = p.get("torch_dtype") or "?"

    orig_w = p.get("original_width") or "?"
    orig_h = p.get("original_height") or "?"
    crop_w = p.get("cropped_width") or "?"
    crop_h = p.get("cropped_height") or "?"
    resize_w = p.get("resized_width") or "?"
    resize_h = p.get("resized_height") or "?"
    crop_pct = p.get("crop_empty_percentage_to_remove") or 0
    crop_enabled = p.get("crop_empty_enabled", False)

    col_meta, col_img = st.columns([1, 2])

    with col_meta:
        st.markdown("##### 📄 Document Info")
        st.markdown(f"**📁 Doc:** {doc_name}")
        st.markdown(f"**🏛️ Dataset:** {dataset}")
        st.markdown(f"**🔑 Doc ID:** `{str(doc_id)[:20]}...`")
        st.markdown(f"**📋 Corpus ID:** {corpus_id}")

        if score is not None:
            st.divider()
            st.markdown("##### 🎯 Relevance")
            if rel_pct is not None:
                st.markdown(f"**Relative:** 🟢 {rel_pct:.1f}%")
                st.progress(rel_pct / 100)
            st.caption(f"Raw score: {score:.4f}")

        st.divider()
        st.markdown("##### 🎨 Visual Metadata")
        st.markdown(f"**🤖 Model:** `{model}`")
        st.markdown(f"**🔲 Tiles:** {num_tiles}")
        st.markdown(f"**🔢 Visual Tokens:** {visual_tokens}")
        st.markdown(f"**📦 Patches/Tile:** {patches_per_tile}")
        st.markdown(f"**⚙️ Dtype:** {torch_dtype}")

        st.divider()
        st.markdown("##### 📐 Dimensions")
        st.markdown(f"**Original:** {orig_w}×{orig_h}")
        st.markdown(f"**Resized:** {resize_w}×{resize_h}")
        if crop_enabled:
            st.markdown(f"**Cropped:** {crop_w}×{crop_h}")
            st.markdown(f"**Crop %:** {int(crop_pct * 100) if crop_pct else 0}%")

    with col_img:
        st.markdown("##### 📷 Document Page")
        tabs = st.tabs(["🖼️ Original", "📷 Resized", "✂️ Cropped"])

        url_o = p.get("original_url")
        url_r = p.get("resized_url") or p.get("page")
        url_c = p.get("cropped_url")

        with tabs[0]:
            if url_o:
                st.image(url_o, width=600)
                st.caption(f"📐 **{orig_w}×{orig_h}**")
            else:
                st.info("No original image available")

        with tabs[1]:
            if url_r:
                st.image(url_r, width=600)
                st.caption(f"📐 **{resize_w}×{resize_h}**")
            else:
                st.info("No resized image available")

        with tabs[2]:
            if url_c:
                st.image(url_c, width=600)
                st.caption(
                    f"📐 **{crop_w}×{crop_h}** | Crop: {int(crop_pct * 100) if crop_pct else 0}%"
                )
            else:
                st.info("No cropped image available")

        with st.expander("🔗 Image URLs"):
            if url_o:
                st.code(url_o, language=None)
            if url_r and url_r != url_o:
                st.code(url_r, language=None)
            if url_c:
                st.code(url_c, language=None)


def render_sample_explorer(collection_name: str, url: str, api_key: str):
    sample_for_filters = sample_points_cached(collection_name, 50, 0, url, api_key)
    datasets = set()
    doc_ids = set()
    for pt in sample_for_filters:
        p = pt.get("payload", {})
        if ds := p.get("dataset"):
            datasets.add(ds)
        if did := (p.get("doc-id") or p.get("filename")):
            doc_ids.add(did)

    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    with c1:
        n_samples = st.slider("Samples", 1, 20, 3, key="pg_n")
    with c2:
        seed = st.number_input("Seed", 0, 9999, 42, key="pg_seed")
    with c3:
        filter_ds = st.selectbox("Dataset", ["All"] + sorted(datasets), key="pg_filter_ds")
    with c4:
        st.write("")
        do_sample = st.button("🎲 Sample", type="primary", key="pg_sample_btn")

    if do_sample:
        points = sample_points_cached(collection_name, n_samples * 5, seed, url, api_key)
        if filter_ds != "All":
            points = [p for p in points if p.get("payload", {}).get("dataset") == filter_ds]
        points = points[:n_samples]
        st.session_state["pg_points"] = points

    points = st.session_state.get("pg_points", [])

    if not points:
        st.caption("Click 'Sample' to load documents")
        return

    st.success(f"**{len(points)} points loaded**")

    for i, pt in enumerate(points):
        p = pt.get("payload", {})

        filename = p.get("filename") or p.get("doc_id") or p.get("source_doc_id") or "Unknown"
        page_num = p.get("page_number") or p.get("page") or "?"

        with st.expander(f"**{i+1}.** {str(filename)[:40]} - Page {page_num}", expanded=(i == 0)):
            render_document_details(pt, p)


def render_rag_query_interface(collection_name: str, model_name: str = None):
    if not collection_name:
        return

    url, api_key = get_qdrant_credentials()

    if not model_name:
        points = sample_points_cached(collection_name, 1, 0, url, api_key)
        if points:
            model_name = points[0].get("payload", {}).get("model_name")
        if not model_name:
            model_name = AVAILABLE_MODELS[1]

    st.caption(f"Model: **{model_name.split('/')[-1] if model_name else 'auto'}**")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c2:
        mode = st.selectbox("Mode", RETRIEVAL_MODES, index=0, key="q_mode")
    with c3:
        top_k = st.slider("Top K", 1, 30, 10, key="q_topk")

    prefetch_k, stage1_mode, stage1_k, stage2_k = 256, "tokens_vs_standard_pooling", 1000, 300

    if mode == "two_stage":
        cc1, cc2 = st.columns(2)
        with cc1:
            stage1_mode = st.selectbox("Stage1", STAGE1_MODES, key="q_s1mode")
        with cc2:
            prefetch_k = st.slider("Prefetch K", 50, 500, 256, key="q_pk")
    elif mode == "three_stage":
        cc1, cc2 = st.columns(2)
        with cc1:
            stage1_k = st.number_input("Stage1 K", 100, 5000, 1000, key="q_s1k")
        with cc2:
            stage2_k = st.number_input("Stage2 K", 50, 1000, 300, key="q_s2k")

    with c1:
        query = st.text_input("Query", placeholder="Enter your search query...", key="q_text")

    if st.button("🔍 Search", type="primary", disabled=not query, key="q_search"):
        with st.spinner("Searching..."):
            results, err = search_collection(
                collection_name,
                query,
                top_k,
                mode,
                prefetch_k,
                stage1_mode,
                stage1_k,
                stage2_k,
                model_name,
            )
            if err:
                st.error("Search failed")
                st.code(err)
            else:
                st.session_state["q_results"] = results

    results = st.session_state.get("q_results", [])
    if results:
        st.success(f"**{len(results)} results**")
        max_score = max(r.get("score_final", r.get("score_stage1", 0)) for r in results) or 1

        for i, r in enumerate(results):
            p = r.get("payload", {})
            score = r.get("score_final", r.get("score_stage1", 0))
            rel = score / max_score * 100

            filename = p.get("filename") or p.get("doc_id") or p.get("source_doc_id") or "Unknown"
            page_num = p.get("page_number") or p.get("page") or "?"

            with st.expander(
                f"**#{i+1}** {str(filename)[:35]} - Page {page_num} | 🎯 {rel:.0f}%",
                expanded=(i < 3),
            ):
                render_document_details(r, p, score=score, rel_pct=rel)


def render_benchmark_tab():
    st.subheader("📊 Benchmarking")

    tab_index, tab_eval, tab_results = st.tabs(["Indexing", "Evaluation", "Results"])

    url, api_key = get_qdrant_credentials()
    collections = get_collections(url, api_key)

    with tab_index:
        render_benchmark_indexing(collections)

    with tab_eval:
        render_benchmark_evaluation(collections)

    with tab_results:
        render_benchmark_results()


def render_benchmark_indexing(collections: List[str]):
    c1, c2, c3 = st.columns(3)
    with c1:
        datasets = st.multiselect(
            "Datasets", BENCHMARK_DATASETS, default=BENCHMARK_DATASETS, key="bi_ds"
        )
    with c2:
        model = st.selectbox("Model", AVAILABLE_MODELS, key="bi_model")
    with c3:
        model_short = model.split("/")[-1].replace("-", "_").replace(".", "_")
        collection = st.text_input(
            "Collection", value=f"vidore_{len(datasets)}ds__{model_short}", key="bi_coll"
        )

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        crop = st.toggle("Crop", value=True, key="bi_crop")
    with c5:
        cloudinary = st.toggle("Cloudinary", value=True, key="bi_cloud")
    with c6:
        grpc = st.toggle("gRPC", value=True, key="bi_grpc")
    with c7:
        recreate = st.toggle("Recreate", value=False, key="bi_recreate")

    crop_pct = st.slider("Crop %", 0.8, 0.99, 0.99, 0.01, key="bi_crop_pct") if crop else 0.99

    config = {
        "datasets": datasets,
        "model": model,
        "collection": collection,
        "crop_empty": crop,
        "crop_percentage": crop_pct,
        "no_cloudinary": not cloudinary,
        "recreate": recreate,
        "resume": False,
        "prefer_grpc": grpc,
        "batch_size": 4,
        "upload_batch_size": 8,
        "qdrant_timeout": 180,
        "qdrant_retries": 5,
        "torch_dtype": "float16",
        "qdrant_vector_dtype": "float16",
    }

    cmd = build_index_command(config)

    col_cmd, col_stats = st.columns([2, 1])
    with col_cmd:
        st.code(cmd, language="bash")
    with col_stats:
        st.metric("Datasets", len(datasets))
        st.metric("Model", model.split("/")[-1])
        run_index = st.button("🚀 Run Index", type="primary", key="bi_run")

    if run_index:
        if not collection:
            st.error("Please select a collection first")
        else:
            run_indexing_with_ui(config)


def render_benchmark_evaluation(collections: List[str]):
    all_docs = sum(DATASET_STATS.get(d, {}).get("docs", 0) for d in BENCHMARK_DATASETS)
    all_queries = sum(DATASET_STATS.get(d, {}).get("queries", 0) for d in BENCHMARK_DATASETS)
    st.markdown(
        f"📊 **Available:** {len(BENCHMARK_DATASETS)} datasets — **{all_docs:,}** docs, **{all_queries:,}** queries"
    )

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        if collections:
            collection = st.selectbox("Collection", collections, key="be_coll")
        else:
            collection = st.text_input("Collection", key="be_coll_txt")
    with c2:
        st.multiselect("Datasets", BENCHMARK_DATASETS, default=BENCHMARK_DATASETS, key="be_ds")
    with c3:
        model = st.selectbox("Model", AVAILABLE_MODELS, key="be_model")

    datasets = st.session_state.get("be_ds", BENCHMARK_DATASETS)
    sel_docs = sum(DATASET_STATS.get(d, {}).get("docs", 0) for d in datasets)
    sel_queries = sum(DATASET_STATS.get(d, {}).get("queries", 0) for d in datasets)
    st.markdown(
        f"🎯 **Selected:** {len(datasets)} dataset(s) — **{sel_docs:,}** docs, **{sel_queries:,}** queries"
    )

    st.markdown("---")

    col_mode, col_topk = st.columns([2, 1])
    with col_mode:
        mode = st.selectbox("Mode", RETRIEVAL_MODES, key="be_mode")
    with col_topk:
        top_k = st.slider("Top K", 10, 100, 100, key="be_topk")

    stage1_mode, prefetch_k, stage1_k, stage2_k = "tokens_vs_standard_pooling", 256, 1000, 300

    if mode == "two_stage":
        cc1, cc2 = st.columns(2)
        with cc1:
            stage1_mode = st.selectbox("Stage1 Mode", STAGE1_MODES, key="be_s1mode")
        with cc2:
            prefetch_k = st.slider("Prefetch K", 50, 1000, 256, key="be_pk")
    elif mode == "three_stage":
        cc1, cc2 = st.columns(2)
        with cc1:
            stage1_k = st.number_input("Stage1 K", 100, 5000, 1000, key="be_s1k")
        with cc2:
            stage2_k = st.number_input("Stage2 K", 50, 1000, 300, key="be_s2k")

    st.markdown("---")

    col_scope, col_grpc, col_spacer = st.columns([2, 1, 1])
    with col_scope:
        scope = st.selectbox("Scope", ["union", "per_dataset"], key="be_scope")
    with col_grpc:
        grpc = st.toggle("gRPC", value=True, key="be_grpc")

    result_prefix_val = st.session_state.get("be_prefix", "")

    config = {
        "datasets": datasets,
        "model": model,
        "collection": collection,
        "mode": mode,
        "top_k": top_k,
        "evaluation_scope": scope,
        "prefer_grpc": grpc,
        "torch_dtype": "float16",
        "qdrant_vector_dtype": "float16",
        "qdrant_timeout": 180,
        "stage1_mode": stage1_mode,
        "prefetch_k": prefetch_k,
        "stage1_k": stage1_k,
        "stage2_k": stage2_k,
        "result_prefix": result_prefix_val,
    }

    cmd = build_eval_command(config)

    python_code = generate_python_eval_code(config)

    col_cmd, col_info = st.columns([2, 1])
    with col_cmd:
        code_tab1, code_tab2 = st.tabs(["🐚 Bash", "🐍 Python"])
        with code_tab1:
            st.code(cmd, language="bash")
        with code_tab2:
            st.code(python_code, language="python")
    with col_info:
        mode_desc = {
            "single_full": "🔹 **Single Full**: Query all visual tokens against full document embeddings in one pass.",
            "single_tiles": "🔸 **Single Tiles**: Query against tile-level embeddings only.",
            "single_global": "🔶 **Single Global**: Query against global (pooled) document embeddings.",
            "two_stage": "🔷 **Two Stage**: Fast prefetch with global/tiles, then rerank with full tokens.",
            "three_stage": "🔶 **Three Stage**: Global → Tiles → Full tokens for maximum precision.",
        }
        scope_desc = {
            "union": "📊 **Union**: Evaluate across all datasets combined as one corpus.",
            "per_dataset": "📁 **Per Dataset**: Evaluate each dataset separately and report individual metrics.",
        }
        st.markdown(mode_desc.get(mode, ""))
        st.markdown(scope_desc.get(scope, ""))
        st.divider()
        st.text_input("Result Prefix", placeholder="optional prefix for output", key="be_prefix")

        run_eval = st.button("🚀 Run Eval", type="primary", key="be_run", use_container_width=True)

    if run_eval:
        if not collection:
            st.error("Please select a collection first")
        else:
            run_evaluation_with_ui(config)


def run_evaluation_with_ui(config: Dict[str, Any]):
    st.divider()

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    output_area = st.empty()

    status_text.info("🚀 Starting evaluation...")
    output_lines = []

    def log(msg):
        output_lines.append(msg)
        output_area.code("\n".join(output_lines[-50:]), language="text")

    try:
        url, api_key = get_qdrant_credentials()
        if not url:
            st.error("QDRANT_URL not configured")
            return

        datasets = config.get("datasets", [])
        collection = config["collection"]
        model = config.get("model", "vidore/colpali-v1.3")
        mode = config.get("mode", "single_full")
        top_k = config.get("top_k", 100)
        prefetch_k = config.get("prefetch_k", 256)
        stage1_mode = config.get("stage1_mode", "tokens_vs_standard_pooling")
        stage1_k = config.get("stage1_k", 1000)
        stage2_k = config.get("stage2_k", 300)
        _evaluation_scope = config.get("evaluation_scope", "union")
        prefer_grpc = config.get("prefer_grpc", True)
        torch_dtype = config.get("torch_dtype", "float16")

        log(f"[Eval] Model: {model}")
        log(f"[Eval] Collection: {collection}")
        log(f"[Eval] Mode: {mode}")
        log(f"[Eval] Datasets: {datasets}")
        status_text.info("📦 Loading embedder...")

        embedder = VisualEmbedder(model_name=model, torch_dtype=torch_dtype)
        log("[Eval] Embedder loaded")

        status_text.info("🔌 Connecting to Qdrant...")
        retriever = MultiVectorRetriever(
            collection_name=collection,
            model_name=model,
            qdrant_url=url,
            qdrant_api_key=api_key,
            prefer_grpc=prefer_grpc,
            embedder=embedder,
        )
        log("[Eval] Retriever connected")

        all_queries = []
        all_qrels: Dict[str, Dict[str, int]] = {}

        for ds_name in datasets:
            status_text.info(f"📚 Loading dataset: {ds_name}")
            corpus, queries, qrels = load_vidore_beir_dataset(ds_name)
            all_queries.extend(queries)
            for qid, rels in qrels.items():
                all_qrels[qid] = rels
            log(f"[Eval] Loaded {ds_name}: {len(corpus)} docs, {len(queries)} queries")

        total_queries = len(all_queries)
        log(f"[Eval] Total queries to evaluate: {total_queries}")

        status_text.info(f"🔍 Embedding {total_queries} queries...")
        query_texts = [q.text for q in all_queries]
        query_embeddings = embedder.embed_queries(query_texts, show_progress=False)
        log("[Eval] Queries embedded")

        ndcg10_vals = []
        recall10_vals = []
        mrr10_vals = []
        latencies = []

        status_text.info("🎯 Running evaluation...")

        for i, (q, qemb) in enumerate(zip(all_queries, query_embeddings)):
            start = time.time()

            try:
                import torch

                if isinstance(qemb, torch.Tensor):
                    qemb_np = qemb.detach().cpu().numpy()
                else:
                    qemb_np = qemb.numpy()
            except ImportError:
                qemb_np = qemb.numpy()

            results = retriever.search_embedded(
                query_embedding=qemb_np,
                top_k=max(100, top_k),
                mode=mode,
                prefetch_k=prefetch_k,
                stage1_mode=stage1_mode,
                stage1_k=stage1_k,
                stage2_k=stage2_k,
            )
            latencies.append((time.time() - start) * 1000)

            ranking = [str(r["id"]) for r in results]
            rels = all_qrels.get(q.query_id, {})

            ndcg10_vals.append(ndcg_at_k(ranking, rels, k=10))
            recall10_vals.append(recall_at_k(ranking, rels, k=10))
            mrr10_vals.append(mrr_at_k(ranking, rels, k=10))

            progress = (i + 1) / total_queries
            progress_bar.progress(progress)
            status_text.info(f"🎯 Evaluating... {i+1}/{total_queries} ({int(progress*100)}%)")

            if (i + 1) % 20 == 0:
                log(f"[Eval] Progress: {i+1}/{total_queries} queries")

        progress_bar.progress(1.0)
        status_text.success("✅ Evaluation complete!")

        final_metrics = {
            "ndcg@10": float(np.mean(ndcg10_vals)),
            "recall@10": float(np.mean(recall10_vals)),
            "mrr@10": float(np.mean(mrr10_vals)),
            "avg_latency_ms": float(np.mean(latencies)),
            "num_queries": total_queries,
        }

        log("")
        log("=" * 40)
        log("RESULTS:")
        log(f"  NDCG@10:  {final_metrics['ndcg@10']:.4f}")
        log(f"  Recall@10: {final_metrics['recall@10']:.4f}")
        log(f"  MRR@10:   {final_metrics['mrr@10']:.4f}")
        log(f"  Avg Latency: {final_metrics['avg_latency_ms']:.1f}ms")
        log("=" * 40)

        st.json(final_metrics)
        st.session_state["last_eval_metrics"] = final_metrics

    except Exception as e:
        status_text.error(f"❌ Error: {e}")
        log(f"ERROR: {e}")
        log(traceback.format_exc())
    finally:
        st.session_state["bench_running"] = False


def run_indexing_with_ui(config: Dict[str, Any]):
    st.divider()

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    output_area = st.empty()

    status_text.info("🚀 Starting indexing...")
    output_lines = []

    def log(msg):
        output_lines.append(msg)
        output_area.code("\n".join(output_lines[-50:]), language="text")

    try:
        url, api_key = get_qdrant_credentials()
        if not url:
            st.error("QDRANT_URL not configured")
            return

        datasets = config.get("datasets", [])
        collection = config["collection"]
        model = config.get("model", "vidore/colpali-v1.3")
        recreate = config.get("recreate", False)
        torch_dtype = config.get("torch_dtype", "float16")
        qdrant_vector_dtype = config.get("qdrant_vector_dtype", "float16")
        prefer_grpc = config.get("prefer_grpc", True)
        batch_size = config.get("batch_size", 4)

        log(f"[Index] Model: {model}")
        log(f"[Index] Collection: {collection}")
        log(f"[Index] Datasets: {datasets}")
        status_text.info("📦 Loading embedder...")

        embedder = VisualEmbedder(model_name=model, torch_dtype=torch_dtype)
        log("[Index] Embedder loaded")

        status_text.info("🔌 Connecting to Qdrant...")
        indexer = QdrantIndexer(
            url=url,
            api_key=api_key,
            collection_name=collection,
            prefer_grpc=prefer_grpc,
            vector_datatype=qdrant_vector_dtype,
        )
        log("[Index] Connected to Qdrant")

        status_text.info("📦 Creating collection...")
        indexer.create_collection(force_recreate=recreate)
        indexer.create_payload_indexes(
            fields=[
                {"field": "dataset", "type": "keyword"},
                {"field": "doc_id", "type": "keyword"},
            ]
        )
        log(f"[Index] Collection '{collection}' ready")

        total_uploaded = 0

        for ds_name in datasets:
            status_text.info(f"📚 Loading dataset: {ds_name}")
            corpus, queries, qrels = load_vidore_beir_dataset(ds_name)
            log(f"[Index] Loaded {ds_name}: {len(corpus)} documents")

            for i in range(0, len(corpus), batch_size):
                batch = corpus[i : i + batch_size]
                images = [doc.image for doc in batch if hasattr(doc, "image") and doc.image]

                if not images:
                    continue

                status_text.info(f"🎨 Embedding batch {i//batch_size + 1}...")
                embeddings = embedder.embed_images(images)

                points = []
                for j, (doc, emb) in enumerate(zip(batch, embeddings)):
                    doc_id = doc.doc_id if hasattr(doc, "doc_id") else str(i + j)
                    emb_np = emb.cpu().numpy() if hasattr(emb, "cpu") else np.array(emb)
                    tile_pooled = emb_np.reshape(-1, 4, emb_np.shape[-1]).mean(axis=1)
                    global_pooled = emb_np.mean(axis=0)

                    points.append(
                        {
                            "id": f"{ds_name}_{doc_id}".replace("/", "_"),
                            "visual_embedding": emb_np,
                            "tile_pooled_embedding": tile_pooled,
                            "experimental_pooled_embedding": tile_pooled,
                            "global_pooled_embedding": global_pooled,
                            "metadata": {"dataset": ds_name, "doc_id": doc_id},
                        }
                    )

                indexer.upload_batch(points)
                total_uploaded += len(points)

                progress = (i + len(batch)) / len(corpus)
                progress_bar.progress(progress)
                log(f"[Index] Uploaded {total_uploaded} points")

        progress_bar.progress(1.0)
        status_text.success(f"✅ Indexing complete! {total_uploaded} documents indexed.")

    except Exception as e:
        status_text.error(f"❌ Error: {e}")
        log(f"ERROR: {e}")
        log(traceback.format_exc())


def render_benchmark_results():
    st.markdown("##### Load Results")

    available = get_available_results()

    if not available:
        st.info("No results found")
        return

    default_select = []
    if st.session_state.get("auto_select_result"):
        auto = st.session_state.pop("auto_select_result")
        if auto in [str(p) for p in available]:
            default_select = [auto]

    selected = st.multiselect(
        "Result files",
        options=[str(p) for p in available],
        format_func=lambda x: Path(x).name[:60],
        default=default_select,
        key="br_files",
    )

    for path in selected:
        data = load_results_file(Path(path))
        if data:
            render_result_card(data, Path(path).name)


def render_result_card(data: Dict[str, Any], filename: str):
    with st.expander(f"📊 {filename[:50]}", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", (data.get("model") or "?").split("/")[-1])
        c2.metric("Mode", data.get("mode", "?"))
        c3.metric("Top K", data.get("top_k", "?"))
        c4.metric("Time", f"{data.get('eval_wall_time_s', 0):.0f}s")

        metrics = data.get("metrics_by_dataset", {})
        if not metrics:
            st.warning("No metrics data")
            return

        rows = []
        for ds, m in metrics.items():
            rows.append(
                {
                    "Dataset": ds.split("/")[-1].replace("_v2", ""),
                    "NDCG@5": m.get("ndcg@5", 0),
                    "NDCG@10": m.get("ndcg@10", 0),
                    "Recall@5": m.get("recall@5", 0),
                    "Recall@10": m.get("recall@10", 0),
                    "MRR@10": m.get("mrr@10", 0),
                    "Latency": m.get("avg_latency_ms", 0),
                    "QPS": m.get("qps", 0),
                }
            )

        df = pd.DataFrame(rows)

        st.dataframe(
            df.style.format(
                {
                    "NDCG@5": "{:.4f}",
                    "NDCG@10": "{:.4f}",
                    "Recall@5": "{:.4f}",
                    "Recall@10": "{:.4f}",
                    "MRR@10": "{:.4f}",
                    "Latency": "{:.1f}",
                    "QPS": "{:.2f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

        chart_data = []
        for ds, m in metrics.items():
            ds_short = ds.split("/")[-1].replace("_v2", "").replace("_", " ").title()
            chart_data.append(
                {"Dataset": ds_short, "Metric": "NDCG@10", "Value": m.get("ndcg@10", 0)}
            )
            chart_data.append(
                {"Dataset": ds_short, "Metric": "Recall@10", "Value": m.get("recall@10", 0)}
            )
            chart_data.append(
                {"Dataset": ds_short, "Metric": "MRR@10", "Value": m.get("mrr@10", 0)}
            )

        chart_df = pd.DataFrame(chart_data)

        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("Dataset:N", title=None),
                y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1]), title="Score"),
                color=alt.Color("Metric:N", scale=alt.Scale(scheme="tableau10")),
                xOffset="Metric:N",
                tooltip=["Dataset", "Metric", alt.Tooltip("Value:Q", format=".4f")],
            )
            .properties(height=300, title="Metrics by Dataset")
        )

        st.altair_chart(chart, use_container_width=True)

        latency_data = [
            {
                "Dataset": ds.split("/")[-1].replace("_v2", ""),
                "Latency (ms)": m.get("avg_latency_ms", 0),
                "QPS": m.get("qps", 0),
            }
            for ds, m in metrics.items()
        ]
        latency_df = pd.DataFrame(latency_data)

        c1, c2 = st.columns(2)
        with c1:
            lat_chart = (
                alt.Chart(latency_df)
                .mark_bar(color="#ff6b6b")
                .encode(
                    x=alt.X("Dataset:N"),
                    y=alt.Y("Latency (ms):Q"),
                    tooltip=["Dataset", alt.Tooltip("Latency (ms):Q", format=".1f")],
                )
                .properties(height=200, title="Avg Latency")
            )
            st.altair_chart(lat_chart, use_container_width=True)

        with c2:
            qps_chart = (
                alt.Chart(latency_df)
                .mark_bar(color="#4ecdc4")
                .encode(
                    x=alt.X("Dataset:N"),
                    y=alt.Y("QPS:Q"),
                    tooltip=["Dataset", alt.Tooltip("QPS:Q", format=".2f")],
                )
                .properties(height=200, title="QPS (Queries/sec)")
            )
            st.altair_chart(qps_chart, use_container_width=True)


def main():
    render_header()
    render_sidebar()

    tab_upload, tab_playground, tab_benchmark = st.tabs(
        ["📤 Upload", "🎮 Playground", "📊 Benchmarking"]
    )

    with tab_upload:
        render_upload_tab()

    with tab_playground:
        render_playground_tab()

    with tab_benchmark:
        render_benchmark_tab()


if __name__ == "__main__":
    main()
