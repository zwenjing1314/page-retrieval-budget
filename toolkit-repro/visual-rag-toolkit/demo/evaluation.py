"""Evaluation runner with UI updates."""

import hashlib
import json
import logging
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st
import torch
from qdrant_client.models import FieldCondition, Filter, MatchValue

from benchmarks.vidore_tatdqa_test.dataset_loader import load_vidore_beir_dataset
from benchmarks.vidore_tatdqa_test.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from demo.qdrant_utils import get_qdrant_credentials
from visual_rag import VisualEmbedder
from visual_rag.retrieval import MultiVectorRetriever

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _stable_uuid(text: str) -> str:
    """Generate a stable UUID from text (same as benchmark script)."""
    hex_str = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"


def _union_point_id(
    *, dataset_name: str, source_doc_id: str, union_namespace: Optional[str]
) -> str:
    """Generate union point ID (same as benchmark script)."""
    ns = f"{union_namespace}::{dataset_name}" if union_namespace else dataset_name
    return _stable_uuid(f"{ns}::{source_doc_id}")


def _remap_qrels_to_union_ids(
    qrels: Dict[str, Dict[str, int]],
    corpus: List[Any],
    dataset_name: str,
    collection_name: str,
) -> Dict[str, Dict[str, int]]:
    """Remap qrels doc_ids from original format to union_doc_id format (matching benchmark)."""
    id_map: Dict[str, str] = {}
    for doc in corpus:
        source_doc_id = str((doc.payload or {}).get("source_doc_id") or doc.doc_id)
        id_map[str(doc.doc_id)] = _union_point_id(
            dataset_name=dataset_name,
            source_doc_id=source_doc_id,
            union_namespace=collection_name,
        )

    remapped: Dict[str, Dict[str, int]] = {}
    for qid, rels in qrels.items():
        out_rels: Dict[str, int] = {}
        for did, score in rels.items():
            mapped = id_map.get(str(did))
            if mapped:
                out_rels[mapped] = int(score)
        if out_rels:
            remapped[qid] = out_rels
    return remapped


def get_doc_id_from_result(r: Dict[str, Any], use_original: bool = True) -> str:
    """Extract document ID from search result.

    Args:
        r: Search result dict with 'id' and 'payload'
        use_original: If True, prefer original doc_id for matching with qrels.
                     If False, prefer union_doc_id (Qdrant point ID).
    """
    payload = r.get("payload", {})
    if use_original:
        doc_id = (
            payload.get("doc_id")
            or payload.get("source_doc_id")
            or payload.get("corpus-id")
            or payload.get("union_doc_id")
            or str(r.get("id", ""))
        )
    else:
        doc_id = payload.get("union_doc_id") or str(r.get("id", "")) or payload.get("doc_id")
    return str(doc_id)


def run_evaluation_with_ui(config: Dict[str, Any]):
    st.divider()

    print("=" * 60)
    print("[EVAL] Starting evaluation via UI")
    print("=" * 60)

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
    prefer_grpc = config.get("prefer_grpc", True)
    torch_dtype = config.get("torch_dtype", "float16")
    evaluation_scope = config.get("evaluation_scope", "union")

    print("[EVAL] ═══════════════════════════════════════════════════")
    print(f"[EVAL] Collection: {collection}")
    print(f"[EVAL] Model: {model}")
    print(f"[EVAL] Mode: {mode}, Scope: {evaluation_scope}")
    print(f"[EVAL] Datasets: {datasets}")
    print(f"[EVAL] Query embedding dtype: {torch_dtype} (vectors already indexed)")
    print("[EVAL] ═══════════════════════════════════════════════════")

    phase1_container = st.container()
    phase2_container = st.container()
    phase3_container = st.container()
    results_container = st.container()

    try:
        with phase1_container:
            st.markdown("##### 🤖 Phase 1: Loading Model")
            model_status = st.empty()
            model_status.info(f"Loading `{model.split('/')[-1]}`...")

            print(f"[EVAL] Loading embedder: {model}")
            torch_dtype_obj = TORCH_DTYPE_MAP.get(torch_dtype, torch.float16)
            qdrant_dtype = config.get("qdrant_vector_dtype", "float16")
            output_dtype_obj = np.float16 if qdrant_dtype == "float16" else np.float32
            embedder = VisualEmbedder(
                model_name=model,
                torch_dtype=torch_dtype_obj,
                output_dtype=output_dtype_obj,
            )
            embedder._load_model()
            print(
                f"[EVAL] Embedder loaded (torch_dtype={torch_dtype}, output_dtype={qdrant_dtype})"
            )

            model_status.success(f"✅ Model `{model.split('/')[-1]}` loaded")

            retriever_status = st.empty()
            retriever_status.info(f"Connecting to collection `{collection}`...")

            print(f"[EVAL] Connecting to Qdrant collection: {collection}")
            retriever = MultiVectorRetriever(
                collection_name=collection,
                model_name=model,
                qdrant_url=url,
                qdrant_api_key=api_key,
                prefer_grpc=prefer_grpc,
                embedder=embedder,
            )
            print("[EVAL] Connected to Qdrant")
            retriever_status.success(f"✅ Connected to `{collection}`")

        with phase2_container:
            st.markdown("##### 📚 Phase 2: Loading Datasets")

            dataset_data = {}
            total_queries = 0
            max_queries_per_ds = config.get("max_queries")

            for ds_name in datasets:
                ds_status = st.empty()
                ds_short = ds_name.split("/")[-1]
                ds_status.info(f"Loading `{ds_short}`...")

                print(f"[EVAL] Loading dataset: {ds_name}")
                corpus, queries, qrels = load_vidore_beir_dataset(ds_name)

                print(f"[EVAL] Remapping qrels to union_doc_id format for collection={collection}")
                remapped_qrels = _remap_qrels_to_union_ids(qrels, corpus, ds_name, collection)
                print(
                    f"[EVAL] Remapped {len(qrels)} -> {len(remapped_qrels)} queries with valid rels"
                )

                if evaluation_scope == "per_dataset" and max_queries_per_ds:
                    queries = queries[:max_queries_per_ds]

                dataset_data[ds_name] = {
                    "queries": queries,
                    "qrels": remapped_qrels,
                    "num_docs": len(corpus),
                }
                total_queries += len(queries)
                print(f"[EVAL] Loaded {ds_name}: {len(corpus)} docs, {len(queries)} queries")
                ds_status.success(f"✅ `{ds_short}`: {len(corpus)} docs, {len(queries)} queries")

            if (
                evaluation_scope == "union"
                and max_queries_per_ds
                and max_queries_per_ds < total_queries
            ):
                total_queries = max_queries_per_ds
                print(f"[EVAL] Will limit to {total_queries} total queries (union mode)")

            embed_status = st.empty()
            embed_status.info("Embedding queries...")

        with phase3_container:
            st.markdown("##### 🎯 Phase 3: Running Evaluation")

            metrics_collectors = {
                "ndcg@5": [],
                "ndcg@10": [],
                "recall@5": [],
                "recall@10": [],
                "mrr@5": [],
                "mrr@10": [],
            }
            latencies = []
            log_lines = []
            metrics_by_dataset = {}

            if evaluation_scope == "per_dataset":
                overall_progress = st.progress(0.0)
                datasets_done = 0

                for ds_name, ds_info in dataset_data.items():
                    ds_short = ds_name.split("/")[-1]
                    st.markdown(f"**Evaluating `{ds_short}`**")

                    queries = ds_info["queries"]
                    qrels = ds_info["qrels"]

                    if not queries:
                        continue

                    print(f"[EVAL] Embedding {len(queries)} queries for {ds_short}...")
                    query_texts = [q.text for q in queries]
                    query_embeddings = embedder.embed_queries(query_texts, show_progress=False)
                    print(f"[EVAL] Queries embedded for {ds_short}")

                    ds_filter = Filter(
                        must=[FieldCondition(key="dataset", match=MatchValue(value=ds_name))]
                    )
                    print(f"[EVAL] Using filter: dataset={ds_name}")

                    progress_bar = st.progress(0.0)
                    eval_status = st.empty()
                    log_area = st.empty()

                    ds_metrics = {
                        "ndcg@5": [],
                        "ndcg@10": [],
                        "recall@5": [],
                        "recall@10": [],
                        "mrr@5": [],
                        "mrr@10": [],
                    }
                    ds_latencies = []
                    ds_log_lines = []

                    eval_status.info(f"Evaluating {len(queries)} queries...")
                    print(
                        f"[EVAL] Starting per-dataset evaluation: {ds_short}, {len(queries)} queries"
                    )

                    for i, (q, qemb) in enumerate(zip(queries, query_embeddings)):
                        start = time.time()

                        if isinstance(qemb, torch.Tensor):
                            qemb_np = qemb.detach().cpu().numpy()
                        else:
                            qemb_np = qemb.numpy() if hasattr(qemb, "numpy") else np.array(qemb)

                        results = retriever.search_embedded(
                            query_embedding=qemb_np,
                            top_k=max(100, top_k),
                            mode=mode,
                            prefetch_k=prefetch_k,
                            stage1_mode=stage1_mode,
                            stage1_k=stage1_k,
                            stage2_k=stage2_k,
                            filter_obj=ds_filter,
                        )
                        ds_latencies.append((time.time() - start) * 1000)
                        latencies.append(ds_latencies[-1])

                        ranking = [str(r["id"]) for r in results]
                        rels = qrels.get(q.query_id, {})

                        if i == 0:
                            print(f"[EVAL] First query for {ds_short} - query_id: {q.query_id}")
                            print(f"[EVAL] Top 3 retrieved doc_ids: {ranking[:3]}")
                            print(f"[EVAL] Expected doc_ids (qrels): {list(rels.keys())[:3]}")
                            print(
                                f"[EVAL] qrels has {len(qrels)} queries, this query in qrels: {q.query_id in qrels}"
                            )
                            if results:
                                r0 = results[0]
                                print(
                                    f"[EVAL] Sample result payload keys: {list(r0.get('payload', {}).keys())}"
                                )
                                p = r0.get("payload", {})
                                print(
                                    f"[EVAL] Sample payload doc_id={p.get('doc_id')}, union_doc_id={p.get('union_doc_id')}, source_doc_id={p.get('source_doc_id')}"
                                )
                            has_match = any(rid in rels for rid in ranking[:10])
                            print(f"[EVAL] Any match in top-10? {has_match}")

                        for k_name, k_val in [("ndcg@5", 5), ("ndcg@10", 10)]:
                            ds_metrics[k_name].append(ndcg_at_k(ranking, rels, k=k_val))
                        for k_name, k_val in [("recall@5", 5), ("recall@10", 10)]:
                            ds_metrics[k_name].append(recall_at_k(ranking, rels, k=k_val))
                        for k_name, k_val in [("mrr@5", 5), ("mrr@10", 10)]:
                            ds_metrics[k_name].append(mrr_at_k(ranking, rels, k=k_val))

                        progress = (i + 1) / len(queries)
                        progress_bar.progress(progress)
                        eval_status.info(
                            f"🎯 {i+1}/{len(queries)} ({int(progress*100)}%) — latency: {np.mean(ds_latencies):.0f}ms"
                        )

                        log_interval = max(5, len(queries) // 10)
                        if (i + 1) % log_interval == 0 and i > 0:
                            cur_ndcg = np.mean(ds_metrics["ndcg@10"])
                            cur_lat = (
                                np.mean(ds_latencies[1:])
                                if len(ds_latencies) > 1
                                else ds_latencies[0]
                            )
                            ds_log_lines.append(
                                f"[{i+1}/{len(queries)}] NDCG@10={cur_ndcg:.4f}, lat={cur_lat:.0f}ms"
                            )
                            log_area.code("\n".join(ds_log_lines[-5:]), language="text")
                            print(
                                f"[EVAL] {ds_short} {i+1}/{len(queries)}: NDCG@10={cur_ndcg:.4f}, lat={cur_lat:.0f}ms"
                            )

                    progress_bar.progress(1.0)
                    ds_final = {k: float(np.mean(v)) for k, v in ds_metrics.items()}
                    ds_final["avg_latency_ms"] = float(np.mean(ds_latencies))
                    ds_final["num_queries"] = len(queries)
                    metrics_by_dataset[ds_name] = ds_final

                    for k, v in ds_metrics.items():
                        metrics_collectors[k].extend(v)

                    eval_status.success(
                        f"✅ `{ds_short}`: NDCG@10={ds_final['ndcg@10']:.4f}, latency={ds_final['avg_latency_ms']:.0f}ms"
                    )
                    print(f"[EVAL] {ds_short} DONE: NDCG@10={ds_final['ndcg@10']:.4f}")

                    datasets_done += 1
                    overall_progress.progress(datasets_done / len(datasets))

                overall_progress.progress(1.0)
                embed_status.success("✅ All queries embedded")
                total_queries = sum(d["num_queries"] for d in metrics_by_dataset.values())

            else:
                all_queries = []
                all_qrels = {}
                for ds_name, ds_info in dataset_data.items():
                    all_queries.extend(ds_info["queries"])
                    for qid, rels in ds_info["qrels"].items():
                        all_qrels[qid] = rels

                sample_qrel_keys = list(all_qrels.keys())[:3]
                sample_doc_ids = []
                for qid in sample_qrel_keys:
                    sample_doc_ids.extend(list(all_qrels[qid].keys())[:2])
                print(f"[EVAL] all_qrels built: {len(all_qrels)} queries")
                print(f"[EVAL] Sample qrel query_ids: {sample_qrel_keys}")
                print(f"[EVAL] Sample qrel doc_ids: {sample_doc_ids[:5]}")

                max_q = config.get("max_queries")
                if max_q and max_q < len(all_queries):
                    all_queries = all_queries[:max_q]
                total_queries = len(all_queries)

                print(f"[EVAL] Embedding {total_queries} queries (union mode)...")
                query_texts = [q.text for q in all_queries]
                query_embeddings = embedder.embed_queries(query_texts, show_progress=False)
                print("[EVAL] Queries embedded")
                embed_status.success(f"✅ {total_queries} queries embedded")

                progress_bar = st.progress(0.0)
                eval_status = st.empty()
                log_area = st.empty()

                eval_status.info(f"Evaluating {total_queries} queries in `{mode}` mode...")
                print(f"[EVAL] Starting union evaluation: {total_queries} queries, mode={mode}")

                for i, (q, qemb) in enumerate(zip(all_queries, query_embeddings)):
                    start = time.time()

                    if isinstance(qemb, torch.Tensor):
                        qemb_np = qemb.detach().cpu().numpy()
                    else:
                        qemb_np = qemb.numpy() if hasattr(qemb, "numpy") else np.array(qemb)

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

                    if i == 0:
                        print(f"[EVAL] First query - query_id: {q.query_id}")
                        print(f"[EVAL] Top 3 retrieved doc_ids: {ranking[:3]}")
                        print(f"[EVAL] Expected doc_ids (qrels): {list(rels.keys())[:3]}")
                        print(
                            f"[EVAL] all_qrels has {len(all_qrels)} queries, this query in qrels: {q.query_id in all_qrels}"
                        )
                        if results:
                            r0 = results[0]
                            print(
                                f"[EVAL] Sample result payload keys: {list(r0.get('payload', {}).keys())}"
                            )
                            p = r0.get("payload", {})
                            print(
                                f"[EVAL] Sample payload doc_id={p.get('doc_id')}, union_doc_id={p.get('union_doc_id')}, source_doc_id={p.get('source_doc_id')}"
                            )
                        has_match = any(rid in rels for rid in ranking[:10])
                        print(f"[EVAL] Any match in top-10? {has_match}")

                    metrics_collectors["ndcg@5"].append(ndcg_at_k(ranking, rels, k=5))
                    metrics_collectors["ndcg@10"].append(ndcg_at_k(ranking, rels, k=10))
                    metrics_collectors["recall@5"].append(recall_at_k(ranking, rels, k=5))
                    metrics_collectors["recall@10"].append(recall_at_k(ranking, rels, k=10))
                    metrics_collectors["mrr@5"].append(mrr_at_k(ranking, rels, k=5))
                    metrics_collectors["mrr@10"].append(mrr_at_k(ranking, rels, k=10))

                    progress = (i + 1) / total_queries
                    progress_bar.progress(progress)
                    eval_status.info(
                        f"🎯 {i+1}/{total_queries} ({int(progress*100)}%) — latency: {np.mean(latencies):.0f}ms"
                    )

                    log_interval = max(10, total_queries // 10)
                    if (i + 1) % log_interval == 0 and i > 0:
                        cur_ndcg = np.mean(metrics_collectors["ndcg@10"])
                        cur_lat = np.mean(latencies[1:]) if len(latencies) > 1 else latencies[0]
                        log_lines.append(
                            f"[{i+1}/{total_queries}] NDCG@10={cur_ndcg:.4f}, lat={cur_lat:.0f}ms"
                        )
                        log_area.code("\n".join(log_lines[-10:]), language="text")
                        print(
                            f"[EVAL] Progress {i+1}/{total_queries}: NDCG@10={cur_ndcg:.4f}, lat={cur_lat:.0f}ms"
                        )

                progress_bar.progress(1.0)
                eval_status.success(f"✅ Evaluation complete! ({total_queries} queries)")

        with results_container:
            st.markdown("##### 📊 Results")

            p95_latency = float(np.percentile(latencies, 95))
            eval_time_s = sum(latencies) / 1000
            qps = total_queries / eval_time_s if eval_time_s > 0 else 0

            final_metrics = {
                "ndcg@5": float(np.mean(metrics_collectors["ndcg@5"])),
                "ndcg@10": float(np.mean(metrics_collectors["ndcg@10"])),
                "recall@5": float(np.mean(metrics_collectors["recall@5"])),
                "recall@10": float(np.mean(metrics_collectors["recall@10"])),
                "mrr@5": float(np.mean(metrics_collectors["mrr@5"])),
                "mrr@10": float(np.mean(metrics_collectors["mrr@10"])),
                "avg_latency_ms": float(np.mean(latencies)),
                "p95_latency_ms": p95_latency,
                "qps": qps,
                "eval_time_s": eval_time_s,
                "num_queries": total_queries,
            }

            print("=" * 60)
            print("[EVAL] FINAL RESULTS:")
            print(f"[EVAL]   NDCG@5:     {final_metrics['ndcg@5']:.4f}")
            print(f"[EVAL]   NDCG@10:    {final_metrics['ndcg@10']:.4f}")
            print(f"[EVAL]   Recall@5:   {final_metrics['recall@5']:.4f}")
            print(f"[EVAL]   Recall@10:  {final_metrics['recall@10']:.4f}")
            print(f"[EVAL]   MRR@5:      {final_metrics['mrr@5']:.4f}")
            print(f"[EVAL]   MRR@10:     {final_metrics['mrr@10']:.4f}")
            print(f"[EVAL]   Avg Latency: {final_metrics['avg_latency_ms']:.1f}ms")
            print(f"[EVAL]   P95 Latency: {final_metrics['p95_latency_ms']:.1f}ms")
            print(f"[EVAL]   QPS:        {final_metrics['qps']:.2f}")
            print(f"[EVAL]   Queries:    {final_metrics['num_queries']}")
            print("=" * 60)

            st.markdown("**Retrieval Metrics**")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("NDCG@5", f"{final_metrics['ndcg@5']:.4f}")
                st.metric("NDCG@10", f"{final_metrics['ndcg@10']:.4f}")
            with c2:
                st.metric("Recall@5", f"{final_metrics['recall@5']:.4f}")
                st.metric("Recall@10", f"{final_metrics['recall@10']:.4f}")
            with c3:
                st.metric("MRR@5", f"{final_metrics['mrr@5']:.4f}")
                st.metric("MRR@10", f"{final_metrics['mrr@10']:.4f}")

            st.markdown("**Performance**")
            c4, c5, c6, c7 = st.columns(4)
            c4.metric("Avg Latency", f"{final_metrics['avg_latency_ms']:.0f}ms")
            c5.metric("P95 Latency", f"{final_metrics['p95_latency_ms']:.0f}ms")
            c6.metric("QPS", f"{final_metrics['qps']:.2f}")
            c7.metric("Eval Time", f"{final_metrics['eval_time_s']:.1f}s")

            with st.expander("📋 Full Results JSON"):
                st.json(final_metrics)

            detailed_report = {
                "generated_at": datetime.now().isoformat(),
                "config": {
                    "collection": collection,
                    "model": model,
                    "datasets": datasets,
                    "mode": mode,
                    "top_k": top_k,
                    "evaluation_scope": config.get("evaluation_scope", "union"),
                    "prefer_grpc": prefer_grpc,
                    "torch_dtype": torch_dtype,
                    "max_queries": config.get("max_queries"),
                },
                "retrieval_metrics": {
                    "ndcg@5": final_metrics["ndcg@5"],
                    "ndcg@10": final_metrics["ndcg@10"],
                    "recall@5": final_metrics["recall@5"],
                    "recall@10": final_metrics["recall@10"],
                    "mrr@5": final_metrics["mrr@5"],
                    "mrr@10": final_metrics["mrr@10"],
                },
                "performance": {
                    "avg_latency_ms": final_metrics["avg_latency_ms"],
                    "p95_latency_ms": final_metrics["p95_latency_ms"],
                    "qps": final_metrics["qps"],
                    "eval_time_s": final_metrics["eval_time_s"],
                    "num_queries": final_metrics["num_queries"],
                },
            }

            if mode == "two_stage":
                detailed_report["config"]["stage1_mode"] = stage1_mode
                detailed_report["config"]["prefetch_k"] = prefetch_k
            elif mode == "three_stage":
                detailed_report["config"]["stage1_k"] = stage1_k
                detailed_report["config"]["stage2_k"] = stage2_k

            if evaluation_scope == "per_dataset" and metrics_by_dataset:
                detailed_report["metrics_by_dataset"] = metrics_by_dataset

                st.markdown("**Per-Dataset Results**")
                for ds_name, ds_metrics in metrics_by_dataset.items():
                    ds_short = ds_name.split("/")[-1]
                    with st.expander(f"📁 {ds_short}"):
                        dc1, dc2, dc3, dc4 = st.columns(4)
                        dc1.metric("NDCG@10", f"{ds_metrics['ndcg@10']:.4f}")
                        dc2.metric("Recall@10", f"{ds_metrics['recall@10']:.4f}")
                        dc3.metric("MRR@10", f"{ds_metrics['mrr@10']:.4f}")
                        dc4.metric("Latency", f"{ds_metrics['avg_latency_ms']:.0f}ms")

            report_json = json.dumps(detailed_report, indent=2)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_report__{collection}__{mode}__{timestamp}.json"

            st.download_button(
                label="📥 Download Detailed Report",
                data=report_json,
                file_name=filename,
                mime="application/json",
                use_container_width=True,
            )

            st.session_state["last_eval_metrics"] = final_metrics

    except Exception as e:
        error_msg = str(e)

        if "not configured in this collection" in error_msg:
            vector_name = (
                error_msg.split("name ")[-1].split(" is")[0] if "name " in error_msg else "unknown"
            )
            st.error(
                f"❌ **Collection Mismatch**: Vector `{vector_name}` not found in collection `{collection}`"
            )
            st.warning(
                f"""
**The selected mode `{mode}` requires vectors that don't exist in this collection.**

**Suggestions:**
- Try `single_full` mode (works with basic collections)
- Use a collection indexed with the Visual RAG Toolkit
- Check that the collection has the required vector types for `{mode}` mode
            """
            )
        else:
            st.error(f"❌ Error: {e}")

        with st.expander("🔍 Full Error Details"):
            st.code(traceback.format_exc(), language="text")
