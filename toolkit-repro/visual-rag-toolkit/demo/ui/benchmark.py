"""Benchmark tab component."""

from pathlib import Path
from typing import Any, Dict, List

import altair as alt
import pandas as pd
import streamlit as st

from demo.commands import (
    build_eval_command,
    build_index_command,
    generate_python_eval_code,
    generate_python_index_code,
)
from demo.config import (
    AVAILABLE_MODELS,
    BENCHMARK_DATASETS,
    DATASET_STATS,
    RETRIEVAL_MODES,
    STAGE1_MODES,
)
from demo.evaluation import run_evaluation_with_ui
from demo.indexing import run_indexing_with_ui
from demo.qdrant_utils import get_collections, get_qdrant_credentials
from demo.results import get_available_results, load_results_file


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
    st.caption("Create a new collection with benchmark datasets")

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
            "New Collection Name", value=f"vidore_{len(datasets)}ds__{model_short}", key="bi_coll"
        )

    sel_docs = sum(DATASET_STATS.get(d, {}).get("docs", 0) for d in datasets)
    sel_queries = sum(DATASET_STATS.get(d, {}).get("queries", 0) for d in datasets)
    st.markdown(
        f"🎯 **Selected:** {len(datasets)} dataset(s) — **{sel_docs:,}** docs, **{sel_queries:,}** queries"
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

    st.markdown("---")

    col_max, col_batch, col_torch, col_qdrant = st.columns([2, 2, 1, 1])
    with col_max:
        max_docs_val = max(sel_docs, 1)
        max_docs = st.number_input(
            "Max Docs (per dataset)",
            min_value=1,
            max_value=max_docs_val,
            value=max_docs_val,
            key="bi_max_docs",
            help="Limit docs per dataset. Useful for quick tests.",
        )
    with col_batch:
        batch_size = st.number_input(
            "Batch Size", min_value=1, max_value=16, value=4, key="bi_batch"
        )
    with col_torch:
        torch_dtype = st.selectbox(
            "Torch dtype", ["float16", "float32"], index=0, key="bi_torch_dtype"
        )
    with col_qdrant:
        qdrant_dtype = st.selectbox(
            "Qdrant dtype", ["float16", "float32"], index=0, key="bi_qdrant_dtype"
        )

    effective_docs = (
        min(max_docs * len(datasets), sel_docs) if max_docs < max_docs_val else sel_docs
    )

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
        "batch_size": batch_size,
        "upload_batch_size": 8,
        "qdrant_timeout": 180,
        "qdrant_retries": 5,
        "torch_dtype": torch_dtype,
        "qdrant_vector_dtype": qdrant_dtype,
        "max_docs": max_docs if max_docs < max_docs_val else None,
    }

    cmd = build_index_command(config)
    python_code = generate_python_index_code(config)

    col_cmd, col_info = st.columns([2, 1])
    with col_cmd:
        code_tab1, code_tab2 = st.tabs(["🐚 Bash", "🐍 Python"])
        with code_tab1:
            st.code(cmd, language="bash")
        with code_tab2:
            st.code(python_code, language="python")
    with col_info:
        st.markdown("<br><br><br>", unsafe_allow_html=True)

        st.metric("Docs to Index", f"{effective_docs:,}")
        st.metric("Model", model.split("/")[-1])
        if effective_docs < sel_docs:
            st.caption(f"Limited from {sel_docs:,} total")
        st.divider()
        run_index = st.button(
            "🚀 Run Index", type="primary", key="bi_run", use_container_width=True
        )

    if run_index:
        if not collection:
            st.error("Please provide a collection name")
        elif not datasets:
            st.error("Please select at least one dataset")
        else:
            run_indexing_with_ui(config)


def render_benchmark_evaluation(collections: List[str]):
    collection = st.session_state.get("active_collection")

    if not collection:
        st.warning("⚠️ Select a collection from the sidebar first")
        return

    st.info(f"**Collection:** `{collection}` (from sidebar)")

    all_docs = sum(DATASET_STATS.get(d, {}).get("docs", 0) for d in BENCHMARK_DATASETS)
    all_queries = sum(DATASET_STATS.get(d, {}).get("queries", 0) for d in BENCHMARK_DATASETS)
    st.markdown(
        f"📊 **Available:** {len(BENCHMARK_DATASETS)} datasets — **{all_docs:,}** docs, **{all_queries:,}** queries"
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        st.multiselect("Datasets", BENCHMARK_DATASETS, default=BENCHMARK_DATASETS, key="be_ds")
    with c2:
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

    col_scope, _, col_grpc, col_nq = st.columns([2, 0.5, 1, 2])
    with col_scope:
        scope = st.selectbox("Scope", ["union", "per_dataset"], key="be_scope")
    with col_grpc:
        st.write("")
        st.write("")
        grpc = st.toggle("gRPC", value=True, key="be_grpc")
    with col_nq:
        max_q_val = max(sel_queries, 1)
        max_queries = st.number_input(
            "Max Queries",
            min_value=1,
            max_value=max_q_val,
            value=max_q_val,
            key="be_max_queries",
            help="Limit number of queries to evaluate (useful for quick tests)",
        )

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
        "max_queries": max_queries,
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
        st.markdown("<br><br><br>", unsafe_allow_html=True)

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
