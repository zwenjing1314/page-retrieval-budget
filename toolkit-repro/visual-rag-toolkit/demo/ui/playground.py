"""Playground tab component."""

import streamlit as st

from demo.config import AVAILABLE_MODELS, RETRIEVAL_MODES, STAGE1_MODES
from demo.qdrant_utils import (
    get_collections,
    get_qdrant_credentials,
    sample_points_cached,
    search_collection,
)
from visual_rag.retrieval import MultiVectorRetriever


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
                    from demo.qdrant_utils import init_embedder
                    embedder, err = init_embedder(model_name)
                    if err:
                        st.warning(f"Failed: {model_short}")
                    else:
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
    def _is_missing(v) -> bool:
        if v is None:
            return True
        if isinstance(v, (list, tuple, dict)) and len(v) == 0:
            return True
        if isinstance(v, str):
            s = v.strip()
            return s == "" or s.lower() in {"na", "n/a", "none", "null", "unknown", "?", "-"}
        return False

    doc_id = p.get("doc_id") or p.get("union_doc_id") or p.get("source_doc_id") or "?"
    corpus_id = p.get("corpus-id") or p.get("source_doc_id") or "?"
    dataset = p.get("dataset") or p.get("source") or None
    model = p.get("model_name") or p.get("model") or None
    model = model.split("/")[-1] if isinstance(model, str) else None
    doc_name = p.get("doc-id") or p.get("filename") or "Unknown"

    num_tiles = p.get("num_tiles")
    visual_tokens = p.get("index_recovery_num_visual_tokens") or p.get("num_visual_tokens")
    patches_per_tile = p.get("patches_per_tile")
    torch_dtype = p.get("torch_dtype")

    orig_w = p.get("original_width")
    orig_h = p.get("original_height")
    crop_w = p.get("cropped_width")
    crop_h = p.get("cropped_height")
    resize_w = p.get("resized_width")
    resize_h = p.get("resized_height")
    crop_pct = p.get("crop_empty_percentage_to_remove")
    crop_enabled = bool(p.get("crop_empty_enabled", False))

    col_meta, col_img = st.columns([1, 2])

    with col_meta:
        st.markdown("##### 📄 Document Info")
        st.markdown(f"**📁 Doc:** {doc_name}")
        if not _is_missing(dataset):
            st.markdown(f"**🏛️ Dataset:** {dataset}")
        if not _is_missing(doc_id) and str(doc_id) != "?":
            st.markdown(f"**🔑 Doc ID:** `{str(doc_id)[:20]}...`")
        if not _is_missing(corpus_id) and str(corpus_id) != "?":
            st.markdown(f"**📋 Corpus ID:** {corpus_id}")

        if score is not None:
            st.divider()
            st.markdown("##### 🎯 Relevance")
            if rel_pct is not None:
                st.markdown(f"**Relative:** 🟢 {rel_pct:.1f}%")
                st.progress(rel_pct / 100)
            st.caption(f"Raw score: {score:.4f}")

        st.divider()
        visual_rows = []
        if not _is_missing(model):
            visual_rows.append(("🤖 Model", f"`{model}`"))
        if not _is_missing(num_tiles):
            visual_rows.append(("🔲 Tiles", str(num_tiles)))
        if not _is_missing(visual_tokens):
            visual_rows.append(("🔢 Visual Tokens", str(visual_tokens)))
        if not _is_missing(patches_per_tile):
            visual_rows.append(("📦 Patches/Tile", str(patches_per_tile)))
        if not _is_missing(torch_dtype):
            visual_rows.append(("⚙️ Dtype", str(torch_dtype)))
        if visual_rows:
            st.markdown("##### 🎨 Visual Metadata")
            for k, v in visual_rows:
                st.markdown(f"**{k}:** {v}")

        st.divider()
        dim_rows = []
        if not _is_missing(orig_w) and not _is_missing(orig_h):
            dim_rows.append(("Original", f"{orig_w}×{orig_h}"))
        if not _is_missing(resize_w) and not _is_missing(resize_h):
            dim_rows.append(("Resized", f"{resize_w}×{resize_h}"))
        if crop_enabled and not _is_missing(crop_w) and not _is_missing(crop_h):
            dim_rows.append(("Cropped", f"{crop_w}×{crop_h}"))
        if dim_rows:
            st.markdown("##### 📐 Dimensions")
            for k, v in dim_rows:
                st.markdown(f"**{k}:** {v}")
        if crop_enabled and not _is_missing(crop_pct):
            try:
                st.markdown(f"**Crop %:** {int(float(crop_pct) * 100)}%")
            except Exception:
                pass

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
                # Display on a checkerboard background to make the crop boundary obvious.
                w_caption = (
                    f"{crop_w}×{crop_h}"
                    if (not _is_missing(crop_w) and not _is_missing(crop_h))
                    else None
                )
                pct_caption = None
                if not _is_missing(crop_pct):
                    try:
                        pct_caption = f"{int(float(crop_pct) * 100)}%"
                    except Exception:
                        pct_caption = None
                st.markdown(
                    f"""
                    <div style="
                        width: 600px;
                        padding: 14px;
                        border-radius: 10px;
                        background-image:
                          linear-gradient(45deg, #e6e6e6 25%, transparent 25%),
                          linear-gradient(-45deg, #e6e6e6 25%, transparent 25%),
                          linear-gradient(45deg, transparent 75%, #e6e6e6 75%),
                          linear-gradient(-45deg, transparent 75%, #e6e6e6 75%);
                        background-size: 24px 24px;
                        background-position: 0 0, 0 12px, 12px -12px, -12px 0px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
                        display: inline-block;
                    ">
                        <img src="{url_c}" style="width: 100%; border-radius: 6px; display:block;" />
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                cap = []
                if w_caption:
                    cap.append(f"📐 **{w_caption}**")
                if pct_caption:
                    cap.append(f"Crop: {pct_caption}")
                if cap:
                    st.caption(" | ".join(cap))
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
