"""Sidebar component."""

import os

import streamlit as st
from qdrant_client.models import VectorParamsDiff

from demo.qdrant_utils import (
    get_collection_stats,
    get_collections,
    get_qdrant_credentials,
    get_vector_sizes,
    init_qdrant_client_with_creds,
    sample_points_cached,
)


def render_sidebar():
    # CSS to make sidebar metrics smaller
    st.markdown(
        """
    <style>
    /* Smaller metrics in sidebar */
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
    }
    /* Smaller expander headers in sidebar */
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        font-size: 0.9rem !important;
    }
    /* Compact subheaders */
    [data-testid="stSidebar"] h3 {
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("🔑 Qdrant Credentials")

        env_url = os.getenv("QDRANT_URL") or os.getenv("SIGIR_QDRANT_URL") or ""
        env_key = os.getenv("QDRANT_API_KEY") or os.getenv("SIGIR_QDRANT_KEY") or ""

        if "qdrant_url_input" not in st.session_state:
            st.session_state["qdrant_url_input"] = env_url
        if "qdrant_key_input" not in st.session_state:
            st.session_state["qdrant_key_input"] = env_key

        qdrant_url = st.text_input(
            "Qdrant URL",
            value=st.session_state["qdrant_url_input"],
            key="qdrant_url_widget",
            placeholder="https://xxx.cloud.qdrant.io:6333",
        )
        qdrant_key = st.text_input(
            "API Key",
            value=st.session_state["qdrant_key_input"],
            key="qdrant_key_widget",
            type="password",
        )

        if (
            qdrant_url != st.session_state["qdrant_url_input"]
            or qdrant_key != st.session_state["qdrant_key_input"]
        ):
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
