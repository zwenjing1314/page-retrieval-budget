"""Upload tab component."""

import inspect
import json
import os
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import streamlit as st

from demo.config import AVAILABLE_MODELS
from demo.qdrant_utils import (
    get_collection_stats,
    get_qdrant_credentials,
    sample_points_cached,
)
from visual_rag.embedding.visual_embedder import VisualEmbedder
from visual_rag.indexing.cloudinary_uploader import CloudinaryUploader
from visual_rag.indexing.pipeline import ProcessingPipeline
from visual_rag.indexing.qdrant_indexer import QdrantIndexer

VECTOR_TYPES = ["initial", "mean_pooling", "experimental_pooling", "global_pooling"]


def _load_metadata_mapping_from_uploaded_json(uploaded_json_file) -> tuple[dict, str]:
    """
    Load a filename->metadata mapping from an uploaded JSON file.

    Supported formats:
    - Flat dict:
        { "Some Report 2023": {"year": 2023, "source": "...", ...}, ... }
    - Nested dict:
        { "filenames": { "Some Report 2023": {...}, ... }, ... }

    Keys are normalized to: lowercase, trimmed, without ".pdf".
    """
    if uploaded_json_file is None:
        return {}, ""

    try:
        raw = uploaded_json_file.getvalue()
        if not raw:
            return {}, "Empty metadata file"
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            return {}, "Metadata file must be a JSON object"

        mapping = data.get("filenames") if isinstance(data.get("filenames"), dict) else data

        # Drop non-mapping keys (common pattern: _description, _usage)
        mapping = {k: v for k, v in mapping.items() if isinstance(k, str) and not k.startswith("_")}

        normalized: dict[str, dict] = {}
        bad = 0
        for k, v in mapping.items():
            if not isinstance(k, str) or not isinstance(v, dict):
                bad += 1
                continue
            key = k.strip().lower()
            if key.endswith(".pdf"):
                key = key[:-4]
            if not key:
                bad += 1
                continue
            normalized[key] = v

        msg = f"Loaded {len(normalized):,} filename metadata mappings"
        if bad:
            msg += f" (ignored {bad:,} non-mapping entries)"
        return normalized, msg
    except Exception as e:
        return {}, f"Failed to parse metadata JSON: {str(e)[:120]}"


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
            vector_dtype = st.selectbox(
                "Vector Dtype", ["float16", "float32"], index=0, key="upload_dtype"
            )
        with c4:
            use_cloudinary = st.toggle("Cloudinary", value=True, key="upload_cloudinary")

        st.markdown("**Performance**")
        p1, p2, p3 = st.columns(3)
        with p1:
            dpi = st.slider(
                "PDF DPI",
                min_value=90,
                max_value=220,
                value=int(st.session_state.get("upload_dpi", 140) or 140),
                step=10,
                key="upload_dpi",
                help="Lower DPI is faster. 120–150 is a good default for PDFs.",
            )
        with p2:
            embed_batch_size = st.slider(
                "Embedding batch",
                min_value=1,
                max_value=32,
                value=int(st.session_state.get("upload_embed_batch", 8) or 8),
                step=1,
                key="upload_embed_batch",
                help="Higher = faster (until you hit GPU/VRAM limits).",
            )
        with p3:
            upload_batch_size = st.slider(
                "Upload batch",
                min_value=1,
                max_value=32,
                value=int(st.session_state.get("upload_upload_batch", 8) or 8),
                step=1,
                key="upload_upload_batch",
                help="How many pages to upsert to Qdrant per batch.",
            )

        vectors_to_index = st.multiselect(
            "Vectors to Index",
            VECTOR_TYPES,
            default=VECTOR_TYPES,
            key="upload_vectors",
            help="Which vector types to store in Qdrant",
        )

        st.markdown("**Crop Settings**")
        cc1, cc2 = st.columns(2)
        with cc1:
            crop_empty = st.toggle("Crop Margins", value=True, key="upload_crop")
        with cc2:
            # Use a slider (instead of free typing) to avoid locale confusion like "0,00".
            # Threshold is std-dev of grayscale intensity (0..255). Smaller = stricter uniformity.
            uniform_rowcol_std_threshold = 0.0
            if crop_empty:
                uniform_rowcol_std_threshold = st.select_slider(
                    "Uniform row/col threshold (any color)",
                    options=[0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0],
                    value=float(
                        st.session_state.get("upload_uniform_rowcol_std_threshold", 0.0) or 0.0
                    ),
                    key="upload_uniform_rowcol_std_threshold",
                    help=(
                        "0 = off (default). Higher values remove more uniform borders, even if they are gray/black. "
                        "Rule: we skip a scanned row/col if std(pixels) ≤ threshold.\n\n"
                        "Suggested:\n"
                        "- 1–2: clean solid borders\n"
                        "- 3–5: light scanner shading\n"
                        "- 8+: aggressive (may remove faint content)"
                    ),
                )

        if crop_empty:
            crop_pct = st.slider(
                "Crop %",
                0.90,
                0.99,
                0.99,
                0.01,
                key="upload_crop_pct",
                help="Remove margins with this % empty space",
            )
        else:
            crop_pct = 0.99

        st.markdown("**File Metadata (optional)**")
        meta_file = st.file_uploader(
            "Metadata mapping (JSON)",
            type=["json"],
            key="upload_metadata_json",
            help=(
                "Optional JSON file that maps PDF filenames to extra metadata fields "
                "(e.g., year/source/district). Supported formats match `filename_metadata.json` "
                "and `metadata_mapping.json`."
            ),
        )
        metadata_mapping, meta_msg = _load_metadata_mapping_from_uploaded_json(meta_file)
        if meta_file:
            if metadata_mapping:
                st.success(meta_msg)
                # Show a tiny preview without overwhelming the UI
                with st.expander("Preview (first 3 entries)", expanded=False):
                    preview_items = list(metadata_mapping.items())[:3]
                    st.json({k: v for k, v in preview_items})
            else:
                st.warning(meta_msg or "No mappings loaded")
        else:
            metadata_mapping = {}

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
                config = {
                    "model_name": model_name,
                    "collection_name": collection_name,
                    "vector_dtype": vector_dtype,
                    "vectors_to_index": vectors_to_index,
                    "crop_empty": crop_empty,
                    "crop_pct": crop_pct,
                    "uniform_rowcol_std_threshold": float(uniform_rowcol_std_threshold or 0.0),
                    "use_cloudinary": use_cloudinary,
                    "metadata_mapping": metadata_mapping,
                    "dpi": int(dpi),
                    "embed_batch_size": int(embed_batch_size),
                    "upload_batch_size": int(upload_batch_size),
                }
                process_pdfs(uploaded_files, config)

    if st.session_state.get("last_upload_result"):
        st.divider()
        render_upload_results()


def process_pdfs(uploaded_files, config):
    model_name = config["model_name"]
    collection_name = config["collection_name"]
    vector_dtype = config["vector_dtype"]
    crop_empty = config["crop_empty"]
    crop_pct = config["crop_pct"]
    uniform_rowcol_std_threshold = float(config.get("uniform_rowcol_std_threshold") or 0.0)
    use_cloudinary = config["use_cloudinary"]
    metadata_mapping = config.get("metadata_mapping") or {}
    dpi = int(config.get("dpi") or 140)
    embed_batch_size = int(config.get("embed_batch_size") or 8)
    upload_batch_size = int(config.get("upload_batch_size") or 8)

    st.divider()

    phase1 = st.container()
    phase2 = st.container()
    phase3 = st.container()
    results_container = st.container()

    try:
        with phase1:
            st.markdown("##### 🤖 Phase 1: Loading Model")
            model_status = st.empty()
            model_short = model_name.split("/")[-1]
            model_status.info(f"Loading `{model_short}`...")

            output_dtype = np.float16 if vector_dtype == "float16" else np.float32
            from demo.qdrant_utils import init_embedder
            embedder, err = init_embedder(model_name)
            if err:
                model_status.error(f"❌ Failed to load model: {err[:100]}")
                return
            embedder.output_dtype = output_dtype
            model_status.success(f"✅ Model `{model_short}` loaded ({vector_dtype})")

        with phase2:
            st.markdown("##### 📦 Phase 2: Setting Up Collection")

            url, api_key = get_qdrant_credentials()
            if not url or not api_key:
                st.error("Qdrant credentials not configured")
                return

            qdrant_status = st.empty()
            qdrant_status.info("Connecting to Qdrant...")

            indexer = QdrantIndexer(
                url=url,
                api_key=api_key,
                collection_name=collection_name,
                prefer_grpc=False,
                vector_datatype=vector_dtype,
                timeout=180,
            )
            qdrant_status.success("✅ Connected to Qdrant")

            coll_status = st.empty()
            collection_exists = False
            try:
                collection_exists = indexer.collection_exists()
            except Exception:
                pass

            if collection_exists:
                coll_status.success(f"✅ Collection `{collection_name}` exists (will append)")
            else:
                coll_status.info(f"Creating collection `{collection_name}`...")
                for attempt in range(3):
                    try:
                        model_lower = (model_name or "").lower()
                        is_colqwen25 = "colqwen2.5" in model_lower or "colqwen2_5" in model_lower
                        if is_colqwen25:
                            exp_names = ["experimental_pooling_gaussian", "experimental_pooling_triangular"]
                        else:
                            exp_names = ["experimental_pooling_3"]
                        indexer.create_collection(
                            force_recreate=False,
                            experimental_vector_names=exp_names,
                        )
                        break
                    except Exception:
                        if attempt < 2:
                            time.sleep(2)
                        else:
                            raise
                coll_status.success(f"✅ Collection `{collection_name}` created")

            idx_status = st.empty()
            idx_status.info("Setting up indexes...")
            try:
                indexer.create_payload_indexes(
                    fields=[
                        {"field": "filename", "type": "keyword"},
                        {"field": "page_number", "type": "integer"},
                    ]
                )
            except Exception:
                pass
            idx_status.success("✅ Indexes ready")

            cloud_status = st.empty()
            cloudinary_uploader = None
            if use_cloudinary:
                cloud_status.info("Connecting to Cloudinary...")
                try:
                    cloudinary_uploader = CloudinaryUploader(folder=f"visual-rag/{collection_name}")
                    cloud_status.success("✅ Cloudinary ready")
                except Exception as e:
                    cloud_status.warning(f"⚠️ Cloudinary unavailable: {str(e)[:30]}")
            else:
                cloud_status.info("☁️ Cloudinary disabled")

            pipeline = ProcessingPipeline(
                embedder=embedder,
                indexer=indexer,
                cloudinary_uploader=cloudinary_uploader,
                metadata_mapping=metadata_mapping,
                config={
                    "processing": {"dpi": dpi},
                    "batching": {
                        "embedding_batch_size": embed_batch_size,
                        "upload_batch_size": upload_batch_size,
                    },
                },
                crop_empty=crop_empty,
                crop_empty_percentage_to_remove=crop_pct,
                **(
                    {"crop_empty_uniform_rowcol_std_threshold": uniform_rowcol_std_threshold}
                    if "crop_empty_uniform_rowcol_std_threshold"
                    in inspect.signature(ProcessingPipeline.__init__).parameters
                    else {}
                ),
            )

        with phase3:
            st.markdown("##### 📄 Phase 3: Processing PDFs")

            overall_progress = st.progress(0.0)
            file_status = st.empty()
            log_area = st.empty()
            log_lines = []

            total_uploaded, total_skipped, total_failed = 0, 0, 0
            file_results = []

            page_status = st.empty()

            for i, f in enumerate(uploaded_files):
                original_filename = f.name
                file_status.info(
                    f"📄 Processing `{original_filename}` ({i+1}/{len(uploaded_files)})"
                )
                t0 = time.perf_counter()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                    tmp_path = Path(tmp.name)

                def progress_cb(stage, current, total, message):
                    if stage == "process" and total > 0:
                        page_status.caption(f"  └─ Page {current}/{total}")
                    elif stage == "embed" and total > 0:
                        # Never show internal function names; keep this human-friendly.
                        page_status.caption(
                            f"  └─ Embedding pages… ({current+1}-{min(current + 1 + (pipeline.embedding_batch_size - 1), total)}/{total})"
                        )
                    elif stage == "convert" and total > 0:
                        page_status.caption(f"  └─ {total} pages found")

                try:
                    result = pipeline.process_pdf(
                        tmp_path,
                        original_filename=original_filename,
                        progress_callback=progress_cb,
                    )
                    elapsed_s = time.perf_counter() - t0
                    uploaded = result.get("uploaded", 0)
                    skipped = result.get("skipped", 0)
                    total_uploaded += uploaded
                    total_skipped += skipped
                    total_pages = int(result.get("total_pages") or 0)
                    sec_per_page = (elapsed_s / total_pages) if total_pages > 0 else None
                    file_results.append(
                        {
                            "file": original_filename,
                            "uploaded": uploaded,
                            "skipped": skipped,
                            "total_pages": total_pages,
                            "elapsed_s": float(elapsed_s),
                            "sec_per_page": (
                                float(sec_per_page) if sec_per_page is not None else None
                            ),
                        }
                    )
                    timing_str = f"{elapsed_s:.1f}s" + (
                        f" ({sec_per_page:.2f}s/page)" if sec_per_page is not None else ""
                    )
                    log_lines.append(
                        f"✓ {original_filename}: {uploaded} uploaded, {skipped} skipped | {timing_str}"
                    )
                except Exception as e:
                    total_failed += 1
                    log_lines.append(f"✗ {original_filename}: {str(e)[:50]}")
                finally:
                    os.unlink(tmp_path)

                page_status.empty()
                overall_progress.progress((i + 1) / len(uploaded_files))
                log_area.code("\n".join(log_lines[-10:]), language="text")

            overall_progress.progress(1.0)
            file_status.success(f"✅ Processed {len(uploaded_files)} file(s)")

        with results_container:
            st.markdown("##### 📊 Results")

            st.success(
                f"✅ **{total_uploaded} pages** uploaded to `{collection_name}`"
                + (f" ({total_skipped} skipped)" if total_skipped else "")
                + (f" ({total_failed} failed)" if total_failed else "")
            )

            if file_results:
                with st.expander("📋 File Details", expanded=False):
                    for fr in file_results:
                        timing = ""
                        if fr.get("elapsed_s") is not None:
                            timing = f" | {fr['elapsed_s']:.1f}s"
                            if fr.get("sec_per_page") is not None:
                                timing += f" ({fr['sec_per_page']:.2f}s/page)"
                        st.text(
                            f"• {fr['file']}: {fr['uploaded']} uploaded"
                            + (f", {fr['skipped']} skipped" if fr.get("skipped") else "")
                            + (f", {fr['total_pages']} pages" if fr.get("total_pages") else "")
                            + timing
                        )

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
            st.session_state["upload_success"] = (
                f"Uploaded {total_uploaded} pages to {collection_name}"
            )
            st.rerun()  # Immediately refresh to show success toast + balloons

    except Exception as e:
        st.error(f"❌ Processing error: {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())


def render_upload_results():
    result = st.session_state.get("last_upload_result", {})
    if not result:
        return

    uploaded = result.get("total_uploaded", 0)
    skipped = result.get("total_skipped", 0)
    failed = result.get("total_failed", 0)
    collection = result.get("collection", "")
    file_results = result.get("file_results", [])

    st.success(
        f"✅ **{uploaded} pages** uploaded to `{collection}`"
        + (f" ({skipped} skipped)" if skipped else "")
        + (f" ({failed} failed)" if failed else "")
    )

    if file_results:
        with st.expander("📋 Details", expanded=False):
            for fr in file_results:
                timing = ""
                if fr.get("elapsed_s") is not None:
                    timing = f" | {fr['elapsed_s']:.1f}s"
                    if fr.get("sec_per_page") is not None:
                        timing += f" ({fr['sec_per_page']:.2f}s/page)"
                st.text(
                    f"• {fr['file']}: {fr['uploaded']} uploaded"
                    + (f", {fr['skipped']} skipped" if fr.get("skipped") else "")
                    + (f", {fr['total_pages']} pages" if fr.get("total_pages") else "")
                    + timing
                )
