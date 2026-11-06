"""Indexing runner with UI updates."""

import hashlib
import traceback
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st
import torch

from benchmarks.vidore_tatdqa_test.dataset_loader import load_vidore_beir_dataset
from demo.qdrant_utils import get_qdrant_credentials
from visual_rag import VisualEmbedder
from visual_rag.embedding.pooling import tile_level_mean_pooling
from visual_rag.indexing.qdrant_indexer import QdrantIndexer

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


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


def run_indexing_with_ui(config: Dict[str, Any]):
    st.divider()

    print("=" * 60)
    print("[INDEX] Starting indexing via UI")
    print("=" * 60)

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
    max_docs = config.get("max_docs")

    print(f"[INDEX] Config: collection={collection}, model={model}")
    print(f"[INDEX] Datasets: {datasets}")
    print(f"[INDEX] max_docs={max_docs}, batch_size={batch_size}, recreate={recreate}")
    print(
        f"[INDEX] torch_dtype={torch_dtype}, qdrant_dtype={qdrant_vector_dtype}, grpc={prefer_grpc}"
    )

    phase1_container = st.container()
    phase2_container = st.container()
    phase3_container = st.container()
    results_container = st.container()

    try:
        with phase1_container:
            st.markdown("##### 🤖 Phase 1: Loading Model")
            model_status = st.empty()
            model_status.info(f"Loading `{model.split('/')[-1]}`...")

            print(f"[INDEX] Loading embedder: {model}")
            torch_dtype_obj = TORCH_DTYPE_MAP.get(torch_dtype, torch.float16)
            output_dtype_obj = np.float16 if qdrant_vector_dtype == "float16" else np.float32
            embedder = VisualEmbedder(
                model_name=model,
                torch_dtype=torch_dtype_obj,
                output_dtype=output_dtype_obj,
            )
            embedder._load_model()
            print(
                f"[INDEX] Embedder loaded (torch_dtype={torch_dtype}, output_dtype={qdrant_vector_dtype})"
            )
            model_status.success(f"✅ Model `{model.split('/')[-1]}` loaded")

        with phase2_container:
            st.markdown("##### 📦 Phase 2: Setting Up Collection")

            indexer_status = st.empty()
            indexer_status.info("Connecting to Qdrant...")

            print("[INDEX] Connecting to Qdrant...")
            indexer = QdrantIndexer(
                url=url,
                api_key=api_key,
                collection_name=collection,
                prefer_grpc=prefer_grpc,
                vector_datatype=qdrant_vector_dtype,
            )
            print("[INDEX] Connected to Qdrant")
            indexer_status.success("✅ Connected to Qdrant")

            coll_status = st.empty()
            action = "Recreating" if recreate else "Creating/verifying"
            coll_status.info(f"{action} collection `{collection}`...")

            print(f"[INDEX] {action} collection: {collection}")
            indexer.create_collection(force_recreate=recreate)
            indexer.create_payload_indexes(
                fields=[
                    {"field": "dataset", "type": "keyword"},
                    {"field": "doc_id", "type": "keyword"},
                    {"field": "source_doc_id", "type": "keyword"},
                ]
            )
            print("[INDEX] Collection ready")
            coll_status.success(f"✅ Collection `{collection}` ready")

        with phase3_container:
            st.markdown("##### 📊 Phase 3: Processing Datasets")

            all_results = []

            for ds_idx, dataset_name in enumerate(datasets):
                ds_short = dataset_name.split("/")[-1]
                ds_container = st.container()

                with ds_container:
                    st.markdown(f"**Dataset {ds_idx + 1}/{len(datasets)}: `{ds_short}`**")

                    load_status = st.empty()
                    load_status.info(f"Loading dataset `{ds_short}`...")

                    print(f"[INDEX] Loading dataset: {dataset_name}")
                    corpus, queries, qrels = load_vidore_beir_dataset(dataset_name)
                    total_docs = len(corpus)
                    print(f"[INDEX] Dataset loaded: {total_docs} docs")
                    load_status.success(f"✅ Loaded {total_docs:,} documents")

                    if max_docs and max_docs < total_docs:
                        corpus = corpus[:max_docs]
                        print(f"[INDEX] Limiting to {max_docs} docs")

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    uploaded = 0
                    failed = 0
                    total = len(corpus)

                    for i, doc in enumerate(corpus):
                        try:
                            doc_id = str(doc.doc_id)
                            image = doc.image
                            if image is None:
                                failed += 1
                                continue

                            status_text.text(f"Processing {i + 1}/{total}: {doc_id[:30]}...")

                            embeddings, token_infos = embedder.embed_images(
                                [image],
                                return_token_info=True,
                                show_progress=False,
                            )
                            emb = embeddings[0]
                            token_info = token_infos[0] if token_infos else {}

                            if hasattr(emb, "cpu"):
                                emb = emb.cpu()
                            emb_np = np.asarray(emb, dtype=output_dtype_obj)

                            initial = emb_np.tolist()
                            global_pool = emb_np.mean(axis=0).tolist()

                            num_tiles = token_info.get("num_tiles")
                            mean_pooling = None
                            experimental_pooling = None

                            if num_tiles and num_tiles > 0:
                                try:
                                    mean_pooling = tile_level_mean_pooling(
                                        emb_np, num_tiles=num_tiles, patches_per_tile=64
                                    ).tolist()
                                except Exception:
                                    pass

                                try:
                                    exp_pool = embedder.experimental_pool_visual_embedding(
                                        emb_np, num_tiles=num_tiles
                                    )
                                    if exp_pool is not None:
                                        experimental_pooling = exp_pool.tolist()
                                except Exception:
                                    pass

                            union_doc_id = _union_point_id(
                                dataset_name=dataset_name,
                                source_doc_id=doc_id,
                                union_namespace=collection,
                            )

                            payload = {
                                "dataset": dataset_name,
                                "doc_id": doc_id,
                                "source_doc_id": doc_id,
                                "union_doc_id": union_doc_id,
                                "num_tiles": num_tiles,
                                "num_visual_tokens": token_info.get("num_visual_tokens"),
                            }

                            vectors = {"initial": initial, "global_pooling": global_pool}
                            if mean_pooling:
                                vectors["mean_pooling"] = mean_pooling
                            if experimental_pooling:
                                vectors["experimental_pooling"] = experimental_pooling

                            indexer.upsert_point(
                                point_id=union_doc_id,
                                vectors=vectors,
                                payload=payload,
                            )

                            uploaded += 1

                        except Exception as e:
                            print(f"[INDEX] Error on doc {i}: {e}")
                            failed += 1

                        progress_bar.progress((i + 1) / total)

                    status_text.text(f"✅ Done: {uploaded} uploaded, {failed} failed")
                    all_results.append(
                        {
                            "dataset": dataset_name,
                            "total": total,
                            "uploaded": uploaded,
                            "failed": failed,
                        }
                    )

        with results_container:
            st.markdown("##### 📋 Results Summary")

            for r in all_results:
                st.write(
                    f"**{r['dataset'].split('/')[-1]}**: {r['uploaded']:,} uploaded, {r['failed']:,} failed"
                )

            st.success("✅ Indexing complete!")

    except Exception as e:
        st.error(f"Indexing error: {e}")
        st.code(traceback.format_exc())
        print(f"[INDEX] ERROR: {e}")
        traceback.print_exc()
