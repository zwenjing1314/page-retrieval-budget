#!/usr/bin/env python3
"""
Visual RAG Toolkit CLI

Provides command-line interface for:
- Processing PDFs (embedding, Cloudinary upload, Qdrant indexing)
- Searching documents
- Managing collections

Usage:
    # Process PDFs (like process_pdfs_saliency_v2.py)
    visual-rag process --reports-dir ./pdfs --metadata-file metadata.json

    # Search
    visual-rag search --query "budget allocation" --collection my_docs

    # Show collection info
    visual-rag info --collection my_docs
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False):
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def cmd_process(args):
    """
    Process PDFs: convert → embed → upload to Cloudinary → index in Qdrant.

    Equivalent to process_pdfs_saliency_v2.py
    """
    from visual_rag import CloudinaryUploader, QdrantIndexer, VisualEmbedder, load_config
    from visual_rag.indexing.pipeline import ProcessingPipeline

    # Load environment
    load_dotenv()

    # Load config
    config = {}
    if args.config and Path(args.config).exists():
        config = load_config(args.config)

    # Get PDFs
    reports_dir = Path(args.reports_dir)
    if not reports_dir.exists():
        logger.error(f"❌ Reports directory not found: {reports_dir}")
        sys.exit(1)

    pdf_paths = sorted(reports_dir.glob("*.pdf")) + sorted(reports_dir.glob("*.PDF"))
    if not pdf_paths:
        logger.error(f"❌ No PDF files found in: {reports_dir}")
        sys.exit(1)

    logger.info(f"📁 Found {len(pdf_paths)} PDF files")

    # Load metadata mapping
    metadata_mapping = {}
    if args.metadata_file:
        metadata_mapping = ProcessingPipeline.load_metadata_mapping(Path(args.metadata_file))

    # Dry run - just show summary
    if args.dry_run:
        logger.info("🏃 DRY RUN MODE")
        logger.info(f"   PDFs: {len(pdf_paths)}")
        logger.info(f"   Metadata entries: {len(metadata_mapping)}")
        logger.info(f"   Collection: {args.collection}")
        logger.info(f"   Cloudinary: {'ENABLED' if not args.no_cloudinary else 'DISABLED'}")

        for pdf in pdf_paths[:10]:
            has_meta = "✓" if pdf.stem.lower() in metadata_mapping else "✗"
            logger.info(f"   {has_meta} {pdf.name}")
        if len(pdf_paths) > 10:
            logger.info(f"   ... and {len(pdf_paths) - 10} more")
        return

    # Get settings
    model_name = args.model or config.get("model", {}).get("name", "vidore/colSmol-500M")
    collection_name = args.collection or config.get("qdrant", {}).get(
        "collection_name", "visual_documents"
    )

    torch_dtype = None
    if args.torch_dtype != "auto":
        import torch

        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[args.torch_dtype]

    logger.info(f"🤖 Initializing embedder: {model_name}")
    embedder = VisualEmbedder(
        model_name=model_name,
        batch_size=args.batch_size,
        torch_dtype=torch_dtype,
        processor_speed=str(getattr(args, "processor_speed", "fast")),
    )

    # Experimental pooling vectors (for additional Qdrant named vectors)
    model_lower = (model_name or "").lower()
    is_colqwen25 = "colqwen2.5" in model_lower or "colqwen2_5" in model_lower
    is_colsmol = "colsmol" in model_lower
    experimental_vector_names = []
    if is_colqwen25:
        # ColQwen2.5: always store both named vectors explicitly.
        experimental_vector_names.extend(
            ["experimental_pooling_gaussian", "experimental_pooling_triangular"]
        )
        if getattr(args, "pooling_windows", None):
            logger.warning(
                "⚠️ --pooling-windows is ignored for ColQwen2.5 (use technique variants instead)."
            )
        if str(
            getattr(args, "experimental_pooling_kernel", "auto") or "auto"
        ).lower().strip() not in ("auto", "gaussian", "triangular"):
            logger.warning(
                "⚠️ --experimental-pooling-kernel is ignored for ColQwen2.5 (fixed gaussian+triangular k=3)."
            )
    else:
        # ColPali-style: optional multiple ks stored as experimental_pooling_{k}
        default_k = 3
        ks = args.pooling_windows if getattr(args, "pooling_windows", None) else [default_k]
        seen_ks = set()
        ks_norm = []
        for k in ks:
            try:
                ki = int(k)
            except Exception:
                continue
            if ki <= 0:
                continue
            if ki in seen_ks:
                continue
            seen_ks.add(ki)
            ks_norm.append(ki)
        if not ks_norm:
            ks_norm = [default_k]
        experimental_vector_names = [f"experimental_pooling_{int(k)}" for k in ks_norm]
    if is_colsmol and bool(getattr(args, "colsmol_experimental_2d", False)):
        experimental_vector_names.append("experimental_pooling_2d")

    # Initialize Qdrant indexer
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url:
        logger.error("❌ QDRANT_URL environment variable not set")
        sys.exit(1)

    logger.info(f"🔌 Connecting to Qdrant: {qdrant_url}")
    indexer = QdrantIndexer(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        prefer_grpc=args.prefer_grpc,
        vector_datatype=args.qdrant_vector_dtype,
    )

    # Create collection if needed
    indexer.create_collection(
        force_recreate=args.force_recreate,
        experimental_vector_names=experimental_vector_names,
    )
    inferred_fields = []
    inferred_fields.append({"field": "filename", "type": "keyword"})
    inferred_fields.append({"field": "page_number", "type": "integer"})
    inferred_fields.append({"field": "has_text", "type": "bool"})

    if metadata_mapping:
        keys = set()
        for _, meta in metadata_mapping.items():
            if isinstance(meta, dict):
                keys.update(meta.keys())
        for k in sorted(keys):
            if k in ("filename", "page_number", "has_text"):
                continue
            inferred_type = "keyword"
            for _, meta in metadata_mapping.items():
                if not isinstance(meta, dict):
                    continue
                v = meta.get(k)
                if isinstance(v, bool):
                    inferred_type = "bool"
                    break
                if isinstance(v, int):
                    inferred_type = "integer"
                    break
                if isinstance(v, float):
                    inferred_type = "float"
                    break
            inferred_fields.append({"field": k, "type": inferred_type})

    indexer.create_payload_indexes(fields=inferred_fields)

    # Initialize Cloudinary uploader (optional)
    cloudinary_uploader = None
    if not args.no_cloudinary:
        try:
            project_name = config.get("project_name", "visual_docs")
            cloudinary_uploader = CloudinaryUploader(folder=project_name)
        except ValueError as e:
            logger.warning(f"⚠️ Cloudinary not configured: {e}")
            logger.warning("   Continuing without Cloudinary uploads")

    # Create pipeline
    pipeline = ProcessingPipeline(
        embedder=embedder,
        indexer=indexer,
        cloudinary_uploader=cloudinary_uploader,
        metadata_mapping=metadata_mapping,
        config=config,
        embedding_strategy=args.strategy,
        crop_empty=bool(getattr(args, "crop_empty", False)),
        crop_empty_percentage_to_remove=float(
            getattr(args, "crop_empty_percentage_to_remove", 0.9)
        ),
        crop_empty_remove_page_number=bool(getattr(args, "crop_empty_remove_page_number", False)),
        max_mean_pool_vectors=getattr(args, "max_mean_pool_vectors", 32),
        pooling_windows=getattr(args, "pooling_windows", None),
        experimental_pooling_kernel=str(getattr(args, "experimental_pooling_kernel", "auto")),
        colsmol_experimental_2d=bool(getattr(args, "colsmol_experimental_2d", False)),
    )

    # Process PDFs
    total_uploaded = 0
    total_skipped = 0
    total_failed = 0

    skip_existing = not args.no_skip_existing

    for pdf_idx, pdf_path in enumerate(pdf_paths, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"📄 [{pdf_idx}/{len(pdf_paths)}] {pdf_path.name}")
        logger.info(f"{'='*60}")

        result = pipeline.process_pdf(
            pdf_path,
            skip_existing=skip_existing,
            upload_to_cloudinary=(not args.no_cloudinary),
            upload_to_qdrant=True,
        )

        total_uploaded += result["uploaded"]
        total_skipped += result["skipped"]
        total_failed += result["failed"]

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("📊 SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"   Total PDFs: {len(pdf_paths)}")
    logger.info(f"   Uploaded: {total_uploaded}")
    logger.info(f"   Skipped: {total_skipped}")
    logger.info(f"   Failed: {total_failed}")

    info = indexer.get_collection_info()
    if info:
        logger.info(f"   Collection points: {info.get('points_count', 'N/A')}")


def cmd_search(args):
    """Search documents."""
    from qdrant_client import QdrantClient

    from visual_rag import VisualEmbedder
    from visual_rag.retrieval import SingleStageRetriever, TwoStageRetriever

    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url:
        logger.error("❌ QDRANT_URL not set")
        sys.exit(1)

    # Initialize
    logger.info(f"🤖 Loading model: {args.model}")
    embedder = VisualEmbedder(
        model_name=args.model, processor_speed=str(getattr(args, "processor_speed", "fast"))
    )

    logger.info("🔌 Connecting to Qdrant")
    grpc_port = 6334 if args.prefer_grpc and urlparse(qdrant_url).port == 6333 else None
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=args.prefer_grpc,
        grpc_port=grpc_port,
        check_compatibility=False,
    )

    def _is_colqwen_model(model_name: str) -> bool:
        return "colqwen" in str(model_name).lower()

    exp_vector_name = "experimental_pooling"
    uses_experimental_vector = str(args.strategy) in (
        "single_experimental_tokens",
        "single_experimental_pooled",
    ) or (
        str(args.strategy) == "two_stage"
        and str(args.stage1_mode)
        in ("pooled_query_vs_experimental_pooling", "tokens_vs_experimental_pooling")
    )
    if (
        getattr(args, "experimental_pooling_technique", None)
        and getattr(args, "experimental_pooling_k", None) is not None
    ):
        raise SystemExit(
            "Use only one of --experimental-pooling-technique or --experimental-pooling-k."
        )

    if getattr(args, "experimental_pooling_technique", None):
        if not uses_experimental_vector:
            logger.warning(
                "--experimental-pooling-technique was provided but this strategy does not use experimental vectors; ignoring."
            )
        else:
            if not _is_colqwen_model(args.model):
                raise SystemExit(
                    "--experimental-pooling-technique is only supported for ColQwen models."
                )
            exp_vector_name = (
                f"experimental_pooling_{str(args.experimental_pooling_technique).strip().lower()}"
            )

    if getattr(args, "experimental_pooling_k", None) is not None:
        if _is_colqwen_model(args.model):
            raise SystemExit(
                "--experimental-pooling-k is intended for ColPali (experimental_pooling_{k}), not ColQwen."
            )
        if not uses_experimental_vector:
            logger.warning(
                "--experimental-pooling-k was provided but this strategy does not use experimental vectors; ignoring."
            )
        elif str(args.stage1_mode) in (
            "pooled_query_vs_experimental_pooling",
            "tokens_vs_experimental_pooling",
        ) or str(args.strategy) in ("single_experimental_tokens", "single_experimental_pooled"):
            exp_vector_name = f"experimental_pooling_{int(args.experimental_pooling_k)}"
        else:
            logger.warning(
                "--experimental-pooling-k was provided but stage1-mode is not experimental; ignoring."
            )

    two_stage = TwoStageRetriever(
        client, args.collection, experimental_vector_name=str(exp_vector_name)
    )
    single_stage = SingleStageRetriever(
        client, args.collection, experimental_vector_name=str(exp_vector_name)
    )

    if str(args.stage1_mode) in (
        "pooled_query_vs_experimental_pooling",
        "tokens_vs_experimental_pooling",
    ):
        try:
            info = client.get_collection(str(args.collection))
            vectors = info.config.params.vectors or {}
            existing = set(str(k) for k in vectors.keys()) if isinstance(vectors, dict) else set()
        except Exception:
            existing = set()
        if existing and exp_vector_name not in existing:
            candidates = sorted([v for v in existing if str(v).startswith("experimental_pooling")])
            raise SystemExit(
                f"Requested experimental vector '{exp_vector_name}' is not present in the collection. "
                f"Available experimental vectors: {candidates or '[]'}. "
                "Re-index (and --force-recreate) to add it."
            )

    # Embed query
    logger.info(f"🔍 Query: {args.query}")
    query_embedding = embedder.embed_query(args.query)

    # Build filter
    filter_obj = None
    if args.year or args.source or args.district:
        filter_obj = two_stage.build_filter(
            year=args.year,
            source=args.source,
            district=args.district,
        )

    # Search
    query_np = query_embedding.detach().cpu().float().numpy()  # .float() for BFloat16
    if args.strategy == "single_full":
        results = single_stage.search(
            query_embedding=query_np,
            top_k=args.top_k,
            strategy="multi_vector",
            filter_obj=filter_obj,
        )
    elif args.strategy == "single_tiles":
        results = single_stage.search(
            query_embedding=query_np,
            top_k=args.top_k,
            strategy="tiles_maxsim",
            filter_obj=filter_obj,
        )
    elif args.strategy == "single_global":
        results = single_stage.search(
            query_embedding=query_np,
            top_k=args.top_k,
            strategy="pooled_global",
            filter_obj=filter_obj,
        )
    elif args.strategy == "single_experimental_tokens":
        results = single_stage.search(
            query_embedding=query_np,
            top_k=args.top_k,
            strategy="experimental_maxsim",
            filter_obj=filter_obj,
        )
    elif args.strategy == "single_experimental_pooled":
        results = single_stage.search(
            query_embedding=query_np,
            top_k=args.top_k,
            strategy="pooled_experimental",
            filter_obj=filter_obj,
        )
    else:
        results = two_stage.search(
            query_embedding=query_np,
            top_k=args.top_k,
            prefetch_k=args.prefetch_k,
            filter_obj=filter_obj,
            stage1_mode=args.stage1_mode,
        )

    # Display results
    logger.info(f"\n📊 Results ({len(results)}):")
    for i, result in enumerate(results, 1):
        payload = result.get("payload", {})
        score = result.get("score_final", result.get("score_stage1", 0))

        filename = payload.get("filename", "N/A")
        page_num = payload.get("page_number", "N/A")
        year = payload.get("year", "N/A")
        source = payload.get("source", "N/A")

        logger.info(f"  {i}. {filename} p.{page_num}")
        logger.info(f"     Score: {score:.4f} | Year: {year} | Source: {source}")

        # Text snippet
        text = payload.get("text", "")
        if text and args.show_text:
            snippet = text[:200].replace("\n", " ")
            logger.info(f"     Text: {snippet}...")


def cmd_info(args):
    """Show collection info."""
    from qdrant_client import QdrantClient

    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url:
        logger.error("❌ QDRANT_URL not set")
        sys.exit(1)

    grpc_port = 6334 if args.prefer_grpc and urlparse(qdrant_url).port == 6333 else None
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=args.prefer_grpc,
        grpc_port=grpc_port,
        check_compatibility=False,
    )

    try:
        info = client.get_collection(args.collection)

        status = info.status
        if hasattr(status, "value"):
            status = status.value

        indexed_count = getattr(info, "indexed_vectors_count", 0) or 0
        if isinstance(indexed_count, dict):
            indexed_count = sum(indexed_count.values())

        logger.info(f"📊 Collection: {args.collection}")
        logger.info(f"   Status: {status}")
        logger.info(f"   Points: {info.points_count}")
        logger.info(f"   Indexed vectors: {indexed_count}")

        # Show vector config
        if hasattr(info, "config") and hasattr(info.config, "params"):
            vectors = getattr(info.config.params, "vectors", {})
            if vectors:
                logger.info(f"   Vectors: {list(vectors.keys())}")

    except Exception as e:
        logger.error(f"❌ Could not get collection info: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="visual-rag",
        description="Visual RAG Toolkit - Visual document retrieval with ColPali",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process PDFs (like process_pdfs_saliency_v2.py)
  visual-rag process --reports-dir ./pdfs --metadata-file metadata.json

  # Process without Cloudinary
  visual-rag process --reports-dir ./pdfs --no-cloudinary

  # Search
  visual-rag search --query "budget allocation" --collection my_docs

  # Search with filters
  visual-rag search --query "budget" --year 2023 --source "Local Government"

  # Show collection info
  visual-rag info --collection my_docs
        """,
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # =========================================================================
    # PROCESS command
    # =========================================================================
    process_parser = subparsers.add_parser(
        "process",
        help="Process PDFs: embed, upload to Cloudinary, index in Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    process_parser.add_argument(
        "--reports-dir", type=str, required=True, help="Directory containing PDF files"
    )
    process_parser.add_argument(
        "--metadata-file",
        type=str,
        help="JSON file with filename → metadata mapping (like filename_metadata.json)",
    )
    process_parser.add_argument(
        "--collection", type=str, default="visual_documents", help="Qdrant collection name"
    )
    process_parser.add_argument(
        "--model",
        type=str,
        default="vidore/colSmol-500M",
        help="Model name (vidore/colSmol-500M, vidore/colpali-v1.3, etc.)",
    )
    process_parser.add_argument("--batch-size", type=int, default=8, help="Embedding batch size")
    process_parser.add_argument("--config", type=str, help="Path to config.yaml file")
    process_parser.add_argument(
        "--no-cloudinary", action="store_true", help="Skip Cloudinary uploads"
    )
    process_parser.add_argument(
        "--crop-empty",
        action="store_true",
        help="Crop empty whitespace from page images before embedding (default: off).",
    )
    process_parser.add_argument(
        "--crop-empty-percentage-to-remove",
        type=float,
        default=0.9,
        help="Kept for traceability; currently does not affect cropping behavior (default: 0.9).",
    )
    process_parser.add_argument(
        "--crop-empty-remove-page-number",
        action="store_true",
        help="If set, attempts to crop away the bottom region that contains sparse page numbers (default: off).",
    )
    process_parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Process all pages even if they exist in Qdrant",
    )
    process_parser.add_argument(
        "--force-recreate", action="store_true", help="Delete and recreate collection"
    )
    process_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be processed without doing it"
    )
    process_parser.add_argument(
        "--strategy",
        type=str,
        default="pooling",
        choices=["pooling", "standard", "all"],
        help="Embedding strategy: 'pooling' (NOVEL), 'standard' (BASELINE), "
        "'all' (embed once, store BOTH for comparison)",
    )
    process_parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype for model weights (default: auto; CUDA->bfloat16, else float32).",
    )
    process_parser.add_argument(
        "--qdrant-vector-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Datatype for vectors stored in Qdrant (default: float16).",
    )
    process_parser.add_argument(
        "--max-mean-pool-vectors",
        type=int,
        default=32,
        help=(
            "Cap ColQwen2.5 adaptive row-mean pooling to at most this many vectors. "
            "Default: 32 (legacy behavior). If <= 0, treated as no cap."
        ),
    )
    process_parser.add_argument(
        "--pooling-windows",
        "--pooling_windows",
        type=int,
        nargs="+",
        default=None,
        help=(
            "ColPali only: experimental pooling window size(s). Provide one int to override the default window, "
            "or multiple ints to index/store multiple experimental vectors as "
            "'experimental_pooling_{k}' (and 'experimental_pooling' aliases the first provided k). "
            "Ignored for ColQwen2.5 (which stores gaussian+triangular variants)."
        ),
    )
    process_parser.add_argument(
        "--experimental-pooling-kernel",
        "--experimental_pooling_kernel",
        type=str,
        default="auto",
        choices=["auto", "legacy", "uniform", "triangular", "gaussian"],
        help=(
            "Experimental pooling kernel. "
            "'legacy' uses the historical ColPali conv-style pooling (N->N+2r; default for ColPali). "
            "'gaussian'/'triangular'/'uniform' use weighted same-length smoothing (N->N). "
            "Ignored for ColQwen2.5 (which stores gaussian+triangular variants with k=3)."
        ),
    )
    process_parser.add_argument(
        "--colsmol-experimental-2d",
        "--colsmol_experimental_2d",
        action="store_true",
        default=False,
        help="For ColSmol indexing, also store 2D 4-neighborhood experimental pooling as 'experimental_pooling_2d'.",
    )
    process_parser.add_argument(
        "--processor-speed",
        type=str,
        default="fast",
        choices=["fast", "slow", "auto"],
        help="Processor implementation: fast (default, with fallback to slow), slow, or auto.",
    )
    process_grpc_group = process_parser.add_mutually_exclusive_group()
    process_grpc_group.add_argument(
        "--prefer-grpc",
        dest="prefer_grpc",
        action="store_true",
        default=True,
        help="Use gRPC for Qdrant client (recommended).",
    )
    process_grpc_group.add_argument(
        "--no-prefer-grpc",
        dest="prefer_grpc",
        action="store_false",
        help="Disable gRPC for Qdrant client.",
    )
    process_parser.set_defaults(func=cmd_process)

    # =========================================================================
    # SEARCH command
    # =========================================================================
    search_parser = subparsers.add_parser(
        "search",
        help="Search documents",
    )
    search_parser.add_argument("--query", type=str, required=True, help="Search query")
    search_parser.add_argument(
        "--collection", type=str, default="visual_documents", help="Qdrant collection name"
    )
    search_parser.add_argument(
        "--model", type=str, default="vidore/colSmol-500M", help="Model name"
    )
    search_parser.add_argument(
        "--processor-speed",
        type=str,
        default="fast",
        choices=["fast", "slow", "auto"],
        help="Processor implementation: fast (default, with fallback to slow), slow, or auto.",
    )
    search_parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    search_parser.add_argument(
        "--strategy",
        type=str,
        default="single_full",
        choices=[
            "single_full",
            "single_tiles",
            "single_global",
            "single_experimental_tokens",
            "single_experimental_pooled",
            "two_stage",
        ],
        help="Search strategy",
    )
    search_parser.add_argument(
        "--prefetch-k", type=int, default=200, help="Prefetch candidates for two-stage retrieval"
    )
    search_parser.add_argument(
        "--stage1-mode",
        type=str,
        default="pooled_query_vs_standard_pooling",
        choices=[
            "pooled_query_vs_standard_pooling",
            "tokens_vs_standard_pooling",
            "pooled_query_vs_experimental_pooling",
            "tokens_vs_experimental_pooling",
            "pooled_query_vs_global",
            # Backwards-compatible aliases (deprecated)
            "pooled_query_vs_tiles",
            "tokens_vs_tiles",
            "pooled_query_vs_experimental",
            "tokens_vs_experimental",
        ],
        help="Stage 1 mode for two-stage retrieval",
    )
    search_parser.add_argument(
        "--experimental-pooling-k",
        "--experimental_pooling_k",
        type=int,
        default=None,
        help=(
            "ColPali only: when using an experimental stage1-mode, select which indexed experimental vector to use "
            "(Qdrant named vector: 'experimental_pooling_{k}'). If omitted, uses 'experimental_pooling'."
        ),
    )
    search_parser.add_argument(
        "--experimental-pooling-technique",
        "--experimental_pooling_technique",
        type=str,
        default=None,
        choices=["gaussian", "triangular"],
        help=(
            "ColQwen only: choose experimental pooling named vector for experimental strategies/stage-1. "
            "Maps to: 'experimental_pooling_gaussian' or 'experimental_pooling_triangular'. "
            "If omitted, uses 'experimental_pooling' (Gaussian alias)."
        ),
    )
    search_parser.add_argument("--year", type=int, help="Filter by year")
    search_parser.add_argument("--source", type=str, help="Filter by source")
    search_parser.add_argument("--district", type=str, help="Filter by district")
    search_parser.add_argument(
        "--show-text", action="store_true", help="Show text snippets in results"
    )
    search_grpc_group = search_parser.add_mutually_exclusive_group()
    search_grpc_group.add_argument(
        "--prefer-grpc",
        dest="prefer_grpc",
        action="store_true",
        default=True,
        help="Use gRPC for Qdrant client (recommended).",
    )
    search_grpc_group.add_argument(
        "--no-prefer-grpc",
        dest="prefer_grpc",
        action="store_false",
        help="Disable gRPC for Qdrant client.",
    )
    search_parser.set_defaults(func=cmd_search)

    # =========================================================================
    # INFO command
    # =========================================================================
    info_parser = subparsers.add_parser(
        "info",
        help="Show collection info",
    )
    info_parser.add_argument(
        "--collection", type=str, default="visual_documents", help="Qdrant collection name"
    )
    info_grpc_group = info_parser.add_mutually_exclusive_group()
    info_grpc_group.add_argument(
        "--prefer-grpc",
        dest="prefer_grpc",
        action="store_true",
        default=True,
        help="Use gRPC for Qdrant client (recommended).",
    )
    info_grpc_group.add_argument(
        "--no-prefer-grpc",
        dest="prefer_grpc",
        action="store_false",
        help="Disable gRPC for Qdrant client.",
    )
    info_parser.set_defaults(func=cmd_info)

    # Parse and execute
    args = parser.parse_args()

    setup_logging(args.debug)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
