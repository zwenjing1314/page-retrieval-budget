#!/usr/bin/env python3
"""
Example: Process PDFs with Visual RAG Toolkit

This example demonstrates the full pipeline:
1. Load configuration
2. Initialize components
3. Process PDFs with metadata
4. Store in Qdrant with saliency metadata

Usage:
    python examples/process_pdfs.py --reports-dir /path/to/pdfs

    # With metadata mapping
    python examples/process_pdfs.py --reports-dir /path/to/pdfs --metadata-file metadata.json

    # Without Cloudinary (local embeddings only)
    python examples/process_pdfs.py --reports-dir /path/to/pdfs --no-cloudinary
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from visual_rag import CloudinaryUploader, QdrantIndexer, VisualEmbedder, load_config  # noqa: E402
from visual_rag.indexing.pipeline import ProcessingPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Process PDFs with Visual RAG Toolkit")
    parser.add_argument(
        "--reports-dir", type=str, required=True, help="Directory containing PDF files"
    )
    parser.add_argument(
        "--metadata-file", type=str, help="JSON file with filename → metadata mapping (optional)"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file path")
    parser.add_argument("--collection", type=str, help="Qdrant collection name (overrides config)")
    parser.add_argument("--model", type=str, help="Model name (overrides config)")
    parser.add_argument("--no-cloudinary", action="store_true", help="Skip Cloudinary uploads")
    parser.add_argument(
        "--no-qdrant", action="store_true", help="Skip Qdrant uploads (just generate embeddings)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip pages that already exist in Qdrant (default: True)",
    )
    parser.add_argument("--force", action="store_true", help="Process all pages even if they exist")

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Load configuration
    config = load_config(args.config)

    # Get PDFs
    reports_dir = Path(args.reports_dir)
    if not reports_dir.exists():
        logger.error(f"Reports directory not found: {reports_dir}")
        sys.exit(1)

    pdf_paths = sorted(reports_dir.glob("*.pdf")) + sorted(reports_dir.glob("*.PDF"))
    if not pdf_paths:
        logger.error(f"No PDF files found in: {reports_dir}")
        sys.exit(1)

    logger.info(f"📁 Found {len(pdf_paths)} PDF files in {reports_dir}")

    # Load metadata mapping if provided
    metadata_mapping = {}
    if args.metadata_file:
        metadata_mapping = ProcessingPipeline.load_metadata_mapping(Path(args.metadata_file))

    # Get settings
    model_name = args.model or config.get("model", {}).get("name", "vidore/colSmol-500M")
    collection_name = args.collection or config.get("qdrant", {}).get(
        "collection_name", "visual_documents"
    )

    # Initialize embedder
    logger.info(f"🤖 Initializing embedder: {model_name}")
    embedder = VisualEmbedder(model_name=model_name)

    # Initialize Qdrant indexer (if not skipped)
    indexer = None
    if not args.no_qdrant:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url:
            logger.error("QDRANT_URL environment variable not set")
            sys.exit(1)

        logger.info(f"🔌 Connecting to Qdrant: {qdrant_url}")
        indexer = QdrantIndexer(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
        )

        # Create collection if needed
        indexer.create_collection()
        indexer.create_payload_indexes()

    # Initialize Cloudinary uploader (if not skipped)
    cloudinary_uploader = None
    if not args.no_cloudinary:
        try:
            cloudinary_uploader = CloudinaryUploader(
                folder=config.get("project_name", "visual_docs"),
            )
        except ValueError as e:
            logger.warning(f"Cloudinary not configured: {e}")
            logger.warning("Continuing without Cloudinary uploads")

    # Create pipeline
    pipeline = ProcessingPipeline(
        embedder=embedder,
        indexer=indexer,
        cloudinary_uploader=cloudinary_uploader,
        metadata_mapping=metadata_mapping,
        config=config,
    )

    # Process PDFs
    total_uploaded = 0
    total_skipped = 0
    total_failed = 0

    skip_existing = args.skip_existing and not args.force

    for pdf_idx, pdf_path in enumerate(pdf_paths, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"📄 [{pdf_idx}/{len(pdf_paths)}] {pdf_path.name}")
        logger.info(f"{'='*60}")

        result = pipeline.process_pdf(
            pdf_path,
            skip_existing=skip_existing,
            upload_to_cloudinary=(not args.no_cloudinary),
            upload_to_qdrant=(not args.no_qdrant),
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

    if indexer:
        info = indexer.get_collection_info()
        if info:
            logger.info(f"   Collection points: {info.get('points_count', 'N/A')}")


if __name__ == "__main__":
    main()
