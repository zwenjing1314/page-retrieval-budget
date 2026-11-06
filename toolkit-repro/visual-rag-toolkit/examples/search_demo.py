#!/usr/bin/env python3
"""
Example: Search with Visual RAG Toolkit

This example demonstrates:
1. Two-stage retrieval (pooled prefetch → MaxSim rerank)
2. Optional saliency map generation
3. Metadata filtering

Usage:
    python examples/search_demo.py --query "What is the budget allocation?"

    # With filters
    python examples/search_demo.py --query "budget" --year 2023 --source "Local Government"

    # With saliency maps
    python examples/search_demo.py --query "budget" --saliency
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from visual_rag import VisualEmbedder  # noqa: E402
from visual_rag.retrieval.two_stage import TwoStageRetriever  # noqa: E402
from visual_rag.visualization import visualize_search_results  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Search with Visual RAG Toolkit")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument(
        "--collection", type=str, default="visual_documents", help="Qdrant collection name"
    )
    parser.add_argument("--model", type=str, default="vidore/colSmol-500M", help="Model name")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument(
        "--prefetch-k", type=int, default=200, help="Candidates for two-stage retrieval"
    )
    parser.add_argument("--year", type=int, help="Filter by year")
    parser.add_argument("--source", type=str, help="Filter by source")
    parser.add_argument("--district", type=str, help="Filter by district")
    parser.add_argument(
        "--saliency", action="store_true", help="Generate saliency maps for results"
    )
    parser.add_argument("--output", type=str, help="Output path for visualization")

    args = parser.parse_args()

    # Load environment
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url:
        logger.error("QDRANT_URL not set")
        sys.exit(1)

    # Initialize components
    logger.info(f"🤖 Loading model: {args.model}")
    embedder = VisualEmbedder(model_name=args.model)

    logger.info(f"🔌 Connecting to Qdrant: {qdrant_url}")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    retriever = TwoStageRetriever(
        qdrant_client=client,
        collection_name=args.collection,
    )

    # Embed query
    logger.info(f"🔍 Query: {args.query}")
    query_embedding = embedder.embed_query(args.query)

    # Build filter
    filter_obj = retriever.build_filter(
        year=args.year,
        source=args.source,
        district=args.district,
    )

    # Search
    results = retriever.search(
        query_embedding=query_embedding.numpy(),
        top_k=args.top_k,
        prefetch_k=args.prefetch_k,
        filter_obj=filter_obj,
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

        # Show text snippet
        text = payload.get("text", "")
        if text:
            snippet = text[:200].replace("\n", " ")
            logger.info(f"     Text: {snippet}...")

    # Visualize results
    if args.output or args.saliency:
        output_path = args.output or "search_results.png"

        logger.info(f"\n🎨 Generating visualization: {output_path}")
        visualize_search_results(
            query=args.query,
            results=results,
            output_path=output_path,
        )
        logger.info(f"   ✅ Saved to {output_path}")


if __name__ == "__main__":
    main()
