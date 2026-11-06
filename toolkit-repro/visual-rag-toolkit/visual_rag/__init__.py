"""
Visual RAG Toolkit - End-to-end visual document retrieval with two-stage pooling.

A modular toolkit for building visual document retrieval systems:

Components:
-----------
- embedding: Visual and text embedding generation (ColPali, etc.)
- indexing: PDF processing, Qdrant indexing, Cloudinary uploads
- retrieval: Single-stage and two-stage retrieval with MaxSim
- visualization: Saliency maps and attention visualization
- cli: Command-line interface

Quick Start:
------------
>>> from visual_rag import VisualEmbedder, PDFProcessor, TwoStageRetriever
>>>
>>> # Process PDFs
>>> processor = PDFProcessor(dpi=140)
>>> images, texts = processor.process_pdf("report.pdf")
>>>
>>> # Generate embeddings
>>> embedder = VisualEmbedder()
>>> embeddings = embedder.embed_images(images)
>>> query_emb = embedder.embed_query("What is the budget?")
>>>
>>> # Search with two-stage retrieval
>>> retriever = TwoStageRetriever(qdrant_client, "my_collection")
>>> results = retriever.search(query_emb, top_k=10)

Each component works independently - use only what you need.
"""

import logging

__version__ = "0.2.0"


def setup_logging(level: str = "INFO", format: str = None) -> None:
    """
    Configure logging for visual_rag package.

    Args:
        level: Log level ("DEBUG", "INFO", "WARNING", "ERROR")
        format: Custom format string. Default shows time, level, and message.

    Example:
        >>> import visual_rag
        >>> visual_rag.setup_logging("INFO")
        >>> # Now you'll see processing logs
    """
    if format is None:
        format = "[%(asctime)s] %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=format,
        datefmt="%H:%M:%S",
    )

    # Also set the visual_rag logger specifically
    logger = logging.getLogger("visual_rag")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))


# Enable INFO logging by default for visual_rag package and all submodules
# This ensures logs like "Processing PDF...", "Embedding pages..." are visible
_logger = logging.getLogger("visual_rag")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    _logger.addHandler(_handler)
_logger.setLevel(logging.INFO)
_logger.propagate = False  # Don't duplicate to root logger

# Import main classes at package level for convenience
# These are optional - if dependencies aren't installed, we catch the error

try:
    from visual_rag.embedding.visual_embedder import VisualEmbedder
except ImportError:
    VisualEmbedder = None

try:
    from visual_rag.indexing.pdf_processor import PDFProcessor
except ImportError:
    PDFProcessor = None

try:
    from visual_rag.indexing.qdrant_indexer import QdrantIndexer
except ImportError:
    QdrantIndexer = None

try:
    from visual_rag.indexing.cloudinary_uploader import CloudinaryUploader
except ImportError:
    CloudinaryUploader = None

try:
    from visual_rag.retrieval.two_stage import TwoStageRetriever
except ImportError:
    TwoStageRetriever = None

try:
    from visual_rag.retrieval.multi_vector import MultiVectorRetriever
except ImportError:
    MultiVectorRetriever = None

try:
    from visual_rag.qdrant_admin import QdrantAdmin
except ImportError:
    QdrantAdmin = None

# demo is lazily imported to avoid RuntimeWarning when running as __main__
# Access via visual_rag.demo() which triggers __getattr__

# Config utilities (always available)
try:
    from visual_rag.config import get, get_section, load_config
except ImportError:
    get = None
    get_section = None
    load_config = None

__all__ = [
    # Version
    "__version__",
    # Main classes
    "VisualEmbedder",
    "PDFProcessor",
    "QdrantIndexer",
    "CloudinaryUploader",
    "TwoStageRetriever",
    "MultiVectorRetriever",
    "QdrantAdmin",
    "demo",
    # Config utilities
    "load_config",
    "get",
    "get_section",
    # Logging
    "setup_logging",
]


def __getattr__(name: str):
    """Lazy import for demo to avoid RuntimeWarning when running as __main__."""
    if name == "demo":
        try:
            from visual_rag.demo_runner import demo

            return demo
        except ImportError:
            return None
    raise AttributeError(f"module 'visual_rag' has no attribute {name!r}")
