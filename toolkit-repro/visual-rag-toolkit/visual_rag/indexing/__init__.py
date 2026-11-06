"""
Indexing module - PDF processing, embedding storage, and CDN uploads.

Components:
- PDFProcessor: Convert PDFs to images and extract text
- QdrantIndexer: Upload embeddings to Qdrant vector database
- CloudinaryUploader: Upload images to Cloudinary CDN
- ProcessingPipeline: End-to-end PDF → Qdrant pipeline
"""

# Lazy imports to avoid failures when optional dependencies aren't installed

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
    from visual_rag.indexing.pipeline import ProcessingPipeline
except ImportError:
    ProcessingPipeline = None

__all__ = [
    "PDFProcessor",
    "QdrantIndexer",
    "CloudinaryUploader",
    "ProcessingPipeline",
]
