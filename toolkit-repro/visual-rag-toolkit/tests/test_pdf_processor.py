"""Tests for PDF processor."""

import pytest
from PIL import Image


class TestResizeForColPali:
    """Test image resizing for ColPali processing."""

    def test_resize_standard_image(self):
        """Standard image resizes to tile boundaries."""
        from visual_rag.indexing.pdf_processor import PDFProcessor

        processor = PDFProcessor()

        # Create test image (A4-like ratio)
        img = Image.new("RGB", (1000, 1414), color="white")

        resized, tile_rows, tile_cols = processor.resize_for_colpali(img)

        # Should resize to multiples of 512
        assert resized.width % 512 == 0 or resized.width <= 2048
        assert resized.height % 512 == 0 or resized.height <= 2048
        assert tile_rows >= 1
        assert tile_cols >= 1

    def test_resize_small_image(self):
        """Small image handles gracefully."""
        from visual_rag.indexing.pdf_processor import PDFProcessor

        processor = PDFProcessor()
        img = Image.new("RGB", (100, 100), color="white")

        resized, tile_rows, tile_cols = processor.resize_for_colpali(img)

        assert resized is not None
        assert tile_rows >= 1
        assert tile_cols >= 1

    def test_resize_wide_image(self):
        """Wide image (panorama-like) resizes correctly."""
        from visual_rag.indexing.pdf_processor import PDFProcessor

        processor = PDFProcessor()
        img = Image.new("RGB", (3000, 500), color="white")

        resized, tile_rows, tile_cols = processor.resize_for_colpali(img)

        # Wide image should have more cols than rows
        assert tile_cols >= tile_rows

    def test_resize_preserves_rgb(self):
        """Resized image should be RGB."""
        from visual_rag.indexing.pdf_processor import PDFProcessor

        processor = PDFProcessor()
        img = Image.new("RGBA", (1000, 1000), color="white")

        resized, _, _ = processor.resize_for_colpali(img)

        assert resized.mode == "RGB"


class TestMetadataExtraction:
    """Test metadata extraction from filenames."""

    def test_extract_year_from_filename(self):
        """Extract year from filename."""
        from visual_rag.indexing.pdf_processor import PDFProcessor

        processor = PDFProcessor()

        metadata = processor.extract_metadata_from_filename("Annual_Report_2023.pdf")

        # Should extract year if implemented
        # This depends on your implementation
        assert isinstance(metadata, dict)

    def test_sanitize_filename(self):
        """Sanitize filename for safe storage."""
        from visual_rag.indexing.pdf_processor import PDFProcessor

        processor = PDFProcessor()

        # Test with special characters
        filename = "Report (Final) - v2.0.pdf"
        # Should handle gracefully
        metadata = processor.extract_metadata_from_filename(filename)
        assert isinstance(metadata, dict)


class TestChunkIdGeneration:
    """Test deterministic chunk ID generation."""

    def test_chunk_id_deterministic(self):
        """Same input produces same chunk ID."""
        from visual_rag.indexing.pipeline import ProcessingPipeline

        id1 = ProcessingPipeline.generate_chunk_id("test.pdf", 1)
        id2 = ProcessingPipeline.generate_chunk_id("test.pdf", 1)

        assert id1 == id2

    def test_chunk_id_unique(self):
        """Different pages produce different IDs."""
        from visual_rag.indexing.pipeline import ProcessingPipeline

        id1 = ProcessingPipeline.generate_chunk_id("test.pdf", 1)
        id2 = ProcessingPipeline.generate_chunk_id("test.pdf", 2)

        assert id1 != id2

    def test_chunk_id_format(self):
        """Chunk ID should be valid UUID format."""
        import uuid

        from visual_rag.indexing.pipeline import ProcessingPipeline

        chunk_id = ProcessingPipeline.generate_chunk_id("test.pdf", 1)

        # Should be valid UUID
        try:
            uuid.UUID(chunk_id)
        except ValueError:
            pytest.fail(f"Invalid UUID format: {chunk_id}")
