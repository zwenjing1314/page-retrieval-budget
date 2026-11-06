"""
PDF Processor - Convert PDFs to images and extract text.

This module works INDEPENDENTLY of embedding and vector storage.
Use it if you just need PDF → images conversion.

Features:
- Batch processing to save memory
- Text extraction with surrogate character handling
- Configurable DPI and quality settings
"""

import gc
import logging
import re
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Process PDFs into images and text for visual retrieval.

    Works independently - no embedding or storage dependencies.

    Args:
        dpi: DPI for image conversion (higher = better quality)
        output_format: Image format (RGB, L, etc.)
        page_batch_size: Pages per batch for memory efficiency

    Example:
        >>> processor = PDFProcessor(dpi=140)
        >>>
        >>> # Convert single PDF
        >>> images, texts = processor.process_pdf(Path("report.pdf"))
        >>>
        >>> # Stream large PDFs
        >>> for images, texts in processor.stream_pdf(Path("large.pdf"), batch_size=10):
        ...     # Process each batch
        ...     pass
    """

    def __init__(
        self,
        dpi: int = 140,
        output_format: str = "RGB",
        page_batch_size: int = 50,
    ):
        self.dpi = dpi
        self.output_format = output_format
        self.page_batch_size = page_batch_size

        # PDF deps are optional: we only require them when calling PDF-specific methods.
        # This keeps the class usable for helper utilities like `resize_for_colpali()`
        # even in minimal installs.
        self._pdf_deps_available = True
        try:
            import pdf2image  # noqa: F401
            import pypdf  # noqa: F401
        except Exception:
            self._pdf_deps_available = False

    def _require_pdf_deps(self) -> None:
        if not self._pdf_deps_available:
            raise ImportError(
                "PDF processing requires `pdf2image` and `pypdf`.\n"
                'Install with: pip install "visual-rag-toolkit[pdf]"'
            )

    def process_pdf(
        self,
        pdf_path: Path,
        dpi: Optional[int] = None,
    ) -> Tuple[List[Image.Image], List[str]]:
        """
        Convert PDF to images and extract text.

        Args:
            pdf_path: Path to PDF file
            dpi: Override default DPI

        Returns:
            Tuple of (list of images, list of page texts)
        """
        self._require_pdf_deps()
        from pdf2image import convert_from_path
        from pypdf import PdfReader

        dpi = dpi or self.dpi
        pdf_path = Path(pdf_path)

        logger.info(f"📄 Processing PDF: {pdf_path.name}")

        # Extract text
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)

        page_texts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            # Handle surrogate characters
            text = self._sanitize_text(text)
            page_texts.append(text)

        # Convert to images in batches
        all_images = []
        for start_page in range(1, total_pages + 1, self.page_batch_size):
            end_page = min(start_page + self.page_batch_size - 1, total_pages)

            batch_images = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                fmt=self.output_format.lower(),
                first_page=start_page,
                last_page=end_page,
            )

            all_images.extend(batch_images)

            del batch_images
            gc.collect()

        assert len(all_images) == len(
            page_texts
        ), f"Mismatch: {len(all_images)} images vs {len(page_texts)} texts"

        logger.info(f"✅ Processed {len(all_images)} pages")
        return all_images, page_texts

    def stream_pdf(
        self,
        pdf_path: Path,
        batch_size: int = 10,
        dpi: Optional[int] = None,
    ) -> Generator[Tuple[List[Image.Image], List[str], int], None, None]:
        """
        Stream PDF processing for large files.

        Yields batches of (images, texts, start_page) without loading
        entire PDF into memory.

        Args:
            pdf_path: Path to PDF file
            batch_size: Pages per batch
            dpi: Override default DPI

        Yields:
            Tuple of (batch_images, batch_texts, start_page_number)
        """
        self._require_pdf_deps()
        from pdf2image import convert_from_path
        from pypdf import PdfReader

        dpi = dpi or self.dpi
        pdf_path = Path(pdf_path)

        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)

        logger.info(f"📄 Streaming PDF: {pdf_path.name} ({total_pages} pages)")

        for start_idx in range(0, total_pages, batch_size):
            end_idx = min(start_idx + batch_size, total_pages)

            # Extract text for batch
            batch_texts = []
            for page_idx in range(start_idx, end_idx):
                text = reader.pages[page_idx].extract_text() or ""
                text = self._sanitize_text(text)
                batch_texts.append(text)

            # Convert images for batch
            batch_images = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                fmt=self.output_format.lower(),
                first_page=start_idx + 1,  # 1-indexed
                last_page=end_idx,
            )

            yield batch_images, batch_texts, start_idx + 1

            del batch_images
            gc.collect()

    def get_page_count(self, pdf_path: Path) -> int:
        """Get number of pages in PDF without loading images."""
        self._require_pdf_deps()
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return len(reader.pages)

    def resize_for_colpali(
        self,
        image: Image.Image,
        max_edge: int = 2048,
        tile_size: int = 512,
    ) -> Tuple[Image.Image, int, int]:
        """
        Resize image following ColPali/Idefics3 processor logic.

        Resizes to fit within tile grid without black padding.

        Args:
            image: PIL Image
            max_edge: Maximum edge length
            tile_size: Size of each tile

        Returns:
            Tuple of (resized_image, tile_rows, tile_cols)
        """
        # Ensure consistent mode for downstream processors (and predictable tests)
        if image.mode != "RGB":
            image = image.convert("RGB")

        w, h = image.size

        # Step 1: Resize so longest edge = max_edge
        if w > h:
            new_w = max_edge
            new_h = int(h * (max_edge / w))
        else:
            new_h = max_edge
            new_w = int(w * (max_edge / h))

        # Step 2: Calculate tile grid
        tile_cols = (new_w + tile_size - 1) // tile_size
        tile_rows = (new_h + tile_size - 1) // tile_size

        # Step 3: Calculate exact dimensions for tiles
        final_w = tile_cols * tile_size
        final_h = tile_rows * tile_size

        # Step 4: Scale to fit within tile grid
        scale_w = final_w / w
        scale_h = final_h / h
        scale = min(scale_w, scale_h)

        scaled_w = int(w * scale)
        scaled_h = int(h * scale)

        resized = image.resize((scaled_w, scaled_h), Image.LANCZOS)

        # Center on white canvas if needed
        if scaled_w != final_w or scaled_h != final_h:
            canvas = Image.new("RGB", (final_w, final_h), (255, 255, 255))
            offset_x = (final_w - scaled_w) // 2
            offset_y = (final_h - scaled_h) // 2
            canvas.paste(resized, (offset_x, offset_y))
            resized = canvas

        return resized, tile_rows, tile_cols

    def _sanitize_text(self, text: str) -> str:
        """Remove invalid Unicode characters (surrogates) from text."""
        if not text:
            return ""

        # Remove surrogate characters (U+D800-U+DFFF)
        return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")

    def extract_metadata_from_filename(
        self,
        filename: str,
        mapping: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Extract metadata from PDF filename.

        Uses mapping if provided, otherwise falls back to pattern matching.

        Args:
            filename: PDF filename (with or without .pdf extension)
            mapping: Optional mapping dict {filename: metadata}

        Returns:
            Metadata dict with year, source, district, etc.
        """
        # Remove extension
        stem = Path(filename).stem
        stem_lower = stem.lower().strip()

        # Try mapping first
        if mapping:
            if stem_lower in mapping:
                return mapping[stem_lower].copy()

            # Try without .pdf
            stem_no_ext = stem_lower.replace(".pdf", "")
            if stem_no_ext in mapping:
                return mapping[stem_no_ext].copy()

        # Fallback: pattern matching
        metadata = {"filename": filename}

        # Extract year
        year_match = re.search(r"(20\d{2})", stem)
        if year_match:
            metadata["year"] = int(year_match.group(1))

        # Detect source type
        if "consolidated" in stem_lower or ("annual" in stem_lower and "oag" in stem_lower):
            metadata["source"] = "Consolidated"
        elif "dlg" in stem_lower or "district local government" in stem_lower:
            metadata["source"] = "Local Government"
            # Try to extract district name
            district_match = re.search(r"([a-z]+)\s+(?:dlg|district local government)", stem_lower)
            if district_match:
                metadata["district"] = district_match.group(1).title()
        elif "hospital" in stem_lower or "referral" in stem_lower:
            metadata["source"] = "Hospital"
        elif "ministry" in stem_lower:
            metadata["source"] = "Ministry"
        elif "project" in stem_lower:
            metadata["source"] = "Project"
        else:
            metadata["source"] = "Unknown"

        return metadata
