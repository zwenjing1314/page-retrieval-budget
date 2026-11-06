"""
Processing Pipeline - Complete PDF → Qdrant pipeline with saliency metadata.

This module combines all components for end-to-end processing:
- PDF → images conversion
- Image resizing for ColPali
- Embedding generation with token info
- Tile-level pooling
- Cloudinary upload (optional)
- Qdrant indexing with full metadata for saliency maps

The metadata stored includes everything needed for saliency visualization:
- Tile structure (num_tiles, tile_rows, tile_cols, patches_per_tile)
- Image dimensions (original and resized)
- Token info (num_visual_tokens, visual_token_indices)
"""

import gc
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """
    End-to-end pipeline for PDF processing and indexing.

    This pipeline:
    1. Converts PDFs to images
    2. Resizes for ColPali processing
    3. Generates embeddings with token info
    4. Computes pooling (strategy-dependent)
    5. Uploads images to Cloudinary (optional)
    6. Stores in Qdrant with full saliency metadata

    Args:
        embedder: VisualEmbedder instance
        indexer: QdrantIndexer instance (optional)
        cloudinary_uploader: CloudinaryUploader instance (optional)
        pdf_processor: PDFProcessor instance (optional, auto-created)
        metadata_mapping: Dict mapping filenames to extra metadata
        config: Configuration dict
        embedding_strategy: How to process embeddings before storing:
            - "pooling" (default): Extract visual tokens only, compute tile-level pooling
              This is our NOVEL contribution - preserves spatial structure while reducing size.
            - "standard": Push ALL tokens as-is (including special tokens, padding)
              This is the baseline approach for comparison.

    Example:
        >>> from visual_rag import VisualEmbedder, QdrantIndexer, CloudinaryUploader
        >>> from visual_rag.indexing.pipeline import ProcessingPipeline
        >>>
        >>> # Our novel pooling strategy (default)
        >>> pipeline = ProcessingPipeline(
        ...     embedder=VisualEmbedder(),
        ...     indexer=QdrantIndexer(url, api_key, "my_collection"),
        ...     embedding_strategy="pooling",  # Visual tokens only + tile pooling
        ... )
        >>>
        >>> # Standard baseline (all tokens, no filtering)
        >>> pipeline_baseline = ProcessingPipeline(
        ...     embedder=VisualEmbedder(),
        ...     indexer=QdrantIndexer(url, api_key, "my_collection_baseline"),
        ...     embedding_strategy="standard",  # All tokens as-is
        ... )
        >>>
        >>> pipeline.process_pdf(Path("report.pdf"))
    """

    # Valid embedding strategies
    # - "pooling": Visual tokens only + tile-level pooling (NOVEL)
    # - "standard": All tokens + global mean (BASELINE)
    # - "all": Embed once, push BOTH representations (efficient comparison)
    STRATEGIES = ["pooling", "standard", "all"]

    def __init__(
        self,
        embedder=None,
        indexer=None,
        cloudinary_uploader=None,
        pdf_processor=None,
        metadata_mapping: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
        embedding_strategy: str = "pooling",
        crop_empty: bool = False,
        crop_empty_percentage_to_remove: float = 0.9,
        crop_empty_remove_page_number: bool = False,
        crop_empty_preserve_border_px: int = 1,
        crop_empty_uniform_rowcol_std_threshold: float = 0.0,
        max_mean_pool_vectors: Optional[int] = 32,
        pooling_windows: Optional[List[int]] = None,
        experimental_pooling_kernel: str = "auto",
        colsmol_experimental_2d: bool = False,
    ):
        self.embedder = embedder
        self.indexer = indexer
        self.cloudinary_uploader = cloudinary_uploader
        self.metadata_mapping = metadata_mapping or {}
        self.config = config or {}

        # Validate and set embedding strategy
        if embedding_strategy not in self.STRATEGIES:
            raise ValueError(
                f"Invalid embedding_strategy: {embedding_strategy}. "
                f"Must be one of: {self.STRATEGIES}"
            )
        self.embedding_strategy = embedding_strategy

        self.crop_empty = bool(crop_empty)
        self.crop_empty_percentage_to_remove = float(crop_empty_percentage_to_remove)
        self.crop_empty_remove_page_number = bool(crop_empty_remove_page_number)
        self.crop_empty_preserve_border_px = int(crop_empty_preserve_border_px)
        self.crop_empty_uniform_rowcol_std_threshold = float(
            crop_empty_uniform_rowcol_std_threshold
        )

        self.max_mean_pool_vectors = max_mean_pool_vectors
        self.pooling_windows = pooling_windows
        self.experimental_pooling_kernel = str(experimental_pooling_kernel or "auto")
        self.colsmol_experimental_2d = bool(colsmol_experimental_2d)

        logger.info(f"📊 Embedding strategy: {embedding_strategy}")
        if embedding_strategy == "pooling":
            logger.info("   → Visual tokens only + tile-level mean pooling (NOVEL)")
        else:
            logger.info("   → All tokens as-is (BASELINE)")

        # Create PDF processor if not provided
        if pdf_processor is None:
            from visual_rag.indexing.pdf_processor import PDFProcessor

            dpi = self.config.get("processing", {}).get("dpi", 140)
            pdf_processor = PDFProcessor(dpi=dpi)
        self.pdf_processor = pdf_processor

        # Config defaults
        self.embedding_batch_size = self.config.get("batching", {}).get("embedding_batch_size", 8)
        self.upload_batch_size = self.config.get("batching", {}).get("upload_batch_size", 8)
        self.delay_between_uploads = self.config.get("delays", {}).get("between_uploads", 0.5)

    def process_pdf(
        self,
        pdf_path: Path,
        skip_existing: bool = True,
        upload_to_cloudinary: bool = True,
        upload_to_qdrant: bool = True,
        original_filename: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Process a single PDF end-to-end.

        Args:
            pdf_path: Path to PDF file
            skip_existing: Skip pages that already exist in Qdrant
            upload_to_cloudinary: Upload images to Cloudinary
            upload_to_qdrant: Upload embeddings to Qdrant
            original_filename: Original filename (use this instead of pdf_path.name for temp files)
            progress_callback: Optional callback(stage, current, total, message) for progress updates

        Returns:
            Dict with processing results:
            {
                "filename": str,
                "total_pages": int,
                "uploaded": int,
                "skipped": int,
                "failed": int,
                "pages": [...],  # Page data with embeddings and metadata
            }
        """
        pdf_path = Path(pdf_path)
        filename = original_filename or pdf_path.name
        logger.info(f"📚 Processing PDF: {filename}")

        # Check existing pages
        existing_ids: Set[str] = set()
        if skip_existing and self.indexer:
            existing_ids = self.indexer.get_existing_ids(filename)
            if existing_ids:
                logger.info(f"   Found {len(existing_ids)} existing pages")

        logger.info("🖼️ Converting PDF to images...")
        if progress_callback:
            progress_callback("convert", 0, 0, "Converting PDF to images...")
        images, texts = self.pdf_processor.process_pdf(pdf_path)
        total_pages = len(images)
        logger.info(f"   ✅ Converted {total_pages} pages")
        if progress_callback:
            progress_callback("convert", total_pages, total_pages, f"Converted {total_pages} pages")

        extra_metadata = self._get_extra_metadata(filename)
        if extra_metadata:
            logger.info(f"   📋 Found extra metadata: {list(extra_metadata.keys())}")

        # Process in batches
        uploaded = 0
        skipped = 0
        failed = 0
        all_pages = []
        upload_queue = []

        for batch_start in range(0, total_pages, self.embedding_batch_size):
            batch_end = min(batch_start + self.embedding_batch_size, total_pages)
            batch_images = images[batch_start:batch_end]
            batch_texts = texts[batch_start:batch_end]

            logger.info(f"📦 Processing pages {batch_start + 1}-{batch_end}/{total_pages}")
            if progress_callback:
                progress_callback(
                    "embed",
                    batch_start,
                    total_pages,
                    f"Embedding pages {batch_start + 1}-{batch_end}",
                )

            pages_to_process = []
            for i, (img, text) in enumerate(zip(batch_images, batch_texts)):
                page_num = batch_start + i + 1
                chunk_id = self.generate_chunk_id(filename, page_num)

                if skip_existing and chunk_id in existing_ids:
                    skipped += 1
                    continue

                pages_to_process.append(
                    {
                        "index": i,
                        "page_num": page_num,
                        "chunk_id": chunk_id,
                        "raw_image": img,
                        "text": text,
                    }
                )

            if not pages_to_process:
                logger.info("   All pages in batch exist, skipping...")
                continue

            # Generate embeddings with token info
            logger.info(f"🤖 Generating embeddings for {len(pages_to_process)} pages...")
            from visual_rag.preprocessing.crop_empty import CropEmptyConfig, crop_empty

            images_to_embed = []
            for p in pages_to_process:
                raw_img = p["raw_image"]
                if self.crop_empty:
                    cropped_img, crop_meta = crop_empty(
                        raw_img,
                        config=CropEmptyConfig(
                            percentage_to_remove=float(self.crop_empty_percentage_to_remove),
                            remove_page_number=bool(self.crop_empty_remove_page_number),
                            preserve_border_px=int(self.crop_empty_preserve_border_px),
                            uniform_rowcol_std_threshold=float(
                                self.crop_empty_uniform_rowcol_std_threshold
                            ),
                        ),
                    )
                    p["embed_image"] = cropped_img
                    p["crop_meta"] = crop_meta
                    images_to_embed.append(cropped_img)
                else:
                    p["embed_image"] = raw_img
                    p["crop_meta"] = None
                    images_to_embed.append(raw_img)

            embeddings, token_infos = self.embedder.embed_images(
                images_to_embed,
                batch_size=self.embedding_batch_size,
                return_token_info=True,
                show_progress=True,
            )

            for idx, page_info in enumerate(pages_to_process):
                raw_img = page_info["raw_image"]
                embed_img = page_info["embed_image"]
                crop_meta = page_info["crop_meta"]
                page_num = page_info["page_num"]
                chunk_id = page_info["chunk_id"]
                text = page_info["text"]
                embedding = embeddings[idx]
                token_info = token_infos[idx]

                if progress_callback:
                    progress_callback(
                        "process",
                        page_num,
                        total_pages,
                        f"Processing page {page_num}/{total_pages}",
                    )

                try:
                    page_data = self._process_single_page(
                        filename=filename,
                        pdf_stem=pdf_path.stem,
                        page_num=page_num,
                        chunk_id=chunk_id,
                        total_pages=total_pages,
                        raw_img=raw_img,
                        embed_img=embed_img,
                        text=text,
                        embedding=embedding,
                        token_info=token_info,
                        extra_metadata=extra_metadata,
                        upload_to_cloudinary=upload_to_cloudinary,
                        crop_meta=crop_meta,
                    )

                    all_pages.append(page_data)

                    if upload_to_qdrant and self.indexer:
                        upload_queue.append(page_data)

                        # Upload in batches
                        if len(upload_queue) >= self.upload_batch_size:
                            count = self._upload_batch(upload_queue)
                            uploaded += count
                            upload_queue = []

                except Exception as e:
                    logger.error(f"   ❌ Failed page {page_num}: {e}")
                    failed += 1

            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Upload remaining pages
        if upload_queue and upload_to_qdrant and self.indexer:
            count = self._upload_batch(upload_queue)
            uploaded += count

        logger.info(
            f"✅ Completed {filename}: {uploaded} uploaded, {skipped} skipped, {failed} failed"
        )

        return {
            "filename": filename,
            "total_pages": total_pages,
            "uploaded": uploaded,
            "skipped": skipped,
            "failed": failed,
            "pages": all_pages,
        }

    def _process_single_page(
        self,
        filename: str,
        pdf_stem: str,
        page_num: int,
        chunk_id: str,
        total_pages: int,
        raw_img,
        embed_img,
        text: str,
        embedding: torch.Tensor,
        token_info: Dict[str, Any],
        extra_metadata: Dict[str, Any],
        upload_to_cloudinary: bool = True,
        crop_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process a single page with full metadata for saliency."""
        from visual_rag.embedding.pooling import global_mean_pooling

        # Resize image for ColPali
        resized_img, tile_rows, tile_cols = self.pdf_processor.resize_for_colpali(embed_img)

        # Use processor's tile info if available (more accurate)
        proc_n_rows = token_info.get("n_rows")
        proc_n_cols = token_info.get("n_cols")
        if proc_n_rows and proc_n_cols:
            tile_rows = proc_n_rows
            tile_cols = proc_n_cols

        # Convert embedding to numpy
        if isinstance(embedding, torch.Tensor):
            if embedding.dtype == torch.bfloat16:
                full_embedding = embedding.cpu().float().numpy()
            else:
                full_embedding = embedding.cpu().numpy()
        else:
            full_embedding = np.array(embedding)
        full_embedding = full_embedding.astype(np.float32)

        # Token info for metadata
        visual_indices = token_info["visual_token_indices"]
        num_visual_tokens = token_info["num_visual_tokens"]

        # =========================================================================
        # STRATEGY: "pooling" (NOVEL) vs "standard" (BASELINE) vs "all" (BOTH)
        # =========================================================================

        # Always compute visual-only embedding (needed for pooling and saliency)
        visual_embedding = full_embedding[visual_indices]

        # Mean pooling cap: <=0 or None => no cap (ColQwen2.5 keeps all effective rows).
        tv = self.max_mean_pool_vectors
        if tv is not None:
            try:
                tv_i = int(tv)
                tv = None if tv_i <= 0 else tv_i
            except Exception:
                tv = 32
        tile_pooled = self.embedder.mean_pool_visual_embedding(
            visual_embedding, token_info, target_vectors=tv
        )

        model_lower = (getattr(self.embedder, "model_name", "") or "").lower()
        is_colqwen25 = "colqwen2.5" in model_lower or "colqwen2_5" in model_lower
        is_colsmol = "colsmol" in model_lower
        experimental_pooled_by_name: Dict[str, Any] = {}
        # ColQwen2.5: keep experimental_pooling as Gaussian alias, and always store both variants explicitly.
        if is_colqwen25:
            exp_gaussian = self.embedder.experimental_pool_visual_embedding(
                visual_embedding,
                token_info,
                target_vectors=tv,
                mean_pool=tile_pooled,
                window_size=3,
                kernel="gaussian",
            )
            exp_triangular = self.embedder.experimental_pool_visual_embedding(
                visual_embedding,
                token_info,
                target_vectors=tv,
                mean_pool=tile_pooled,
                window_size=3,
                kernel="triangular",
            )
            experimental_pooled_by_name["experimental_pooling"] = exp_gaussian
            experimental_pooled_by_name["experimental_pooling_gaussian"] = exp_gaussian
            experimental_pooled_by_name["experimental_pooling_triangular"] = exp_triangular
            ks_norm = [3]
            canonical_k = 3
            kernel = "gaussian"
        else:
            kernel_arg = (
                str(getattr(self, "experimental_pooling_kernel", "auto") or "auto").lower().strip()
            )
            if kernel_arg == "auto":
                kernel = "legacy"
            else:
                kernel = kernel_arg
            default_k = 3
            ks = self.pooling_windows if self.pooling_windows else [default_k]
            # Normalize + keep order, avoid duplicates.
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

            canonical_k = int(ks_norm[0])
            for k in ks_norm:
                exp = self.embedder.experimental_pool_visual_embedding(
                    visual_embedding,
                    token_info,
                    target_vectors=tv,
                    mean_pool=tile_pooled,
                    window_size=int(k),
                    kernel=str(kernel),
                )
                experimental_pooled_by_name[f"experimental_pooling_{int(k)}"] = exp
                if int(k) == canonical_k:
                    experimental_pooled_by_name["experimental_pooling"] = exp

        if is_colsmol and bool(getattr(self, "colsmol_experimental_2d", False)):
            try:
                from visual_rag.embedding.pooling import colsmol_tile_4n_pooling_from_tiles

                n_rows = (token_info or {}).get("n_rows")
                n_cols = (token_info or {}).get("n_cols")
                if n_rows and n_cols:
                    exp2d = colsmol_tile_4n_pooling_from_tiles(
                        tile_pooled,
                        n_rows=int(n_rows),
                        n_cols=int(n_cols),
                        has_global=True,
                        include_self=True,
                        output_dtype=self.embedder.output_dtype,
                    )
                    experimental_pooled_by_name["experimental_pooling_2d"] = exp2d
            except Exception:
                pass
        global_pooled = global_mean_pooling(full_embedding)
        global_pooling = (
            self.embedder.global_pool_from_mean_pool(tile_pooled)
            if tile_pooled.size
            else global_pooled
        )

        num_tiles = int(tile_pooled.shape[0])
        patches_per_tile = int(visual_embedding.shape[0] // max(num_tiles, 1)) if num_tiles else 0
        if tile_rows and tile_cols and int(tile_rows) * int(tile_cols) + 1 == num_tiles:
            pass
        else:
            tile_rows = token_info.get("n_rows") or None
            tile_cols = token_info.get("n_cols") or None

        if self.embedding_strategy == "pooling":
            # NOVEL APPROACH: Visual tokens only + tile-level pooling
            embedding_for_initial = visual_embedding
            embedding_for_pooling = tile_pooled
            global_pooling = (
                self.embedder.global_pool_from_mean_pool(tile_pooled)
                if tile_pooled.size
                else global_pooled
            )

        elif self.embedding_strategy == "standard":
            # BASELINE: All tokens + global mean
            embedding_for_initial = full_embedding
            embedding_for_pooling = global_pooled.reshape(1, -1)
            global_pooling = global_pooled

        else:  # "all" - Push BOTH representations (efficient for comparison)
            # Embed once, store multiple vector representations
            # This allows comparing both strategies without re-embedding
            embedding_for_initial = visual_embedding  # Use visual for search
            embedding_for_pooling = tile_pooled  # Use tile-level for fast prefetch
            global_pooling = (
                self.embedder.global_pool_from_mean_pool(tile_pooled)
                if tile_pooled.size
                else global_pooled
            )

            # ALSO store standard representations as additional vectors
            # These will be added to metadata for optional use
            pass  # Extra vectors handled in return dict below

        # Upload to Cloudinary
        original_url = None
        cropped_url = None
        resized_url = None

        if upload_to_cloudinary and self.cloudinary_uploader:
            base_filename = f"{pdf_stem}_page_{page_num}"
            if self.crop_empty:
                original_url, cropped_url, resized_url = (
                    self.cloudinary_uploader.upload_original_cropped_and_resized(
                        raw_img, embed_img, resized_img, base_filename
                    )
                )
            else:
                original_url, resized_url = self.cloudinary_uploader.upload_original_and_resized(
                    raw_img, resized_img, base_filename
                )

        # Sanitize text
        safe_text = self._sanitize_text(text[:10000]) if text else ""

        metadata = {
            "filename": filename,
            "page_number": page_num,
            "total_pages": total_pages,
            "has_text": bool(text and text.strip()),
            "text": safe_text,
            # Image URLs
            "page": resized_url or "",  # For display
            "original_url": original_url or "",
            "cropped_url": cropped_url or "",
            "resized_url": resized_url or "",
            # Dimensions (needed for saliency overlay)
            "original_width": raw_img.width,
            "original_height": raw_img.height,
            "cropped_width": int(embed_img.width) if self.crop_empty else int(raw_img.width),
            "cropped_height": int(embed_img.height) if self.crop_empty else int(raw_img.height),
            "resized_width": resized_img.width,
            "resized_height": resized_img.height,
            # Tile structure (needed for saliency)
            "num_tiles": num_tiles,
            "tile_rows": tile_rows,
            "tile_cols": tile_cols,
            "patches_per_tile": patches_per_tile,
            # Token info (needed for saliency)
            "num_visual_tokens": num_visual_tokens,
            "visual_token_indices": visual_indices,
            "total_tokens": len(full_embedding),  # Total tokens in raw embedding
            # Strategy used (important for paper comparison)
            "embedding_strategy": self.embedding_strategy,
            "model_name": getattr(self.embedder, "model_name", None),
            "experimental_pooling_windows": ks_norm,
            "experimental_pooling_default_window": canonical_k,
            "experimental_pooling_kernel": str(kernel),
            "colsmol_experimental_2d": (
                bool(getattr(self, "colsmol_experimental_2d", False)) if is_colsmol else None
            ),
            "max_mean_pool_vectors": (
                int(self.max_mean_pool_vectors) if self.max_mean_pool_vectors is not None else None
            ),
            "crop_empty_enabled": bool(self.crop_empty),
            "crop_empty_crop_box": (crop_meta or {}).get("crop_box"),
            "crop_empty_remove_page_number": bool(self.crop_empty_remove_page_number),
            "crop_empty_percentage_to_remove": float(self.crop_empty_percentage_to_remove),
            "crop_empty_preserve_border_px": int(self.crop_empty_preserve_border_px),
            "crop_empty_uniform_rowcol_std_threshold": float(
                self.crop_empty_uniform_rowcol_std_threshold
            ),
            # Extra metadata (year, district, etc.)
            **extra_metadata,
        }

        result = {
            "id": chunk_id,
            "visual_embedding": embedding_for_initial,  # "initial" vector in Qdrant
            "tile_pooled_embedding": embedding_for_pooling,  # "mean_pooling" vector in Qdrant
            "experimental_pooled_embedding": experimental_pooled_by_name,  # ColQwen: gaussian/triangular; ColPali: "experimental_pooling_{k}"
            "global_pooled_embedding": global_pooling,  # "global_pooling" vector in Qdrant
            "metadata": metadata,
            "image": raw_img,
            "resized_image": resized_img,
        }

        # For "all" strategy, include BOTH representations for comparison
        if self.embedding_strategy == "all":
            result["extra_vectors"] = {
                # Standard baseline vectors (for comparison)
                "full_embedding": full_embedding,  # All tokens [total, 128]
                "global_pooled": global_pooled,  # Global mean [128]
                # Pooling vectors (already in main result)
                "visual_embedding": visual_embedding,  # Visual only [visual, 128]
                "tile_pooled": tile_pooled,  # Tile-level [tiles, 128]
            }

        return result

    def _upload_batch(self, upload_queue: List[Dict[str, Any]]) -> int:
        """Upload batch to Qdrant."""
        if not upload_queue or not self.indexer:
            return 0

        logger.info(f"📤 Uploading batch of {len(upload_queue)} pages...")

        count = self.indexer.upload_batch(
            upload_queue,
            delay_between_batches=self.delay_between_uploads,
        )

        return count

    def _get_extra_metadata(self, filename: str) -> Dict[str, Any]:
        """Get extra metadata for a filename."""
        if not self.metadata_mapping:
            return {}

        # Normalize filename
        filename_clean = filename.replace(".pdf", "").replace(".PDF", "").strip().lower()

        # Try exact match
        if filename_clean in self.metadata_mapping:
            return self.metadata_mapping[filename_clean].copy()

        # Try fuzzy match
        from difflib import SequenceMatcher

        best_match = None
        best_score = 0.0

        for known_filename, metadata in self.metadata_mapping.items():
            score = SequenceMatcher(None, filename_clean, known_filename.lower()).ratio()
            if score > best_score and score > 0.75:
                best_score = score
                best_match = metadata

        if best_match:
            logger.debug(f"Fuzzy matched '{filename}' with score {best_score:.2f}")
            return best_match.copy()

        return {}

    def _sanitize_text(self, text: str) -> str:
        """Remove invalid Unicode characters."""
        if not text:
            return ""
        return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")

    @staticmethod
    def generate_chunk_id(filename: str, page_number: int) -> str:
        """Generate deterministic chunk ID."""
        content = f"{filename}:page:{page_number}"
        hash_obj = hashlib.sha256(content.encode())
        hex_str = hash_obj.hexdigest()[:32]
        return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"

    @staticmethod
    def load_metadata_mapping(json_path: Path) -> Dict[str, Dict[str, Any]]:
        """
        Load metadata mapping from JSON file.

        Expected format:
        {
            "filenames": {
                "Report Name 2023": {"year": 2023, "source": "Local Government", ...},
                ...
            }
        }

        Or simple format:
        {
            "Report Name 2023": {"year": 2023, "source": "Local Government", ...},
            ...
        }
        """
        import json

        with open(json_path, "r") as f:
            data = json.load(f)

        # Check if nested under "filenames"
        if "filenames" in data and isinstance(data["filenames"], dict):
            mapping = data["filenames"]
        else:
            mapping = data

        # Normalize keys to lowercase
        normalized = {}
        for filename, metadata in mapping.items():
            key = filename.lower().strip().replace(".pdf", "")
            normalized[key] = metadata

        logger.info(f"📖 Loaded metadata for {len(normalized)} files")
        return normalized
