"""
Visual Embedder - Generate visual and text embeddings for document retrieval.

This module provides a flexible interface that supports:
- ColPali models (ColSmol, ColPali, ColQwen2)
- Other vision-language models (future)
- Image embedding with tile-aware processing
- Query embedding with special token filtering

The embedder is BACKEND-AGNOSTIC - configure which model to use via the
`backend` parameter or model_name.
"""

import gc
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VisualEmbedder:
    """
    Visual document embedder supporting multiple backends.

    Currently supports:
    - ColPali family (ColSmol-500M, ColPali, ColQwen2)
    - More backends can be added

    Args:
        model_name: HuggingFace model name (e.g., "vidore/colSmol-500M")
        backend: Backend type ("colpali", "auto"). "auto" detects from model_name.
        device: Device to use (auto, cuda, mps, cpu)
        torch_dtype: Data type for model weights
        batch_size: Batch size for image processing
        filter_special_tokens: Filter special tokens from query embeddings

    Example:
        >>> # Auto-detect backend from model name
        >>> embedder = VisualEmbedder(model_name="vidore/colSmol-500M")
        >>>
        >>> # Embed images
        >>> image_embeddings = embedder.embed_images(images)
        >>>
        >>> # Embed query
        >>> query_embedding = embedder.embed_query("What is the budget?")
        >>>
        >>> # Get token info for saliency maps
        >>> embeddings, token_infos = embedder.embed_images(
        ...     images, return_token_info=True
        ... )
    """

    # Known model families and their backends
    MODEL_BACKENDS = {
        "colsmol": "colpali",
        "colpali": "colpali",
        "colqwen": "colpali",
        "colidefics": "colpali",
    }

    def __init__(
        self,
        model_name: str = "vidore/colSmol-500M",
        backend: str = "auto",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        output_dtype: Optional[np.dtype] = None,
        batch_size: int = 4,
        filter_special_tokens: bool = True,
        processor_speed: str = "fast",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.filter_special_tokens = filter_special_tokens
        if processor_speed not in ("fast", "slow", "auto"):
            raise ValueError("processor_speed must be one of: fast, slow, auto")
        self.processor_speed = processor_speed

        if os.getenv("VISUALRAG_INCLUDE_SPECIAL_TOKENS"):
            self.filter_special_tokens = False
            logger.info("Special token filtering disabled via VISUALRAG_INCLUDE_SPECIAL_TOKENS")

        if backend == "auto":
            backend = self._detect_backend(model_name)
        self.backend = backend

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        if torch_dtype is None:
            if device == "cuda":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
        self.torch_dtype = torch_dtype

        if output_dtype is None:
            if torch_dtype == torch.float16:
                output_dtype = np.float16
            else:
                output_dtype = np.float32
        self.output_dtype = output_dtype

        self._model = None
        self._processor = None
        self._image_token_id = None

        logger.info("🤖 VisualEmbedder initialized")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Backend: {backend}")
        logger.info(
            f"   Device: {device}, torch_dtype: {torch_dtype}, output_dtype: {output_dtype}"
        )

    def _detect_backend(self, model_name: str) -> str:
        """Auto-detect backend from model name."""
        model_lower = model_name.lower()

        for key, backend in self.MODEL_BACKENDS.items():
            if key in model_lower:
                logger.debug(f"Detected backend '{backend}' from model name")
                return backend

        # Default to colpali for unknown models
        logger.warning(f"Unknown model '{model_name}', defaulting to 'colpali' backend")
        return "colpali"

    def _load_model(self):
        """Lazy load the model when first needed."""
        if self._model is not None:
            return

        if self.backend == "colpali":
            self._load_colpali_model()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _load_colpali_model(self):
        """Load ColPali-family model."""
        try:
            from colpali_engine.models import (
                ColIdefics3,
                ColIdefics3Processor,
                ColPali,
                ColPaliProcessor,
                ColQwen2,
                ColQwen2Processor,
            )
        except ImportError:
            raise ImportError(
                "colpali_engine not installed. Install with: "
                "pip install visual-rag-toolkit[embedding] or "
                "pip install colpali-engine"
            )
        try:
            # Newer colpali-engine versions add ColQwen2.5 support
            from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        except Exception:
            ColQwen2_5 = None
            ColQwen2_5_Processor = None

        logger.info(f"🤖 Loading ColPali model: {self.model_name}")
        logger.info(f"   Device: {self.device}, dtype: {self.torch_dtype}")

        def _processor_kwargs():
            if self.processor_speed == "auto":
                return {}
            return {"use_fast": self.processor_speed == "fast"}

        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(self.model_name)
        model_type = str(getattr(cfg, "model_type", "") or "").lower()

        if model_type == "colpali" or "colpali" in (self.model_name or "").lower():
            self._model = ColPali.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
            ).eval()
            try:
                self._processor = ColPaliProcessor.from_pretrained(
                    self.model_name, **_processor_kwargs()
                )
            except TypeError:
                self._processor = ColPaliProcessor.from_pretrained(self.model_name)
            except Exception:
                if self.processor_speed == "fast":
                    self._processor = ColPaliProcessor.from_pretrained(
                        self.model_name, use_fast=False
                    )
                else:
                    raise
            self._image_token_id = self._processor.image_token_id
            logger.info("✅ Loaded ColPali backend")
            return

        model_lower = (self.model_name or "").lower()
        is_qwen25 = (
            "colqwen2.5" in model_lower
            or "colqwen2_5" in model_lower
            or "qwen2_5" in model_type
            or "qwen2.5" in model_type
        )
        if is_qwen25:
            if ColQwen2_5 is None or ColQwen2_5_Processor is None:
                raise ImportError(
                    "ColQwen2.5 requires a newer colpali-engine. Install/upgrade with:\n"
                    '  pip install "transformers>=4.45.0"\n'
                    "  pip install git+https://github.com/illuin-tech/colpali\n"
                    "or ensure colpali-engine>=0.3.7 is installed."
                )
            # Attention backend selection:
            # - CUDA: prefer FlashAttention2 when available
            # - MPS: default to eager (SDPA on MPS can produce NaNs for some batched query shapes)
            # - Allow override via env var.
            attn_implementation = os.getenv("VISUALRAG_ATTN_IMPLEMENTATION") or None
            if attn_implementation is None:
                if str(self.device) == "mps":
                    attn_implementation = "eager"
                elif self.device != "cpu":
                    try:
                        from transformers.utils.import_utils import is_flash_attn_2_available

                        if is_flash_attn_2_available():
                            attn_implementation = "flash_attention_2"
                    except Exception:
                        pass
            self._model = ColQwen2_5.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                attn_implementation=attn_implementation,
            ).eval()
            try:
                self._processor = ColQwen2_5_Processor.from_pretrained(
                    self.model_name, **_processor_kwargs()
                )
            except TypeError:
                self._processor = ColQwen2_5_Processor.from_pretrained(self.model_name)
            except Exception:
                if self.processor_speed == "fast":
                    self._processor = ColQwen2_5_Processor.from_pretrained(
                        self.model_name, use_fast=False
                    )
                else:
                    raise
            self._image_token_id = self._processor.image_token_id
            logger.info("✅ Loaded ColQwen2.5 backend")
            return

        if model_type.startswith("qwen2") or "colqwen" in model_lower:
            self._model = ColQwen2.from_pretrained(
                self.model_name,
                dtype=self.torch_dtype,
                device_map=self.device,
            ).eval()
            try:
                self._processor = ColQwen2Processor.from_pretrained(
                    self.model_name, device_map=self.device, **_processor_kwargs()
                )
            except TypeError:
                self._processor = ColQwen2Processor.from_pretrained(
                    self.model_name, device_map=self.device
                )
            except Exception:
                if self.processor_speed == "fast":
                    self._processor = ColQwen2Processor.from_pretrained(
                        self.model_name, device_map=self.device, use_fast=False
                    )
                else:
                    raise
            self._image_token_id = self._processor.image_token_id
            logger.info("✅ Loaded ColQwen2 backend")
            return

        attn_implementation = "eager"
        if self.device != "cpu":
            try:
                import flash_attn  # noqa

                attn_implementation = "flash_attention_2"
                logger.info("   Using FlashAttention2")
            except ImportError:
                pass

        self._model = ColIdefics3.from_pretrained(
            self.model_name,
            dtype=self.torch_dtype,
            device_map=self.device,
            attn_implementation=attn_implementation,
        ).eval()
        try:
            self._processor = ColIdefics3Processor.from_pretrained(
                self.model_name, **_processor_kwargs()
            )
        except TypeError:
            self._processor = ColIdefics3Processor.from_pretrained(self.model_name)
        except Exception:
            if self.processor_speed == "fast":
                self._processor = ColIdefics3Processor.from_pretrained(
                    self.model_name, use_fast=False
                )
            else:
                raise
        self._image_token_id = self._processor.image_token_id

        logger.info("✅ Model loaded successfully")

    @property
    def model(self):
        self._load_model()
        return self._model

    @property
    def processor(self):
        self._load_model()
        return self._processor

    @property
    def image_token_id(self):
        self._load_model()
        return self._image_token_id

    def embed_query(
        self,
        query_text: str,
        filter_special_tokens: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Generate embedding for a text query.

        By default, filters out special tokens (CLS, SEP, PAD) to keep only
        meaningful text tokens for better MaxSim matching.

        Args:
            query_text: Natural language query string
            filter_special_tokens: Override instance-level setting

        Returns:
            Query embedding tensor of shape [num_tokens, embedding_dim]
        """
        should_filter = (
            filter_special_tokens
            if filter_special_tokens is not None
            else self.filter_special_tokens
        )

        with torch.no_grad():
            processed = self.processor.process_queries([query_text]).to(self.model.device)
            embedding = self.model(**processed)

        # Remove batch dimension: [1, tokens, dim] -> [tokens, dim]
        if embedding.dim() == 3:
            embedding = embedding.squeeze(0)

        # Safety: surface NaN/Inf early for single-query embeddings
        try:
            if bool(torch.isnan(embedding).any()) or bool(torch.isinf(embedding).any()):
                logger.error(
                    "NaN/Inf detected in single query embedding. "
                    "model=%s device=%s torch_dtype=%s query=%r",
                    str(self.model_name),
                    str(self.device),
                    str(self.torch_dtype),
                    str(query_text)[:2000],
                )
        except Exception:
            pass

        if should_filter:
            # Filter special tokens based on attention mask
            attention_mask = processed.get("attention_mask")
            if attention_mask is not None:
                # Keep only tokens with attention_mask = 1
                valid_mask = attention_mask.squeeze(0).bool()
                embedding = embedding[valid_mask]

                # Additionally filter padding tokens if present
                input_ids = processed.get("input_ids")
                if input_ids is not None:
                    input_ids = input_ids.squeeze(0)[valid_mask]
                    # Filter common special token IDs
                    # IDs >= 4 are usually real tokens for most tokenizers
                    non_special_mask = input_ids >= 4
                    if non_special_mask.any():
                        embedding = embedding[non_special_mask]

            logger.debug(f"Query embedding: {embedding.shape[0]} tokens after filtering")
        else:
            logger.debug(f"Query embedding: {embedding.shape[0]} tokens (unfiltered)")

        return embedding

    def embed_queries(
        self,
        query_texts: List[str],
        batch_size: Optional[int] = None,
        filter_special_tokens: Optional[bool] = None,
        show_progress: bool = True,
    ) -> List[torch.Tensor]:
        """
        Generate embeddings for a list of text queries.

        Returns a list of tensors, each of shape [num_tokens, embedding_dim].
        """
        should_filter = (
            filter_special_tokens
            if filter_special_tokens is not None
            else self.filter_special_tokens
        )
        batch_size = batch_size or self.batch_size

        # Optional: reduce padding variance by bucketing queries by length (helps on some backends).
        # Controlled via VISUALRAG_SORT_QUERIES_BY_LENGTH=1|0 (default: 1 on MPS+ColQwen2.5, else 0).
        model_lower = (self.model_name or "").lower()
        is_colqwen25 = "colqwen2.5" in model_lower or "colqwen2_5" in model_lower
        env_sort = os.getenv("VISUALRAG_SORT_QUERIES_BY_LENGTH")
        if env_sort is None:
            sort_by_len = bool(self.device == "mps" and is_colqwen25)
        else:
            sort_by_len = str(env_sort).strip().lower() not in ("0", "false", "no", "off")

        if sort_by_len and len(query_texts) > 1:
            tok = getattr(self.processor, "tokenizer", None)
            try:
                if tok is not None:
                    lengths = [
                        len(tok(q, add_special_tokens=True).get("input_ids", []))
                        for q in query_texts
                    ]
                else:
                    lengths = [len(str(q)) for q in query_texts]
                order = sorted(range(len(query_texts)), key=lambda idx: lengths[idx])
                inv = [0] * len(order)
                for pos, idx in enumerate(order):
                    inv[idx] = pos
                query_texts_sorted = [query_texts[i] for i in order]
            except Exception:
                query_texts_sorted = query_texts
                order = None
                inv = None
        else:
            query_texts_sorted = query_texts
            order = None
            inv = None

        outputs: List[torch.Tensor] = []
        fallback_count = 0
        nan_log_path: Optional[Path] = None
        nan_logged = 0
        iterator = range(0, len(query_texts_sorted), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="📝 Embedding queries", unit="batch")

        for i in iterator:
            batch = query_texts_sorted[i : i + batch_size]
            with torch.no_grad():
                processed = self.processor.process_queries(batch).to(self.model.device)
                batch_embeddings = self.model(**processed)

            if isinstance(batch_embeddings, torch.Tensor) and batch_embeddings.dim() == 3:
                attn = processed.get("attention_mask") if should_filter else None
                input_ids = processed.get("input_ids") if should_filter else None

                for j in range(batch_embeddings.shape[0]):
                    emb = batch_embeddings[j]
                    if should_filter and attn is not None:
                        valid_mask = attn[j].bool()
                        emb = emb[valid_mask]
                        if input_ids is not None:
                            ids = input_ids[j][valid_mask]
                            non_special_mask = ids >= 4
                            if non_special_mask.any():
                                emb = emb[non_special_mask]
                    # ColQwen2.5 on MPS can produce NaNs when batching queries.
                    # If we detect NaNs/Infs, recompute the query embedding individually (stable).
                    try:
                        has_nan = bool(torch.isnan(emb).any())
                        has_inf = bool(torch.isinf(emb).any())
                        if has_nan or has_inf:
                            # Persist a reproducible sample for debugging.
                            try:
                                if nan_log_path is None:
                                    log_dir = os.getenv("VISUALRAG_NAN_LOG_DIR") or str(
                                        Path("results") / "nan_samples"
                                    )
                                    Path(log_dir).mkdir(parents=True, exist_ok=True)
                                    safe_model = (
                                        str(self.model_name or "model")
                                        .replace("/", "_")
                                        .replace(" ", "_")
                                        .replace(":", "_")
                                    )
                                    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                    nan_log_path = (
                                        Path(log_dir) / f"nan_queries__{safe_model}__{ts}.jsonl"
                                    )

                                rec = {
                                    "ts": datetime.now(timezone.utc).isoformat(),
                                    "model_name": str(self.model_name),
                                    "device": str(self.device),
                                    "torch_dtype": str(self.torch_dtype),
                                    "output_dtype": str(self.output_dtype),
                                    "processor_speed": str(
                                        getattr(self, "processor_speed", "unknown")
                                    ),
                                    "filter_special_tokens": bool(should_filter),
                                    "batch_size": int(batch_size),
                                    "global_query_index": int(i + j),
                                    "query_text": str(batch[j]),
                                    "has_nan": bool(has_nan),
                                    "has_inf": bool(has_inf),
                                    "torch_version": str(getattr(torch, "__version__", "")),
                                }
                                with nan_log_path.open("a", encoding="utf-8") as f:
                                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                nan_logged += 1
                                if nan_logged <= 3:
                                    logger.warning(
                                        "NaN/Inf detected in batched query embedding (idx=%d). "
                                        "Logged sample to %s. Recomputing this query individually.",
                                        int(i + j),
                                        str(nan_log_path),
                                    )
                            except Exception:
                                pass
                            fallback_count += 1
                            emb = self.embed_query(batch[j], filter_special_tokens=should_filter)
                    except Exception:
                        pass
                    outputs.append(emb)
            else:
                outputs.extend(batch_embeddings)

            del processed, batch_embeddings
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

        if fallback_count > 0:
            logger.warning(
                "embed_queries(): detected NaN/Inf in %d/%d queries; "
                "recomputed those queries individually for stability.",
                int(fallback_count),
                int(len(query_texts)),
            )
            if nan_log_path is not None and nan_logged > 0:
                logger.warning(
                    "NaN/Inf samples written to %s (%d rows).", str(nan_log_path), int(nan_logged)
                )
        if order is None or inv is None:
            return outputs
        # Unsort back to the caller's original query order.
        out_unsorted: List[torch.Tensor] = [outputs[inv[i]] for i in range(len(inv))]
        return out_unsorted

    def embed_images(
        self,
        images: List[Image.Image],
        batch_size: Optional[int] = None,
        return_token_info: bool = False,
        show_progress: bool = True,
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[Dict[str, Any]]]]:
        """
        Generate embeddings for a list of images.

        Args:
            images: List of PIL Images
            batch_size: Override instance batch size
            return_token_info: Also return token metadata (for saliency maps)
            show_progress: Show progress bar

        Returns:
            If return_token_info=False:
                List of embedding tensors [num_patches, dim]
            If return_token_info=True:
                Tuple of (embeddings, token_infos)

        Token info contains:
            - visual_token_indices: Indices of visual tokens in embedding
            - num_visual_tokens: Count of visual tokens
            - n_rows, n_cols: Tile grid dimensions
            - num_tiles: Total tiles (n_rows × n_cols + 1 global)
        """
        batch_size = batch_size or self.batch_size
        if (
            self.device == "mps"
            and "colpali" in (self.model_name or "").lower()
            and int(batch_size) > 1
        ):
            batch_size = 1

        embeddings = []
        token_infos = [] if return_token_info else None

        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="🎨 Embedding", unit="batch")

        for i in iterator:
            batch = images[i : i + batch_size]
            current_batch_size = len(batch)

            while current_batch_size > 0:
                try:
                    sub_batch = batch[:current_batch_size]
                    self._embed_image_batch(
                        sub_batch, embeddings, token_infos, return_token_info
                    )
                    if current_batch_size < len(batch):
                        remaining = batch[current_batch_size:]
                        for single in remaining:
                            self._embed_image_batch(
                                [single], embeddings, token_infos, return_token_info
                            )
                    break
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "out of memory" not in str(e).lower() and "CUDA" not in str(e):
                        raise
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    new_size = max(1, current_batch_size // 2)
                    logger.warning(
                        f"⚠️ CUDA OOM with batch_size={current_batch_size}, "
                        f"retrying with batch_size={new_size}"
                    )
                    print(
                        f"[OOM] Reducing batch size from {current_batch_size} to {new_size}"
                    )
                    if new_size == current_batch_size:
                        raise
                    current_batch_size = new_size

        if return_token_info:
            return embeddings, token_infos
        return embeddings

    def _embed_image_batch(
        self,
        batch: List[Image.Image],
        embeddings: list,
        token_infos: Optional[list],
        return_token_info: bool,
    ):
        with torch.no_grad():
            processed = self.processor.process_images(batch).to(self.model.device)

            if return_token_info:
                input_ids = processed["input_ids"]
                batch_n_rows = processed.get("n_rows")
                batch_n_cols = processed.get("n_cols")
                batch_grid_thw = processed.get("image_grid_thw", None)
                if batch_grid_thw is None:
                    batch_grid_thw = processed.get("grid_thw", None)
                if batch_grid_thw is None:
                    batch_grid_thw = processed.get("image_grid", None)

                for j in range(input_ids.shape[0]):
                    image_token_mask = input_ids[j] == self.image_token_id
                    visual_indices = torch.where(image_token_mask)[0].cpu().numpy().tolist()

                    n_rows = batch_n_rows[j].item() if batch_n_rows is not None else None
                    n_cols = batch_n_cols[j].item() if batch_n_cols is not None else None
                    grid_t = grid_h = grid_w = None
                    grid_h_eff = grid_w_eff = None
                    if batch_grid_thw is not None:
                        try:
                            g = batch_grid_thw[j]
                            if hasattr(g, "dim") and g.dim() == 2:
                                g = g[0]
                            t, h, w = [int(x) for x in g.detach().cpu().tolist()]
                            grid_t, grid_h, grid_w = t, h, w
                            num_visual = int(len(visual_indices))
                            if int(h) * int(w) == num_visual:
                                grid_h_eff, grid_w_eff = int(h), int(w)
                            elif (
                                h % 2 == 0 and w % 2 == 0 and (h // 2) * (w // 2) == num_visual
                            ):
                                grid_h_eff, grid_w_eff = int(h // 2), int(w // 2)
                        except Exception:
                            pass

                    token_infos.append({
                        "visual_token_indices": visual_indices,
                        "num_visual_tokens": len(visual_indices),
                        "n_rows": n_rows,
                        "n_cols": n_cols,
                        "num_tiles": (n_rows * n_cols + 1) if n_rows and n_cols else None,
                        "grid_t": grid_t,
                        "grid_h": grid_h,
                        "grid_w": grid_w,
                        "grid_h_eff": grid_h_eff,
                        "grid_w_eff": grid_w_eff,
                    })

            batch_embeddings = self.model(**processed)

        if isinstance(batch_embeddings, torch.Tensor) and batch_embeddings.dim() == 3:
            for j in range(batch_embeddings.shape[0]):
                embeddings.append(batch_embeddings[j].cpu())
        else:
            embeddings.extend([e.cpu() for e in batch_embeddings])

        del processed, batch_embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def extract_visual_embedding(
        self,
        full_embedding: torch.Tensor,
        token_info: Dict[str, Any],
    ) -> np.ndarray:
        """
        Extract only visual token embeddings from full embedding.

        Filters out special tokens, keeping only visual patches for MaxSim.

        Args:
            full_embedding: Full embedding [all_tokens, dim]
            token_info: Token info dict from embed_images

        Returns:
            Visual embedding array [num_visual_tokens, dim]
        """
        visual_indices = token_info["visual_token_indices"]

        if isinstance(full_embedding, torch.Tensor):
            if full_embedding.dtype == torch.bfloat16:
                visual_emb = full_embedding[visual_indices].cpu().float().numpy()
            else:
                visual_emb = full_embedding[visual_indices].cpu().numpy()
        else:
            visual_emb = np.array(full_embedding)[visual_indices]

        return visual_emb.astype(self.output_dtype)

    def mean_pool_visual_embedding(
        self,
        visual_embedding: Union[torch.Tensor, np.ndarray],
        token_info: Optional[Dict[str, Any]] = None,
        *,
        target_vectors: Optional[int] = 32,
    ) -> np.ndarray:
        from visual_rag.embedding.pooling import (
            adaptive_row_mean_pooling_from_grid,
            colpali_row_mean_pooling,
            tile_level_mean_pooling,
        )

        model_lower = (self.model_name or "").lower()
        is_colsmol = "colsmol" in model_lower
        is_colqwen25 = "colqwen2.5" in model_lower or "colqwen2_5" in model_lower
        target_vectors_cap: Optional[int]
        if target_vectors is None:
            target_vectors_cap = None
        else:
            try:
                tv = int(target_vectors)
            except Exception:
                tv = 32
            target_vectors_cap = None if tv <= 0 else tv
        # For non-dynamic models, default to the historical fixed 32 vectors when unset.
        if not is_colqwen25 and target_vectors_cap is None:
            target_vectors_cap = 32

        if isinstance(visual_embedding, torch.Tensor):
            if visual_embedding.dtype == torch.bfloat16:
                visual_np = visual_embedding.cpu().float().numpy()
            else:
                visual_np = visual_embedding.cpu().numpy().astype(np.float32)
        else:
            visual_np = np.array(visual_embedding, dtype=np.float32)

        if is_colsmol:
            n_rows = (token_info or {}).get("n_rows")
            n_cols = (token_info or {}).get("n_cols")
            num_tiles = int(n_rows) * int(n_cols) + 1 if n_rows and n_cols else 13
            return tile_level_mean_pooling(
                visual_np, num_tiles=num_tiles, patches_per_tile=64, output_dtype=self.output_dtype
            )

        # ColQwen2.5 supports dynamic resolutions. The processor provides a pre-merge grid (grid_h/grid_w),
        # but the *effective* token grid is (grid_h_eff, grid_w_eff) due to 2×2 spatial merge.
        # We follow the dynamic shape by default:
        # - if target_vectors is unset (None or <=0), return all effective rows (no cap, no upsampling)
        # - else, return <= target_vectors rows (no upsampling).
        num_tokens = int(visual_np.shape[0])
        if is_colqwen25:
            grid_h_eff = (token_info or {}).get("grid_h_eff")
            grid_w_eff = (token_info or {}).get("grid_w_eff")
            if grid_h_eff and grid_w_eff and int(grid_h_eff) * int(grid_w_eff) == int(num_tokens):
                # Compute row means over the *effective* grid.
                target_rows = int(grid_h_eff)
                if target_vectors_cap is not None:
                    target_rows = min(int(target_vectors_cap), int(grid_h_eff))
                pooled_rows = adaptive_row_mean_pooling_from_grid(
                    visual_np,
                    grid_h=int(grid_h_eff),
                    grid_w=int(grid_w_eff),
                    target_rows=target_rows,
                    output_dtype=self.output_dtype,
                )
                return pooled_rows

        # Fallback: infer a square grid if possible
        grid = int(round(float(num_tokens) ** 0.5))
        if grid * grid == num_tokens:
            # For ColQwen2.5 with unset cap, keep all rows (grid) rather than defaulting to 32.
            effective_target_rows = (
                int(grid)
                if (is_colqwen25 and target_vectors_cap is None)
                else int(target_vectors_cap)
            )
            if int(grid) == int(effective_target_rows):
                return colpali_row_mean_pooling(
                    visual_np, grid_size=int(effective_target_rows), output_dtype=self.output_dtype
                )
            return adaptive_row_mean_pooling_from_grid(
                visual_np,
                grid_h=int(grid),
                grid_w=int(grid),
                target_rows=int(effective_target_rows),
                output_dtype=self.output_dtype,
            )

        # Last-resort: treat tokens as a sequence and adaptively mean-pool chunks to target_vectors rows.
        # If unset (None/<=0), fall back to 32 to avoid returning extremely large multi-vectors.
        tv_last = int(target_vectors_cap or 32)
        edges = np.linspace(0, num_tokens, tv_last + 1)
        pooled = np.zeros((tv_last, int(visual_np.shape[1])), dtype=np.float32)
        for i in range(tv_last):
            s = int(np.floor(edges[i]))
            e = int(np.ceil(edges[i + 1]))
            s = max(0, min(s, num_tokens - 1))
            e = max(s + 1, min(e, num_tokens))
            pooled[i] = visual_np[s:e].mean(axis=0)
        return pooled.astype(self.output_dtype)

    def global_pool_from_mean_pool(self, mean_pool: np.ndarray) -> np.ndarray:
        if mean_pool.size == 0:
            return np.zeros((128,), dtype=self.output_dtype)
        return mean_pool.mean(axis=0).astype(self.output_dtype)

    def experimental_pool_visual_embedding(
        self,
        visual_embedding: Union[torch.Tensor, np.ndarray],
        token_info: Optional[Dict[str, Any]] = None,
        *,
        target_vectors: Optional[int] = 32,
        mean_pool: Optional[np.ndarray] = None,
        window_size: Optional[int] = None,
        kernel: Optional[str] = None,
    ) -> np.ndarray:
        from visual_rag.embedding.pooling import (
            colpali_experimental_pooling_from_rows,
            colsmol_experimental_pooling,
            weighted_row_smoothing_same_length,
        )

        model_lower = (self.model_name or "").lower()
        is_colsmol = "colsmol" in model_lower
        is_colqwen25 = "colqwen2.5" in model_lower or "colqwen2_5" in model_lower

        if isinstance(visual_embedding, torch.Tensor):
            if visual_embedding.dtype == torch.bfloat16:
                visual_np = visual_embedding.cpu().float().numpy()
            else:
                visual_np = visual_embedding.cpu().numpy().astype(np.float32)
        else:
            visual_np = np.array(visual_embedding, dtype=np.float32)

        if is_colsmol:
            if (
                mean_pool is not None
                and getattr(mean_pool, "shape", None) is not None
                and int(mean_pool.shape[0]) > 0
            ):
                num_tiles = int(mean_pool.shape[0])
            else:
                num_tiles = (token_info or {}).get("num_tiles")
                if num_tiles is None:
                    num_visual_tokens = (token_info or {}).get("num_visual_tokens")
                    if num_visual_tokens is None:
                        num_visual_tokens = int(visual_np.shape[0])
                    patches_per_tile = 64
                    num_tiles = int(num_visual_tokens) // patches_per_tile
                    if int(num_tiles) * patches_per_tile != int(num_visual_tokens):
                        num_tiles = int(num_tiles) + 1
                num_tiles = int(num_tiles)
            return colsmol_experimental_pooling(
                visual_np, num_tiles=num_tiles, patches_per_tile=64, output_dtype=self.output_dtype
            )

        rows = (
            mean_pool
            if mean_pool is not None
            else self.mean_pool_visual_embedding(
                visual_np, token_info, target_vectors=target_vectors
            )
        )
        # Kernel selection:
        # - legacy: ColPali-style conv pooling that produces N+2r rows (historical behavior)
        # - uniform/triangular/gaussian: weighted smoothing that preserves row count (N -> N)
        k = (kernel or ("gaussian" if is_colqwen25 else "legacy")).lower().strip()
        if k in ("legacy", "legacy_conv", "conv"):
            # Allow overriding window size; keep legacy defaults.
            window = int(window_size) if window_size is not None else (5 if is_colqwen25 else 3)
            return colpali_experimental_pooling_from_rows(
                rows, window_size=window, output_dtype=self.output_dtype
            )

        # Weighted same-length smoothing defaults:
        # - ColQwen2.5: gaussian k=3
        # - ColPali: user-controlled; default to k=3 to mirror legacy scale
        window = int(window_size) if window_size is not None else 3
        return weighted_row_smoothing_same_length(
            rows,
            window_size=window,
            kernel=(
                "gaussian"
                if k == "gaussian"
                else ("triangular" if k == "triangular" else "uniform")
            ),
            output_dtype=self.output_dtype,
        )


# Backward compatibility alias
ColPaliEmbedder = VisualEmbedder
