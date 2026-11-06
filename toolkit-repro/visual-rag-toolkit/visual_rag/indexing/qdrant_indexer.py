"""
Qdrant Indexer - Upload embeddings to Qdrant vector database.

Works INDEPENDENTLY of PDF processing and embedding generation.
Use it if you already have embeddings and just need to upload.

Features:
- Named vectors for multi-vector and pooled search
- Batch uploading with retry logic
- Skip-existing for incremental updates
- Configurable payload indexes
"""

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    from qdrant_client.http.models import Distance, VectorParams
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    qdrant_models = None
    Distance = None
    VectorParams = None
    FieldCondition = None
    Filter = None
    MatchValue = None

logger = logging.getLogger(__name__)


class QdrantIndexer:
    """
    Upload visual embeddings to Qdrant.

    Works independently - just needs embeddings and metadata.

    Args:
        url: Qdrant server URL
        api_key: Qdrant API key
        collection_name: Name of the collection
        timeout: Request timeout in seconds
        prefer_grpc: Use gRPC protocol (faster but may have issues)

    Example:
        >>> indexer = QdrantIndexer(
        ...     url="https://your-cluster.qdrant.io:6333",
        ...     api_key="your-api-key",
        ...     collection_name="my_collection",
        ... )
        >>>
        >>> # Create collection
        >>> indexer.create_collection()
        >>>
        >>> # Upload points
        >>> indexer.upload_batch(points)
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        collection_name: str,
        timeout: int = 60,
        prefer_grpc: bool = False,
        vector_datatype: str = "float32",
    ):
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant client not installed. "
                "Install with: pip install visual-rag-toolkit[qdrant]"
            )

        self.collection_name = collection_name
        self.timeout = timeout
        if vector_datatype not in ("float32", "float16"):
            raise ValueError("vector_datatype must be 'float32' or 'float16'")
        self.vector_datatype = vector_datatype
        self._np_vector_dtype = np.float16 if vector_datatype == "float16" else np.float32

        grpc_port = None
        if prefer_grpc:
            try:
                parsed = urlparse(url)
                port = parsed.port
                if port == 6333:
                    grpc_port = 6334
            except Exception:
                grpc_port = None

        def _make_client(use_grpc: bool):
            return QdrantClient(
                url=url,
                api_key=api_key,
                timeout=timeout,
                prefer_grpc=bool(use_grpc),
                grpc_port=grpc_port,
                check_compatibility=False,
            )

        self.client = _make_client(prefer_grpc)
        if prefer_grpc:
            try:
                _ = self.client.get_collections()
            except Exception as e:
                msg = str(e)
                if "StatusCode.PERMISSION_DENIED" in msg or "http2 header with status: 403" in msg:
                    self.client = _make_client(False)
                else:
                    raise

        logger.info(f"🔌 Connected to Qdrant: {url}")
        logger.info(f"   Collection: {collection_name}")
        logger.info(f"   Vector datatype: {self.vector_datatype}")

    def collection_exists(self) -> bool:
        """Check if collection exists."""
        collections = self.client.get_collections().collections
        return any(c.name == self.collection_name for c in collections)

    def create_collection(
        self,
        embedding_dim: int = 128,
        force_recreate: bool = False,
        enable_quantization: bool = False,
        indexing_threshold: int = 20000,
        full_scan_threshold: int = 0,
        experimental_vector_names: Optional[List[str]] = None,
    ) -> bool:
        """
        Create collection with multi-vector support.

        Creates named vectors:
        - initial: Full multi-vector embeddings (num_patches × dim)
        - mean_pooling: Tile-level pooled vectors (num_tiles × dim)
        - experimental_pooling: Experimental multi-vector pooling (varies by model)
        - experimental_pooling_{k}: (ColPali) Optional additional experimental poolings with different window sizes
        - experimental_pooling_gaussian / experimental_pooling_triangular: (ColQwen) Technique variants (k=3)
        - global_pooling: Single vector pooled representation (dim)

        Args:
            embedding_dim: Embedding dimension (128 for ColSmol)
            force_recreate: Delete and recreate if exists
            enable_quantization: Enable int8 quantization
            indexing_threshold: Qdrant optimizer indexing threshold (set 0 to always build ANN indexes)

        Returns:
            True if created, False if already existed
        """
        # Normalize requested experimental vector names (always include canonical "experimental_pooling")
        exp_names: List[str] = ["experimental_pooling"]
        if experimental_vector_names:
            for n in experimental_vector_names:
                s = str(n).strip()
                if not s:
                    continue
                exp_names.append(s)
        # Unique while preserving order
        seen = set()
        exp_names = [x for x in exp_names if not (x in seen or seen.add(x))]

        if self.collection_exists():
            if force_recreate:
                logger.info(f"🗑️ Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                # Verify the collection has the required named vectors.
                try:
                    info = self.client.get_collection(self.collection_name)
                    vectors = getattr(getattr(info, "config", None), "params", None)
                    vectors = getattr(vectors, "vectors", None)
                    existing = set()
                    if isinstance(vectors, dict):
                        existing = set(str(k) for k in vectors.keys())
                    missing = set(exp_names) - existing
                    if missing:
                        raise ValueError(
                            "Collection exists but is missing required experimental vectors: "
                            f"{sorted(missing)}. Recreate the collection to add new named vectors."
                        )
                except Exception as e:
                    # If we cannot verify, keep legacy behavior.
                    logger.debug(f"Could not verify existing vector schema: {e}")
                logger.info(f"✅ Collection already exists: {self.collection_name}")
                return False

        logger.info(f"📦 Creating collection: {self.collection_name}")

        # Multi-vector config for ColBERT-style MaxSim
        multivector_config = qdrant_models.MultiVectorConfig(
            comparator=qdrant_models.MultiVectorComparator.MAX_SIM
        )

        # Vector configs - simplified for compatibility
        datatype = (
            qdrant_models.Datatype.FLOAT16
            if self.vector_datatype == "float16"
            else qdrant_models.Datatype.FLOAT32
        )
        vectors_config = {
            "initial": VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
                on_disk=True,
                multivector_config=multivector_config,
                datatype=datatype,
            ),
            "mean_pooling": VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
                on_disk=False,
                multivector_config=multivector_config,
                datatype=datatype,
            ),
            "global_pooling": VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
                on_disk=False,
                datatype=datatype,
            ),
        }
        for exp_name in exp_names:
            vectors_config[str(exp_name)] = VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
                on_disk=False,
                multivector_config=multivector_config,
                datatype=datatype,
            )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            optimizers_config=qdrant_models.OptimizersConfigDiff(
                indexing_threshold=int(indexing_threshold),
            ),
        )

        # Create required payload index for skip_existing functionality
        # This index is needed for filtering by filename when checking existing docs
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="filename",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
            logger.info("   📇 Created payload index: filename")
        except Exception as e:
            logger.warning(f"   ⚠️ Could not create filename index: {e}")

        logger.info(f"✅ Collection created: {self.collection_name}")
        return True

    def create_payload_indexes(
        self,
        fields: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Create payload indexes for filtering.

        Args:
            fields: List of {field, type} dicts
                   type can be: integer, keyword, bool, float, text
        """
        type_mapping = {
            "integer": qdrant_models.PayloadSchemaType.INTEGER,
            "keyword": qdrant_models.PayloadSchemaType.KEYWORD,
            "bool": qdrant_models.PayloadSchemaType.BOOL,
            "float": qdrant_models.PayloadSchemaType.FLOAT,
            "text": qdrant_models.PayloadSchemaType.TEXT,
        }

        if not fields:
            return
        # Cache between calls so multi-dataset runs don't spam logs.
        if not hasattr(self, "_ensured_payload_indexes"):
            self._ensured_payload_indexes = set()
            self._payload_indexes_skip_logged = False

        requested_fields: List[str] = []
        for fc in fields:
            try:
                requested_fields.append(str(fc["field"]))
            except Exception:
                continue
        requested_set = set(requested_fields)

        # Qdrant exposes indexed payload fields in collection_info.payload_schema
        existing_indexed: set[str] = set()
        try:
            info = self.client.get_collection(self.collection_name)
            payload_schema = getattr(info, "payload_schema", None) or {}
            if isinstance(payload_schema, dict):
                existing_indexed = set(str(k) for k in payload_schema.keys())
        except Exception as e:
            logger.debug(f"Could not read existing payload schema: {e}")

        already = existing_indexed | set(self._ensured_payload_indexes)
        missing = requested_set - already

        if not missing:
            # Log this only once per process to avoid repetition across datasets.
            if not getattr(self, "_payload_indexes_skip_logged", False):
                logger.info("📇 Payload indexes already exist — skipping creation")
                self._payload_indexes_skip_logged = True
            self._ensured_payload_indexes |= requested_set
            return

        logger.info(f"📇 Creating payload indexes ({len(missing)} new)...")

        for field_config in fields:
            field_name = str(field_config["field"])
            if field_name not in missing:
                continue
            field_type_str = field_config.get("type", "keyword")
            field_type = type_mapping.get(field_type_str, qdrant_models.PayloadSchemaType.KEYWORD)

            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
                self._ensured_payload_indexes.add(field_name)
                logger.info(f"   ✅ {field_name} ({field_type_str})")
            except Exception as e:
                # If Qdrant reports it already exists anyway, treat it as ensured.
                self._ensured_payload_indexes.add(field_name)
                logger.debug(f"   Index {field_name} might already exist: {e}")

    def upload_batch(
        self,
        points: List[Dict[str, Any]],
        max_retries: int = 3,
        delay_between_batches: float = 0.5,
        wait: bool = True,
        stop_event=None,
    ) -> int:
        """
        Upload a batch of points to Qdrant.

        Each point should have:
        - id: Unique point ID (string or UUID)
        - visual_embedding: Full embedding [num_patches, dim]
        - tile_pooled_embedding: Pooled embedding [num_tiles, dim]
        - experimental_pooled_embedding: Experimental pooled embedding [*, dim]
        - global_pooled_embedding: Pooled embedding [dim]
        - metadata: Payload dict

        Args:
            points: List of point dicts
            max_retries: Retry attempts on failure
            delay_between_batches: Delay after upload
            wait: Wait for operation to complete on Qdrant server
            stop_event: Optional threading.Event used to cancel uploads early

        Returns:
            Number of successfully uploaded points
        """
        if not points:
            return 0

        def _is_cancelled() -> bool:
            return stop_event is not None and getattr(stop_event, "is_set", lambda: False)()

        def _is_payload_too_large_error(e: Exception) -> bool:
            msg = str(e)
            if ("JSON payload" in msg and "larger than allowed" in msg) or (
                "Payload error:" in msg and "limit:" in msg
            ):
                return True
            content = getattr(e, "content", None)
            if content is not None:
                try:
                    if isinstance(content, (bytes, bytearray)):
                        text = content.decode("utf-8", errors="ignore")
                    else:
                        text = str(content)
                except Exception:
                    text = ""
                if ("JSON payload" in text and "larger than allowed" in text) or (
                    "Payload error" in text and "limit" in text
                ):
                    return True
            resp = getattr(e, "response", None)
            if resp is not None:
                try:
                    text = str(getattr(resp, "text", "") or "")
                except Exception:
                    text = ""
                if ("JSON payload" in text and "larger than allowed" in text) or (
                    "Payload error" in text and "limit" in text
                ):
                    return True
            return False

        def _to_list(val):
            if isinstance(val, np.ndarray):
                return val.tolist()
            return val

        def _build_qdrant_points(
            batch_points: List[Dict[str, Any]],
        ) -> List[qdrant_models.PointStruct]:
            qdrant_points: List[qdrant_models.PointStruct] = []
            for p in batch_points:
                global_pooled = p.get("global_pooled_embedding")
                if global_pooled is None:
                    tile_pooled = np.array(p["tile_pooled_embedding"], dtype=np.float32)
                    global_pooled = tile_pooled.mean(axis=0)
                global_pooled = np.array(global_pooled, dtype=np.float32).reshape(-1)

                initial = np.array(p["visual_embedding"], dtype=np.float32).astype(
                    self._np_vector_dtype, copy=False
                )
                mean_pooling = np.array(p["tile_pooled_embedding"], dtype=np.float32).astype(
                    self._np_vector_dtype, copy=False
                )
                global_pooling = global_pooled.astype(self._np_vector_dtype, copy=False)

                exp_val = p.get("experimental_pooled_embedding")
                exp_vectors: Dict[str, Any] = {}
                if isinstance(exp_val, dict):
                    for k, v in exp_val.items():
                        if v is None:
                            continue
                        exp_vectors[str(k)] = np.array(v, dtype=np.float32).astype(
                            self._np_vector_dtype, copy=False
                        )
                elif exp_val is not None:
                    exp_vectors["experimental_pooling"] = np.array(
                        exp_val, dtype=np.float32
                    ).astype(self._np_vector_dtype, copy=False)

                qdrant_points.append(
                    qdrant_models.PointStruct(
                        id=p["id"],
                        vector={
                            "initial": _to_list(initial),
                            "mean_pooling": _to_list(mean_pooling),
                            "global_pooling": _to_list(global_pooling),
                            **{name: _to_list(val) for name, val in exp_vectors.items()},
                        },
                        payload=p["metadata"],
                    )
                )
            return qdrant_points

        # Upload with retry
        for attempt in range(max_retries):
            try:
                if _is_cancelled():
                    return 0
                qdrant_points = _build_qdrant_points(points)
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=qdrant_points,
                    wait=wait,
                )

                if delay_between_batches > 0:
                    if _is_cancelled():
                        return 0
                    time.sleep(delay_between_batches)

                return len(points)

            except Exception as e:
                if _is_payload_too_large_error(e) and len(points) > 1:
                    mid = len(points) // 2
                    left = points[:mid]
                    right = points[mid:]
                    logger.warning(
                        f"Upload payload too large for {len(points)} points; splitting into {len(left)} + {len(right)}"
                    )
                    return self.upload_batch(
                        left,
                        max_retries=max_retries,
                        delay_between_batches=delay_between_batches,
                        wait=wait,
                        stop_event=stop_event,
                    ) + self.upload_batch(
                        right,
                        max_retries=max_retries,
                        delay_between_batches=delay_between_batches,
                        wait=wait,
                        stop_event=stop_event,
                    )

                logger.warning(f"Upload attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    if _is_cancelled():
                        return 0
                    time.sleep(2**attempt)  # Exponential backoff

        logger.error(f"❌ Upload failed after {max_retries} attempts")
        return 0

    def check_exists(self, chunk_id: str) -> bool:
        """Check if a point already exists."""
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[chunk_id],
                with_payload=False,
                with_vectors=False,
            )
            return len(result) > 0
        except Exception:
            return False

    def get_existing_ids(self, filename: str) -> Set[str]:
        """Get all point IDs for a specific file.

        Requires a payload index on 'filename' field. If the index doesn't exist,
        this method will attempt to create it automatically.
        """
        existing_ids = set()
        offset = None

        try:
            while True:
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
                    ),
                    limit=100,
                    offset=offset,
                    with_payload=["page_number"],
                    with_vectors=False,
                )

                points, next_offset = results

                for point in points:
                    existing_ids.add(str(point.id))

                if next_offset is None or len(points) == 0:
                    break
                offset = next_offset

        except Exception as e:
            error_msg = str(e).lower()
            if "index required" in error_msg or "index" in error_msg and "filename" in error_msg:
                # Missing payload index - try to create it
                logger.warning(
                    "⚠️ Missing 'filename' payload index. Creating it now... "
                    "(skip_existing requires this index for filtering)"
                )
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="filename",
                        field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
                    )
                    logger.info("   ✅ Created 'filename' index. Retrying query...")
                    # Retry the query
                    return self.get_existing_ids(filename)
                except Exception as idx_err:
                    logger.warning(f"   ❌ Could not create index: {idx_err}")
                    logger.warning("   Returning empty set - all pages will be processed")
                    return set()
            else:
                logger.warning(f"⚠️ Error checking existing IDs: {e}")
                return set()

        return existing_ids

    def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)

            status = info.status
            if hasattr(status, "value"):
                status = status.value

            indexed_count = getattr(info, "indexed_vectors_count", 0) or 0
            if isinstance(indexed_count, dict):
                indexed_count = sum(indexed_count.values())

            return {
                "status": str(status),
                "points_count": getattr(info, "points_count", 0),
                "indexed_vectors_count": indexed_count,
            }
        except Exception as e:
            logger.warning(f"Could not get collection info: {e}")
            return None

    @staticmethod
    def generate_point_id(filename: str, page_number: int) -> str:
        """
        Generate deterministic point ID from filename and page.

        Returns a valid UUID string.
        """
        content = f"{filename}:page:{page_number}"
        hash_obj = hashlib.sha256(content.encode())
        hex_str = hash_obj.hexdigest()[:32]
        # Format as UUID
        return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"
