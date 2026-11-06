"""
Pooling strategies for multi-vector embeddings.

Provides:
- Tile-level mean pooling: Preserves spatial structure (num_tiles × dim)
- Global mean pooling: Single vector (1 × dim)
- MaxSim scoring for ColBERT-style late interaction
"""

import logging
from typing import Literal, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _infer_output_dtype(
    embedding: Union[torch.Tensor, np.ndarray],
    output_dtype: Optional[np.dtype] = None,
) -> np.dtype:
    """Infer output dtype: use provided, else match input (fp16→fp16, bf16→fp32, fp32→fp32)."""
    if output_dtype is not None:
        return output_dtype
    if isinstance(embedding, torch.Tensor):
        if embedding.dtype == torch.float16:
            return np.float16
        return np.float32
    if isinstance(embedding, np.ndarray) and embedding.dtype == np.float16:
        return np.float16
    return np.float32


def tile_level_mean_pooling(
    embedding: Union[torch.Tensor, np.ndarray],
    num_tiles: int,
    patches_per_tile: int = 64,
    output_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    Compute tile-level mean pooling for multi-vector embeddings.

    Instead of collapsing to 1×dim (global pooling), this preserves spatial
    structure by computing mean per tile → num_tiles × dim.

    This is our NOVEL contribution for scalable visual retrieval:
    - Faster than full MaxSim (fewer vectors to compare)
    - More accurate than global pooling (preserves spatial info)
    - Ideal for two-stage retrieval (prefetch with pooled, rerank with full)

    Args:
        embedding: Visual token embeddings [num_visual_tokens, dim]
        num_tiles: Number of tiles (including global tile)
        patches_per_tile: Patches per tile (64 for ColSmol)
        output_dtype: Output dtype (default: infer from input, fp16→fp16, bf16→fp32)

    Returns:
        Tile-level pooled embeddings [num_tiles, dim]

    Example:
        >>> # Image with 4×3 tiles + 1 global = 13 tiles
        >>> # Each tile has 64 patches → 832 visual tokens
        >>> pooled = tile_level_mean_pooling(embedding, num_tiles=13)
        >>> print(pooled.shape)  # (13, 128)
    """
    out_dtype = _infer_output_dtype(embedding, output_dtype)
    if isinstance(embedding, torch.Tensor):
        if embedding.dtype == torch.bfloat16:
            emb_np = embedding.cpu().float().numpy()
        else:
            emb_np = embedding.cpu().numpy().astype(np.float32)
    else:
        emb_np = np.array(embedding, dtype=np.float32)

    num_visual_tokens = emb_np.shape[0]
    expected_tokens = num_tiles * patches_per_tile

    if num_visual_tokens != expected_tokens:
        logger.debug(f"Token count mismatch: {num_visual_tokens} vs expected {expected_tokens}")
        actual_tiles = num_visual_tokens // patches_per_tile
        if actual_tiles * patches_per_tile != num_visual_tokens:
            actual_tiles += 1
        num_tiles = actual_tiles

    tile_embeddings = []
    for tile_idx in range(num_tiles):
        start_idx = tile_idx * patches_per_tile
        end_idx = min(start_idx + patches_per_tile, num_visual_tokens)

        if start_idx >= num_visual_tokens:
            break

        tile_patches = emb_np[start_idx:end_idx]
        tile_mean = tile_patches.mean(axis=0)
        tile_embeddings.append(tile_mean)

    return np.array(tile_embeddings, dtype=out_dtype)


def colpali_row_mean_pooling(
    embedding: Union[torch.Tensor, np.ndarray],
    grid_size: int = 32,
    output_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    out_dtype = _infer_output_dtype(embedding, output_dtype)
    if isinstance(embedding, torch.Tensor):
        if embedding.dtype == torch.bfloat16:
            emb_np = embedding.cpu().float().numpy()
        else:
            emb_np = embedding.cpu().numpy().astype(np.float32)
    else:
        emb_np = np.array(embedding, dtype=np.float32)

    num_tokens, dim = emb_np.shape
    expected = int(grid_size) * int(grid_size)
    if num_tokens != expected:
        raise ValueError(
            f"Expected {expected} visual tokens for grid_size={grid_size}, got {num_tokens}"
        )

    grid = emb_np.reshape(int(grid_size), int(grid_size), int(dim))
    pooled = grid.mean(axis=1)
    return pooled.astype(out_dtype)


def adaptive_row_mean_pooling_from_grid(
    embedding: Union[torch.Tensor, np.ndarray],
    *,
    grid_h: int,
    grid_w: int,
    target_rows: int = 32,
    output_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    Row-mean pooling for arbitrary H×W patch grids with adaptive down/up-sampling to `target_rows`.

    This is useful for dynamic-resolution models (e.g., ColQwen2.5) where the number of
    visual tokens (patches) is not fixed to a 32×32 grid.

    Steps:
    1) reshape tokens to [H, W, dim]
    2) mean over columns -> [H, dim]
    3) adaptive mean-pool rows to `target_rows` -> [target_rows, dim]
    """
    out_dtype = _infer_output_dtype(embedding, output_dtype)
    if isinstance(embedding, torch.Tensor):
        if embedding.dtype == torch.bfloat16:
            emb_np = embedding.cpu().float().numpy()
        else:
            emb_np = embedding.cpu().numpy().astype(np.float32)
    else:
        emb_np = np.array(embedding, dtype=np.float32)

    num_tokens, dim = emb_np.shape
    expected = int(grid_h) * int(grid_w)
    if num_tokens != expected:
        raise ValueError(
            f"Expected {expected} visual tokens for grid_h×grid_w={grid_h}×{grid_w}, got {num_tokens}"
        )

    grid = emb_np.reshape(int(grid_h), int(grid_w), int(dim))
    rows = grid.mean(axis=1)  # [H, dim]

    h = int(rows.shape[0])
    target_rows = int(target_rows)
    if target_rows <= 0:
        raise ValueError("target_rows must be > 0")
    if h == target_rows:
        return rows.astype(out_dtype)
    if h == 1:
        return np.repeat(rows, repeats=target_rows, axis=0).astype(out_dtype)

    # Adaptive average pooling along the row dimension.
    # We use evenly spaced bins over [0, H) and mean rows per bin.
    edges = np.linspace(0, h, target_rows + 1)
    pooled = np.zeros((target_rows, int(dim)), dtype=np.float32)
    for i in range(target_rows):
        start = int(np.floor(edges[i]))
        end = int(np.ceil(edges[i + 1]))
        start = max(0, min(start, h - 1))
        end = max(start + 1, min(end, h))
        pooled[i] = rows[start:end].mean(axis=0)

    return pooled.astype(out_dtype)


def colsmol_experimental_pooling(
    embedding: Union[torch.Tensor, np.ndarray],
    num_tiles: int,
    patches_per_tile: int = 64,
    output_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    out_dtype = _infer_output_dtype(embedding, output_dtype)
    if isinstance(embedding, torch.Tensor):
        if embedding.dtype == torch.bfloat16:
            emb_np = embedding.cpu().float().numpy()
        else:
            emb_np = embedding.cpu().numpy().astype(np.float32)
    else:
        emb_np = np.array(embedding, dtype=np.float32)

    num_visual_tokens, dim = emb_np.shape
    if num_tiles <= 0:
        raise ValueError("num_tiles must be > 0")
    if patches_per_tile <= 0:
        raise ValueError("patches_per_tile must be > 0")

    last_tile_start = (int(num_tiles) - 1) * int(patches_per_tile)
    if last_tile_start >= num_visual_tokens:
        actual_tiles = int(num_visual_tokens) // int(patches_per_tile)
        if actual_tiles * int(patches_per_tile) != int(num_visual_tokens):
            actual_tiles += 1
        if actual_tiles <= 0:
            raise ValueError(
                f"Not enough tokens for num_tiles={num_tiles}, patches_per_tile={patches_per_tile}: got {num_visual_tokens}"
            )
        num_tiles = actual_tiles
        last_tile_start = (int(num_tiles) - 1) * int(patches_per_tile)

    prefix = emb_np[:last_tile_start]
    last_tile = emb_np[
        last_tile_start : min(last_tile_start + int(patches_per_tile), num_visual_tokens)
    ]

    if prefix.size:
        prefix_tiles = prefix.reshape(-1, int(patches_per_tile), int(dim))
        prefix_means = prefix_tiles.mean(axis=1)
    else:
        prefix_means = np.zeros((0, int(dim)), dtype=out_dtype)

    return np.concatenate([prefix_means.astype(out_dtype), last_tile.astype(out_dtype)], axis=0)


def colpali_experimental_pooling_from_rows(
    row_vectors: Union[torch.Tensor, np.ndarray],
    *,
    window_size: int = 3,
    output_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    Experimental "convolution-style" pooling with an odd `window_size` (default: 3).

    For N input rows and radius r = window_size//2, produces N + 2r output vectors.
    Each output position uses a clipped window around a (possibly out-of-range) center:
      center = i - r, i in [0, N + 2r - 1]
      window = rows[max(0, center-r) : min(N, center+r+1)]

    For window_size=3 and N=32 rows: produces 34 vectors (same as previous implementation).
    """
    out_dtype = _infer_output_dtype(row_vectors, output_dtype)
    if isinstance(row_vectors, torch.Tensor):
        if row_vectors.dtype == torch.bfloat16:
            rows = row_vectors.cpu().float().numpy()
        else:
            rows = row_vectors.cpu().numpy().astype(np.float32)
    else:
        rows = np.array(row_vectors, dtype=np.float32)

    n, dim = rows.shape
    if n < 1:
        raise ValueError("row_vectors must be non-empty")

    window_size = int(window_size)
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if window_size == 1 or n == 1:
        return rows.astype(out_dtype)

    r = window_size // 2
    # Preserve legacy edge-case behavior for the default window_size=3:
    # - n=1 -> (1, dim)
    # - n=2 -> (3, dim): [row0, mean(row0,row1), row1]
    # - n>=3 -> (n+2, dim)
    if int(window_size) == 3 and int(n) == 2:
        mid = rows.mean(axis=0)
        return np.stack([rows[0], mid, rows[1]], axis=0).astype(out_dtype)
    out = np.zeros((n + 2 * r, dim), dtype=np.float32)
    for i in range(n + 2 * r):
        center = i - r
        lo = max(0, center - r)
        hi = min(n - 1, center + r)
        out[i] = rows[lo : hi + 1].mean(axis=0)
    return out.astype(out_dtype)


def weighted_row_smoothing_same_length(
    row_vectors: Union[torch.Tensor, np.ndarray],
    *,
    window_size: int = 3,
    kernel: Literal["uniform", "triangular", "gaussian"] = "gaussian",
    sigma: Optional[float] = None,
    output_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    Smooth row vectors with a weighted 1D kernel, returning the SAME number of rows (N -> N).

    Unlike `colpali_experimental_pooling_from_rows`, this does NOT add extra border vectors.
    It is designed to be a better "experimental pooling" for backbones that already include
    learned local mixing (e.g. Qwen2/2.5-VL PatchMerger).

    Supports any positive `window_size` (odd or even). Even sizes are centered between two rows.
    """
    out_dtype = _infer_output_dtype(row_vectors, output_dtype)
    if isinstance(row_vectors, torch.Tensor):
        if row_vectors.dtype == torch.bfloat16:
            rows = row_vectors.cpu().float().numpy()
        else:
            rows = row_vectors.cpu().numpy().astype(np.float32)
    else:
        rows = np.array(row_vectors, dtype=np.float32)

    n, dim = rows.shape
    if n < 1:
        raise ValueError("row_vectors must be non-empty")

    k = int(window_size)
    if k < 1:
        raise ValueError("window_size must be >= 1")
    if k == 1 or n == 1:
        return rows.astype(out_dtype)

    kernel = str(kernel).lower().strip()
    if kernel not in ("uniform", "triangular", "gaussian"):
        raise ValueError(f"Unknown kernel={kernel}. Choose uniform|triangular|gaussian.")

    # Center can be .0 (odd k) or .5 (even k)
    center = (k - 1) / 2.0
    positions = np.arange(k, dtype=np.float32)
    dist = np.abs(positions - center)

    if kernel == "uniform":
        w = np.ones((k,), dtype=np.float32)
    elif kernel == "triangular":
        # Linear decay. For odd k=5: [1,2,3,2,1]. For even k=4: [1.5,2.5,2.5,1.5].
        w = (center + 1.0) - dist
        w = np.clip(w, 0.0, None).astype(np.float32)
    else:  # gaussian
        # Reasonable default sigma: proportional to radius, but not too tiny for k=2/3.
        if sigma is None:
            # radius ~= center; sigma ~= radius/2, but min 0.5
            sigma_eff = max(0.5, float(center) / 2.0)
        else:
            sigma_eff = float(sigma)
            if sigma_eff <= 0:
                raise ValueError("sigma must be > 0")
        w = np.exp(-0.5 * (dist / sigma_eff) ** 2).astype(np.float32)

    # Normalize weights
    w_sum = float(w.sum())
    if w_sum <= 0:
        return rows.astype(out_dtype)
    w = w / w_sum

    # Window indexing: for each i, take indices [i-left, i-left+k)
    left = k // 2  # for odd: r; for even: k/2 (slightly "right of center")
    out = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        acc = np.zeros((dim,), dtype=np.float32)
        w_acc = 0.0
        start = i - left
        for t in range(k):
            j = start + t
            if j < 0 or j >= n:
                continue
            wt = float(w[t])
            acc += wt * rows[j]
            w_acc += wt
        if w_acc > 0:
            out[i] = acc / w_acc
        else:
            out[i] = rows[i]
    return out.astype(out_dtype)


def colsmol_tile_4n_pooling_from_tiles(
    tile_vectors: Union[torch.Tensor, np.ndarray],
    *,
    n_rows: int,
    n_cols: int,
    has_global: bool = True,
    include_self: bool = True,
    output_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    ColSmol-specific experimental pooling: 2D 4-neighborhood pooling over the tile grid.

    Expects tile vectors ordered row-major for the tile grid, followed by an optional global tile.
    Returns the same number of vectors as input (grid tiles [+ global]).
    """
    out_dtype = _infer_output_dtype(tile_vectors, output_dtype)
    if isinstance(tile_vectors, torch.Tensor):
        if tile_vectors.dtype == torch.bfloat16:
            tiles = tile_vectors.cpu().float().numpy()
        else:
            tiles = tile_vectors.cpu().numpy().astype(np.float32)
    else:
        tiles = np.array(tile_vectors, dtype=np.float32)

    n_rows = int(n_rows)
    n_cols = int(n_cols)
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("n_rows and n_cols must be > 0")
    grid_n = n_rows * n_cols
    if tiles.shape[0] < grid_n:
        raise ValueError(
            f"Expected at least {grid_n} tile vectors for n_rows×n_cols={n_rows}×{n_cols}, got {tiles.shape[0]}"
        )

    grid = tiles[:grid_n].reshape(n_rows, n_cols, -1)
    out_grid = np.zeros_like(grid, dtype=np.float32)

    for r in range(n_rows):
        for c in range(n_cols):
            neigh = []
            if include_self:
                neigh.append(grid[r, c])
            if r > 0:
                neigh.append(grid[r - 1, c])
            if r + 1 < n_rows:
                neigh.append(grid[r + 1, c])
            if c > 0:
                neigh.append(grid[r, c - 1])
            if c + 1 < n_cols:
                neigh.append(grid[r, c + 1])
            out_grid[r, c] = np.stack(neigh, axis=0).mean(axis=0)

    out_list = [out_grid.reshape(grid_n, -1)]
    if has_global and tiles.shape[0] > grid_n:
        # Keep global tile unchanged (it already summarizes the page).
        out_list.append(tiles[grid_n : grid_n + 1])

    out = np.concatenate(out_list, axis=0)
    return out.astype(out_dtype)


def global_mean_pooling(
    embedding: Union[torch.Tensor, np.ndarray],
    output_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    Compute global mean pooling → single vector.

    This is the simplest pooling but loses all spatial information.
    Use for fastest retrieval when accuracy can be sacrificed.

    Args:
        embedding: Multi-vector embeddings [num_tokens, dim]
        output_dtype: Output dtype (default: infer from input, fp16→fp16, bf16→fp32)

    Returns:
        Pooled vector [dim]
    """
    out_dtype = _infer_output_dtype(embedding, output_dtype)
    if isinstance(embedding, torch.Tensor):
        if embedding.dtype == torch.bfloat16:
            emb_np = embedding.cpu().float().numpy()
        else:
            emb_np = embedding.cpu().numpy()
    else:
        emb_np = np.array(embedding)

    return emb_np.mean(axis=0).astype(out_dtype)


def compute_maxsim_score(
    query_embedding: np.ndarray,
    doc_embedding: np.ndarray,
    normalize: bool = True,
) -> float:
    """
    Compute ColBERT-style MaxSim late interaction score.

    For each query token, finds max similarity with any document token,
    then sums across query tokens.

    This is the standard scoring for ColBERT/ColPali:
    score = Σ_q max_d (sim(q, d))

    Args:
        query_embedding: Query embeddings [num_query_tokens, dim]
        doc_embedding: Document embeddings [num_doc_tokens, dim]
        normalize: L2 normalize embeddings before scoring (recommended)

    Returns:
        MaxSim score (higher is better)

    Example:
        >>> query = embedder.embed_query("budget allocation")
        >>> doc = embeddings[0]  # From embed_images
        >>> score = compute_maxsim_score(query, doc)
    """
    if normalize:
        # L2 normalize
        query_norm = query_embedding / (
            np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-8
        )
        doc_norm = doc_embedding / (np.linalg.norm(doc_embedding, axis=1, keepdims=True) + 1e-8)
    else:
        query_norm = query_embedding
        doc_norm = doc_embedding

    # Compute similarity matrix: [num_query, num_doc]
    similarity_matrix = np.dot(query_norm, doc_norm.T)

    # MaxSim: For each query token, take max similarity with any doc token
    max_similarities = similarity_matrix.max(axis=1)

    # Sum across query tokens
    score = float(max_similarities.sum())

    return score


def compute_maxsim_batch(
    query_embedding: np.ndarray,
    doc_embeddings: list,
    normalize: bool = True,
) -> list:
    """
    Compute MaxSim scores for multiple documents efficiently.

    Args:
        query_embedding: Query embeddings [num_query_tokens, dim]
        doc_embeddings: List of document embeddings
        normalize: L2 normalize embeddings

    Returns:
        List of MaxSim scores
    """
    # Pre-normalize query once
    if normalize:
        query_norm = query_embedding / (
            np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-8
        )
    else:
        query_norm = query_embedding

    scores = []
    for doc_emb in doc_embeddings:
        if normalize:
            doc_norm = doc_emb / (np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-8)
        else:
            doc_norm = doc_emb

        sim_matrix = np.dot(query_norm, doc_norm.T)
        max_sims = sim_matrix.max(axis=1)
        scores.append(float(max_sims.sum()))

    return scores
