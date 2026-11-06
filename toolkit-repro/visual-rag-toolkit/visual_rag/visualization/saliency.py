"""
Saliency Map Generation for Visual Document Retrieval.

Generates attention/saliency maps to visualize which parts of documents
are most relevant to a query.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def generate_saliency_map(
    query_embedding: np.ndarray,
    doc_embedding: np.ndarray,
    image: Image.Image,
    token_info: Optional[Dict[str, Any]] = None,
    colormap: str = "Reds",
    alpha: float = 0.5,
    threshold_percentile: float = 50.0,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Generate saliency map showing which parts of the image match the query.

    Computes patch-level relevance scores and overlays them on the image.

    Args:
        query_embedding: Query embeddings [num_query_tokens, dim]
        doc_embedding: Document visual embeddings [num_visual_tokens, dim]
        image: Original PIL Image
        token_info: Optional token info with n_rows, n_cols for tile grid
        colormap: Matplotlib colormap name (Reds, viridis, jet, etc.)
        alpha: Overlay transparency (0-1)
        threshold_percentile: Only highlight patches above this percentile

    Returns:
        Tuple of (annotated_image, patch_scores)

    Example:
        >>> query = embedder.embed_query("budget allocation")
        >>> doc = visual_embedding  # From embed_images
        >>> annotated, scores = generate_saliency_map(
        ...     query_embedding=query.numpy(),
        ...     doc_embedding=doc,
        ...     image=page_image,
        ...     token_info=token_info,
        ... )
        >>> annotated.save("saliency.png")
    """
    # Ensure numpy arrays
    if hasattr(query_embedding, "numpy"):
        query_np = query_embedding.numpy()
    elif hasattr(query_embedding, "cpu"):
        query_np = query_embedding.cpu().float().numpy()  # .float() for BFloat16
    else:
        query_np = np.array(query_embedding, dtype=np.float32)

    if hasattr(doc_embedding, "numpy"):
        doc_np = doc_embedding.numpy()
    elif hasattr(doc_embedding, "cpu"):
        doc_np = doc_embedding.cpu().float().numpy()  # .float() for BFloat16
    else:
        doc_np = np.array(doc_embedding, dtype=np.float32)

    # Normalize embeddings
    query_norm = query_np / (np.linalg.norm(query_np, axis=1, keepdims=True) + 1e-8)
    doc_norm = doc_np / (np.linalg.norm(doc_np, axis=1, keepdims=True) + 1e-8)

    # Compute similarity matrix: [num_query, num_doc]
    similarity_matrix = np.dot(query_norm, doc_norm.T)

    # Get max similarity per document patch (best match from any query token)
    patch_scores = similarity_matrix.max(axis=0)

    # Normalize to [0, 1]
    score_min, score_max = patch_scores.min(), patch_scores.max()
    if score_max - score_min > 1e-8:
        patch_scores_norm = (patch_scores - score_min) / (score_max - score_min)
    else:
        patch_scores_norm = np.zeros_like(patch_scores)

    # Determine grid dimensions
    if token_info and token_info.get("n_rows") and token_info.get("n_cols"):
        n_rows = token_info["n_rows"]
        n_cols = token_info["n_cols"]
        num_tiles = n_rows * n_cols + 1  # +1 for global tile
        patches_per_tile = 64  # ColSmol standard

        # Reshape to tile grid (excluding global tile)
        try:
            # Skip global tile patches at the end
            tile_patches = num_tiles * patches_per_tile
            if len(patch_scores_norm) >= tile_patches:
                grid_patches = patch_scores_norm[: n_rows * n_cols * patches_per_tile]
            else:
                grid_patches = patch_scores_norm

            # Reshape: [tiles * patches_per_tile] -> [tiles, patches_per_tile]
            # Then mean per tile
            num_grid_tiles = n_rows * n_cols
            grid_patches = grid_patches[: num_grid_tiles * patches_per_tile]
            tile_scores = grid_patches.reshape(num_grid_tiles, patches_per_tile).mean(axis=1)
            tile_scores = tile_scores.reshape(n_rows, n_cols)
        except Exception as e:
            logger.warning(f"Could not reshape to tile grid: {e}")
            tile_scores = None
    else:
        tile_scores = None
        n_rows = n_cols = None

    # Create overlay
    annotated = create_saliency_overlay(
        image=image,
        scores=tile_scores if tile_scores is not None else patch_scores_norm,
        colormap=colormap,
        alpha=alpha,
        threshold_percentile=threshold_percentile,
        grid_rows=n_rows,
        grid_cols=n_cols,
    )

    return annotated, patch_scores


def create_saliency_overlay(
    image: Image.Image,
    scores: np.ndarray,
    colormap: str = "Reds",
    alpha: float = 0.5,
    threshold_percentile: float = 50.0,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
) -> Image.Image:
    """
    Create colored overlay on image based on scores.

    Args:
        image: Base PIL Image
        scores: Score array - 1D [num_patches] or 2D [rows, cols]
        colormap: Matplotlib colormap name
        alpha: Overlay transparency
        threshold_percentile: Only color patches above this percentile
        grid_rows, grid_cols: Grid dimensions (auto-detected if not provided)

    Returns:
        Annotated PIL Image
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, returning original image")
        return image

    img_array = np.array(image)
    h, w = img_array.shape[:2]

    # Handle 2D scores (tile grid)
    if scores.ndim == 2:
        rows, cols = scores.shape
    elif grid_rows and grid_cols:
        rows, cols = grid_rows, grid_cols
        # Reshape if possible
        if len(scores) == rows * cols:
            scores = scores.reshape(rows, cols)
        else:
            # Fallback: estimate grid from score count
            num_patches = len(scores)
            aspect = w / h
            cols = int(np.sqrt(num_patches * aspect))
            rows = max(1, num_patches // cols)
            scores = scores[: rows * cols].reshape(rows, cols)
    else:
        # Auto-estimate grid
        num_patches = len(scores) if scores.ndim == 1 else scores.size
        aspect = w / h
        cols = max(1, int(np.sqrt(num_patches * aspect)))
        rows = max(1, num_patches // cols)

        if rows * cols > len(scores) if scores.ndim == 1 else scores.size:
            cols = max(1, cols - 1)

        if scores.ndim == 1:
            scores = scores[: rows * cols].reshape(rows, cols)

    # Get colormap
    cmap = plt.cm.get_cmap(colormap)

    # Calculate threshold
    threshold = np.percentile(scores, threshold_percentile)

    # Calculate cell dimensions
    cell_h = h // rows
    cell_w = w // cols

    # Create RGBA overlay
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            score = scores[i, j]

            if score >= threshold:
                y1 = i * cell_h
                y2 = min((i + 1) * cell_h, h)
                x1 = j * cell_w
                x2 = min((j + 1) * cell_w, w)

                # Normalize score for coloring (above threshold)
                norm_score = (score - threshold) / (1.0 - threshold + 1e-8)
                norm_score = min(1.0, max(0.0, norm_score))

                # Get color
                color = cmap(norm_score)[:3]
                color_uint8 = (np.array(color) * 255).astype(np.uint8)

                overlay[y1:y2, x1:x2, :3] = color_uint8
                overlay[y1:y2, x1:x2, 3] = int(alpha * 255 * norm_score)

    # Blend with original
    overlay_img = Image.fromarray(overlay, "RGBA")
    result = Image.alpha_composite(image.convert("RGBA"), overlay_img)

    return result.convert("RGB")


def visualize_search_results(
    query: str,
    results: List[Dict[str, Any]],
    query_embedding: Optional[np.ndarray] = None,
    embeddings: Optional[List[np.ndarray]] = None,
    output_path: Optional[str] = None,
    max_results: int = 5,
    show_saliency: bool = False,
) -> Optional[Image.Image]:
    """
    Visualize search results as a grid of images with scores.

    Args:
        query: Original query text
        results: List of search results with 'payload' containing 'page' (image URL/base64)
        query_embedding: Query embedding for saliency (optional)
        embeddings: Document embeddings for saliency (optional)
        output_path: Path to save visualization (optional)
        max_results: Maximum results to show
        show_saliency: Generate saliency overlays (requires query_embedding & embeddings)

    Returns:
        Combined visualization image if successful
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required for visualization")
        return None

    results = results[:max_results]
    n = len(results)

    if n == 0:
        logger.warning("No results to visualize")
        return None

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for idx, (result, ax) in enumerate(zip(results, axes)):
        payload = result.get("payload", {})
        score = result.get("score_final", result.get("score_stage1", 0))

        # Try to load image from payload
        page_data = payload.get("page", "")
        image = None

        if page_data.startswith("data:image"):
            # Base64 encoded
            try:
                import base64
                from io import BytesIO

                b64_data = page_data.split(",")[1]
                image = Image.open(BytesIO(base64.b64decode(b64_data)))
            except Exception as e:
                logger.debug(f"Could not decode base64 image: {e}")
        elif page_data.startswith("http"):
            # URL - try to fetch
            try:
                import urllib.request
                from io import BytesIO

                with urllib.request.urlopen(page_data, timeout=5) as response:
                    image = Image.open(BytesIO(response.read()))
            except Exception as e:
                logger.debug(f"Could not fetch image URL: {e}")

        if image:
            ax.imshow(image)
        else:
            # Show placeholder
            ax.text(0.5, 0.5, "No image", ha="center", va="center", fontsize=12, color="gray")

        # Add title
        title = f"Rank {idx + 1}\nScore: {score:.3f}"
        if payload.get("filename"):
            title += f"\n{payload['filename'][:30]}"
        if payload.get("page_number") is not None:
            title += f" p.{payload['page_number'] + 1}"

        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Add query as suptitle
    query_display = query[:80] + "..." if len(query) > 80 else query
    plt.suptitle(f"Query: {query_display}", fontsize=11, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"💾 Saved visualization to: {output_path}")

    # Convert to PIL Image for return
    from io import BytesIO

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    result_image = Image.open(buf)

    plt.close()

    return result_image
