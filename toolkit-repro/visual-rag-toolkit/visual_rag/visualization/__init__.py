"""
Visualization module - Saliency maps and attention visualization.

This module provides:
- Saliency map generation showing query-document relevance
- Attention heatmaps for visual token analysis
"""

from visual_rag.visualization.saliency import (
    create_saliency_overlay,
    generate_saliency_map,
    visualize_search_results,
)

__all__ = [
    "generate_saliency_map",
    "create_saliency_overlay",
    "visualize_search_results",
]
