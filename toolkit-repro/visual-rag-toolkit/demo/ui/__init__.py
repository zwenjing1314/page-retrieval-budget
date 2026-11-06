"""UI components for the demo app."""

from demo.ui.benchmark import render_benchmark_tab
from demo.ui.header import render_header
from demo.ui.playground import render_playground_tab
from demo.ui.sidebar import render_sidebar
from demo.ui.upload import render_upload_tab

__all__ = [
    "render_header",
    "render_sidebar",
    "render_upload_tab",
    "render_playground_tab",
    "render_benchmark_tab",
]
