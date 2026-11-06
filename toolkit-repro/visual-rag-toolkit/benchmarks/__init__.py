"""
Benchmark utilities and dataset loaders used by the demo UI.

Note: The `benchmarks/` folder is primarily for research/evaluation scripts, but the
Streamlit demo imports some loaders/metrics from here. Making this directory a
package (via `__init__.py`) ensures imports like `benchmarks.vidore_tatdqa_test`
work in Docker/Spaces environments.
"""

__all__ = []
