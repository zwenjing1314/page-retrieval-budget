"""Main entry point for the Visual RAG Toolkit demo application."""

import sys
from pathlib import Path

# Ensure repo root is in sys.path for local development
# (In HF Space / Docker, PYTHONPATH is already set correctly)
_app_dir = Path(__file__).resolve().parent
_repo_root = _app_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dotenv import load_dotenv  # noqa: E402

# Load .env from the repo root (works both locally and in Docker)
if (_repo_root / ".env").exists():
    load_dotenv(_repo_root / ".env")
if (_app_dir / ".env").exists():
    load_dotenv(_app_dir / ".env")

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Visual RAG Toolkit",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from demo.ui.benchmark import render_benchmark_tab  # noqa: E402
from demo.ui.header import render_header  # noqa: E402
from demo.ui.playground import render_playground_tab  # noqa: E402
from demo.ui.sidebar import render_sidebar  # noqa: E402
from demo.ui.upload import render_upload_tab  # noqa: E402


def main():
    render_header()
    render_sidebar()

    tab_upload, tab_playground, tab_benchmark = st.tabs(
        ["📤 Upload", "🎮 Playground", "📊 Benchmarking"]
    )

    with tab_upload:
        render_upload_tab()

    with tab_playground:
        render_playground_tab()

    with tab_benchmark:
        render_benchmark_tab()


if __name__ == "__main__":
    main()
