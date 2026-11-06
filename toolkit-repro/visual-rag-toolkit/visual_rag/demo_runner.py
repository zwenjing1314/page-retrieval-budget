"""
Launch the Streamlit demo from an installed package.

Why:
- After `pip install visual-rag-toolkit`, the repo layout isn't present.
- We package the `demo/` module and expose `visual_rag.demo()` + `visual-rag-demo`.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def demo(
    *,
    host: str = "0.0.0.0",
    port: int = 7860,
    headless: bool = True,
    open_browser: bool = False,
    extra_args: Optional[list[str]] = None,
) -> int:
    """
    Launch the Streamlit demo UI.

    Requirements:
    - `visual-rag-toolkit[ui,qdrant,embedding,pdf]` (or `visual-rag-toolkit[all]`)

    Returns:
        Streamlit process exit code.
    """
    try:
        import streamlit  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Streamlit is not installed. Install with:\n"
            '  pip install "visual-rag-toolkit[ui,qdrant,embedding,pdf]"'
        ) from e

    # Resolve the installed demo entrypoint path.
    mod = importlib.import_module("demo.app")
    app_path = Path(getattr(mod, "__file__", "")).resolve()
    if not app_path.exists():  # pragma: no cover
        raise RuntimeError("Could not locate installed demo app (demo.app).")

    # Build a stable Streamlit invocation.
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    cmd += ["--server.address", str(host)]
    cmd += ["--server.port", str(int(port))]
    # headless=true prevents browser from auto-opening; open_browser overrides
    should_be_headless = headless and not open_browser
    cmd += ["--server.headless", "true" if should_be_headless else "false"]
    cmd += ["--browser.gatherUsageStats", "false"]
    cmd += ["--server.runOnSave", "false"]

    if extra_args:
        cmd += list(extra_args)

    env = os.environ.copy()
    # Make sure the demo doesn't spam internal Streamlit warnings in logs.
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    print("Launching Streamlit demo:", " ".join(cmd), file=sys.stderr, flush=True)
    return subprocess.call(cmd, env=env)


def main() -> None:
    p = argparse.ArgumentParser(description="Launch the Visual RAG Toolkit Streamlit demo.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument(
        "--no-headless", action="store_true", help="Run with a browser window (not headless)."
    )
    p.add_argument("--open", action="store_true", help="Open browser automatically.")
    args, unknown = p.parse_known_args()

    rc = demo(
        host=args.host,
        port=args.port,
        headless=(not args.no_headless),
        open_browser=bool(args.open),
        extra_args=unknown,
    )
    raise SystemExit(rc)


if __name__ == "__main__":  # pragma: no cover
    main()
