"""Results file handling utilities."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_results_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def get_available_results() -> List[Path]:
    results_dir = Path(__file__).parent.parent / "results"
    if not results_dir.exists():
        return []
    results = []
    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            for f in subdir.glob("*.json"):
                if "index_failures" not in f.name:
                    results.append(f)
    return sorted(results, key=lambda x: x.stat().st_mtime, reverse=True)


def find_main_result_file(collection: str, mode: str) -> Optional[Path]:
    results = get_available_results()
    for r in results:
        if collection in str(r) and mode in r.name:
            if "__vidore_" not in r.name:
                return r
    return results[0] if results else None
