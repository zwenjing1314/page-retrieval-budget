"""
Configuration utilities for Visual RAG Toolkit.

Provides:
- YAML configuration loading with caching
- Environment variable overrides
- Convenience getters for common settings
"""

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Global config cache (raw YAML only; env overrides applied on demand)
_raw_config_cache: Optional[Dict[str, Any]] = None
_raw_config_cache_path: Optional[str] = None


def _env_qdrant_url() -> Optional[str]:
    """Get Qdrant URL from environment. Prefers QDRANT_URL."""
    return os.getenv("QDRANT_URL") or os.getenv("SIGIR_QDRANT_URL")  # legacy fallback


def _env_qdrant_api_key() -> Optional[str]:
    """Get Qdrant API key from environment. Prefers QDRANT_API_KEY."""
    return os.getenv("QDRANT_API_KEY") or os.getenv("SIGIR_QDRANT_KEY")  # legacy fallback


def load_config(
    config_path: Optional[str] = None,
    force_reload: bool = False,
    apply_env_overrides: bool = True,
) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Uses caching to avoid repeated file I/O.
    Environment variables can override config values.

    Args:
        config_path: Path to config file (auto-detected if None)
        force_reload: Bypass cache and reload from file

    Returns:
        Configuration dictionary
    """
    global _raw_config_cache, _raw_config_cache_path

    # Determine the effective config path (used for caching)
    effective_path: Optional[str] = None

    # Find config file
    if config_path is None:
        config_path = os.getenv("VISUALRAG_CONFIG")

        if config_path is None:
            # Check common locations
            search_paths = [
                Path.cwd() / "config.yaml",
                Path.cwd() / "visual_rag.yaml",
                Path.home() / ".visual_rag" / "config.yaml",
            ]

            for path in search_paths:
                if path.exists():
                    config_path = str(path)
                    break
    effective_path = str(config_path) if config_path else None

    # Return cached raw config if available.
    # - If caller doesn't specify a path (effective_path is None), use whatever was
    #   loaded most recently (common pattern in apps).
    # - If a path is specified, only reuse cache when it matches.
    if (
        _raw_config_cache is not None
        and not force_reload
        and (effective_path is None or _raw_config_cache_path == effective_path)
    ):
        cfg = copy.deepcopy(_raw_config_cache)
        return _apply_env_overrides(cfg) if apply_env_overrides else cfg

    # Load YAML if file exists
    config = {}
    if config_path and Path(config_path).exists():
        try:
            import yaml

            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}

            logger.info(f"Loaded config from: {config_path}")
        except ImportError:
            logger.warning("PyYAML not installed, using environment variables only")
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")

    # Cache RAW config (no env overrides)
    _raw_config_cache = copy.deepcopy(config)
    _raw_config_cache_path = effective_path

    # Return resolved or raw depending on caller preference
    cfg = copy.deepcopy(config)
    return _apply_env_overrides(cfg) if apply_env_overrides else cfg


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides."""

    env_mappings = {
        # Qdrant
        "QDRANT_URL": ["qdrant", "url"],
        "QDRANT_API_KEY": ["qdrant", "api_key"],
        "QDRANT_COLLECTION": ["qdrant", "collection"],
        # Model
        "VISUALRAG_MODEL": ["model", "name"],
        "COLPALI_MODEL_NAME": ["model", "name"],  # Alias
        "EMBEDDING_BATCH_SIZE": ["model", "batch_size"],
        # Cloudinary
        "CLOUDINARY_CLOUD_NAME": ["cloudinary", "cloud_name"],
        "CLOUDINARY_API_KEY": ["cloudinary", "api_key"],
        "CLOUDINARY_API_SECRET": ["cloudinary", "api_secret"],
        # Processing
        "PDF_DPI": ["processing", "dpi"],
        "JPEG_QUALITY": ["processing", "jpeg_quality"],
        # Search
        "SEARCH_STRATEGY": ["search", "strategy"],
        "PREFETCH_K": ["search", "prefetch_k"],
        # Special token handling
        "VISUALRAG_INCLUDE_SPECIAL_TOKENS": ["embedding", "include_special_tokens"],
    }

    for env_var, path in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            # Navigate to the right place in config
            current = config
            for key in path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Convert value to appropriate type
            final_key = path[-1]
            if final_key in current:
                existing_type = type(current[final_key])
                # Use `is` for type comparisons (Ruff E721).
                if existing_type is bool:
                    value = value.lower() in ("true", "1", "yes", "on")
                elif existing_type is int:
                    value = int(value)
                elif existing_type is float:
                    value = float(value)

            current[final_key] = value
            logger.debug(f"Config override: {'.'.join(path)} = {value}")

    return config


def get(key: str, default: Any = None) -> Any:
    """
    Get a configuration value by dot-notation path.

    Examples:
        >>> get("qdrant.url")
        >>> get("model.name", "vidore/colSmol-500M")
        >>> get("search.strategy", "multi_vector")
    """
    config = load_config(apply_env_overrides=True)

    keys = key.split(".")
    current = config

    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default

    return current


def get_section(section: str, *, apply_env_overrides: bool = True) -> Dict[str, Any]:
    """Get an entire configuration section."""
    config = load_config(apply_env_overrides=apply_env_overrides)
    return config.get(section, {})


# Convenience getters
def get_qdrant_config() -> Dict[str, Any]:
    """Get Qdrant configuration with defaults."""
    return {
        "url": get("qdrant.url", _env_qdrant_url()),
        "api_key": get("qdrant.api_key", _env_qdrant_api_key()),
        "collection": get("qdrant.collection", "visual_documents"),
    }


def get_model_config() -> Dict[str, Any]:
    """Get model configuration with defaults."""
    return {
        "name": get("model.name", "vidore/colSmol-500M"),
        "batch_size": get("model.batch_size", 4),
        "device": get("model.device", "auto"),
    }


def get_processing_config() -> Dict[str, Any]:
    """Get processing configuration with defaults."""
    return {
        "dpi": get("processing.dpi", 140),
        "jpeg_quality": get("processing.jpeg_quality", 95),
        "page_batch_size": get("processing.page_batch_size", 50),
    }


def get_search_config() -> Dict[str, Any]:
    """Get search configuration with defaults."""
    return {
        "strategy": get("search.strategy", "multi_vector"),
        "prefetch_k": get("search.prefetch_k", 200),
        "top_k": get("search.top_k", 10),
    }
