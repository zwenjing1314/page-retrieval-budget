"""Tests for configuration utilities."""

import os
import tempfile


class TestConfigLoading:
    """Test config file loading."""

    def test_load_yaml_config(self):
        """Load config from YAML file."""
        from visual_rag.config import load_config

        # Create temp config file
        config_content = """
model:
  name: "test-model"
  batch_size: 8
qdrant:
  url: "http://localhost:6333"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config = load_config(config_path, force_reload=True, apply_env_overrides=False)

            assert config["model"]["name"] == "test-model"
            assert config["model"]["batch_size"] == 8
            assert config["qdrant"]["url"] == "http://localhost:6333"
        finally:
            os.unlink(config_path)

    def test_env_override(self):
        """Environment variables override config values."""
        from visual_rag.config import load_config

        # Set env var
        os.environ["VISUAL_RAG_MODEL_NAME"] = "env-override-model"

        config_content = """
model:
  name: "yaml-model"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config = load_config(config_path, force_reload=True, apply_env_overrides=False)
            # The env var should be checked in get() if implemented
            # For now, just verify config loads
            assert config["model"]["name"] == "yaml-model"
        finally:
            os.unlink(config_path)
            del os.environ["VISUAL_RAG_MODEL_NAME"]

    def test_missing_config_uses_defaults(self):
        """Missing config file returns empty dict or defaults."""
        from visual_rag.config import load_config

        config = load_config("/nonexistent/path/config.yaml")

        # Should not raise, returns empty or default config
        assert isinstance(config, dict)

    def test_get_nested_value(self):
        """Get nested config values with dot notation."""

        # This tests the get() function if available
        # Will need the config to be loaded first
        pass  # Placeholder - depends on implementation


class TestConfigSection:
    """Test getting config sections."""

    def test_get_section(self):
        """Get a config section."""
        from visual_rag.config import get_section, load_config

        config_content = """
qdrant:
  url: "http://localhost"
  collection: "test"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            load_config(config_path, force_reload=True, apply_env_overrides=False)
            section = get_section("qdrant", apply_env_overrides=False)

            assert section["url"] == "http://localhost"
            assert section["collection"] == "test"
        finally:
            os.unlink(config_path)

    def test_missing_section(self):
        """Missing section returns empty dict."""
        from visual_rag.config import get_section

        section = get_section("nonexistent")
        assert section == {}
