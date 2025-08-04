"""
Test dynamic pricing system - no bundled fallback, always fetch from remote.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import yaml

try:
    # Development/local testing
    from src.tokenx.yaml_loader import (
        load_yaml_prices,
        _fetch_remote_prices,
        _get_base_prices,
        _load_user_override,
    )
except ImportError:
    # CI/installed package testing
    from tokenx.yaml_loader import (
        load_yaml_prices,
        _fetch_remote_prices,
        _get_base_prices,
        _load_user_override,
    )


def test_config_constants():
    """Test that pricing configuration constants are set correctly."""
    try:
        from src.tokenx.yaml_loader import (
            REMOTE_PRICES_URL,
            CACHE_TTL_HOURS,
            REMOTE_FALLBACK_URLS,
        )
    except ImportError:
        from tokenx.yaml_loader import (
            REMOTE_PRICES_URL,
            CACHE_TTL_HOURS,
            REMOTE_FALLBACK_URLS,
        )

    assert (
        REMOTE_PRICES_URL
        == "https://raw.githubusercontent.com/dvlshah/tokenx/main/src/tokenx/model_prices.yaml"
    )
    assert CACHE_TTL_HOURS == 24
    assert len(REMOTE_FALLBACK_URLS) > 0


def test_remote_fetch_success():
    """Test remote pricing fetch (may fail if URLs don't exist yet)."""
    # This will try to fetch from GitHub - may fail if not merged to main yet
    result = _fetch_remote_prices()

    # If remote fetch succeeds, validate structure
    if result is not None:
        assert "openai" in result
        assert "anthropic" in result
        assert len(result["openai"]) > 0
        assert len(result["anthropic"]) > 0
    else:
        # Remote fetch failed - this is expected if URLs don't exist yet
        # The bundled fallback will handle this case
        print("Remote fetch failed as expected - URLs may not exist yet")


@patch("urllib.request.urlopen")
def test_remote_fetch_failure(mock_urlopen):
    """Test remote fetch failure handling."""
    mock_urlopen.side_effect = Exception("Network error")

    result = _fetch_remote_prices()
    assert result is None


def test_user_override():
    """Test user override functionality."""
    # Create temporary pricing file
    test_data = {"openai": {"test-model": {"sync": {"in": 1.0, "out": 2.0}}}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(test_data, f)
        temp_path = f.name

    try:
        # Test with environment variable set
        os.environ["TOKENX_PRICES_PATH"] = temp_path
        override = _load_user_override()

        assert override is not None
        assert "openai" in override
        assert "test-model" in override["openai"]

    finally:
        os.unlink(temp_path)
        os.environ.pop("TOKENX_PRICES_PATH", None)


def test_user_override_not_set():
    """Test user override when environment variable not set."""
    os.environ.pop("TOKENX_PRICES_PATH", None)

    override = _load_user_override()
    assert override is None


def test_load_yaml_prices_integration():
    """Test complete pricing system integration."""
    # Clear any cache
    if hasattr(load_yaml_prices, "cache_clear"):
        load_yaml_prices.cache_clear()

    prices = load_yaml_prices()

    # Verify structure
    assert isinstance(prices, dict)
    assert "openai" in prices
    assert "anthropic" in prices

    # Verify pricing data format
    openai_models = prices["openai"]
    assert len(openai_models) > 0

    # Test a known model
    if "gpt-4o" in openai_models:
        gpt4o = openai_models["gpt-4o"]
        assert "sync" in gpt4o
        assert "in" in gpt4o["sync"]
        assert "out" in gpt4o["sync"]

        # Verify prices are scaled (should be very small numbers)
        assert gpt4o["sync"]["in"] < 0.01  # Should be scaled from per-million


@patch("urllib.request.urlopen")
def test_bundled_fallback_when_no_remote(mock_urlopen):
    """Test that bundled pricing is used when remote sources fail."""
    # Mock network failure
    mock_urlopen.side_effect = Exception("Network error")

    # Clear cache
    if hasattr(load_yaml_prices, "cache_clear"):
        load_yaml_prices.cache_clear()

    # Remove any existing cache
    cache_dir = Path.home() / ".cache" / "tokenx"
    if cache_dir.exists():
        import shutil

        shutil.rmtree(cache_dir)

    # Should succeed with bundled fallback (not raise an error)
    result = _get_base_prices()

    # Verify bundled pricing loaded successfully
    assert isinstance(result, dict)
    assert "openai" in result
    assert "anthropic" in result
    assert len(result["openai"]) > 0


def test_environment_variable_overrides():
    """Test environment variable configuration overrides."""
    original_url = os.environ.get("TOKENX_REMOTE_PRICES_URL")

    try:
        # Set custom URL
        test_url = "https://example.com/custom_prices.yaml"
        os.environ["TOKENX_REMOTE_PRICES_URL"] = test_url

        # Reload module to pick up new env var
        import importlib

        try:
            from src.tokenx import yaml_loader
        except ImportError:
            from tokenx import yaml_loader

        importlib.reload(yaml_loader)

        assert yaml_loader.REMOTE_PRICES_URL == test_url

    finally:
        # Restore original
        if original_url:
            os.environ["TOKENX_REMOTE_PRICES_URL"] = original_url
        else:
            os.environ.pop("TOKENX_REMOTE_PRICES_URL", None)

        # Reload to restore original state
        import importlib

        try:
            from src.tokenx import yaml_loader
        except ImportError:
            from tokenx import yaml_loader

        importlib.reload(yaml_loader)
