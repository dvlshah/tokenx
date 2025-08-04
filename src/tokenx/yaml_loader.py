"""
YAML Loader Module for TokenX

Handles loading and scaling of price data from remote YAML files with local caching.
Always fetches latest pricing from remote source - no bundled fallback files.
"""

import functools
import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


def deep_merge(source: dict, destination: dict) -> dict:
    """Deeply merges source dict into destination dict."""
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            destination[key] = value
    return destination


REMOTE_PRICES_URL = os.environ.get(
    "TOKENX_REMOTE_PRICES_URL",
    "https://raw.githubusercontent.com/dvlshah/tokenx/main/src/tokenx/model_prices.yaml",
)
REMOTE_FALLBACK_URLS = [
    "https://cdn.jsdelivr.net/gh/dvlshah/tokenx@main/src/tokenx/model_prices.yaml"
]
CACHE_TTL_HOURS = int(os.environ.get("TOKENX_CACHE_TTL_HOURS", "24"))
USER_OVERRIDE_ENV_VAR = "TOKENX_PRICES_PATH"

CACHE_DIR = Path(
    os.environ.get("TOKENX_CACHE_DIR", str(Path.home() / ".cache" / "tokenx"))
).expanduser()
CACHED_PRICES_PATH = CACHE_DIR / os.environ.get(
    "TOKENX_CACHE_FILENAME", "model_prices.yaml"
)


def _ensure_cache_dir() -> None:
    """Ensure the cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _is_cache_valid() -> bool:
    """Check if the cached file exists and is within TTL."""
    if not CACHED_PRICES_PATH.exists():
        return False

    # Check if cache is within TTL
    cache_age_hours = (time.time() - CACHED_PRICES_PATH.stat().st_mtime) / 3600
    return cache_age_hours < CACHE_TTL_HOURS


def _fetch_remote_prices(url: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch prices from remote URL and return parsed YAML."""
    if url is None:
        url = REMOTE_PRICES_URL

    try:
        print(f"tokenx: Fetching latest model prices from {url}...")
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read().decode("utf-8")
            return yaml.safe_load(content)
    except Exception as e:
        print(f"tokenx: Failed to fetch from {url}: {e}")
        return None


def _save_to_cache(prices_data: Dict[str, Any]) -> None:
    """Save prices data to local cache."""
    try:
        _ensure_cache_dir()
        with CACHED_PRICES_PATH.open("w", encoding="utf-8") as f:
            yaml.dump(prices_data, f, default_flow_style=False)
        print("tokenx: Cached latest prices locally.")
    except Exception as e:
        print(f"tokenx: Failed to cache prices: {e}")


def _load_from_cache() -> Optional[Dict[str, Any]]:
    """Load prices from local cache."""
    try:
        if CACHED_PRICES_PATH.exists():
            print("tokenx: Using cached model prices.")
            with CACHED_PRICES_PATH.open(encoding="utf-8") as f:
                return yaml.safe_load(f)
    except Exception as e:
        print(f"tokenx: Failed to load cached prices: {e}")
    return None


def _load_user_override() -> Optional[Dict[str, Any]]:
    """Load user override prices from environment variable path."""
    override_path = os.environ.get(USER_OVERRIDE_ENV_VAR)
    if not override_path:
        return None

    override_file = Path(override_path)
    if not override_file.exists():
        print(f"tokenx: Warning - User override file not found: {override_path}")
        return None

    try:
        print(f"tokenx: Loading user override from {override_path}")
        with override_file.open(encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"tokenx: Warning - Failed to load user override: {e}")
        return None


def _get_base_prices() -> Dict[str, Any]:
    """Get base prices: cache → remote → fallback URLs → error."""
    # Check if cache is valid
    if _is_cache_valid():
        cached_prices = _load_from_cache()
        if cached_prices is not None:
            return cached_prices

    # Try to fetch from primary remote URL
    remote_prices = _fetch_remote_prices()
    if remote_prices is not None:
        _save_to_cache(remote_prices)
        return remote_prices

    # Try fallback URLs
    for fallback_url in REMOTE_FALLBACK_URLS:
        print("tokenx: Trying fallback URL...")
        remote_prices = _fetch_remote_prices(fallback_url)
        if remote_prices is not None:
            _save_to_cache(remote_prices)
            return remote_prices

    # Use stale cache if remote fetch failed
    if CACHED_PRICES_PATH.exists():
        print("tokenx: All remote sources failed, using stale cache.")
        cached_prices = _load_from_cache()
        if cached_prices is not None:
            return cached_prices

    # Fall back to bundled pricing file
    try:
        import pkg_resources  # type: ignore

        bundled_path = pkg_resources.resource_filename("tokenx", "model_prices.yaml")
        print("tokenx: Loading bundled pricing as last resort.")
        with open(bundled_path, "r", encoding="utf-8") as f:
            bundled_prices = yaml.safe_load(f)
            if bundled_prices:
                return bundled_prices
    except Exception:
        # Try direct path approach
        try:
            from pathlib import Path

            current_dir = Path(__file__).parent
            bundled_path = current_dir / "model_prices.yaml"
            if bundled_path.exists():
                print("tokenx: Loading bundled pricing as last resort.")
                with open(bundled_path, "r", encoding="utf-8") as f:
                    bundled_prices = yaml.safe_load(f)
                    if bundled_prices:
                        return bundled_prices
        except Exception:
            pass

    # No pricing data available
    raise RuntimeError(
        "tokenx: Unable to load pricing data. "
        "No remote sources available and no cached data found. "
        "Check your internet connection or set TOKENX_PRICES_PATH to a local pricing file."
    )


@functools.lru_cache(maxsize=1)
def load_yaml_prices() -> Dict[str, Any]:
    """
    Load and scale the price data from remote sources with local caching.

    Priority chain:
    1. User override file (via TOKENX_PRICES_PATH env var) - highest priority
    2. Local cache (valid for configured TTL hours)
    3. Remote source of truth (GitHub repository)
    4. Fallback URLs
    5. Stale cache (if all remotes fail)

    Scales values from "per-million" to "per single token"
    so later calculations can multiply by raw token counts.

    Returns:
        dict: Nested dictionary with provider -> model -> tier -> price type -> value

    Raises:
        RuntimeError: If no pricing data can be loaded from any source
    """
    # Step 1: Get base prices (cache → remote → fallback URLs → stale cache)
    raw_prices = _get_base_prices()

    # Step 2: Apply user override if available (highest priority)
    user_override = _load_user_override()
    if user_override is not None:
        raw_prices = deep_merge(user_override, raw_prices.copy())

    # Step 3: Process and scale the final prices
    return _process_and_scale_prices(raw_prices)


def _process_and_scale_prices(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Process raw prices and scale them from per-million to per-token."""
    # Check if the YAML is already in the new format
    if any(
        isinstance(raw.get(key), dict)
        and any(isinstance(v, dict) for v in raw[key].values())
        for key in raw
    ):
        # New format: provider -> model -> tier -> price type -> value
        result: Dict[str, Any] = {}
        for provider, models in raw.items():
            result[provider] = {}
            for model, tiers in models.items():
                result[provider][model] = {}
                for tier, prices in tiers.items():
                    result[provider][model][tier] = _scale_prices(prices)
        return result

    # Old format: model -> tier -> price type -> value
    # Add "openai" as the provider for backward compatibility
    return {
        "openai": {
            model: {tier: _scale_prices(spec) for tier, spec in tiers.items()}
            for model, tiers in raw.items()
        }
    }


def _scale_prices(prices: Dict[str, float]) -> Dict[str, float]:
    """
    Scale price values from per-million to per-token.

    Args:
        prices: Dictionary of price types to values

    Returns:
        dict: Scaled prices
    """
    return {k: (v / 1e6 if v is not None else None) for k, v in prices.items()}
