"""
YAML Loader Module for LLM Meter

Handles loading and scaling of price data from YAML files.
"""

import functools
from pathlib import Path
from typing import Dict, Any

import yaml


# Path to the price sheet
YAML_PATH = Path(__file__).parent / "model_prices.yaml"


@functools.lru_cache(maxsize=1)
def load_yaml_prices() -> Dict[str, Any]:
    """
    Load and scale the price data from the YAML file.

    Scales values from "per-million" to "per single token"
    so later calculations can multiply by raw token counts.

    Returns:
        dict: Nested dictionary with provider -> model -> tier -> price type -> value
    """
    if not YAML_PATH.exists():
        raise FileNotFoundError(f"Price sheet not found: {YAML_PATH}")

    with YAML_PATH.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Check if the YAML is already in the new format
    if any(
        isinstance(raw.get(key), dict)
        and any(isinstance(v, dict) for v in raw[key].values())
        for key in raw
    ):
        # New format: provider -> model -> tier -> price type -> value
        result = {}
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
