"""
LLM Cost Calculator (YAML-driven, provider-aware)
================================================

- Loads model prices dynamically from remote sources with local caching
- Supports multiple providers (OpenAI, Anthropic, Gemini)
- Handles pricing tiers and prompt-caching discounts
- Provides:
    · calculate_cost()     – direct cost calculation
    · cost_from_usage()    – cost from response usage data
    · costed() decorator   – wrap any fn and attach USD spend

Dependencies
------------
  pip install pyyaml tiktoken
"""

from __future__ import annotations

import functools
from typing import Any, Dict, Callable, Union

from .providers import ProviderRegistry
from .yaml_loader import load_yaml_prices
from .errors import TokenExtractionError
from .constants import DEFAULT_TIER, DEFAULT_ENABLE_CACHING, PROVIDER_OPENAI


# Global cache for pricing data - loaded lazily on first use
_PRICE_TABLE_CACHE = None
_PRICE_PER_TOKEN_CACHE = None


def _ensure_pricing_loaded():
    """Ensure pricing data is loaded, with helpful error message for new users."""
    global _PRICE_TABLE_CACHE, _PRICE_PER_TOKEN_CACHE

    if _PRICE_TABLE_CACHE is None:
        try:
            _PRICE_TABLE_CACHE = load_yaml_prices()
            # Legacy OpenAI-only pricing dict for backward compatibility
            # PRICE_PER_TOKEN was designed when TokenX only supported OpenAI
            # and expects a flat {model: prices} structure, not {provider: {model: prices}}
            _PRICE_PER_TOKEN_CACHE = {
                model: tiers[DEFAULT_TIER]
                for model, tiers in _PRICE_TABLE_CACHE.get(PROVIDER_OPENAI, {}).items()
                if DEFAULT_TIER in tiers
            }
        except Exception as e:
            # Provide helpful error message for new users
            raise RuntimeError(
                f"Failed to load pricing data: {e}\n\n"
                "💡 Quick Fix for New Users:\n"
                "1. Check your internet connection for automatic pricing updates\n"
                "2. Or set custom pricing: export TOKENX_PRICES_PATH=/path/to/pricing.yaml\n"
                "3. See https://github.com/dvlshah/tokenx#dynamic-pricing for details\n"
            ) from e


# For backward compatibility - these are now lazy-loaded module attributes
def _get_price_table():
    """Backward compatibility: lazy-loaded pricing table."""
    _ensure_pricing_loaded()
    return _PRICE_TABLE_CACHE


def _get_price_per_token():
    """Backward compatibility: lazy-loaded OpenAI pricing."""
    _ensure_pricing_loaded()
    return _PRICE_PER_TOKEN_CACHE


# Module-level lazy attributes using __getattr__ (Python 3.7+)
def __getattr__(name):
    """Lazy loading for backward compatibility attributes."""
    if name == "PRICE_TABLE":
        return _get_price_table()
    elif name == "PRICE_PER_TOKEN":
        return _get_price_per_token()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


class CostCalculator:
    """Base cost calculator for any LLM provider."""

    @staticmethod
    def for_provider(
        provider_name: str,
        model: str,
        *,
        tier: str = DEFAULT_TIER,
        enable_caching: bool = DEFAULT_ENABLE_CACHING,
    ) -> Union[CostCalculator, "OpenAICostCalculator"]:
        """
        Factory method to create a cost calculator for a specific provider.

        Args:
            provider_name: Provider name (e.g., "openai", "anthropic", "gemini")
            model: Model name (e.g., "gpt-4o", "claude-3.5-sonnet")
            tier: Pricing tier (default: "sync")
            enable_caching: Whether to discount cached tokens (default: True)

        Returns:
            CostCalculator: A cost calculator for the specified provider
        """
        provider = ProviderRegistry.get_provider(provider_name)
        if provider is None:
            raise ValueError(f"Provider {provider_name!r} not found")

        # Registry-based factory dispatch - no hardcoded conditionals
        calculator_class = ProviderRegistry.get_calculator_class(provider_name)

        if calculator_class:
            return calculator_class(model, tier=tier, enable_caching=enable_caching)  # type: ignore

        return CostCalculator(
            provider_name=provider_name,
            model=model,
            tier=tier,
            enable_caching=enable_caching,
        )

    def __init__(
        self,
        provider_name: str,
        model: str,
        *,
        tier: str = DEFAULT_TIER,
        enable_caching: bool = DEFAULT_ENABLE_CACHING,
    ):
        """
        Initialize a cost calculator for any LLM provider.

        Args:
            provider_name: Provider name (e.g., "openai", "anthropic", "gemini")
            model: Model name (e.g., "gpt-4o", "claude-3.5-sonnet")
            tier: Pricing tier (default: "sync")
            enable_caching: Whether to discount cached tokens (default: True)
        """
        self.provider_name = provider_name
        self.model = model
        self.tier = tier
        self.enable_caching = enable_caching

        # Get the provider implementation
        self.provider = ProviderRegistry.get_provider(provider_name)
        if self.provider is None:
            raise ValueError(f"Provider {provider_name!r} not found")

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, cached_tokens: int = 0
    ) -> float:
        """
        Calculate cost from token counts.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached tokens (default: 0)

        Returns:
            float: Cost in USD
        """
        if self.provider is None:
            raise ValueError("Provider not initialized")
        return self.provider.calculate_cost(
            self.model,
            input_tokens,
            output_tokens,
            cached_tokens=cached_tokens if self.enable_caching else 0,
            tier=self.tier,
        )

    def cost_from_usage(self, usage: Dict[str, Any]) -> float:
        """
        Calculate cost from a usage object.

        Args:
            usage: Usage data from provider's response

        Returns:
            float: Cost in USD
        """
        if self.provider is None:
            raise ValueError("Provider not initialized")
        input_tokens, output_tokens, cached_tokens = self.provider.extract_tokens(usage)
        return self.calculate_cost(input_tokens, output_tokens, cached_tokens)

    def cost_from_response(self, response: Any) -> float:
        """
        Calculate cost from a full response object.

        Args:
            response: Response object from provider

        Returns:
            float: Cost in USD
        """
        if self.provider is None:
            raise ValueError("Provider not initialized")

        # Use the provider adapter to extract tokens (handles special cases)
        try:
            input_tokens, output_tokens, cached_tokens = self.provider.extract_tokens(
                response
            )
            # Check if provider supports enhanced cost calculation with response object
            if (
                hasattr(self.provider, "calculate_cost")
                and "response" in self.provider.calculate_cost.__code__.co_varnames
            ):
                return self.provider.calculate_cost(
                    self.model,
                    input_tokens,
                    output_tokens,
                    cached_tokens,
                    self.tier,
                    response,
                )
            else:
                return self.calculate_cost(input_tokens, output_tokens, cached_tokens)
        except Exception:
            # Fallback to old method for backward compatibility
            usage_data = None

            # Try to extract usage from response object attributes or dict keys
            if hasattr(response, "usage"):
                usage_data = response.usage
            elif isinstance(response, dict) and "usage" in response:
                usage_data = response["usage"]

            # If no valid usage data structure was found, raise an error
            if usage_data is None:
                # Raise TokenExtractionError directly here as the structure is wrong
                # Use the provider associated with this calculator instance
                raise TokenExtractionError(
                    "Response object does not contain expected 'usage' attribute or key.",
                    self.provider_name,
                    type(response).__name__,
                )

            # Proceed to calculate cost using the extracted usage data
            return self.cost_from_usage(usage_data)

    def costed(self, expects_usage: bool = False) -> Callable[..., Any]:
        """
        Decorator to add cost tracking to a function.

        Args:
            expects_usage: Whether the function returns usage data
                If True, expects (result, usage) tuple
                If False, tries to extract usage from result

        Returns:
            Callable: Decorator function
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = fn(*args, **kwargs)

                if expects_usage:
                    # Function returns (result, usage) tuple
                    usage_data = result[1] if len(result) > 1 else None
                    if usage_data is None:
                        raise ValueError("Function did not return usage data")

                    cost = self.cost_from_usage(usage_data)
                    if self.provider is None:
                        raise ValueError("Provider not initialized")
                    input_tokens, output_tokens, cached_tokens = (
                        self.provider.extract_tokens(usage_data)
                    )
                else:
                    # Try to extract usage from result
                    cost = self.cost_from_response(result)
                    if self.provider is None:
                        raise ValueError("Provider not initialized")
                    input_tokens, output_tokens, cached_tokens = (
                        self.provider.extract_tokens(
                            result.usage if hasattr(result, "usage") else result
                        )
                    )

                return {
                    "provider": self.provider_name,
                    "model": self.model,
                    "tier": self.tier,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_tokens": cached_tokens,
                    "usd": round(cost, 6),
                }

            return wrapper

        return decorator


# For backward compatibility
class OpenAICostCalculator(CostCalculator):
    """Backward-compatible OpenAI cost calculator."""

    def __init__(
        self,
        model: str,
        *,
        tier: str = DEFAULT_TIER,
        enable_caching: bool = DEFAULT_ENABLE_CACHING,
    ):
        """
        Initialize an OpenAI cost calculator.

        Args:
            model: Model name (e.g., "gpt-4o", "gpt-3.5-turbo")
            tier: Pricing tier (default: "sync")
            enable_caching: Whether to discount cached tokens (default: True)
        """
        super().__init__(
            provider_name=PROVIDER_OPENAI,
            model=model,
            tier=tier,
            enable_caching=enable_caching,
        )

        # Set up the tokenizer for backward compatibility
        if hasattr(self.provider, "get_encoding_for_model"):
            self.enc = self.provider.get_encoding_for_model(model)  # type: ignore
        else:
            raise AttributeError("Provider does not support encoding")

    def _count(self, text: str) -> int:
        """Return BPE token count for text (backward compatibility)."""
        if hasattr(self.provider, "count_tokens"):
            return self.provider.count_tokens(text, self.model)  # type: ignore
        else:
            raise AttributeError("Provider does not support token counting")

    def blended_cost(
        self,
        prompt: str,
        completion: str,
        cached_prompt_tokens: int = 0,
    ) -> float:
        """
        Calculate cost from raw strings (backward compatibility).

        Args:
            prompt: Prompt text
            completion: Completion text
            cached_prompt_tokens: Number of cached tokens

        Returns:
            float: Cost in USD
        """
        return self.calculate_cost(
            self._count(prompt),
            self._count(completion),
            cached_prompt_tokens,
        )


# Register the OpenAI calculator class for backward compatibility
ProviderRegistry.register_calculator_class(PROVIDER_OPENAI, OpenAICostCalculator)
