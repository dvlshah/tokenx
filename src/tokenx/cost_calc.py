"""
LLM Cost Calculator (YAML-driven, provider-aware)
================================================

- Loads model prices from `model_prices.yaml` at import-time
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


# For backward compatibility with existing code
PRICE_TABLE = load_yaml_prices()
PRICE_PER_TOKEN = {
    model: tiers["sync"]
    for model, tiers in PRICE_TABLE.get("openai", {}).items()
    if "sync" in tiers
}


class CostCalculator:
    """Base cost calculator for any LLM provider."""

    @staticmethod
    def for_provider(
        provider_name: str,
        model: str,
        *,
        tier: str = "sync",
        enable_caching: bool = True,
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

        # For OpenAI, return the OpenAICostCalculator for backward compatibility
        if provider_name == "openai":
            return OpenAICostCalculator(model, tier=tier, enable_caching=enable_caching)

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
        tier: str = "sync",
        enable_caching: bool = True,
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

    def costed(self, expects_usage: bool = False) -> Callable:
        """
        Decorator to add cost tracking to a function.

        Args:
            expects_usage: Whether the function returns usage data
                If True, expects (result, usage) tuple
                If False, tries to extract usage from result

        Returns:
            Callable: Decorator function
        """

        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                result = fn(*args, **kwargs)

                if expects_usage:
                    # Function returns (result, usage) tuple
                    usage_data = result[1] if len(result) > 1 else None
                    if usage_data is None:
                        raise ValueError("Function did not return usage data")

                    cost = self.cost_from_usage(usage_data)
                    input_tokens, output_tokens, cached_tokens = (
                        self.provider.extract_tokens(usage_data)
                    )
                else:
                    # Try to extract usage from result
                    cost = self.cost_from_response(result)
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
        tier: str = "sync",
        enable_caching: bool = True,
    ):
        """
        Initialize an OpenAI cost calculator.

        Args:
            model: Model name (e.g., "gpt-4o", "gpt-3.5-turbo")
            tier: Pricing tier (default: "sync")
            enable_caching: Whether to discount cached tokens (default: True)
        """
        super().__init__(
            provider_name="openai",
            model=model,
            tier=tier,
            enable_caching=enable_caching,
        )

        # Set up the tokenizer for backward compatibility
        self.enc = self.provider.get_encoding_for_model(model)

    def _count(self, text: str) -> int:
        """Return BPE token count for text (backward compatibility)."""
        return self.provider.count_tokens(text, self.model)

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
