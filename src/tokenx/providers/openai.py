"""
OpenAI Provider Adapter Implementation

This module implements the OpenAI provider adapter with enhanced error handling.
"""

from typing import Any, Dict, Optional, Tuple

import tiktoken

from .base import ProviderAdapter
from ..yaml_loader import load_yaml_prices
from ..errors import enhance_provider_adapter, TokenExtractionError, PricingError


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI API cost calculation."""

    def __init__(self):
        """Initialize the OpenAI adapter."""
        self._prices = load_yaml_prices().get("openai", {})

    @property
    def provider_name(self) -> str:
        """Return the provider name identifier."""
        return "openai"

    def matches_function(self, func: Any, args: tuple, kwargs: dict) -> bool:
        """
        Determine if this function is from the OpenAI provider.

        Checks for OpenAI client in the function's module or arguments.
        """
        # Check module name for OpenAI indicators
        module_name = func.__module__ if hasattr(func, "__module__") else ""
        if "openai" in module_name.lower():
            return True

        # Check first argument for OpenAI client
        if args and hasattr(args[0], "__class__"):
            class_name = args[0].__class__.__name__
            if "openai" in class_name.lower():
                return True

        # Check kwargs for OpenAI model names
        if "model" in kwargs and isinstance(kwargs["model"], str):
            model = kwargs["model"].lower()
            return (
                model.startswith("gpt-")
                or model.startswith("text-")
                or model.startswith("o")  # For o1, o3, etc.
                or model in self._prices
            )

        return False

    def _normalize_usage(self, usage: Any) -> Dict[str, Any]:
        """
        Normalize usage data from different response formats.

        Args:
            usage: Usage data from response

        Returns:
            dict: Normalized usage data with input_tokens, output_tokens, and cached_tokens
        """
        # Initialize with None to detect if values were actually found
        result = {"input_tokens": None, "output_tokens": None, "cached_tokens": 0}

        # Handle attribute-based access (Pydantic models)
        if hasattr(usage, "__dict__") or hasattr(usage, "__getattr__"):
            # Input tokens (prompt_tokens or input_tokens)
            if hasattr(usage, "prompt_tokens"):
                result["input_tokens"] = usage.prompt_tokens
            elif hasattr(usage, "input_tokens"):
                result["input_tokens"] = usage.input_tokens

            # Output tokens (completion_tokens or output_tokens)
            if hasattr(usage, "completion_tokens"):
                result["output_tokens"] = usage.completion_tokens
            elif hasattr(usage, "output_tokens"):
                result["output_tokens"] = usage.output_tokens

            # Check for cached tokens in prompt_tokens_details or input_tokens_details
            details = None
            if hasattr(usage, "prompt_tokens_details"):
                details = usage.prompt_tokens_details
            elif hasattr(usage, "input_tokens_details"):
                details = usage.input_tokens_details

            if details is not None:
                if hasattr(details, "cached_tokens"):
                    result["cached_tokens"] = details.cached_tokens
                elif hasattr(details, "get") and callable(details.get):
                    result["cached_tokens"] = details.get("cached_tokens", 0)

        # Handle dictionary-based access
        elif isinstance(usage, dict):
            # Input tokens - use None as default to detect missing keys
            result["input_tokens"] = usage.get("prompt_tokens") or usage.get(
                "input_tokens"
            )

            # Output tokens - use None as default
            result["output_tokens"] = usage.get("completion_tokens") or usage.get(
                "output_tokens"
            )

            # Cached tokens
            details = usage.get("prompt_tokens_details", {}) or usage.get(
                "input_tokens_details", {}
            )
            if isinstance(details, dict):
                result["cached_tokens"] = details.get("cached_tokens", 0)

        # Ensure required tokens were found
        if result["input_tokens"] is None or result["output_tokens"] is None:
            raise TokenExtractionError(
                "Could not extract required 'input_tokens' or 'output_tokens' from usage data.",
                self.provider_name,
                type(usage).__name__,
            )

        # Fill in 0 for any Nones that weren't required (like cached_tokens if details missing)
        # or if the initial value was 0 from the get() calls.
        result["input_tokens"] = result["input_tokens"] or 0
        result["output_tokens"] = result["output_tokens"] or 0
        # cached_tokens already defaults to 0

        return result

    def extract_tokens(self, response: Any) -> Tuple[int, int, int]:
        """
        Extract token counts from an OpenAI response object.

        Args:
            response: Provider-specific response object

        Returns:
            tuple: (input_tokens, output_tokens, cached_tokens)

        Raises:
            TokenExtractionError: If token extraction fails
        """
        # Handle different response shapes to extract usage information
        usage = None

        # Try to extract usage from response object
        if hasattr(response, "usage"):
            usage = response.usage
        elif isinstance(response, dict) and "usage" in response:
            usage = response["usage"]
        else:
            # Use the response itself as usage data
            usage = response

        # If we couldn't extract anything meaningful, raise an error
        if usage is None:
            raise TokenExtractionError(
                "Could not extract usage data from response. "
                "Expected 'usage' attribute or key.",
                self.provider_name,
                type(response).__name__,
            )

        # Use normalize_usage to extract token counts
        normalized = self._normalize_usage(usage)
        return (
            normalized["input_tokens"],
            normalized["output_tokens"],
            normalized["cached_tokens"],
        )

    def detect_model(self, func: Any, args: tuple, kwargs: dict) -> Optional[str]:
        """
        Try to identify model name from function and arguments.

        Note: Auto model detection is disabled. Users must explicitly specify the model.

        Args:
            func: The function being called
            args: Function arguments
            kwargs: Function keyword arguments

        Returns:
            str: Model name if provided in kwargs, None otherwise
        """
        # Only check kwargs for explicit model
        if "model" in kwargs and isinstance(kwargs["model"], str):
            return kwargs["model"]

        return None

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        tier: str = "sync",
    ) -> float:
        """
        Calculate cost in USD based on token usage.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached input tokens (default: 0)
            tier: Pricing tier (default: "sync")

        Returns:
            float: Cost in USD

        Raises:
            PricingError: If pricing information is not available
        """
        if model not in self._prices:
            raise PricingError(
                f"Price for model={model!r} not found in YAML",
                self.provider_name,
                model,
                available_models=list(self._prices.keys()),
            )

        if tier not in self._prices[model]:
            raise PricingError(
                f"Price for model={model!r} tier={tier!r} not found in YAML",
                self.provider_name,
                model,
                tier,
                available_models=list(self._prices.keys()),
            )

        price = self._prices[model][tier]

        # Ensure cached tokens don't exceed input tokens
        cached_tokens = max(0, min(cached_tokens, input_tokens))
        uncached_tokens = input_tokens - cached_tokens

        # Calculate cost
        cost = uncached_tokens * price["in"]

        # Add cached token cost if available
        if price.get("cached_in") is not None:
            cost += cached_tokens * price["cached_in"]

        # Add output token cost if available
        if price.get("out") is not None:
            cost += output_tokens * price["out"]

        return cost

    def get_encoding_for_model(self, model: str):
        """
        Get the appropriate tiktoken encoding for a given model.

        Args:
            model: Model name

        Returns:
            tiktoken.Encoding: The encoding for the model

        Raises:
            TokenCountingError: If encoding retrieval fails
        """
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            # Fall back to cl100k_base for newer models
            return tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str, model: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: Text to count tokens for
            model: Model name to use for token counting

        Returns:
            int: Number of tokens
        """
        encoding = self.get_encoding_for_model(model)
        return len(encoding.encode(text))


# Apply enhanced error handling to the adapter
def create_openai_adapter():
    """
    Create an OpenAI adapter with enhanced error handling.

    Returns:
        OpenAIAdapter: An enhanced OpenAI adapter
    """
    adapter = OpenAIAdapter()
    return enhance_provider_adapter(adapter)
