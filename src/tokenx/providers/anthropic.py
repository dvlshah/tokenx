"""
Anthropic Provider Adapter Implementation

Enhanced with proper prompt caching cost calculation:
- Cache reads: Use 'cached_hit' pricing (0.1x base input price)
- Cache writes (5m): Use 'cached_in' pricing (1.25x base input price)
- Cache writes (1h): Use 'cached_1h' pricing (2x base input price)
- Regular input: Use base 'in' pricing

Fixes applied in v0.2.8+ for accurate Anthropic prompt caching costs.
"""

from typing import Any, Dict, Optional, Tuple

from .base import ProviderAdapter, Usage
from ..yaml_loader import load_yaml_prices
from ..errors import enhance_provider_adapter, TokenExtractionError, PricingError


class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic API cost calculation."""

    def __init__(self) -> None:
        """Initialize the Anthropic adapter."""
        self._prices = load_yaml_prices().get("anthropic", {})

    @property
    def provider_name(self) -> str:
        """Return the provider name identifier."""
        return "anthropic"

    def matches_function(
        self, func: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> bool:
        """
        Determine if this function is from the Anthropic provider.
        Checks for Anthropic client in the function's module or arguments,
        or Anthropic model names in kwargs.
        """
        module_name = getattr(func, "__module__", "")
        if "anthropic" in module_name.lower():
            return True

        # Check if the first argument is an Anthropic client instance
        if args:
            client_arg = args[0]
            client_class = getattr(client_arg, "__class__", None)
            if client_class:
                class_name = getattr(client_class, "__name__", "")
                class_module = getattr(client_class, "__module__", "")
                # Check common client names like Anthropic, AsyncAnthropic
                if (
                    "anthropic" in class_name.lower()
                    or "anthropic" in class_module.lower()
                ):
                    return True

        # Check for model name in kwargs (e.g., "claude-...")
        model_kwarg = kwargs.get("model")
        if isinstance(model_kwarg, str) and "claude" in model_kwarg.lower():
            return True

        return False

    def _extract_anthropic_usage_fields(
        self, usage_data: Any
    ) -> Dict[str, Optional[int]]:
        """
        Extract standard and cache-specific token fields from Anthropic usage data.
        Defaults cache-related fields to 0 if not present or None.
        Now includes detailed cache creation breakdown for accurate pricing.
        """
        result = {
            "input_tokens": None,
            "output_tokens": None,
            "cache_read_input_tokens": 0,  # Default to 0
            "cache_creation_input_tokens": 0,  # Default to 0
            # Enhanced: Cache creation breakdown for accurate pricing
            "cache_creation_5m_tokens": 0,  # 5-minute TTL cache writes
            "cache_creation_1h_tokens": 0,  # 1-hour TTL cache writes
        }

        # Try attribute-based access (Pydantic models)
        if hasattr(usage_data, "__dict__") or hasattr(usage_data, "__getattr__"):
            result["input_tokens"] = getattr(usage_data, "input_tokens", None)
            result["output_tokens"] = getattr(usage_data, "output_tokens", None)
            # Ensure cache fields default to 0 if attribute is missing or None
            cache_read = getattr(usage_data, "cache_read_input_tokens", 0)
            result["cache_read_input_tokens"] = (
                cache_read if cache_read is not None else 0
            )
            cache_creation = getattr(usage_data, "cache_creation_input_tokens", 0)
            result["cache_creation_input_tokens"] = (
                cache_creation if cache_creation is not None else 0
            )

            # Extract detailed cache creation breakdown if available
            cache_creation_detail = getattr(usage_data, "cache_creation", None)
            if cache_creation_detail:
                result["cache_creation_5m_tokens"] = int(
                    getattr(cache_creation_detail, "ephemeral_5m_input_tokens", 0) or 0
                )
                result["cache_creation_1h_tokens"] = int(
                    getattr(cache_creation_detail, "ephemeral_1h_input_tokens", 0) or 0
                )

        # Fallback to dictionary-based access
        elif isinstance(usage_data, dict):
            result["input_tokens"] = usage_data.get("input_tokens")
            result["output_tokens"] = usage_data.get("output_tokens")
            # Ensure cache fields default to 0 if key is missing or value is None
            cache_read = usage_data.get("cache_read_input_tokens", 0)
            result["cache_read_input_tokens"] = (
                cache_read if cache_read is not None else 0
            )
            cache_creation = usage_data.get("cache_creation_input_tokens", 0)
            result["cache_creation_input_tokens"] = (
                cache_creation if cache_creation is not None else 0
            )

            # Extract detailed cache creation breakdown if available
            cache_creation_detail = usage_data.get("cache_creation", {})
            if isinstance(cache_creation_detail, dict):
                result["cache_creation_5m_tokens"] = int(
                    cache_creation_detail.get("ephemeral_5m_input_tokens", 0) or 0
                )
                result["cache_creation_1h_tokens"] = int(
                    cache_creation_detail.get("ephemeral_1h_input_tokens", 0) or 0
                )

        # Ensure token counts are integers (input/output can be None if not found)
        for key in ["input_tokens", "output_tokens"]:
            if result.get(key) is not None:
                try:
                    result[key] = int(result[key])  # type: ignore
                except (ValueError, TypeError):
                    # If conversion fails for input/output, set to None as they are critical
                    result[key] = None

        # For cache fields, ensure they are int, defaulting to 0 if conversion fails
        for key in [
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "cache_creation_5m_tokens",
            "cache_creation_1h_tokens",
        ]:
            current_val = result.get(key, 0)  # Default to 0 if key somehow missing
            if current_val is not None:
                try:
                    result[key] = int(current_val)
                except (ValueError, TypeError):
                    result[key] = 0  # Default to 0 if not an integer
            else:
                result[key] = 0  # Default to 0 if None
        return result

    def usage_from_response(self, response: Any) -> Usage:
        """
        Extract standardized usage information from an Anthropic response.

        Args:
            response: Anthropic response object (Message, etc.)

        Returns:
            Usage: Standardized usage data with Anthropic-specific fields

        Raises:
            TokenExtractionError: If usage data cannot be extracted
        """
        usage_data = None

        # Try to extract usage from response object
        if hasattr(response, "usage") and response.usage is not None:
            usage_data = response.usage
        elif (
            isinstance(response, dict)
            and "usage" in response
            and response["usage"] is not None
        ):
            usage_data = response["usage"]
        # Some Anthropic SDK versions/methods might return token counts at the top level of the response
        elif hasattr(response, "input_tokens") and hasattr(response, "output_tokens"):
            usage_data = (
                response  # The response object itself contains the token counts
            )
        elif (
            isinstance(response, dict)
            and "input_tokens" in response
            and "output_tokens" in response
        ):
            usage_data = response

        if usage_data is None:
            raise TokenExtractionError(
                "Could not extract usage data from Anthropic response. "
                "Expected 'usage' attribute/key or top-level 'input_tokens'/'output_tokens'.",
                self.provider_name,
                type(response).__name__,
            )

        extracted_fields = self._extract_anthropic_usage_fields(usage_data)

        input_tokens = extracted_fields["input_tokens"]
        output_tokens = extracted_fields["output_tokens"]
        cached_tokens = (
            extracted_fields["cache_read_input_tokens"] or 0
        )  # Map cache_read to cached_tokens

        # Ensure required tokens were found
        if input_tokens is None or output_tokens is None:
            raise TokenExtractionError(
                "Could not extract 'input_tokens' or 'output_tokens' from Anthropic usage data.",
                self.provider_name,
                type(usage_data).__name__,
            )

        # Create Usage dataclass with Anthropic-specific cache fields
        extra_fields = {
            "provider": "anthropic",
            "cache_creation_input_tokens": extracted_fields.get(
                "cache_creation_input_tokens", 0
            ),
            "cache_read_input_tokens": extracted_fields.get(
                "cache_read_input_tokens", 0
            ),
            # Enhanced: Store detailed cache creation breakdown for accurate cost calculation
            "cache_creation_5m_tokens": extracted_fields.get(
                "cache_creation_5m_tokens", 0
            ),
            "cache_creation_1h_tokens": extracted_fields.get(
                "cache_creation_1h_tokens", 0
            ),
            "raw_usage": usage_data if isinstance(usage_data, dict) else None,
        }

        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            extra_fields=extra_fields,
        )

    def extract_tokens(self, response: Any) -> Tuple[int, int, int]:
        """
        Extract standard and cache-specific token counts from an Anthropic response object.
        Maps cache_read_input_tokens to the 'cached_tokens' return value.
        """
        usage_data = None

        # Try to extract usage from response object
        if hasattr(response, "usage") and response.usage is not None:
            usage_data = response.usage
        elif (
            isinstance(response, dict)
            and "usage" in response
            and response["usage"] is not None
        ):
            usage_data = response["usage"]
        # Some Anthropic SDK versions/methods might return token counts at the top level of the response
        elif hasattr(response, "input_tokens") and hasattr(response, "output_tokens"):
            usage_data = (
                response  # The response object itself contains the token counts
            )
        elif (
            isinstance(response, dict)
            and "input_tokens" in response
            and "output_tokens" in response
        ):
            usage_data = response

        if usage_data is None:
            raise TokenExtractionError(
                "Could not extract usage data from Anthropic response. "
                "Expected 'usage' attribute/key or top-level 'input_tokens'/'output_tokens'.",
                self.provider_name,
                type(response).__name__,
            )

        extracted_fields = self._extract_anthropic_usage_fields(usage_data)

        input_tokens = extracted_fields["input_tokens"]
        output_tokens = extracted_fields["output_tokens"]
        cached_tokens = (
            extracted_fields["cache_read_input_tokens"] or 0
        )  # Map cache_read to cached_tokens

        # Ensure required tokens were found
        if input_tokens is None or output_tokens is None:
            raise TokenExtractionError(
                "Could not extract 'input_tokens' or 'output_tokens' from Anthropic usage data.",
                self.provider_name,
                type(usage_data).__name__,
            )
        return input_tokens, output_tokens, cached_tokens

    def detect_model(
        self, func: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Optional[str]:
        """
        Try to identify model name from function and arguments.
        The @measure_cost decorator requires explicit model, so this is supplementary.
        """
        if "model" in kwargs and isinstance(kwargs["model"], str):
            return kwargs["model"]
        return None

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,  # This is the total input tokens (including cached read portions)
        output_tokens: int,
        cached_tokens: int = 0,  # This is the number of tokens READ from cache (cache_read_input_tokens)
        tier: str = "sync",
        response: Optional[Any] = None,
    ) -> float:
        """
        Calculate cost in USD based on token usage for Anthropic models with proper cache pricing.

        Pricing Structure:
        - cached_tokens (cache reads): Use 'cached_hit' pricing (0.1x base)
        - cache_creation_input_tokens: Use 'cached_in' (5m, 1.25x) or 'cached_1h' (1h, 2x)
        - regular input_tokens: Use base 'in' pricing
        """
        if not self._prices:
            raise PricingError(
                f"No pricing information loaded for provider {self.provider_name}.",
                self.provider_name,
            )
        if model not in self._prices:
            raise PricingError(
                f"Price for model={model!r} not found in YAML for Anthropic.",
                self.provider_name,
                model,
                available_models=list(self._prices.keys()),
            )

        if tier not in self._prices[model]:
            raise PricingError(
                f"Price for model={model!r} tier={tier!r} not found in YAML for Anthropic.",
                self.provider_name,
                model,
                tier,
                available_models=list(self._prices[model].keys()),
            )

        price_info = self._prices[model][tier]
        cost = 0.0

        # Extract cache creation info from response if available
        cache_creation_5m_tokens = 0
        cache_creation_1h_tokens = 0
        total_cache_creation_tokens = 0

        if response is not None:
            try:
                # Try to extract cache creation breakdown from response
                usage_fields = self._extract_anthropic_usage_fields(
                    getattr(response, "usage", response)
                )
                cache_creation_5m_tokens = (
                    usage_fields.get("cache_creation_5m_tokens", 0) or 0
                )
                cache_creation_1h_tokens = (
                    usage_fields.get("cache_creation_1h_tokens", 0) or 0
                )
                total_cache_creation_tokens = (
                    usage_fields.get("cache_creation_input_tokens", 0) or 0
                )

                # If no detailed breakdown, assume all cache creation is 5m
                if (
                    total_cache_creation_tokens > 0
                    and (cache_creation_5m_tokens + cache_creation_1h_tokens) == 0
                ):
                    cache_creation_5m_tokens = total_cache_creation_tokens
            except Exception:
                # If extraction fails, continue with basic calculation
                pass

        # 1. CACHE READ COSTS (cache_read_input_tokens)
        # Use 'cached_hit' pricing (0.1x base) for tokens read from cache
        actual_cached_read = min(cached_tokens, input_tokens)
        if actual_cached_read > 0:
            cached_hit_price = price_info.get("cached_hit")
            if cached_hit_price is not None:
                cost += actual_cached_read * cached_hit_price

        # 2. CACHE WRITE COSTS (cache_creation_input_tokens)
        # Use 'cached_in' (5m, 1.25x) or 'cached_1h' (1h, 2x) pricing
        if cache_creation_5m_tokens > 0:
            cached_5m_price = price_info.get("cached_in")  # 5m cache writes
            if cached_5m_price is not None:
                cost += cache_creation_5m_tokens * cached_5m_price

        if cache_creation_1h_tokens > 0:
            cached_1h_price = price_info.get("cached_1h")  # 1h cache writes
            if cached_1h_price is not None:
                cost += cache_creation_1h_tokens * cached_1h_price

        # 3. REGULAR INPUT TOKEN COSTS
        # Calculate uncached input tokens (total - cache_read - cache_creation)
        total_special_tokens = (
            actual_cached_read + cache_creation_5m_tokens + cache_creation_1h_tokens
        )
        uncached_input_tokens = max(0, input_tokens - total_special_tokens)

        if uncached_input_tokens > 0 and price_info.get("in") is not None:
            cost += uncached_input_tokens * price_info["in"]

        # 4. OUTPUT TOKEN COSTS (unchanged)
        if price_info.get("out") is not None:
            cost += output_tokens * price_info["out"]

        return cost


def create_anthropic_adapter() -> Any:
    """
    Create an Anthropic adapter with enhanced error handling.

    Returns:
        AnthropicAdapter: An enhanced Anthropic adapter
    """
    adapter = AnthropicAdapter()
    enhanced = enhance_provider_adapter(adapter)  # ensure fallbacks are applied
    return enhanced
