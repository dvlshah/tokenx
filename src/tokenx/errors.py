"""
Error handling module for tokenx.

This module defines exception classes and utilities for error handling
across all provider adapters in tokenx.
"""

from typing import Any, Callable, List, Optional, Tuple

from .constants import CHARS_PER_TOKEN_ESTIMATE


class LLMMeterError(Exception):
    """Base exception for all tokenx errors."""

    pass


class ProviderError(LLMMeterError):
    """Base exception for provider-related errors."""

    pass


class TokenExtractionError(ProviderError):
    """Exception raised when token extraction fails."""

    def __init__(
        self, message: str, provider: str, response_type: Optional[str] = None
    ):
        self.provider = provider
        self.response_type = response_type

        # Enhance message with provider info
        enhanced_message = f"[{provider}] {message}"
        if response_type:
            enhanced_message = (
                f"[{provider}] [Response type: {response_type}] {message}"
            )

        super().__init__(enhanced_message)


class PricingError(ProviderError):
    """Exception raised when pricing information is not available."""

    def __init__(
        self,
        message: str,
        provider: str,
        model: Optional[str] = None,
        tier: Optional[str] = None,
        available_models: Optional[List[str]] = None,
    ):
        self.provider = provider
        self.model = model
        self.tier = tier
        self.available_models = available_models or []

        # Enhance message with provider and model info
        enhanced_message = f"[{provider}]"
        if model:
            enhanced_message += f" [Model: {model}]"
        if tier:
            enhanced_message += f" [Tier: {tier}]"

        enhanced_message += f" {message}"

        # Add available models if provided
        if available_models:
            if len(available_models) > 10:
                # Too many models to list them all
                model_sample = ", ".join(available_models[:10])
                enhanced_message += f"\n\nAvailable models include: {model_sample}, and {len(available_models) - 10} more."
            else:
                model_list = ", ".join(available_models)
                enhanced_message += f"\n\nAvailable models: {model_list}"

        super().__init__(enhanced_message)


class ModelDetectionError(ProviderError):
    """Exception raised when model detection fails."""

    def __init__(self, message: str, provider: str):
        self.provider = provider
        enhanced_message = f"[{provider}] {message}"
        super().__init__(enhanced_message)


class TokenCountingError(ProviderError):
    """Exception raised when token counting fails."""

    def __init__(self, message: str, provider: str, model: Optional[str] = None):
        self.provider = provider
        self.model = model

        enhanced_message = f"[{provider}]"
        if model:
            enhanced_message += f" [Model: {model}]"

        enhanced_message += f" {message}"
        super().__init__(enhanced_message)


# Fallback utilities
def extract_tokens_with_fallbacks(
    extract_func: Callable[..., Any], response: Any, provider_name: str
) -> Tuple[int, int, int]:
    """
    Extract tokens from a response with multiple fallback strategies.
    If extract_func raises TokenExtractionError, it's re-raised.
    Fallbacks are attempted for other exceptions from extract_func.
    """
    response_type = type(response).__name__

    try:
        return extract_func(response)  # type: ignore
    except TokenExtractionError:
        raise  # Re-raise adapter-identified errors
    except Exception as e:
        return _attempt_fallback_extraction(e, response, provider_name, response_type)


def _attempt_fallback_extraction(
    original_exception: Exception, response: Any, provider_name: str, response_type: str
) -> Tuple[int, int, int]:
    """Attempt fallback token extraction strategies."""
    fallback_attempts: List[str] = []

    # Fallback 1: Direct usage parsing
    tokens = _try_direct_usage_parsing(response, fallback_attempts)
    if tokens:
        return tokens

    # Fallback 2: Content estimation
    tokens = _try_content_estimation(response, fallback_attempts)
    if tokens:
        return tokens

    # All fallbacks failed
    return _raise_fallback_error(
        original_exception, response, provider_name, response_type, fallback_attempts
    )


def _try_direct_usage_parsing(
    response: Any, fallback_attempts: List[str]
) -> Optional[Tuple[int, int, int]]:
    """Try to extract tokens directly from usage data."""
    usage_data = _extract_usage_data(response)
    if not usage_data:
        return None

    try:
        fallback_attempts.append("direct usage parsing")

        # Extract token fields
        input_tokens = _get_token_field(usage_data, ["prompt_tokens", "input_tokens"])
        output_tokens = _get_token_field(
            usage_data, ["completion_tokens", "output_tokens"]
        )
        cached_tokens = _get_token_field(usage_data, ["cached_tokens"], default=0)

        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            return (
                input_tokens,
                output_tokens,
                int(cached_tokens) if cached_tokens is not None else 0,
            )
    except Exception:
        pass
    return None


def _try_content_estimation(
    response: Any, fallback_attempts: List[str]
) -> Optional[Tuple[int, int, int]]:
    """Try to estimate tokens from response content."""
    try:
        fallback_attempts.append("choices content estimation")

        # Extract content from choices
        content = _extract_choice_content(response)
        if content is None:
            return None

        # Estimate output tokens using constant
        output_tokens = max(1, len(content) // CHARS_PER_TOKEN_ESTIMATE)

        # Try to find input tokens
        input_tokens = _find_input_tokens(response)

        return (input_tokens, output_tokens, 0)
    except Exception:
        pass
    return None


def _extract_usage_data(response: Any) -> Any:
    """Extract usage data from response."""
    if hasattr(response, "usage") and response.usage is not None:
        return response.usage
    elif (
        isinstance(response, dict)
        and "usage" in response
        and response["usage"] is not None
    ):
        return response["usage"]
    return None


def _get_token_field(
    usage_data: Any, field_names: List[str], default: Optional[int] = None
) -> Optional[int]:
    """Get a token field from usage data, trying multiple field names."""
    for field_name in field_names:
        if hasattr(usage_data, field_name):
            value = getattr(usage_data, field_name)
            if value is not None:
                try:
                    return int(value)
                except (ValueError, TypeError):
                    continue
        elif isinstance(usage_data, dict) and field_name in usage_data:
            value = usage_data[field_name]
            if value is not None:
                try:
                    return int(value)
                except (ValueError, TypeError):
                    continue
    return default


def _extract_choice_content(response: Any) -> Optional[str]:
    """Extract content from response choices."""
    if not hasattr(response, "choices") or not response.choices:
        return None

    choice = response.choices[0]

    # Try object-style access
    if hasattr(choice, "message") and hasattr(choice.message, "content"):
        content = choice.message.content
        return content if isinstance(content, str) else ""

    # Try dict-style access
    if isinstance(choice, dict) and "message" in choice:
        message_dict = choice["message"]
        if isinstance(message_dict, dict) and "content" in message_dict:
            content = message_dict["content"]
            return content if isinstance(content, str) else ""

    return None


def _find_input_tokens(response: Any) -> int:
    """Find input tokens from various response locations."""
    # Direct attribute access
    if hasattr(response, "prompt_tokens") and isinstance(response.prompt_tokens, int):
        return response.prompt_tokens

    # Usage attribute access
    if hasattr(response, "usage") and hasattr(response.usage, "prompt_tokens"):
        tokens = response.usage.prompt_tokens
        if isinstance(tokens, int):
            return tokens

    # Dict access
    if isinstance(response, dict):
        if "prompt_tokens" in response and isinstance(response["prompt_tokens"], int):
            return response["prompt_tokens"]
        elif "usage" in response and isinstance(response["usage"], dict):
            usage = response["usage"]
            if "prompt_tokens" in usage and isinstance(usage["prompt_tokens"], int):
                return usage["prompt_tokens"]

    return 0  # Default fallback


def _raise_fallback_error(
    original_exception: Exception,
    response: Any,
    provider_name: str,
    response_type: str,
    fallback_attempts: List[str],
) -> Tuple[int, int, int]:
    """Raise a detailed error when all fallbacks fail."""
    available_attrs = dir(response) if hasattr(response, "__dict__") else "N/A"
    if isinstance(available_attrs, list) and len(available_attrs) > 20:
        available_attrs = available_attrs[:20] + ["..."]

    error_msg = (
        f"Failed to extract tokens from response after trying fallbacks.\n"
        f"Initial adapter error: [{type(original_exception).__name__}] {str(original_exception)}\n\n"
        f"Response details:\n"
        f"- Type: {response_type}\n"
        f"- Available attributes: {str(available_attrs)}\n\n"
        f"Tried fallback methods: {', '.join(fallback_attempts) if fallback_attempts else 'None'}\n\n"
        f"Tips:\n"
        f"- Check if your provider SDK version is supported\n"
        f"- Ensure your model is included in the pricing configuration\n"
        f"- For streaming responses, consider aggregating usage after completion\n"
        f"- If using a custom client, ensure it returns proper usage data"
    )
    raise TokenExtractionError(
        error_msg, provider_name, response_type
    ) from original_exception


def enhance_provider_adapter(adapter: Any) -> Any:
    """
    Apply enhanced error handling to a provider adapter.

    This function enhances error handling for the common provider methods.

    Args:
        adapter: The provider adapter instance to enhance

    Returns:
        The enhanced adapter
    """
    provider_name = adapter.provider_name

    # Enhance extract_tokens method
    original_extract_tokens = adapter.extract_tokens

    def enhanced_extract_tokens(response: Any) -> Tuple[int, int, int]:
        return extract_tokens_with_fallbacks(
            original_extract_tokens, response, provider_name
        )

    adapter.extract_tokens = enhanced_extract_tokens

    # Enhance calculate_cost method
    original_calculate_cost = adapter.calculate_cost

    def enhanced_calculate_cost(
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        tier: str = "sync",
    ) -> float:
        try:
            result = original_calculate_cost(
                model, input_tokens, output_tokens, cached_tokens, tier
            )
            return float(result)
        except ValueError as e:
            available_models = (
                list(adapter._prices.keys()) if hasattr(adapter, "_prices") else []
            )

            if "not found in YAML" in str(e):
                # Enhance the pricing error message
                error_msg = (
                    f"Price information not found for the specified configuration.\n"
                    f"Original error: {str(e)}\n\n"
                    f"Tips:\n"
                    f"- Check if the model name exactly matches the entries in the pricing data\n"
                    f"- Pricing is automatically updated from remote sources\n"
                    f"- Consider using a similar model with known pricing"
                )
                raise PricingError(
                    error_msg, provider_name, model, tier, available_models
                ) from e
            raise

    adapter.calculate_cost = enhanced_calculate_cost

    # If adapter has get_encoding_for_model method, enhance it too
    if hasattr(adapter, "get_encoding_for_model"):
        original_get_encoding = adapter.get_encoding_for_model

        def enhanced_get_encoding(model: str) -> Any:
            try:
                return original_get_encoding(model)
            except Exception as e:
                error_msg = (
                    f"Failed to get encoding for model.\n"
                    f"Original error: {str(e)}\n\n"
                    f"Tips:\n"
                    f"- Check if the tokenizer library is installed\n"
                    f"- Ensure the model name is supported by the tokenizer"
                )
                raise TokenCountingError(error_msg, provider_name, model) from e

        adapter.get_encoding_for_model = enhanced_get_encoding

    return adapter
