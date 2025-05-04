"""
Error handling module for tokenx.

This module defines exception classes and utilities for error handling
across all provider adapters in tokenx.
"""

from typing import Any, Callable, List, Optional, Tuple


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
    extract_func: Callable, response: Any, provider_name: str
) -> Tuple[int, int, int]:
    """
    Extract tokens from a response with multiple fallback strategies.

    Args:
        extract_func: Original token extraction function
        response: The response object to extract tokens from
        provider_name: Name of the provider for error reporting

    Returns:
        tuple: (input_tokens, output_tokens, cached_tokens)

    Raises:
        TokenExtractionError: If token extraction fails after all fallbacks
    """
    response_type = type(response).__name__

    # Try standard extraction first
    try:
        return extract_func(response)
    except Exception as e:
        # Track fallback attempts for error reporting
        fallback_attempts = []

        # First fallback: Try extracting directly from usage if available
        if hasattr(response, "usage"):
            try:
                fallback_attempts.append("usage attribute")
                return extract_func(response.usage)
            except Exception:
                pass
        elif isinstance(response, dict) and "usage" in response:
            try:
                fallback_attempts.append("usage dict key")
                return extract_func(response["usage"])
            except Exception:
                pass

        # Second fallback: Try parsing choices if available (for stream responses)
        try:
            fallback_attempts.append("choices content estimation")
            if hasattr(response, "choices") and response.choices:
                # Estimate tokens from content length for output tokens
                content = ""
                if hasattr(response.choices[0], "message") and hasattr(
                    response.choices[0].message, "content"
                ):
                    content = response.choices[0].message.content
                elif (
                    isinstance(response.choices[0], dict)
                    and "message" in response.choices[0]
                ):
                    if (
                        isinstance(response.choices[0]["message"], dict)
                        and "content" in response.choices[0]["message"]
                    ):
                        content = response.choices[0]["message"]["content"]

                # Rough estimate (4 chars ≈ 1 token)
                output_tokens = max(1, len(content) // 4)

                # Estimate input tokens from prompt if available
                input_tokens = 0
                if hasattr(response, "prompt_tokens"):
                    input_tokens = response.prompt_tokens
                elif hasattr(response, "usage") and hasattr(
                    response.usage, "prompt_tokens"
                ):
                    input_tokens = response.usage.prompt_tokens
                elif isinstance(response, dict):
                    if "prompt_tokens" in response:
                        input_tokens = response["prompt_tokens"]
                    elif "usage" in response and "prompt_tokens" in response["usage"]:
                        input_tokens = response["usage"]["prompt_tokens"]

                return (
                    input_tokens,
                    output_tokens,
                    0,
                )  # Assume no cached tokens in fallback
        except Exception:
            pass

        # All fallbacks failed, raise a detailed error
        available_attrs = dir(response) if hasattr(response, "__dict__") else "N/A"
        if isinstance(available_attrs, list) and len(available_attrs) > 20:
            # Trim very long attribute list
            available_attrs = available_attrs[:20] + ["..."]

        error_msg = (
            f"Failed to extract tokens from response.\n"
            f"Original error: {str(e)}\n\n"
            f"Response details:\n"
            f"- Type: {response_type}\n"
            f"- Available attributes: {available_attrs}\n\n"
            f"Tried fallback methods: {', '.join(fallback_attempts)}\n\n"
            f"Tips:\n"
            f"- Check if your provider SDK version is supported\n"
            f"- Ensure your model is included in model_prices.yaml\n"
            f"- For streaming responses, consider aggregating usage after completion\n"
            f"- If using a custom client, ensure it returns proper usage data"
        )
        raise TokenExtractionError(error_msg, provider_name, response_type) from e


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
            return original_calculate_cost(
                model, input_tokens, output_tokens, cached_tokens, tier
            )
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
                    f"- Check if the model name exactly matches the entries in model_prices.yaml\n"
                    f"- Update your model_prices.yaml with the latest pricing\n"
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

        def enhanced_get_encoding(model: str):
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
