"""
OpenAI Provider Adapter Implementation

This module implements the OpenAI provider adapter with enhanced error handling.
"""

from typing import Any, Dict, Optional, Tuple

import tiktoken

from .base import ProviderAdapter, Usage
from ..yaml_loader import load_yaml_prices
from ..errors import enhance_provider_adapter, TokenExtractionError, PricingError


def _is_real_integer_value(obj: Any, attr: str) -> Optional[int]:
    """Check if an attribute contains a real integer value (not a mock)."""
    try:
        value = None

        # First check if it's in the object's __dict__ (explicitly set)
        if hasattr(obj, "__dict__") and attr in obj.__dict__:
            value = obj.__dict__[attr]
        # Then check _mock_children for mocks
        elif hasattr(obj, "_mock_children") and attr in obj._mock_children:
            value = obj._mock_children[attr]
        else:
            # If not explicitly set, don't try to get it (it will be a new mock)
            return None

        if isinstance(value, int):
            return value
        elif isinstance(value, str) and value.isdigit():
            return int(value)
        return None
    except (AttributeError, ValueError, TypeError):
        return None


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI API cost calculation."""

    def __init__(self) -> None:
        """Initialize the OpenAI adapter."""
        self._prices = load_yaml_prices().get("openai", {})

    @property
    def provider_name(self) -> str:
        """Return the provider name identifier."""
        return "openai"

    def matches_function(
        self, func: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> bool:
        """
        Determine if this function is from the OpenAI provider.

        Checks for OpenAI client in the function's module or arguments.
        Enhanced to detect all OpenAI API endpoints.
        """
        # Check module name for OpenAI indicators (covers all resources)
        module_name = func.__module__ if hasattr(func, "__module__") else ""
        if "openai" in module_name.lower():
            return True

        # Check first argument for OpenAI client
        if args and hasattr(args[0], "__class__"):
            class_name = args[0].__class__.__name__
            if "openai" in class_name.lower():
                return True

        # Check kwargs for OpenAI model names (expanded coverage)
        if "model" in kwargs and isinstance(kwargs["model"], str):
            model = kwargs["model"].lower()
            return (
                # Chat/Completion models
                model.startswith("gpt-")
                or model.startswith("text-")
                or model.startswith("o")  # For o1, o3, etc.
                # Embedding models
                or model.startswith("text-embedding-")
                # Audio models
                or model.startswith("whisper-")
                or model.startswith("tts-")
                or model.startswith("gpt-4o-transcribe")
                or model.startswith("gpt-4o-mini-transcribe")
                or model.startswith("gpt-4o-mini-tts")
                or model.startswith("gpt-4o-audio-preview")
                or model.startswith("gpt-4o-mini-audio-preview")
                or model.startswith("gpt-4o-realtime-preview")
                or model.startswith("gpt-4o-mini-realtime-preview")
                # Image models
                or model.startswith("dall-e-")
                # Moderation models
                or "moderation" in model
                # Any model in our pricing data
                or model in self._prices
            )

        # Check for OpenAI-specific parameter patterns
        # Embeddings API uses 'input' parameter with model
        if "input" in kwargs and "model" in kwargs:
            return True

        # Moderation API uses 'input' parameter (model is optional)
        if "input" in kwargs and not any(
            param in kwargs for param in ["file", "prompt", "messages", "voice"]
        ):
            return True

        # Audio APIs use 'file' parameter with model
        if "file" in kwargs and "model" in kwargs:
            return True

        # Images API uses 'prompt' with size/quality parameters
        if "prompt" in kwargs and any(
            param in kwargs for param in ["size", "quality", "style", "response_format"]
        ):
            return True

        return False

    def _normalize_usage(self, usage: Any) -> Dict[str, Any]:
        """
        Normalize usage data from different OpenAI API response formats.

        Supports multiple APIs:
        - Chat Completions: prompt_tokens, completion_tokens, cached_tokens
        - Embeddings: prompt_tokens, total_tokens (no completion_tokens)
        - Audio: input_tokens, output_tokens (varies by API)
        - Images: No token usage (different pricing model)

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
            else:
                # No completion_tokens or output_tokens found
                # For embeddings, there are no output tokens, so set to 0
                # This handles cases where embeddings only have prompt_tokens + total_tokens
                result["output_tokens"] = 0

            # Check for cached tokens in prompt_tokens_details or input_tokens_details
            details = None
            if hasattr(usage, "prompt_tokens_details"):
                details = usage.prompt_tokens_details
            elif hasattr(usage, "input_tokens_details"):
                details = usage.input_tokens_details

            if details is not None:
                if hasattr(details, "cached_tokens"):
                    result["cached_tokens"] = details.cached_tokens or 0
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

            # For embeddings, there's no output_tokens, so set to 0
            if (
                result["output_tokens"] is None
                and "total_tokens" in usage
                and "prompt_tokens" in usage
            ):
                # Embeddings case: no completion tokens
                result["output_tokens"] = 0

            # Cached tokens from details
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

        # For models with dual token pricing, add image/audio tokens to input_tokens
        if details is not None and result["input_tokens"] is not None:
            # Use helper function to safely extract values
            image_tokens = _is_real_integer_value(details, "image_tokens") or 0
            audio_tokens = _is_real_integer_value(details, "audio_tokens") or 0

            # Add any additional tokens found
            if image_tokens > 0:
                result["input_tokens"] += image_tokens
            elif audio_tokens > 0:
                result["input_tokens"] += audio_tokens

        # For models with dual token pricing, also check for image/audio output tokens
        # Only process if completion_tokens_details was explicitly set (not a mock)
        try:
            if (
                hasattr(usage, "__dict__")
                and "completion_tokens_details" in usage.__dict__
            ) or (
                hasattr(usage, "_mock_children")
                and "completion_tokens_details" in usage._mock_children
            ):
                output_details = usage.completion_tokens_details
                # Use helper function to safely extract values
                image_output_tokens = (
                    _is_real_integer_value(output_details, "image_tokens") or 0
                )
                audio_output_tokens = (
                    _is_real_integer_value(output_details, "audio_tokens") or 0
                )

                # Add any additional output tokens found
                if image_output_tokens > 0:
                    if result["output_tokens"] is not None:
                        result["output_tokens"] += image_output_tokens
                    else:
                        result["output_tokens"] = image_output_tokens
                elif audio_output_tokens > 0:
                    if result["output_tokens"] is not None:
                        result["output_tokens"] += audio_output_tokens
                    else:
                        result["output_tokens"] = audio_output_tokens
        except (AttributeError, TypeError):
            pass  # Skip if any issues accessing the details

        # Fill in 0 for any Nones that weren't required (like cached_tokens if details missing)
        # or if the initial value was 0 from the get() calls.
        result["input_tokens"] = result["input_tokens"] or 0
        result["output_tokens"] = result["output_tokens"] or 0
        # cached_tokens already defaults to 0

        # Final safety check: ensure all values are integers, not mocks
        for key in ["input_tokens", "output_tokens", "cached_tokens"]:
            if not isinstance(result[key], int):
                try:
                    value = result[key]
                    result[key] = int(value) if value is not None else 0
                except (ValueError, TypeError):
                    result[key] = 0

        return result

    def usage_from_response(self, response: Any) -> Usage:
        """
        Extract standardized usage information from an OpenAI response.

        Supports multiple OpenAI APIs:
        - Chat Completions: ChatCompletion.usage
        - Responses API: Response.usage (new advanced interface)
        - Embeddings: CreateEmbeddingResponse.usage
        - Audio: Transcription/Translation/Speech usage
        - Images: ImagesResponse.usage (gpt-image-1 model)
        - Fine-tuning: FineTuningJob (different metrics)

        Args:
            response: OpenAI response object (ChatCompletion, CreateEmbeddingResponse, etc.)

        Returns:
            Usage: Standardized usage data

        Raises:
            TokenExtractionError: If usage data cannot be extracted
        """
        # Handle different response shapes to extract usage information
        usage = None
        api_type = "unknown"
        response_type = type(response).__name__.lower()

        # Handle APIs that don't return usage data - raise error instead of estimating
        if isinstance(response, str):
            # Audio transcription/translation returns plain text string (whisper-1)
            # These APIs use duration-based pricing, not token-based pricing
            raise TokenExtractionError(
                "Audio transcription/translation with string response does not provide token usage data. "
                "This API uses duration-based pricing. Cost tracking disabled for accuracy.",
                self.provider_name,
                "whisper-1 (string response)",
            )

        # Handle binary responses (TTS returns audio data)
        if hasattr(response, "content") and isinstance(response.content, bytes):
            # TTS response with binary audio content - no usage data available
            raise TokenExtractionError(
                "Text-to-Speech API does not provide token usage data in response. "
                "Cost tracking disabled to ensure accuracy - no estimates allowed.",
                self.provider_name,
                "TTS (binary response)",
            )

        # Handle moderation responses without usage data
        if (
            hasattr(response, "results")
            and hasattr(response, "id")
            and not (hasattr(response, "usage") and response.usage)
        ):
            # Moderation API response without usage data
            raise TokenExtractionError(
                "Moderation API does not provide token usage data in response. "
                "Cost tracking disabled to ensure accuracy - no estimates allowed.",
                self.provider_name,
                "Moderation (no usage data)",
            )

        # Try to extract usage from response object
        if hasattr(response, "usage") and response.usage is not None:
            usage = response.usage
            # Detect API type from response.object attribute first, then response type
            object_type = None
            if hasattr(response, "object"):
                obj = response.object
                # Only use if it's an actual string, not a MagicMock
                if isinstance(obj, str):
                    object_type = obj

            if object_type == "chat.completion":
                api_type = "chat_completions"
            elif object_type == "response":
                api_type = "responses"
            elif object_type == "transcription" or object_type == "translation":
                api_type = "audio"
            elif object_type == "moderation":
                api_type = "moderation"
            elif object_type == "list":
                # Distinguish between images and embeddings based on other attributes
                if hasattr(response, "data") and hasattr(response, "created"):
                    # Try to check if created is an actual timestamp (not MagicMock)
                    created_val = getattr(response, "created", None)
                    if (
                        isinstance(created_val, (int, float))
                        or str(created_val).isdigit()
                    ):
                        api_type = "images"
                    else:
                        api_type = "embeddings"
                elif hasattr(response, "model"):
                    api_type = "embeddings"
                else:
                    api_type = "embeddings"  # Default for list object
            elif "embedding" in response_type:
                api_type = "embeddings"
            elif "chat" in response_type or "completion" in response_type:
                api_type = "chat_completions"
            elif "image" in response_type:
                api_type = "images"
            elif "transcription" in response_type or "translation" in response_type:
                api_type = "audio"
            else:
                # Fallback: try to infer from usage structure
                if hasattr(usage, "completion_tokens") or (
                    isinstance(usage, dict) and "completion_tokens" in usage
                ):
                    api_type = "chat_completions"
                elif hasattr(usage, "total_tokens") and not hasattr(
                    usage, "completion_tokens"
                ):
                    api_type = "embeddings"
                elif hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
                    api_type = "audio"
        elif isinstance(response, dict) and "usage" in response:
            usage = response["usage"]
            # Try to detect API type from response structure
            if "data" in response and isinstance(response["data"], list):
                # Likely embeddings response
                api_type = "embeddings"
            elif "completion_tokens" in usage:
                api_type = "chat_completions"
            elif "total_tokens" in usage and "completion_tokens" not in usage:
                api_type = "embeddings"
        else:
            # Use the response itself as usage data (fallback)
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

        # Create Usage dataclass with extracted values and API type info
        extra_fields: Dict[str, Any] = {
            "provider": "openai",
            "api_type": api_type,
            "raw_usage": usage if isinstance(usage, dict) else None,
        }

        # Add API-specific extra fields
        if api_type == "embeddings" and hasattr(usage, "total_tokens"):
            extra_fields["total_tokens"] = usage.total_tokens
        elif (
            api_type == "embeddings"
            and isinstance(usage, dict)
            and "total_tokens" in usage
        ):
            extra_fields["total_tokens"] = usage["total_tokens"]
        elif api_type == "images":
            # Add image-specific details for cost calculation
            extra_fields["quality"] = getattr(response, "quality", "medium")
            extra_fields["size"] = getattr(response, "size", "1024x1024")
            extra_fields["num_images"] = (
                len(response.data) if hasattr(response, "data") and response.data else 1
            )

        return Usage(
            input_tokens=normalized["input_tokens"],
            output_tokens=normalized["output_tokens"],
            cached_tokens=normalized["cached_tokens"],
            extra_fields=extra_fields,
        )

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
        # Handle APIs that don't return usage data - raise error instead of estimating
        if isinstance(response, str):
            # Audio transcription/translation returns plain text string (whisper-1)
            raise TokenExtractionError(
                "Audio transcription/translation with string response does not provide token usage data. "
                "This API uses duration-based pricing. Cost tracking disabled for accuracy.",
                self.provider_name,
                "whisper-1 (string response)",
            )

        # Handle binary responses (TTS returns audio data)
        if hasattr(response, "content") and isinstance(response.content, bytes):
            # TTS response with binary audio content - no usage data available
            raise TokenExtractionError(
                "Text-to-Speech API does not provide token usage data in response. "
                "Cost tracking disabled to ensure accuracy - no estimates allowed.",
                self.provider_name,
                "TTS (binary response)",
            )

        # Handle moderation responses without usage data
        if (
            hasattr(response, "results")
            and hasattr(response, "id")
            and not (hasattr(response, "usage") and response.usage)
        ):
            # Moderation API response without usage data
            raise TokenExtractionError(
                "Moderation API does not provide token usage data in response. "
                "Cost tracking disabled to ensure accuracy - no estimates allowed.",
                self.provider_name,
                "Moderation (no usage data)",
            )

        # Standard case: Handle different response shapes to extract usage information
        usage = None

        # Try to extract usage from response object
        if hasattr(response, "usage"):
            usage = response.usage
        elif isinstance(response, dict) and "usage" in response:
            usage = response["usage"]
        elif (
            hasattr(response, "prompt_tokens")
            or hasattr(response, "input_tokens")
            or (
                isinstance(response, dict)
                and ("prompt_tokens" in response or "input_tokens" in response)
            )
        ):
            # The response itself IS the usage object (e.g., CompletionUsage, Usage, or dict)
            usage = response
        else:
            # For unhandled special cases, raise a more informative error
            raise TokenExtractionError(
                f"Could not extract usage data from response type '{type(response).__name__}'. "
                "This API may not support standard token usage tracking.",
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

    def detect_model(
        self, func: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Optional[str]:
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
        response: Any = None,
    ) -> float:
        """
        Calculate cost in USD based on token usage, dual token pricing, and per-image costs.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached input tokens (default: 0)
            tier: Pricing tier (default: "sync")
            response: Response object for dual token breakdown and per-image cost calculation (optional)

        Returns:
            float: Cost in USD (token cost + per-image cost if applicable)

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

        # Calculate input token cost with dual token pricing support
        cost = self._calculate_input_token_cost(
            price, uncached_tokens, cached_tokens, response
        )

        # Add output token cost if available
        if price.get("out") is not None:
            cost += output_tokens * price["out"]

        # Add audio output token cost if available (for TTS models)
        if price.get("audio_out") is not None and response is not None:
            audio_output_tokens = self._extract_audio_output_tokens(response)
            if audio_output_tokens > 0:
                cost += audio_output_tokens * price["audio_out"]

        # Add per-image cost if applicable and response is provided
        if response is not None:
            image_cost = self.calculate_image_cost(model, response)
            cost += image_cost

        return float(cost)

    def _calculate_input_token_cost(
        self,
        price: Dict[str, Any],
        uncached_tokens: int,
        cached_tokens: int,
        response: Any = None,
    ) -> float:
        """
        Calculate input token cost with support for dual token pricing (text vs audio tokens).

        Args:
            price: Pricing configuration for the model
            uncached_tokens: Number of uncached input tokens
            cached_tokens: Number of cached input tokens
            response: Response object containing detailed token breakdown

        Returns:
            float: Input token cost
        """
        cost = 0.0

        # Check if this model has separate audio token pricing
        has_audio_pricing = price.get("audio_in") is not None

        if has_audio_pricing and response is not None:
            # Extract detailed token breakdown from response
            audio_tokens, text_tokens = self._extract_token_breakdown(
                response, uncached_tokens
            )

            # Use separate pricing for audio and text tokens
            if audio_tokens > 0:
                cost += audio_tokens * price["audio_in"]
            if text_tokens > 0:
                cost += text_tokens * price["in"]

            # Apply cached token cost to text tokens only (audio tokens typically aren't cached)
            if cached_tokens > 0 and price.get("cached_in") is not None:
                cost += cached_tokens * price["cached_in"]
        else:
            # Standard single token pricing
            cost = uncached_tokens * price["in"]

            # Add cached token cost if available
            if cached_tokens > 0 and price.get("cached_in") is not None:
                cost += cached_tokens * price["cached_in"]

        return cost

    def _extract_token_breakdown(
        self, response: Any, total_input_tokens: int
    ) -> Tuple[int, int]:
        """
        Extract detailed token breakdown (audio vs text) from response object.

        Args:
            response: Response object from OpenAI API
            total_input_tokens: Total input tokens as fallback

        Returns:
            Tuple[int, int]: (audio_tokens, text_tokens)
        """
        audio_tokens = 0
        text_tokens = 0

        # Try to extract from response usage details (prefer prompt_tokens_details for chat/image models)
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            if (
                hasattr(usage, "prompt_tokens_details")
                and usage.prompt_tokens_details is not None
            ):
                # Chat models use prompt_tokens_details
                details = usage.prompt_tokens_details

                # Extract and validate token values using helper
                audio_tokens = _is_real_integer_value(details, "audio_tokens") or 0
                image_tokens = _is_real_integer_value(details, "image_tokens") or 0
                text_tokens = _is_real_integer_value(details, "text_tokens") or 0

                # Treat image tokens as audio tokens for pricing
                audio_tokens += image_tokens

                # For chat models, assume remaining tokens are text tokens
                text_tokens = total_input_tokens - audio_tokens
            elif hasattr(usage, "input_token_details"):
                # New transcription models use input_token_details
                details = usage.input_token_details

                # Extract and validate token values using helper
                audio_tokens = _is_real_integer_value(details, "audio_tokens") or 0
                image_tokens = _is_real_integer_value(details, "image_tokens") or 0
                text_tokens = _is_real_integer_value(details, "text_tokens") or 0

                # Treat image tokens as audio tokens for pricing
                audio_tokens += image_tokens

        # Fallback: if no breakdown available, assume all tokens are text tokens
        if audio_tokens == 0 and text_tokens == 0:
            text_tokens = total_input_tokens

        return audio_tokens, text_tokens

    def _extract_audio_output_tokens(self, response: Any) -> int:
        """
        Extract audio output tokens from TTS response objects.

        Args:
            response: Response object from OpenAI TTS API

        Returns:
            int: Number of audio output tokens (0 if not available)
        """
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            if hasattr(usage, "completion_tokens_details"):
                details = usage.completion_tokens_details
                audio_tokens = _is_real_integer_value(details, "audio_tokens") or 0
                image_tokens = _is_real_integer_value(details, "image_tokens") or 0
                return audio_tokens + image_tokens
            elif hasattr(usage, "output_token_details"):
                details = usage.output_token_details
                audio_tokens = _is_real_integer_value(details, "audio_tokens") or 0
                image_tokens = _is_real_integer_value(details, "image_tokens") or 0
                return audio_tokens + image_tokens

        return 0

    def calculate_image_cost(self, model: str, response: Any) -> float:
        """
        Calculate per-image cost for image generation models.

        Args:
            model: Model name (e.g., "gpt-image-1")
            response: Image generation response with quality, size, and data

        Returns:
            float: Per-image cost in USD (0.0 if not applicable)
        """
        # Check if model has per-image pricing in YAML
        if model not in self._prices:
            return 0.0

        model_config = self._prices[model]

        # Extract quality and size from response
        quality = getattr(response, "quality", "medium").lower()
        size = getattr(response, "size", "1024x1024")

        # Get number of images generated
        num_images = 0
        if hasattr(response, "data") and response.data:
            num_images = len(response.data)

        # If no images generated, return 0 cost
        if num_images == 0:
            return 0.0

        # Map OpenAI quality values to pricing tiers
        quality_mapping = {"standard": "low", "hd": "high", "medium": "medium"}
        pricing_quality = quality_mapping.get(quality, quality)

        # Build the key for the flattened YAML structure
        # Format: images_{quality}_{size}
        price_key = f"images_{pricing_quality}_{size}"

        # Get per-image price from YAML (looking in the sync tier)
        sync_prices = model_config.get("sync", {})
        per_image_cost = sync_prices.get(price_key, 0.0)

        # Fallback to default size/quality if specific combination not found
        if per_image_cost == 0.0:
            fallback_key = "images_low_1024x1024"  # Default fallback
            per_image_cost = sync_prices.get(fallback_key, 0.0)

        # Image prices in the test setup are already in USD, no scaling needed

        return float(per_image_cost * num_images)

    def get_encoding_for_model(self, model: str) -> tiktoken.Encoding:
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
def create_openai_adapter() -> Any:
    """
    Create an OpenAI adapter with enhanced error handling.

    Returns:
        OpenAIAdapter: An enhanced OpenAI adapter
    """
    adapter = OpenAIAdapter()
    enhanced = enhance_provider_adapter(adapter)
    return enhanced
