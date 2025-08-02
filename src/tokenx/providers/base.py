"""
Base Provider Interface for LLM Cost Calculation

All provider adapters must implement this interface to ensure consistent
token counting and cost calculation across different LLM providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict


@dataclass(frozen=True)
class Usage:
    """
    Canonical usage data structure for LLM API responses.

    This dataclass provides a standardized interface for token usage information
    across all LLM providers, ensuring consistent data extraction and processing.
    """

    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    total_tokens: Optional[int] = None
    # Provider-specific fields can be stored here
    extra_fields: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate and compute derived fields after initialization."""
        # Auto-compute total_tokens if not provided
        if self.total_tokens is None:
            object.__setattr__(
                self, "total_tokens", self.input_tokens + self.output_tokens
            )

        # Ensure cached_tokens doesn't exceed input_tokens
        if self.cached_tokens > self.input_tokens:
            object.__setattr__(self, "cached_tokens", self.input_tokens)

        # Validate non-negative values
        for field_name, value in [
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("cached_tokens", self.cached_tokens),
        ]:
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")


class BaseExtractor(ABC):
    """
    Abstract base class for extracting usage information from provider responses.

    This interface ensures that all provider adapters implement a consistent
    method for extracting token usage data from their specific response formats.
    """

    @abstractmethod
    def usage_from_response(self, response: Any) -> Usage:
        """
        Extract standardized usage information from a provider response.

        This method must be implemented by all provider adapters to convert
        their provider-specific response format into the canonical Usage dataclass.

        Args:
            response: Provider-specific response object (e.g., OpenAI ChatCompletion,
                     Anthropic Message, etc.)

        Returns:
            Usage: Standardized usage information with token counts

        Raises:
            TokenExtractionError: If usage data cannot be extracted from response
        """
        pass


class ProviderAdapter(BaseExtractor):
    """
    Base adapter interface for LLM providers.

    This class combines the BaseExtractor interface for usage extraction
    with provider-specific functionality for cost calculation and detection.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name identifier."""
        pass

    @abstractmethod
    def matches_function(
        self, func: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> bool:
        """
        Determine if this function is from this provider.

        Args:
            func: The function to check
            args: Function arguments
            kwargs: Function keyword arguments

        Returns:
            bool: True if the function is from this provider
        """
        pass

    def extract_tokens(self, response: Any) -> Tuple[int, int, int]:
        """
        Extract token counts from a response object.

        This method provides backward compatibility by delegating to the new
        usage_from_response method and converting the result to a tuple.

        Args:
            response: Provider-specific response object

        Returns:
            tuple: (input_tokens, output_tokens, cached_tokens)
        """
        usage = self.usage_from_response(response)
        return (usage.input_tokens, usage.output_tokens, usage.cached_tokens)

    @abstractmethod
    def detect_model(
        self, func: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Optional[str]:
        """
        Try to identify model name from function and arguments.

        Args:
            func: The function being called
            args: Function arguments
            kwargs: Function keyword arguments

        Returns:
            str: Model name if detected, None otherwise
        """
        pass

    @abstractmethod
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        tier: str = "sync",
        response: Optional[Any] = None,
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
        """
        pass
