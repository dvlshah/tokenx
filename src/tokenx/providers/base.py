"""
Base Provider Interface for LLM Cost Calculation

All provider adapters must implement this interface to ensure consistent
token counting and cost calculation across different LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple


class ProviderAdapter(ABC):
    """Base adapter interface for LLM providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name identifier."""
        pass

    @abstractmethod
    def matches_function(self, func: Any, args: tuple, kwargs: dict) -> bool:
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

    @abstractmethod
    def extract_tokens(self, response: Any) -> Tuple[int, int, int]:
        """
        Extract token counts from a response object.

        Args:
            response: Provider-specific response object

        Returns:
            tuple: (input_tokens, output_tokens, cached_tokens)
        """
        pass

    @abstractmethod
    def detect_model(self, func: Any, args: tuple, kwargs: dict) -> Optional[str]:
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
