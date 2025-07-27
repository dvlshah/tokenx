"""
Tests for BaseExtractor ABC and Usage dataclass.

These tests verify that the abstract interface is properly enforced
and that CI will fail if providers don't implement required methods.
"""

import pytest
from typing import Any, Optional

from tokenx.providers.base import BaseExtractor, ProviderAdapter, Usage


class TestBaseExtractor:
    """Test the BaseExtractor abstract base class."""

    def test_cannot_instantiate_base_extractor(self):
        """Test that BaseExtractor cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BaseExtractor()

        assert "Can't instantiate abstract class BaseExtractor" in str(exc_info.value)
        assert "usage_from_response" in str(exc_info.value)

    def test_cannot_instantiate_incomplete_extractor(self):
        """Test that incomplete implementations cannot be instantiated."""

        class IncompleteExtractor(BaseExtractor):
            # Missing usage_from_response method
            pass

        with pytest.raises(TypeError) as exc_info:
            IncompleteExtractor()

        assert "Can't instantiate abstract class IncompleteExtractor" in str(
            exc_info.value
        )
        assert "usage_from_response" in str(exc_info.value)

    def test_cannot_instantiate_incomplete_provider_adapter(self):
        """Test that ProviderAdapter without usage_from_response cannot be instantiated."""

        class IncompleteProvider(ProviderAdapter):
            @property
            def provider_name(self) -> str:
                return "incomplete"

            def matches_function(self, func: Any, args: tuple, kwargs: dict) -> bool:
                return False

            # Missing usage_from_response method

            def detect_model(
                self, func: Any, args: tuple, kwargs: dict
            ) -> Optional[str]:
                return None

            def calculate_cost(
                self,
                model: str,
                input_tokens: int,
                output_tokens: int,
                cached_tokens: int = 0,
                tier: str = "sync",
            ) -> float:
                return 0.0

        with pytest.raises(TypeError) as exc_info:
            IncompleteProvider()

        assert "Can't instantiate abstract class IncompleteProvider" in str(
            exc_info.value
        )
        assert "usage_from_response" in str(exc_info.value)

    def test_complete_extractor_can_be_instantiated(self):
        """Test that complete implementations can be instantiated."""

        class CompleteExtractor(BaseExtractor):
            def usage_from_response(self, response: Any) -> Usage:
                return Usage(input_tokens=10, output_tokens=5, cached_tokens=0)

        # Should not raise any exceptions
        extractor = CompleteExtractor()
        assert isinstance(extractor, BaseExtractor)

        # Test the method works
        usage = extractor.usage_from_response({"some": "response"})
        assert isinstance(usage, Usage)
        assert usage.input_tokens == 10
        assert usage.output_tokens == 5

    def test_complete_provider_adapter_can_be_instantiated(self):
        """Test that complete ProviderAdapter implementations work."""

        class CompleteProvider(ProviderAdapter):
            @property
            def provider_name(self) -> str:
                return "complete"

            def matches_function(self, func: Any, args: tuple, kwargs: dict) -> bool:
                return False

            def usage_from_response(self, response: Any) -> Usage:
                return Usage(input_tokens=100, output_tokens=50, cached_tokens=10)

            def detect_model(
                self, func: Any, args: tuple, kwargs: dict
            ) -> Optional[str]:
                return "test-model"

            def calculate_cost(
                self,
                model: str,
                input_tokens: int,
                output_tokens: int,
                cached_tokens: int = 0,
                tier: str = "sync",
            ) -> float:
                return 0.001

        # Should not raise any exceptions
        provider = CompleteProvider()
        assert isinstance(provider, ProviderAdapter)
        assert isinstance(provider, BaseExtractor)

        # Test usage_from_response works
        usage = provider.usage_from_response({"mock": "response"})
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cached_tokens == 10

        # Test backward compatibility - extract_tokens should delegate to usage_from_response
        tokens = provider.extract_tokens({"mock": "response"})
        assert tokens == (100, 50, 10)


class TestUsage:
    """Test the Usage dataclass."""

    def test_basic_usage_creation(self):
        """Test basic Usage dataclass creation."""
        usage = Usage(input_tokens=100, output_tokens=50)

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cached_tokens == 0  # default
        assert usage.total_tokens == 150  # computed
        assert usage.extra_fields is None  # default

    def test_usage_with_all_fields(self):
        """Test Usage with all fields specified."""
        extra = {"provider": "test", "model": "test-model"}
        usage = Usage(
            input_tokens=200,
            output_tokens=100,
            cached_tokens=50,
            total_tokens=300,
            extra_fields=extra,
        )

        assert usage.input_tokens == 200
        assert usage.output_tokens == 100
        assert usage.cached_tokens == 50
        assert usage.total_tokens == 300
        assert usage.extra_fields == extra

    def test_usage_validation_negative_tokens(self):
        """Test that Usage validates non-negative token counts."""

        # Negative input tokens
        with pytest.raises(ValueError) as exc_info:
            Usage(input_tokens=-1, output_tokens=50)
        assert "input_tokens must be non-negative" in str(exc_info.value)

        # Negative output tokens
        with pytest.raises(ValueError) as exc_info:
            Usage(input_tokens=100, output_tokens=-5)
        assert "output_tokens must be non-negative" in str(exc_info.value)

        # Negative cached tokens
        with pytest.raises(ValueError) as exc_info:
            Usage(input_tokens=100, output_tokens=50, cached_tokens=-10)
        assert "cached_tokens must be non-negative" in str(exc_info.value)

    def test_usage_cached_tokens_capping(self):
        """Test that cached_tokens is capped at input_tokens."""
        usage = Usage(input_tokens=100, output_tokens=50, cached_tokens=150)

        # cached_tokens should be capped at input_tokens
        assert usage.cached_tokens == 100
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_usage_total_tokens_computation(self):
        """Test automatic total_tokens computation."""
        usage = Usage(input_tokens=75, output_tokens=25)

        # total_tokens should be automatically computed
        assert usage.total_tokens == 100

        # Manual total_tokens should be preserved
        usage2 = Usage(input_tokens=75, output_tokens=25, total_tokens=200)
        assert usage2.total_tokens == 200

    def test_usage_immutability(self):
        """Test that Usage dataclass is immutable (frozen=True)."""
        usage = Usage(input_tokens=100, output_tokens=50)

        with pytest.raises(AttributeError):
            usage.input_tokens = 200  # type: ignore

        with pytest.raises(AttributeError):
            usage.output_tokens = 75  # type: ignore
