"""
Fixed test file for the OpenAI adapter.

This file tests the OpenAI adapter functionality with correct import paths.
"""

from unittest.mock import MagicMock, patch

# Fix: Use absolute imports instead of relative imports
from tokenx.providers.openai import OpenAIAdapter


class TestOpenAIAdapter:
    def setup_method(self):
        """Test setup."""
        self.adapter = OpenAIAdapter()

    def test_matches_function(self):
        """Test detection of OpenAI functions."""

        # Test module name matching
        def mock_openai_fn():
            pass

        mock_openai_fn.__module__ = "openai.chat_completions"
        assert self.adapter.matches_function(mock_openai_fn, (), {})

        # Test model name matching
        assert self.adapter.matches_function(lambda: None, (), {"model": "gpt-4o"})

        # Test negative case
        def mock_non_openai_fn():
            pass

        mock_non_openai_fn.__module__ = "anthropic.api"
        assert not self.adapter.matches_function(mock_non_openai_fn, (), {})

    def test_extract_tokens(self):
        """Test token extraction from response."""
        # Test dict usage
        mock_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 20},
        }
        input_tokens, output_tokens, cached_tokens = self.adapter.extract_tokens(
            mock_usage
        )
        assert input_tokens == 100
        assert output_tokens == 50
        assert cached_tokens == 20

        # Test response with usage attribute
        class MockResponse:
            def __init__(self):
                self.usage = {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "prompt_tokens_details": {"cached_tokens": 20},
                }

            # Add get method to mock dict-like behavior
            def get(self, key, default=None):
                if key == "usage":
                    return self.usage
                return default

        input_tokens, output_tokens, cached_tokens = self.adapter.extract_tokens(
            MockResponse()
        )
        assert input_tokens == 100
        assert output_tokens == 50
        assert cached_tokens == 20

    def test_normalize_usage(self):
        """Test the usage normalization function."""
        # Test dictionary format
        usage_dict = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 20},
        }
        normalized = self.adapter._normalize_usage(usage_dict)
        assert normalized["input_tokens"] == 100
        assert normalized["output_tokens"] == 50
        assert normalized["cached_tokens"] == 20

        # Test attribute format
        class MockUsageDetails:
            def __init__(self, cached_tokens):
                self.cached_tokens = cached_tokens

        class MockUsage:
            def __init__(self):
                self.prompt_tokens = 200
                self.completion_tokens = 100
                self.prompt_tokens_details = MockUsageDetails(30)

        normalized = self.adapter._normalize_usage(MockUsage())
        assert normalized["input_tokens"] == 200
        assert normalized["output_tokens"] == 100
        assert normalized["cached_tokens"] == 30

    def test_detect_model(self):
        """Test model detection from function and arguments."""
        # Only model in kwargs should be detected
        model = self.adapter.detect_model(None, (), {"model": "gpt-4o"})
        assert model == "gpt-4o"

        # Model should not be detected from other sources
        model = self.adapter.detect_model(None, (MagicMock(model="gpt-4o"),), {})
        assert model is None

    @patch("tokenx.providers.openai.tiktoken.encoding_for_model")
    def test_get_encoding_for_model(self, mock_encoding):
        """Test getting encoding for a known model."""
        mock_encoding.return_value = "encoding"
        encoding = self.adapter.get_encoding_for_model("gpt-4o")
        mock_encoding.assert_called_with("gpt-4o")
        assert encoding == "encoding"

    @patch("tokenx.providers.openai.load_yaml_prices")
    def test_calculate_cost(self, mock_load_prices):
        """Test cost calculation with proper pricing."""
        # Mock the pricing data
        mock_load_prices.return_value = {
            "openai": {
                "gpt-4o": {
                    "sync": {
                        "in": 0.00000250,  # per token
                        "cached_in": 0.00000125,  # per token
                        "out": 0.00001000,  # per token
                    }
                }
            }
        }

        # Create adapter with mocked prices
        adapter = OpenAIAdapter()

        # Calculate cost
        cost = adapter.calculate_cost(
            "gpt-4o", input_tokens=100, output_tokens=50, cached_tokens=20
        )

        # Verify cost calculation
        expected_cost = (
            (100 - 20) * 0.00000250  # uncached input tokens
            + 20 * 0.00000125  # cached input tokens
            + 50 * 0.00001000  # output tokens
        )
        assert cost == expected_cost
