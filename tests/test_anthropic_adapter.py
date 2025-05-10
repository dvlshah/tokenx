"""
Tests for the Anthropic provider adapter.
"""

import pytest
from unittest.mock import MagicMock, patch

from tokenx.providers.anthropic import create_anthropic_adapter
from tokenx.errors import TokenExtractionError, PricingError


@pytest.fixture
def adapter():
    # Patch load_yaml_prices to return controlled prices for testing
    with patch("tokenx.providers.anthropic.load_yaml_prices") as mock_load_prices:
        mock_load_prices.return_value = {
            "anthropic": {
                "claude-3-sonnet-20240229": {
                    "sync": {
                        "in": 3.00 / 1e6,  # Scaled price
                        "out": 15.00 / 1e6,  # Scaled price
                    }
                },
                "claude-3-opus-20240229": {
                    "sync": {
                        "in": 15.00 / 1e6,
                        "out": 75.00 / 1e6,
                        "cached_in": 7.50
                        / 1e6,  # Hypothetical cached price for testing
                    }
                },
                "model-no-tier": {
                    # Missing 'sync' tier
                },
            }
        }
        # Use the create_anthropic_adapter to ensure enhancements are applied
        adapter_instance = create_anthropic_adapter()
        yield adapter_instance


class TestAnthropicAdapter:
    def test_provider_name(self, adapter):
        assert adapter.provider_name == "anthropic"

    def test_matches_function(self, adapter):
        # Mock Anthropic client and function
        class MockAnthropicClient:
            pass

        mock_anthropic_client_instance = MockAnthropicClient()

        def anthropic_sdk_call():
            pass

        anthropic_sdk_call.__module__ = "anthropic.some.module"

        def another_providers_call():
            pass

        another_providers_call.__module__ = "openai.api"

        assert adapter.matches_function(anthropic_sdk_call, (), {})
        assert adapter.matches_function(MagicMock(__module__="anthropic"), (), {})
        assert adapter.matches_function(
            MagicMock(), (mock_anthropic_client_instance,), {}
        )
        assert adapter.matches_function(
            MagicMock(), (), {"model": "claude-3-opus-20240229"}
        )
        assert not adapter.matches_function(another_providers_call, (), {})
        assert not adapter.matches_function(MagicMock(), (), {"model": "gpt-4"})

    def test_extract_tokens_from_response_object(self, adapter):
        mock_response = MagicMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.usage.cache_read_input_tokens = None  # Or simply don't define it

        input_tokens, output_tokens, cached_tokens = adapter.extract_tokens(
            mock_response
        )
        assert input_tokens == 100
        assert output_tokens == 50
        assert (
            cached_tokens == 0
        )  # EXPECT 0 because cache_read_input_tokens was not provided or was None

    def test_extract_tokens_from_dict(self, adapter):
        response_dict = {"usage": {"input_tokens": 200, "output_tokens": 75}}
        input_tokens, output_tokens, cached_tokens = adapter.extract_tokens(
            response_dict
        )
        assert input_tokens == 200
        assert output_tokens == 75
        assert (
            cached_tokens == 0
        )  # EXPECT 0 because cache_read_input_tokens was not in the dict

    def test_extract_tokens_top_level(self, adapter):
        # Mock for top-level extraction.
        # If 'cache_read_input_tokens' is not a top-level field, cached_tokens will be 0.
        mock_response_spec = {
            "input_tokens": 50,
            "output_tokens": 25,
            # 'cached_tokens': 3, # Old field
            "usage": None,  # Ensure usage is not present
            "cache_read_input_tokens": None,  # Explicitly not providing it or setting to None
        }
        response_top_level_obj = MagicMock(**mock_response_spec)
        # If an attribute is not in spec or explicitly set to None, getattr would raise AttributeError or return None
        # depending on how MagicMock is configured. For safety, ensure it's not there if not intended.
        if not hasattr(response_top_level_obj, "cache_read_input_tokens"):
            response_top_level_obj.cache_read_input_tokens = 0  # Default if not present

        input_tokens, output_tokens, cached_tokens = adapter.extract_tokens(
            response_top_level_obj
        )
        assert input_tokens == 50
        assert output_tokens == 25
        assert (
            cached_tokens == 0
        )  # EXPECT 0 because cache_read_input_tokens was not a top-level field or was None

    def test_extract_tokens_missing_usage(self, adapter):
        # Case 1: response.usage is None, and no other way to get tokens
        # Mock is configured to only have 'usage' and 'choices' attributes for this test's purpose.
        # 'choices' is included to explicitly control its behavior for fallbacks.
        mock_response_no_usage_details = MagicMock(
            spec=["usage", "choices", "input_tokens", "output_tokens"]
        )
        mock_response_no_usage_details.usage = None
        mock_response_no_usage_details.choices = None  # Prevent choices fallback
        del mock_response_no_usage_details.input_tokens
        del mock_response_no_usage_details.output_tokens

        with pytest.raises(TokenExtractionError, match="Could not extract usage data"):
            adapter.extract_tokens(mock_response_no_usage_details)

        # Case 2: response is an empty dict (this should already work if the adapter logic is correct)
        with pytest.raises(TokenExtractionError, match="Could not extract usage data"):
            adapter.extract_tokens({})

    def test_extract_tokens_missing_token_counts(self, adapter):
        # Configure mock so it doesn't fall back to 'choices' unexpectedly
        # Define only the attributes relevant to this test case.
        mock_response_missing_fields = MagicMock(spec=["usage", "choices"])
        mock_response_missing_fields.usage.input_tokens = None  # Explicitly None
        mock_response_missing_fields.usage.output_tokens = 50
        mock_response_missing_fields.choices = None  # Prevent choices fallback

        with pytest.raises(
            TokenExtractionError,
            match="Could not extract 'input_tokens' or 'output_tokens'",
        ):
            adapter.extract_tokens(mock_response_missing_fields)

    def test_detect_model(self, adapter):
        assert (
            adapter.detect_model(None, (), {"model": "claude-3-sonnet-20240229"})
            == "claude-3-sonnet-20240229"
        )
        assert adapter.detect_model(None, (), {}) is None

    def test_extract_tokens_from_response_object_with_cache(self, adapter):
        # Mock response with usage object including cache metrics
        mock_usage = MagicMock(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=20,
            # Add other potential usage attributes with default values or None
            total_tokens=None,  # Explicitly None
        )
        mock_response = MagicMock(usage=mock_usage)

        input_tokens, output_tokens, cached_tokens = adapter.extract_tokens(
            mock_response
        )
        assert input_tokens == 100
        assert output_tokens == 50
        assert (
            cached_tokens == 30
        )  # Should map cache_read_input_tokens to cached_tokens

        # Verify the extra fields are available via the internal method (though not returned by extract_tokens tuple)
        extracted_fields = adapter._extract_anthropic_usage_fields(mock_response.usage)
        assert extracted_fields["cache_read_input_tokens"] == 30
        assert extracted_fields["cache_creation_input_tokens"] == 20
        assert extracted_fields["input_tokens"] == 100
        assert extracted_fields["output_tokens"] == 50

    def test_extract_tokens_from_dict_with_cache(self, adapter):
        # Mock response with usage dictionary including cache metrics
        response_dict = {
            "usage": {
                "input_tokens": 200,
                "output_tokens": 75,
                "cache_read_input_tokens": 40,
                "cache_creation_input_tokens": 30,
                "total_tokens": 275,  # Example total
            }
        }
        input_tokens, output_tokens, cached_tokens = adapter.extract_tokens(
            response_dict
        )
        assert input_tokens == 200
        assert output_tokens == 75
        assert (
            cached_tokens == 40
        )  # Should map cache_read_input_tokens to cached_tokens

        # Verify the extra fields are available via the internal method
        extracted_fields = adapter._extract_anthropic_usage_fields(
            response_dict["usage"]
        )
        assert extracted_fields["cache_read_input_tokens"] == 40
        assert extracted_fields["cache_creation_input_tokens"] == 30
        assert extracted_fields["input_tokens"] == 200
        assert extracted_fields["output_tokens"] == 75

    # ... (existing tests - extract_tokens_top_level, missing_usage, missing_token_counts - ensure these still pass) ...

    def test_calculate_cost_with_cached_pricing(self, mocker):
        """Test cost calculation when 'cached_in' price is available."""
        # Mock pricing data with a 'cached_in' price for a specific model
        mock_load_prices = mocker.patch("tokenx.providers.anthropic.load_yaml_prices")
        mock_load_prices.return_value = {
            "anthropic": {
                "claude-caching-model": {
                    "sync": {
                        "in": 10.00 / 1e6,  # Uncached input price
                        "cached_in": 5.00 / 1e6,  # Cached input price
                        "out": 40.00 / 1e6,  # Output price
                    }
                }
            }
        }
        # Re-create adapter to load the mock prices
        adapter = create_anthropic_adapter()

        # Simulate token counts with tokens read from cache
        total_input = 1000
        cached_read = 300  # These tokens were read from cache (cache_read_input_tokens)
        uncached_input = (
            total_input - cached_read
        )  # These were processed by model (total_input - cache_read)
        output = 500

        # Calculate cost using the adapter's calculate_cost method
        # Pass cache_read_input_tokens as cached_tokens
        cost = adapter.calculate_cost(
            "claude-caching-model",
            input_tokens=total_input,
            output_tokens=output,
            cached_tokens=cached_read,  # Pass cache_read_input_tokens here
            tier="sync",
        )

        # Expected cost calculation: (uncached * uncached_price) + (cached_read * cached_price) + (output * output_price)
        expected_cost = (
            (uncached_input * (10.00 / 1e6))
            + (cached_read * (5.00 / 1e6))
            + (output * (40.00 / 1e6))
        )

        assert cost == pytest.approx(expected_cost)

    def test_calculate_cost_without_cached_pricing_but_with_cached_tokens(
        self, adapter
    ):
        """Test cost calculation when 'cached_in' price is missing, but cached_tokens > 0."""
        # Prices for sonnet: in: 3.00/1M, out: 15.00/1M (no cached_in defined in fixture)
        # Simulate token counts with tokens read from cache, but no specific cached_in price
        total_input = 1000
        cached_read = 300  # These tokens were read from cache
        output = 500

        # Pass cache_read_input_tokens as cached_tokens
        cost = adapter.calculate_cost(
            "claude-3-sonnet-20240229",
            input_tokens=total_input,
            output_tokens=output,
            cached_tokens=cached_read,  # Pass cache_read_input_tokens here
            tier="sync",
        )

        # Expected cost calculation: (total_input * standard_in_price) + (output * output_price)
        # All input tokens use the standard 'in' price if 'cached_in' is not available.
        expected_cost = (total_input * (3.00 / 1e6)) + (output * (15.00 / 1e6))

        assert cost == pytest.approx(expected_cost)

    def test_calculate_cost_missing_model(self, adapter):
        with pytest.raises(
            PricingError, match="Price for model='nonexistent-model' not found"
        ):
            adapter.calculate_cost("nonexistent-model", 100, 50)

    def test_calculate_cost_missing_tier(self, adapter):
        with pytest.raises(
            PricingError, match="Price for model='model-no-tier' tier='sync' not found"
        ):
            adapter.calculate_cost("model-no-tier", 100, 50, tier="sync")

    def test_calculate_cost_no_prices_loaded(self):
        # Test scenario where _prices is empty for the provider
        with patch("tokenx.providers.anthropic.load_yaml_prices") as mock_load_prices:
            mock_load_prices.return_value = {"other_provider": {}}  # No 'anthropic' key
            adapter_no_prices = create_anthropic_adapter()
            with pytest.raises(
                PricingError,
                match="No pricing information loaded for provider anthropic",
            ):
                adapter_no_prices.calculate_cost("claude-3-sonnet-20240229", 100, 50)
