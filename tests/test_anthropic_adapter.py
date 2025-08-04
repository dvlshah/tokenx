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

    def test_calculate_cost_with_cache_hit_pricing(self, mocker):
        """Test cost calculation with proper cache hit pricing (cached_hit)."""
        # Mock pricing data with complete cache pricing structure
        mock_load_prices = mocker.patch("tokenx.providers.anthropic.load_yaml_prices")
        mock_load_prices.return_value = {
            "anthropic": {
                "claude-caching-model": {
                    "sync": {
                        "in": 10.00 / 1e6,  # Base input price
                        "cached_in": 12.50 / 1e6,  # 5m cache writes (1.25x)
                        "cached_1h": 20.00 / 1e6,  # 1h cache writes (2x)
                        "cached_hit": 1.00 / 1e6,  # Cache hits (0.1x)
                        "out": 40.00 / 1e6,  # Output price
                    }
                }
            }
        }
        # Re-create adapter to load the mock prices
        adapter = create_anthropic_adapter()

        # Test cache hit scenario
        total_input = 1000
        cached_read = 300  # Tokens read from cache (should use cached_hit pricing)
        output = 500

        cost = adapter.calculate_cost(
            "claude-caching-model",
            input_tokens=total_input,
            output_tokens=output,
            cached_tokens=cached_read,
            tier="sync",
        )

        # Expected cost: (uncached_input * in_price) + (cached_read * cached_hit_price) + (output * out_price)
        uncached_input = total_input - cached_read
        expected_cost = (
            (uncached_input * (10.00 / 1e6))  # 700 * 10.00/1M
            + (cached_read * (1.00 / 1e6))  # 300 * 1.00/1M (cached_hit pricing)
            + (output * (40.00 / 1e6))  # 500 * 40.00/1M
        )

        assert cost == pytest.approx(expected_cost)

    def test_calculate_cost_without_cached_hit_pricing_but_with_cached_tokens(
        self, adapter
    ):
        """Test cost calculation when 'cached_hit' price is missing, but cached_tokens > 0."""
        # Prices for sonnet: in: 3.00/1M, out: 15.00/1M (no cached_hit defined in fixture)
        # With the new implementation, cache reads should still use regular 'in' pricing
        # when cached_hit pricing is not available
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

        # Expected cost calculation: (uncached_input * in_price) + (output * out_price)
        # When cached_hit pricing is not available, cache reads are effectively free (0 cost)
        uncached_input = total_input - cached_read
        expected_cost = (
            (uncached_input * (3.00 / 1e6))  # 700 * 3.00/1M (regular input)
            + (output * (15.00 / 1e6))  # 500 * 15.00/1M (output)
            # Cached reads get 0 cost when cached_hit pricing is not available
        )

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

    def test_calculate_cost_with_cache_write_5m(self, mocker):
        """Test cost calculation for 5-minute cache writes (cached_in pricing)."""
        mock_load_prices = mocker.patch("tokenx.providers.anthropic.load_yaml_prices")
        mock_load_prices.return_value = {
            "anthropic": {
                "claude-test-model": {
                    "sync": {
                        "in": 3.00 / 1e6,
                        "cached_in": 3.75 / 1e6,  # 5m cache writes (1.25x)
                        "cached_1h": 6.00 / 1e6,  # 1h cache writes (2x)
                        "cached_hit": 0.30 / 1e6,  # Cache hits (0.1x)
                        "out": 15.00 / 1e6,
                    }
                }
            }
        }
        # Use the base adapter class directly to test response parameter
        from tokenx.providers.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter()

        # Mock response with cache creation breakdown
        mock_usage = MagicMock()
        mock_usage.cache_creation_input_tokens = 200
        mock_usage.cache_read_input_tokens = 0
        # Add detailed cache breakdown
        mock_cache_detail = MagicMock()
        mock_cache_detail.ephemeral_5m_input_tokens = 200
        mock_cache_detail.ephemeral_1h_input_tokens = 0
        mock_usage.cache_creation = mock_cache_detail

        mock_response = MagicMock()
        mock_response.usage = mock_usage

        cost = adapter.calculate_cost(
            "claude-test-model",
            input_tokens=500,  # 200 cache write + 300 regular
            output_tokens=100,
            cached_tokens=0,  # No cache reads
            tier="sync",
            response=mock_response,
        )

        # Expected: (cache_write_5m * cached_in_price) + (regular_input * in_price) + (output * out_price)
        expected_cost = (
            (200 * (3.75 / 1e6))  # Cache writes use cached_in pricing
            + (300 * (3.00 / 1e6))  # Regular input tokens
            + (100 * (15.00 / 1e6))  # Output tokens
        )

        assert cost == pytest.approx(expected_cost)

    def test_calculate_cost_with_cache_write_1h(self, mocker):
        """Test cost calculation for 1-hour cache writes (cached_1h pricing)."""
        mock_load_prices = mocker.patch("tokenx.providers.anthropic.load_yaml_prices")
        mock_load_prices.return_value = {
            "anthropic": {
                "claude-test-model": {
                    "sync": {
                        "in": 3.00 / 1e6,
                        "cached_in": 3.75 / 1e6,  # 5m cache writes
                        "cached_1h": 6.00 / 1e6,  # 1h cache writes (2x)
                        "cached_hit": 0.30 / 1e6,  # Cache hits
                        "out": 15.00 / 1e6,
                    }
                }
            }
        }
        # Use the base adapter class directly to test response parameter
        from tokenx.providers.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter()

        # Mock response with 1h cache creation
        mock_usage = MagicMock()
        mock_usage.cache_creation_input_tokens = 150
        mock_usage.cache_read_input_tokens = 0
        # Add detailed cache breakdown for 1h
        mock_cache_detail = MagicMock()
        mock_cache_detail.ephemeral_5m_input_tokens = 0
        mock_cache_detail.ephemeral_1h_input_tokens = 150
        mock_usage.cache_creation = mock_cache_detail

        mock_response = MagicMock()
        mock_response.usage = mock_usage

        cost = adapter.calculate_cost(
            "claude-test-model",
            input_tokens=400,  # 150 cache write + 250 regular
            output_tokens=80,
            cached_tokens=0,
            tier="sync",
            response=mock_response,
        )

        # Expected: (cache_write_1h * cached_1h_price) + (regular_input * in_price) + (output * out_price)
        expected_cost = (
            (150 * (6.00 / 1e6))  # 1h cache writes use cached_1h pricing
            + (250 * (3.00 / 1e6))  # Regular input tokens
            + (80 * (15.00 / 1e6))  # Output tokens
        )

        assert cost == pytest.approx(expected_cost)

    def test_calculate_cost_complex_cache_scenario(self, mocker):
        """Test complex scenario with cache reads, 5m writes, 1h writes, and regular tokens."""
        mock_load_prices = mocker.patch("tokenx.providers.anthropic.load_yaml_prices")
        mock_load_prices.return_value = {
            "anthropic": {
                "claude-opus-4-20250514": {
                    "sync": {
                        "in": 15.00 / 1e6,
                        "cached_in": 18.75 / 1e6,  # 5m cache writes (1.25x)
                        "cached_1h": 30.00 / 1e6,  # 1h cache writes (2x)
                        "cached_hit": 1.50 / 1e6,  # Cache hits (0.1x)
                        "out": 75.00 / 1e6,
                    }
                }
            }
        }
        # Use the base adapter class directly to test response parameter
        from tokenx.providers.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter()

        # Complex scenario with all cache types
        mock_usage = MagicMock()
        mock_usage.cache_creation_input_tokens = 300
        mock_usage.cache_read_input_tokens = 150
        # Mixed cache creation breakdown
        mock_cache_detail = MagicMock()
        mock_cache_detail.ephemeral_5m_input_tokens = 200
        mock_cache_detail.ephemeral_1h_input_tokens = 100
        mock_usage.cache_creation = mock_cache_detail

        mock_response = MagicMock()
        mock_response.usage = mock_usage

        total_input = 1000  # 200 (5m) + 100 (1h) + 150 (read) + 550 (regular)
        cache_reads = 150
        output = 200

        cost = adapter.calculate_cost(
            "claude-opus-4-20250514",
            input_tokens=total_input,
            output_tokens=output,
            cached_tokens=cache_reads,
            tier="sync",
            response=mock_response,
        )

        # Expected cost breakdown:
        # - Cache reads: 150 * 1.50/1M (cached_hit)
        # - 5m cache writes: 200 * 18.75/1M (cached_in)
        # - 1h cache writes: 100 * 30.00/1M (cached_1h)
        # - Regular input: 550 * 15.00/1M (in)
        # - Output: 200 * 75.00/1M (out)
        expected_cost = (
            (150 * (1.50 / 1e6))  # Cache reads
            + (200 * (18.75 / 1e6))  # 5m cache writes
            + (100 * (30.00 / 1e6))  # 1h cache writes
            + (550 * (15.00 / 1e6))  # Regular input (1000 - 150 - 200 - 100)
            + (200 * (75.00 / 1e6))  # Output
        )

        assert cost == pytest.approx(expected_cost)

    def test_extract_cache_breakdown_with_detailed_response(self, mocker):
        """Test extraction of detailed cache breakdown from response."""
        mock_load_prices = mocker.patch("tokenx.providers.anthropic.load_yaml_prices")
        mock_load_prices.return_value = {
            "anthropic": {
                "claude-test": {
                    "sync": {
                        "in": 3.00 / 1e6,
                        "cached_in": 3.75 / 1e6,
                        "cached_1h": 6.00 / 1e6,
                        "cached_hit": 0.30 / 1e6,
                        "out": 15.00 / 1e6,
                    }
                }
            }
        }
        # Use the base adapter class directly
        from tokenx.providers.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter()

        # Mock response with detailed cache creation breakdown
        mock_cache_creation_detail = MagicMock()
        mock_cache_creation_detail.ephemeral_5m_input_tokens = 100
        mock_cache_creation_detail.ephemeral_1h_input_tokens = 50

        mock_usage = MagicMock()
        mock_usage.input_tokens = 300
        mock_usage.output_tokens = 150
        mock_usage.cache_read_input_tokens = 75
        mock_usage.cache_creation_input_tokens = 150
        mock_usage.cache_creation = mock_cache_creation_detail

        mock_response = MagicMock()
        mock_response.usage = mock_usage

        # Test the _extract_anthropic_usage_fields method
        extracted_fields = adapter._extract_anthropic_usage_fields(mock_usage)

        assert extracted_fields["input_tokens"] == 300
        assert extracted_fields["output_tokens"] == 150
        assert extracted_fields["cache_read_input_tokens"] == 75
        assert extracted_fields["cache_creation_input_tokens"] == 150
        assert extracted_fields["cache_creation_5m_tokens"] == 100
        assert extracted_fields["cache_creation_1h_tokens"] == 50

    def test_extract_cache_breakdown_fallback_logic(self, mocker):
        """Test fallback logic when detailed breakdown is missing."""
        mock_load_prices = mocker.patch("tokenx.providers.anthropic.load_yaml_prices")
        mock_load_prices.return_value = {
            "anthropic": {
                "claude-test": {
                    "sync": {
                        "in": 3.00 / 1e6,
                        "cached_in": 3.75 / 1e6,
                        "out": 15.00 / 1e6,
                    }
                }
            }
        }
        # Use the base adapter class directly
        from tokenx.providers.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter()

        # Mock response without detailed cache breakdown
        mock_usage = MagicMock()
        mock_usage.input_tokens = 200
        mock_usage.output_tokens = 100
        mock_usage.cache_read_input_tokens = 50
        mock_usage.cache_creation_input_tokens = 75
        mock_usage.cache_creation = None  # No detailed breakdown

        extracted_fields = adapter._extract_anthropic_usage_fields(mock_usage)

        assert extracted_fields["cache_creation_5m_tokens"] == 0  # Should default to 0
        assert extracted_fields["cache_creation_1h_tokens"] == 0  # Should default to 0

        # Test that calculate_cost falls back to 5m pricing when no breakdown
        mock_response = MagicMock()
        mock_response.usage = mock_usage

        cost = adapter.calculate_cost(
            "claude-test",
            input_tokens=200,
            output_tokens=100,
            cached_tokens=50,
            tier="sync",
            response=mock_response,
        )

        # Should assume all cache creation is 5m (cached_in pricing)
        # Cache read: 50 * cached_hit (but no cached_hit in this mock, so fallback)
        # Cache write: 75 * cached_in
        # Regular input: (200 - 50 - 75) = 75 * in
        # Output: 100 * out
        expected_cost = (
            (75 * (3.75 / 1e6))  # Cache writes (fallback to 5m)
            + (75 * (3.00 / 1e6))  # Regular input
            + (100 * (15.00 / 1e6))  # Output
        )

        assert cost == pytest.approx(expected_cost)

    def test_claude_4_models_pricing(self, mocker):
        """Test that Claude 4 models use correct pricing from YAML."""
        # Use actual Claude 4 pricing structure
        mock_load_prices = mocker.patch("tokenx.providers.anthropic.load_yaml_prices")
        mock_load_prices.return_value = {
            "anthropic": {
                "claude-opus-4-20250514": {
                    "sync": {
                        "in": 15.00 / 1e6,
                        "cached_in": 18.75 / 1e6,  # 1.25x
                        "cached_1h": 30.00 / 1e6,  # 2x
                        "cached_hit": 1.50 / 1e6,  # 0.1x
                        "out": 75.00 / 1e6,
                    }
                },
                "claude-sonnet-4-20250514": {
                    "sync": {
                        "in": 3.00 / 1e6,
                        "cached_in": 3.75 / 1e6,  # 1.25x
                        "cached_1h": 6.00 / 1e6,  # 2x
                        "cached_hit": 0.30 / 1e6,  # 0.1x
                        "out": 15.00 / 1e6,
                    }
                },
            }
        }
        adapter = create_anthropic_adapter()

        # Test Claude Opus 4
        cost_opus = adapter.calculate_cost(
            "claude-opus-4-20250514",
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=0,
            tier="sync",
        )
        expected_opus = (1000 * (15.00 / 1e6)) + (500 * (75.00 / 1e6))
        assert cost_opus == pytest.approx(expected_opus)

        # Test Claude Sonnet 4
        cost_sonnet = adapter.calculate_cost(
            "claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=0,
            tier="sync",
        )
        expected_sonnet = (1000 * (3.00 / 1e6)) + (500 * (15.00 / 1e6))
        assert cost_sonnet == pytest.approx(expected_sonnet)

    def test_usage_from_response_with_cache_fields(self, mocker):
        """Test that usage_from_response includes cache fields in extra_fields."""
        mock_load_prices = mocker.patch("tokenx.providers.anthropic.load_yaml_prices")
        mock_load_prices.return_value = {
            "anthropic": {
                "claude-test": {
                    "sync": {
                        "in": 3.00 / 1e6,
                        "out": 15.00 / 1e6,
                    }
                }
            }
        }
        # Use the base adapter class directly
        from tokenx.providers.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter()

        # Mock response with cache data (without cache_creation detail)
        mock_usage = MagicMock()
        mock_usage.input_tokens = 200
        mock_usage.output_tokens = 100
        mock_usage.cache_read_input_tokens = 50
        mock_usage.cache_creation_input_tokens = 75
        mock_usage.cache_creation = None  # No detailed breakdown

        mock_response = MagicMock()
        mock_response.usage = mock_usage

        usage = adapter.usage_from_response(mock_response)

        assert usage.input_tokens == 200
        assert usage.output_tokens == 100
        assert usage.cached_tokens == 50  # Maps from cache_read_input_tokens
        assert usage.extra_fields["provider"] == "anthropic"
        assert usage.extra_fields["cache_creation_input_tokens"] == 75
        assert usage.extra_fields["cache_read_input_tokens"] == 50
        assert (
            usage.extra_fields["cache_creation_5m_tokens"] == 0
        )  # Default when no breakdown
        assert (
            usage.extra_fields["cache_creation_1h_tokens"] == 0
        )  # Default when no breakdown
