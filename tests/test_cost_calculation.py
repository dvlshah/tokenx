"""
Comprehensive tests for cost calculation functionality.

This module combines testing for:
- CostCalculator interface and functionality
- OpenAICostCalculator mathematical accuracy
- Backward compatibility
- Provider factory methods
- Decorator functionality
"""

import math
import pytest
from tokenx.cost_calc import CostCalculator, OpenAICostCalculator, PRICE_PER_TOKEN


class TestCostCalculator:
    """Test the CostCalculator interface and functionality."""

    def test_provider_factory(self):
        """Test the provider factory method."""
        calc = CostCalculator.for_provider("openai", "gpt-4o")
        assert calc.provider_name == "openai"
        assert calc.model == "gpt-4o"
        assert isinstance(calc, OpenAICostCalculator)  # Backward compatibility

        # Test invalid provider
        with pytest.raises(ValueError):
            CostCalculator.for_provider("invalid", "model")

    def test_calculate_cost(self, mocker):
        """Test cost calculation."""
        # Mock the provider's calculate_cost method
        mock_provider = mocker.MagicMock()
        mock_provider.calculate_cost.return_value = 0.001
        mocker.patch(
            "tokenx.providers.ProviderRegistry.get_provider", return_value=mock_provider
        )

        calc = CostCalculator("openai", "gpt-4o")
        cost = calc.calculate_cost(100, 50, 20)

        # Verify the provider's calculate_cost was called with the right args
        mock_provider.calculate_cost.assert_called_once_with(
            "gpt-4o", 100, 50, cached_tokens=20, tier="sync"
        )
        assert cost == 0.001

    def test_cost_from_response(self, mocker):
        """Test extracting cost from a response object."""
        # Mock the provider and its methods
        mock_provider = mocker.MagicMock()
        mock_provider.extract_tokens.return_value = (100, 50, 20)
        mock_provider.calculate_cost.return_value = 0.001
        mocker.patch(
            "tokenx.providers.ProviderRegistry.get_provider", return_value=mock_provider
        )

        calc = CostCalculator("openai", "gpt-4o")

        # Test with response object
        class MockResponse:
            usage = {"prompt_tokens": 100, "completion_tokens": 50}

        cost = calc.cost_from_response(MockResponse())
        assert cost == 0.001
        # The method may call extract_tokens more than once (direct + fallback)
        assert mock_provider.extract_tokens.call_count >= 1

        # Test with dictionary response
        mock_provider.extract_tokens.reset_mock()
        response_dict = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}
        cost = calc.cost_from_response(response_dict)
        assert cost == 0.001
        # The method may call extract_tokens more than once (direct + fallback)
        assert mock_provider.extract_tokens.call_count >= 1

    def test_costed_decorator(self, mocker):
        """Test the costed decorator."""
        # Mock the provider and its methods
        mock_provider = mocker.MagicMock()
        mock_provider.extract_tokens.return_value = (100, 50, 20)
        mock_provider.calculate_cost.return_value = 0.001
        mocker.patch(
            "tokenx.providers.ProviderRegistry.get_provider", return_value=mock_provider
        )

        calc = CostCalculator("openai", "gpt-4o")

        # Test decorator on a function
        @calc.costed()
        def mock_function():
            return {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}

        result = mock_function()
        assert "usd" in result
        assert result["usd"] == 0.001
        assert result["provider"] == "openai"
        assert result["model"] == "gpt-4o"
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50


class TestOpenAICostCalculatorAccuracy:
    """Test mathematical accuracy of OpenAI cost calculations."""

    # Models to test for mathematical accuracy
    _MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "o3",
        "o4-mini",
        "gpt-3.5-turbo-0125",
    ]

    @pytest.mark.parametrize("model", _MODELS)
    def test_blended_cost_accuracy(self, model):
        """Test blended cost calculation accuracy with real token counting."""
        calc = OpenAICostCalculator(model, enable_caching=True)

        # Use varied string to avoid token collapse surprises
        prompt = "alpha bravo charlie delta echo foxtrot " * 25
        completion = "one two three four five " * 10

        p_tok = calc._count(prompt)
        c_tok = calc._count(completion)
        cached = min(1024, p_tok)  # same clamp the library uses

        got = calc.blended_cost(prompt, completion, cached)

        price = PRICE_PER_TOKEN[model]
        expected = (
            (p_tok - cached) * price["in"]
            + cached * price["cached_in"]
            + c_tok * price["out"]
        )
        assert math.isclose(got, expected, rel_tol=1e-9)

    def test_cost_from_usage_accuracy(self):
        """Test cost calculation from usage data with mathematical precision."""
        calc = OpenAICostCalculator("gpt-4o-mini", enable_caching=True)

        prompt_tok, comp_tok, cached_tok = 1600, 100, 1024
        usage = {
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tok,
            "prompt_tokens_details": {"cached_tokens": cached_tok},
        }

        price = PRICE_PER_TOKEN["gpt-4o-mini"]
        expected = (
            (prompt_tok - cached_tok) * price["in"]
            + cached_tok * price["cached_in"]
            + comp_tok * price["out"]
        )
        assert math.isclose(calc.cost_from_usage(usage), expected, rel_tol=1e-9)


class TestBackwardCompatibility:
    """Test backward compatibility for existing OpenAI calculator interface."""

    def test_openai_calculator_legacy_interface(self):
        """Test that OpenAICostCalculator works as before."""
        calc = OpenAICostCalculator("gpt-4o")

        # Test the _count method (token counting)
        token_count = calc._count("Hello, world!")
        assert token_count > 0

        # Test blended_cost method
        cost = calc.blended_cost("Hello", "world", 0)
        assert cost > 0

        # Test with caching
        cost_with_cache = calc.blended_cost("Hello", "world", 1)
        assert cost_with_cache < cost  # Should be cheaper with caching

    def test_legacy_instantiation(self):
        """Test legacy instantiation patterns still work."""
        # Direct instantiation
        calc1 = OpenAICostCalculator("gpt-4o")
        assert calc1.model == "gpt-4o"
        assert calc1.provider_name == "openai"

        # Factory method instantiation
        calc2 = CostCalculator.for_provider("openai", "gpt-4o")
        assert isinstance(calc2, OpenAICostCalculator)
        assert calc2.model == "gpt-4o"
