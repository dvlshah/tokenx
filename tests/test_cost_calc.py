import pytest
from tokenx.cost_calc import CostCalculator, OpenAICostCalculator


class TestCostCalculator:
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
        mock_provider.extract_tokens.assert_called_once()

        # Test with dictionary response
        mock_provider.extract_tokens.reset_mock()
        response_dict = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}
        cost = calc.cost_from_response(response_dict)
        assert cost == 0.001
        mock_provider.extract_tokens.assert_called_once()

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
