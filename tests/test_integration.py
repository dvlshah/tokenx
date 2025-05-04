import time
from tokenx.metrics import measure_latency, measure_cost


class TestIntegration:
    def test_decorators_combined(self, mocker):
        """Test latency and cost decorators working together."""
        # Mock the provider detection and cost calculation
        mock_calculator = mocker.MagicMock()
        mock_calculator.costed.return_value = lambda func: lambda: {
            "provider": "openai",
            "model": "gpt-4o",
            "input_tokens": 100,
            "output_tokens": 50,
            "cached_tokens": 20,
            "usd": 0.001,
        }

        # Patch for_provider to return our mock calculator
        mocker.patch(
            "tokenx.metrics.CostCalculator.for_provider", return_value=mock_calculator
        )

        @measure_latency
        @measure_cost(provider="openai", model="gpt-4o")
        def mock_openai_function():
            time.sleep(0.01)  # Add latency
            return {
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }

        result, metrics = mock_openai_function()

        # Check metrics contains both latency and cost
        assert "latency_ms" in metrics
        assert "cost_usd" in metrics
        assert metrics["cost_usd"] == 0.001
        assert metrics["provider"] == "openai"
        assert metrics["model"] == "gpt-4o"
