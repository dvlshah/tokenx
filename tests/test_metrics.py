import pytest
import time
import asyncio
from tokenx.metrics import measure_latency, measure_cost


class TestMetricsDecorators:
    def test_measure_latency_sync(self):
        """Test the latency decorator with sync functions."""

        @measure_latency
        def mock_function():
            time.sleep(0.01)
            return "result"

        result, metrics = mock_function()
        assert result == "result"
        assert "latency_ms" in metrics
        assert metrics["latency_ms"] >= 10  # At least 10ms

    @pytest.mark.asyncio
    async def test_measure_latency_async(self):
        """Test the latency decorator with async functions."""

        @measure_latency
        async def mock_function():
            await asyncio.sleep(0.01)
            return "result"

        result, metrics = await mock_function()
        assert result == "result"
        assert "latency_ms" in metrics
        assert metrics["latency_ms"] >= 10  # At least 10ms

    def test_measure_cost_explicit(self, mocker):
        """Test the cost decorator with explicit provider and model."""
        # Mock metrics that will be returned
        mock_metrics = {
            "provider": "openai",
            "model": "gpt-4o",
            "input_tokens": 100,
            "output_tokens": 50,
            "cached_tokens": 20,
            "usd": 0.001,
        }

        # Create a mock calculator
        mock_calc = mocker.MagicMock()

        # When `calculator.costed()(lambda: resp)()` is called in the code,
        # we need to make sure our mocks return the right values at each step
        mock_decorated_lambda = mocker.MagicMock(return_value=mock_metrics)
        mock_decorator = mocker.MagicMock(return_value=mock_decorated_lambda)
        mock_calc.costed.return_value = mock_decorator

        # Patch for_provider to return our mock calculator
        mocker.patch(
            "tokenx.metrics.CostCalculator.for_provider", return_value=mock_calc
        )

        # Create a function with the decorator
        @measure_cost(provider="openai", model="gpt-4o")
        def mock_function():
            return {
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }

        # Call the function and check the results
        result, metrics = mock_function()
        assert "cost_usd" in metrics
        assert metrics["cost_usd"] == 0.001
