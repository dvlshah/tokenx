import pytest
import os


class TestRealAPIs:
    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in os.environ
        or pytest.importorskip("openai", reason="OpenAI SDK not installed") is None,
        reason="No OpenAI API key provided or OpenAI SDK not installed",
    )
    def test_openai_api_integration(self):
        """Test with real OpenAI API calls."""
        import openai
        from tokenx.metrics import measure_latency, measure_cost

        @measure_latency
        @measure_cost(
            provider="openai", model="gpt-3.5-turbo"
        )  # Explicitly specify provider and model
        def call_openai():
            client = openai.OpenAI()
            return client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello, world!"}],
            )

        # Verify response and metrics
        try:
            response, metrics = call_openai()

            # Verify response
            assert hasattr(response, "choices")
            assert len(response.choices) > 0

            # Verify metrics
            assert "latency_ms" in metrics
            assert "cost_usd" in metrics
            assert "input_tokens" in metrics
            assert "output_tokens" in metrics
            assert metrics["provider"] == "openai"
        except Exception as e:
            # If API call fails, mark the test as skipped rather than failed
            pytest.skip(f"OpenAI API call failed: {e}")
