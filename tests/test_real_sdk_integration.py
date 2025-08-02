"""
Integration tests with real OpenAI and Anthropic SDKs.
This is a temporary test file to validate registry functionality with actual SDKs.

NOTE: These tests require:
1. pip install openai anthropic python-dotenv
2. Create .env file in project root with OPENAI_API_KEY and ANTHROPIC_API_KEY
3. Real API calls will be made (small cost)

Run with: PYTHONPATH=src python -m pytest tests/test_real_sdk_integration.py -v -s
"""

import os
import pytest
from pathlib import Path
from typing import Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    # Find .env file in project root
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        print(f"🔧 Loading .env from: {env_path}")
        load_dotenv(env_path)
    else:
        print(f"⚠️  No .env file found at: {env_path}")
except ImportError:
    print("⚠️  python-dotenv not installed. Skipping .env loading.")

# Import SDKs with proper error handling
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    print("⚠️  OpenAI SDK not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    print("⚠️  Anthropic SDK not available. Install with: pip install anthropic")
    ANTHROPIC_AVAILABLE = False

# Import our tokenx components
from tokenx.metrics import measure_cost, measure_latency
from tokenx.cost_calc import CostCalculator, OpenAICostCalculator
from tokenx.providers import register_provider
from tokenx.providers.base import ProviderAdapter, Usage


class TestRealSDKIntegration:
    """Test tokenx functionality with real OpenAI and Anthropic SDKs."""

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not available")
    def test_openai_cost_calculation_real_api(self):
        """Test cost calculation with real OpenAI API calls."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set in environment")

        client = openai.OpenAI(api_key=api_key)

        @measure_cost("openai", "gpt-3.5-turbo")
        @measure_latency
        def test_completion(prompt: str):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
            )
            return response

        # Make a test call
        result, metrics = test_completion("Say hello in one word")

        # Verify we got metrics
        assert "cost_usd" in metrics
        assert "latency_ms" in metrics
        assert "input_tokens" in metrics
        assert "output_tokens" in metrics
        assert metrics["cost_usd"] > 0
        assert metrics["latency_ms"] > 0
        assert metrics["input_tokens"] > 0
        assert metrics["output_tokens"] > 0

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not available")
    def test_anthropic_cost_calculation_real_api(self):
        """Test cost calculation with real Anthropic API calls."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set in environment")

        client = anthropic.Anthropic(api_key=api_key)

        @measure_cost("anthropic", "claude-3-haiku-20240307")
        @measure_latency
        def test_completion(prompt: str):
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
            )
            return response

        # Make a test call
        result, metrics = test_completion("Say hello in one word")

        # Verify we got metrics
        assert "cost_usd" in metrics
        assert "latency_ms" in metrics
        assert "input_tokens" in metrics
        assert "output_tokens" in metrics
        assert metrics["cost_usd"] > 0
        assert metrics["latency_ms"] > 0
        assert metrics["input_tokens"] > 0
        assert metrics["output_tokens"] > 0

    def test_cost_calculator_instantiation(self):
        """Test that we can create cost calculators for registered providers."""
        # Test OpenAI calculator
        openai_calc = CostCalculator.for_provider("openai", "gpt-3.5-turbo")
        assert openai_calc.provider_name == "openai"
        assert openai_calc.model == "gpt-3.5-turbo"

        # Test Anthropic calculator
        anthropic_calc = CostCalculator.for_provider(
            "anthropic", "claude-3-haiku-20240307"
        )
        assert anthropic_calc.provider_name == "anthropic"
        assert anthropic_calc.model == "claude-3-haiku-20240307"

        # Test cost calculation works
        cost = openai_calc.calculate_cost(1000, 100, 0)
        assert cost > 0

        cost = anthropic_calc.calculate_cost(1000, 100, 0)
        assert cost > 0

    def test_backward_compatibility_openai_calculator(self):
        """Test that OpenAICostCalculator still works for backward compatibility."""
        calc = OpenAICostCalculator("gpt-3.5-turbo")
        assert calc.provider_name == "openai"
        assert calc.model == "gpt-3.5-turbo"

        # Test cost calculation
        cost = calc.calculate_cost(1000, 100, 0)
        assert cost > 0

    def test_custom_provider_registration_integration(self):
        """Test that custom provider registration works end-to-end."""

        # Register a mock provider
        @register_provider("mock_integration")
        class MockIntegrationProvider(ProviderAdapter):
            @property
            def provider_name(self) -> str:
                return "mock_integration"

            def matches_function(self, func, args, kwargs) -> bool:
                return kwargs.get("model", "").startswith("mock-")

            def usage_from_response(self, response) -> Usage:
                # Mock response parsing - handle both standard and custom formats
                if hasattr(response, "usage"):
                    usage = response.usage
                    if isinstance(usage, dict):
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)
                        if input_tokens > 0 and output_tokens > 0:
                            return Usage(
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                cached_tokens=0,
                            )
                if hasattr(response, "token_usage"):
                    return Usage(
                        input_tokens=response.token_usage["input"],
                        output_tokens=response.token_usage["output"],
                        cached_tokens=0,
                    )
                return Usage(
                    input_tokens=100, output_tokens=50, cached_tokens=0
                )  # Default values

            def extract_tokens(self, response) -> tuple:
                usage = self.usage_from_response(response)
                return (usage.input_tokens, usage.output_tokens, usage.cached_tokens)

            def detect_model(self, func, args, kwargs) -> Optional[str]:
                return kwargs.get("model")

            def calculate_cost(
                self,
                model: str,
                input_tokens: int,
                output_tokens: int,
                cached_tokens: int = 0,
                tier: str = "sync",
            ) -> float:
                # Simple pricing: $0.001 per 1000 tokens
                return (input_tokens + output_tokens) * 0.000001

        # Test that we can create a calculator for this provider
        calc = CostCalculator.for_provider("mock_integration", "mock-model-v1")
        assert calc.provider_name == "mock_integration"
        assert calc.model == "mock-model-v1"

        # Test cost calculation
        cost = calc.calculate_cost(1000, 500, 0)
        expected_cost = 1500 * 0.000001  # (1000 + 500) * 0.000001
        assert abs(cost - expected_cost) < 1e-10

        # Test token extraction with mock response
        class MockResponse:
            def __init__(self):
                self.usage = {"input_tokens": 150, "output_tokens": 75}

        mock_response = MockResponse()
        tokens = calc.provider.extract_tokens(mock_response)
        assert tokens == (150, 75, 0)

    def test_registry_stress_test(self):
        """Stress test the registry with multiple provider registrations."""

        providers_to_test = []

        # Register multiple test providers
        for i in range(5):
            provider_name = f"stress_test_{i}"

            @register_provider(provider_name)
            class StressTestProvider(ProviderAdapter):
                def __init__(self, provider_id=i):
                    self.provider_id = provider_id

                @property
                def provider_name(self) -> str:
                    return f"stress_test_{self.provider_id}"

                def matches_function(self, func, args, kwargs) -> bool:
                    return False

                def usage_from_response(self, response) -> Usage:
                    return Usage(
                        input_tokens=self.provider_id * 10,
                        output_tokens=self.provider_id * 5,
                        cached_tokens=0,
                    )

                def extract_tokens(self, response) -> tuple:
                    usage = self.usage_from_response(response)
                    return (
                        usage.input_tokens,
                        usage.output_tokens,
                        usage.cached_tokens,
                    )

                def detect_model(self, func, args, kwargs) -> Optional[str]:
                    return f"stress-model-{self.provider_id}"

                def calculate_cost(
                    self,
                    model: str,
                    input_tokens: int,
                    output_tokens: int,
                    cached_tokens: int = 0,
                    tier: str = "sync",
                ) -> float:
                    return (input_tokens + output_tokens) * (
                        0.000001 * (self.provider_id + 1)
                    )

            providers_to_test.append(provider_name)

        # Test that all providers are accessible
        for provider_name in providers_to_test:
            calc = CostCalculator.for_provider(provider_name, f"model-{provider_name}")
            assert calc.provider_name == provider_name

            # Extract provider_id from name to test cost calculation
            provider_id = int(provider_name.split("_")[-1])
            expected_input = provider_id * 10
            expected_output = provider_id * 5
            expected_cost = (expected_input + expected_output) * (
                0.000001 * (provider_id + 1)
            )

            # Test cost calculation
            cost = calc.calculate_cost(expected_input, expected_output, 0)
            assert abs(cost - expected_cost) < 1e-10

            # Test token extraction
            mock_response = (
                object()
            )  # Any object will work for our mock usage_from_response
            tokens = calc.provider.extract_tokens(mock_response)
            assert tokens == (expected_input, expected_output, 0)

    def test_provider_interface_enforcement(self):
        """Test that provider registration enforces the BaseExtractor interface."""

        # This should fail because the provider doesn't implement usage_from_response
        with pytest.raises(TypeError) as exc_info:

            @register_provider("incomplete_provider")
            class IncompleteProvider(ProviderAdapter):
                @property
                def provider_name(self) -> str:
                    return "incomplete"

                def matches_function(self, func, args, kwargs) -> bool:
                    return False

                # Missing usage_from_response method!

                def detect_model(self, func, args, kwargs) -> Optional[str]:
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

            # This should fail when we try to instantiate it
            CostCalculator.for_provider("incomplete_provider", "test-model")

        assert "usage_from_response" in str(exc_info.value)
