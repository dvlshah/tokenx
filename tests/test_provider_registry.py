"""
Unit tests for the provider registry system.
"""

import pytest
from typing import Any, Optional

from tokenx.providers import (
    register_provider,
    ProviderRegistry,
    ProviderAlreadyRegisteredError,
    _PROVIDER_REGISTRY,
)
from tokenx.providers.base import ProviderAdapter, Usage
from tokenx.cost_calc import CostCalculator


class TestProviderRegistry:
    """Test the @register_provider decorator and registry functionality."""

    def test_register_provider_decorator_success(self):
        """Test successful provider registration with decorator."""

        @register_provider("test")
        class TestAdapter(ProviderAdapter):
            @property
            def provider_name(self) -> str:
                return "test"

            def matches_function(self, func: Any, args: tuple, kwargs: dict) -> bool:
                return False

            def usage_from_response(self, response: Any) -> Usage:
                return Usage(input_tokens=10, output_tokens=5, cached_tokens=0)

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

        # Verify the provider was registered in the global registry
        assert "test" in _PROVIDER_REGISTRY
        assert _PROVIDER_REGISTRY["test"] == TestAdapter

        # Clean up for other tests
        del _PROVIDER_REGISTRY["test"]

    def test_register_provider_duplicate_error(self):
        """Test that duplicate registration raises ProviderAlreadyRegisteredError."""

        @register_provider("duplicate")
        class FirstAdapter(ProviderAdapter):
            @property
            def provider_name(self) -> str:
                return "duplicate"

            def matches_function(self, func: Any, args: tuple, kwargs: dict) -> bool:
                return False

            def usage_from_response(self, response: Any) -> Usage:
                return Usage(input_tokens=10, output_tokens=5, cached_tokens=0)

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
                return 0.001

        # Attempting to register another provider with the same name should fail
        with pytest.raises(ProviderAlreadyRegisteredError) as exc_info:

            @register_provider("duplicate")
            class SecondAdapter(ProviderAdapter):
                @property
                def provider_name(self) -> str:
                    return "duplicate"

                def matches_function(
                    self, func: Any, args: tuple, kwargs: dict
                ) -> bool:
                    return False

                def usage_from_response(self, response: Any) -> Usage:
                    return Usage(input_tokens=20, output_tokens=10, cached_tokens=0)

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
                    return 0.002

        assert "already registered" in str(exc_info.value)

        # Clean up
        del _PROVIDER_REGISTRY["duplicate"]

    def test_register_provider_invalid_class(self):
        """Test that non-ProviderAdapter classes are rejected."""

        with pytest.raises(TypeError) as exc_info:

            @register_provider("invalid")
            class NotAnAdapter:
                pass

        assert "must inherit from ProviderAdapter" in str(exc_info.value)

    def test_cost_calculator_uses_registry(self):
        """Test that CostCalculator.for_provider() uses the registry without conditionals."""

        # Register a test provider
        @register_provider("registry_test")
        class RegistryTestAdapter(ProviderAdapter):
            @property
            def provider_name(self) -> str:
                return "registry_test"

            def matches_function(self, func: Any, args: tuple, kwargs: dict) -> bool:
                return False

            def usage_from_response(self, response: Any) -> Usage:
                return Usage(input_tokens=50, output_tokens=25, cached_tokens=5)

            def detect_model(
                self, func: Any, args: tuple, kwargs: dict
            ) -> Optional[str]:
                return "registry-model"

            def calculate_cost(
                self,
                model: str,
                input_tokens: int,
                output_tokens: int,
                cached_tokens: int = 0,
                tier: str = "sync",
            ) -> float:
                return input_tokens * 0.0001 + output_tokens * 0.0002

        # Test that CostCalculator can find and use the registered provider
        calc = CostCalculator.for_provider("registry_test", "test-model")
        assert calc.provider_name == "registry_test"
        assert calc.model == "test-model"

        # Test that the calculator can actually calculate costs
        cost = calc.calculate_cost(100, 50, 10)
        assert cost == 0.02  # 100 * 0.0001 + 50 * 0.0002

        # Clean up
        del _PROVIDER_REGISTRY["registry_test"]

    def test_provider_registry_get_provider_from_global_registry(self):
        """Test that ProviderRegistry.get_provider() can instantiate from global registry."""

        # Register a provider class but don't instantiate it yet
        @register_provider("lazy_test")
        class LazyTestAdapter(ProviderAdapter):
            @property
            def provider_name(self) -> str:
                return "lazy_test"

            def matches_function(self, func: Any, args: tuple, kwargs: dict) -> bool:
                return False

            def usage_from_response(self, response: Any) -> Usage:
                return Usage(input_tokens=75, output_tokens=35, cached_tokens=0)

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
                return 0.005

        # Get the provider - should instantiate it from the global registry
        provider = ProviderRegistry.get_provider("lazy_test")
        assert provider is not None
        assert provider.provider_name == "lazy_test"

        # Should return the same instance on subsequent calls
        provider2 = ProviderRegistry.get_provider("lazy_test")
        assert provider is provider2

        # Clean up
        del _PROVIDER_REGISTRY["lazy_test"]
        if "lazy_test" in ProviderRegistry._providers:
            del ProviderRegistry._providers["lazy_test"]

    def test_unknown_provider_error(self):
        """Test that unknown providers raise ValueError."""

        with pytest.raises(ValueError) as exc_info:
            CostCalculator.for_provider("nonexistent", "model")

        assert "not found" in str(exc_info.value)

    def test_concurrent_provider_registration(self):
        """Test that concurrent provider registrations are thread-safe."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        errors = []
        successful_registrations = []

        def register_provider_thread(provider_id: int):
            """Function to register a provider in a separate thread."""
            try:
                provider_name = f"concurrent_test_{provider_id}"

                @register_provider(provider_name)
                class ConcurrentTestAdapter(ProviderAdapter):
                    def __init__(self, provider_id=provider_id):
                        self.provider_id = provider_id

                    @property
                    def provider_name(self) -> str:
                        return f"concurrent_test_{self.provider_id}"

                    def matches_function(
                        self, func: Any, args: tuple, kwargs: dict
                    ) -> bool:
                        return False

                    def usage_from_response(self, response: Any) -> Usage:
                        return Usage(
                            input_tokens=self.provider_id,
                            output_tokens=self.provider_id,
                            cached_tokens=0,
                        )

                    def detect_model(
                        self, func: Any, args: tuple, kwargs: dict
                    ) -> Optional[str]:
                        return f"concurrent-model-{self.provider_id}"

                    def calculate_cost(
                        self,
                        model: str,
                        input_tokens: int,
                        output_tokens: int,
                        cached_tokens: int = 0,
                        tier: str = "sync",
                    ) -> float:
                        return 0.001 * self.provider_id

                successful_registrations.append(provider_name)

            except Exception as e:
                errors.append(f"Thread {provider_id}: {str(e)}")

        # Run multiple threads trying to register providers concurrently
        num_threads = 10
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(register_provider_thread, i) for i in range(num_threads)
            ]

            # Wait for all threads to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    errors.append(f"Thread execution error: {str(e)}")

        # Verify results
        assert len(errors) == 0, f"Concurrent registration errors: {errors}"
        assert len(successful_registrations) == num_threads

        # Verify all providers are accessible
        for provider_name in successful_registrations:
            calc = CostCalculator.for_provider(provider_name, "test-model")
            assert calc.provider_name == provider_name

        # Clean up
        for provider_name in successful_registrations:
            if provider_name in _PROVIDER_REGISTRY:
                del _PROVIDER_REGISTRY[provider_name]
            if provider_name in ProviderRegistry._providers:
                del ProviderRegistry._providers[provider_name]

    def teardown_method(self):
        """Clean up after each test."""
        # Remove any test providers that might have been left behind
        test_providers = [
            name
            for name in _PROVIDER_REGISTRY.keys()
            if name.startswith("test") or "test" in name
        ]
        for provider_name in test_providers:
            if provider_name in _PROVIDER_REGISTRY:
                del _PROVIDER_REGISTRY[provider_name]
            if provider_name in ProviderRegistry._providers:
                del ProviderRegistry._providers[provider_name]
