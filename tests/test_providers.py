from tokenx.providers import ProviderRegistry
from tokenx.providers.openai import OpenAIAdapter


class TestProviderRegistry:
    def test_provider_registration(self):
        """Test that providers can be registered and retrieved."""
        registry = ProviderRegistry()
        provider = OpenAIAdapter()
        registry.register(provider)
        assert registry.get_provider("openai") == provider

    def test_autodiscovery(self):
        """Test that providers are automatically discovered."""
        ProviderRegistry._initialized = False  # Reset initialization
        providers = ProviderRegistry.get_all_providers()
        assert any(p.provider_name == "openai" for p in providers)

    def test_provider_detection(self):
        """Test detection of providers from function calls."""

        # Mock OpenAI function
        def mock_openai_fn():
            pass

        mock_openai_fn.__module__ = "openai.api_resources"

        provider = ProviderRegistry.detect_provider(
            mock_openai_fn, (), {"model": "gpt-4o"}
        )
        assert provider is not None
        assert provider.provider_name == "openai"
