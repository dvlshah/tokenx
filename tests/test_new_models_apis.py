"""
Tests for new models and APIs added in this branch.

This module tests all the new models and API endpoints that were added
as part of the 9x OpenAI coverage expansion.
"""

from unittest.mock import patch

from tokenx.providers.openai import OpenAIAdapter
from tokenx.cost_calc import CostCalculator


class TestNewModels:
    """Test new models added to the pricing YAML."""

    def setup_method(self):
        """Test setup - use real pricing YAML."""
        self.adapter = OpenAIAdapter()

    def test_o3_models_pricing(self):
        """Test o3 model family pricing."""
        # o3 model
        cost = self.adapter.calculate_cost("o3", 1000, 500, 0, "sync")
        expected_cost = (1000 * 2.00 + 500 * 8.00) / 1_000_000
        assert abs(cost - expected_cost) < 1e-10

        # o3 with flex pricing
        cost_flex = self.adapter.calculate_cost("o3", 1000, 500, 0, "flex")
        expected_flex = (1000 * 1.00 + 500 * 4.00) / 1_000_000
        assert abs(cost_flex - expected_flex) < 1e-10

        # o3-pro model
        cost_pro = self.adapter.calculate_cost("o3-pro", 1000, 500, 0, "sync")
        expected_pro = (1000 * 20.00 + 500 * 80.00) / 1_000_000
        assert abs(cost_pro - expected_pro) < 1e-10

        # o3-deep-research model
        cost_deep = self.adapter.calculate_cost(
            "o3-deep-research", 1000, 500, 100, "sync"
        )
        expected_deep = ((1000 - 100) * 10.00 + 100 * 2.50 + 500 * 40.00) / 1_000_000
        assert abs(cost_deep - expected_deep) < 1e-10

        # o3-mini model
        cost_mini = self.adapter.calculate_cost("o3-mini", 1000, 500, 50, "sync")
        expected_mini = ((1000 - 50) * 1.10 + 50 * 0.55 + 500 * 4.40) / 1_000_000
        assert abs(cost_mini - expected_mini) < 1e-10

    def test_o4_models_pricing(self):
        """Test o4 model family pricing."""
        # o4-mini model
        cost = self.adapter.calculate_cost("o4-mini", 1000, 500, 100, "sync")
        expected_cost = ((1000 - 100) * 1.10 + 100 * 0.275 + 500 * 4.40) / 1_000_000
        assert abs(cost - expected_cost) < 1e-10

        # o4-mini with flex pricing
        cost_flex = self.adapter.calculate_cost("o4-mini", 1000, 500, 100, "flex")
        expected_flex = ((1000 - 100) * 0.55 + 100 * 0.138 + 500 * 2.20) / 1_000_000
        assert abs(cost_flex - expected_flex) < 1e-10

        # o4-mini-deep-research model
        cost_deep = self.adapter.calculate_cost(
            "o4-mini-deep-research", 1000, 500, 50, "sync"
        )
        expected_deep = ((1000 - 50) * 2.00 + 50 * 0.50 + 500 * 8.00) / 1_000_000
        assert abs(cost_deep - expected_deep) < 1e-10

    def test_gpt_4_1_models_pricing(self):
        """Test GPT-4.1 model family pricing."""
        # gpt-4.1 model
        cost = self.adapter.calculate_cost("gpt-4.1", 1000, 500, 100, "sync")
        expected_cost = ((1000 - 100) * 2.00 + 100 * 0.50 + 500 * 8.00) / 1_000_000
        assert abs(cost - expected_cost) < 1e-10

        # gpt-4.1-mini model
        cost_mini = self.adapter.calculate_cost("gpt-4.1-mini", 1000, 500, 50, "sync")
        expected_mini = ((1000 - 50) * 0.40 + 50 * 0.10 + 500 * 1.60) / 1_000_000
        assert abs(cost_mini - expected_mini) < 1e-10

        # gpt-4.1-nano model
        cost_nano = self.adapter.calculate_cost("gpt-4.1-nano", 1000, 500, 25, "sync")
        expected_nano = ((1000 - 25) * 0.10 + 25 * 0.025 + 500 * 0.40) / 1_000_000
        assert abs(cost_nano - expected_nano) < 1e-10

    def test_new_preview_models_pricing(self):
        """Test new preview models pricing."""
        # gpt-4.5-preview model (very expensive)
        cost = self.adapter.calculate_cost("gpt-4.5-preview", 100, 50, 20, "sync")
        expected_cost = ((100 - 20) * 75.00 + 20 * 37.50 + 50 * 150.00) / 1_000_000
        assert abs(cost - expected_cost) < 1e-10

        # codex-mini-latest model
        cost_codex = self.adapter.calculate_cost(
            "codex-mini-latest", 1000, 500, 100, "sync"
        )
        expected_codex = ((1000 - 100) * 1.50 + 100 * 0.375 + 500 * 6.00) / 1_000_000
        assert abs(cost_codex - expected_codex) < 1e-10

    def test_search_models_pricing(self):
        """Test search preview models pricing."""
        # gpt-4o-mini-search-preview (no caching discount)
        cost = self.adapter.calculate_cost(
            "gpt-4o-mini-search-preview", 1000, 500, 0, "sync"
        )
        expected_cost = (1000 * 0.15 + 500 * 0.60) / 1_000_000
        assert abs(cost - expected_cost) < 1e-10

        # gpt-4o-search-preview (no caching discount)
        cost_4o = self.adapter.calculate_cost(
            "gpt-4o-search-preview", 1000, 500, 0, "sync"
        )
        expected_4o = (1000 * 2.50 + 500 * 10.00) / 1_000_000
        assert abs(cost_4o - expected_4o) < 1e-10

    def test_computer_use_model_pricing(self):
        """Test computer use preview model pricing."""
        cost = self.adapter.calculate_cost("computer-use-preview", 1000, 500, 0, "sync")
        expected_cost = (1000 * 3.00 + 500 * 12.00) / 1_000_000
        assert abs(cost - expected_cost) < 1e-10

    def test_audio_models_basic_pricing(self):
        """Test basic pricing for new audio models (without dual tokens)."""
        # gpt-4o-transcribe model
        cost_transcribe = self.adapter.calculate_cost(
            "gpt-4o-transcribe", 1000, 200, 0, "sync"
        )
        expected_transcribe = (1000 * 2.50 + 200 * 10.00) / 1_000_000
        assert abs(cost_transcribe - expected_transcribe) < 1e-10

        # gpt-4o-mini-transcribe model
        cost_mini_transcribe = self.adapter.calculate_cost(
            "gpt-4o-mini-transcribe", 1000, 200, 0, "sync"
        )
        expected_mini_transcribe = (1000 * 1.25 + 200 * 5.00) / 1_000_000
        assert abs(cost_mini_transcribe - expected_mini_transcribe) < 1e-10

        # gpt-4o-mini-tts model (no text output)
        cost_tts = self.adapter.calculate_cost("gpt-4o-mini-tts", 500, 0, 0, "sync")
        expected_tts = (500 * 0.60) / 1_000_000
        assert abs(cost_tts - expected_tts) < 1e-10

    def test_moderation_models_free_pricing(self):
        """Test that moderation models are correctly priced as FREE."""
        # omni-moderation-latest
        cost_omni = self.adapter.calculate_cost(
            "omni-moderation-latest", 1000, 0, 0, "sync"
        )
        assert cost_omni == 0.0

        # text-moderation-latest
        cost_text_latest = self.adapter.calculate_cost(
            "text-moderation-latest", 1000, 0, 0, "sync"
        )
        assert cost_text_latest == 0.0

        # text-moderation-stable
        cost_text_stable = self.adapter.calculate_cost(
            "text-moderation-stable", 1000, 0, 0, "sync"
        )
        assert cost_text_stable == 0.0

    def test_realtime_models_pricing(self):
        """Test realtime models pricing."""
        # gpt-4o-realtime-preview
        cost_4o = self.adapter.calculate_cost(
            "gpt-4o-realtime-preview", 1000, 500, 100, "sync"
        )
        expected_4o = ((1000 - 100) * 40.00 + 100 * 2.50 + 500 * 80.00) / 1_000_000
        assert abs(cost_4o - expected_4o) < 1e-10

        # gpt-4o-mini-realtime-preview
        cost_mini = self.adapter.calculate_cost(
            "gpt-4o-mini-realtime-preview", 1000, 500, 50, "sync"
        )
        expected_mini = ((1000 - 50) * 10.00 + 50 * 0.30 + 500 * 20.00) / 1_000_000
        assert abs(cost_mini - expected_mini) < 1e-10


class TestNewAPIEndpoints:
    """Test new API endpoints and their behavior."""

    def setup_method(self):
        """Test setup."""
        with patch("tokenx.providers.openai.load_yaml_prices") as mock_prices:
            mock_prices.return_value = {
                "openai": {"gpt-4o": {"sync": {"in": 2.50, "out": 10.00}}}
            }
            self.adapter = OpenAIAdapter()

    def test_audio_preview_api_detection(self):
        """Test audio preview API detection."""
        # gpt-4o-audio-preview should be detected
        assert self.adapter.matches_function(
            lambda: None,
            (),
            {
                "model": "gpt-4o-audio-preview",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        # gpt-4o-mini-audio-preview should be detected
        assert self.adapter.matches_function(
            lambda: None,
            (),
            {
                "model": "gpt-4o-mini-audio-preview",
                "input": "audio_content",
                "modalities": ["text", "audio"],
            },
        )

    def test_realtime_api_detection(self):
        """Test realtime API detection."""
        # Realtime API typically uses WebSocket connections
        assert self.adapter.matches_function(
            lambda: None,
            (),
            {
                "model": "gpt-4o-realtime-preview",
                "session": {"instructions": "You are helpful"},
            },
        )

    def test_new_embedding_models(self):
        """Test new embedding models if any were added."""
        # Test existing embedding models work with new detection
        assert self.adapter.matches_function(
            lambda: None,
            (),
            {"model": "text-embedding-3-small", "input": "Text to embed"},
        )

        assert self.adapter.matches_function(
            lambda: None,
            (),
            {"model": "text-embedding-3-large", "input": ["Text 1", "Text 2"]},
        )


class TestComprehensiveCoverage:
    """Test comprehensive coverage of all major OpenAI APIs."""

    def setup_method(self):
        """Test setup."""
        self.adapter = OpenAIAdapter()

    def test_comprehensive_model_coverage(self):
        """Test that all new models can be instantiated with CostCalculator."""
        new_models = [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4.5-preview",
            "o3",
            "o3-pro",
            "o3-deep-research",
            "o3-mini",
            "o4-mini",
            "o4-mini-deep-research",
            "codex-mini-latest",
            "gpt-4o-mini-search-preview",
            "gpt-4o-search-preview",
            "computer-use-preview",
            "gpt-4o-transcribe",
            "gpt-4o-mini-transcribe",
            "gpt-4o-mini-tts",
            "gpt-4o-audio-preview",
            "gpt-4o-mini-audio-preview",
            "gpt-4o-realtime-preview",
            "gpt-4o-mini-realtime-preview",
            "gpt-image-1",
            "omni-moderation-latest",
            "text-moderation-latest",
            "text-moderation-stable",
        ]

        for model in new_models:
            # Should not raise PricingError
            calc = CostCalculator.for_provider("openai", model)
            assert calc.model == model
            assert calc.provider_name == "openai"

            # Should be able to calculate basic cost
            cost = calc.calculate_cost(100, 50, 0)
            assert cost >= 0  # Cost should be non-negative

    def test_tier_support_for_new_models(self):
        """Test tier support (sync/flex) for models that have it."""
        # Models with flex pricing
        flex_models = ["o3", "o4-mini"]

        for model in flex_models:
            calc_sync = CostCalculator.for_provider("openai", model, tier="sync")
            calc_flex = CostCalculator.for_provider("openai", model, tier="flex")

            cost_sync = calc_sync.calculate_cost(1000, 500, 0)
            cost_flex = calc_flex.calculate_cost(1000, 500, 0)

            # Flex should be cheaper than sync
            assert cost_flex < cost_sync

    def test_cached_token_support(self):
        """Test cached token support for models that have it."""
        # Models with caching support
        cached_models = ["gpt-4o", "gpt-4o-mini", "o1", "o3", "o3-mini"]

        for model in cached_models:
            if model in ["o1"]:  # Skip if not in current YAML
                continue

            calc = CostCalculator.for_provider("openai", model)

            cost_no_cache = calc.calculate_cost(1000, 500, 0)
            cost_with_cache = calc.calculate_cost(1000, 500, 200)  # 200 cached tokens

            # With caching should be cheaper
            assert cost_with_cache < cost_no_cache
