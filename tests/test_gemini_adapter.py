import pytest
from unittest.mock import MagicMock, patch

from tokenx.providers.gemini import create_gemini_adapter
from tokenx.errors import TokenExtractionError, PricingError


@pytest.fixture
def adapter():
    with patch("tokenx.providers.gemini.load_yaml_prices") as mock_load:
        mock_load.return_value = {
            "gemini": {
                "gemini-test-model": {
                    "sync": {"in": 1.0, "cached_in": 0.5, "out": 2.0}
                }
            }
        }
        yield create_gemini_adapter()


class TestGeminiAdapter:
    def test_provider_name(self, adapter):
        assert adapter.provider_name == "gemini"

    def test_matches_function(self, adapter):
        def gemini_fn():
            pass

        gemini_fn.__module__ = "google.generativeai.models"
        assert adapter.matches_function(gemini_fn, (), {})

        class MockModel:
            pass

        assert adapter.matches_function(lambda: None, (MockModel(),), {}) is False

        assert not adapter.matches_function(lambda: None, (), {"model": "gpt-4"})
        assert adapter.matches_function(lambda: None, (), {"model": "gemini-pro"})

    def test_extract_tokens_dict(self, adapter):
        resp = {
            "usage_metadata": {
                "prompt_token_count": 10,
                "candidates_token_count": 5,
                "cached_content_token_count": 2,
            }
        }
        t_in, t_out, t_cached = adapter.extract_tokens(resp)
        assert (t_in, t_out, t_cached) == (10, 5, 2)

    def test_extract_tokens_attr(self, adapter):
        usage = MagicMock(
            prompt_token_count=20,
            candidates_token_count=7,
            cached_content_token_count=None,
        )
        resp = MagicMock(usage_metadata=usage)
        assert adapter.extract_tokens(resp) == (20, 7, 0)

    def test_extract_tokens_missing(self, adapter):
        with pytest.raises(TokenExtractionError):
            adapter.extract_tokens({})

    def test_detect_model(self, adapter):
        assert (
            adapter.detect_model(None, (), {"model_name": "gemini-pro"})
            == "gemini-pro"
        )
        assert adapter.detect_model(None, (), {}) is None

    def test_calculate_cost(self, adapter):
        cost = adapter.calculate_cost(
            "gemini-test-model", input_tokens=100, output_tokens=50, cached_tokens=20
        )
        expected = (100 - 20) * 1.0 + 20 * 0.5 + 50 * 2.0
        assert cost == expected

    def test_calculate_cost_missing_model(self, adapter):
        with pytest.raises(PricingError):
            adapter.calculate_cost("missing-model", 1, 1)
