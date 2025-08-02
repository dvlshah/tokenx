"""
Tests for dual token pricing functionality in OpenAI adapter.

This module tests the new dual token pricing feature for audio models that
separate audio tokens from text tokens for more accurate cost calculation.
"""

from unittest.mock import MagicMock, patch

from tokenx.providers.openai import OpenAIAdapter


class TestDualTokenPricing:
    """Test dual token pricing functionality."""

    def setup_method(self):
        """Test setup."""
        with patch("tokenx.providers.openai.load_yaml_prices") as mock_prices:
            mock_prices.return_value = {
                "openai": {
                    "gpt-4o-mini-transcribe": {
                        "sync": {
                            "in": 0.00000125,  # Text input tokens: $1.25/1M
                            "out": 0.000005,  # Text output tokens: $5.00/1M
                            "audio_in": 0.000003,  # Audio input tokens: $3.00/1M
                        }
                    },
                    "gpt-4o-mini-tts": {
                        "sync": {
                            "in": 0.0000006,  # Text input tokens: $0.60/1M
                            "audio_out": 0.000012,  # Audio output tokens: $12.00/1M
                        }
                    },
                }
            }
            self.adapter = OpenAIAdapter()

    def test_extract_token_breakdown_audio_input(self):
        """Test _extract_token_breakdown method for audio input models."""
        # Mock response with audio and text token breakdown
        mock_response = MagicMock()

        # Create proper mock details for transcription models (use input_token_details)
        mock_input_details = MagicMock()
        mock_input_details.audio_tokens = 800  # Actual integer
        mock_input_details.text_tokens = 200  # Actual integer
        mock_response.usage.input_token_details = mock_input_details

        # Remove prompt_tokens_details completely to ensure input_token_details is used
        mock_response.usage.prompt_tokens_details = None

        audio_tokens, text_tokens = self.adapter._extract_token_breakdown(
            mock_response, total_input_tokens=1000
        )

        assert audio_tokens == 800
        assert text_tokens == 200

    def test_extract_token_breakdown_fallback(self):
        """Test _extract_token_breakdown fallback when details not available."""
        # Mock response without token breakdown details
        mock_response = MagicMock()
        # Remove the input_token_details attribute entirely
        delattr(mock_response.usage, "input_token_details")
        delattr(mock_response.usage, "prompt_tokens_details")

        audio_tokens, text_tokens = self.adapter._extract_token_breakdown(
            mock_response, total_input_tokens=1000
        )

        # Should fallback to all tokens being text tokens
        assert audio_tokens == 0
        assert text_tokens == 1000

    def test_extract_audio_output_tokens(self):
        """Test _extract_audio_output_tokens method."""
        # Mock response with audio output tokens
        mock_response = MagicMock()
        mock_response.usage.completion_tokens_details.audio_tokens = 500

        audio_out_tokens = self.adapter._extract_audio_output_tokens(mock_response)
        assert audio_out_tokens == 500

    def test_extract_audio_output_tokens_fallback(self):
        """Test _extract_audio_output_tokens fallback when not available."""
        # Mock response without audio output token details
        mock_response = MagicMock()
        del mock_response.usage.completion_tokens_details.audio_tokens

        audio_out_tokens = self.adapter._extract_audio_output_tokens(mock_response)
        assert audio_out_tokens == 0

    def test_calculate_input_token_cost_with_audio(self):
        """Test _calculate_input_token_cost with audio pricing."""
        price = {"in": 0.00000125, "audio_in": 0.000003}

        # Mock response with audio/text breakdown
        mock_response = MagicMock()

        # Create proper mock details
        mock_input_details = MagicMock()
        mock_input_details.audio_tokens = 600  # Actual integer
        mock_input_details.text_tokens = 400  # Actual integer
        mock_response.usage.input_token_details = mock_input_details
        mock_response.usage.prompt_tokens_details = None

        cost = self.adapter._calculate_input_token_cost(
            price, uncached_tokens=1000, cached_tokens=0, response=mock_response
        )

        expected_cost = (
            600 * 0.000003  # 600 audio tokens
            + 400 * 0.00000125  # 400 text tokens
        )
        assert abs(cost - expected_cost) < 1e-10

    def test_calculate_input_token_cost_no_audio_pricing(self):
        """Test _calculate_input_token_cost without audio pricing."""
        price = {"in": 0.00000125}  # No audio_in pricing

        mock_response = MagicMock()

        cost = self.adapter._calculate_input_token_cost(
            price, uncached_tokens=1000, cached_tokens=0, response=mock_response
        )

        expected_cost = 1000 * 0.00000125  # All tokens at text rate
        assert abs(cost - expected_cost) < 1e-10

    def test_dual_token_cost_calculation_transcribe_model(self):
        """Test complete cost calculation for transcription model with dual pricing."""
        # Test gpt-4o-mini-transcribe model
        cost = self.adapter.calculate_cost(
            model="gpt-4o-mini-transcribe",
            input_tokens=1000,
            output_tokens=200,
            cached_tokens=0,
            tier="sync",
            response=None,  # No breakdown available, should use text rates
        )

        expected_cost = (
            1000 * 0.00000125  # Input at text rate
            + 200 * 0.000005  # Output at text rate
        )
        assert abs(cost - expected_cost) < 1e-10

    def test_dual_token_cost_calculation_tts_model(self):
        """Test complete cost calculation for TTS model with audio output pricing."""
        # Mock response with audio output tokens
        mock_response = MagicMock()
        mock_response.usage.completion_tokens_details.audio_tokens = 150

        cost = self.adapter.calculate_cost(
            model="gpt-4o-mini-tts",
            input_tokens=100,
            output_tokens=150,
            cached_tokens=0,
            tier="sync",
            response=mock_response,
        )

        expected_cost = (
            100 * 0.0000006  # Input text tokens
            + 150 * 0.000012  # Output audio tokens
        )
        assert abs(cost - expected_cost) < 1e-10

    def test_dual_token_integration_with_caching(self):
        """Test dual token pricing with cached tokens."""
        price = {"in": 0.00000125, "cached_in": 0.000000625, "audio_in": 0.000003}

        # Mock response with audio/text breakdown
        mock_response = MagicMock()

        # Create proper mock details
        mock_input_details = MagicMock()
        mock_input_details.audio_tokens = 400  # Actual integer
        mock_input_details.text_tokens = 600  # Actual integer
        mock_response.usage.input_token_details = mock_input_details
        mock_response.usage.prompt_tokens_details = None

        cost = self.adapter._calculate_input_token_cost(
            price, uncached_tokens=800, cached_tokens=200, response=mock_response
        )

        # The implementation adds cached tokens on top of regular text tokens
        expected_cost = (
            400 * 0.000003  # 400 audio tokens (no caching)
            + 600 * 0.00000125  # 600 text tokens at regular rate
            + 200 * 0.000000625  # 200 cached tokens at cached rate (additive)
        )
        assert abs(cost - expected_cost) < 1e-10

    def test_model_detection_for_dual_pricing(self):
        """Test that dual pricing models are properly detected."""
        # Test transcription model detection
        assert self.adapter.matches_function(
            lambda: None, (), {"model": "gpt-4o-mini-transcribe", "file": "audio.mp3"}
        )

        # Test TTS model detection
        assert self.adapter.matches_function(
            lambda: None,
            (),
            {"model": "gpt-4o-mini-tts", "voice": "alloy", "input": "Hello world"},
        )
