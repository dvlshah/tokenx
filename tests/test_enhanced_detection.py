"""
Tests for enhanced function detection and "no estimates" policy.

This module tests the enhanced parameter-based function detection and
the "no estimates" policy that raises TokenExtractionError instead of
providing estimates for APIs without usage data.
"""

import pytest
from unittest.mock import MagicMock, patch

from tokenx.providers.openai import OpenAIAdapter
from tokenx.errors import TokenExtractionError


class TestEnhancedFunctionDetection:
    """Test enhanced function detection capabilities."""

    def setup_method(self):
        """Test setup."""
        with patch("tokenx.providers.openai.load_yaml_prices") as mock_prices:
            mock_prices.return_value = {"openai": {}}
            self.adapter = OpenAIAdapter()

    def test_audio_api_detection_by_file_parameter(self):
        """Test detection of audio APIs by file parameter."""
        # Transcription API
        assert self.adapter.matches_function(
            lambda: None, (), {"file": "audio.mp3", "model": "whisper-1"}
        )

        # Translation API
        assert self.adapter.matches_function(
            lambda: None, (), {"file": "german_audio.wav", "model": "whisper-1"}
        )

        # New transcription models
        assert self.adapter.matches_function(
            lambda: None, (), {"file": "interview.m4a", "model": "gpt-4o-transcribe"}
        )

    def test_tts_api_detection_by_parameters(self):
        """Test detection of TTS APIs by voice/input parameters."""
        # Standard TTS
        assert self.adapter.matches_function(
            lambda: None,
            (),
            {"model": "tts-1", "voice": "alloy", "input": "Hello world"},
        )

        # HD TTS
        assert self.adapter.matches_function(
            lambda: None,
            (),
            {"model": "tts-1-hd", "voice": "echo", "input": "High quality speech"},
        )

        # New TTS models
        assert self.adapter.matches_function(
            lambda: None,
            (),
            {"model": "gpt-4o-mini-tts", "voice": "nova", "input": "Modern TTS"},
        )

    def test_image_api_detection_by_parameters(self):
        """Test detection of image APIs by prompt/size parameters."""
        # DALL-E 2
        assert self.adapter.matches_function(
            lambda: None, (), {"prompt": "A beautiful sunset", "size": "1024x1024"}
        )

        # DALL-E 3 with quality
        assert self.adapter.matches_function(
            lambda: None,
            (),
            {"prompt": "Abstract art", "size": "1024x1536", "quality": "hd"},
        )

        # gpt-image-1 with style
        assert self.adapter.matches_function(
            lambda: None,
            (),
            {
                "prompt": "Mountain landscape",
                "style": "vivid",
                "response_format": "url",
            },
        )

    def test_moderation_api_detection(self):
        """Test detection of moderation APIs."""
        # Standard moderation
        assert self.adapter.matches_function(
            lambda: None, (), {"input": "Check this text for violations"}
        )

        # With model specification
        assert self.adapter.matches_function(
            lambda: None,
            (),
            {"input": ["Text 1", "Text 2"], "model": "omni-moderation-latest"},
        )

    def test_embeddings_api_detection(self):
        """Test detection of embeddings APIs."""
        assert self.adapter.matches_function(
            lambda: None,
            (),
            {"input": "Text to embed", "model": "text-embedding-3-small"},
        )

        assert self.adapter.matches_function(
            lambda: None,
            (),
            {"input": ["Text 1", "Text 2"], "model": "text-embedding-3-large"},
        )

    def test_chat_api_detection_by_messages(self):
        """Test detection of chat APIs by messages parameter."""
        assert self.adapter.matches_function(
            lambda: None,
            (),
            {"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4o"},
        )

        assert self.adapter.matches_function(
            lambda: None,
            (),
            {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hi"},
                ],
                "model": "gpt-4o-mini",
            },
        )

    def test_module_based_detection(self):
        """Test module-based detection still works."""
        # OpenAI module detection
        mock_fn = MagicMock()
        mock_fn.__module__ = "openai.resources.chat.completions"
        assert self.adapter.matches_function(mock_fn, (), {})

        mock_fn.__module__ = "openai.resources.embeddings"
        assert self.adapter.matches_function(mock_fn, (), {})

        mock_fn.__module__ = "openai.resources.audio.transcriptions"
        assert self.adapter.matches_function(mock_fn, (), {})

    def test_model_name_detection(self):
        """Test model name-based detection."""
        # GPT models
        assert self.adapter.matches_function(lambda: None, (), {"model": "gpt-4o"})

        assert self.adapter.matches_function(lambda: None, (), {"model": "gpt-4o-mini"})

        # o1/o3 models
        assert self.adapter.matches_function(lambda: None, (), {"model": "o1-preview"})

        assert self.adapter.matches_function(lambda: None, (), {"model": "o3-mini"})

        # Audio models
        assert self.adapter.matches_function(lambda: None, (), {"model": "whisper-1"})

        assert self.adapter.matches_function(lambda: None, (), {"model": "tts-1-hd"})

    def test_negative_detection_cases(self):
        """Test cases that should NOT be detected as OpenAI."""
        # Non-OpenAI modules
        mock_fn = MagicMock()
        mock_fn.__module__ = "anthropic.resources.messages"
        assert not self.adapter.matches_function(mock_fn, (), {})

        # Non-OpenAI model names
        assert not self.adapter.matches_function(
            lambda: None, (), {"model": "claude-3-sonnet"}
        )

        # Insufficient parameters
        assert not self.adapter.matches_function(
            lambda: None,
            (),
            {"prompt": "test"},  # Just prompt, no size/quality
        )

        # Empty parameters
        assert not self.adapter.matches_function(lambda: None, (), {})


class TestNoEstimatesPolicy:
    """Test the 'no estimates' policy - TokenExtractionError instead of estimates."""

    def setup_method(self):
        """Test setup."""
        with patch("tokenx.providers.openai.load_yaml_prices") as mock_prices:
            mock_prices.return_value = {"openai": {}}
            self.adapter = OpenAIAdapter()

    def test_string_response_raises_error(self):
        """Test that string responses raise TokenExtractionError."""
        # Whisper-1 returns plain text string
        with pytest.raises(TokenExtractionError) as exc_info:
            self.adapter.extract_tokens("This is the transcribed text")

        assert "Audio transcription/translation with string response" in str(
            exc_info.value
        )
        assert "duration-based pricing" in str(exc_info.value)
        assert "accuracy" in str(exc_info.value)

    def test_binary_response_raises_error(self):
        """Test that binary TTS responses raise TokenExtractionError."""
        # Mock TTS response with binary content
        mock_response = MagicMock()
        mock_response.content = b"binary audio data here"

        with pytest.raises(TokenExtractionError) as exc_info:
            self.adapter.extract_tokens(mock_response)

        assert "Text-to-Speech API does not provide token usage data" in str(
            exc_info.value
        )
        assert "no estimates allowed" in str(exc_info.value)

    def test_moderation_response_without_usage_raises_error(self):
        """Test that moderation responses without usage raise TokenExtractionError."""
        # Mock moderation response without usage
        mock_response = MagicMock()
        mock_response.results = [{"flagged": False, "categories": {}}]
        mock_response.id = "modr-123"
        mock_response.usage = None  # No usage data

        with pytest.raises(TokenExtractionError) as exc_info:
            self.adapter.extract_tokens(mock_response)

        assert "Moderation API does not provide token usage data" in str(exc_info.value)
        assert "no estimates allowed" in str(exc_info.value)

    def test_unknown_response_type_raises_error(self):
        """Test that unknown response types raise TokenExtractionError."""

        # Mock response that doesn't match any known patterns
        class UnknownResponse:
            def __init__(self):
                self.data = "some data"
                self.metadata = {"type": "unknown"}

        with pytest.raises(TokenExtractionError) as exc_info:
            self.adapter.extract_tokens(UnknownResponse())

        assert "Could not extract usage data from response type" in str(exc_info.value)
        assert "UnknownResponse" in str(exc_info.value)

    def test_dict_without_usage_raises_error(self):
        """Test that dict responses without usage keys raise error."""
        # Dict that doesn't have usage or token information
        invalid_dict = {"data": "some data", "id": "resp-123", "metadata": {}}

        with pytest.raises(TokenExtractionError) as exc_info:
            self.adapter.extract_tokens(invalid_dict)

        assert "Could not extract usage data from response type 'dict'" in str(
            exc_info.value
        )

    def test_valid_responses_still_work(self):
        """Test that valid responses with usage data still work correctly."""
        # Valid dict with usage
        valid_dict = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 20},
        }

        input_tokens, output_tokens, cached_tokens = self.adapter.extract_tokens(
            valid_dict
        )
        assert input_tokens == 100
        assert output_tokens == 50
        assert cached_tokens == 20

    def test_normalize_usage_validation(self):
        """Test that _normalize_usage validates required fields."""
        # Missing required input_tokens
        with pytest.raises(TokenExtractionError) as exc_info:
            self.adapter._normalize_usage({"completion_tokens": 50})

        assert "Could not extract required 'input_tokens'" in str(exc_info.value)

        # Missing required output_tokens (but allow embeddings case with total_tokens)
        with pytest.raises(TokenExtractionError) as exc_info:
            self.adapter._normalize_usage({"some_other_field": 100})

        assert "Could not extract required" in str(exc_info.value)

    def test_error_context_information(self):
        """Test that TokenExtractionError provides helpful context."""
        with pytest.raises(TokenExtractionError) as exc_info:
            self.adapter.extract_tokens("text response")

        error = exc_info.value
        assert error.provider == "openai"
        assert "whisper-1" in error.response_type
        assert "duration-based pricing" in str(error)
