"""
Test enhanced OpenAI SDK coverage for multiple API endpoints.

This module tests the expanded OpenAI adapter functionality for:
- Chat Completions (existing)
- Embeddings API (new)
- Audio APIs (new)
- Images API (new)
- Enhanced function detection
"""

import pytest
from unittest.mock import Mock, patch

from tokenx.providers.openai import OpenAIAdapter
from tokenx.providers.base import Usage
from tokenx.errors import TokenExtractionError


class TestOpenAIEnhancedCoverage:
    """Test enhanced OpenAI SDK coverage."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("tokenx.providers.openai.load_yaml_prices") as mock_prices:
            mock_prices.return_value = {
                "openai": {
                    "gpt-4o": {"sync": {"in": 2.50e-6, "out": 10.00e-6}},
                    "text-embedding-3-small": {"sync": {"in": 0.02e-6}},
                    "whisper-1": {"sync": {"in": 6.00e-6}},
                    "tts-1": {"sync": {"in": 15.00e-6}},
                    "dall-e-3": {"sync": {"in": 0.040}},  # Per image
                }
            }
            self.adapter = OpenAIAdapter()

    def test_enhanced_function_detection_chat(self):
        """Test enhanced function detection for chat completions."""

        def mock_chat_fn():
            pass

        mock_chat_fn.__module__ = "openai.resources.chat.completions"

        # Should detect OpenAI module
        assert self.adapter.matches_function(mock_chat_fn, (), {})

        # Should detect GPT model
        assert self.adapter.matches_function(
            mock_chat_fn, (), {"model": "gpt-4o", "messages": []}
        )

    def test_enhanced_function_detection_embeddings(self):
        """Test function detection for embeddings API."""

        def mock_embeddings_fn():
            pass

        mock_embeddings_fn.__module__ = "openai.resources.embeddings"

        # Should detect OpenAI module
        assert self.adapter.matches_function(mock_embeddings_fn, (), {})

        # Should detect embedding model
        assert self.adapter.matches_function(
            mock_embeddings_fn, (), {"model": "text-embedding-3-small"}
        )

        # Should detect embeddings parameters
        assert self.adapter.matches_function(
            mock_embeddings_fn,
            (),
            {"model": "text-embedding-3-small", "input": ["text"]},
        )

    def test_enhanced_function_detection_audio(self):
        """Test function detection for audio APIs."""

        def mock_audio_fn():
            pass

        mock_audio_fn.__module__ = "openai.resources.audio.transcriptions"

        # Should detect OpenAI module
        assert self.adapter.matches_function(mock_audio_fn, (), {})

        # Should detect whisper model
        assert self.adapter.matches_function(mock_audio_fn, (), {"model": "whisper-1"})

        # Should detect audio parameters
        assert self.adapter.matches_function(
            mock_audio_fn, (), {"model": "whisper-1", "file": Mock()}
        )

        # Should detect TTS model
        assert self.adapter.matches_function(
            mock_audio_fn, (), {"model": "tts-1", "voice": "alloy", "input": "text"}
        )

    def test_enhanced_function_detection_images(self):
        """Test function detection for images API."""

        def mock_images_fn():
            pass

        mock_images_fn.__module__ = "openai.resources.images"

        # Should detect OpenAI module
        assert self.adapter.matches_function(mock_images_fn, (), {})

        # Should detect DALL-E model
        assert self.adapter.matches_function(mock_images_fn, (), {"model": "dall-e-3"})

        # Should detect images parameters
        assert self.adapter.matches_function(
            mock_images_fn, (), {"prompt": "a cat painting", "size": "1024x1024"}
        )

    def test_enhanced_function_detection_negative(self):
        """Test that non-OpenAI functions are not detected."""

        def mock_non_openai_fn():
            pass

        mock_non_openai_fn.__module__ = "anthropic.api"

        # Should not detect non-OpenAI modules
        assert not self.adapter.matches_function(mock_non_openai_fn, (), {})

        # Should not detect non-OpenAI models
        assert not self.adapter.matches_function(
            mock_non_openai_fn, (), {"model": "claude-3-haiku"}
        )

    def test_usage_from_response_chat_completions(self):
        """Test usage extraction from chat completions response."""

        # Simple class-based mock for chat completion usage
        class PromptTokensDetails:
            def __init__(self):
                self.cached_tokens = 10

        class ChatUsage:
            def __init__(self):
                self.prompt_tokens = 100
                self.completion_tokens = 50
                self.total_tokens = 150
                self.prompt_tokens_details = PromptTokensDetails()

        class ChatCompletion:
            def __init__(self):
                self.usage = ChatUsage()

        mock_response = ChatCompletion()

        usage = self.adapter.usage_from_response(mock_response)

        assert isinstance(usage, Usage)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cached_tokens == 10
        assert usage.total_tokens == 150
        assert usage.extra_fields["provider"] == "openai"
        assert usage.extra_fields["api_type"] == "chat_completions"

    def test_usage_from_response_embeddings(self):
        """Test usage extraction from embeddings response."""

        # Simple class-based mock for embeddings usage
        class EmbeddingsUsage:
            def __init__(self):
                self.prompt_tokens = 75
                self.total_tokens = 75
                # No completion_tokens attribute

        class CreateEmbeddingResponse:
            def __init__(self):
                self.usage = EmbeddingsUsage()

        mock_response = CreateEmbeddingResponse()

        usage = self.adapter.usage_from_response(mock_response)

        assert isinstance(usage, Usage)
        assert usage.input_tokens == 75
        assert usage.output_tokens == 0  # No completion tokens for embeddings
        assert usage.cached_tokens == 0
        assert usage.total_tokens == 75  # Auto-computed as input + output
        assert usage.extra_fields["provider"] == "openai"
        assert usage.extra_fields["api_type"] == "embeddings"
        assert usage.extra_fields["total_tokens"] == 75

    def test_usage_from_response_embeddings_dict(self):
        """Test usage extraction from embeddings response as dict."""
        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "usage": {"prompt_tokens": 50, "total_tokens": 50},
        }

        usage = self.adapter.usage_from_response(mock_response)

        assert isinstance(usage, Usage)
        assert usage.input_tokens == 50
        assert usage.output_tokens == 0  # No completion tokens for embeddings
        assert usage.cached_tokens == 0
        assert usage.total_tokens == 50
        assert usage.extra_fields["provider"] == "openai"
        assert usage.extra_fields["api_type"] == "embeddings"

    def test_usage_from_response_audio_transcription(self):
        """Test usage extraction from audio transcription response."""

        # Simple class-based mock for audio usage
        class AudioUsage:
            def __init__(self):
                self.input_tokens = 120  # Audio input tokens
                self.output_tokens = 25  # Transcribed text tokens

        class Transcription:
            def __init__(self):
                self.usage = AudioUsage()

        mock_response = Transcription()

        usage = self.adapter.usage_from_response(mock_response)

        assert isinstance(usage, Usage)
        assert usage.input_tokens == 120
        assert usage.output_tokens == 25
        assert usage.cached_tokens == 0
        assert usage.total_tokens == 145
        assert usage.extra_fields["provider"] == "openai"
        assert usage.extra_fields["api_type"] == "audio"

    def test_usage_from_response_with_cached_tokens(self):
        """Test usage extraction with cached tokens."""
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 200
        mock_response.usage.completion_tokens = 100
        mock_response.usage.prompt_tokens_details = Mock()
        mock_response.usage.prompt_tokens_details.cached_tokens = 50
        type(mock_response).__name__ = "ChatCompletion"

        usage = self.adapter.usage_from_response(mock_response)

        assert usage.input_tokens == 200
        assert usage.output_tokens == 100
        assert usage.cached_tokens == 50
        assert usage.total_tokens == 300

    def test_usage_from_response_dict_with_cached_tokens(self):
        """Test usage extraction from dict response with cached tokens."""
        mock_response = {
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 75,
                "total_tokens": 225,
                "prompt_tokens_details": {"cached_tokens": 25},
            }
        }

        usage = self.adapter.usage_from_response(mock_response)

        assert usage.input_tokens == 150
        assert usage.output_tokens == 75
        assert usage.cached_tokens == 25
        assert usage.total_tokens == 225

    def test_usage_from_response_no_usage_error(self):
        """Test error when no usage data is found."""

        # Use a simple class without usage attribute instead of Mock
        class SomeResponse:
            pass

        mock_response = SomeResponse()

        with pytest.raises(TokenExtractionError) as exc_info:
            self.adapter.usage_from_response(mock_response)

        # The error now comes from normalize_usage, not the initial extraction
        assert "Could not extract required" in str(exc_info.value)
        assert "SomeResponse" in str(exc_info.value)

    def test_usage_from_response_missing_required_tokens_error(self):
        """Test error when required token fields are missing."""

        # Use a simple class with empty usage instead of Mock
        class EmptyUsage:
            pass  # No token attributes at all

        class ChatCompletion:
            def __init__(self):
                self.usage = EmptyUsage()

        mock_response = ChatCompletion()

        with pytest.raises(TokenExtractionError) as exc_info:
            self.adapter.usage_from_response(mock_response)

        assert "Could not extract required" in str(exc_info.value)

    def test_extract_tokens_backward_compatibility(self):
        """Test that extract_tokens still works for backward compatibility."""
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 80
        mock_response.usage.completion_tokens = 40
        mock_response.usage.prompt_tokens_details = Mock()
        mock_response.usage.prompt_tokens_details.cached_tokens = 5
        type(mock_response).__name__ = "ChatCompletion"

        tokens = self.adapter.extract_tokens(mock_response)

        assert tokens == (80, 40, 5)  # (input, output, cached)

    def test_detect_model_from_kwargs(self):
        """Test model detection from various kwargs patterns."""
        # Chat model
        model = self.adapter.detect_model(Mock(), (), {"model": "gpt-4o"})
        assert model == "gpt-4o"

        # Embedding model
        model = self.adapter.detect_model(
            Mock(), (), {"model": "text-embedding-3-small"}
        )
        assert model == "text-embedding-3-small"

        # Audio model
        model = self.adapter.detect_model(Mock(), (), {"model": "whisper-1"})
        assert model == "whisper-1"

        # No model specified
        model = self.adapter.detect_model(Mock(), (), {})
        assert model is None

    def test_calculate_cost_different_models(self):
        """Test cost calculation for different model types."""
        # Chat model cost
        cost = self.adapter.calculate_cost("gpt-4o", 1000, 500, 0)
        expected = 1000 * 2.50e-6 + 500 * 10.00e-6  # input + output
        assert abs(cost - expected) < 1e-10

        # Embedding model cost (no output tokens)
        cost = self.adapter.calculate_cost("text-embedding-3-small", 1000, 0, 0)
        expected = 1000 * 0.02e-6  # only input tokens
        assert abs(cost - expected) < 1e-10

    def test_normalize_usage_edge_cases(self):
        """Test usage normalization edge cases."""
        # Test with minimal embedding usage
        usage_dict = {"prompt_tokens": 100, "total_tokens": 100}
        normalized = self.adapter._normalize_usage(usage_dict)

        assert normalized["input_tokens"] == 100
        assert normalized["output_tokens"] == 0  # Embeddings have no output
        assert normalized["cached_tokens"] == 0

        # Test with audio usage pattern
        usage_dict = {"input_tokens": 150, "output_tokens": 30}
        normalized = self.adapter._normalize_usage(usage_dict)

        assert normalized["input_tokens"] == 150
        assert normalized["output_tokens"] == 30
        assert normalized["cached_tokens"] == 0


class TestOpenAIRealSDKCompatibility:
    """Test compatibility with real OpenAI SDK response structures."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("tokenx.providers.openai.load_yaml_prices") as mock_prices:
            mock_prices.return_value = {"openai": {}}  # Minimal pricing
            self.adapter = OpenAIAdapter()

    def test_real_chat_completion_structure(self):
        """Test with realistic ChatCompletion response structure."""
        # Based on actual OpenAI SDK response structure
        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        }

        usage = self.adapter.usage_from_response(mock_response)

        assert usage.input_tokens == 12
        assert usage.output_tokens == 8
        assert usage.cached_tokens == 0
        assert usage.total_tokens == 20

    def test_real_embeddings_structure(self):
        """Test with realistic CreateEmbeddingResponse structure."""
        mock_response = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        }

        usage = self.adapter.usage_from_response(mock_response)

        assert usage.input_tokens == 8
        assert usage.output_tokens == 0  # Embeddings don't have completion tokens
        assert usage.cached_tokens == 0
        assert usage.total_tokens == 8
        assert usage.extra_fields["api_type"] == "embeddings"

    @patch("tokenx.providers.openai.tiktoken.encoding_for_model")
    def test_get_encoding_for_model(self, mock_encoding):
        """Test getting encoding for a known model."""
        mock_encoding.return_value = "mock_encoding"
        encoding = self.adapter.get_encoding_for_model("gpt-4o")
        mock_encoding.assert_called_with("gpt-4o")
        assert encoding == "mock_encoding"

    @patch("tokenx.providers.openai.tiktoken.get_encoding")
    @patch("tokenx.providers.openai.tiktoken.encoding_for_model")
    def test_get_encoding_for_model_fallback(
        self, mock_encoding_for_model, mock_get_encoding
    ):
        """Test encoding fallback for unknown models."""
        # Simulate KeyError for unknown model
        mock_encoding_for_model.side_effect = KeyError("Unknown model")
        mock_get_encoding.return_value = "fallback_encoding"

        encoding = self.adapter.get_encoding_for_model("unknown-model")

        mock_encoding_for_model.assert_called_with("unknown-model")
        mock_get_encoding.assert_called_with("cl100k_base")
        assert encoding == "fallback_encoding"
