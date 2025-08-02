"""
Tests for hybrid pricing functionality in OpenAI adapter.

This module tests the hybrid pricing feature for gpt-image-1 model that
combines token-based costs with per-image costs for accurate billing.
"""

from unittest.mock import MagicMock, patch

from tokenx.providers.openai import OpenAIAdapter


class TestHybridPricing:
    """Test hybrid pricing functionality for gpt-image-1 model."""

    def setup_method(self):
        """Test setup."""
        with patch("tokenx.providers.openai.load_yaml_prices") as mock_prices:
            mock_prices.return_value = {
                "openai": {
                    "gpt-image-1": {
                        "sync": {
                            # Text token pricing (per token rates)
                            "in": 0.000005,  # $5.00 per 1M input tokens (text)
                            "cached_in": 0.00000125,  # $1.25 per 1M cached input tokens (text)
                            "out": None,  # No text output tokens
                            # Image token pricing (per token rates) - use audio_in for dual pricing
                            "audio_in": 0.00001,  # $10.00 per 1M input tokens (image)
                            "audio_cached_in": 0.0000025,  # $2.50 per 1M cached input tokens (image)
                            "audio_out": 0.00004,  # $40.00 per 1M output tokens (image)
                            # Per-image costs
                            "images_low_1024x1024": 0.011000,  # $0.011 per image
                            "images_low_1024x1536": 0.016000,  # $0.016 per image
                            "images_low_1536x1024": 0.016000,  # $0.016 per image
                            "images_medium_1024x1024": 0.042000,  # $0.042 per image
                            "images_medium_1024x1536": 0.063000,  # $0.063 per image
                            "images_medium_1536x1024": 0.063000,  # $0.063 per image
                            "images_high_1024x1024": 0.167000,  # $0.167 per image
                            "images_high_1024x1536": 0.250000,  # $0.250 per image
                            "images_high_1536x1024": 0.250000,  # $0.250 per image
                        }
                    }
                }
            }
            self.adapter = OpenAIAdapter()

    def test_calculate_image_cost_low_quality(self):
        """Test calculate_image_cost for low quality images."""
        # Mock response with low quality 1024x1024 image
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(revised_prompt="test", url="http://example.com/image.png")
        ]

        # Mock request parameters
        size = "1024x1024"
        quality = "standard"  # Maps to "low" in pricing

        # Mock the response to have size and quality attributes
        mock_response.size = size
        mock_response.quality = quality

        cost = self.adapter.calculate_image_cost("gpt-image-1", mock_response)

        assert cost == 0.011000  # $0.011 for one low quality 1024x1024 image

    def test_calculate_image_cost_high_quality_multiple(self):
        """Test calculate_image_cost for multiple high quality images."""
        # Mock response with 2 high quality images
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(revised_prompt="test1", url="http://example.com/image1.png"),
            MagicMock(revised_prompt="test2", url="http://example.com/image2.png"),
        ]

        # Mock request parameters
        size = "1024x1536"
        quality = "hd"  # Maps to "high" in pricing

        # Mock the response to have size and quality attributes
        mock_response.size = size
        mock_response.quality = quality

        cost = self.adapter.calculate_image_cost("gpt-image-1", mock_response)

        assert cost == 0.500000  # $0.250 * 2 for two high quality 1024x1536 images

    def test_calculate_image_cost_medium_quality(self):
        """Test calculate_image_cost for medium quality images."""
        # Mock response with medium quality image
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(revised_prompt="test", url="http://example.com/image.png")
        ]

        # Mock request parameters - medium is implied when not standard or hd
        size = "1536x1024"
        quality = "medium"

        # Mock the response to have size and quality attributes
        mock_response.size = size
        mock_response.quality = quality

        cost = self.adapter.calculate_image_cost("gpt-image-1", mock_response)

        assert cost == 0.063000  # $0.063 for one medium quality 1536x1024 image

    def test_calculate_image_cost_unsupported_size(self):
        """Test calculate_image_cost with unsupported size falls back to base cost."""
        # Mock response
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(revised_prompt="test", url="http://example.com/image.png")
        ]

        # Unsupported size
        size = "512x512"
        quality = "standard"

        # Mock the response to have size and quality attributes
        mock_response.size = size
        mock_response.quality = quality

        cost = self.adapter.calculate_image_cost("gpt-image-1", mock_response)

        # Should fallback to default low quality 1024x1024 pricing
        assert cost == 0.011000

    def test_hybrid_cost_calculation_text_and_images(self):
        """Test complete hybrid cost calculation with both tokens and images."""
        # Mock response with token usage and generated images
        mock_response = MagicMock()

        # Token usage
        mock_response.usage.prompt_tokens = 100  # Text input tokens
        mock_response.usage.completion_tokens = 0  # No text output

        # Create proper mock details with actual integers
        mock_prompt_details = MagicMock()
        mock_prompt_details.image_tokens = 50  # Actual integer
        mock_response.usage.prompt_tokens_details = mock_prompt_details

        mock_completion_details = MagicMock()
        mock_completion_details.image_tokens = 200  # Actual integer
        mock_response.usage.completion_tokens_details = mock_completion_details

        # Generated images
        mock_response.data = [
            MagicMock(revised_prompt="test", url="http://example.com/image.png")
        ]

        # Calculate cost with hybrid pricing
        total_cost = self.adapter.calculate_cost(
            model="gpt-image-1",
            input_tokens=150,  # 100 text + 50 image tokens
            output_tokens=0,  # No standard output tokens for image model
            cached_tokens=0,
            tier="sync",
            response=mock_response,
        )

        # Expected breakdown:
        # Text tokens: 100 * $0.000005 = $0.0005
        # Image input tokens: 50 * $0.00001 = $0.0005
        # Image output tokens: 200 * $0.00004 = $0.008
        # Per-image cost: 1 * $0.011 = $0.011
        expected_token_cost = (
            100 * 0.000005  # Text input
            + 50 * 0.00001  # Image input
            + 200 * 0.00004  # Image output
        )
        expected_image_cost = 0.011000  # One standard quality image
        expected_total = expected_token_cost + expected_image_cost

        # Use reasonable tolerance for floating point comparison
        assert abs(total_cost - expected_total) < 1e-6

    def test_hybrid_cost_with_caching(self):
        """Test hybrid cost calculation with cached tokens."""
        # Mock response with cached tokens
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 200
        mock_response.usage.completion_tokens = 0

        # Create proper mock details with actual integers
        mock_prompt_details = MagicMock()
        mock_prompt_details.cached_tokens = 50  # Actual integer
        mock_prompt_details.image_tokens = 100  # Actual integer
        mock_response.usage.prompt_tokens_details = mock_prompt_details

        mock_completion_details = MagicMock()
        mock_completion_details.image_tokens = 150  # Actual integer
        mock_response.usage.completion_tokens_details = mock_completion_details

        # Generated images
        mock_response.data = [
            MagicMock(revised_prompt="test1", url="http://example.com/image1.png"),
            MagicMock(revised_prompt="test2", url="http://example.com/image2.png"),
        ]

        total_cost = self.adapter.calculate_cost(
            model="gpt-image-1",
            input_tokens=200,
            output_tokens=0,  # No standard output tokens for image model
            cached_tokens=50,
            tier="sync",
            response=mock_response,
        )

        # Expected breakdown:
        # Text tokens: (200-100-50) * $0.000005 + 50 * $0.00000125 = 50 uncached text + 50 cached text
        # Image input tokens: 100 * $0.00001
        # Image output tokens: 150 * $0.00004
        # Per-image cost: 2 * $0.011 = $0.022
        expected_cost = (
            50 * 0.000005  # Uncached text tokens
            + 50 * 0.00000125  # Cached text tokens
            + 100 * 0.00001  # Image input tokens
            + 150 * 0.00004  # Image output tokens
            + 0.022000  # Two images
        )

        # Use reasonable tolerance for floating point comparison
        assert abs(total_cost - expected_cost) < 1e-6

    def test_image_model_detection(self):
        """Test that gpt-image-1 model is properly detected."""
        # Test with image generation parameters
        assert self.adapter.matches_function(
            lambda: None,
            (),
            {
                "model": "gpt-image-1",
                "prompt": "A beautiful sunset",
                "size": "1024x1024",
                "quality": "standard",
            },
        )

        # Test with different image parameters
        assert self.adapter.matches_function(
            lambda: None,
            (),
            {
                "prompt": "A mountain landscape",
                "size": "1024x1536",
                "quality": "hd",
                "style": "vivid",
            },
        )

    def test_quality_mapping(self):
        """Test quality parameter mapping to pricing tiers."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]

        # Test standard -> low mapping
        mock_response.size = "1024x1024"
        mock_response.quality = "standard"
        cost_standard = self.adapter.calculate_image_cost("gpt-image-1", mock_response)
        assert cost_standard == 0.011000  # Low quality price

        # Test hd -> high mapping
        mock_response.quality = "hd"
        cost_hd = self.adapter.calculate_image_cost("gpt-image-1", mock_response)
        assert cost_hd == 0.167000  # High quality price

        # Test default -> medium mapping
        mock_response.quality = "medium"
        cost_default = self.adapter.calculate_image_cost("gpt-image-1", mock_response)
        assert cost_default == 0.042000  # Medium quality price

    def test_hybrid_pricing_edge_cases(self):
        """Test edge cases in hybrid pricing."""
        mock_response = MagicMock()
        mock_response.data = []  # No images generated

        # Test with no images generated
        mock_response.size = "1024x1024"
        mock_response.quality = "standard"
        cost = self.adapter.calculate_image_cost("gpt-image-1", mock_response)
        assert cost == 0.0  # No cost if no images generated

        # Test with more images requested than generated
        mock_response.data = [MagicMock()]  # Only 1 image generated
        cost = self.adapter.calculate_image_cost("gpt-image-1", mock_response)
        assert cost == 0.011000  # Cost for 1 image actually generated
