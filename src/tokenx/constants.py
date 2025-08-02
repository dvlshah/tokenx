"""
Constants for tokenx package.

This module defines commonly used constants to avoid magic strings
and improve maintainability.
"""

# Provider names
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GEMINI = "gemini"

# Pricing tiers
TIER_SYNC = "sync"
TIER_FLEX = "flex"
TIER_BATCH = "batch"

# Token field names
TOKEN_FIELD_INPUT = "input_tokens"
TOKEN_FIELD_OUTPUT = "output_tokens"
TOKEN_FIELD_CACHED = "cached_tokens"
TOKEN_FIELD_PROMPT = "prompt_tokens"
TOKEN_FIELD_COMPLETION = "completion_tokens"
TOKEN_FIELD_TOTAL = "total_tokens"

# Usage field names
USAGE_FIELD = "usage"

# Currency
CURRENCY_USD = "USD"

# API types
API_TYPE_CHAT = "chat_completions"
API_TYPE_EMBEDDINGS = "embeddings"
API_TYPE_AUDIO = "audio"
API_TYPE_IMAGES = "images"
API_TYPE_MODERATION = "moderation"
API_TYPE_RESPONSES = "responses"

# Default values
DEFAULT_CACHED_TOKENS = 0
DEFAULT_TIER = TIER_SYNC
DEFAULT_ENABLE_CACHING = True

# OpenAI specific constants
OPENAI_MODEL_PREFIXES = [
    "gpt-",
    "text-",
    "o1-",
    "o3-",
    "text-embedding-",
    "whisper-",
    "tts-",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "gpt-4o-mini-tts",
    "gpt-4o-audio-preview",
    "gpt-4o-mini-audio-preview",
    "gpt-4o-realtime-preview",
    "gpt-4o-mini-realtime-preview",
    "dall-e-",
]

# Quality mappings for image generation
IMAGE_QUALITY_MAPPING = {"standard": "low", "hd": "high", "medium": "medium"}

# Default image settings
DEFAULT_IMAGE_SIZE = "1024x1024"
DEFAULT_IMAGE_QUALITY = "medium"
DEFAULT_IMAGE_FALLBACK_KEY = "images_low_1024x1024"

# Token estimation constants
CHARS_PER_TOKEN_ESTIMATE = 4  # Rough approximation for content estimation
