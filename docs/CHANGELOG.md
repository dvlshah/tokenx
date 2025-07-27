# Changelog

## [Unreleased]

### Added
- **BaseExtractor Interface**: Formal abstract base class for provider adapters
  - All providers must implement `usage_from_response` method returning standardized `Usage` dataclass
  - Ensures consistent token extraction across providers with compile-time enforcement
  - Maintains backward compatibility through `extract_tokens` delegation
- **Usage Dataclass**: Immutable, validated data structure for token information
  - Auto-computation of `total_tokens`, validation of non-negative values
  - Capping of `cached_tokens` at `input_tokens`, support for provider-specific `extra_fields`
- Full mypy --strict compliance across entire codebase

### Changed
- Custom provider registration now requires implementing `usage_from_response` abstract method
- Updated documentation examples to use new BaseExtractor interface

## [0.2.6] - 2025-05-10

### Added
- **Anthropic Provider Support**:
    - Full cost and latency tracking for Anthropic Claude models.
    - Integration with Anthropic's Messages API.
    - Extraction of Anthropic's prompt caching beta metrics:
        - `cache_read_input_tokens` is mapped to `cached_tokens` in the metrics output and used for potential cost discounts if a `cached_in` price is defined.
        - `cache_creation_input_tokens` is added directly to the metrics output.
    - Requires users to enable the Anthropic beta client (e.g., `client.beta`) and use `cache_control` in messages to receive cache metrics.
    - Updated Anthropic model pricing to `model_prices.yaml`.
- New unit tests for the Anthropic adapter, including caching scenarios.

### Changed
- Updated `README.md` provider compatibility matrix and feature list.
- `measure_cost` decorator now includes `cache_creation_input_tokens` in the metrics dictionary for Anthropic calls when available.
- Refined `extract_tokens_with_fallbacks` in `src/tokenx/errors.py` to better handle provider-specific `TokenExtractionError`s and improve fallback logic.
- Ensured `cost_usd` key is consistently present in the metrics output from `measure_cost` for backward compatibility.

### Fixed
- Corrected test assertions for Anthropic adapter related to default cached token values.

## [0.2.0] - 2025-05-03

### Added
- Provider architecture for supporting multiple LLM providers
- Provider registry with auto-discovery mechanism
- Provider interface for implementing new adapters
- OpenAI provider adapter implementation with enhanced error handling
- Support for all OpenAI response formats (dict, Pydantic models)
- Support for cached token extraction and pricing
- Centralized error handling module with detailed error messages
- Enhanced documentation with provider compatibility matrix
- Contributing guidelines

### Changed
- Refactored cost calculation to use provider adapters
- Updated metrics decorators to require explicit provider and model
- Enhanced token extraction to handle different response formats
- Improved error messages for common failure scenarios
- Updated tests to reflect new API requirements

### Maintained
- Full backward compatibility with existing OpenAI code
- Same decorator-based API for cost and latency measurement
- Support for different pricing tiers and caching discounts

## [0.1.0] - 2025-04-01

### Added
- Initial release with OpenAI cost calculation
- Support for latency measurement via decorators
- Token counting with tiktoken
- Support for different pricing tiers
- Caching discount support