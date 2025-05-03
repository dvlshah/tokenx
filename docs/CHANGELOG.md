# Changelog

All notable changes to the tokenx project will be documented in this file.

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