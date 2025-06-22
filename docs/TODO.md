# tokenx Project Status & Roadmap

## What Works

1. **Provider Interface Architecture**: Core provider pattern is implemented and works correctly
2. **YAML Configuration**: Multi-provider YAML structure properly loads and scales pricing data
3. **Cost Calculator Core**: Base calculation logic for tokens-to-cost conversion works correctly
4. **OpenAI Support**: Full OpenAI support is in place, with robust cost calculations working, including cached token discounts.
5. **Anthropic Support**: Full Anthropic support is in place, with robust cost calculations, including extraction of prompt caching beta metrics (`cache_read_input_tokens` as `cached_tokens`, and `cache_creation_input_tokens`).
6. **Provider Registry**: Auto-discovery mechanism for providers is implemented
7. **Backward Compatibility**: Core calculations maintain backward compatibility with the old API
8. **Error Handling**: Comprehensive error handling with detailed messages is implemented
9. **Token Extraction**: Support for OpenAI response formats (dict, Pydantic) and Anthropic response formats is working.
10. **Cached Token Handling**: Support for extracting and pricing cached tokens for OpenAI and Anthropic (prompt caching beta) is working.

## TODO List for Phase 2 (Anthropic Support) - COMPLETED on 10/05/2025

1. **Anthropic Provider Setup**:
   - [x] Create Anthropic adapter class implementing ProviderAdapter
   - [x] Implement token extraction from Anthropic responses
   - [x] Implement extraction of Anthropic cache metrics (`cache_read_input_tokens`, `cache_creation_input_tokens`)
   - [x] Add detection logic for Anthropic client

2. **Anthropic Pricing**:
   - [x] Add Anthropic model pricing to YAML configuration
   - [x] Support Anthropic-specific pricing structures, including potential for `cached_in` rates.

3. **Testing for Anthropic**:
   - [x] Develop unit tests for Anthropic adapter, including cache metric scenarios.
   - [x] Create integration tests for Anthropic API (manual live tests performed, CI for unit tests).

## TODO List for Phase 3 (Google Gemini Support)

1. **Gemini Provider Setup**:
   - [x] Create Gemini adapter class implementing ProviderAdapter
   - [x] Implement token extraction from Gemini responses
   - [x] Add detection logic for Gemini client

2. **Gemini Pricing**:
   - [x] Add Gemini model pricing to YAML configuration
   - [ ] Support Gemini-specific pricing structures and volume-based pricing

3. **Testing for Gemini**:
   - [x] Develop unit tests for Gemini adapter
   - [ ] Create integration tests for Gemini API

## Future Enhancements

1. **Documentation**:
   - [ ] Add more examples for each provider, especially for advanced caching scenarios.
   - [ ] Create usage guides for common scenarios
   - [ ] Add API reference documentation

2. **Metrics Improvements**:
   - [ ] Enhance latency tracking with additional breakdowns (e.g., time to first token for streaming).
   - [ ] Support for custom metrics collection.

3. **Integration Features**:
   - [ ] Add logging integrations (e.g., structlog).
   - [ ] Add monitoring dashboard integrations (e.g., Prometheus, Datadog exporters).
   - [ ] Create CLI tools for cost estimation from text or files.

4. **Performance Optimization**:
   - [ ] Add caching for token counting (if `tiktoken` becomes a bottleneck, unlikely for typical use).
   - [ ] Optimize token extraction for speed on very large/frequent responses.
   - [ ] Reduce memory usage for large responses.

5. **Streaming Support Enhancements**:
    - [ ] Investigate robust ways to aggregate token usage from streaming responses if the final response object isn't returned by the decorated function.