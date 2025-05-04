"""
Test script for error handling in tokenx.

This script demonstrates the enhanced error handling capabilities
for the OpenAI provider adapter.
"""


def test_error_handling():
    """Test the enhanced error handling for provider adapters."""
    from tokenx.providers.openai import create_openai_adapter
    from tokenx.errors import TokenExtractionError, PricingError

    # Create an enhanced OpenAI adapter
    adapter = create_openai_adapter()

    print("=== Testing Error Handling ===\n")

    # Test 1: Invalid model pricing
    print("Test 1: Invalid Model Error")
    try:
        adapter.calculate_cost("nonexistent-model", 100, 50)
    except PricingError as e:
        print(f"✅ Successfully caught PricingError:\n{e}\n")

    # Test 2: Invalid response format
    print("Test 2: Invalid Response Format")
    try:
        # Create a response with no usage data
        class BadResponse:
            pass

        adapter.extract_tokens(BadResponse())
    except TokenExtractionError as e:
        print(f"✅ Successfully caught TokenExtractionError:\n{e}\n")

    # Test 3: Test fallback extraction
    print("Test 3: Response Format with Usage as Attribute")

    class UsageWithAttribute:
        def __init__(self):
            self.prompt_tokens = 100
            self.completion_tokens = 50

    class ResponseWithUsageAttribute:
        def __init__(self):
            self.usage = UsageWithAttribute()

    try:
        input_tokens, output_tokens, cached_tokens = adapter.extract_tokens(
            ResponseWithUsageAttribute()
        )
        print("✅ Successfully extracted tokens from Pydantic-like response:")
        print(f"   Input tokens: {input_tokens}")
        print(f"   Output tokens: {output_tokens}")
        print(f"   Cached tokens: {cached_tokens}\n")
    except Exception as e:
        print(f"❌ Failed to extract tokens: {e}\n")

    # Test 4: Test dictionary format
    print("Test 4: Response Format with Usage as Dictionary")

    response_dict = {
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 50},
        }
    }

    try:
        input_tokens, output_tokens, cached_tokens = adapter.extract_tokens(
            response_dict
        )
        print("✅ Successfully extracted tokens from dictionary response:")
        print(f"   Input tokens: {input_tokens}")
        print(f"   Output tokens: {output_tokens}")
        print(f"   Cached tokens: {cached_tokens}\n")
    except Exception as e:
        print(f"❌ Failed to extract tokens: {e}\n")

    # Test 5: Cost calculation with cached tokens
    print("Test 5: Cost Calculation with Cached Tokens")

    try:
        # Use a model that exists in the YAML file
        cost = adapter.calculate_cost("gpt-4o", 100, 50, 20)
        print(f"✅ Successfully calculated cost: ${cost:.6f}\n")
    except Exception as e:
        print(f"❌ Failed to calculate cost: {e}\n")


if __name__ == "__main__":
    test_error_handling()
