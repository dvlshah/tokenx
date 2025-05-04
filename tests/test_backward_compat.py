from tokenx.cost_calc import OpenAICostCalculator


class TestBackwardCompatibility:
    def test_openai_calculator(self):
        """Test that OpenAICostCalculator works as before."""
        calc = OpenAICostCalculator("gpt-4o")

        # Test the _count method (token counting)
        token_count = calc._count("Hello, world!")
        assert token_count > 0

        # Test blended_cost method
        cost = calc.blended_cost("Hello", "world", 0)
        assert cost > 0

        # Test with caching
        cost_with_cache = calc.blended_cost("Hello", "world", 1)
        assert cost_with_cache < cost  # Should be cheaper with caching
