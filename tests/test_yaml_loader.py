import os
import yaml
import tempfile
from tokenx.yaml_loader import load_yaml_prices


class TestYAMLLoader:
    def test_load_multiformat_yaml(self):
        """Test loading the new multi-provider YAML format."""

        # Create a temporary YAML file with the new format
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(
                {
                    "openai": {
                        "gpt-4o": {"sync": {"in": 2.50, "cached_in": 1.25, "out": 10.0}}
                    }
                },
                tmp,
            )
            tmp_path = tmp.name

        try:
            # Use user override to test with our temporary file
            original_override = os.environ.get("TOKENX_PRICES_PATH")
            os.environ["TOKENX_PRICES_PATH"] = tmp_path

            # Clear the cache to ensure the file is reloaded
            load_yaml_prices.cache_clear()

            prices = load_yaml_prices()
            assert "openai" in prices
            assert "gpt-4o" in prices["openai"]
            assert prices["openai"]["gpt-4o"]["sync"]["in"] == 2.50 / 1e6

        finally:
            # Cleanup
            os.unlink(tmp_path)
            if original_override:
                os.environ["TOKENX_PRICES_PATH"] = original_override
            else:
                os.environ.pop("TOKENX_PRICES_PATH", None)
            load_yaml_prices.cache_clear()

    def test_backward_compatibility(self):
        """Test backward compatibility with old format."""
        from tokenx.cost_calc import PRICE_PER_TOKEN

        # Verify imported objects have expected structure
        assert any(
            model in PRICE_PER_TOKEN for model in ["gpt-4o", "gpt-3.5-turbo-0125", "o3"]
        )

        # Check price scaling
        for model, prices in PRICE_PER_TOKEN.items():
            assert prices["in"] < 1.0  # Should be scaled down from per-million
