from pathlib import Path
import yaml
import tempfile
from tokenx.yaml_loader import load_yaml_prices


class TestYAMLLoader:
    def test_load_multiformat_yaml(self, monkeypatch):
        """Test loading the new multi-provider YAML format."""

        # Create a temporary YAML file with the new format
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml") as tmp:
            yaml.dump(
                {
                    "openai": {
                        "gpt-4o": {"sync": {"in": 2.50, "cached_in": 1.25, "out": 10.0}}
                    }
                },
                tmp,
            )
            tmp.flush()

            # Monkeypatch the YAML path to use our temporary file
            monkeypatch.setattr("tokenx.yaml_loader.YAML_PATH", Path(tmp.name))

            # Clear the cache to ensure the file is reloaded
            load_yaml_prices.cache_clear()

            prices = load_yaml_prices()
            assert "openai" in prices
            assert "gpt-4o" in prices["openai"]
            assert prices["openai"]["gpt-4o"]["sync"]["in"] == 2.50 / 1e6

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
