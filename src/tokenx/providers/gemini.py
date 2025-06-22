"""Gemini Provider Adapter Implementation."""

from typing import Any, Dict, Optional, Tuple

from .base import ProviderAdapter
from ..yaml_loader import load_yaml_prices
from ..errors import enhance_provider_adapter, TokenExtractionError, PricingError


class GeminiAdapter(ProviderAdapter):
    """Adapter for Google Gemini API cost calculation."""

    def __init__(self) -> None:
        self._prices = load_yaml_prices().get("gemini", {})

    @property
    def provider_name(self) -> str:
        return "gemini"

    def matches_function(self, func: Any, args: tuple, kwargs: dict) -> bool:
        module_name = getattr(func, "__module__", "").lower()
        if "generativeai" in module_name or "generativelanguage" in module_name:
            return True

        if args:
            cls = getattr(args[0], "__class__", None)
            if cls and "generativemodel" in cls.__name__.lower():
                return True

        model = kwargs.get("model") or kwargs.get("model_name")
        if isinstance(model, str) and "gemini" in model.lower():
            return True

        return False

    def _get_usage(self, response: Any) -> Optional[Any]:
        if hasattr(response, "usage_metadata"):
            return response.usage_metadata
        if hasattr(response, "usage"):
            return response.usage
        if isinstance(response, dict):
            if "usage_metadata" in response:
                return response["usage_metadata"]
            if "usage" in response:
                return response["usage"]
        return None

    def extract_tokens(self, response: Any) -> Tuple[int, int, int]:
        usage = self._get_usage(response)
        if usage is None:
            raise TokenExtractionError(
                "Could not extract usage data from Gemini response.",
                self.provider_name,
                type(response).__name__,
            )

        def _field(obj: Any, name: str) -> Optional[int]:
            if hasattr(obj, name):
                val = getattr(obj, name)
                if val is not None:
                    return int(val)
            if isinstance(obj, Dict):
                val = obj.get(name)
                if val is not None:
                    return int(val)
            return None

        input_tokens = _field(usage, "prompt_token_count")
        if input_tokens is None:
            input_tokens = _field(usage, "prompt_tokens")

        output_tokens = _field(usage, "candidates_token_count")
        if output_tokens is None:
            output_tokens = _field(usage, "output_tokens")

        cached_tokens = _field(usage, "cached_content_token_count") or 0

        if input_tokens is None or output_tokens is None:
            raise TokenExtractionError(
                "Could not extract 'prompt_token_count' or 'candidates_token_count'.",
                self.provider_name,
                type(usage).__name__,
            )

        return input_tokens, output_tokens, cached_tokens

    def detect_model(self, func: Any, args: tuple, kwargs: dict) -> Optional[str]:
        model = kwargs.get("model") or kwargs.get("model_name")
        if isinstance(model, str):
            return model
        return None

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        tier: str = "sync",
    ) -> float:
        if model not in self._prices:
            raise PricingError(
                f"Price for model={model!r} not found in YAML",
                self.provider_name,
                model,
                available_models=list(self._prices.keys()),
            )
        if tier not in self._prices[model]:
            raise PricingError(
                f"Price for model={model!r} tier={tier!r} not found in YAML",
                self.provider_name,
                model,
                tier,
                available_models=list(self._prices[model].keys()),
            )

        price = self._prices[model][tier]

        cached_tokens = max(0, min(cached_tokens, input_tokens))
        cost = 0.0
        if price.get("cached_in") is not None:
            cost += (input_tokens - cached_tokens) * price["in"]
            cost += cached_tokens * price["cached_in"]
        else:
            cost += input_tokens * price.get("in", 0)

        if price.get("out") is not None:
            cost += output_tokens * price["out"]

        return cost


def create_gemini_adapter() -> GeminiAdapter:
    adapter = GeminiAdapter()
    return enhance_provider_adapter(adapter)
