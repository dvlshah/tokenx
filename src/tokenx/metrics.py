"""
metrics.py
----------
Two decorators that add a `.metrics` dictionary to any function returning
an LLM provider response object (sync **or** async):

    @measure_cost(provider="openai", model="gpt-4o", tier="sync", enable_caching=True)
    @measure_latency

- Works with any LLM provider (OpenAI, Anthropic, Google Gemini)
- Decorators are order-agnostic; they merge their keys in the returned
  tuple: (response, metrics_dict).
"""

from __future__ import annotations

import functools
import inspect
import time
from typing import Any, Callable, Dict, Tuple, Union

from .cost_calc import CostCalculator
from .constants import (
    DEFAULT_TIER,
    DEFAULT_ENABLE_CACHING,
    CURRENCY_USD,
    PROVIDER_ANTHROPIC,
)

ResponseT = Any  # Provider response object type alias
ReturnT = Union[ResponseT, Tuple[ResponseT, Dict[str, Any]]]


# ──────────────────────────────────────────────────────────────────────────────
# Helper: merge / create metrics dict
# ──────────────────────────────────────────────────────────────────────────────
def _merge_metrics(ret: ReturnT, **new_data: Any) -> Tuple[ResponseT, Dict[str, Any]]:
    """
    Accepts:
        • plain response           → returns (response, {new_data})
        • (response, metrics_dict) → merges and returns
    """
    if isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[1], dict):
        resp, metrics = ret
        metrics.update(new_data)
        return resp, metrics
    else:
        return ret, dict(new_data)


# ──────────────────────────────────────────────────────────────────────────────
# Decorator 1  – latency measurement
# ──────────────────────────────────────────────────────────────────────────────
def measure_latency(fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Adds `latency_ms` (float, wall-clock) to .metrics.
    Works with sync **or** async functions transparently.
    """
    is_async = inspect.iscoroutinefunction(fn)

    async def _aw(*a: Any, **kw: Any) -> Any:
        start = time.perf_counter_ns()
        ret = await fn(*a, **kw)
        return _merge_metrics(ret, latency_ms=(time.perf_counter_ns() - start) / 1e6)

    def _sync(*a: Any, **kw: Any) -> Any:
        start = time.perf_counter_ns()
        ret = fn(*a, **kw)
        return _merge_metrics(ret, latency_ms=(time.perf_counter_ns() - start) / 1e6)

    return functools.wraps(fn)(_aw if is_async else _sync)


# ──────────────────────────────────────────────────────────────────────────────
# Decorator 2  – cost measurement
# ──────────────────────────────────────────────────────────────────────────────
def measure_cost(
    provider: str,
    model: str,
    *,
    tier: str = DEFAULT_TIER,
    enable_caching: bool = DEFAULT_ENABLE_CACHING,
) -> Callable[..., Any]:
    """
    Adds `cost_usd` and token counts to .metrics by analyzing response usage.
    Includes provider-specific cache metrics if available.

    Parameters
    ----------
    provider : str
        Provider name, e.g., "openai", "anthropic"
    model : str
        Model name, e.g., "gpt-4o", "claude-3.5-sonnet"
    tier : str, optional
        Pricing tier, e.g., "sync" or "flex"
    enable_caching : bool, optional
        Whether to discount cached tokens if provider supports it.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        is_async = inspect.iscoroutinefunction(fn)

        def get_calculator() -> CostCalculator:
            # Use the specified provider and model
            return CostCalculator.for_provider(
                provider,
                model,
                tier=tier,
                enable_caching=enable_caching,
            )

        def get_cost_metrics(resp: Any, calculator: CostCalculator) -> Dict[str, Any]:
            # Cost is calculated by the provider adapter.
            cost_metrics = calculator.costed()(lambda: resp)()
            cost_metrics["cost_usd"] = cost_metrics["usd"]
            cost_metrics["currency"] = CURRENCY_USD  # Explicit currency indicator

            # Add provider-specific metrics if the adapter supports extracting them
            if (
                hasattr(calculator.provider, "_extract_anthropic_usage_fields")
                and provider == PROVIDER_ANTHROPIC
            ):
                try:
                    # Need to extract usage data first to pass to _extract_anthropic_usage_fields
                    usage_data = None
                    if hasattr(resp, "usage"):
                        usage_data = resp.usage
                    elif isinstance(resp, dict) and "usage" in resp:
                        usage_data = resp["usage"]
                    # Handle top-level token counts if usage data is not nested
                    elif hasattr(resp, "input_tokens") and hasattr(
                        resp, "output_tokens"
                    ):
                        usage_data = resp
                    elif (
                        isinstance(resp, dict)
                        and "input_tokens" in resp
                        and "output_tokens" in resp
                    ):
                        usage_data = resp

                    if usage_data is not None:
                        anthropic_fields = (
                            calculator.provider._extract_anthropic_usage_fields(  # type: ignore
                                usage_data
                            )
                        )
                        # Add relevant Anthropic-specific cache metrics
                        # cache_read_input_tokens is already mapped to 'cached_tokens'
                        cost_metrics["cache_creation_input_tokens"] = (
                            anthropic_fields.get("cache_creation_input_tokens", 0)
                        )
                        # We could add cache_read_input_tokens explicitly too, but it's redundant with 'cached_tokens'
                        # cost_metrics["anthropic_cache_read_input_tokens"] = anthropic_fields.get("cache_read_input_tokens", 0)

                except Exception as e:
                    # Log a warning if extraction of extra fields fails, but don't crash
                    print(
                        f"Warning: Could not extract extra Anthropic cache metrics: {e}"
                    )

            return dict(cost_metrics)

        async def _aw(*args: Any, **kwargs: Any) -> Any:
            calculator = get_calculator()
            resp = await fn(*args, **kwargs)
            # Extract actual response if it's a tuple from latency decorator
            actual_response = (
                resp[0]
                if isinstance(resp, tuple)
                and len(resp) == 2
                and isinstance(resp[1], dict)
                else resp
            )
            # Pass the actual response object to get_cost_metrics
            return _merge_metrics(resp, **get_cost_metrics(actual_response, calculator))

        def _sync(*args: Any, **kwargs: Any) -> Any:
            calculator = get_calculator()
            resp = fn(*args, **kwargs)
            # Extract actual response if it's a tuple from latency decorator
            actual_response = (
                resp[0]
                if isinstance(resp, tuple)
                and len(resp) == 2
                and isinstance(resp[1], dict)
                else resp
            )
            # Pass the actual response object to get_cost_metrics
            return _merge_metrics(resp, **get_cost_metrics(actual_response, calculator))

        return functools.wraps(fn)(_aw if is_async else _sync)

    return decorator
