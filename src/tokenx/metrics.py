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

ResponseT = Any  # Provider response object type alias
ReturnT = Union[ResponseT, Tuple[ResponseT, Dict[str, Any]]]


# ──────────────────────────────────────────────────────────────────────────────
# Helper: merge / create metrics dict
# ──────────────────────────────────────────────────────────────────────────────
def _merge_metrics(ret: ReturnT, **new_data) -> Tuple[ResponseT, Dict[str, Any]]:
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
def measure_latency(fn: Callable) -> Callable:
    """
    Adds `latency_ms` (float, wall-clock) to .metrics.
    Works with sync **or** async functions transparently.
    """
    is_async = inspect.iscoroutinefunction(fn)

    async def _aw(*a, **kw):
        start = time.perf_counter_ns()
        ret = await fn(*a, **kw)
        return _merge_metrics(ret, latency_ms=(time.perf_counter_ns() - start) / 1e6)

    def _sync(*a, **kw):
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
    tier: str = "sync",
    enable_caching: bool = True,
) -> Callable:
    """
    Adds `cost_usd` and token counts to .metrics by analyzing response.usage.

    Parameters
    ----------
    provider : str
        Provider name, e.g., "openai", "anthropic"
    model : str
        Model name, e.g., "gpt-4o", "claude-3.5-sonnet"
    tier : str, optional
        Pricing tier, e.g., "sync" or "flex"
    enable_caching : bool, optional
        Whether to discount cached tokens
    """

    def decorator(fn: Callable) -> Callable:
        is_async = inspect.iscoroutinefunction(fn)

        def get_calculator():
            # Use the specified provider and model
            return CostCalculator.for_provider(
                provider,
                model,
                tier=tier,
                enable_caching=enable_caching,
            )

        def get_cost_metrics(resp, calculator):
            # Cost is calculated by the provider adapter
            cost_metrics = calculator.costed()(lambda: resp)()
            # Add cost_usd for backward compatibility
            cost_metrics["cost_usd"] = cost_metrics["usd"]
            return cost_metrics

        async def _aw(*args, **kwargs):
            calculator = get_calculator()
            resp = await fn(*args, **kwargs)
            return _merge_metrics(resp, **get_cost_metrics(resp, calculator))

        def _sync(*args, **kwargs):
            calculator = get_calculator()
            resp = fn(*args, **kwargs)
            return _merge_metrics(resp, **get_cost_metrics(resp, calculator))

        return functools.wraps(fn)(_aw if is_async else _sync)

    return decorator
