# test_costs.py  (drop in same folder)
"""
Robust tests for cost_calc.py

We query the calculator for the *actual* token counts of our synthetic
prompts so that the expected-cost formula matches reality.
"""

import math

import pytest

from tokenx.cost_calc import PRICE_PER_TOKEN, OpenAICostCalculator as Calc

_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o3",
    "o4-mini",
    "gpt-3.5-turbo-0125",
]


@pytest.mark.parametrize("model", _MODELS)
def test_blended_cost(model):
    calc = Calc(model, enable_caching=True)

    # a reasonably varied string → no token collapse surprises
    prompt = "alpha bravo charlie delta echo foxtrot " * 25
    completion = "one two three four five " * 10

    p_tok = calc._count(prompt)
    c_tok = calc._count(completion)
    cached = min(1024, p_tok)  # same clamp the library uses

    got = calc.blended_cost(prompt, completion, cached)

    price = PRICE_PER_TOKEN[model]
    expected = (
        (p_tok - cached) * price["in"]
        + cached * price["cached_in"]
        + c_tok * price["out"]
    )
    assert math.isclose(got, expected, rel_tol=1e-9)


def test_cost_from_usage():
    calc = Calc("gpt-4o-mini", enable_caching=True)

    prompt_tok, comp_tok, cached_tok = 1600, 100, 1024
    usage = {
        "prompt_tokens": prompt_tok,
        "completion_tokens": comp_tok,
        "prompt_tokens_details": {"cached_tokens": cached_tok},
    }

    price = PRICE_PER_TOKEN["gpt-4o-mini"]
    expected = (
        (prompt_tok - cached_tok) * price["in"]
        + cached_tok * price["cached_in"]
        + comp_tok * price["out"]
    )
    assert math.isclose(calc.cost_from_usage(usage), expected, rel_tol=1e-9)
