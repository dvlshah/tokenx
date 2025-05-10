<p align="center">
  <img src="https://raw.githubusercontent.com/dvlshah/tokenx/main/docs/assets/logo.png"
       alt="tokenX logo" width="280">
</p>

<p align="center"><em>Track cost and latency of your LLM calls </em></p>

<p align="center">
  <a href="https://pypi.org/project/tokenx-core/"><img src="https://img.shields.io/pypi/v/tokenx-core?logo=pypi&label=pypi" alt="PyPI version"></a>
  <a href="https://github.com/dvlshah/tokenx/actions/workflows/test.yml"><img src="https://github.com/dvlshah/tokenx/actions/workflows/test.yml/badge.svg?branch=main" alt="CI status"></a>
  <!-- <a href="https://codecov.io/gh/dvlshah/tokenx"><img src="https://img.shields.io/codecov/c/github/dvlshah/tokenx?logo=codecov" alt="Coverage"></a> -->
  <a href="https://pypi.org/project/tokenx-core/"><img src="https://img.shields.io/pypi/pyversions/tokenx-core?logo=python&label=python" alt="Python versions"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT license"></a>
</p>

<p align="center"><strong>👉 Like what you see? <a href="https://github.com/dvlshah/tokenx/stargazers">Star the repo</a>  and <a href="https://github.com/dvlshah">follow @dvlshah</a> for updates!</strong></p>

> **Decorator in → Metrics out.**
> Monitor cost & latency of any LLM function without touching its body.

```bash
pip install tokenx-core[openai]          # 1️⃣ install
```

```python
from tokenx.metrics import measure_cost, measure_latency   # 2️⃣ decorate
from openai import OpenAI

@measure_latency
@measure_cost(provider="openai", model="gpt-4o-mini")
def ask(prompt: str):
    return OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

resp, m = ask("Hello!")                                   # 3️⃣ run
print(m["cost_usd"], "USD |", m["latency_ms"], "ms")
```

---

## 🤔 Why tokenx?

Integrating with LLM APIs often involves hidden costs and variable performance. Manually tracking token usage and calculating costs across different models and providers is tedious and error-prone. `tokenx` simplifies this by:

*   **Effortless Integration:** Add monitoring with simple decorators, no need to refactor your API call logic.
*   **Accurate Cost Tracking:** Uses up-to-date, configurable pricing (including caching discounts) for precise cost analysis.
*   **Performance Insights:** Easily measure API call latency to identify bottlenecks.
*   **Multi-Provider Ready:** Designed to consistently monitor costs across different LLM vendors (OpenAI currently supported, more coming soon!).

---

## 🏗️ Architecture (1‑min overview)

```mermaid
flowchart LR
  %% --- subgraph: user's code ---
  subgraph User_Code
    A["API call"]
  end

  %% --- main pipeline ---
  A -- decorators -->   B["tokenx wrapper"]
  B -- cost      -->   C["Cost Calculator"]
  B -- latency   -->   D["Latency Timer"]
  C -- lookup    -->   E["model_prices.yaml"]
  B -- metrics   -->   F["Structured JSON (stdout / exporter)"]
```

*No vendor lock‑in:*  pure‑Python wrapper emits plain dicts—pipe them to Prometheus, Datadog, or stdout.

---

## 💡 Features at a glance

* **Track & save money** – live USD costing with cached‑token discounts
* **Trace latency** – pinpoint slow models or network hops
* **Plug‑&‑play decorators** – wrap any sync or async function
* **Provider plug‑ins** – OpenAI today, Anthropic & Gemini next
* **Typed** – 100 % `py.typed`, 95 %+ mypy coverage
* **Zero deps** – slims Docker images

---

## 📦 Installation

```bash
pip install tokenx-core                 # stable
pip install tokenx-core[openai]         # with OpenAI provider extras
pip install tokenx-core[anthropic]      # with Anthropic provider extras
pip install tokenx-core[all]            # with all provider extras
```

---

## 🚀 Quick Start

Here's how to monitor your OpenAI API calls with just two lines of code:

```python
from tokenx.metrics import measure_cost, measure_latency
from openai import OpenAI

@measure_latency
@measure_cost(provider="openai", model="gpt-4o-mini")  # Always specify provider and model
def call_openai():
    client = OpenAI()
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello, world!"}]
    )

response, metrics = call_openai()

# Access your metrics
print(f"Cost: ${metrics['cost_usd']:.6f}")
print(f"Latency: {metrics['latency_ms']:.2f}ms")
print(f"Tokens: {metrics['input_tokens']} in, {metrics['output_tokens']} out")
print(f"Cached tokens: {metrics['cached_tokens']}")  # New in v0.2.0
```

Here's how to monitor your Anthropic API calls with just two lines of code:

```python
from tokenx.metrics import measure_cost, measure_latency
from anthropic import Anthropic # Or AsyncAnthropic

@measure_latency
@measure_cost(provider="anthropic", model="claude-3-haiku-20240307")
def call_anthropic(prompt: str):
    # For Anthropic prompt caching metrics, initialize client with .beta
    # and use 'cache_control' in your messages.
    # Example: client = Anthropic(api_key="YOUR_KEY").beta
    client = Anthropic() # Standard client
    return client.messages.create(
        model="claude-3-haiku-20240307", # Ensure model matches decorator
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )

response_claude, metrics_claude = call_anthropic("Why is the sky blue?")
print(f"Cost: ${metrics_claude['cost_usd']:.6f} USD")
print(f"Latency: {metrics_claude['latency_ms']:.2f} ms")
print(f"Input Tokens: {metrics_claude['input_tokens']}")
print(f"Output Tokens: {metrics_claude['output_tokens']}")
# For Anthropic, 'cached_tokens' reflects 'cache_read_input_tokens'.
# 'cache_creation_input_tokens' will also be in metrics if caching beta is active.
print(f"Cached (read) tokens: {metrics_claude.get('cached_tokens', 0)}")
print(f"Cache creation tokens: {metrics_claude.get('cache_creation_input_tokens', 'N/A (requires caching beta)')}")```
```

## 🔍 Detailed Usage

### Cost Tracking

The `measure_cost` decorator requires explicit provider and model specification:

```python
@measure_cost(provider="openai", model="gpt-4o")  # Explicit specification required
def my_function(): ...

@measure_cost(provider="openai", model="gpt-4o", tier="flex")  # Optional tier
def my_function(): ...
```

### Latency Measurement

The `measure_latency` decorator works with both sync and async functions:

```python
@measure_latency
def sync_function(): ...

@measure_latency
async def async_function(): ...
```

### Combining Decorators

Decorators can be combined in any order:

```python
@measure_latency
@measure_cost(provider="openai", model="gpt-4o")
def my_function(): ...

# Equivalent to:
@measure_cost(provider="openai", model="gpt-4o")
@measure_latency
def my_function(): ...
```

### Async Usage

Both decorators work seamlessly with `async` functions:

```python
import asyncio
from tokenx.metrics import measure_cost, measure_latency
from openai import AsyncOpenAI # Use Async client

@measure_latency
@measure_cost(provider="openai", model="gpt-4o-mini")
async def call_openai_async():
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Tell me an async joke!"}]
    )
    return response

async def main():
    response, metrics = await call_openai_async()
    print(metrics)

# asyncio.run(main()) # Example of how to run it
```

---

## 🔄 Provider Compatibility

tokenx is designed to work with multiple LLM providers. Here's the current compatibility matrix:

| Provider | Status | SDK Version | Response Formats | Models | Cache Metrics Support |
|----------|--------|-------------|-----------------|--------|-----------------------|
| OpenAI | ✅ | >= 1.0.0 | Dict, Pydantic | All models (GPT-4, GPT-3.5, etc.) | ✅ (cached_tokens) |
| Anthropic | ✅ | >= 0.20.0 (approx for cache beta fields) | Dict, Pydantic-like | Claude models (Claude 3 Opus, Sonnet, Haiku) | ✅ (Prompt Caching Beta via cached_tokens & cache_creation_input_tokens) |
| Google | 🔜 | - | - | Gemini models (coming soon) | - |

### OpenAI Support Details

- **SDK Versions**: Compatible with OpenAI Python SDK v1.0.0 and newer.
- **Response Formats**: Supports dictionary responses from older SDK versions and Pydantic model responses from newer SDK versions, with cached token extraction from `prompt_tokens_details.cached_tokens`.
- **API Types**: Supports Chat Completions API and Traditional Completions API, with support for the newer Responses API coming soon.

### Anthropic Support Details

- **SDK Versions**: Compatible with Anthropic Python SDK (e.g., v0.20.0+ for full prompt caching beta fields).
- **Response Formats**: Supports Pydantic-like response objects.
- **Token Extraction**: Extracts input_tokens and output_tokens from the usage object.
- **Caching (Prompt Caching Beta)**: To utilize Anthropic's prompt caching and see related metrics, enable the beta feature in your client (`client = Anthropic().beta`) and define `cache_control` checkpoints in messages. tokenx maps `cache_read_input_tokens` to `cached_tokens` (for cost discounts if `cached_in` is defined) and includes `cache_creation_input_tokens` directly in metrics.
- **API Types**: Supports Messages API.

## 🛠️ Advanced Configuration

### Custom Pricing

Prices are loaded from the `model_prices.yaml` file. You can update this file when new models are released or prices change:

```yaml
openai:
  gpt-4o:
    sync:
      in: 2.50        # USD per million input tokens
      cached_in: 1.25 # USD per million cached tokens (OpenAI specific)
      out: 10.00      # USD per million output tokens
anthropic:
  claude-3-haiku-20240307:
    sync:
      in: 0.25        # USD per million input tokens
      # cached_in: 0.10 # Example: if Anthropic offered a specific rate for cache_read_input_tokens
      out: 1.25       # USD per million output tokens
  # If 'cached_in' is not specified for an Anthropic model,
  # all input_tokens (including cache_read_input_tokens) are billed at the 'in' rate.
```

### Error Handling

tokenx provides detailed error messages to help diagnose issues:

```python
from tokenx.errors import TokenExtractionError, PricingError

try:
    calculator = CostCalculator.for_provider("openai", "gpt-4o")
    cost = calculator.cost_from_response(response)
except TokenExtractionError as e:
    print(f"Token extraction failed: {e}")
except PricingError as e:
    print(f"Pricing error: {e}")
```

## 📊 Example Metrics Output

When you use the decorators, you'll get a structured metrics dictionary:

```python
{
    "provider": "openai",
    "model": "gpt-4o-mini",
    "tier": "sync",
    "input_tokens": 12,
    "output_tokens": 48,
    "cached_tokens": 20,        # New in v0.2.0
    "cost_usd": 0.000348,       # $0.000348 USD
    "latency_ms": 543.21        # 543.21 milliseconds
}
```

For Anthropic (when prompt caching beta is active and cache is utilized):

```python
{
    "provider": "anthropic",
    "model": "claude-3-haiku-20240307",
    "tier": "sync",
    "input_tokens": 25,                     # Total input tokens for the request
    "output_tokens": 60,
    "cached_tokens": 10,                    # Populated from Anthropic's 'cache_read_input_tokens'
    "cache_creation_input_tokens": 15,      # Anthropic's 'cache_creation_input_tokens'
    "cost_usd": 0.0000XX,
    "usd": 0.0000XX,
    "latency_ms": 750.21
}
```

If Anthropic's prompt caching beta is not used or no cache interaction occurs, cached_tokens will be 0 and cache_creation_input_tokens might be 0 or absent.

## 🤝 Contributing

```bash
git clone https://github.com/dvlshah/tokenx.git
pre-commit install
pip install -e .[dev]   # or `poetry install`
pytest -q && mypy src/
```

See [CONTRIBUTING.md](https://github.com/dvlshah/tokenx/blob/main/docs/CONTRIBUTING.md) for details.

---

## 📝 Changelog

See [CHANGELOG.md](https://github.com/dvlshah/tokenx/blob/main/docs/CHANGELOG.md) for full history.

---

## 📜 License

MIT © 2025 Deval Shah

<p align="center"><em>If tokenX saves you time or money, please consider <a href="https://github.com/sponsors/dvlshah">sponsoring</a> or giving a ⭐ – it really helps!</em></p>
