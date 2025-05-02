# 🧮 llm-meter

> Plug-and-play decorators for tracking **USD cost** + **latency** of LLM calls.

```bash
pip install llm-meter
```

## Quick demo

```python
from llm_meter import measure_cost, measure_latency
import openai

@measure_latency
@measure_cost("gpt-4.1-mini")
def ask(q):
    return openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": q}],
    )

_, m = ask("Hi!")
print(m)  # {'cost_usd': 0.000018, 'latency_ms': 742.1}
```

MIT © 2025 YOUR_NAME
