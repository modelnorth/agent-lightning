# Connect OpenAI

## 1-line wrap

```python
from openai import OpenAI
import agent_lightning as al

client = OpenAI(api_key="sk-...")
client = al.wrap_openai(client, agent_id="my_agent")

# Use exactly as before — all calls traced
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Manual tracing

```python
tracer = al.AsyncTracer(agent_id="my_agent")

with tracer.run() as run:
    tracer.log_prompt("Hello!", model="gpt-4o")
    response = client.chat.completions.create(...)
    tracer.log_response(response.choices[0].message.content,
                        model="gpt-4o",
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens)
    run.add_reward(my_scorer(response))
```

## Use OpenAI as training backend

```python
backend = al.OpenAIBackend(api_key="sk-...", model="gpt-4o-mini")
apo = al.APO(store=store, agent_id="x", gradient_backend=backend)
```
