# Quickstart

## Install

```bash
pip install agent-lightning
```

## Step 1 — Trace your agent

```python
import agent_lightning as al

store  = al.LightningStore()
tracer = al.AsyncTracer(agent_id="my_agent", store=store)

with tracer.run() as run:
    tracer.log_prompt("What is Python?")
    answer = my_llm("What is Python?")
    tracer.log_response(answer)
    run.add_reward(score(answer))
```

## Step 2 — View runs

```bash
agl dashboard
# → http://localhost:7860
```

## Step 3 — Optimize

```bash
export OPENROUTER_API_KEY=sk-or-...
agl train --agent-id my_agent --model qwen/qwen3-8b:free --rounds 3
```

## Step 4 — Use the better prompt

```python
prompt = tracer.get_optimized_prompt("You are a helpful assistant.")
```

## OpenAI 1-line integration

```python
from openai import OpenAI
import agent_lightning as al

client = al.wrap_openai(OpenAI(), agent_id="my_agent")
# use exactly as before
```

## Full autonomous loop

See [how-to/autonomous-loop.md](how-to/autonomous-loop.md).
