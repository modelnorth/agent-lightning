# ⚡ Agent Lightning v0.2

**Drop-in AI agent optimization toolkit. Better than Microsoft's.**

- **100+ models** via OpenRouter — not hardcoded to OpenAI
- **APO beam search** with textual gradients (same algo, model-agnostic)
- **Multi-model consensus** — novel feature, not in Microsoft's version
- **Autonomous orchestrator** — fires, trains, A/B tests, deploys automatically
- **OTel spans** — plug into Jaeger, Datadog, Grafana Tempo
- **Zero deps core** — SQLite only, works offline with Ollama/Qwen3

```bash
pip install agent-lightning
pip install "agent-lightning[openrouter]"   # + aiohttp for OpenRouter
pip install "agent-lightning[all]"           # everything
```

---

## 60-second quickstart

```python
import agent_lightning as al

store  = al.LightningStore()
tracer = al.AsyncTracer(agent_id="my_agent", store=store)

# Record runs exactly as before
with tracer.run() as run:
    tracer.log_prompt("What is Python?")
    answer = my_llm("What is Python?")
    tracer.log_response(answer)
    run.add_reward(score(answer))

# Get optimized prompt back (set by orchestrator automatically)
prompt = tracer.get_optimized_prompt(fallback="You are a helpful assistant.")
```

## OpenAI — 1-line wrap (v0.1 compat)

```python
client = al.wrap_openai(OpenAI(), agent_id="my_agent")
# use exactly as before — all calls traced automatically
```

---

## The Autonomous Loop

```python
from agent_lightning import *

store  = LightningStore()
tracer = AsyncTracer(agent_id="my_agent", store=store)

# 1. Setup APO with OpenRouter (cost arbitrage)
apo = APO(
    store=store, agent_id="my_agent",
    gradient_backend=OpenRouterBackend.fast(api_key="sk-or-..."),   # cheap: Qwen3-8B free
    eval_backend=OpenRouterBackend.strong(api_key="sk-or-..."),     # strong: Claude/GPT-4o
    initial_prompt="You are a helpful assistant.",
    beam_width=4, beam_rounds=3,
)

# 2. Orchestrator fires automatically every 50 runs
orchestrator = Orchestrator(
    store=store, algorithm=apo,
    config=OrchestratorConfig(
        agent_id="my_agent",
        trigger=EveryNRuns(50),
        auto_deploy=True,
    ),
    on_improvement=lambda p, score: print(f"Better prompt deployed! score={score:.3f}"),
)
orchestrator.start()   # background thread, non-blocking

# 3. Your agent runs unchanged
while True:
    prompt = tracer.get_optimized_prompt("You are a helpful assistant.")
    with tracer.run() as run:
        tracer.log_prompt(user_input, role="user")
        response = call_llm(user_input, system=prompt)
        tracer.log_response(response)
        run.add_reward(evaluate(response))
```

---

## OpenRouter — 100+ Models

```python
# Free tier for bulk gradient computation
gradient = OpenRouterBackend(api_key="sk-or-...", model="qwen/qwen3-8b:free")

# Strong model for final eval
eval_b   = OpenRouterBackend(api_key="sk-or-...", model="anthropic/claude-3-haiku")

# Or use tier factories
fast     = OpenRouterBackend.fast(api_key="sk-or-...")       # cheapest
balanced = OpenRouterBackend.balanced(api_key="sk-or-...")   # mid
strong   = OpenRouterBackend.strong(api_key="sk-or-...")     # best quality
```

**Available tiers:**
| Tier | Models |
|------|--------|
| fast | `qwen/qwen3-8b:free`, `llama-3.1-8b:free`, `gemma-3-4b:free`, `mistral-7b:free` |
| balanced | `claude-3-haiku`, `gemini-flash-1.5`, `qwen-2.5-72b`, `mistral-nemo` |
| strong | `claude-sonnet-4`, `gemini-2.5-pro`, `gpt-4o`, `deepseek-r1` |

## Multi-Model Consensus (novel vs Microsoft)

```python
consensus = MultiModelConsensus(
    backends=[
        OpenRouterBackend(key, model="qwen/qwen3-8b:free"),
        OpenRouterBackend(key, model="meta-llama/llama-3.1-8b-instruct:free"),
        OpenRouterBackend(key, model="mistralai/mistral-7b-instruct:free"),
    ],
    judge=OpenRouterBackend(key, model="anthropic/claude-3-haiku"),
)
best_prompt = await consensus.optimize_prompt(current, good_examples, bad_examples)
```

## Offline / Air-Gapped (Ollama)

```python
backend = OllamaBackend(model="qwen3:8b")  # local, no internet
apo = APO(store=store, agent_id="x", gradient_backend=backend)
```

## APO — Automatic Prompt Optimization

Textual gradients + beam search. Beats single-shot reflection.

```python
apo = APO(
    store=store, agent_id="my_agent",
    gradient_backend=OpenRouterBackend.fast(api_key="..."),
    eval_backend=OpenRouterBackend.strong(api_key="..."),
    initial_prompt="You are a helpful assistant.",
    beam_width=4,       # keep top 4 candidates each round
    branch_factor=3,    # 3 gradient edits per parent
    beam_rounds=3,      # 3 optimization rounds
    reward_fn=my_scorer,  # optional: your own eval function
)
result = await apo.run(train_dataset=train, val_dataset=val)
print(result["best_prompt"])
print(result["best_score"])
print(result["history"])   # full optimization trace
```

## Async Rollouts + Hooks

```python
class MyHook(al.Hook):
    async def on_rollout_start(self, rollout, **kw):
        print(f"Starting rollout {rollout.rollout_id}")
    async def on_rollout_end(self, rollout, attempt, status, **kw):
        print(f"Done: reward={attempt.reward}")

tracer = AsyncTracer(agent_id="x", store=store, hooks=[MyHook()])

async with tracer.rollout(task=my_task) as (rollout, attempt):
    tracer.log_prompt("input")
    tracer.log_response("output")
    attempt.reward = 0.9
```

## OTel Compatibility

```python
from agent_lightning.otel import OTelExporter, OTelSpanAdapter

exporter = OTelExporter(endpoint="http://jaeger:4318/v1/traces")
await exporter.export(spans)          # send to Jaeger/Datadog/Tempo

# Or export to file
exporter.export_to_file(spans, "traces.jsonl")
```

## Triggers

```python
EveryNRuns(n=50)                       # after every N completed runs
Scheduled(interval_seconds=3600)       # every hour
Manual()                               # only when .trigger() called
OnImprovement(min_avg_reward=0.6)      # when quality drops below threshold
```

## CLI

```bash
agl dashboard                          # web UI at localhost:7860
agl stats my_agent                     # run statistics
agl rollouts my_agent                  # rollout queue status
agl prompt my_agent                    # show current optimized prompt
agl export my_agent -o data.jsonl      # export runs for fine-tuning
agl models                             # list OpenRouter model tiers
agl train --agent-id my_agent \
          --api-key sk-or-... \
          --model qwen/qwen3-8b:free \
          --rounds 3
```

---

## How We Beat Microsoft

| Feature | Microsoft Agent Lightning | Ours v0.2 |
|---------|--------------------------|-----------|
| LLM backend | Hardcoded `AsyncOpenAI` | **OpenRouter = 100+ models** |
| Cost arbitrage | ❌ | ✅ cheap gradient, strong eval |
| Multi-model consensus | ❌ | ✅ novel feature |
| Offline / air-gapped | ❌ | ✅ via OllamaBackend |
| APO algorithm | ✅ OpenAI only | ✅ **any backend** |
| Beam search | ✅ | ✅ |
| Textual gradients | ✅ | ✅ |
| Autonomous orchestrator | ✅ | ✅ |
| OTel spans | ✅ | ✅ |
| Rollout queue | ✅ complex | ✅ simpler, SQLite |
| Async tracer | ✅ | ✅ |
| Hooks | ✅ | ✅ |
| A/B eval + t-test | ❌ | ✅ |
| GPU RL (VERL) | ✅ | ❌ (roadmap) |
| Install complexity | Docker + vLLM + FSDP | `pip install` |

---

## Examples

| File | What it shows |
|------|---------------|
| `examples/quickstart.py` | v0.1 minimal tracing |
| `examples/v2_quickstart.py` | v0.2 full autonomous loop |
| `examples/v2_openrouter_apo.py` | APO with cost arbitrage |
| `examples/v2_orchestrator.py` | Background auto-optimization |
| `examples/v2_multi_model_consensus.py` | 3-model consensus |
| `examples/v2_ollama_offline.py` | Offline with Ollama/Qwen3 |
| `examples/openai_agent.py` | 1-line OpenAI wrap |
| `examples/tool_agent.py` | Tool call tracing |
| `examples/multi_agent.py` | Multi-agent pipeline |

## Tests

```bash
cd agent_lightning
python -m pytest tests/ -v     # requires pytest
# or run directly:
python -c "import sys; sys.path.insert(0,'.'); exec(open('tests/test_models.py').read())"
```

---

## License

MIT — use it, fork it, ship it.
