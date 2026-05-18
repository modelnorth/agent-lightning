# ⚡ Agent Lightning

**Drop-in AI agent optimization toolkit. Plug in, record runs, train with 100+ models, ship better prompts automatically.**

```bash
pip install agent-lightning
pip install "agent-lightning[openrouter]"   # + OpenRouter support
```

## Why Agent Lightning?

| | Microsoft Agent Lightning | **Ours** |
|---|---|---|
| LLM backend | Hardcoded OpenAI | **OpenRouter = 100+ models** |
| Cost arbitrage | ❌ | ✅ cheap gradient, strong eval |
| Multi-model consensus | ❌ | ✅ novel feature |
| Offline / Ollama | ❌ | ✅ air-gapped deployments |
| Install | Docker + vLLM + FSDP | `pip install` |
| A/B eval + t-test | ❌ | ✅ |

## 30-second example

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

# Autonomous optimization in background
apo = al.APO(store=store, agent_id="my_agent",
             gradient_backend=al.OpenRouterBackend.fast(api_key="sk-or-..."))
orchestrator = al.Orchestrator(store, apo,
    config=al.OrchestratorConfig(trigger=al.EveryNRuns(50)))
orchestrator.start()  # non-blocking
```

→ [Quickstart](quickstart.md) to get running in 5 minutes.
