# Autonomous Training Loop

The Orchestrator runs APO in the background — your agent improves automatically.

## Setup

```python
import agent_lightning as al

store = al.LightningStore()

apo = al.APO(
    store=store, agent_id="my_agent",
    gradient_backend=al.OpenRouterBackend.fast(api_key="sk-or-..."),
    initial_prompt="You are a helpful assistant.",
)

orchestrator = al.Orchestrator(
    store=store, algorithm=apo,
    config=al.OrchestratorConfig(
        agent_id="my_agent",
        trigger=al.EveryNRuns(50),     # fire every 50 completed runs
        auto_deploy=True,               # deploy if better
        min_runs_to_train=10,           # wait for at least 10 runs
    ),
    on_improvement=lambda p, score: print(f"Deployed! score={score:.3f}"),
)
orchestrator.start()   # background thread, non-blocking
```

## Triggers

```python
al.EveryNRuns(n=50)                    # after N runs
al.Scheduled(interval_seconds=3600)    # every hour
al.Manual()                            # call trigger.trigger() manually
al.OnImprovement(min_avg_reward=0.6)   # when quality drops below threshold
```

## With A/B evaluation before deploy

```python
evaluator = al.Evaluator(
    backend=al.OpenRouterBackend.balanced(...),
    reward_fn=my_reward_fn,
    n_samples=20,
)
orchestrator = al.Orchestrator(
    ...,
    evaluator=evaluator,
    config=al.OrchestratorConfig(require_eval=True),
)
```

## Agent side — zero changes needed

```python
with tracer.run() as run:
    # get_optimized_prompt returns the latest deployed prompt automatically
    prompt = tracer.get_optimized_prompt("You are a helpful assistant.")
    response = call_llm(user_input, system=prompt)
    run.add_reward(score(response))
```
