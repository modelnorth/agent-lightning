# Multi-Agent Workflows

Trace individual agents inside a larger pipeline. Each agent gets its own
`AsyncTracer` with a distinct `agent_id`. They share one `LightningStore`.

## Setup

```python
import agent_lightning as al

shared_store = al.LightningStore()

planner    = al.AsyncTracer(agent_id="planner",    store=shared_store)
researcher = al.AsyncTracer(agent_id="researcher", store=shared_store)
writer     = al.AsyncTracer(agent_id="writer",     store=shared_store)
```

## Trace each agent independently

```python
async def run_pipeline(query: str):
    # Planner
    with planner.run() as run:
        planner.log_prompt(query)
        plan = await my_planner_llm(query)
        planner.log_response(plan)
        run.add_reward(quality_score(plan))

    # Researcher
    for step in parse_plan(plan):
        with researcher.run() as run:
            researcher.log_prompt(step)
            researcher.log_tool_call("web_search", {"query": step})
            results = await search(step)
            researcher.log_tool_result("web_search", results)
            researcher.log_response(str(results))
            run.add_reward(relevance_score(results, step))

    # Writer
    with writer.run() as run:
        writer.log_prompt(str(results))
        final = await my_writer_llm(results)
        writer.log_response(final)
        run.add_reward(quality_score(final))

    return final
```

## Optimize only one agent

```python
# Only optimize the writer — leave planner and researcher alone
apo = al.APO(
    store=shared_store,
    agent_id="writer",          # targets only writer runs
    gradient_backend=al.OpenRouterBackend.fast(api_key="sk-or-..."),
    initial_prompt="You are a technical writer.",
)
result = await apo.run()
```

## Per-agent stats

```python
for agent_id in ["planner", "researcher", "writer"]:
    stats = shared_store.stats(agent_id=agent_id)
    print(f"{agent_id}: {stats['total']} runs, avg_reward={stats['avg_reward']:.3f}")
```

## Async rollout pattern

```python
async with planner.rollout(task={"query": query}) as (rollout, attempt):
    planner.log_prompt(query)
    plan = await my_planner_llm(query)
    planner.log_response(plan)
    attempt.reward = quality_score(plan)
```
