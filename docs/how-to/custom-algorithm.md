# Write a Custom Algorithm

Subclass `BaseAlgorithm` to implement any training logic.

## Minimal example

```python
from agent_lightning.algorithms.base_algorithm import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    def __init__(self, store, agent_id="default", backend=None):
        super().__init__(store=store)
        self.agent_id = agent_id
        self.backend  = backend

    async def run(self, train_dataset=None, val_dataset=None):
        # 1. Load runs
        runs = self.store.list_runs(
            agent_id=self.agent_id, status="completed", limit=100
        )
        if not runs:
            return {"success": False, "message": "no runs"}

        # 2. Your logic — e.g. pick the best prompt
        best_run   = max(runs, key=lambda r: r.total_reward)
        best_prompt = best_run.get_prompts()[0] if best_run.get_prompts() else ""

        # 3. Write back to store
        self.store.save_optimized_prompt(
            self.agent_id, best_prompt, trainer="MyAlgorithm"
        )

        return {
            "success":     True,
            "best_prompt": best_prompt,
            "best_score":  best_run.total_reward,
        }
```

## Plug into Orchestrator

```python
algo = MyAlgorithm(store=store, agent_id="my_agent", backend=backend)

orchestrator = al.Orchestrator(
    store=store, algorithm=algo,
    config=al.OrchestratorConfig(
        agent_id="my_agent",
        trigger=al.EveryNRuns(20),
    ),
)
orchestrator.start()
```

## Full example

See `examples/custom_algorithm/custom_algorithm.py` for the BestOfN recipe.
