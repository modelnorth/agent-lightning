# Custom Algorithm

Shows how to implement your own training algorithm by subclassing `BaseAlgorithm`.

This example implements **BestOfN Synthesis**: takes the top-K highest-reward prompts
and asks an LLM to synthesize a better one combining their strengths.

## Run
```bash
python custom_algorithm.py
```

## How to write your own
```python
from agent_lightning.algorithms.base_algorithm import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    async def run(self, train_dataset=None, val_dataset=None):
        runs = self.store.list_runs(agent_id=self.agent_id)
        # ... your logic ...
        return {"success": True, "best_prompt": "...", "best_score": 0.9}

# Plug into orchestrator
orchestrator = al.Orchestrator(store=store, algorithm=MyAlgorithm(store, "agent1"))
orchestrator.start()
```
