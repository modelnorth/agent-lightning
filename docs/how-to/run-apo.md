# Run APO (Automatic Prompt Optimization)

APO uses textual gradients + beam search to find better prompts.

## Minimal example

```python
apo = al.APO(
    store=store,
    agent_id="my_agent",
    gradient_backend=al.OpenRouterBackend.fast(api_key="sk-or-..."),
    initial_prompt="You are a helpful assistant.",
)
result = await apo.run(train_dataset=my_tasks)
print(result["best_prompt"])
```

## With your own reward function

```python
def my_reward(response: str, task_input: str) -> float:
    return 1.0 if "correct keyword" in response.lower() else 0.2

apo = al.APO(
    ...
    reward_fn=my_reward,
    eval_backend=al.OpenRouterBackend.balanced(...),
)
result = await apo.run(train_dataset=train, val_dataset=val)
```

## Tuning beam search

```python
apo = al.APO(
    beam_width=4,       # candidates kept each round (default 4)
    branch_factor=3,    # edits per parent (default 3)
    beam_rounds=3,      # optimization rounds (default 3)
)
```

## Cost vs quality tradeoff

| Setting | Cost | Quality |
|---------|------|---------|
| `beam_width=2, rounds=1` | Low | Baseline |
| `beam_width=4, rounds=3` | Medium | Good |
| `beam_width=6, rounds=5` | High | Best |

## From CLI

```bash
agl train --agent-id my_agent \
          --model qwen/qwen3-8b:free \
          --beam-width 4 \
          --rounds 3
```
