# Algorithms

## APO — Automatic Prompt Optimization

APO uses **textual gradients + beam search** to iteratively improve prompts.

### How it works

```
Initial prompt (seed)
        │
        ▼ Round 1
┌──────────────────────────────────────┐
│  For each parent in beam:            │
│    1. Sample K failed rollouts       │
│    2. Ask gradient model:            │
│       "What's wrong with this        │
│        prompt given these outputs?"  │
│    3. Ask edit model:                │
│       "Fix these weaknesses"         │
│    4. Get branch_factor candidates   │
└──────────────────────────────────────┘
        │
        ▼ Evaluate all candidates on val set
        │
        ▼ Keep top beam_width by score
        │
        ▼ Round 2, 3, ...
        │
        ▼ Best prompt → store
```

### Model-agnostic advantage

Microsoft's APO hardcodes `AsyncOpenAI`. Ours accepts any `LLMBackend`:

```python
# OpenRouter (100+ models)
apo = APO(gradient_backend=OpenRouterBackend.fast(...))

# Local Ollama (offline)
apo = APO(gradient_backend=OllamaBackend(model="qwen3:8b"))

# OpenAI
apo = APO(gradient_backend=OpenAIBackend(model="gpt-4o-mini"))
```

### Cost arbitrage

```python
apo = APO(
    gradient_backend=OpenRouterBackend.fast(...),   # $0 (free tier) for critiques
    eval_backend=OpenRouterBackend.strong(...),     # strong model for scoring only
)
```

### Beam search parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `beam_width` | 4 | Candidates kept each round |
| `branch_factor` | 3 | New edits per parent |
| `beam_rounds` | 3 | Total optimization rounds |
| `diversity_temperature` | 0.9 | Variety in generated edits |

Total LLM calls ≈ `beam_width × branch_factor × beam_rounds × 2`

## BestOfN (contrib)

Simpler than APO. Takes top-K prompts by reward, asks LLM to synthesize a better one.
Good when you have many existing runs to learn from.

See `examples/custom_algorithm/` and `contrib/recipes/`.

## Writing your own

See [how-to/custom-algorithm.md](../how-to/custom-algorithm.md).
