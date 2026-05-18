# Use OpenRouter (100+ Models)

OpenRouter gives you access to 100+ models through a single API key.
Get one at [openrouter.ai](https://openrouter.ai).

## Basic usage

```python
backend = al.OpenRouterBackend(
    api_key="sk-or-...",          # or set OPENROUTER_API_KEY env var
    model="qwen/qwen3-8b:free",   # free tier
)
```

## Model tiers

```python
# Cheap — for bulk gradient computation
fast = al.OpenRouterBackend.fast(api_key="sk-or-...")

# Balanced — good quality/cost ratio  
balanced = al.OpenRouterBackend.balanced(api_key="sk-or-...")

# Strong — for final evaluation
strong = al.OpenRouterBackend.strong(api_key="sk-or-...")
```

## Cost arbitrage

```python
apo = al.APO(
    gradient_backend=al.OpenRouterBackend.fast(...),   # cheap for critiques
    eval_backend=al.OpenRouterBackend.strong(...),     # strong for scoring
)
```

## Fallback chains

```python
backend = al.OpenRouterBackend(
    model="anthropic/claude-3-haiku",
    fallback_models=["google/gemini-flash-1.5", "qwen/qwen3-8b:free"],
)
# If Claude fails, tries Gemini, then Qwen
```

## Multi-model consensus

```python
consensus = al.MultiModelConsensus(
    backends=[
        al.OpenRouterBackend(key, "qwen/qwen3-8b:free"),
        al.OpenRouterBackend(key, "mistralai/mistral-7b-instruct:free"),
        al.OpenRouterBackend(key, "meta-llama/llama-3.1-8b-instruct:free"),
    ],
    judge=al.OpenRouterBackend(key, "anthropic/claude-3-haiku"),
)
best = await consensus.best_of("Improve this prompt: ...", score_fn=len)
```

## Available free models

```bash
agl models
```
