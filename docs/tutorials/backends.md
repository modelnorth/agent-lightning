# Backends

All trainers and algorithms use the `LLMBackend` ABC. Swap providers without
changing any training code.

## LLMBackend interface

```python
class LLMBackend(ABC):
    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse: ...

    # Convenience wrapper
    async def complete_text(self, prompt: str, system: str = None) -> str: ...
```

## OpenRouterBackend ⭐ recommended

```python
backend = al.OpenRouterBackend(
    api_key="sk-or-...",               # or OPENROUTER_API_KEY env var
    model="qwen/qwen3-8b:free",        # default: free tier
    fallback_models=["mistralai/mistral-7b-instruct:free"],
)
```

**Tier factories:**
```python
al.OpenRouterBackend.fast(...)      # cheapest / free tier
al.OpenRouterBackend.balanced(...)  # quality/cost balance
al.OpenRouterBackend.strong(...)    # best quality
```

## OllamaBackend — offline

```python
backend = al.OllamaBackend(
    model="qwen3:8b",
    base_url="http://localhost:11434",
)
```

Requires: `ollama serve && ollama pull qwen3:8b`

## OpenAIBackend

```python
backend = al.OpenAIBackend(
    api_key="sk-...",
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1",  # or any compatible endpoint
)
```

## LiteLLMBackend — catch-all

```python
backend = al.LiteLLMBackend(
    model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
)
```

Supports 100+ providers via LiteLLM. Install: `pip install litellm`

## MultiModelConsensus

```python
consensus = al.MultiModelConsensus(
    backends=[backend_a, backend_b, backend_c],
    judge=strong_backend,  # picks the best candidate
)
best = await consensus.best_of(prompt, score_fn=len)
```

## Custom backend

```python
class MyBackend(al.LLMBackend):
    async def complete(self, messages, temperature=0.7, max_tokens=2048, **kw):
        # call your API
        return al.LLMResponse(
            content="response text",
            model="my-model",
            input_tokens=10,
            output_tokens=20,
        )
```
