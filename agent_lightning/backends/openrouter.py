"""
OpenRouterBackend — 100+ models via one API key.

The killer feature: model arbitrage, fallback chains, multi-model consensus.

Models: https://openrouter.ai/models
"""
from __future__ import annotations
import asyncio, os, time, logging
from typing import Dict, List, Optional, Any
from .base import LLMBackend, LLMMessage, LLMResponse

log = logging.getLogger(__name__)

# Curated model tiers for cost arbitrage
OPENROUTER_MODELS = {
    # Free / ultra-cheap — use for bulk gradient computation
    "fast": [
        "qwen/qwen3-8b:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "google/gemma-3-4b-it:free",
        "mistralai/mistral-7b-instruct:free",
    ],
    # Mid-tier — good balance
    "balanced": [
        "anthropic/claude-3-haiku",
        "google/gemini-flash-1.5",
        "qwen/qwen-2.5-72b-instruct",
        "mistralai/mistral-nemo",
    ],
    # Strong — use for final eval / consensus judge
    "strong": [
        "anthropic/claude-sonnet-4",
        "google/gemini-2.5-pro",
        "openai/gpt-4o",
        "deepseek/deepseek-r1",
    ],
}


class OpenRouterBackend(LLMBackend):
    """
    Backend powered by OpenRouter — access 100+ models with one API key.

    Features:
    - Model arbitrage (use cheap model for bulk, strong for eval)
    - Automatic fallback: if primary model fails, try next in fallback_models
    - Usage tracking across models
    - Multi-model consensus (see MultiModelConsensus)

    Example:
        backend = OpenRouterBackend(
            api_key="sk-or-...",
            model="qwen/qwen3-8b:free",          # cheap for bulk ops
            fallback_models=["mistralai/mistral-7b-instruct:free"],
        )
        response = await backend.complete_text("Improve this prompt: ...")
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str = None,
        model: str = "qwen/qwen3-8b:free",
        fallback_models: List[str] = None,
        site_url: str = "https://github.com/modelNorth/agent-lightning",
        site_name: str = "Agent Lightning",
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.api_key        = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model          = model
        self.fallback_models= fallback_models or []
        self.site_url       = site_url
        self.site_name      = site_name
        self.timeout        = timeout
        self.max_retries    = max_retries
        self._usage: Dict[str, Dict] = {}

    async def complete(self, messages: List[LLMMessage], temperature: float = 0.7,
                       max_tokens: int = 2048, model: str = None, **kwargs) -> LLMResponse:
        """Call OpenRouter. Falls back through fallback_models on failure."""
        import aiohttp
        models_to_try = [model or self.model] + self.fallback_models
        last_error = None

        for m in models_to_try:
            for attempt in range(self.max_retries):
                try:
                    t0 = time.time()
                    resp = await self._call(m, messages, temperature, max_tokens, **kwargs)
                    latency = (time.time() - t0) * 1000
                    self._track_usage(m, resp)
                    resp.latency_ms = latency
                    return resp
                except Exception as e:
                    last_error = e
                    log.warning("OpenRouter %s attempt %d failed: %s", m, attempt+1, e)
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"All OpenRouter models failed. Last error: {last_error}")

    async def _call(self, model: str, messages: List[LLMMessage],
                    temperature: float, max_tokens: int, **kwargs) -> LLMResponse:
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp required: pip install aiohttp")

        payload = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.BASE_URL}/chat/completions",
                json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"OpenRouter HTTP {resp.status}: {text[:200]}")
                data = await resp.json()

        content = data["choices"][0]["message"]["content"] or ""
        usage   = data.get("usage", {})
        return LLMResponse(
            content=content, model=model,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            raw=data,
        )

    def _track_usage(self, model: str, resp: LLMResponse):
        if model not in self._usage:
            self._usage[model] = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self._usage[model]["calls"] += 1
        self._usage[model]["input_tokens"]  += resp.input_tokens  or 0
        self._usage[model]["output_tokens"] += resp.output_tokens or 0

    @property
    def usage_stats(self) -> Dict[str, Dict]:
        return dict(self._usage)

    @classmethod
    def fast(cls, api_key: str = None, **kwargs) -> "OpenRouterBackend":
        """Factory: cheap model for bulk gradient computation."""
        return cls(api_key=api_key, model=OPENROUTER_MODELS["fast"][0],
                   fallback_models=OPENROUTER_MODELS["fast"][1:], **kwargs)

    @classmethod
    def balanced(cls, api_key: str = None, **kwargs) -> "OpenRouterBackend":
        return cls(api_key=api_key, model=OPENROUTER_MODELS["balanced"][0],
                   fallback_models=OPENROUTER_MODELS["balanced"][1:], **kwargs)

    @classmethod
    def strong(cls, api_key: str = None, **kwargs) -> "OpenRouterBackend":
        """Factory: strong model for final evaluation."""
        return cls(api_key=api_key, model=OPENROUTER_MODELS["strong"][0],
                   fallback_models=OPENROUTER_MODELS["strong"][1:], **kwargs)


class MultiModelConsensus:
    """
    Run the same optimization through N models, take intersection/best.
    Novel vs Microsoft: they single-model. We multi-model.

    Example:
        consensus = MultiModelConsensus(
            backends=[
                OpenRouterBackend(model="anthropic/claude-3-haiku"),
                OpenRouterBackend(model="qwen/qwen-2.5-72b-instruct"),
                OpenRouterBackend(model="google/gemini-flash-1.5"),
            ]
        )
        best_prompt = await consensus.optimize_prompt(current, good_examples, bad_examples)
    """

    def __init__(self, backends: List[LLMBackend], judge: LLMBackend = None):
        self.backends = backends
        self.judge    = judge or backends[0]

    async def complete_all(self, messages: List[LLMMessage], **kwargs) -> List[LLMResponse]:
        """Call all backends in parallel."""
        tasks = [b.complete(messages, **kwargs) for b in self.backends]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, LLMResponse)]

    async def best_of(self, prompt: str, system: str = None,
                      score_fn=None, **kwargs) -> str:
        """
        Generate N responses, score each, return best.
        Falls back to longest response if no score_fn.
        """
        msgs = []
        if system: msgs.append(LLMMessage("system", system))
        msgs.append(LLMMessage("user", prompt))
        responses = await self.complete_all(msgs, **kwargs)
        if not responses:
            return ""
        if score_fn:
            scored = [(score_fn(r.content), r.content) for r in responses]
            return max(scored, key=lambda x: x[0])[1]
        # Default: longest meaningful response
        return max(responses, key=lambda r: len(r.content)).content

    async def optimize_prompt(self, current_prompt: str, good_examples: List[str],
                               bad_examples: List[str]) -> str:
        """
        Run APO-style prompt optimization across all models, then judge the best candidate.
        """
        from ..algorithms.apo import _OPTIMIZER_SYSTEM
        good_text = "\n---\n".join(good_examples[:5]) or "None"
        bad_text  = "\n---\n".join(bad_examples[:3])  or "None"

        user_msg = f"""Current prompt:\n```\n{current_prompt}\n```

HIGH REWARD examples:\n{good_text}

LOW REWARD examples:\n{bad_text}

Write an improved system prompt:"""

        # All models propose a candidate
        candidates = await self.best_of(
            prompt=user_msg, system=_OPTIMIZER_SYSTEM, temperature=0.8,
        )

        return candidates
