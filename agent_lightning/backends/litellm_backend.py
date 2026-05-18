"""LiteLLM backend — catch-all for any provider."""
from __future__ import annotations
import time
from typing import List
from .base import LLMBackend, LLMMessage, LLMResponse


class LiteLLMBackend(LLMBackend):
    """
    LiteLLM catch-all backend. Supports 100+ providers via unified interface.
    Use when you need a provider not covered by other backends.

    Example:
        backend = LiteLLMBackend(model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0")
    """
    def __init__(self, model: str, **kwargs):
        self.model  = model
        self.kwargs = kwargs

    async def complete(self, messages: List[LLMMessage], temperature=0.7, max_tokens=2048, **kw) -> LLMResponse:
        try:
            import litellm
        except ImportError:
            raise ImportError("pip install litellm")
        t0 = time.time()
        resp = await litellm.acompletion(
            model=self.model,
            messages=[m.to_dict() for m in messages],
            temperature=temperature, max_tokens=max_tokens,
            **{**self.kwargs, **kw},
        )
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            model=self.model,
            input_tokens=getattr(resp.usage, "prompt_tokens", None),
            output_tokens=getattr(resp.usage, "completion_tokens", None),
            latency_ms=(time.time()-t0)*1000, raw=resp,
        )
