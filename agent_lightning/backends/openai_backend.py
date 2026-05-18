"""OpenAI backend."""
from __future__ import annotations
import os, time
from typing import List
from .base import LLMBackend, LLMMessage, LLMResponse


class OpenAIBackend(LLMBackend):
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini", base_url: str = None):
        self.api_key  = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model    = model
        self.base_url = base_url

    async def complete(self, messages: List[LLMMessage], temperature=0.7, max_tokens=2048, **kw) -> LLMResponse:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("pip install openai")
        kw2 = {"api_key": self.api_key}
        if self.base_url: kw2["base_url"] = self.base_url
        client = AsyncOpenAI(**kw2)
        t0 = time.time()
        resp = await client.chat.completions.create(
            model=self.model, messages=[m.to_dict() for m in messages],
            temperature=temperature, max_tokens=max_tokens,
        )
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            model=self.model,
            input_tokens=resp.usage.prompt_tokens if resp.usage else None,
            output_tokens=resp.usage.completion_tokens if resp.usage else None,
            latency_ms=(time.time()-t0)*1000, raw=resp,
        )
