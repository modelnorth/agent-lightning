"""Ollama backend — local models, zero API cost. Perfect for air-gapped deployments."""
from __future__ import annotations
import time
from typing import List
from .base import LLMBackend, LLMMessage, LLMResponse


class OllamaBackend(LLMBackend):
    """
    Ollama backend for local models.

    Example (Qwen3-8B on your ModelNorth stack):
        backend = OllamaBackend(model="qwen3:8b", base_url="http://localhost:11434")
    """
    def __init__(self, model: str = "qwen3:8b", base_url: str = "http://localhost:11434"):
        self.model    = model
        self.base_url = base_url.rstrip("/")

    async def complete(self, messages: List[LLMMessage], temperature=0.7, max_tokens=2048, **kw) -> LLMResponse:
        try:
            import aiohttp
        except ImportError:
            raise ImportError("pip install aiohttp")
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        t0 = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/api/chat", json=payload) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Ollama HTTP {resp.status}: {await resp.text()}")
                data = await resp.json()
        content = data.get("message", {}).get("content", "")
        return LLMResponse(
            content=content, model=self.model,
            latency_ms=(time.time()-t0)*1000, raw=data,
        )
