"""LLMBackend ABC — model-agnostic LLM interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LLMMessage:
    role: str
    content: str
    def to_dict(self): return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    content: str
    model: str = ""
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    raw: Any = None

    @property
    def total_tokens(self): return (self.input_tokens or 0) + (self.output_tokens or 0)


class LLMBackend(ABC):
    """Model-agnostic LLM interface. All trainers/algorithms use this."""

    @abstractmethod
    async def complete(self, messages: List[LLMMessage], temperature: float = 0.7,
                       max_tokens: int = 2048, **kwargs) -> LLMResponse: ...

    async def complete_text(self, prompt: str, system: str = None,
                            temperature: float = 0.7, max_tokens: int = 2048) -> str:
        msgs = []
        if system: msgs.append(LLMMessage("system", system))
        msgs.append(LLMMessage("user", prompt))
        return (await self.complete(msgs, temperature=temperature, max_tokens=max_tokens)).content

    @property
    def model_name(self) -> str: return getattr(self, "model", "unknown")
