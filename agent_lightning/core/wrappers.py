"""
Drop-in wrappers that instrument existing agent setups with minimal code changes.
"""

from __future__ import annotations

import time
import functools
from typing import Any, Callable, Dict, List, Optional

from .models import Step, StepType
from .tracer import Tracer
from .run_store import RunStore


class AgentWrapper:
    """
    Generic wrapper that intercepts LLM calls and records them.

    Works with any callable that takes messages and returns a response.

    Example:
        def my_llm(messages):
            return openai.chat.completions.create(...)

        wrapper = AgentWrapper(my_llm, agent_id="my_agent")
        result = wrapper(messages)  # same interface, now recorded
    """

    def __init__(
        self,
        fn: Callable,
        agent_id: str = "default",
        store: Optional[RunStore] = None,
        extract_content: Optional[Callable] = None,
        extract_tokens: Optional[Callable] = None,
    ):
        self.fn = fn
        self.tracer = Tracer(agent_id=agent_id, store=store or RunStore())
        self.extract_content = extract_content or self._default_content
        self.extract_tokens = extract_tokens or self._default_tokens
        self._active_run = None
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs) -> Any:
        messages = args[0] if args else kwargs.get("messages", [])

        with self.tracer.run() as run:
            # Log the prompt
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        run.add_step(Step(
                            step_type=StepType.PROMPT,
                            content=content,
                            metadata={"role": role},
                        ))

            t0 = time.time()
            result = self.fn(*args, **kwargs)
            latency = (time.time() - t0) * 1000

            content = self.extract_content(result)
            tokens_in, tokens_out = self.extract_tokens(result)

            run.add_step(Step(
                step_type=StepType.LLM_RESPONSE,
                content=content,
                latency_ms=latency,
                input_tokens=tokens_in,
                output_tokens=tokens_out,
            ))

            self._active_run = run
            return result

    @staticmethod
    def _default_content(result: Any) -> str:
        if hasattr(result, "choices"):
            return result.choices[0].message.content or ""
        if isinstance(result, str):
            return result
        return str(result)

    @staticmethod
    def _default_tokens(result: Any):
        if hasattr(result, "usage") and result.usage:
            return result.usage.prompt_tokens, result.usage.completion_tokens
        return None, None


def wrap_openai(client, agent_id: str = "default", store: Optional[RunStore] = None):
    """
    Wrap an OpenAI client so every chat.completions.create call is traced.

    Example:
        from openai import OpenAI
        import agent_lightning as al

        client = OpenAI()
        client = al.wrap_openai(client, agent_id="my_agent")

        # Now use client exactly as before — runs are recorded automatically.
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}]
        )
    """
    tracer = Tracer(agent_id=agent_id, store=store or RunStore())
    original_create = client.chat.completions.create

    @functools.wraps(original_create)
    def patched_create(*args, **kwargs):
        messages = kwargs.get("messages", args[0] if args else [])
        model = kwargs.get("model", "unknown")

        with tracer.run() as run:
            for msg in (messages or []):
                if isinstance(msg, dict):
                    run.add_step(Step(
                        step_type=StepType.PROMPT,
                        content=msg.get("content", ""),
                        model=model,
                        metadata={"role": msg.get("role", "user")},
                    ))

            t0 = time.time()
            response = original_create(*args, **kwargs)
            latency = (time.time() - t0) * 1000

            content = ""
            if hasattr(response, "choices") and response.choices:
                content = response.choices[0].message.content or ""

            tokens_in = tokens_out = None
            if hasattr(response, "usage") and response.usage:
                tokens_in = response.usage.prompt_tokens
                tokens_out = response.usage.completion_tokens

            run.add_step(Step(
                step_type=StepType.LLM_RESPONSE,
                content=content,
                model=model,
                latency_ms=latency,
                input_tokens=tokens_in,
                output_tokens=tokens_out,
            ))

        return response

    client.chat.completions.create = patched_create
    client._lightning_tracer = tracer
    return client


def wrap_langchain_llm(llm, agent_id: str = "default", store: Optional[RunStore] = None):
    """
    Wrap a LangChain BaseLLM/BaseChatModel to record invocations.

    Example:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o")
        llm = wrap_langchain_llm(llm, agent_id="lc_agent")
    """
    tracer = Tracer(agent_id=agent_id, store=store or RunStore())
    original_invoke = llm.invoke

    @functools.wraps(original_invoke)
    def patched_invoke(input, *args, **kwargs):
        with tracer.run() as run:
            input_str = str(input)[:2000]
            run.add_step(Step(step_type=StepType.PROMPT, content=input_str))

            t0 = time.time()
            result = original_invoke(input, *args, **kwargs)
            latency = (time.time() - t0) * 1000

            content = result.content if hasattr(result, "content") else str(result)
            run.add_step(Step(
                step_type=StepType.LLM_RESPONSE,
                content=content,
                latency_ms=latency,
            ))

        return result

    llm.invoke = patched_invoke
    llm._lightning_tracer = tracer
    return llm
