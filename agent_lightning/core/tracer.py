"""
Tracer - the core recording primitive.

Usage:
    tracer = Tracer(agent_id="my_agent")
    
    with tracer.run() as run:
        run.add_step(Step(step_type=StepType.PROMPT, content="..."))
        run.add_reward(1.0)
"""

from __future__ import annotations

import time
import functools
import contextlib
from typing import Any, Callable, Dict, Generator, List, Optional, TYPE_CHECKING

from .models import Run, Step, StepType, Reward
from .run_store import RunStore

if TYPE_CHECKING:
    pass


class Tracer:
    """
    Records agent runs and stores them via a RunStore.

    One Tracer instance per agent (or per agent role in multi-agent workflows).
    """

    def __init__(
        self,
        agent_id: str = "default",
        store: Optional[RunStore] = None,
        auto_save: bool = True,
    ):
        self.agent_id = agent_id
        self.store = store or RunStore()
        self.auto_save = auto_save
        self._active_run: Optional[Run] = None

    @contextlib.contextmanager
    def run(
        self,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Generator[Run, None, None]:
        """
        Context manager that creates, tracks, and saves a Run.

        Example:
            with tracer.run(tags=["prod"]) as run:
                run.add_step(Step(step_type=StepType.PROMPT, content=prompt))
                result = my_llm(prompt)
                run.add_step(Step(step_type=StepType.LLM_RESPONSE, content=result))
                run.add_reward(score_fn(result))
        """
        active_run = Run(
            agent_id=self.agent_id,
            session_id=session_id,
            tags=tags or [],
            metadata=metadata or {},
        )
        self._active_run = active_run
        try:
            yield active_run
            active_run.finish("completed")
        except Exception as e:
            active_run.error = str(e)
            active_run.finish("failed")
            raise
        finally:
            if self.auto_save:
                self.store.save(active_run)
            self._active_run = None

    def log_step(self, step: Step) -> None:
        """Add a step to the currently active run (if any)."""
        if self._active_run is not None:
            self._active_run.add_step(step)

    def log_reward(self, value: float, label: str = "default", **metadata) -> None:
        """Add a reward to the currently active run."""
        if self._active_run is not None:
            self._active_run.add_reward(value, label, **metadata)

    def log_prompt(self, content: str, model: Optional[str] = None, **kwargs) -> None:
        self.log_step(Step(step_type=StepType.PROMPT, content=content, model=model, **kwargs))

    def log_response(self, content: str, model: Optional[str] = None,
                     input_tokens: Optional[int] = None,
                     output_tokens: Optional[int] = None,
                     latency_ms: Optional[float] = None) -> None:
        self.log_step(Step(
            step_type=StepType.LLM_RESPONSE,
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        ))

    def log_tool_call(self, tool_name: str, tool_args: Dict[str, Any]) -> None:
        self.log_step(Step(
            step_type=StepType.TOOL_CALL,
            tool_name=tool_name,
            tool_args=tool_args,
            content=f"Calling {tool_name}",
        ))

    def log_tool_result(self, tool_name: str, result: Any, error: Optional[str] = None) -> None:
        self.log_step(Step(
            step_type=StepType.TOOL_RESULT,
            tool_name=tool_name,
            content=result,
            tool_error=error,
        ))

    def get_optimized_prompt(self, fallback: str) -> str:
        """
        Return the latest optimized prompt for this agent, or fallback.
        Trainers write optimized prompts back here.
        """
        return self.store.get_optimized_prompt(self.agent_id) or fallback

    @property
    def current_run(self) -> Optional[Run]:
        return self._active_run


def trace(
    agent_id: str = "default",
    store: Optional[RunStore] = None,
    reward_fn: Optional[Callable] = None,
):
    """
    Decorator that wraps a function to automatically trace its execution.

    Example:
        @trace(agent_id="summarizer", reward_fn=lambda r: 1.0 if len(r) < 200 else 0.0)
        def summarize(text: str) -> str:
            return llm(f"Summarize: {text}")
    """
    tracer = Tracer(agent_id=agent_id, store=store or RunStore())

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with tracer.run() as run:
                run.metadata["fn_name"] = fn.__name__
                run.metadata["args"] = str(args)[:500]
                run.metadata["kwargs"] = str(kwargs)[:500]

                t0 = time.time()
                result = fn(*args, **kwargs)
                latency = (time.time() - t0) * 1000

                run.add_step(Step(
                    step_type=StepType.LLM_RESPONSE,
                    content=str(result)[:2000],
                    latency_ms=latency,
                ))

                if reward_fn is not None:
                    try:
                        r = reward_fn(result)
                        run.add_reward(float(r))
                    except Exception:
                        pass

                return result
        wrapper._tracer = tracer
        return wrapper
    return decorator
