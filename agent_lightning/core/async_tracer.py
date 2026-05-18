"""
AsyncTracer v0.2 — async-first tracer with OTel span emission,
hooks support, and rollout/attempt lifecycle management.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import time
import uuid
from typing import Any, Callable, Dict, Generator, List, Optional, TYPE_CHECKING

from .models import Run, Step, StepType, Span, Reward, Rollout, Attempt, RolloutStatus
from .store import LightningStore

log = logging.getLogger(__name__)


class Hook:
    """Lifecycle callbacks called by the Runner around each rollout."""
    async def on_rollout_start(self, rollout: Rollout, **kw): pass
    async def on_trace_start(self, rollout: Rollout, attempt: Attempt, **kw): pass
    async def on_trace_end(self, rollout: Rollout, attempt: Attempt, **kw): pass
    async def on_rollout_end(self, rollout: Rollout, attempt: Attempt, status: str, **kw): pass


class AsyncTracer:
    """
    Async-first tracer. Instruments agent code and streams OTel-compatible
    spans to the LightningStore.

    Usage:
        tracer = AsyncTracer(agent_id="my_agent", store=store)

        async with tracer.run() as run:
            tracer.log_prompt("Hello")
            tracer.log_response("World")
            run.add_reward(1.0)
    """

    def __init__(
        self,
        agent_id: str = "default",
        store: Optional[LightningStore] = None,
        hooks: Optional[List[Hook]] = None,
        auto_save: bool = True,
        worker_id: Optional[str] = None,
    ):
        self.agent_id   = agent_id
        self.store      = store or LightningStore()
        self.hooks      = hooks or []
        self.auto_save  = auto_save
        self.worker_id  = worker_id or str(uuid.uuid4())[:8]
        self._active_run: Optional[Run]     = None
        self._current_rollout: Optional[Rollout]  = None
        self._current_attempt: Optional[Attempt]  = None
        self._seq: int  = 0

    # ── async context manager (new API) ───────────────────────────────────────

    @contextlib.asynccontextmanager
    async def rollout(
        self,
        task: Any = None,
        rollout_id: str = None,
        resources: Dict[str, Any] = None,
        tags: List[str] = None,
        mode: str = "train",
    ):
        """
        Async context manager for a single rollout execution.

        async with tracer.rollout(task=my_task) as (rollout, attempt):
            tracer.log_prompt("hello")
            tracer.log_response("world")
            attempt.reward = 1.0
        """
        r = Rollout(
            rollout_id=rollout_id or str(uuid.uuid4()),
            agent_id=self.agent_id,
            task=task, resources=resources or {},
            tags=tags or [], mode=mode,
        )
        attempt = r.new_attempt(worker_id=self.worker_id)
        self._current_rollout = r
        self._current_attempt = attempt
        self._seq = 0

        # Fire hooks
        for h in self.hooks:
            await h.on_rollout_start(r)
            await h.on_trace_start(r, attempt)

        try:
            yield r, attempt
            attempt.finish(reward=attempt.reward)
            r.status = RolloutStatus.COMPLETED
        except Exception as e:
            attempt.finish(error=str(e))
            r.status = RolloutStatus.FAILED
            raise
        finally:
            # Fire hooks
            for h in self.hooks:
                await h.on_trace_end(r, attempt)
                await h.on_rollout_end(r, attempt, r.status.value)

            if self.auto_save:
                await self.store.enqueue_rollout(r)
                await self.store.update_rollout_status(r.rollout_id, r.status,
                                                       reward=attempt.reward, error=attempt.error)
            self._current_rollout = None
            self._current_attempt = None

    # ── sync context manager (v0.1 compat) ───────────────────────────────────

    @contextlib.contextmanager
    def run(
        self,
        session_id: str = None,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
    ):
        """v0.1 compatible sync context manager."""
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

    # ── span emission ─────────────────────────────────────────────────────────

    def _emit_span(self, span: Span):
        """Emit span to store (fire-and-forget async from sync context)."""
        span.rollout_id = self._current_rollout.rollout_id if self._current_rollout else None
        span.attempt_id = self._current_attempt.attempt_id if self._current_attempt else None
        span.attributes["agent_id"] = self.agent_id
        span.sequence_id = self._seq
        self._seq += 1

        # Also attach to v0.1 run if active
        if self._active_run:
            step = Step(
                step_type=span.step_type or StepType.CUSTOM,
                content=span.content,
                model=span.model,
                input_tokens=span.input_tokens,
                output_tokens=span.output_tokens,
                latency_ms=span.latency_ms,
                tool_name=span.tool_name,
                tool_args=span.tool_args,
                tool_error=span.tool_error,
                reward=span.reward,
            )
            self._active_run.add_step(step)

        # Async emit (non-blocking)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.store.add_span(span))
        except RuntimeError:
            pass  # No event loop — fire-and-forget skipped in sync context

    def log_prompt(self, content: str, model: str = None, role: str = "user", **kwargs):
        self._emit_span(Span(
            name="agent.prompt", step_type=StepType.PROMPT,
            content=content, model=model,
            attributes={"role": role, **kwargs},
        ))

    def log_response(self, content: str, model: str = None,
                     input_tokens: int = None, output_tokens: int = None,
                     latency_ms: float = None):
        self._emit_span(Span(
            name="agent.llm_response", step_type=StepType.LLM_RESPONSE,
            content=content, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            latency_ms=latency_ms,
        ))

    def log_tool_call(self, tool_name: str, tool_args: Dict[str, Any]):
        self._emit_span(Span(
            name=f"agent.tool.{tool_name}", step_type=StepType.TOOL_CALL,
            tool_name=tool_name, tool_args=tool_args,
            content=f"Calling {tool_name}",
        ))

    def log_tool_result(self, tool_name: str, result: Any, error: str = None):
        self._emit_span(Span(
            name=f"agent.tool_result.{tool_name}", step_type=StepType.TOOL_RESULT,
            tool_name=tool_name, content=result, tool_error=error,
        ))

    def log_reward(self, value: float, label: str = "default", **metadata):
        reward = Reward(value=value, label=label, metadata=metadata)
        self._emit_span(Span(
            name="agent.reward", step_type=StepType.REWARD,
            reward=reward, content=value,
        ))
        # Also attach to active run
        if self._active_run:
            self._active_run.add_reward(value, label, **metadata)
        # Attach to active attempt
        if self._current_attempt:
            self._current_attempt.reward = value

    def get_optimized_prompt(self, fallback: str = "") -> str:
        return self.store.get_optimized_prompt(self.agent_id) or fallback

    @property
    def current_run(self) -> Optional[Run]:
        return self._active_run


# ── backwards compat alias ────────────────────────────────────────────────────
Tracer = AsyncTracer


def trace(agent_id: str = "default", store=None, reward_fn=None):
    """@trace decorator — wraps any function for automatic tracing."""
    _store = store or LightningStore()
    tracer = AsyncTracer(agent_id=agent_id, store=_store)

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with tracer.run() as run:
                run.metadata["fn"] = fn.__name__
                t0 = time.time()
                result = fn(*args, **kwargs)
                latency = (time.time() - t0) * 1000
                run.add_step(Step(step_type=StepType.LLM_RESPONSE,
                                  content=str(result)[:2000], latency_ms=latency))
                if reward_fn:
                    try: run.add_reward(float(reward_fn(result)))
                    except: pass
                return result
        wrapper._tracer = tracer
        return wrapper
    return decorator
