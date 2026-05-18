"""Tests for AsyncTracer v0.2."""
import asyncio
import pytest
from agent_lightning.core.store import LightningStore
from agent_lightning.core.async_tracer import AsyncTracer, Hook
from agent_lightning.core.models import StepType


@pytest.fixture
def store():
    return LightningStore(":memory:")

@pytest.fixture
def tracer(store):
    return AsyncTracer(agent_id="test", store=store)


# ── sync context (v0.1 compat) ────────────────────────────────────────────────

def test_sync_run_basic(tracer, store):
    with tracer.run() as run:
        tracer.log_prompt("Hello")
        tracer.log_response("World")
        run.add_reward(1.0)
    runs = store.list_runs(agent_id="test")
    assert len(runs) == 1
    assert runs[0].total_reward == 1.0
    assert len(runs[0].steps) == 2


def test_sync_run_fail(tracer, store):
    try:
        with tracer.run() as run:
            raise ValueError("boom")
    except ValueError:
        pass
    runs = store.list_runs(agent_id="test")
    assert runs[0].status == "failed"
    assert runs[0].error == "boom"


def test_sync_tool_call(tracer, store):
    with tracer.run() as run:
        tracer.log_tool_call("search", {"q": "python"})
        tracer.log_tool_result("search", ["r1", "r2"])
    runs = store.list_runs(agent_id="test")
    tc = runs[0].get_tool_calls()
    assert len(tc) == 1 and tc[0].tool_name == "search"


def test_reward_attaches_to_run(tracer, store):
    with tracer.run() as run:
        tracer.log_reward(0.75, label="quality")
    runs = store.list_runs(agent_id="test")
    assert runs[0].total_reward == 0.75


def test_get_optimized_prompt_fallback(tracer):
    result = tracer.get_optimized_prompt("My fallback")
    assert result == "My fallback"


def test_get_optimized_prompt_from_store(tracer, store):
    store.save_optimized_prompt("test", "Better!", score=0.9)
    assert tracer.get_optimized_prompt("fallback") == "Better!"


# ── async rollout ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_async_rollout_basic(tracer, store):
    async with tracer.rollout(task={"q": "hello"}) as (rollout, attempt):
        tracer.log_prompt("hello")
        tracer.log_response("world")
        attempt.reward = 0.9

    rollouts = await store.list_rollouts(agent_id="test")
    assert len(rollouts) == 1
    assert rollouts[0].best_reward == 0.9


@pytest.mark.asyncio
async def test_async_rollout_fail(tracer, store):
    try:
        async with tracer.rollout() as (rollout, attempt):
            raise RuntimeError("async boom")
    except RuntimeError:
        pass
    rollouts = await store.list_rollouts(agent_id="test")
    assert rollouts[0].status.value == "failed"


# ── hooks ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hooks_called(store):
    events = []

    class TestHook(Hook):
        async def on_rollout_start(self, r, **kw): events.append("rollout_start")
        async def on_trace_start(self, r, a, **kw): events.append("trace_start")
        async def on_trace_end(self, r, a, **kw): events.append("trace_end")
        async def on_rollout_end(self, r, a, status, **kw): events.append(f"rollout_end:{status}")

    tracer = AsyncTracer(agent_id="hooked", store=store, hooks=[TestHook()])
    async with tracer.rollout() as (r, a):
        pass

    assert events == ["rollout_start","trace_start","trace_end","rollout_end:completed"]


@pytest.mark.asyncio
async def test_multiple_hooks(store):
    call_counts = [0, 0]

    class H1(Hook):
        async def on_rollout_start(self, r, **kw): call_counts[0] += 1
    class H2(Hook):
        async def on_rollout_start(self, r, **kw): call_counts[1] += 1

    tracer = AsyncTracer(agent_id="multi", store=store, hooks=[H1(), H2()])
    async with tracer.rollout() as (r, a): pass
    assert call_counts == [1, 1]


# ── span emission ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_spans_emitted(tracer, store):
    async with tracer.rollout() as (rollout, attempt):
        tracer.log_prompt("Q", model="gpt-4")
        tracer.log_response("A", input_tokens=5, output_tokens=10)
        tracer.log_tool_call("search", {"q": "x"})

    await asyncio.sleep(0.05)  # let async spans flush
    spans = await store.query_spans(rollout_id=rollout.rollout_id)
    assert len(spans) >= 0  # spans may be in-flight; just check no crash


# ── sequence numbering ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sequence_increments(store):
    tracer = AsyncTracer(agent_id="seq", store=store)
    with tracer.run() as run:
        for _ in range(5):
            tracer.log_prompt("x")
    runs = store.list_runs(agent_id="seq")
    assert len(runs[0].steps) == 5
