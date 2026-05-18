"""Tests for LightningStore v0.2 async features."""
import asyncio
import pytest
from agent_lightning.core.store import LightningStore
from agent_lightning.core.models import (Rollout, Attempt, Span, StepType,
                                          RolloutStatus, PromptTemplate)


@pytest.fixture
def store():
    return LightningStore(":memory:")


def make_rollout(agent_id="test", mode="train") -> Rollout:
    return Rollout(agent_id=agent_id, task={"q": "hello"}, mode=mode)


# ── Rollout queue ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_enqueue_dequeue(store):
    r = make_rollout()
    await store.enqueue_rollout(r)
    dequeued = await store.dequeue_rollout(agent_id="test", timeout=0.5)
    assert dequeued is not None
    assert dequeued.rollout_id == r.rollout_id


@pytest.mark.asyncio
async def test_dequeue_empty_returns_none(store):
    result = await store.dequeue_rollout(agent_id="nobody", timeout=0.1)
    assert result is None


@pytest.mark.asyncio
async def test_enqueue_batch(store):
    rollouts = [make_rollout() for _ in range(5)]
    await store.enqueue_batch(rollouts)
    listed = await store.list_rollouts(agent_id="test")
    assert len(listed) == 5


@pytest.mark.asyncio
async def test_update_rollout_status(store):
    r = make_rollout()
    attempt = r.new_attempt(worker_id="w1")
    await store.enqueue_rollout(r)
    await store.update_rollout_status(r.rollout_id, RolloutStatus.COMPLETED, reward=0.9)
    fetched = await store.get_rollout(r.rollout_id)
    assert fetched.status == RolloutStatus.COMPLETED


@pytest.mark.asyncio
async def test_list_rollouts_filter(store):
    for _ in range(3):
        r = make_rollout(mode="train"); await store.enqueue_rollout(r)
    for _ in range(2):
        r = make_rollout(mode="val"); await store.enqueue_rollout(r)
    train = await store.list_rollouts(mode="train")
    val   = await store.list_rollouts(mode="val")
    assert len(train) == 3
    assert len(val) == 2


# ── Spans ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_add_and_query_spans(store):
    span = Span(
        name="agent.prompt", step_type=StepType.PROMPT,
        content="Hello", rollout_id="r1",
        attributes={"agent_id": "test"},
    )
    await store.add_span(span)
    spans = await store.query_spans(rollout_id="r1")
    assert len(spans) == 1
    assert spans[0].content == "Hello"


@pytest.mark.asyncio
async def test_query_spans_by_type(store):
    for st in [StepType.PROMPT, StepType.LLM_RESPONSE, StepType.TOOL_CALL]:
        await store.add_span(Span(step_type=st, name=f"agent.{st.value}",
                                   rollout_id="r2", attributes={"agent_id":"test"}))
    prompts = await store.query_spans(rollout_id="r2", step_type="prompt")
    assert len(prompts) == 1


@pytest.mark.asyncio
async def test_get_triplets(store):
    from agent_lightning.core.models import Reward
    rid = "r3"
    await store.add_span(Span(step_type=StepType.PROMPT, content="Q", rollout_id=rid,
                               name="agent.prompt", sequence_id=0, attributes={"agent_id":"test"}))
    await store.add_span(Span(step_type=StepType.LLM_RESPONSE, content="A", rollout_id=rid,
                               name="agent.response", sequence_id=1, attributes={"agent_id":"test"}))
    await store.add_span(Span(step_type=StepType.REWARD, rollout_id=rid, name="agent.reward",
                               reward=Reward(0.9), sequence_id=2, attributes={"agent_id":"test"}))
    triplets = await store.get_triplets(rid)
    assert len(triplets) == 1
    assert triplets[0][0] == "Q"
    assert triplets[0][1] == "A"


# ── Resources ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_set_get_resource(store):
    pt = PromptTemplate(template="Be concise.", score=0.92)
    await store.set_resource("main_prompt", "agent1", pt, kind="prompt", score=0.92)
    val = await store.get_resource("main_prompt", "agent1")
    assert val is not None
    assert val["template"] == "Be concise."


@pytest.mark.asyncio
async def test_resource_versioning(store):
    pt1 = PromptTemplate(template="v1")
    pt2 = PromptTemplate(template="v2")
    await store.set_resource("p", "a1", pt1)
    await store.set_resource("p", "a1", pt2)
    val = await store.get_resource("p", "a1")
    assert val["template"] == "v2"


@pytest.mark.asyncio
async def test_get_latest_prompt(store):
    await store.set_resource("main_prompt", "ag", PromptTemplate(template="Better!"), kind="prompt")
    p = await store.get_latest_prompt("ag")
    assert p == "Better!"


@pytest.mark.asyncio
async def test_rollout_stats(store):
    for _ in range(3):
        r = make_rollout(); await store.enqueue_rollout(r)
        await store.update_rollout_status(r.rollout_id, RolloutStatus.COMPLETED)
    stats = await store.rollout_stats(agent_id="test")
    assert stats["completed"] >= 3


# ── Rollout model ─────────────────────────────────────────────────────────────

def test_rollout_attempt():
    r = Rollout(agent_id="x")
    assert r.best_reward is None
    a = r.new_attempt("worker1")
    assert r.status == RolloutStatus.RUNNING
    a.finish(reward=0.8)
    assert r.best_reward == 0.8


def test_rollout_retry():
    r = Rollout(agent_id="x", max_attempts=3)
    assert r.can_retry()
    for i in range(2):
        a = r.new_attempt(); a.finish(error="fail")
    assert r.can_retry()
    a = r.new_attempt(); a.finish(error="fail again")
    assert not r.can_retry()


def test_attempt_duration():
    from datetime import datetime, timezone, timedelta
    a = Attempt(rollout_id="r")
    a.started_at  = datetime(2024,1,1,0,0,0, tzinfo=timezone.utc)
    a.finished_at = datetime(2024,1,1,0,0,1, tzinfo=timezone.utc)
    assert a.duration_ms == 1000.0
