"""Tests for triggers, evaluator, and orchestrator logic."""
import asyncio
import math
import pytest
from unittest.mock import AsyncMock, MagicMock
from agent_lightning.core.store import LightningStore
from agent_lightning.core.models import Run, Step, StepType
from agent_lightning.orchestrator.triggers import EveryNRuns, Scheduled, Manual, OnImprovement
from agent_lightning.orchestrator.evaluator import Evaluator, ABTest, _normal_cdf
from agent_lightning.orchestrator.orchestrator import Orchestrator, OrchestratorConfig


@pytest.fixture
def store():
    return LightningStore(":memory:")


def make_run(store, agent_id="test", reward=1.0, status="completed"):
    run = Run(agent_id=agent_id)
    run.add_step(Step(step_type=StepType.PROMPT, content="q"))
    run.add_step(Step(step_type=StepType.LLM_RESPONSE, content="a"))
    run.add_reward(reward); run.finish(status); store.save(run)
    return run


# ── Triggers ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_every_n_runs_fires(store):
    trigger = EveryNRuns(n=5)
    for _ in range(4):
        make_run(store)
    assert not await trigger.should_fire(store, "test")
    make_run(store)
    assert await trigger.should_fire(store, "test")


@pytest.mark.asyncio
async def test_every_n_runs_doesnt_double_fire(store):
    trigger = EveryNRuns(n=3)
    for _ in range(3): make_run(store)
    assert await trigger.should_fire(store, "test")
    assert not await trigger.should_fire(store, "test")


@pytest.mark.asyncio
async def test_manual_trigger(store):
    t = Manual()
    assert not await t.should_fire(store, "test")
    t.trigger()
    assert await t.should_fire(store, "test")
    assert not await t.should_fire(store, "test")  # one-shot


@pytest.mark.asyncio
async def test_on_improvement_trigger(store):
    t = OnImprovement(min_avg_reward=0.6, window=5)
    # Not enough runs
    assert not await t.should_fire(store, "test")
    for _ in range(5):
        make_run(store, reward=0.3)  # below threshold
    assert await t.should_fire(store, "test")


@pytest.mark.asyncio
async def test_on_improvement_no_fire_high_reward(store):
    t = OnImprovement(min_avg_reward=0.5, window=5)
    for _ in range(5):
        make_run(store, reward=0.9)  # above threshold
    assert not await t.should_fire(store, "test")


# ── Evaluator ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_evaluator_picks_winner():
    from agent_lightning.backends.base import LLMBackend, LLMMessage, LLMResponse

    class GoodBadBackend(LLMBackend):
        """Returns better answer for prompt_b."""
        async def complete(self, messages, **kw):
            system = next((m.content for m in messages if m.role == "system"), "")
            if "concise" in system:
                return LLMResponse("Short answer.")
            return LLMResponse("This is a very long and verbose answer that goes on and on.")

    def reward(response, item): return 1.0 if len(response) < 50 else 0.2

    evaluator = Evaluator(
        backend=GoodBadBackend(),
        reward_fn=reward,
        n_samples=5,
        significance_threshold=1.0,  # always "significant" for test
    )

    result = await evaluator.evaluate(
        "You are verbose.", "You are concise.",
        dataset=[f"Question {i}" for i in range(10)]
    )
    assert result.score_b > result.score_a
    assert result.winner == "b"


def test_normal_cdf():
    assert abs(_normal_cdf(0) - 0.5) < 0.01
    assert _normal_cdf(3) > 0.99
    assert _normal_cdf(-3) < 0.01


def test_ab_test_tracking():
    ab = ABTest("Prompt A", "Prompt B", split=0.5)
    for _ in range(10): ab.record("a", 0.5)
    for _ in range(10): ab.record("b", 0.8)
    stats = ab.stats
    assert stats["winner"] == "b"
    assert stats["avg_b"] == 0.8


def test_ab_test_sample():
    ab = ABTest("A", "B", split=1.0)  # always return A
    prompt, variant = ab.sample()
    assert prompt == "A" and variant == "a"


# ── Orchestrator ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_orchestrator_train_now():
    """Orchestrator.train_now() calls algorithm.run() and deploys if successful."""
    store = LightningStore(":memory:")
    for _ in range(5):
        make_run(store)

    # Mock algorithm
    mock_algo = MagicMock()
    mock_algo.run = AsyncMock(return_value={
        "success": True,
        "best_prompt": "Improved prompt!",
        "best_score": 0.95,
    })

    orch = Orchestrator(
        store=store,
        algorithm=mock_algo,
        config=OrchestratorConfig(
            agent_id="test",
            auto_deploy=True,
            require_eval=False,
        ),
    )

    result = await orch.train_now()
    assert result["prompt"] == "Improved prompt!"
    assert result["score"] == 0.95
    mock_algo.run.assert_called_once()

    # Check deployed to store
    deployed = store.get_optimized_prompt("test")
    assert deployed == "Improved prompt!"


@pytest.mark.asyncio
async def test_orchestrator_no_deploy_on_failure():
    """If algorithm fails, nothing is deployed."""
    store = LightningStore(":memory:")
    mock_algo = MagicMock()
    mock_algo.run = AsyncMock(return_value={"success": False, "message": "no data"})

    orch = Orchestrator(store=store, algorithm=mock_algo,
                        config=OrchestratorConfig(agent_id="test", auto_deploy=True, require_eval=False))
    await orch.train_now()
    assert store.get_optimized_prompt("test") is None


def test_orchestrator_status():
    store = LightningStore(":memory:")
    mock_algo = MagicMock()
    orch = Orchestrator(store=store, algorithm=mock_algo,
                        config=OrchestratorConfig(agent_id="x", trigger=EveryNRuns(10)))
    status = orch.status
    assert status["agent_id"] == "x"
    assert status["trigger"] == "EveryNRuns"
    assert not status["running"]
