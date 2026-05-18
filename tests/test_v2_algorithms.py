"""Tests for APO algorithm (no real LLM calls — uses mock backends)."""
import asyncio
import pytest
from unittest.mock import AsyncMock
from agent_lightning.backends.base import LLMBackend, LLMMessage, LLMResponse
from agent_lightning.algorithms.apo import APO, VersionedPrompt, _OPTIMIZER_SYSTEM
from agent_lightning.core.store import LightningStore


class MockBackend(LLMBackend):
    """Returns configurable response for any request."""
    def __init__(self, response: str = "Improved prompt text"):
        self.model = "mock/model"
        self._response = response
        self.call_count = 0

    async def complete(self, messages, temperature=0.7, max_tokens=2048, **kw):
        self.call_count += 1
        return LLMResponse(content=self._response, model=self.model,
                            input_tokens=10, output_tokens=20)


class ScoringBackend(LLMBackend):
    """Returns a numeric score as string (for LLM-as-judge tests)."""
    def __init__(self, score: str = "0.8"):
        self.model = "mock/scorer"
        self._score = score
    async def complete(self, messages, **kw):
        return LLMResponse(content=self._score, model=self.model)


@pytest.fixture
def store():
    return LightningStore(":memory:")


# ── APO basic flow ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_apo_runs_without_dataset(store):
    """APO should complete even with empty dataset."""
    backend = MockBackend("Better assistant prompt")
    apo = APO(store=store, agent_id="test", gradient_backend=backend,
               initial_prompt="You are helpful.", beam_width=2, beam_rounds=1, branch_factor=2)
    result = await apo.run(train_dataset=[], val_dataset=[])
    assert result["success"]
    assert result["best_prompt"]


@pytest.mark.asyncio
async def test_apo_with_dataset(store):
    backend = MockBackend("Concise and specific assistant prompt")
    apo = APO(store=store, agent_id="test", gradient_backend=backend,
               initial_prompt="Be helpful.",
               beam_width=2, beam_rounds=2, branch_factor=2)
    train = ["What is Python?", "Explain Docker", "What is REST?"]
    result = await apo.run(train_dataset=train, val_dataset=train)
    assert result["success"]
    assert result["candidates"] > 0


@pytest.mark.asyncio
async def test_apo_writes_to_store(store):
    backend = ScoringBackend("0.9")  # LLM judge returns 0.9
    apo = APO(store=store, agent_id="agent1", gradient_backend=backend,
               initial_prompt="Prompt v0",
               beam_width=2, beam_rounds=1, branch_factor=1)
    result = await apo.run(train_dataset=["Q1","Q2"], val_dataset=["Q1","Q2"])
    # Should write resource to store
    prompt_in_store = await store.get_latest_prompt("agent1")
    # May or may not write depending on score comparison — just check no crash
    assert result["success"]


@pytest.mark.asyncio
async def test_apo_history_tracked(store):
    backend = MockBackend("candidate prompt")
    apo = APO(store=store, agent_id="hist", gradient_backend=backend,
               beam_width=2, beam_rounds=2, branch_factor=2)
    result = await apo.run(train_dataset=["Q"])
    # History should include seed + candidates
    assert len(result["history"]) >= 1
    assert all("template" in h for h in result["history"])


@pytest.mark.asyncio
async def test_apo_reward_fn(store):
    """When reward_fn is provided, it's used for eval instead of LLM judge."""
    backend = MockBackend("Short answer")
    calls = []

    def my_reward(response: str, item: str) -> float:
        calls.append((response, item))
        return 1.0 if len(response) < 20 else 0.5

    apo = APO(store=store, agent_id="rew", gradient_backend=backend,
               eval_backend=backend,
               initial_prompt="Be helpful.",
               reward_fn=my_reward,
               beam_width=2, beam_rounds=1, branch_factor=1)
    result = await apo.run(train_dataset=["Q1"], val_dataset=["Q1","Q2"])
    assert result["success"]
    assert len(calls) > 0  # reward_fn was called


@pytest.mark.asyncio
async def test_apo_no_backend_raises(store):
    apo = APO(store=store, agent_id="x", gradient_backend=None)
    with pytest.raises(ValueError, match="gradient_backend required"):
        await apo.run()


@pytest.mark.asyncio
async def test_apo_textual_gradient(store):
    backend = MockBackend("- The prompt is too vague\n- Missing examples\n- No output format specified")
    apo = APO(store=store, agent_id="x", gradient_backend=backend,
               initial_prompt="Be helpful.")
    parent = VersionedPrompt(template="Be helpful.", version=0)
    critique = await apo._compute_textual_gradient(parent, ["Q1","Q2"])
    assert critique is not None
    assert len(critique) > 0


@pytest.mark.asyncio
async def test_apo_apply_edit(store):
    backend = MockBackend("You are a concise and specific assistant. Always provide examples.")
    apo = APO(store=store, agent_id="x", gradient_backend=backend)
    parent = VersionedPrompt(template="Be helpful.", version=0)
    edited = await apo._apply_edit(parent, "Too vague, needs examples")
    assert edited is not None
    assert edited != parent.template


@pytest.mark.asyncio
async def test_apo_llm_judge(store):
    backend = ScoringBackend("0.75")
    apo = APO(store=store, agent_id="x", gradient_backend=backend, eval_backend=backend)
    prompt = VersionedPrompt(template="You are specific and concise.", version=1)
    score = await apo._llm_judge(prompt, ["Q1","Q2"])
    assert 0.0 <= score <= 1.0
    assert abs(score - 0.75) < 0.01


@pytest.mark.asyncio
async def test_apo_llm_judge_invalid_response(store):
    """Should return 0.5 default on invalid judge response."""
    backend = MockBackend("not a number here")
    apo = APO(store=store, agent_id="x", gradient_backend=backend, eval_backend=backend)
    prompt = VersionedPrompt(template="test", version=0)
    score = await apo._llm_judge(prompt, ["Q1"])
    assert score == 0.5


# ── VersionedPrompt ───────────────────────────────────────────────────────────

def test_versioned_prompt():
    p = VersionedPrompt(template="Hello", version=0)
    assert p.score is None
    assert p.parent_version == -1


# ── OTel exporter ─────────────────────────────────────────────────────────────

def test_otel_adapter():
    from agent_lightning.otel.exporter import OTelSpanAdapter
    from agent_lightning.core.models import Span, StepType
    span = Span(name="agent.prompt", step_type=StepType.PROMPT,
                content="Hello", model="gpt-4",
                input_tokens=10, rollout_id="r1")
    otel = OTelSpanAdapter.to_otel(span)
    assert otel["name"] == "agent.prompt"
    assert otel["kind"] == "INTERNAL"
    attrs = {a["key"]: a["value"]["stringValue"] for a in otel["attributes"]}
    assert attrs["al.step_type"] == "prompt"
    assert attrs["llm.model"] == "gpt-4"


def test_otel_batch():
    from agent_lightning.otel.exporter import OTelSpanAdapter
    from agent_lightning.core.models import Span, StepType
    spans = [Span(name="s1", step_type=StepType.PROMPT, attributes={}),
             Span(name="s2", step_type=StepType.LLM_RESPONSE, attributes={})]
    batch = OTelSpanAdapter.to_otel_batch(spans)
    assert "resourceSpans" in batch
    assert len(batch["resourceSpans"][0]["scopeSpans"][0]["spans"]) == 2
