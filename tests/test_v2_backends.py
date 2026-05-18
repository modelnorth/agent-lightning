"""Tests for backends (no real API calls — uses mocks)."""
import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agent_lightning.backends.base import LLMBackend, LLMMessage, LLMResponse
from agent_lightning.backends.openrouter import OpenRouterBackend, MultiModelConsensus, OPENROUTER_MODELS


# ── LLMMessage / LLMResponse ──────────────────────────────────────────────────

def test_llm_message_to_dict():
    m = LLMMessage("user", "Hello")
    assert m.to_dict() == {"role": "user", "content": "Hello"}


def test_llm_response_total_tokens():
    r = LLMResponse("hi", input_tokens=10, output_tokens=5)
    assert r.total_tokens == 15


def test_llm_response_no_tokens():
    r = LLMResponse("hi")
    assert r.total_tokens == 0


# ── OpenRouterBackend ─────────────────────────────────────────────────────────

def test_openrouter_model_tiers():
    assert "fast" in OPENROUTER_MODELS
    assert "balanced" in OPENROUTER_MODELS
    assert "strong" in OPENROUTER_MODELS
    assert len(OPENROUTER_MODELS["fast"]) > 0


def test_openrouter_factory_fast():
    b = OpenRouterBackend.fast(api_key="test")
    assert b.model == OPENROUTER_MODELS["fast"][0]
    assert len(b.fallback_models) > 0


def test_openrouter_factory_strong():
    b = OpenRouterBackend.strong(api_key="test")
    assert b.model == OPENROUTER_MODELS["strong"][0]


def test_openrouter_usage_tracking():
    b = OpenRouterBackend(api_key="test")
    resp = LLMResponse("hi", model="qwen/test", input_tokens=10, output_tokens=5)
    b._track_usage("qwen/test", resp)
    b._track_usage("qwen/test", resp)
    assert b.usage_stats["qwen/test"]["calls"] == 2
    assert b.usage_stats["qwen/test"]["input_tokens"] == 20


@pytest.mark.asyncio
async def test_openrouter_mock_call():
    """Mock the aiohttp call and verify parsing."""
    b = OpenRouterBackend(api_key="sk-test", model="qwen/test")

    mock_response_data = {
        "choices": [{"message": {"content": "Test response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }

    with patch("aiohttp.ClientSession") as MockSession:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post.return_value = mock_resp
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        MockSession.return_value = mock_session

        result = await b.complete([LLMMessage("user", "Hello")])
        assert result.content == "Test response"
        assert result.input_tokens == 10
        assert result.output_tokens == 20


@pytest.mark.asyncio
async def test_openrouter_fallback():
    """If primary model fails, fallback model should be tried."""
    b = OpenRouterBackend(api_key="sk-test", model="primary/model",
                          fallback_models=["fallback/model"], max_retries=1)

    call_count = {"n": 0}
    async def fake_call(model, messages, temp, max_tok, **kw):
        call_count["n"] += 1
        if model == "primary/model":
            raise RuntimeError("primary failed")
        return LLMResponse("fallback response", model=model)

    b._call = fake_call
    result = await b.complete([LLMMessage("user", "test")])
    assert result.content == "fallback response"
    assert call_count["n"] >= 2


# ── MultiModelConsensus ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_multi_model_consensus_complete_all():
    """All backends called in parallel."""

    class MockBackend(LLMBackend):
        def __init__(self, name, response):
            self.model = name
            self._response = response
        async def complete(self, messages, **kw):
            return LLMResponse(content=self._response, model=self.model)

    backends = [
        MockBackend("m1", "Response from model 1"),
        MockBackend("m2", "Response from model 2 is longer"),
        MockBackend("m3", "R3"),
    ]
    consensus = MultiModelConsensus(backends=backends)
    responses = await consensus.complete_all([LLMMessage("user", "test")])
    assert len(responses) == 3
    contents = {r.content for r in responses}
    assert "Response from model 1" in contents


@pytest.mark.asyncio
async def test_multi_model_best_of_score_fn():
    class MockBackend(LLMBackend):
        def __init__(self, resp): self._resp = resp; self.model = "m"
        async def complete(self, messages, **kw): return LLMResponse(self._resp)

    backends = [MockBackend("short"), MockBackend("this is the longest response here")]
    consensus = MultiModelConsensus(backends=backends)
    best = await consensus.best_of("test", score_fn=lambda r: len(r))
    assert best == "this is the longest response here"


# ── LLMBackend ABC ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_complete_text_convenience():
    class EchoBackend(LLMBackend):
        async def complete(self, messages, **kw):
            content = " | ".join(f"{m.role}:{m.content}" for m in messages)
            return LLMResponse(content=content)

    b = EchoBackend()
    result = await b.complete_text("Hello", system="System msg")
    assert "system:System msg" in result
    assert "user:Hello" in result
