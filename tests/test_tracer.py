"""Tests for Tracer and decorators."""

import pytest
from agent_lightning.core.tracer import Tracer, trace
from agent_lightning.core.run_store import RunStore
from agent_lightning.core.models import StepType


@pytest.fixture
def store():
    return RunStore(db_path=":memory:")


@pytest.fixture
def tracer(store):
    return Tracer(agent_id="test", store=store)


def test_tracer_basic_run(tracer, store):
    with tracer.run() as run:
        tracer.log_prompt("Hello")
        tracer.log_response("World")
        run.add_reward(1.0)

    runs = store.list_runs(agent_id="test")
    assert len(runs) == 1
    assert runs[0].status == "completed"
    assert len(runs[0].steps) == 2
    assert runs[0].total_reward == 1.0


def test_tracer_run_with_metadata(tracer, store):
    with tracer.run(session_id="sess1", tags=["t1", "t2"], metadata={"user": "bob"}) as run:
        pass

    saved = store.get(run.run_id)
    assert saved.session_id == "sess1"
    assert "t1" in saved.tags
    assert saved.metadata["user"] == "bob"


def test_tracer_run_fails_gracefully(tracer, store):
    with pytest.raises(ValueError):
        with tracer.run() as run:
            raise ValueError("oops")

    runs = store.list_runs(agent_id="test")
    assert runs[0].status == "failed"
    assert runs[0].error == "oops"


def test_tracer_log_tool_call(tracer, store):
    with tracer.run() as run:
        tracer.log_tool_call("calculator", {"expression": "2+2"})
        tracer.log_tool_result("calculator", 4)

    saved = store.get(run.run_id)
    calls = saved.get_tool_calls()
    assert len(calls) == 1
    assert calls[0].tool_name == "calculator"


def test_tracer_optimized_prompt(tracer, store):
    store.save_optimized_prompt("test", "Better prompt!", score=0.9)
    result = tracer.get_optimized_prompt(fallback="Fallback")
    assert result == "Better prompt!"


def test_tracer_fallback_prompt(tracer):
    result = tracer.get_optimized_prompt(fallback="My fallback")
    assert result == "My fallback"


def test_tracer_current_run(tracer):
    assert tracer.current_run is None
    with tracer.run() as run:
        assert tracer.current_run is run
    assert tracer.current_run is None


def test_trace_decorator():
    store = RunStore(db_path=":memory:")

    @trace(agent_id="decorated", store=store, reward_fn=lambda r: 1.0 if "ok" in r else 0.0)
    def my_agent(question: str) -> str:
        return f"ok: answer to {question}"

    result = my_agent("What is Python?")
    assert "ok" in result

    runs = store.list_runs(agent_id="decorated")
    assert len(runs) == 1
    assert runs[0].total_reward == 1.0


def test_trace_decorator_no_reward():
    store = RunStore(db_path=":memory:")

    @trace(agent_id="no_reward", store=store)
    def my_fn(x):
        return x * 2

    my_fn(5)
    runs = store.list_runs(agent_id="no_reward")
    assert len(runs) == 1


def test_multiple_runs(tracer, store):
    for i in range(10):
        with tracer.run() as run:
            tracer.log_prompt(f"Question {i}")
            tracer.log_response(f"Answer {i}")
            run.add_reward(float(i) / 10)

    runs = store.list_runs(agent_id="test")
    assert len(runs) == 10
