"""Tests for RunStore."""

import pytest
from agent_lightning.core.run_store import RunStore
from agent_lightning.core.models import Run, Step, StepType, Reward


@pytest.fixture
def store():
    return RunStore(db_path=":memory:")


def make_run(agent_id="test", reward=1.0, status="completed"):
    run = Run(agent_id=agent_id)
    run.add_step(Step(step_type=StepType.PROMPT, content="hello"))
    run.add_step(Step(step_type=StepType.LLM_RESPONSE, content="world"))
    run.add_reward(reward)
    run.finish(status)
    return run


def test_save_and_get(store):
    run = make_run()
    store.save(run)
    loaded = store.get(run.run_id)
    assert loaded is not None
    assert loaded.run_id == run.run_id
    assert loaded.agent_id == run.agent_id


def test_get_missing(store):
    assert store.get("nonexistent") is None


def test_list_runs_empty(store):
    runs = store.list_runs()
    assert runs == []


def test_list_runs(store):
    for i in range(5):
        store.save(make_run(agent_id="a", reward=float(i)))
    runs = store.list_runs(agent_id="a")
    assert len(runs) == 5


def test_list_runs_filter_status(store):
    store.save(make_run(status="completed"))
    store.save(make_run(status="failed"))
    completed = store.list_runs(status="completed")
    failed = store.list_runs(status="failed")
    assert len(completed) >= 1
    assert len(failed) >= 1


def test_list_runs_min_reward(store):
    store.save(make_run(reward=0.5))
    store.save(make_run(reward=1.5))
    runs = store.list_runs(min_reward=1.0)
    assert all(r.total_reward >= 1.0 for r in runs)


def test_count(store):
    assert store.count() == 0
    store.save(make_run(agent_id="x"))
    store.save(make_run(agent_id="x"))
    store.save(make_run(agent_id="y"))
    assert store.count() == 3
    assert store.count(agent_id="x") == 2
    assert store.count(agent_id="y") == 1


def test_stats(store):
    store.save(make_run(reward=0.0))
    store.save(make_run(reward=1.0))
    stats = store.stats()
    assert stats["total"] == 2
    assert stats["completed"] == 2
    assert abs(stats["avg_reward"] - 0.5) < 0.01
    assert stats["max_reward"] == 1.0
    assert stats["min_reward"] == 0.0


def test_optimized_prompt(store):
    assert store.get_optimized_prompt("agent1") is None
    store.save_optimized_prompt("agent1", "Better prompt!", trainer="PromptTuner", score=0.95)
    result = store.get_optimized_prompt("agent1")
    assert result == "Better prompt!"


def test_optimized_prompt_overwrite(store):
    store.save_optimized_prompt("agent1", "First prompt")
    store.save_optimized_prompt("agent1", "Second prompt")
    assert store.get_optimized_prompt("agent1") == "Second prompt"


def test_step_roundtrip(store):
    run = Run(agent_id="roundtrip")
    run.add_step(Step(
        step_type=StepType.TOOL_CALL,
        tool_name="search",
        tool_args={"query": "test"},
        content="Calling search",
    ))
    run.add_step(Step(
        step_type=StepType.TOOL_RESULT,
        tool_name="search",
        content={"results": ["a", "b"]},
    ))
    run.finish()
    store.save(run)

    loaded = store.get(run.run_id)
    assert len(loaded.steps) == 2
    tc = loaded.steps[0]
    assert tc.step_type == StepType.TOOL_CALL
    assert tc.tool_name == "search"
    assert tc.tool_args == {"query": "test"}


def test_export_jsonl(store, tmp_path):
    for _ in range(3):
        store.save(make_run())
    path = str(tmp_path / "export.jsonl")
    n = store.export_jsonl(path)
    assert n == 3
    import json
    with open(path) as f:
        lines = [json.loads(l) for l in f]
    assert len(lines) == 3
