"""Tests for core data models."""

import pytest
from datetime import datetime, timezone
from agent_lightning.core.models import Run, Step, StepType, Reward


def test_step_defaults():
    s = Step()
    assert s.step_id is not None
    assert s.step_type == StepType.CUSTOM
    assert s.content is None


def test_step_to_dict():
    s = Step(step_type=StepType.PROMPT, content="Hello", model="gpt-4")
    d = s.to_dict()
    assert d["step_type"] == "prompt"
    assert d["content"] == "Hello"
    assert d["model"] == "gpt-4"


def test_step_with_reward():
    r = Reward(value=0.9, label="quality")
    s = Step(step_type=StepType.LLM_RESPONSE, reward=r)
    d = s.to_dict()
    assert d["reward"]["value"] == 0.9
    assert d["reward"]["label"] == "quality"


def test_run_defaults():
    r = Run()
    assert r.run_id is not None
    assert r.agent_id == "default"
    assert r.status == "running"
    assert r.steps == []
    assert r.rewards == []


def test_run_add_step():
    run = Run()
    step = Step(step_type=StepType.PROMPT, content="test")
    run.add_step(step)
    assert len(run.steps) == 1
    assert run.steps[0].content == "test"


def test_run_add_reward():
    run = Run()
    run.add_reward(0.8)
    run.add_reward(0.6)
    assert len(run.rewards) == 2
    assert abs(run.total_reward - 1.4) < 0.001


def test_run_finish():
    run = Run()
    run.finish("completed")
    assert run.status == "completed"
    assert run.finished_at is not None


def test_run_duration():
    run = Run()
    run.finish()
    assert run.duration_ms is not None
    assert run.duration_ms >= 0


def test_run_total_tokens():
    run = Run()
    run.add_step(Step(step_type=StepType.PROMPT, input_tokens=100, output_tokens=0))
    run.add_step(Step(step_type=StepType.LLM_RESPONSE, input_tokens=0, output_tokens=50))
    assert run.total_tokens == 150


def test_run_get_prompts():
    run = Run()
    run.add_step(Step(step_type=StepType.PROMPT, content="Hello"))
    run.add_step(Step(step_type=StepType.LLM_RESPONSE, content="Hi"))
    run.add_step(Step(step_type=StepType.PROMPT, content="Goodbye"))
    prompts = run.get_prompts()
    assert len(prompts) == 2
    assert prompts[0] == "Hello"


def test_run_get_tool_calls():
    run = Run()
    run.add_step(Step(step_type=StepType.TOOL_CALL, tool_name="search"))
    run.add_step(Step(step_type=StepType.TOOL_RESULT, content="results"))
    assert len(run.get_tool_calls()) == 1


def test_run_to_dict():
    run = Run(agent_id="test", tags=["a", "b"])
    run.add_step(Step(step_type=StepType.PROMPT, content="hi"))
    run.add_reward(1.0)
    run.finish()
    d = run.to_dict()
    assert d["agent_id"] == "test"
    assert d["tags"] == ["a", "b"]
    assert len(d["steps"]) == 1
    assert d["total_reward"] == 1.0
    assert d["status"] == "completed"


def test_run_zero_reward():
    run = Run()
    assert run.total_reward == 0.0


def test_step_type_values():
    assert StepType.PROMPT.value == "prompt"
    assert StepType.LLM_RESPONSE.value == "llm_response"
    assert StepType.TOOL_CALL.value == "tool_call"
    assert StepType.TOOL_RESULT.value == "tool_result"
