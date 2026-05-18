"""Tests for trainers."""

import pytest
from agent_lightning.core.run_store import RunStore
from agent_lightning.core.models import Run, Step, StepType
from agent_lightning.trainers.rl_trainer import RLTrainer
from agent_lightning.trainers.sft_trainer import SFTTrainer, run_to_chat_example


@pytest.fixture
def store():
    return RunStore(db_path=":memory:")


def make_run(store, agent_id="test", prompt="You are helpful.", reward=1.0):
    run = Run(agent_id=agent_id)
    run.add_step(Step(step_type=StepType.PROMPT, content=prompt, metadata={"role": "system"}))
    run.add_step(Step(step_type=StepType.PROMPT, content="Hello", metadata={"role": "user"}))
    run.add_step(Step(step_type=StepType.LLM_RESPONSE, content="Hi there!"))
    run.add_reward(reward)
    run.finish()
    store.save(run)
    return run


def test_rl_trainer_no_variants(store):
    make_run(store, reward=0.9)
    make_run(store, reward=0.5)
    trainer = RLTrainer(store=store, agent_id="test")
    result = trainer.run()
    assert result["success"]
    assert result["best_prompt"] is not None


def test_rl_trainer_with_variants(store):
    variants = [
        "You are a concise assistant.",
        "You are a detailed assistant.",
        "You are a Socratic assistant.",
    ]
    make_run(store, prompt=variants[0], reward=0.9)
    make_run(store, prompt=variants[0], reward=0.8)
    make_run(store, prompt=variants[1], reward=0.3)
    make_run(store, prompt=variants[2], reward=0.5)

    trainer = RLTrainer(store=store, agent_id="test", prompt_variants=variants)
    result = trainer.run()
    assert result["success"]
    assert "variant_scores" in result
    assert len(result["variant_scores"]) == 3


def test_rl_trainer_no_runs(store):
    trainer = RLTrainer(store=store, agent_id="empty_agent")
    result = trainer.run()
    assert not result["success"]


def test_sft_trainer_export(store, tmp_path):
    for _ in range(5):
        make_run(store, reward=0.8)

    trainer = SFTTrainer(
        store=store,
        agent_id="test",
        min_reward=0.5,
        upload=False,
        export_path=str(tmp_path / "train.jsonl"),
    )
    result = trainer.run()
    assert result["success"]
    assert result["n_examples"] > 0
    assert result["job_id"] is None

    import json
    with open(result["export_path"]) as f:
        lines = [json.loads(l) for l in f]
    assert len(lines) > 0
    assert "messages" in lines[0]


def test_run_to_chat_example():
    run = Run(agent_id="test")
    run.add_step(Step(step_type=StepType.PROMPT, content="Hello", metadata={"role": "user"}))
    run.add_step(Step(step_type=StepType.LLM_RESPONSE, content="Hi!"))

    ex = run_to_chat_example(run)
    assert ex is not None
    assert len(ex["messages"]) == 2
    assert ex["messages"][0]["role"] == "user"
    assert ex["messages"][1]["role"] == "assistant"


def test_run_to_chat_example_insufficient():
    run = Run(agent_id="test")
    run.add_step(Step(step_type=StepType.PROMPT, content="Lone prompt"))
    ex = run_to_chat_example(run)
    assert ex is None


def test_sft_min_reward_filter(store, tmp_path):
    make_run(store, reward=0.1)  # below threshold
    make_run(store, reward=0.9)  # above threshold

    trainer = SFTTrainer(
        store=store,
        agent_id="test",
        min_reward=0.5,
        upload=False,
        export_path=str(tmp_path / "filtered.jsonl"),
    )
    result = trainer.run()
    assert result["runs_used"] == 1
