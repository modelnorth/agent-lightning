"""
Agent Lightning Quickstart
==========================

The fastest way to add tracing to any existing agent.
This example uses a fake LLM so you don't need an API key.
"""

import agent_lightning as al

# ── 1. Create a tracer ───────────────────────────────────────────────────────
tracer = al.Tracer(agent_id="my_first_agent")

# ── 2. Simulate a few agent runs ─────────────────────────────────────────────
def fake_llm(prompt: str) -> str:
    """Stand-in for any real LLM call."""
    return f"Response to: {prompt[:40]}..."


for i in range(5):
    user_question = f"Question {i}: What is the capital of France?"

    with tracer.run(tags=["quickstart"]) as run:
        # Log the prompt
        run.add_step(al.Step(
            step_type=al.StepType.PROMPT,
            content=user_question,
        ))

        # Call your LLM
        answer = fake_llm(user_question)

        # Log the response
        run.add_step(al.Step(
            step_type=al.StepType.LLM_RESPONSE,
            content=answer,
        ))

        # Score the run (reward signal)
        reward = 1.0 if "Paris" in answer or i % 2 == 0 else 0.0
        run.add_reward(reward)
        print(f"Run {i+1}: reward={reward:.1f}")

# ── 3. Check what was stored ──────────────────────────────────────────────────
store = tracer.store
stats = store.stats(agent_id="my_first_agent")
print(f"\nStats: {stats}")

runs = store.list_runs(agent_id="my_first_agent")
print(f"\nRecorded {len(runs)} runs")
print(f"Average reward: {stats['avg_reward']:.3f}")
print(f"\nDone! Run `python -m agent_lightning dashboard` to see the UI.")
