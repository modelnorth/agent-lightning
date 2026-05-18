"""
Prompt Optimization Example
=============================

Shows the full loop:
  1. Run agent with baseline prompt
  2. Score each run
  3. Run PromptTuner to get improved prompt
  4. Use improved prompt in next run

No real API key needed for the agent part.
PromptTuner requires an OpenAI-compatible key (or set OPENAI_API_KEY).
"""

import os
import agent_lightning as al
from agent_lightning.trainers import PromptTuner

store = al.RunStore()
tracer = al.Tracer(agent_id="tuning_demo", store=store)

# ── Phase 1: Collect runs with baseline prompt ────────────────────────────────
BASELINE_PROMPT = "You are a helpful assistant. Answer questions clearly."

print("Phase 1: Collecting runs...")

test_questions = [
    ("What is Python?", "programming language", 1.0),
    ("Explain recursion", "calls itself", 1.0),
    ("What is 2+2?", "4", 1.0),
    ("Tell me a story", "once upon", 0.3),  # off-topic, low reward
    ("What is DevOps?", "development operations", 0.8),
]

for question, expected_keyword, base_reward in test_questions:
    with tracer.run(tags=["baseline"]) as run:
        # Log system prompt
        run.add_step(al.Step(
            step_type=al.StepType.PROMPT,
            content=BASELINE_PROMPT,
            metadata={"role": "system"},
        ))
        # Log user message
        run.add_step(al.Step(
            step_type=al.StepType.PROMPT,
            content=question,
            metadata={"role": "user"},
        ))
        # Fake response
        response = f"Here is information about {question.lower()}: {expected_keyword} is relevant here."
        run.add_step(al.Step(
            step_type=al.StepType.LLM_RESPONSE,
            content=response,
        ))
        # Reward based on keyword presence
        reward = base_reward if expected_keyword.lower() in response.lower() else 0.1
        run.add_reward(reward)
        print(f"  Q: {question[:35]:<35} reward={reward:.1f}")

stats = store.stats(agent_id="tuning_demo")
print(f"\nBaseline avg reward: {stats['avg_reward']:.3f}")

# ── Phase 2: Optimize the prompt ─────────────────────────────────────────────
print("\nPhase 2: Running PromptTuner...")

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("  Note: OPENAI_API_KEY not set. Showing dry-run output.")
    print("  Set your key and re-run to get a real optimized prompt.")
    print("\n  What PromptTuner would do:")
    print("  - Load top 3 high-reward runs")
    print("  - Load bottom 2 low-reward runs")
    print("  - Ask GPT to rewrite the system prompt")
    print("  - Save optimized prompt back to the store")
else:
    tuner = PromptTuner(
        store=store,
        agent_id="tuning_demo",
        current_prompt=BASELINE_PROMPT,
        api_key=api_key,
        model="gpt-4o-mini",
    )
    result = tuner.run()
    if result["success"]:
        print(f"\nOptimized prompt:\n{result['optimized_prompt']}")
    else:
        print(f"Tuner error: {result['message']}")

# ── Phase 3: Use optimized prompt in next run ─────────────────────────────────
print("\nPhase 3: Using optimized prompt...")
active_prompt = tracer.get_optimized_prompt(fallback=BASELINE_PROMPT)
print(f"Active prompt (first 80 chars): {active_prompt[:80]}...")
print("\nDone! The agent will automatically use the better prompt next time.")
