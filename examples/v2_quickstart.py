"""
Agent Lightning v0.2 — Quickstart

Shows the full autonomous loop in ~40 lines:
  1. Trace your agent with AsyncTracer
  2. Orchestrator fires APO after N runs
  3. Better prompt deployed automatically
  4. Agent picks it up with get_optimized_prompt()

No real API key needed — uses mock LLM + mock reward.
Set OPENROUTER_API_KEY to use real optimization.
"""

import asyncio
import os
import agent_lightning as al

async def main():
    # ── Setup ────────────────────────────────────────────────────────────────
    store  = al.LightningStore(":memory:")
    tracer = al.AsyncTracer(agent_id="demo_agent", store=store)

    # ── Mock LLM + reward ────────────────────────────────────────────────────
    def mock_llm(prompt: str, system: str) -> str:
        if "concise" in system.lower():
            return f"Short: {prompt[:20]}"
        return f"Here is a detailed answer about: {prompt}"

    def score(response: str) -> float:
        # Reward shorter, direct answers
        return max(0.0, 1.0 - len(response) / 200)

    # ── Collect 12 runs ──────────────────────────────────────────────────────
    questions = [f"Question {i}: explain topic {i}" for i in range(12)]
    system_prompt = tracer.get_optimized_prompt("You are a helpful assistant.")

    print("Collecting runs...")
    for q in questions:
        with tracer.run() as run:
            tracer.log_prompt(q)
            answer = mock_llm(q, system_prompt)
            tracer.log_response(answer)
            run.add_reward(score(answer))

    stats = store.stats(agent_id="demo_agent")
    print(f"Collected {stats['total']} runs, avg_reward={stats['avg_reward']:.3f}")

    # ── Run APO (with real key) or show dry-run ───────────────────────────────
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        print("\nRunning APO optimization via OpenRouter...")
        apo = al.APO(
            store=store,
            agent_id="demo_agent",
            gradient_backend=al.OpenRouterBackend.fast(api_key=api_key),
            eval_backend=al.OpenRouterBackend.balanced(api_key=api_key),
            initial_prompt="You are a helpful assistant.",
            beam_width=3,
            beam_rounds=2,
        )
        result = await apo.run()
        print(f"Best score: {result['best_score']:.4f}")
        print(f"Optimized prompt: {result['best_prompt'][:120]}...")
    else:
        print("\n[Dry run] Set OPENROUTER_API_KEY to run real APO optimization")
        store.save_optimized_prompt("demo_agent",
            "You are a concise assistant. Keep answers under 2 sentences.",
            trainer="demo", score=0.85)

    # ── Agent picks up better prompt automatically ────────────────────────────
    better_prompt = tracer.get_optimized_prompt("You are a helpful assistant.")
    print(f"\nAgent now using: '{better_prompt[:60]}...'")

    final_answer = mock_llm("Explain Python", better_prompt)
    print(f"Answer quality: {score(final_answer):.3f}")
    print("\n✅ Full loop complete. Run `agl dashboard` to see runs.")

asyncio.run(main())
