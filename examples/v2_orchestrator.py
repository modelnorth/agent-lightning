"""
Autonomous Orchestrator — set it and forget it.

The orchestrator runs in a background thread.
Every 10 runs (configurable), it:
  1. Fires APO via OpenRouter
  2. A/B tests new vs old prompt
  3. Auto-deploys if significantly better
  4. Calls your on_improvement callback

Your agent code stays completely unchanged.
"""

import asyncio
import os
import time
import agent_lightning as al

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")

async def main():
    store  = al.LightningStore(":memory:")
    tracer = al.AsyncTracer(agent_id="auto_agent", store=store)

    # ── Callback when a better prompt is found ────────────────────────────────
    def on_improvement(new_prompt: str, score: float):
        print(f"\n🚀 New prompt deployed! score={score:.4f}")
        print(f"   Preview: {new_prompt[:80]}...")

    # ── Setup Orchestrator ────────────────────────────────────────────────────
    backend = al.OpenRouterBackend.fast(api_key=OPENROUTER_KEY) if OPENROUTER_KEY \
              else None  # demo mode without API key

    if backend:
        apo = al.APO(
            store=store, agent_id="auto_agent",
            gradient_backend=backend,
            initial_prompt="You are a helpful assistant.",
            beam_width=3, beam_rounds=2,
        )
        orchestrator = al.Orchestrator(
            store=store,
            algorithm=apo,
            config=al.OrchestratorConfig(
                agent_id="auto_agent",
                trigger=al.EveryNRuns(n=10),
                auto_deploy=True,
                require_eval=False,       # skip A/B eval for demo speed
                min_runs_to_train=10,
                poll_interval=2.0,
            ),
            on_improvement=on_improvement,
        )
        orchestrator.start()
        print("✅ Orchestrator started in background")
    else:
        orchestrator = None
        print("⚠️  No OPENROUTER_API_KEY — running without live optimization")

    # ── Simulate your agent running normally ──────────────────────────────────
    print("\nAgent running... (generating 20 runs)")

    for i in range(20):
        current_prompt = tracer.get_optimized_prompt("You are a helpful assistant.")
        with tracer.run() as run:
            tracer.log_prompt(f"Question {i}")
            # Simulate response quality improving with better prompts
            is_good_prompt = "concise" in current_prompt.lower() or i > 10
            response = f"{'Short answer' if is_good_prompt else 'Long verbose answer that goes on and on'}: topic {i}"
            tracer.log_response(response)
            run.add_reward(0.9 if is_good_prompt else 0.3)
        print(f"  run {i+1:02d}: reward={run.total_reward:.1f} prompt_len={len(current_prompt)}")
        time.sleep(0.1)

    if orchestrator:
        time.sleep(3)  # Let orchestrator process
        print(f"\nOrchestrator status: {orchestrator.status}")
        orchestrator.stop()

    # Final stats
    stats = store.stats(agent_id="auto_agent")
    print(f"\nFinal stats: {stats}")

asyncio.run(main())
