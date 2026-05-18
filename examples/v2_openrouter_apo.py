"""
OpenRouter APO — Automatic Prompt Optimization with 100+ models.

This is the feature that beats Microsoft's implementation:
  - Cheap model (qwen3-8b:free) for gradient computation
  - Strong model (claude-haiku) for final evaluation
  - Multi-model consensus option

Requires: OPENROUTER_API_KEY
"""

import asyncio
import os
import agent_lightning as al

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "sk-or-your-key-here")

# Sample training data — in real use, these come from your RunStore
TRAIN_DATA = [
    "What is Python used for?",
    "Explain machine learning briefly",
    "What is an API?",
    "How does HTTPS work?",
    "What is Docker?",
]

VAL_DATA = [
    "What is a neural network?",
    "Explain REST APIs",
    "What is Git?",
]


def reward_fn(response: str, item: str) -> float:
    """Score: reward concise, relevant answers."""
    if len(response) < 50:
        return 0.2  # too short
    if len(response) > 500:
        return 0.4  # too verbose
    keywords = item.lower().split()
    hits = sum(1 for k in keywords if k in response.lower())
    return min(1.0, 0.5 + hits * 0.1)


async def main():
    store = al.LightningStore()  # defaults to ~/.agent_lightning/store_v2.db

    # ── Cost arbitrage: cheap for gradients, strong for eval ─────────────────
    gradient_backend = al.OpenRouterBackend(
        api_key=OPENROUTER_KEY,
        model="qwen/qwen3-8b:free",            # free tier for bulk ops
        fallback_models=["meta-llama/llama-3.1-8b-instruct:free"],
    )
    eval_backend = al.OpenRouterBackend(
        api_key=OPENROUTER_KEY,
        model="anthropic/claude-3-haiku",       # stronger for evaluation
        fallback_models=["google/gemini-flash-1.5"],
    )

    # ── APO with beam search + textual gradients ──────────────────────────────
    apo = al.APO(
        store=store,
        agent_id="my_agent",
        gradient_backend=gradient_backend,
        eval_backend=eval_backend,
        initial_prompt="You are a helpful assistant. Answer questions clearly.",
        beam_width=4,           # keep top 4 candidates each round
        branch_factor=3,        # 3 gradient edits per parent
        beam_rounds=3,          # 3 rounds of optimization
        reward_fn=reward_fn,
    )

    print("Running APO...")
    print(f"  Gradient model: {gradient_backend.model}")
    print(f"  Eval model:     {eval_backend.model}")
    print(f"  Beam: width={apo.beam_width} rounds={apo.beam_rounds}")
    print()

    result = await apo.run(train_dataset=TRAIN_DATA, val_dataset=VAL_DATA)

    print(f"\n{'─'*60}")
    print(f"✅ APO complete!")
    print(f"   Candidates evaluated: {result['candidates']}")
    print(f"   Best score:           {result['best_score']:.4f}")
    print(f"\nOptimized prompt:")
    print(f"  {result['best_prompt']}")

    # Optimization trace
    print(f"\nOptimization history:")
    for h in result["history"]:
        score_str = f"{h['score']:.4f}" if h['score'] is not None else "  n/a"
        print(f"  v{h['version']:02d} score={score_str}  {h['template'][:60]}...")

    # Show model usage
    print(f"\nToken usage:")
    for model, usage in gradient_backend.usage_stats.items():
        print(f"  {model}: {usage['calls']} calls, {usage['input_tokens']+usage['output_tokens']} tokens")

asyncio.run(main())
