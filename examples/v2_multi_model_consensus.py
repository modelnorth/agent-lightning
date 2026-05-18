"""
Multi-Model Consensus — novel feature not in Microsoft's version.

Run the same prompt optimization through 3 different models simultaneously.
Take the best result.

Why this works better:
- Different models have different blind spots
- Consensus catches what single-model reflection misses
- No extra cost if using free-tier models
"""
import asyncio, os
import agent_lightning as al

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "sk-or-demo")

async def main():
    consensus = al.MultiModelConsensus(
        backends=[
            al.OpenRouterBackend(OPENROUTER_KEY, model="qwen/qwen3-8b:free"),
            al.OpenRouterBackend(OPENROUTER_KEY, model="meta-llama/llama-3.1-8b-instruct:free"),
            al.OpenRouterBackend(OPENROUTER_KEY, model="mistralai/mistral-7b-instruct:free"),
        ],
        judge=al.OpenRouterBackend(OPENROUTER_KEY, model="anthropic/claude-3-haiku"),
    )

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY to run multi-model consensus")
        print("\nWhat it does:")
        print("  1. Sends same prompt to Qwen3-8B, Llama-3.1-8B, Mistral-7B simultaneously")
        print("  2. Each model proposes an improved prompt")
        print("  3. Claude-Haiku judges and picks the best one")
        print("  4. Result stored back to LightningStore")
        return

    current_prompt = "You are a helpful assistant. Answer questions."
    good_examples  = ["Clear concise explanation of Python", "Step-by-step Docker setup guide"]
    bad_examples   = ["Overly verbose answer that misses the point"]

    print("Running multi-model consensus optimization...")
    best = await consensus.optimize_prompt(current_prompt, good_examples, bad_examples)
    print(f"\nConsensus best prompt:\n{best}")

asyncio.run(main())
