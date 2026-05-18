"""
Offline / Air-Gapped Optimization with Ollama.

Perfect for ModelNorth sovereign deployments — no cloud API needed.
Uses Qwen3-8B locally via Ollama for the full APO loop.

Requires: ollama pull qwen3:8b
"""
import asyncio
import agent_lightning as al

async def main():
    store  = al.LightningStore(":memory:")
    tracer = al.AsyncTracer(agent_id="offline_agent", store=store)

    # Seed some runs
    for i in range(5):
        with tracer.run() as run:
            tracer.log_prompt(f"Explain concept {i}")
            tracer.log_response(f"Here is a detailed explanation of concept {i}...")
            run.add_reward(0.5 + i * 0.05)

    # Optimize entirely locally — no internet required
    local_backend = al.OllamaBackend(
        model="qwen3:8b",                     # or llama3.2, mistral, phi3, etc.
        base_url="http://localhost:11434",
    )

    apo = al.APO(
        store=store,
        agent_id="offline_agent",
        gradient_backend=local_backend,
        eval_backend=local_backend,
        initial_prompt="You are a helpful assistant.",
        beam_width=2,
        beam_rounds=1,
    )

    print("Optimizing with local Ollama (qwen3:8b)...")
    print("(This requires: ollama serve && ollama pull qwen3:8b)")

    try:
        result = await apo.run()
        print(f"\nOptimized prompt: {result['best_prompt']}")
    except Exception as e:
        print(f"Ollama not running: {e}")
        print("Start with: ollama serve")

asyncio.run(main())
