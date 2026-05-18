"""
Custom Algorithm — How to write your own training algorithm.

Shows how to subclass BaseAlgorithm and plug it into the Orchestrator.
This implements a simple "best-of-N resampling" algorithm:
  1. Load top-K runs by reward
  2. Extract their prompts
  3. Ask the LLM to synthesize a new prompt combining their strengths
  4. Write it back to the store

Swap in any logic you want — this is just one pattern.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import agent_lightning as al
from agent_lightning.algorithms.base_algorithm import BaseAlgorithm
from agent_lightning.backends.base import LLMBackend, LLMMessage


class BestOfNAlgorithm(BaseAlgorithm):
    """
    Synthesizes a new prompt by asking an LLM to combine the best
    prompts from the top-K highest-reward runs.

    Different from APO: no beam search, no gradients.
    Just: "here are the 5 prompts that worked best — make a better one."
    """

    def __init__(self, store, agent_id="default", backend: LLMBackend = None,
                 top_k: int = 5, n_candidates: int = 3):
        super().__init__(store=store)
        self.agent_id    = agent_id
        self.backend     = backend
        self.top_k       = top_k
        self.n_candidates = n_candidates

    async def get_resources(self):
        prompt = self.store.get_optimized_prompt(self.agent_id)
        return {"main_prompt": prompt} if prompt else {}

    async def run(self, train_dataset=None, val_dataset=None):
        if not self.backend:
            return {"success": False, "message": "backend required"}

        # Load top-K runs
        runs = self.store.list_runs(
            agent_id=self.agent_id, status="completed", limit=200
        )
        if not runs:
            return {"success": False, "message": "no runs available"}

        top_runs = sorted(runs, key=lambda r: r.total_reward, reverse=True)[:self.top_k]

        # Extract their prompts
        top_prompts = []
        for run in top_runs:
            prompts = run.get_prompts()
            if prompts:
                top_prompts.append((prompts[0][:200], run.total_reward))

        if not top_prompts:
            return {"success": False, "message": "no prompts found in runs"}

        # Ask LLM to synthesize
        prompt_examples = "\n".join(
            f"Prompt {i+1} (reward={r:.3f}):\n{p}"
            for i, (p, r) in enumerate(top_prompts)
        )

        synthesis_request = f"""Here are the {len(top_prompts)} best-performing prompts for an AI agent,
ranked by reward score. Synthesize a single better prompt that combines their strengths.

{prompt_examples}

Write a single improved prompt that incorporates the best elements of all of these:"""

        candidates = []
        for _ in range(self.n_candidates):
            resp = await self.backend.complete(
                [
                    al.LLMMessage("system", "You are an expert prompt engineer. Return only the prompt text."),
                    al.LLMMessage("user", synthesis_request),
                ],
                temperature=0.8, max_tokens=1000,
            )
            if resp.content:
                candidates.append(resp.content.strip())

        if not candidates:
            return {"success": False, "message": "LLM returned no candidates"}

        # Pick longest (heuristic: more specific = better for this algo)
        best = max(candidates, key=len)

        # Write to store
        await self.store.set_resource(
            "main_prompt", self.agent_id,
            al.PromptTemplate(template=best),
            kind="prompt",
        )
        self.store.save_optimized_prompt(self.agent_id, best, trainer="BestOfN")

        return {
            "success":     True,
            "best_prompt": best,
            "best_score":  top_runs[0].total_reward,
            "message":     f"Synthesized from top-{len(top_prompts)} prompts",
            "candidates":  len(candidates),
        }


# ── Demo ───────────────────────────────────────────────────────────────────────

async def demo():
    print("⚡ Custom Algorithm Demo — BestOfN Synthesis")
    print("=" * 50)

    store  = al.LightningStore(":memory:")
    tracer = al.AsyncTracer(agent_id="custom_demo", store=store)

    # Seed some runs with different prompts
    prompts_and_rewards = [
        ("You are a concise assistant. Keep answers short.", 0.9),
        ("You are a helpful assistant. Always be clear.", 0.7),
        ("You are an expert. Provide detailed technical answers.", 0.8),
        ("You are a teacher. Explain step by step.", 0.85),
        ("Answer questions accurately and briefly.", 0.75),
    ]

    for prompt, reward in prompts_and_rewards:
        with tracer.run() as run:
            run.add_step(al.Step(step_type=al.StepType.PROMPT, content=prompt,
                                  metadata={"role": "system"}))
            run.add_step(al.Step(step_type=al.StepType.LLM_RESPONSE, content="A response."))
            run.add_reward(reward)

    print(f"Seeded {store.count(agent_id='custom_demo')} runs")

    # Mock backend
    class SynthBackend(al.LLMBackend):
        async def complete(self, msgs, **kw):
            return al.LLMResponse(
                "You are a precise and concise expert assistant. Explain step by step "
                "when needed, but keep answers focused and accurate.",
                model="mock"
            )

    # Run custom algorithm
    algo = BestOfNAlgorithm(
        store=store, agent_id="custom_demo",
        backend=SynthBackend(), top_k=3, n_candidates=2,
    )
    result = await algo.run()

    print(f"\nResult: success={result['success']}")
    print(f"Best prompt: {result['best_prompt']}")
    print(f"Source: {result['message']}")

    # Plug into orchestrator
    print("\nPlugging into Orchestrator...")
    orchestrator = al.Orchestrator(
        store=store, algorithm=algo,
        config=al.OrchestratorConfig(
            agent_id="custom_demo",
            trigger=al.EveryNRuns(5),
            auto_deploy=True,
            require_eval=False,
        ),
    )
    print(f"Status: {orchestrator.status}")
    print("\n✅ Custom algorithm plugged in. Start orchestrator with orchestrator.start()")


if __name__ == "__main__":
    asyncio.run(demo())
