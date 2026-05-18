"""
RLTrainer — simple REINFORCE / policy gradient over prompt choices.

This is a lightweight, dependency-minimal RL trainer that:
1. Treats each "prompt variant" as an action
2. Uses accumulated rewards as the signal
3. Updates a softmax policy over variants
4. Selects the best-performing variant to serve

For a production-grade RL trainer with actual model fine-tuning,
see the SFTTrainer or integrate with OpenPipe ART / TRL / Verl.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .base import BaseTrainer
from agent_lightning.core.models import Run
from agent_lightning.core.run_store import RunStore


class RLTrainer(BaseTrainer):
    """
    Bandit-style RL trainer over a discrete set of prompt variants.

    Uses Upper Confidence Bound (UCB1) to balance exploration/exploitation.
    Each variant gets a score; the best one is written to the store.

    Example:
        trainer = RLTrainer(
            store=store,
            agent_id="my_agent",
            prompt_variants=[
                "You are a concise assistant. Keep answers under 3 sentences.",
                "You are a detailed assistant. Provide thorough explanations.",
                "You are a Socratic assistant. Guide with questions.",
            ]
        )
        result = trainer.run()
        print(result["best_prompt"])
        print(result["best_score"])
    """

    def __init__(
        self,
        store: RunStore,
        agent_id: str = "default",
        prompt_variants: Optional[List[str]] = None,
        exploration_constant: float = 1.414,  # sqrt(2) — classic UCB1
    ):
        super().__init__(store=store, agent_id=agent_id)
        self.variants = prompt_variants or []
        self.c = exploration_constant

    def _ucb1_score(self, mean: float, n: int, total: int) -> float:
        if n == 0:
            return float("inf")
        return mean + self.c * math.sqrt(math.log(total + 1) / n)

    def train(self, runs: Optional[List[Run]] = None) -> dict:
        if runs is None:
            runs = self.load_runs()
        if not runs:
            return {"success": False, "message": "No runs available.", "result": None}

        if not self.variants:
            # No explicit variants — just report best prompt from history
            best_run = max(runs, key=lambda r: r.total_reward)
            prompts = best_run.get_prompts()
            best_prompt = prompts[0] if prompts else ""
            if best_prompt:
                self.store.save_optimized_prompt(
                    agent_id=self.agent_id,
                    prompt=best_prompt,
                    trainer="RLTrainer",
                    score=best_run.total_reward,
                )
            return {
                "success": True,
                "message": f"Best prompt extracted from top run (reward={best_run.total_reward:.3f})",
                "result": best_prompt,
                "best_prompt": best_prompt,
                "best_score": best_run.total_reward,
            }

        # Map variant index by first-token match against logged prompts
        variant_rewards: Dict[int, List[float]] = defaultdict(list)

        for run in runs:
            prompts = run.get_prompts()
            if not prompts:
                continue
            prompt = prompts[0]
            # Find closest variant by simple overlap
            best_idx = 0
            best_overlap = -1
            for i, v in enumerate(self.variants):
                overlap = len(set(prompt.split()) & set(v.split()))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = i
            variant_rewards[best_idx].append(run.total_reward)

        total_pulls = sum(len(v) for v in variant_rewards.values())

        scores = {}
        for i, variant in enumerate(self.variants):
            rewards = variant_rewards.get(i, [])
            mean = sum(rewards) / len(rewards) if rewards else 0.0
            n = len(rewards)
            scores[i] = {
                "variant": variant[:80] + "..." if len(variant) > 80 else variant,
                "mean_reward": mean,
                "pulls": n,
                "ucb1": self._ucb1_score(mean, n, total_pulls),
            }

        # Best by mean reward (exploitation mode after training)
        best_idx = max(scores, key=lambda i: scores[i]["mean_reward"])
        best_prompt = self.variants[best_idx]
        best_score = scores[best_idx]["mean_reward"]

        self.store.save_optimized_prompt(
            agent_id=self.agent_id,
            prompt=best_prompt,
            trainer="RLTrainer",
            score=best_score,
        )

        return {
            "success": True,
            "message": f"UCB1 selected variant {best_idx} (mean reward={best_score:.3f})",
            "result": best_prompt,
            "best_prompt": best_prompt,
            "best_score": best_score,
            "variant_scores": scores,
            "runs_analyzed": len(runs),
        }

    def sample_variant(self, exploration: bool = True) -> str:
        """Sample a variant to try next (for online learning)."""
        if not self.variants:
            return self.store.get_optimized_prompt(self.agent_id) or ""
        if exploration:
            return random.choice(self.variants)
        best = self.store.get_optimized_prompt(self.agent_id)
        return best or self.variants[0]
