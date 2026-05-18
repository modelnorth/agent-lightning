"""
Chain-of-Thought Recipe — optimize prompts to elicit step-by-step reasoning.

Rewards responses that show explicit reasoning steps before giving a final answer.
Works particularly well with math, logic, and coding tasks.

Usage:
    from contrib.recipes.chain_of_thought import CoTOptimizer
    opt = CoTOptimizer(store=store, backend=backend)
    result = await opt.run(tasks=my_tasks)
"""
from __future__ import annotations
import re, sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import agent_lightning as al
from agent_lightning.algorithms.apo import APO


def cot_reward(response: str, item) -> float:
    """
    Reward responses that show chain-of-thought reasoning.
    Looks for: numbered steps, "step", "therefore", "because", "so".
    """
    if not response or len(response) < 30:
        return 0.0

    score = 0.0
    r = response.lower()

    # Numbered steps
    if re.search(r'\b(step\s*\d+|^\d+\.)', r, re.MULTILINE):
        score += 0.3
    # Reasoning connectors
    connectors = ["therefore", "because", "since", "thus", "so we", "this means", "we get"]
    for c in connectors:
        if c in r:
            score += 0.1
            break
    # Final answer marker
    if re.search(r'\b(answer|result|final|=)\b', r):
        score += 0.2
    # Length bonus (CoT tends to be longer)
    if len(response) > 100:
        score += 0.2
    # Actual answer correctness if expected answer provided
    expected = str(item.get("answer", "") if isinstance(item, dict) else "").lower()
    if expected and expected in r:
        score += 0.2

    return min(1.0, score)


class CoTOptimizer:
    """Optimize prompts to elicit chain-of-thought reasoning."""

    DEFAULT_PROMPT = "You are a helpful assistant. Answer questions clearly."

    def __init__(self, store, agent_id="cot_agent", backend=None,
                 beam_width=3, beam_rounds=2):
        self.store       = store
        self.agent_id    = agent_id
        self.backend     = backend
        self.beam_width  = beam_width
        self.beam_rounds = beam_rounds

    async def run(self, tasks: List[Dict]) -> Dict:
        apo = APO(
            store=self.store, agent_id=self.agent_id,
            gradient_backend=self.backend, eval_backend=self.backend,
            initial_prompt=self.DEFAULT_PROMPT,
            beam_width=self.beam_width, beam_rounds=self.beam_rounds,
            reward_fn=cot_reward,
        )
        return await apo.run(
            train_dataset=tasks[:int(len(tasks)*0.8)],
            val_dataset=tasks[int(len(tasks)*0.8):],
        )
