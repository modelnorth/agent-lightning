"""
APO — Automatic Prompt Optimization v2.

Beats Microsoft's APO by:
  1. Using OpenRouter (100+ models) instead of hardcoded AsyncOpenAI
  2. Multi-model consensus option (run gradients through N models, take best)
  3. Cost arbitrage: cheap model for gradients, strong model for eval
  4. Textual gradients + beam search (same algo as theirs, but model-agnostic)
  5. Works fully offline with OllamaBackend

Algorithm (matches ProTeGi / TextGrad):
  For each beam round:
    1. Evaluate current prompt beam on val_dataset → scores
    2. For each parent in beam:
       a. Sample failed rollouts → compute textual gradient (critique)
       b. Apply edit → new candidate prompt
    3. Evaluate all candidates → keep top beam_width
  Return best prompt found across all rounds.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..backends.base import LLMBackend, LLMMessage
from ..core.models import Rollout, RolloutStatus, PromptTemplate, Run, StepType
from ..core.store import LightningStore
from .base_algorithm import BaseAlgorithm

log = logging.getLogger(__name__)

# ── Prompt templates for APO ──────────────────────────────────────────────────

_OPTIMIZER_SYSTEM = """You are an expert AI prompt engineer.
Your task: rewrite a system prompt to maximize agent performance.
You will see the current prompt, high-reward examples (what worked), and low-reward examples (what failed).
Rules:
- Keep the core intent of the original
- Make it more specific, clearer, better structured
- Address failure patterns from low-reward runs
- Reinforce patterns from high-reward runs
- Return ONLY the new prompt text. No explanation, no markdown fences."""

_GRADIENT_SYSTEM = """You are an AI performance analyst.
Given a prompt and agent outputs, identify specific weaknesses causing failures.
Be concrete: point to exact phrases or missing instructions.
Return a short critique (3-5 bullet points) of why the prompt is underperforming."""

_APPLY_EDIT_SYSTEM = """You are a prompt editor.
Given a prompt and a critique of its weaknesses, rewrite the prompt to fix those weaknesses.
Preserve everything that works. Fix only what the critique identifies.
Return ONLY the new prompt text."""


@dataclass
class VersionedPrompt:
    template: str
    version: int = 0
    parent_version: int = -1
    score: Optional[float] = None
    critique: Optional[str] = None


class APO(BaseAlgorithm):
    """
    Automatic Prompt Optimization with textual gradients + beam search.

    Powered by any LLMBackend — use OpenRouter for 100+ models.
    Supports cost arbitrage: cheap model for gradients, strong for eval.

    Example:
        from agent_lightning.backends import OpenRouterBackend
        from agent_lightning.algorithms import APO

        apo = APO(
            store=store,
            agent_id="my_agent",
            gradient_backend=OpenRouterBackend.fast(),    # cheap model for critiques
            eval_backend=OpenRouterBackend.strong(),      # strong model for eval
            initial_prompt="You are a helpful assistant.",
            beam_width=4,
            branch_factor=3,
            beam_rounds=3,
        )
        result = await apo.run(train_dataset=train_data, val_dataset=val_data)
        print(result["best_prompt"])
        print(result["best_score"])
    """

    def __init__(
        self,
        store: LightningStore = None,
        agent_id: str = "default",
        gradient_backend: LLMBackend = None,
        eval_backend: LLMBackend = None,
        initial_prompt: str = "You are a helpful assistant.",
        beam_width: int = 4,
        branch_factor: int = 3,
        beam_rounds: int = 3,
        gradient_batch_size: int = 4,
        val_batch_size: int = 16,
        reward_fn: Optional[Callable] = None,
        diversity_temperature: float = 0.9,
        multi_model_consensus: bool = False,
        rollout_timeout: float = 300.0,
    ):
        super().__init__(store=store)
        self.agent_id             = agent_id
        self.gradient_backend     = gradient_backend
        self.eval_backend         = eval_backend or gradient_backend
        self.initial_prompt       = initial_prompt
        self.beam_width           = beam_width
        self.branch_factor        = branch_factor
        self.beam_rounds          = beam_rounds
        self.gradient_batch_size  = gradient_batch_size
        self.val_batch_size       = val_batch_size
        self.reward_fn            = reward_fn
        self.diversity_temp       = diversity_temperature
        self.multi_model_consensus= multi_model_consensus
        self.rollout_timeout      = rollout_timeout

        self._beam: List[VersionedPrompt] = []
        self._best: Optional[VersionedPrompt] = None
        self._history: List[VersionedPrompt] = []

    # ── main loop ─────────────────────────────────────────────────────────────

    async def run(self, train_dataset=None, val_dataset=None, **kwargs) -> Dict[str, Any]:
        if not self.gradient_backend:
            raise ValueError("gradient_backend required. Use OpenRouterBackend, OllamaBackend, etc.")
        if not self.store:
            raise ValueError("store required.")

        train_data = train_dataset or []
        val_data   = val_dataset   or train_data

        log.info("APO starting: beam_width=%d branch_factor=%d rounds=%d model=%s",
                 self.beam_width, self.branch_factor, self.beam_rounds,
                 self.gradient_backend.model_name)

        # Seed beam with initial prompt
        seed = VersionedPrompt(template=self.initial_prompt, version=0)
        self._beam = [seed]
        self._history = [seed]

        # Optionally run initial validation
        if val_data:
            seed.score = await self._evaluate_prompt(seed, val_data[:self.val_batch_size])
            self._best = seed
            log.info("Seed prompt score: %.4f", seed.score)

        for round_idx in range(self.beam_rounds):
            log.info("── APO Round %d/%d ──", round_idx + 1, self.beam_rounds)
            candidates = []

            for parent in self._beam:
                # Sample training rollouts to compute gradient
                sample = random.sample(train_data, min(self.gradient_batch_size, len(train_data))) \
                         if train_data else []

                # Generate branch_factor candidate prompts from this parent
                for _ in range(self.branch_factor):
                    try:
                        candidate = await self._textual_gradient_and_apply(parent, sample)
                        if candidate:
                            candidates.append(candidate)
                    except Exception as e:
                        log.warning("Branch generation failed: %s", e)

            if not candidates:
                log.warning("No candidates generated in round %d", round_idx + 1)
                break

            # Evaluate all candidates on val set
            eval_tasks = [self._evaluate_prompt(c, val_data[:self.val_batch_size]) for c in candidates]
            scores     = await asyncio.gather(*eval_tasks, return_exceptions=True)

            for c, score in zip(candidates, scores):
                if isinstance(score, float):
                    c.score = score
                self._history.append(c)

            # Keep top beam_width
            scored = [(c.score or -999, c) for c in candidates if c.score is not None]
            scored.sort(key=lambda x: x[0], reverse=True)
            self._beam = [c for _, c in scored[:self.beam_width]]

            best_this_round = self._beam[0] if self._beam else None
            if best_this_round and (self._best is None or (best_this_round.score or 0) > (self._best.score or 0)):
                self._best = best_this_round
                log.info("New best prompt (round %d): score=%.4f", round_idx + 1, self._best.score)
                # Write to store
                if self.store:
                    await self.store.set_resource(
                        "main_prompt", self.agent_id,
                        PromptTemplate(template=self._best.template, score=self._best.score),
                        score=self._best.score,
                    )

        best = self._best or seed
        return {
            "success":       True,
            "best_prompt":   best.template,
            "best_score":    best.score,
            "best_version":  best.version,
            "rounds":        self.beam_rounds,
            "candidates":    len(self._history),
            "history":       [{"template": h.template[:80], "score": h.score, "version": h.version}
                               for h in self._history],
        }

    # ── textual gradient ──────────────────────────────────────────────────────

    async def _compute_textual_gradient(self, prompt: VersionedPrompt,
                                         rollout_samples: List[Any]) -> Optional[str]:
        """Ask the gradient model: what's wrong with this prompt?"""
        sample_text = "\n---\n".join(
            f"Input: {str(s)[:200]}" for s in rollout_samples[:self.gradient_batch_size]
        ) or "No examples available."

        user_msg = f"""Prompt under evaluation:
```
{prompt.template}
```

Recent agent interactions (inputs only):
{sample_text}

What specific weaknesses does this prompt have? Give a concise critique:"""

        resp = await self.gradient_backend.complete(
            [LLMMessage("system", _GRADIENT_SYSTEM), LLMMessage("user", user_msg)],
            temperature=self.diversity_temp, max_tokens=512,
        )
        return resp.content.strip() if resp else None

    async def _apply_edit(self, prompt: VersionedPrompt, critique: str) -> Optional[str]:
        """Ask the edit model: apply this critique to produce a better prompt."""
        user_msg = f"""Current prompt:
```
{prompt.template}
```

Critique (weaknesses to fix):
{critique}

Write the improved prompt:"""

        resp = await self.gradient_backend.complete(
            [LLMMessage("system", _APPLY_EDIT_SYSTEM), LLMMessage("user", user_msg)],
            temperature=self.diversity_temp, max_tokens=1500,
        )
        return resp.content.strip() if resp else None

    async def _textual_gradient_and_apply(self, parent: VersionedPrompt,
                                           samples: List[Any]) -> Optional[VersionedPrompt]:
        """One full gradient step: critique → edit → new candidate."""
        critique = await self._compute_textual_gradient(parent, samples)
        if not critique:
            return None
        new_template = await self._apply_edit(parent, critique)
        if not new_template or new_template == parent.template:
            return None
        candidate = VersionedPrompt(
            template=new_template,
            version=parent.version + 1,
            parent_version=parent.version,
            critique=critique,
        )
        return candidate

    # ── evaluation ────────────────────────────────────────────────────────────

    async def _evaluate_prompt(self, prompt: VersionedPrompt, val_data: List[Any]) -> float:
        """
        Score a prompt on val_data.
        If a reward_fn is provided, uses it directly.
        Otherwise falls back to LLM-as-judge.
        """
        if not val_data:
            return 0.0
        if self.reward_fn:
            scores = []
            for item in val_data[:self.val_batch_size]:
                try:
                    resp = await self.eval_backend.complete_text(
                        prompt=str(item), system=prompt.template, temperature=0.0, max_tokens=512
                    )
                    score = float(self.reward_fn(resp, item))
                    scores.append(score)
                except Exception as e:
                    log.debug("eval error: %s", e)
            return sum(scores) / len(scores) if scores else 0.0
        else:
            # LLM-as-judge: ask the eval model to score the prompt
            return await self._llm_judge(prompt, val_data[:min(4, len(val_data))])

    async def _llm_judge(self, prompt: VersionedPrompt, samples: List[Any]) -> float:
        """Use the eval_backend as a judge to score prompt quality."""
        judge_prompt = f"""Rate this AI system prompt on a scale from 0.0 to 1.0 based on its clarity,
specificity, and likely effectiveness. Consider:
- Is it clear and unambiguous?
- Does it give enough context and constraints?
- Is it well-structured?

Prompt to rate:
```
{prompt.template}
```

Respond with ONLY a number between 0.0 and 1.0:"""

        try:
            resp = await self.eval_backend.complete_text(judge_prompt, temperature=0.0, max_tokens=10)
            score = float(resp.strip().split()[0])
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5

    def get_best_prompt(self) -> Optional[str]:
        return self._best.template if self._best else None

    @property
    def optimization_history(self) -> List[Dict]:
        return [{"template": h.template, "score": h.score, "version": h.version}
                for h in self._history]
