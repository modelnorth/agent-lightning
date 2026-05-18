"""
Evaluator — A/B test old vs new prompt before auto-deploying.
"""
from __future__ import annotations
import asyncio, logging, random, math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple
from ..backends.base import LLMBackend, LLMMessage

log = logging.getLogger(__name__)


@dataclass
class EvalResult:
    prompt_a: str
    prompt_b: str
    score_a: float
    score_b: float
    winner: str            # "a" | "b" | "tie"
    p_value: float = 1.0   # statistical significance
    n_samples: int = 0
    significant: bool = False

    @property
    def improvement(self) -> float:
        return self.score_b - self.score_a


class Evaluator:
    """
    Compare two prompts on a dataset and pick the winner.

    Uses t-test approximation for statistical significance.
    Only auto-deploys if improvement is statistically significant.
    """

    def __init__(
        self,
        backend: LLMBackend,
        reward_fn: Callable,
        n_samples: int = 20,
        significance_threshold: float = 0.05,
        min_improvement: float = 0.02,
    ):
        self.backend   = backend
        self.reward_fn = reward_fn
        self.n_samples = n_samples
        self.alpha     = significance_threshold
        self.min_improvement = min_improvement

    async def evaluate(self, prompt_a: str, prompt_b: str,
                        dataset: List[Any]) -> EvalResult:
        """Run both prompts on dataset, return winner."""
        samples = random.sample(dataset, min(self.n_samples, len(dataset)))

        async def score_prompt(prompt: str) -> List[float]:
            scores = []
            tasks = [self._score_one(prompt, item) for item in samples]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, float):
                    scores.append(r)
            return scores

        scores_a, scores_b = await asyncio.gather(score_prompt(prompt_a), score_prompt(prompt_b))

        if not scores_a or not scores_b:
            return EvalResult(prompt_a, prompt_b, 0.0, 0.0, "tie")

        avg_a = sum(scores_a) / len(scores_a)
        avg_b = sum(scores_b) / len(scores_b)
        p = self._ttest_p(scores_a, scores_b)
        significant = p < self.alpha and abs(avg_b - avg_a) >= self.min_improvement
        winner = "b" if avg_b > avg_a else ("a" if avg_a > avg_b else "tie")

        log.info("A/B eval: A=%.4f B=%.4f p=%.4f winner=%s significant=%s",
                 avg_a, avg_b, p, winner, significant)

        return EvalResult(
            prompt_a=prompt_a, prompt_b=prompt_b,
            score_a=avg_a, score_b=avg_b,
            winner=winner, p_value=p,
            n_samples=len(samples), significant=significant,
        )

    async def _score_one(self, prompt: str, item: Any) -> float:
        try:
            resp = await self.backend.complete_text(
                prompt=str(item), system=prompt, temperature=0.0, max_tokens=512
            )
            return float(self.reward_fn(resp, item))
        except Exception as e:
            log.debug("score_one failed: %s", e)
            return 0.0

    @staticmethod
    def _ttest_p(a: List[float], b: List[float]) -> float:
        """Welch's t-test approximation."""
        if len(a) < 2 or len(b) < 2:
            return 1.0
        mean_a = sum(a) / len(a)
        mean_b = sum(b) / len(b)
        var_a  = sum((x - mean_a) ** 2 for x in a) / (len(a) - 1)
        var_b  = sum((x - mean_b) ** 2 for x in b) / (len(b) - 1)
        se     = math.sqrt(var_a / len(a) + var_b / len(b))
        if se == 0:
            return 1.0
        t = abs(mean_a - mean_b) / se
        # Approximate p-value using normal CDF (valid for large N)
        p = 2 * (1 - _normal_cdf(t))
        return max(0.0, min(1.0, p))


def _normal_cdf(x: float) -> float:
    """Approximation of standard normal CDF."""
    t = 1 / (1 + 0.2316419 * abs(x))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    p = 1 - (1 / math.sqrt(2 * math.pi)) * math.exp(-x * x / 2) * poly
    return p if x >= 0 else 1 - p


class ABTest:
    """
    Lightweight A/B test without a dataset.
    Splits live traffic between two prompts and tracks rewards.
    Use with the Orchestrator for online A/B testing.
    """
    def __init__(self, prompt_a: str, prompt_b: str, split: float = 0.5):
        self.prompt_a = prompt_a
        self.prompt_b = prompt_b
        self.split    = split
        self.rewards_a: List[float] = []
        self.rewards_b: List[float] = []

    def sample(self) -> Tuple[str, str]:
        """Returns (prompt, variant_label)."""
        if random.random() < self.split:
            return self.prompt_a, "a"
        return self.prompt_b, "b"

    def record(self, variant: str, reward: float):
        if variant == "a":
            self.rewards_a.append(reward)
        else:
            self.rewards_b.append(reward)

    @property
    def stats(self):
        avg_a = sum(self.rewards_a) / len(self.rewards_a) if self.rewards_a else 0.0
        avg_b = sum(self.rewards_b) / len(self.rewards_b) if self.rewards_b else 0.0
        return {
            "n_a": len(self.rewards_a), "avg_a": avg_a,
            "n_b": len(self.rewards_b), "avg_b": avg_b,
            "winner": "b" if avg_b > avg_a else ("a" if avg_a > avg_b else "tie"),
        }
