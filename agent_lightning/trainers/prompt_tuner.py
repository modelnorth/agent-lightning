"""
PromptTuner — iterative prompt optimization via LLM reflection.

Algorithm:
  1. Load top-K runs (sorted by reward)
  2. Extract current prompt + outcome examples
  3. Ask an LLM meta-optimizer: "Given these successes and failures, rewrite the prompt to maximize reward"
  4. Evaluate new prompt on held-out runs (optional)
  5. Write best prompt back to the store

Works with any OpenAI-compatible endpoint.
No GPU required. Extremely practical first step.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .base import BaseTrainer
from agent_lightning.core.models import Run
from agent_lightning.core.run_store import RunStore


_OPTIMIZER_SYSTEM = """You are an expert AI prompt engineer.
Your job is to improve a system prompt for an AI agent based on its past run history.

You will be given:
- The current system prompt
- Examples of SUCCESSFUL runs (high reward)
- Examples of FAILED or LOW-REWARD runs

Your task: rewrite the system prompt to maximize performance.

Rules:
- Keep the core intent of the original prompt
- Make it more specific, clearer, and better structured
- Address failure patterns you see in the low-reward runs
- Reinforce what worked in the high-reward runs
- Return ONLY the new prompt text, no explanation
"""


class PromptTuner(BaseTrainer):
    """
    Improve an agent's system prompt using LLM-based reflection.

    Requires an LLM backend (OpenAI-compatible API).

    Example:
        from agent_lightning import RunStore
        from agent_lightning.trainers import PromptTuner

        store = RunStore()
        tuner = PromptTuner(
            store=store,
            agent_id="my_agent",
            current_prompt="You are a helpful assistant.",
            api_key="sk-...",
            model="gpt-4o",
        )
        result = tuner.run()
        print(result["optimized_prompt"])
    """

    def __init__(
        self,
        store: RunStore,
        agent_id: str = "default",
        current_prompt: str = "",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        n_good_examples: int = 5,
        n_bad_examples: int = 5,
        top_reward_percentile: float = 0.5,
    ):
        super().__init__(store=store, agent_id=agent_id)
        self.current_prompt = current_prompt
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.n_good = n_good_examples
        self.n_bad = n_bad_examples
        self.top_percentile = top_reward_percentile

    def _build_examples(self, runs: List[Run]) -> tuple:
        sorted_runs = sorted(runs, key=lambda r: r.total_reward, reverse=True)
        n = len(sorted_runs)
        split = max(1, int(n * self.top_percentile))
        good = sorted_runs[:min(self.n_good, split)]
        bad = sorted_runs[split:][-self.n_bad:] if split < n else []
        return good, bad

    def _format_run(self, run: Run) -> str:
        prompts = run.get_prompts()
        prompt_text = prompts[0][:300] if prompts else "(no prompt logged)"
        tool_calls = run.get_tool_calls()
        tool_summary = ", ".join(t.tool_name for t in tool_calls[:5] if t.tool_name) or "none"
        return (
            f"  Reward: {run.total_reward:.2f}\n"
            f"  Prompt snippet: {prompt_text}\n"
            f"  Tools used: {tool_summary}\n"
            f"  Steps: {len(run.steps)}\n"
        )

    def _call_llm(self, user_message: str) -> str:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required for PromptTuner. "
                "Install with: pip install openai"
            )

        kwargs: Dict[str, Any] = {"api_key": self.api_key or "dummy"}
        if self.base_url:
            kwargs["base_url"] = self.base_url

        client = openai.OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _OPTIMIZER_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=1500,
        )
        return response.choices[0].message.content or ""

    def train(self, runs: Optional[List[Run]] = None) -> dict:
        if runs is None:
            runs = self.load_runs()
        if not runs:
            return {"success": False, "message": "No runs available.", "result": None, "optimized_prompt": self.current_prompt}

        good, bad = self._build_examples(runs)

        good_text = "\n---\n".join(self._format_run(r) for r in good) or "None"
        bad_text = "\n---\n".join(self._format_run(r) for r in bad) or "None"

        user_msg = f"""Current system prompt:
```
{self.current_prompt}
```

HIGH REWARD RUNS (these worked well):
{good_text}

LOW REWARD RUNS (these need improvement):
{bad_text}

Total runs analyzed: {len(runs)}
Average reward: {sum(r.total_reward for r in runs) / len(runs):.3f}

Please write an improved system prompt:"""

        try:
            optimized = self._call_llm(user_msg)
            self.store.save_optimized_prompt(
                agent_id=self.agent_id,
                prompt=optimized,
                trainer="PromptTuner",
            )
            return {
                "success": True,
                "message": f"Prompt optimized from {len(runs)} runs.",
                "result": optimized,
                "optimized_prompt": optimized,
                "runs_analyzed": len(runs),
                "good_examples": len(good),
                "bad_examples": len(bad),
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"LLM call failed: {e}",
                "result": None,
                "optimized_prompt": self.current_prompt,
            }
