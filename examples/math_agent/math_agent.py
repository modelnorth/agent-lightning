"""
Math Agent — Real APO example with structured dataset.

This is a production-style example showing the full loop:
  1. Load real math tasks from math_tasks.jsonl
  2. Run a math-solving agent (via OpenRouter or mock)
  3. Score each answer for correctness
  4. Run APO to optimize the system prompt
  5. Measure improvement on held-out validation set

Mirrors Microsoft's room_selector_apo.py but with OpenRouter
and model-agnostic backends.

Usage:
    # Mock run (no API key needed):
    python math_agent.py

    # Real run with OpenRouter:
    OPENROUTER_API_KEY=sk-or-... python math_agent.py

    # With specific model:
    OPENROUTER_API_KEY=sk-or-... python math_agent.py --model qwen/qwen3-8b:free
"""

import asyncio
import json
import os
import re
import sys
import argparse
from pathlib import Path

# Add parent to path for local dev
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import agent_lightning as al


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_tasks(path: str = None) -> list:
    path = path or Path(__file__).parent / "math_tasks.jsonl"
    tasks = []
    with open(path) as f:
        for line in f:
            tasks.append(json.loads(line.strip()))
    return tasks


# ── Scoring ────────────────────────────────────────────────────────────────────

def extract_number(text: str) -> str:
    """Pull the final number/fraction from a response."""
    text = text.strip()
    # Look for boxed answer: \boxed{42}
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m: return m.group(1).strip()
    # Last number in response
    numbers = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', text)
    return numbers[-1] if numbers else text.split('\n')[-1].strip()


def score_response(response: str, task: dict) -> float:
    """
    Score 0.0–1.0. Full credit for exact match, partial for close numeric.
    """
    expected = str(task["answer"]).strip().lower()
    got       = extract_number(response).lower()

    # Exact string match
    if got == expected:
        return 1.0

    # Numeric closeness
    try:
        exp_num = float(expected.replace("/", "/"))
        got_num = float(got)
        if abs(exp_num - got_num) < 0.01:
            return 1.0
        if abs(exp_num - got_num) / max(abs(exp_num), 1) < 0.05:
            return 0.7
    except (ValueError, ZeroDivisionError):
        pass

    # Partial: expected appears in response
    if expected in response.lower():
        return 0.8

    return 0.0


# ── Mock LLM (no API key) ──────────────────────────────────────────────────────

class MockMathBackend(al.LLMBackend):
    """
    Deterministic mock that solves easy math correctly,
    struggles on hard problems without good prompting.
    """
    EASY_ANSWERS = {
        "36": "36", "5": "5", "80": "80", "5040": "5040",
        "15": "15", "4": "4", "39": "39", "80cm": "80",
        "5050": "5050", "120": "120",
    }

    def __init__(self, good_prompt_keywords=None):
        self.model = "mock/math"
        self.good_keywords = good_prompt_keywords or ["step", "show", "reasoning"]

    async def complete(self, messages, temperature=0.0, max_tokens=512, **kw):
        system = next((m.content for m in messages if m.role == "system"), "")
        user   = next((m.content for m in messages if m.role == "user"), "")

        is_good_prompt = any(kw in system.lower() for kw in self.good_keywords)

        # Simulate better accuracy with better prompts
        import re, random
        numbers = re.findall(r'\d+', user)
        if numbers and is_good_prompt:
            # Good prompt → attempt to solve
            n = int(numbers[0]) if numbers else 10
            answer = str(n * 2 - 4) if "%" not in user else str(int(n * 0.15))
            return al.LLMResponse(
                f"Let me work through this step by step.\nThe answer is {answer}.",
                model=self.model, input_tokens=50, output_tokens=20,
            )
        else:
            # Bad prompt → vague response
            return al.LLMResponse(
                "This is a math problem. The result could be many things.",
                model=self.model, input_tokens=40, output_tokens=15,
            )


# ── Agent ──────────────────────────────────────────────────────────────────────

async def run_agent(backend, system_prompt: str, task: dict) -> tuple:
    """Run the math agent on one task. Returns (response, score)."""
    messages = [
        al.LLMMessage("system", system_prompt),
        al.LLMMessage("user",   task["question"]),
    ]
    resp  = await backend.complete(messages, temperature=0.0, max_tokens=512)
    score = score_response(resp.content, task)
    return resp.content, score


async def evaluate_prompt(backend, prompt: str, tasks: list) -> float:
    """Evaluate a prompt on a list of tasks, return mean score."""
    scores = []
    for task in tasks:
        _, score = await run_agent(backend, prompt, task)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args):
    print("⚡ Agent Lightning — Math Agent APO Example")
    print("=" * 50)

    # Load dataset
    tasks     = load_tasks()
    train     = tasks[:14]
    val       = tasks[14:]
    print(f"Dataset: {len(tasks)} tasks ({len(train)} train, {len(val)} val)")

    # Setup store + tracer
    store  = al.LightningStore(":memory:")
    tracer = al.AsyncTracer(agent_id="math_agent", store=store)

    # Setup backend
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        backend = al.OpenRouterBackend(
            api_key=api_key,
            model=args.model,
            fallback_models=["qwen/qwen3-8b:free", "meta-llama/llama-3.1-8b-instruct:free"],
        )
        print(f"Backend: OpenRouter ({args.model})")
    else:
        backend = MockMathBackend()
        print("Backend: Mock (set OPENROUTER_API_KEY for real inference)")

    # Baseline prompt
    baseline_prompt = "You are a math assistant. Solve the given problem."

    # ── Phase 1: Collect baseline runs ──────────────────────────────────────
    print(f"\nPhase 1: Collecting baseline runs ({len(train)} tasks)...")
    baseline_scores = []

    for task in train:
        response, score = await run_agent(backend, baseline_prompt, task)
        baseline_scores.append(score)

        with tracer.run() as run:
            run.add_step(al.Step(
                step_type=al.StepType.PROMPT,
                content=task["question"],
                metadata={"task_id": task["id"], "category": task["category"]},
            ))
            run.add_step(al.Step(
                step_type=al.StepType.LLM_RESPONSE,
                content=response,
            ))
            run.add_reward(score, label="correctness",
                           task_id=task["id"], difficulty=task["difficulty"])

    baseline_avg = sum(baseline_scores) / len(baseline_scores)
    print(f"  Baseline avg score: {baseline_avg:.3f}")

    # ── Phase 2: APO optimization ────────────────────────────────────────────
    print(f"\nPhase 2: Running APO optimization...")

    def reward_fn(response: str, task_dict) -> float:
        return score_response(response, task_dict)

    apo = al.APO(
        store=store,
        agent_id="math_agent",
        gradient_backend=backend,
        eval_backend=backend,
        initial_prompt=baseline_prompt,
        beam_width=args.beam_width,
        beam_rounds=args.rounds,
        branch_factor=2,
        reward_fn=reward_fn,
    )

    # Provide train tasks as dataset
    result = await apo.run(
        train_dataset=[t["question"] for t in train],
        val_dataset=[t["question"] for t in val],
    )

    print(f"  APO complete: {result['candidates']} candidates evaluated")
    print(f"  Best score: {result['best_score']:.4f}")
    print(f"\nOptimized prompt:\n  {result['best_prompt'][:120]}...")

    # ── Phase 3: Evaluate optimized prompt ──────────────────────────────────
    print(f"\nPhase 3: Evaluating optimized prompt on val set ({len(val)} tasks)...")
    optimized_prompt = result["best_prompt"]
    optimized_scores = []

    for task in val:
        _, score = await run_agent(backend, optimized_prompt, task)
        optimized_scores.append(score)

    optimized_avg = sum(optimized_scores) / len(optimized_scores)

    # ── Results ──────────────────────────────────────────────────────────────
    improvement = optimized_avg - baseline_avg
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"  Baseline  (val): {baseline_avg:.4f}")
    print(f"  Optimized (val): {optimized_avg:.4f}")
    print(f"  Improvement:     {improvement:+.4f} ({improvement/max(baseline_avg,0.001)*100:+.1f}%)")
    print(f"  Total runs logged: {store.count(agent_id='math_agent')}")

    if improvement > 0:
        print("\n✅ APO improved the math agent!")
    else:
        print("\n⚠️  No improvement this run (try more rounds or a stronger model)")

    # Optimization trace
    if args.verbose:
        print("\nOptimization history:")
        for h in result["history"]:
            score_str = f"{h['score']:.4f}" if h["score"] is not None else "  n/a "
            print(f"  v{h['version']:02d} score={score_str}  {h['template'][:70]}...")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Math Agent APO Example")
    parser.add_argument("--model",      default="qwen/qwen3-8b:free")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--rounds",     type=int, default=2)
    parser.add_argument("--verbose",    action="store_true")
    asyncio.run(main(parser.parse_args()))
