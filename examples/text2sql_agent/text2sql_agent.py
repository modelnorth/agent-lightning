"""
Text-to-SQL Agent — APO with structured database tasks.

Optimizes a system prompt for converting natural language questions
into correct SQL queries. Uses exact-match + keyword scoring.

Mirrors Microsoft's evaluation on Text2SQL tasks.

Usage:
    python text2sql_agent.py
    OPENROUTER_API_KEY=sk-or-... python text2sql_agent.py --model qwen/qwen-2.5-72b-instruct
"""

import asyncio
import json
import os
import re
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import agent_lightning as al


def load_tasks(path=None):
    path = path or Path(__file__).parent / "sql_tasks.jsonl"
    return [json.loads(l) for l in open(path)]


def score_sql(response: str, task: dict) -> float:
    """Score SQL response: keyword presence + structure."""
    expected = task["answer"].lower()
    response  = response.lower()

    # Extract SQL block if present
    sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL)
    if sql_match:
        response = sql_match.group(1)

    score = 0.0
    # Key SQL keywords present
    keywords = re.findall(r'\b(select|from|where|join|group by|having|order by|limit|avg|sum|count|max|min)\b', expected)
    for kw in keywords:
        if kw in response:
            score += 1.0 / len(keywords) if keywords else 0
    # Table names present
    tables = re.findall(r'from\s+(\w+)', expected)
    for t in tables:
        if t in response:
            score = min(1.0, score + 0.1)
    # Exact match bonus
    if expected.strip() in response.strip():
        score = 1.0
    return min(1.0, score)


class MockSQLBackend(al.LLMBackend):
    def __init__(self):
        self.model = "mock/sql"

    async def complete(self, messages, **kw):
        system = next((m.content for m in messages if m.role == "system"), "")
        user   = next((m.content for m in messages if m.role == "user"), "")
        good   = "schema" in system.lower() or "sql" in system.lower()

        if good:
            # Extract table name heuristically
            tables = re.findall(r'\b([a-z_]+)\(', user)
            table  = tables[0] if tables else "table1"
            sql    = f"SELECT * FROM {table} WHERE id > 0"
        else:
            sql = "I need more information to write that query."

        return al.LLMResponse(
            f"```sql\n{sql}\n```",
            model=self.model, input_tokens=60, output_tokens=25,
        )


async def main(args):
    print("⚡ Agent Lightning — Text-to-SQL APO Example")
    print("=" * 50)

    tasks = load_tasks()
    train, val = tasks[:8], tasks[8:]
    print(f"Dataset: {len(tasks)} SQL tasks ({len(train)} train, {len(val)} val)")

    store  = al.LightningStore(":memory:")
    tracer = al.AsyncTracer(agent_id="sql_agent", store=store)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    backend = al.OpenRouterBackend(api_key=api_key, model=args.model) if api_key else MockSQLBackend()
    print(f"Backend: {'OpenRouter ' + args.model if api_key else 'Mock'}")

    baseline_prompt = "You are a SQL assistant. Convert questions to SQL."

    # Collect baseline runs
    print(f"\nPhase 1: Baseline evaluation...")
    baseline_scores = []
    for task in train:
        msgs = [
            al.LLMMessage("system", baseline_prompt),
            al.LLMMessage("user", f"Schema: {task['schema']}\nQuestion: {task['question']}"),
        ]
        resp  = await backend.complete(msgs, temperature=0.0)
        score = score_sql(resp.content, task)
        baseline_scores.append(score)

        with tracer.run() as run:
            run.add_step(al.Step(step_type=al.StepType.PROMPT, content=task["question"],
                                  metadata={"difficulty": task["difficulty"]}))
            run.add_step(al.Step(step_type=al.StepType.LLM_RESPONSE, content=resp.content))
            run.add_reward(score, label="sql_correctness")

    baseline_avg = sum(baseline_scores) / len(baseline_scores)
    print(f"  Baseline: {baseline_avg:.4f}")

    # APO
    print(f"\nPhase 2: APO optimization (beam_width={args.beam_width}, rounds={args.rounds})...")

    def reward_fn(response, task_input):
        # task_input is the question string; approximate scoring
        has_select = "select" in response.lower()
        has_from   = "from" in response.lower()
        return 1.0 if (has_select and has_from) else 0.2

    apo = al.APO(
        store=store, agent_id="sql_agent",
        gradient_backend=backend, eval_backend=backend,
        initial_prompt=baseline_prompt,
        beam_width=args.beam_width, beam_rounds=args.rounds, branch_factor=2,
        reward_fn=reward_fn,
    )
    result = await apo.run(
        train_dataset=[f"Schema: {t['schema']}\nQuestion: {t['question']}" for t in train],
        val_dataset=[f"Schema: {t['schema']}\nQuestion: {t['question']}" for t in val],
    )

    print(f"  APO done: {result['candidates']} candidates, best_score={result['best_score']:.4f}")
    print(f"  Optimized: {result['best_prompt'][:100]}...")

    # Validate
    print(f"\nPhase 3: Validation...")
    opt_scores = []
    for task in val:
        msgs = [
            al.LLMMessage("system", result["best_prompt"]),
            al.LLMMessage("user", f"Schema: {task['schema']}\nQuestion: {task['question']}"),
        ]
        resp  = await backend.complete(msgs, temperature=0.0)
        score = score_sql(resp.content, task)
        opt_scores.append(score)

    opt_avg = sum(opt_scores) / len(opt_scores)
    print(f"\n{'='*50}")
    print(f"  Baseline:  {baseline_avg:.4f}")
    print(f"  Optimized: {opt_avg:.4f}  ({opt_avg-baseline_avg:+.4f})")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="qwen/qwen-2.5-72b-instruct")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--rounds",     type=int, default=2)
    asyncio.run(main(parser.parse_args()))
