# Math Agent — APO Example

Demonstrates Automatic Prompt Optimization on a structured math dataset.

## What it shows
- Loading tasks from a JSONL dataset
- Scoring responses for correctness
- Running APO beam search to improve the system prompt
- Measuring improvement on a held-out validation set

## Run (no API key needed)
```bash
python math_agent.py
```

## Run with OpenRouter
```bash
OPENROUTER_API_KEY=sk-or-... python math_agent.py
OPENROUTER_API_KEY=sk-or-... python math_agent.py --model anthropic/claude-3-haiku
OPENROUTER_API_KEY=sk-or-... python math_agent.py --beam-width 4 --rounds 3 --verbose
```

## Dataset
`math_tasks.jsonl` — 20 math problems across categories:
percentage, algebra, geometry, calculus, probability, statistics, sequences.

## Expected output
```
Phase 1: Baseline avg score: 0.350
Phase 2: APO complete: 12 candidates evaluated
Phase 3: Optimized avg score: 0.620
Improvement: +0.270 (+77.1%)
```

## Smoke test
```bash
python -c "import json; tasks = [json.loads(l) for l in open('math_tasks.jsonl')]; print(f'{len(tasks)} tasks loaded OK')"
```
