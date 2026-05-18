# Text-to-SQL Agent — APO Example

Optimizes a system prompt for converting natural language to SQL.

## Run
```bash
python text2sql_agent.py
OPENROUTER_API_KEY=sk-or-... python text2sql_agent.py --model qwen/qwen-2.5-72b-instruct
```

## Dataset
`sql_tasks.jsonl` — 12 SQL tasks: easy SELECT to hard correlated subqueries.

## Smoke test
```bash
python -c "import json; t=[json.loads(l) for l in open('sql_tasks.jsonl')]; print(f'{len(t)} tasks OK')"
```
