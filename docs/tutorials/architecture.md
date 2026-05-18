# Architecture

## Overview

```
Your Agent
    │
    ▼
AsyncTracer                   ← records every prompt, tool call, response, reward
    │  emits OTel Spans
    ▼
LightningStore                ← central hub: rollout queue + spans + resources
    │  (SQLite / Postgres / REST)
    │
    ├── Orchestrator          ← autonomous loop: trigger → train → eval → deploy
    │       │
    │       ├── Trigger       ← when to fire (EveryNRuns, Scheduled, Manual...)
    │       ├── Algorithm     ← what to do (APO, BestOfN, custom...)
    │       │       │
    │       │       └── LLMBackend  ← which model (OpenRouter, Ollama, OpenAI...)
    │       └── Evaluator     ← A/B test before deploying
    │
    └── OTelExporter          ← push spans to Jaeger / Datadog / Tempo
```

## Core concepts

### Run / Rollout
One complete agent execution from start to finish.
- **Run** — v0.1 simple API
- **Rollout** — v0.2 with retry/attempt support

### Step / Span
A single traceable unit: a prompt sent, an LLM response, a tool call, a reward.
Spans are OTel-compatible and can be exported to any observability backend.

### Resource
A named, versioned artifact written by the Algorithm and read by the Agent.
The main resource is `main_prompt` — the current best system prompt.

### Algorithm
The brain. Reads rollouts from the store, runs optimization, writes updated
resources back. Pluggable — implement `BaseAlgorithm` for custom logic.

### Orchestrator
The clock. Checks triggers on an interval, fires the Algorithm when they fire,
runs the Evaluator to validate improvements, deploys if better.

## Execution modes

**SharedMemory** (default): single process, threads. Good for dev and single-machine.

**ClientServer**: REST API server exposes LightningStore. Runners on other machines
connect via `RestStoreClient`. Good for distributed multi-agent systems.

## Data flow

```
1. Agent runs → AsyncTracer.run() context
2. Spans streamed to LightningStore.add_span()
3. Orchestrator polls trigger every N seconds
4. Trigger fires → Algorithm.run() called
5. Algorithm reads spans + runs from store
6. Algorithm calls LLMBackend (OpenRouter / Ollama / OpenAI)
7. APO: gradient → edit → eval → beam update (N rounds)
8. Best prompt written to store via set_resource()
9. Evaluator A/B tests old vs new on val dataset
10. If significant improvement → store.save_optimized_prompt()
11. Agent calls get_optimized_prompt() → gets better prompt next run
```
