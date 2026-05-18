# Changelog

## [0.3.0] — 2025-06-01

### Added
- `PostgresStore` — asyncpg-backed production store
- `RestStoreClient` / `create_rest_app` — HTTP store API for distributed runners
- `examples/math_agent/` — real math dataset + full APO example
- `examples/text2sql_agent/` — Text-to-SQL optimization example
- `examples/custom_algorithm/` — BestOfN custom algorithm example
- `contrib/recipes/rag_optimizer.py` — RAG prompt optimization recipe
- `contrib/recipes/chain_of_thought.py` — CoT elicitation recipe
- `scripts/prepare_dataset.py` — dataset normalization tool
- `scripts/benchmark.py` — baseline vs optimized comparison
- GitHub Actions CI/CD workflows
- `RAI_README.md` — Responsible AI documentation
- `CONTRIBUTING.md` and `CHANGELOG.md`
- Full mkdocs documentation site

### Changed
- `pyproject.toml` — added `postgres`, `serve` extras

---

## [0.2.0] — 2025-05-18

### Added
- `OpenRouterBackend` — 100+ models, fallback chains, cost arbitrage
- `MultiModelConsensus` — run optimization across N models simultaneously
- `OllamaBackend` — local offline inference
- `OpenAIBackend`, `LiteLLMBackend`
- `APO` algorithm — textual gradients + beam search, model-agnostic
- `Orchestrator` — autonomous background training loop
- `OrchestratorConfig`, `EveryNRuns`, `Scheduled`, `Manual`, `OnImprovement` triggers
- `Evaluator` — A/B test with Welch t-test significance
- `ABTest` — live traffic splitting
- `AsyncTracer` with `async with tracer.rollout()` and `Hook` callbacks
- `LightningStore` — rollout queue, spans, resources, versioning
- `Rollout`, `Attempt`, `Span`, `PromptTemplate` models
- `OTelExporter` — OpenTelemetry span export
- `agl` CLI — dashboard, train, stats, export, models, rollouts, prompt

---

## [0.1.0] — 2025-05-01

### Added
- `Tracer` context manager + `@trace` decorator
- `RunStore` (SQLite)
- `PromptTuner`, `RLTrainer`, `SFTTrainer`
- `wrap_openai`, `wrap_langchain_llm`
- Basic dashboard (FastAPI + HTML)
