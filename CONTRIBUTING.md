# Contributing to Agent Lightning

## Quick start

```bash
git clone https://github.com/modelNorth/agent-lightning
cd agent-lightning
pip install -e ".[dev]"
python examples/quickstart.py   # verify setup
```

## Running tests

```bash
pytest tests/ -v
```

## Project structure

```
agent_lightning/
├── core/          # tracer, store, models — zero external deps
├── backends/      # LLM provider adapters
├── algorithms/    # APO and custom training algorithms
├── orchestrator/  # autonomous training loop + triggers + evaluator
├── otel/          # OpenTelemetry span export
├── cli/           # agl command-line interface
└── dashboard/     # web UI (FastAPI + embedded HTML)

contrib/           # community recipes
examples/          # runnable demos with datasets
tests/             # test suite
scripts/           # dev tooling
docs/              # mkdocs documentation
```

## Adding a new backend

1. Create `agent_lightning/backends/my_provider.py`
2. Subclass `LLMBackend` and implement `async complete()`
3. Add to `agent_lightning/backends/__init__.py`
4. Add tests in `tests/test_v2_backends.py`
5. Add an entry in `docs/how-to/connect-my-provider.md`

## Adding a new algorithm

1. Create `agent_lightning/algorithms/my_algo.py`
2. Subclass `BaseAlgorithm` and implement `async run()`
3. Return `{"success": bool, "best_prompt": str, "best_score": float}`
4. Add to `agent_lightning/algorithms/__init__.py`
5. Add a usage example in `examples/custom_algorithm/`

## Adding a contrib recipe

Drop a `.py` file in `contrib/recipes/`. No approval needed for contrib.
Include a module docstring explaining what it does and how to use it.

## Commit style

```
feat: add PostgresStore backend
fix: triplet reward ordering in get_triplets
docs: add distributed runners how-to
test: add APO beam search edge cases
chore: bump version to 0.3.0
```

## Pull request checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] New feature has tests
- [ ] Examples still run (`python examples/quickstart.py`)
- [ ] No breaking changes to public API (or documented in CHANGELOG)
