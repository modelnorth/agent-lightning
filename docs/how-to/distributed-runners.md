# Distributed Runners

Run agents on multiple machines, all connected to a central store.

## Architecture

```
Machine A (store server)          Machines B, C, D (runners)
┌─────────────────────┐           ┌──────────────────┐
│  LightningStore     │ ←── HTTP ─│  RestStoreClient │
│  + REST API server  │           │  + your agent    │
└─────────────────────┘           └──────────────────┘
```

## Start store server

```bash
# Machine A
agl serve --host 0.0.0.0 --port 8000

# Or in Python:
from agent_lightning.core.rest_server import run_rest_server
run_rest_server(host="0.0.0.0", port=8000)
```

## Connect runners

```python
# Machines B, C, D
from agent_lightning.core.rest_server import RestStoreClient
import agent_lightning as al

store  = RestStoreClient("http://store-server:8000")
await store.connect()

tracer = al.AsyncTracer(agent_id="my_agent", store=store)

# Use exactly like local store
with tracer.run() as run:
    tracer.log_prompt("Hello")
    tracer.log_response("World")
    run.add_reward(1.0)
```

## API docs

Once the server is running, visit:
```
http://store-server:8000/docs
```
Full interactive Swagger UI.

## Postgres for production

```python
from agent_lightning.core.store_postgres import PostgresStore

store = PostgresStore("postgresql://user:pass@host/agentlightning")
await store.connect()
# Same API as LightningStore
```
