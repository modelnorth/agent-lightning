"""
REST Store Server — expose LightningStore over HTTP.

Enables distributed runners on separate machines/containers to
connect to a central store. Client-Server execution strategy.

Run the server:
    python -m agent_lightning.core.rest_server
    # or
    agl serve --host 0.0.0.0 --port 8000

Connect a remote runner:
    store = RestStoreClient("http://store-server:8000")
    await store.enqueue_rollout(rollout)
    rollout = await store.dequeue_rollout(agent_id="my_agent")
"""
from __future__ import annotations
import json, logging, os
from typing import Any, Dict, List, Optional
from .store import LightningStore
from .models import Rollout, Span, RolloutStatus, PromptTemplate

log = logging.getLogger(__name__)


def create_rest_app(store: LightningStore = None):
    """Create FastAPI app exposing LightningStore over HTTP."""
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("pip install 'agent-lightning[dashboard]'")

    _store = store or LightningStore()
    app = FastAPI(title="Agent Lightning Store API", version="0.3.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    # ── Health ────────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.3.0"}

    # ── Rollouts ──────────────────────────────────────────────────────────────

    @app.post("/rollouts")
    async def enqueue_rollout(data: dict):
        r = _store._deserialize_rollout(data)
        await _store.enqueue_rollout(r)
        return {"rollout_id": r.rollout_id, "status": r.status.value}

    @app.post("/rollouts/batch")
    async def enqueue_batch(data: List[dict]):
        rollouts = [_store._deserialize_rollout(d) for d in data]
        await _store.enqueue_batch(rollouts)
        return {"enqueued": len(rollouts)}

    @app.get("/rollouts/dequeue")
    async def dequeue_rollout(agent_id: Optional[str] = None, timeout: float = 2.0):
        r = await _store.dequeue_rollout(agent_id=agent_id, timeout=timeout)
        if not r:
            return JSONResponse(status_code=204, content={"message": "no rollout available"})
        return r.to_dict()

    @app.get("/rollouts/{rollout_id}")
    async def get_rollout(rollout_id: str):
        r = await _store.get_rollout(rollout_id)
        if not r:
            raise HTTPException(404, f"Rollout {rollout_id} not found")
        return r.to_dict()

    @app.patch("/rollouts/{rollout_id}/status")
    async def update_rollout_status(rollout_id: str, data: dict):
        status = RolloutStatus(data.get("status", "completed"))
        await _store.update_rollout_status(
            rollout_id, status,
            reward=data.get("reward"),
            error=data.get("error"),
        )
        return {"ok": True}

    @app.get("/rollouts")
    async def list_rollouts(
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        mode: Optional[str] = None,
        limit: int = 100,
    ):
        rollouts = await _store.list_rollouts(
            agent_id=agent_id, status=status, mode=mode, limit=limit
        )
        return [r.to_dict() for r in rollouts]

    @app.post("/rollouts/wait")
    async def wait_for_rollouts(data: dict):
        rollout_ids = data.get("rollout_ids", [])
        timeout     = data.get("timeout", 300.0)
        rollouts = await _store.wait_for_rollouts(rollout_ids, timeout=timeout)
        return [r.to_dict() for r in rollouts]

    # ── Spans ─────────────────────────────────────────────────────────────────

    @app.post("/spans")
    async def add_span(data: dict):
        span = _store._deserialize_span(data)
        await _store.add_span(span)
        return {"span_id": span.span_id}

    @app.post("/spans/batch")
    async def add_spans_batch(data: List[dict]):
        for d in data:
            await _store.add_span(_store._deserialize_span(d))
        return {"added": len(data)}

    @app.get("/spans")
    async def query_spans(
        rollout_id: Optional[str] = None,
        agent_id:   Optional[str] = None,
        step_type:  Optional[str] = None,
        limit: int = 500,
    ):
        spans = await _store.query_spans(
            rollout_id=rollout_id, agent_id=agent_id,
            step_type=step_type, limit=limit,
        )
        return [s.to_dict() for s in spans]

    @app.get("/spans/{rollout_id}/triplets")
    async def get_triplets(rollout_id: str):
        triplets = await _store.get_triplets(rollout_id)
        return [{"prompt": t[0], "response": t[1], "reward": t[2]} for t in triplets]

    # ── Resources ─────────────────────────────────────────────────────────────

    @app.put("/resources/{agent_id}/{key}")
    async def set_resource(agent_id: str, key: str, data: dict):
        await _store.set_resource(
            key=key, agent_id=agent_id,
            value=data.get("value", data),
            kind=data.get("kind", "prompt"),
            score=data.get("score"),
        )
        return {"ok": True}

    @app.get("/resources/{agent_id}/{key}")
    async def get_resource(agent_id: str, key: str):
        val = await _store.get_resource(key, agent_id)
        if val is None:
            raise HTTPException(404, f"Resource {key} not found for {agent_id}")
        return val

    @app.get("/resources/{agent_id}")
    async def list_resources(agent_id: str):
        return await _store.list_resources(agent_id=agent_id)

    @app.get("/resources/{agent_id}/prompt/latest")
    async def get_latest_prompt(agent_id: str):
        prompt = await _store.get_latest_prompt(agent_id)
        return {"agent_id": agent_id, "prompt": prompt}

    # ── Stats ─────────────────────────────────────────────────────────────────

    @app.get("/stats/rollouts")
    async def rollout_stats(agent_id: Optional[str] = None):
        return await _store.rollout_stats(agent_id=agent_id)

    @app.get("/stats/runs")
    async def run_stats(agent_id: Optional[str] = None):
        return _store.stats(agent_id=agent_id)

    return app


class RestStoreClient:
    """
    HTTP client that implements the same async API as LightningStore.
    Use this in distributed runners connecting to a remote store server.

    Example:
        # On runner machine:
        store = RestStoreClient("http://store-server:8000")
        await store.connect()
        rollout = await store.dequeue_rollout(agent_id="my_agent")
        # ... execute rollout ...
        await store.update_rollout_status(rollout.rollout_id, RolloutStatus.COMPLETED, reward=0.9)
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self._session = None

    async def connect(self):
        try:
            import aiohttp
        except ImportError:
            raise ImportError("pip install aiohttp")
        import aiohttp
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        resp = await self._get("/health")
        log.info("RestStoreClient connected to %s — %s", self.base_url, resp)

    async def close(self):
        if self._session:
            await self._session.close()

    async def _get(self, path: str, **params) -> Any:
        import aiohttp
        async with self._session.get(f"{self.base_url}{path}", params=params) as r:
            if r.status == 204:
                return None
            r.raise_for_status()
            return await r.json()

    async def _post(self, path: str, data: Any) -> Any:
        async with self._session.post(f"{self.base_url}{path}", json=data) as r:
            r.raise_for_status()
            return await r.json()

    async def _put(self, path: str, data: Any) -> Any:
        async with self._session.put(f"{self.base_url}{path}", json=data) as r:
            r.raise_for_status()
            return await r.json()

    async def _patch(self, path: str, data: Any) -> Any:
        async with self._session.patch(f"{self.base_url}{path}", json=data) as r:
            r.raise_for_status()
            return await r.json()

    async def enqueue_rollout(self, rollout: Rollout) -> Rollout:
        await self._post("/rollouts", rollout.to_dict())
        return rollout

    async def enqueue_batch(self, rollouts):
        await self._post("/rollouts/batch", [r.to_dict() for r in rollouts])
        return rollouts

    async def dequeue_rollout(self, agent_id=None, timeout=5.0):
        params = {"timeout": min(timeout, self.timeout - 1)}
        if agent_id: params["agent_id"] = agent_id
        data = await self._get("/rollouts/dequeue", **params)
        if not data: return None
        from .store import LightningStore
        return LightningStore.__new__(LightningStore)._deserialize_rollout(data)

    async def get_rollout(self, rollout_id):
        data = await self._get(f"/rollouts/{rollout_id}")
        if not data: return None
        from .store import LightningStore
        return LightningStore.__new__(LightningStore)._deserialize_rollout(data)

    async def update_rollout_status(self, rollout_id, status, reward=None, error=None):
        await self._patch(f"/rollouts/{rollout_id}/status",
                          {"status": status.value, "reward": reward, "error": error})

    async def list_rollouts(self, agent_id=None, status=None, mode=None, limit=100):
        params = {"limit": limit}
        if agent_id: params["agent_id"] = agent_id
        if status:   params["status"]   = status
        if mode:     params["mode"]     = mode
        data = await self._get("/rollouts", **params)
        from .store import LightningStore
        s = LightningStore.__new__(LightningStore)
        return [s._deserialize_rollout(d) for d in data]

    async def add_span(self, span):
        await self._post("/spans", span.to_dict())
        return span

    async def query_spans(self, rollout_id=None, agent_id=None, step_type=None, limit=500):
        params = {"limit": limit}
        if rollout_id: params["rollout_id"] = rollout_id
        if agent_id:   params["agent_id"]   = agent_id
        if step_type:  params["step_type"]  = step_type
        data = await self._get("/spans", **params)
        from .store import LightningStore
        s = LightningStore.__new__(LightningStore)
        return [s._deserialize_span(d) for d in data]

    async def get_triplets(self, rollout_id):
        data = await self._get(f"/spans/{rollout_id}/triplets")
        return [(d["prompt"], d["response"], d["reward"]) for d in data]

    async def set_resource(self, key, agent_id, value, kind="prompt", score=None):
        data = value.to_dict() if hasattr(value, "to_dict") else {"value": value}
        await self._put(f"/resources/{agent_id}/{key}", {**data, "kind": kind, "score": score})

    async def get_resource(self, key, agent_id=None):
        try:
            return await self._get(f"/resources/{agent_id or 'default'}/{key}")
        except Exception:
            return None

    async def get_latest_prompt(self, agent_id, key="main_prompt"):
        data = await self._get(f"/resources/{agent_id}/prompt/latest")
        return data.get("prompt") if data else None

    async def rollout_stats(self, agent_id=None):
        params = {}
        if agent_id: params["agent_id"] = agent_id
        return await self._get("/stats/rollouts", **params)

    def save_optimized_prompt(self, agent_id, prompt, trainer="unknown", score=None):
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            self.set_resource("main_prompt", agent_id,
                              {"template": prompt, "trainer": trainer}, score=score)
        )

    def get_optimized_prompt(self, agent_id):
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.get_latest_prompt(agent_id))


def run_rest_server(host="0.0.0.0", port=8000, db_path=None):
    """Start the REST store server."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("pip install uvicorn")
    store = LightningStore(db_path)
    app   = create_rest_app(store)
    print(f"\n⚡ Agent Lightning Store API running at http://{host}:{port}")
    print(f"   Docs: http://{host}:{port}/docs\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_rest_server()
