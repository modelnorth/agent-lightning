"""
PostgresStore — asyncpg-backed LightningStore for production deployments.

Use when:
  - Multiple runner processes writing simultaneously
  - > 10k runs/day
  - You need SQL queries on span metadata
  - Horizontal scaling across machines

Install: pip install "agent-lightning[postgres]"

Example:
    store = PostgresStore("postgresql://user:pass@localhost/agentlightning")
    await store.connect()
"""
from __future__ import annotations
import json, logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from .models import (Rollout, Attempt, Span, PromptTemplate,
                     RolloutStatus, Run, Step, StepType, Reward)

log = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rollouts (
    rollout_id  TEXT PRIMARY KEY,
    agent_id    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    mode        TEXT NOT NULL DEFAULT 'train',
    data        JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_rollouts_status  ON rollouts(status);
CREATE INDEX IF NOT EXISTS idx_rollouts_agent   ON rollouts(agent_id);
CREATE INDEX IF NOT EXISTS idx_rollouts_mode    ON rollouts(mode);

CREATE TABLE IF NOT EXISTS spans (
    span_id     TEXT PRIMARY KEY,
    rollout_id  TEXT,
    attempt_id  TEXT,
    agent_id    TEXT,
    step_type   TEXT,
    seq         INTEGER,
    data        JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_spans_rollout ON spans(rollout_id);
CREATE INDEX IF NOT EXISTS idx_spans_agent   ON spans(agent_id);

CREATE TABLE IF NOT EXISTS resources (
    key         TEXT,
    agent_id    TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'prompt',
    data        JSONB NOT NULL,
    version     INTEGER NOT NULL DEFAULT 0,
    score       REAL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (key, agent_id)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    agent_id     TEXT NOT NULL,
    session_id   TEXT,
    data         JSONB NOT NULL,
    status       TEXT NOT NULL DEFAULT 'running',
    total_reward REAL DEFAULT 0.0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_runs_agent  ON runs(agent_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
"""


class PostgresStore:
    """
    Production-grade Postgres backend for LightningStore.
    Drop-in replacement — same async API as LightningStore.

    Example:
        store = PostgresStore("postgresql://user:pass@host/db")
        await store.connect()
        # use exactly like LightningStore
        await store.enqueue_rollout(rollout)
        rollout = await store.dequeue_rollout(agent_id="my_agent")
    """

    def __init__(self, dsn: str):
        self.dsn  = dsn
        self._pool = None

    async def connect(self):
        """Initialize connection pool and create schema."""
        try:
            import asyncpg
        except ImportError:
            raise ImportError("pip install asyncpg")
        self._pool = await asyncpg.create_pool(self.dsn, min_size=2, max_size=10)
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
        log.info("PostgresStore connected: %s", self.dsn.split("@")[-1])

    async def close(self):
        if self._pool:
            await self._pool.close()

    def _p(self): return self._pool

    # ── Rollouts ──────────────────────────────────────────────────────────────

    async def enqueue_rollout(self, rollout: Rollout) -> Rollout:
        async with self._p().acquire() as conn:
            await conn.execute(
                """INSERT INTO rollouts (rollout_id,agent_id,status,mode,data,created_at)
                   VALUES ($1,$2,$3,$4,$5,$6)
                   ON CONFLICT (rollout_id) DO UPDATE SET
                   status=EXCLUDED.status, data=EXCLUDED.data, updated_at=NOW()""",
                rollout.rollout_id, rollout.agent_id, rollout.status.value,
                rollout.mode, json.dumps(rollout.to_dict()),
                rollout.created_at,
            )
        return rollout

    async def enqueue_batch(self, rollouts: List[Rollout]) -> List[Rollout]:
        for r in rollouts:
            await self.enqueue_rollout(r)
        return rollouts

    async def dequeue_rollout(self, agent_id: str = None, timeout: float = 5.0) -> Optional[Rollout]:
        import asyncio
        deadline = __import__('time').time() + timeout
        while __import__('time').time() < deadline:
            row = await self._try_dequeue(agent_id)
            if row:
                return self._deser_rollout(json.loads(row["data"]))
            await asyncio.sleep(0.1)
        return None

    async def _try_dequeue(self, agent_id: str = None):
        clause = "AND agent_id = $2" if agent_id else ""
        params = ["pending"] + ([agent_id] if agent_id else [])
        async with self._p().acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    f"""SELECT rollout_id, data FROM rollouts
                        WHERE status=$1 {clause}
                        ORDER BY created_at ASC LIMIT 1
                        FOR UPDATE SKIP LOCKED""",
                    *params,
                )
                if row:
                    await conn.execute(
                        "UPDATE rollouts SET status='running', updated_at=NOW() WHERE rollout_id=$1",
                        row["rollout_id"],
                    )
        return row

    async def get_rollout(self, rollout_id: str) -> Optional[Rollout]:
        async with self._p().acquire() as conn:
            row = await conn.fetchrow("SELECT data FROM rollouts WHERE rollout_id=$1", rollout_id)
        return self._deser_rollout(json.loads(row["data"])) if row else None

    async def list_rollouts(self, agent_id=None, status=None, mode=None, limit=100) -> List[Rollout]:
        clauses, params, i = [], [], 1
        if agent_id: clauses.append(f"agent_id=${i}"); params.append(agent_id); i+=1
        if status:   clauses.append(f"status=${i}");   params.append(status);   i+=1
        if mode:     clauses.append(f"mode=${i}");     params.append(mode);     i+=1
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        async with self._p().acquire() as conn:
            rows = await conn.fetch(
                f"SELECT data FROM rollouts {where} ORDER BY created_at DESC LIMIT ${i}",
                *params,
            )
        return [self._deser_rollout(json.loads(r["data"])) for r in rows]

    async def update_rollout_status(self, rollout_id, status, reward=None, error=None):
        row = await self.get_rollout(rollout_id)
        if not row: return
        row.status = status
        if row.attempts:
            row.attempts[-1].status = status
            row.attempts[-1].reward = reward
            row.attempts[-1].finished_at = datetime.now(timezone.utc)
        async with self._p().acquire() as conn:
            await conn.execute(
                "UPDATE rollouts SET status=$1, data=$2, updated_at=NOW() WHERE rollout_id=$3",
                status.value, json.dumps(row.to_dict()), rollout_id,
            )

    async def wait_for_rollouts(self, rollout_ids, timeout=300.0):
        import asyncio, time
        deadline = time.time() + timeout
        while time.time() < deadline:
            rollouts = [await self.get_rollout(rid) for rid in rollout_ids]
            if all(r and r.is_done for r in rollouts if r):
                return [r for r in rollouts if r]
            await asyncio.sleep(0.5)
        return []

    # ── Spans ─────────────────────────────────────────────────────────────────

    async def add_span(self, span: Span) -> Span:
        async with self._p().acquire() as conn:
            await conn.execute(
                """INSERT INTO spans (span_id,rollout_id,attempt_id,agent_id,step_type,seq,data)
                   VALUES ($1,$2,$3,$4,$5,$6,$7) ON CONFLICT DO NOTHING""",
                span.span_id, span.rollout_id, span.attempt_id,
                span.attributes.get("agent_id","unknown"),
                span.step_type.value if span.step_type else None,
                span.sequence_id, json.dumps(span.to_dict()),
            )
        return span

    async def query_spans(self, rollout_id=None, agent_id=None, step_type=None, limit=500):
        clauses, params, i = [], [], 1
        if rollout_id: clauses.append(f"rollout_id=${i}"); params.append(rollout_id); i+=1
        if agent_id:   clauses.append(f"agent_id=${i}");   params.append(agent_id);   i+=1
        if step_type:  clauses.append(f"step_type=${i}");  params.append(step_type);  i+=1
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        async with self._p().acquire() as conn:
            rows = await conn.fetch(
                f"SELECT data FROM spans {where} ORDER BY seq ASC, created_at ASC LIMIT ${i}",
                *params,
            )
        return [self._deser_span(json.loads(r["data"])) for r in rows]

    async def get_triplets(self, rollout_id: str) -> List[Tuple[str, str, float]]:
        spans = await self.query_spans(rollout_id=rollout_id)
        triplets, prompt, response, reward = [], None, None, 0.0
        for s in spans:
            if s.step_type == StepType.REWARD and s.reward: reward = s.reward.value
            elif s.step_type == StepType.PROMPT and isinstance(s.content, str): prompt = s.content
            elif s.step_type == StepType.LLM_RESPONSE and isinstance(s.content, str):
                response = s.content
                if prompt and response:
                    triplets.append((prompt, response, reward))
                    prompt = response = None; reward = 0.0
        return triplets

    # ── Resources ─────────────────────────────────────────────────────────────

    async def set_resource(self, key, agent_id, value, kind="prompt", score=None):
        data = value.to_dict() if hasattr(value, "to_dict") else {"value": value}
        async with self._p().acquire() as conn:
            existing = await conn.fetchval(
                "SELECT version FROM resources WHERE key=$1 AND agent_id=$2", key, agent_id
            )
            version = (existing + 1) if existing is not None else 0
            await conn.execute(
                """INSERT INTO resources (key,agent_id,kind,data,version,score,updated_at)
                   VALUES ($1,$2,$3,$4,$5,$6,NOW())
                   ON CONFLICT (key,agent_id) DO UPDATE SET
                   data=EXCLUDED.data, version=EXCLUDED.version,
                   score=EXCLUDED.score, updated_at=NOW()""",
                key, agent_id, kind, json.dumps(data), version, score,
            )

    async def get_resource(self, key, agent_id=None):
        async with self._p().acquire() as conn:
            if agent_id:
                row = await conn.fetchrow(
                    "SELECT data FROM resources WHERE key=$1 AND agent_id=$2", key, agent_id)
            else:
                row = await conn.fetchrow(
                    "SELECT data FROM resources WHERE key=$1 ORDER BY updated_at DESC LIMIT 1", key)
        return json.loads(row["data"]) if row else None

    async def get_latest_prompt(self, agent_id, key="main_prompt"):
        val = await self.get_resource(key, agent_id)
        if val: return val.get("template") or val.get("prompt") or val.get("value")
        return None

    # ── v0.1 Run compat ───────────────────────────────────────────────────────

    async def save_run(self, run: Run):
        async with self._p().acquire() as conn:
            await conn.execute(
                """INSERT INTO runs (run_id,agent_id,session_id,data,status,total_reward,created_at,finished_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                   ON CONFLICT (run_id) DO UPDATE SET data=EXCLUDED.data, status=EXCLUDED.status""",
                run.run_id, run.agent_id, run.session_id,
                json.dumps(run.to_dict()), run.status, run.total_reward,
                run.created_at, run.finished_at,
            )

    async def rollout_stats(self, agent_id=None):
        clause = "WHERE agent_id=$1" if agent_id else ""
        params = [agent_id] if agent_id else []
        async with self._p().acquire() as conn:
            row = await conn.fetchrow(
                f"""SELECT COUNT(*) as total,
                    SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status='failed'    THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN status='pending'   THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN status='running'   THEN 1 ELSE 0 END) as running
                    FROM rollouts {clause}""", *params,
            )
        return dict(row) if row else {}

    def save_optimized_prompt(self, agent_id, prompt, trainer="unknown", score=None):
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            self.set_resource("main_prompt", agent_id,
                              {"template": prompt, "trainer": trainer}, kind="prompt", score=score)
        )

    def get_optimized_prompt(self, agent_id):
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.get_latest_prompt(agent_id))

    # ── deserializers (shared with SQLite store) ──────────────────────────────
    def _deser_rollout(self, d): 
        from .store import LightningStore
        return LightningStore.__new__(LightningStore)._deserialize_rollout(d)

    def _deser_span(self, d):
        from .store import LightningStore
        return LightningStore.__new__(LightningStore)._deserialize_span(d)
