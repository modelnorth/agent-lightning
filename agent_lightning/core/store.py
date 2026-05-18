"""
LightningStore v0.2 — central hub for rollouts, spans, resources, and task queue.

Async-first. Supports:
  - SQLite (default, zero deps)
  - In-memory (tests)
  - Postgres (optional, via asyncpg)
  - REST client/server (distributed runners)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import (
    Rollout, Attempt, Span, PromptTemplate,
    RolloutStatus, Run, Step, StepType, Reward
)

log = logging.getLogger(__name__)


class LightningStore:
    """
    Async task queue + span storage + resource registry.

    The single source of truth shared by Algorithm and Runner.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            d = Path.home() / ".agent_lightning"
            d.mkdir(parents=True, exist_ok=True)
            db_path = str(d / "store_v2.db")
        self.db_path = db_path
        self._lock = threading.Lock()
        self._memory_conn: Optional[sqlite3.Connection] = None
        if db_path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
        self._rollout_event = None  # lazy init
        self._init_db()

    # ── connection ────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        if self._memory_conn:
            return self._memory_conn
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _exec(self, q: str, p=None, commit=False):
        p = p or []
        if self._memory_conn:
            cur = self._memory_conn.execute(q, p)
            if commit:
                self._memory_conn.commit()
            return cur
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(q, p)
            rows = cur.fetchall()
            return rows

    def _exec_fetch(self, q: str, p=None):
        p = p or []
        if self._memory_conn:
            return self._memory_conn.execute(q, p).fetchall()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(q, p).fetchall()

    def _exec_one(self, q: str, p=None):
        p = p or []
        if self._memory_conn:
            return self._memory_conn.execute(q, p).fetchone()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(q, p).fetchone()

    def _exec_write(self, q: str, p=None):
        p = p or []
        with self._lock:
            if self._memory_conn:
                self._memory_conn.execute(q, p)
                self._memory_conn.commit()
            else:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(q, p)

    def _init_db(self):
        with self._lock:
            c = self._conn()
            c.executescript("""
                CREATE TABLE IF NOT EXISTS rollouts (
                    rollout_id TEXT PRIMARY KEY,
                    agent_id   TEXT NOT NULL,
                    status     TEXT NOT NULL DEFAULT 'pending',
                    mode       TEXT NOT NULL DEFAULT 'train',
                    data       JSON NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_rollouts_status  ON rollouts(status);
                CREATE INDEX IF NOT EXISTS idx_rollouts_agent   ON rollouts(agent_id);
                CREATE INDEX IF NOT EXISTS idx_rollouts_mode    ON rollouts(mode);

                CREATE TABLE IF NOT EXISTS spans (
                    span_id    TEXT PRIMARY KEY,
                    rollout_id TEXT,
                    attempt_id TEXT,
                    agent_id   TEXT,
                    step_type  TEXT,
                    seq        INTEGER,
                    data       JSON NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_spans_rollout ON spans(rollout_id);
                CREATE INDEX IF NOT EXISTS idx_spans_agent   ON spans(agent_id);

                CREATE TABLE IF NOT EXISTS resources (
                    key        TEXT PRIMARY KEY,
                    agent_id   TEXT NOT NULL,
                    kind       TEXT NOT NULL DEFAULT 'prompt',
                    data       JSON NOT NULL,
                    version    INTEGER NOT NULL DEFAULT 0,
                    score      REAL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id     TEXT PRIMARY KEY,
                    agent_id   TEXT NOT NULL,
                    session_id TEXT,
                    data       JSON NOT NULL,
                    status     TEXT NOT NULL DEFAULT 'running',
                    total_reward REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_runs_agent  ON runs(agent_id);
                CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
            """)
            if self._memory_conn:
                c.commit()
            else:
                c.close()

    # ── Rollout queue ─────────────────────────────────────────────────────────

    async def enqueue_rollout(self, rollout: Rollout) -> Rollout:
        """Add a rollout to the pending queue."""
        self._exec_write(
            "INSERT OR REPLACE INTO rollouts (rollout_id,agent_id,status,mode,data,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
            (rollout.rollout_id, rollout.agent_id, rollout.status.value, rollout.mode,
             json.dumps(rollout.to_dict()), rollout.created_at.isoformat(),
             datetime.now(timezone.utc).isoformat())
        )
        # Signal waiting runners
        pass  # runners poll

        log.debug("enqueued rollout %s", rollout.rollout_id)
        return rollout

    async def enqueue_batch(self, rollouts: List[Rollout]) -> List[Rollout]:
        for r in rollouts:
            await self.enqueue_rollout(r)
        return rollouts

    async def dequeue_rollout(self, agent_id: str = None, timeout: float = 5.0) -> Optional[Rollout]:
        """Pop a pending rollout. Blocks up to timeout seconds."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            row = self._dequeue_one(agent_id)
            if row:
                return self._deserialize_rollout(json.loads(row["data"]))
            await asyncio.sleep(0.1)
        return None

    def _dequeue_one(self, agent_id: str = None) -> Optional[Any]:
        with self._lock:
            clause = "AND agent_id = ?" if agent_id else ""
            params = ["pending"] + ([agent_id] if agent_id else [])
            row = self._exec_one(
                f"SELECT rollout_id, data FROM rollouts WHERE status=? {clause} ORDER BY created_at ASC LIMIT 1",
                params
            )
            if row:
                self._exec_write(
                    "UPDATE rollouts SET status='running', updated_at=? WHERE rollout_id=?",
                    (datetime.now(timezone.utc).isoformat(), row["rollout_id"])
                )
            return row

    async def update_rollout_status(self, rollout_id: str, status: RolloutStatus,
                                     reward: float = None, error: str = None) -> None:
        data_row = self._exec_one("SELECT data FROM rollouts WHERE rollout_id=?", (rollout_id,))
        if not data_row:
            return
        rollout = self._deserialize_rollout(json.loads(data_row["data"]))
        rollout.status = status
        if rollout.attempts:
            rollout.attempts[-1].status = status
            rollout.attempts[-1].reward = reward
            rollout.attempts[-1].error = error
            rollout.attempts[-1].finished_at = datetime.now(timezone.utc)
        self._exec_write(
            "UPDATE rollouts SET status=?, data=?, updated_at=? WHERE rollout_id=?",
            (status.value, json.dumps(rollout.to_dict()),
             datetime.now(timezone.utc).isoformat(), rollout_id)
        )

    async def get_rollout(self, rollout_id: str) -> Optional[Rollout]:
        row = self._exec_one("SELECT data FROM rollouts WHERE rollout_id=?", (rollout_id,))
        return self._deserialize_rollout(json.loads(row["data"])) if row else None

    async def list_rollouts(self, agent_id: str = None, status: str = None,
                             mode: str = None, limit: int = 100) -> List[Rollout]:
        clauses, params = [], []
        if agent_id: clauses.append("agent_id=?"); params.append(agent_id)
        if status:   clauses.append("status=?");   params.append(status)
        if mode:     clauses.append("mode=?");     params.append(mode)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._exec_fetch(
            f"SELECT data FROM rollouts {where} ORDER BY created_at DESC LIMIT ?",
            params + [limit]
        )
        return [self._deserialize_rollout(json.loads(r["data"])) for r in rows]

    async def wait_for_rollouts(self, rollout_ids: List[str], timeout: float = 300.0) -> List[Rollout]:
        """Block until all rollout_ids are done (completed or failed)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            rollouts = [await self.get_rollout(rid) for rid in rollout_ids]
            if all(r and r.is_done for r in rollouts if r):
                return [r for r in rollouts if r]
            await asyncio.sleep(0.5)
        log.warning("wait_for_rollouts timed out for %d rollouts", len(rollout_ids))
        return [r for r in (await self.list_rollouts(limit=9999)) if r.rollout_id in rollout_ids]

    # ── Spans ─────────────────────────────────────────────────────────────────

    async def add_span(self, span: Span) -> Span:
        self._exec_write(
            "INSERT OR REPLACE INTO spans (span_id,rollout_id,attempt_id,agent_id,step_type,seq,data,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (span.span_id, span.rollout_id, span.attempt_id,
             span.attributes.get("agent_id", "unknown"),
             span.step_type.value if span.step_type else None,
             span.sequence_id, json.dumps(span.to_dict()),
             span.start_time.isoformat())
        )
        return span

    async def query_spans(self, rollout_id: str = None, agent_id: str = None,
                          step_type: str = None, limit: int = 500) -> List[Span]:
        clauses, params = [], []
        if rollout_id: clauses.append("rollout_id=?"); params.append(rollout_id)
        if agent_id:   clauses.append("agent_id=?");   params.append(agent_id)
        if step_type:  clauses.append("step_type=?");  params.append(step_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._exec_fetch(
            f"SELECT data FROM spans {where} ORDER BY seq ASC, created_at ASC LIMIT ?",
            params + [limit]
        )
        return [self._deserialize_span(json.loads(r["data"])) for r in rows]

    async def get_triplets(self, rollout_id: str) -> List[Tuple[str, str, float]]:
        """
        Extract (prompt, response, reward) triplets for RL/SFT.
        The Adapter pattern — transforms raw spans into learning signals.
        """
        spans = await self.query_spans(rollout_id=rollout_id)
        triplets = []
        prompt = response = None
        reward = 0.0
        for s in spans:
            if s.step_type == StepType.REWARD and s.reward:
                reward = s.reward.value
            elif s.step_type == StepType.PROMPT and isinstance(s.content, str):
                prompt = s.content
            elif s.step_type == StepType.LLM_RESPONSE and isinstance(s.content, str):
                response = s.content
                if prompt and response:
                    triplets.append((prompt, response, reward))
                    prompt = response = None
                    reward = 0.0
        return triplets

    # ── Resources ─────────────────────────────────────────────────────────────

    async def set_resource(self, key: str, agent_id: str, value: Any,
                           kind: str = "prompt", score: float = None) -> None:
        data = value.to_dict() if hasattr(value, "to_dict") else {"value": value}
        version_row = self._exec_one("SELECT version FROM resources WHERE key=? AND agent_id=?", (key, agent_id))
        version = (version_row["version"] + 1) if version_row else 0
        self._exec_write(
            "INSERT OR REPLACE INTO resources (key,agent_id,kind,data,version,score,updated_at) VALUES (?,?,?,?,?,?,?)",
            (key, agent_id, kind, json.dumps(data), version, score,
             datetime.now(timezone.utc).isoformat())
        )
        log.info("resource updated: key=%s agent=%s version=%d score=%s", key, agent_id, version, score)

    async def get_resource(self, key: str, agent_id: str = None) -> Optional[Dict[str, Any]]:
        if agent_id:
            row = self._exec_one("SELECT data, kind FROM resources WHERE key=? AND agent_id=?", (key, agent_id))
        else:
            row = self._exec_one("SELECT data, kind FROM resources WHERE key=? ORDER BY updated_at DESC LIMIT 1", (key,))
        if not row:
            return None
        return json.loads(row["data"])

    async def get_latest_prompt(self, agent_id: str, key: str = "main_prompt") -> Optional[str]:
        row = self._exec_one(
            "SELECT data FROM resources WHERE key=? AND agent_id=? AND kind='prompt' ORDER BY version DESC LIMIT 1",
            (key, agent_id)
        )
        if row:
            d = json.loads(row["data"])
            return d.get("template") or d.get("prompt") or d.get("value")
        return None

    async def list_resources(self, agent_id: str = None) -> List[Dict[str, Any]]:
        if agent_id:
            rows = self._exec_fetch("SELECT key,kind,version,score,data FROM resources WHERE agent_id=?", (agent_id,))
        else:
            rows = self._exec_fetch("SELECT key,kind,version,score,data FROM resources", [])
        return [{"key": r["key"], "kind": r["kind"], "version": r["version"],
                 "score": r["score"], **json.loads(r["data"])} for r in rows]

    # ── v0.1 Run compat ───────────────────────────────────────────────────────

    def save(self, run: Run) -> None:
        with self._lock:
            if self._memory_conn:
                self._memory_conn.execute(
                    "INSERT OR REPLACE INTO runs (run_id,agent_id,session_id,data,status,total_reward,created_at,finished_at) VALUES (?,?,?,?,?,?,?,?)",
                    (run.run_id, run.agent_id, run.session_id, json.dumps(run.to_dict()),
                     run.status, run.total_reward, run.created_at.isoformat(),
                     run.finished_at.isoformat() if run.finished_at else None)
                )
                self._memory_conn.commit()
            else:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO runs (run_id,agent_id,session_id,data,status,total_reward,created_at,finished_at) VALUES (?,?,?,?,?,?,?,?)",
                        (run.run_id, run.agent_id, run.session_id, json.dumps(run.to_dict()),
                         run.status, run.total_reward, run.created_at.isoformat(),
                         run.finished_at.isoformat() if run.finished_at else None)
                    )

    def get(self, run_id: str) -> Optional[Run]:
        row = self._exec_one("SELECT data FROM runs WHERE run_id=?", (run_id,))
        return self._deserialize_run(json.loads(row["data"])) if row else None

    def list_runs(self, agent_id=None, status=None, limit=100, offset=0,
                  min_reward=None, max_reward=None) -> List[Run]:
        clauses, params = [], []
        if agent_id:   clauses.append("agent_id=?");        params.append(agent_id)
        if status:     clauses.append("status=?");          params.append(status)
        if min_reward is not None: clauses.append("total_reward>=?"); params.append(min_reward)
        if max_reward is not None: clauses.append("total_reward<=?"); params.append(max_reward)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._exec_fetch(
            f"SELECT data FROM runs {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params + [limit, offset]
        )
        return [self._deserialize_run(json.loads(r["data"])) for r in rows]

    def count(self, agent_id=None) -> int:
        clause = "WHERE agent_id=?" if agent_id else ""
        params = [agent_id] if agent_id else []
        row = self._exec_one(f"SELECT COUNT(*) FROM runs {clause}", params)
        return row[0] if row else 0

    def stats(self, agent_id=None) -> Dict[str, Any]:
        clause = "WHERE agent_id=?" if agent_id else ""
        params = [agent_id] if agent_id else []
        q = f"""SELECT COUNT(*) as total, AVG(total_reward) as avg_reward,
                MAX(total_reward) as max_reward, MIN(total_reward) as min_reward,
                SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failed
                FROM runs {clause}"""
        row = self._exec_one(q, params)
        if row:
            return dict(zip(["total","avg_reward","max_reward","min_reward","completed","failed"], row))
        return {"total":0,"avg_reward":0,"max_reward":0,"min_reward":0,"completed":0,"failed":0}

    def save_optimized_prompt(self, agent_id, prompt, trainer="unknown", score=None):
        self._exec_write(
            "INSERT OR REPLACE INTO resources (key,agent_id,kind,data,version,score,updated_at) VALUES (?,?,?,?,?,?,?)",
            ("main_prompt", agent_id, "prompt",
             json.dumps({"template": prompt, "trainer": trainer}),
             0, score, datetime.now(timezone.utc).isoformat())
        )

    def get_optimized_prompt(self, agent_id) -> Optional[str]:
        row = self._exec_one(
            "SELECT data FROM resources WHERE key='main_prompt' AND agent_id=?", (agent_id,)
        )
        if row:
            d = json.loads(row["data"])
            return d.get("template") or d.get("prompt")
        return None

    def export_jsonl(self, path: str, agent_id=None) -> int:
        runs = self.list_runs(agent_id=agent_id, limit=999999)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in runs:
                f.write(json.dumps(r.to_dict()) + "\n")
        return len(runs)

    # ── stats for dashboard ───────────────────────────────────────────────────

    async def rollout_stats(self, agent_id: str = None) -> Dict[str, Any]:
        clause = "WHERE agent_id=?" if agent_id else ""
        params = [agent_id] if agent_id else []
        row = self._exec_one(
            f"SELECT COUNT(*) as total,"
            f" SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as completed,"
            f" SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failed,"
            f" SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) as pending,"
            f" SUM(CASE WHEN status='running' THEN 1 ELSE 0 END) as running"
            f" FROM rollouts {clause}", params
        )
        if row:
            return dict(zip(["total","completed","failed","pending","running"], row))
        return {"total":0,"completed":0,"failed":0,"pending":0,"running":0}

    # ── deserializers ─────────────────────────────────────────────────────────

    def _deserialize_rollout(self, d: Dict) -> Rollout:
        attempts = []
        for ad in d.get("attempts", []):
            a = Attempt(
                attempt_id=ad["attempt_id"], rollout_id=ad["rollout_id"],
                status=RolloutStatus(ad["status"]), worker_id=ad.get("worker_id"),
                reward=ad.get("reward"), error=ad.get("error"),
            )
            if ad.get("started_at"):  a.started_at  = datetime.fromisoformat(ad["started_at"])
            if ad.get("finished_at"): a.finished_at = datetime.fromisoformat(ad["finished_at"])
            attempts.append(a)
        r = Rollout(
            rollout_id=d["rollout_id"], agent_id=d["agent_id"],
            status=RolloutStatus(d["status"]),
            task=d.get("task"), resources=d.get("resources", {}),
            attempts=attempts, max_attempts=d.get("max_attempts", 3),
            tags=d.get("tags", []), metadata=d.get("metadata", {}),
            mode=d.get("mode", "train"),
        )
        if d.get("created_at"): r.created_at = datetime.fromisoformat(d["created_at"])
        return r

    def _deserialize_span(self, d: Dict) -> Span:
        s = Span(
            span_id=d["span_id"], trace_id=d.get("trace_id", d["span_id"]),
            parent_id=d.get("parent_id"), name=d.get("name", "agent.step"),
            kind=d.get("kind", "INTERNAL"), status=d.get("status", "OK"),
            rollout_id=d.get("rollout_id"), attempt_id=d.get("attempt_id"),
            sequence_id=d.get("sequence_id"), attributes=d.get("attributes", {}),
            events=d.get("events", []), content=d.get("content"),
            model=d.get("model"), input_tokens=d.get("input_tokens"),
            output_tokens=d.get("output_tokens"), latency_ms=d.get("latency_ms"),
            tool_name=d.get("tool_name"), tool_args=d.get("tool_args"),
            tool_error=d.get("tool_error"),
        )
        if d.get("step_type"): s.step_type = StepType(d["step_type"])
        if d.get("start_time"): s.start_time = datetime.fromisoformat(d["start_time"])
        if d.get("end_time"):   s.end_time   = datetime.fromisoformat(d["end_time"])
        if d.get("reward"):
            rd = d["reward"]; s.reward = Reward(value=rd["value"], label=rd.get("label","default"))
        return s

    def _deserialize_run(self, d: Dict) -> Run:
        steps = []
        for sd in d.get("steps", []):
            step = Step(
                step_id=sd["step_id"], step_type=StepType(sd["step_type"]),
                content=sd.get("content"), metadata=sd.get("metadata", {}),
                latency_ms=sd.get("latency_ms"), model=sd.get("model"),
                input_tokens=sd.get("input_tokens"), output_tokens=sd.get("output_tokens"),
                tool_name=sd.get("tool_name"), tool_args=sd.get("tool_args"),
            )
            if sd.get("reward"):
                rd = sd["reward"]; step.reward = Reward(rd["value"], rd["label"])
            steps.append(step)
        rewards = [Reward(r["value"], r["label"], r.get("metadata",{})) for r in d.get("rewards",[])]
        run = Run(
            run_id=d["run_id"], agent_id=d["agent_id"], session_id=d.get("session_id"),
            steps=steps, rewards=rewards, metadata=d.get("metadata",{}),
            tags=d.get("tags",[]), status=d.get("status","completed"),
            error=d.get("error"), optimized_prompt=d.get("optimized_prompt"),
        )
        if d.get("created_at"):  run.created_at  = datetime.fromisoformat(d["created_at"])
        if d.get("finished_at"): run.finished_at = datetime.fromisoformat(d["finished_at"])
        return run


# ── v0.1 compat alias ─────────────────────────────────────────────────────────
RunStore = LightningStore
