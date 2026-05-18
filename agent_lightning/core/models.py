"""
Core data models for Agent Lightning v0.2.

Extends v0.1 with Rollout/Attempt/Span concepts matching
the Algorithm ↔ Runner ↔ Store loop.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


# ── Step / Span types ────────────────────────────────────────────────────────

class StepType(str, Enum):
    PROMPT       = "prompt"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL    = "tool_call"
    TOOL_RESULT  = "tool_result"
    REWARD       = "reward"
    CUSTOM       = "custom"


class RolloutStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    RETRYING  = "retrying"


# ── Reward ────────────────────────────────────────────────────────────────────

@dataclass
class Reward:
    value: float
    label: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Span (OTel-compatible) ────────────────────────────────────────────────────

@dataclass
class Span:
    """
    OTel-compatible span. Maps 1:1 with our Step but uses OTel naming
    so external tools (Jaeger, Datadog) can consume them directly.
    """
    span_id:      str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id:     str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id:    Optional[str] = None
    name:         str = "agent.step"
    kind:         str = "INTERNAL"          # INTERNAL | CLIENT | SERVER
    start_time:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time:     Optional[datetime] = None
    attributes:   Dict[str, Any] = field(default_factory=dict)
    events:       List[Dict[str, Any]] = field(default_factory=list)
    status:       str = "OK"               # OK | ERROR
    rollout_id:   Optional[str] = None
    attempt_id:   Optional[str] = None
    sequence_id:  Optional[int] = None

    # Agent Lightning semantic fields
    step_type:    Optional[StepType] = None
    content:      Any = None
    model:        Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens:Optional[int] = None
    latency_ms:   Optional[float] = None
    tool_name:    Optional[str] = None
    tool_args:    Optional[Dict[str, Any]] = None
    tool_error:   Optional[str] = None
    reward:       Optional[Reward] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id":       self.span_id,
            "trace_id":      self.trace_id,
            "parent_id":     self.parent_id,
            "name":          self.name,
            "kind":          self.kind,
            "start_time":    self.start_time.isoformat(),
            "end_time":      self.end_time.isoformat() if self.end_time else None,
            "attributes":    self.attributes,
            "events":        self.events,
            "status":        self.status,
            "rollout_id":    self.rollout_id,
            "attempt_id":    self.attempt_id,
            "sequence_id":   self.sequence_id,
            "step_type":     self.step_type.value if self.step_type else None,
            "content":       self.content,
            "model":         self.model,
            "input_tokens":  self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms":    self.latency_ms,
            "tool_name":     self.tool_name,
            "tool_args":     self.tool_args,
            "tool_error":    self.tool_error,
            "reward":        {"value": self.reward.value, "label": self.reward.label} if self.reward else None,
        }


# ── Step (v0.1 compat alias) ──────────────────────────────────────────────────

@dataclass
class Step:
    step_id:      str = field(default_factory=lambda: str(uuid.uuid4()))
    step_type:    StepType = StepType.CUSTOM
    content:      Any = None
    metadata:     Dict[str, Any] = field(default_factory=dict)
    timestamp:    datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms:   Optional[float] = None
    model:        Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens:Optional[int] = None
    temperature:  Optional[float] = None
    tool_name:    Optional[str] = None
    tool_args:    Optional[Dict[str, Any]] = None
    tool_error:   Optional[str] = None
    reward:       Optional[Reward] = None

    def to_span(self, rollout_id: str = None, attempt_id: str = None) -> Span:
        """Convert Step to OTel Span."""
        return Span(
            name=f"agent.{self.step_type.value}",
            step_type=self.step_type,
            content=self.content,
            model=self.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latency_ms=self.latency_ms,
            tool_name=self.tool_name,
            tool_args=self.tool_args,
            tool_error=self.tool_error,
            reward=self.reward,
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            attributes=self.metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id":       self.step_id,
            "step_type":     self.step_type.value,
            "content":       self.content,
            "metadata":      self.metadata,
            "timestamp":     self.timestamp.isoformat(),
            "latency_ms":    self.latency_ms,
            "model":         self.model,
            "input_tokens":  self.input_tokens,
            "output_tokens": self.output_tokens,
            "temperature":   self.temperature,
            "tool_name":     self.tool_name,
            "tool_args":     self.tool_args,
            "tool_error":    self.tool_error,
            "reward":        {"value": self.reward.value, "label": self.reward.label, "metadata": self.reward.metadata} if self.reward else None,
        }


# ── Attempt ───────────────────────────────────────────────────────────────────

@dataclass
class Attempt:
    """A single execution attempt of a Rollout. One rollout can have multiple attempts."""
    attempt_id:  str = field(default_factory=lambda: str(uuid.uuid4()))
    rollout_id:  str = ""
    status:      RolloutStatus = RolloutStatus.PENDING
    worker_id:   Optional[str] = None
    started_at:  Optional[datetime] = None
    finished_at: Optional[datetime] = None
    reward:      Optional[float] = None
    error:       Optional[str] = None
    spans:       List[Span] = field(default_factory=list)

    def start(self, worker_id: str = None):
        self.status = RolloutStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)
        self.worker_id = worker_id

    def finish(self, reward: float = None, error: str = None):
        self.status = RolloutStatus.FAILED if error else RolloutStatus.COMPLETED
        self.finished_at = datetime.now(timezone.utc)
        self.reward = reward
        self.error = error

    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds() * 1000
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_id":  self.attempt_id,
            "rollout_id":  self.rollout_id,
            "status":      self.status.value,
            "worker_id":   self.worker_id,
            "started_at":  self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "reward":      self.reward,
            "error":       self.error,
            "duration_ms": self.duration_ms,
            "n_spans":     len(self.spans),
        }


# ── Rollout ───────────────────────────────────────────────────────────────────

@dataclass
class Rollout:
    """
    A unit of work. The algorithm enqueues rollouts; runners dequeue and execute them.
    One rollout → one or more Attempts (retry on failure).
    """
    rollout_id:   str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id:     str = "default"
    task:         Any = None              # input data for this rollout
    resources:    Dict[str, Any] = field(default_factory=dict)  # injected by algorithm
    status:       RolloutStatus = RolloutStatus.PENDING
    attempts:     List[Attempt] = field(default_factory=list)
    max_attempts: int = 3
    tags:         List[str] = field(default_factory=list)
    metadata:     Dict[str, Any] = field(default_factory=dict)
    created_at:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    mode:         str = "train"           # train | val

    def new_attempt(self, worker_id: str = None) -> Attempt:
        attempt = Attempt(rollout_id=self.rollout_id)
        attempt.start(worker_id=worker_id)
        self.attempts.append(attempt)
        self.status = RolloutStatus.RUNNING
        return attempt

    def current_attempt(self) -> Optional[Attempt]:
        return self.attempts[-1] if self.attempts else None

    def can_retry(self) -> bool:
        failed = sum(1 for a in self.attempts if a.status == RolloutStatus.FAILED)
        return failed < self.max_attempts

    @property
    def best_reward(self) -> Optional[float]:
        rewards = [a.reward for a in self.attempts if a.reward is not None]
        return max(rewards) if rewards else None

    @property
    def is_done(self) -> bool:
        return self.status in (RolloutStatus.COMPLETED, RolloutStatus.FAILED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rollout_id":   self.rollout_id,
            "agent_id":     self.agent_id,
            "status":       self.status.value,
            "attempts":     [a.to_dict() for a in self.attempts],
            "max_attempts": self.max_attempts,
            "tags":         self.tags,
            "metadata":     self.metadata,
            "created_at":   self.created_at.isoformat(),
            "mode":         self.mode,
            "best_reward":  self.best_reward,
        }


# ── Resource ──────────────────────────────────────────────────────────────────

@dataclass
class PromptTemplate:
    """A tunable prompt template resource."""
    template:   str = ""
    version:    int = 0
    score:      Optional[float] = None
    metadata:   Dict[str, Any] = field(default_factory=dict)

    def format(self, **kwargs) -> str:
        try:
            return self.template.format(**kwargs)
        except KeyError:
            return self.template

    def to_dict(self) -> Dict[str, Any]:
        return {"template": self.template, "version": self.version,
                "score": self.score, "metadata": self.metadata}


# ── Run (v0.1 compat) ─────────────────────────────────────────────────────────

@dataclass
class Run:
    """v0.1 compatibility. Wraps a Rollout in the simpler original API."""
    run_id:          str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id:        str = "default"
    session_id:      Optional[str] = None
    steps:           List[Step] = field(default_factory=list)
    rewards:         List[Reward] = field(default_factory=list)
    metadata:        Dict[str, Any] = field(default_factory=dict)
    tags:            List[str] = field(default_factory=list)
    created_at:      datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at:     Optional[datetime] = None
    status:          str = "running"
    error:           Optional[str] = None
    optimized_prompt:Optional[str] = None

    def add_step(self, step: Step) -> None:
        self.steps.append(step)

    def add_reward(self, value: float, label: str = "default", **metadata) -> None:
        self.rewards.append(Reward(value=value, label=label, metadata=metadata))

    def finish(self, status: str = "completed") -> None:
        self.status = status
        self.finished_at = datetime.now(timezone.utc)

    @property
    def total_reward(self) -> float:
        return sum(r.value for r in self.rewards) if self.rewards else 0.0

    @property
    def duration_ms(self) -> Optional[float]:
        if self.finished_at and self.created_at:
            return (self.finished_at - self.created_at).total_seconds() * 1000
        return None

    @property
    def total_tokens(self) -> int:
        return sum((s.input_tokens or 0) + (s.output_tokens or 0) for s in self.steps)

    def get_prompts(self) -> List[str]:
        return [s.content for s in self.steps if s.step_type == StepType.PROMPT and isinstance(s.content, str)]

    def get_tool_calls(self) -> List[Step]:
        return [s for s in self.steps if s.step_type == StepType.TOOL_CALL]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id, "agent_id": self.agent_id,
            "session_id": self.session_id,
            "steps": [s.to_dict() for s in self.steps],
            "rewards": [{"value": r.value, "label": r.label, "metadata": r.metadata} for r in self.rewards],
            "metadata": self.metadata, "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "status": self.status, "error": self.error,
            "optimized_prompt": self.optimized_prompt,
            "total_reward": self.total_reward,
            "duration_ms": self.duration_ms,
            "total_tokens": self.total_tokens,
        }
