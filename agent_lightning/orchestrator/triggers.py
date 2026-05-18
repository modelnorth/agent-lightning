"""
Triggers — when should the orchestrator fire a training run?
"""
from __future__ import annotations
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..core.store import LightningStore


class BaseTrigger(ABC):
    @abstractmethod
    async def should_fire(self, store: "LightningStore", agent_id: str) -> bool: ...
    def reset(self): pass


class EveryNRuns(BaseTrigger):
    """Fire after every N completed runs."""
    def __init__(self, n: int = 50):
        self.n = n
        self._last_count = 0

    async def should_fire(self, store, agent_id):
        count = store.count(agent_id=agent_id)
        if count >= self._last_count + self.n:
            self._last_count = count
            return True
        return False

    def reset(self): self._last_count = 0


class Scheduled(BaseTrigger):
    """Fire every N seconds."""
    def __init__(self, interval_seconds: float = 3600):
        self.interval = interval_seconds
        self._last_fire = 0.0

    async def should_fire(self, store, agent_id):
        if time.time() - self._last_fire >= self.interval:
            self._last_fire = time.time()
            return True
        return False


class Manual(BaseTrigger):
    """Fire only when manually triggered via .trigger()."""
    def __init__(self):
        self._triggered = False

    async def should_fire(self, store, agent_id):
        if self._triggered:
            self._triggered = False
            return True
        return False

    def trigger(self):
        self._triggered = True


class OnImprovement(BaseTrigger):
    """Fire when reward drops below threshold — reactive optimization."""
    def __init__(self, min_avg_reward: float = 0.6, window: int = 20):
        self.min_avg_reward = min_avg_reward
        self.window = window

    async def should_fire(self, store, agent_id):
        runs = store.list_runs(agent_id=agent_id, status="completed", limit=self.window)
        if len(runs) < self.window:
            return False
        avg = sum(r.total_reward for r in runs) / len(runs)
        return avg < self.min_avg_reward
