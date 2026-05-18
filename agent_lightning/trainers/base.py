"""
BaseTrainer — all trainers inherit from this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from agent_lightning.core.models import Run
from agent_lightning.core.run_store import RunStore


class BaseTrainer(ABC):
    """
    Base class for all Agent Lightning trainers.

    Trainers read completed runs from a RunStore,
    apply an improvement method, and write results
    (optimized prompts, model artifacts) back to the store.
    """

    def __init__(self, store: RunStore, agent_id: str = "default"):
        self.store = store
        self.agent_id = agent_id

    def load_runs(
        self,
        min_reward: Optional[float] = None,
        limit: int = 500,
    ) -> List[Run]:
        """Load completed runs for this agent."""
        return self.store.list_runs(
            agent_id=self.agent_id,
            status="completed",
            min_reward=min_reward,
            limit=limit,
        )

    @abstractmethod
    def train(self, runs: Optional[List[Run]] = None) -> dict:
        """
        Run the training / optimization loop.

        Returns a dict with at minimum:
            {"success": bool, "message": str, "result": Any}
        """
        ...

    def run(self, min_reward: Optional[float] = None) -> dict:
        """Load runs and train in one call."""
        runs = self.load_runs(min_reward=min_reward)
        if not runs:
            return {"success": False, "message": "No runs found to train on.", "result": None}
        return self.train(runs)
