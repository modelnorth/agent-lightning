"""BaseAlgorithm — the brain of the training loop."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from ..core.store import LightningStore
    from ..core.models import Rollout, PromptTemplate


class BaseAlgorithm(ABC):
    """
    The Algorithm is the brain: it decides what to run, learns from results,
    and updates resources (prompts, model weights).
    """
    def __init__(self, store: "LightningStore" = None):
        self.store = store

    @abstractmethod
    async def run(self, train_dataset=None, val_dataset=None) -> Dict[str, Any]:
        """Execute the algorithm loop. Returns result dict."""
        ...

    async def on_rollout_complete(self, rollout: "Rollout") -> None:
        """Called when a rollout finishes. Override for online learning."""
        pass

    async def get_resources(self) -> Dict[str, Any]:
        """Return current resources (prompts, model refs) for injection into runners."""
        return {}
