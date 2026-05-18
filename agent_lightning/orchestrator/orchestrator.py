"""
Orchestrator — the autonomous training loop.

Ties together: Store ↔ Algorithm ↔ Runner ↔ Evaluator
Runs as a background asyncio task.

Two execution modes (matching Microsoft's architecture):
  - SharedMemory: single process, threads (dev/debug)
  - ClientServer:  REST API server exposing the store (distributed runners)
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .triggers import BaseTrigger, EveryNRuns
from .evaluator import Evaluator
from ..core.store import LightningStore
from ..core.models import PromptTemplate

if TYPE_CHECKING:
    from ..algorithms.base_algorithm import BaseAlgorithm
    from ..backends.base import LLMBackend

log = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    agent_id:          str = "default"
    trigger:           BaseTrigger = field(default_factory=lambda: EveryNRuns(50))
    poll_interval:     float = 5.0        # seconds between trigger checks
    auto_deploy:       bool = True         # deploy if evaluator says better
    require_eval:      bool = True         # run A/B eval before deploying
    min_runs_to_train: int = 10            # don't train until this many runs exist
    max_train_iterations: int = 999999    # safety cap
    log_level:         str = "INFO"


class Orchestrator:
    """
    Autonomous training loop that runs in the background.

    Example:
        orchestrator = Orchestrator(
            store=store,
            algorithm=APO(
                store=store,
                agent_id="my_agent",
                gradient_backend=OpenRouterBackend.fast(api_key="sk-or-..."),
                initial_prompt="You are a helpful assistant.",
            ),
            config=OrchestratorConfig(
                agent_id="my_agent",
                trigger=EveryNRuns(50),
                auto_deploy=True,
            ),
        )
        orchestrator.start()          # non-blocking background thread
        # ... your agent keeps running ...
        orchestrator.stop()
    """

    def __init__(
        self,
        store: LightningStore,
        algorithm: "BaseAlgorithm",
        config: OrchestratorConfig = None,
        evaluator: Optional[Evaluator] = None,
        train_dataset: Optional[List[Any]] = None,
        val_dataset: Optional[List[Any]] = None,
        on_improvement: Optional[Callable[[str, float], None]] = None,
    ):
        self.store          = store
        self.algorithm      = algorithm
        self.config         = config or OrchestratorConfig()
        self.evaluator      = evaluator
        self.train_dataset  = train_dataset or []
        self.val_dataset    = val_dataset or self.train_dataset
        self.on_improvement = on_improvement

        self._running    = False
        self._task: Optional[asyncio.Task] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._iterations = 0
        self._best_score: Optional[float] = None
        self._current_prompt: Optional[str] = None

        logging.basicConfig(level=getattr(logging, config.log_level if config else "INFO"))

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self, blocking: bool = False) -> "Orchestrator":
        """
        Start the orchestrator.
        blocking=False → background thread (typical usage)
        blocking=True  → runs in current event loop (for async scripts)
        """
        self._running = True
        if blocking:
            asyncio.run(self._loop_async())
        else:
            self._thread = threading.Thread(target=self._run_in_thread, daemon=True)
            self._thread.start()
            log.info("⚡ Orchestrator started (background) agent=%s trigger=%s",
                     self.config.agent_id, type(self.config.trigger).__name__)
        return self

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        log.info("Orchestrator stopped. Iterations: %d", self._iterations)

    def _run_in_thread(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._loop_async())
        finally:
            self._loop.close()

    # ── main loop ─────────────────────────────────────────────────────────────

    async def _loop_async(self):
        log.info("Orchestrator loop started")
        while self._running and self._iterations < self.config.max_train_iterations:
            try:
                await asyncio.sleep(self.config.poll_interval)

                # Check run count threshold
                count = self.store.count(agent_id=self.config.agent_id)
                if count < self.config.min_runs_to_train:
                    log.debug("Not enough runs yet (%d/%d)", count, self.config.min_runs_to_train)
                    continue

                # Check trigger
                should = await self.config.trigger.should_fire(self.store, self.config.agent_id)
                if not should:
                    continue

                self._iterations += 1
                log.info("🔥 Orchestrator firing (iteration %d, %d runs available)",
                         self._iterations, count)

                await self._train_iteration()

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Orchestrator loop error: %s", e, exc_info=True)
                await asyncio.sleep(self.config.poll_interval * 2)

    async def _train_iteration(self):
        """One full train → eval → deploy cycle."""
        t0 = time.time()

        # Run algorithm
        try:
            result = await self.algorithm.run(
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
            )
        except Exception as e:
            log.error("Algorithm failed: %s", e, exc_info=True)
            return

        if not result.get("success", False) or not result.get("best_prompt"):
            log.warning("Algorithm returned no result: %s", result.get("message", ""))
            return

        new_prompt = result["best_prompt"]
        new_score  = result.get("best_score")

        log.info("Algorithm done in %.1fs — new_score=%s", time.time()-t0, new_score)

        # Evaluate vs current
        should_deploy = True
        if self.config.require_eval and self.evaluator and self._current_prompt:
            try:
                eval_result = await self.evaluator.evaluate(
                    self._current_prompt, new_prompt, self.val_dataset
                )
                should_deploy = eval_result.winner == "b" and eval_result.significant
                log.info("A/B eval: A=%.4f B=%.4f winner=%s significant=%s deploy=%s",
                         eval_result.score_a, eval_result.score_b,
                         eval_result.winner, eval_result.significant, should_deploy)
            except Exception as e:
                log.warning("Evaluator failed, deploying anyway: %s", e)

        if should_deploy and self.config.auto_deploy:
            self._current_prompt = new_prompt
            self._best_score     = new_score
            self.store.save_optimized_prompt(
                self.config.agent_id, new_prompt,
                trainer="Orchestrator", score=new_score
            )
            log.info("✅ Deployed new prompt (score=%.4f)", new_score or 0)

            if self.on_improvement:
                try:
                    self.on_improvement(new_prompt, new_score or 0.0)
                except Exception as e:
                    log.debug("on_improvement callback error: %s", e)
        else:
            log.info("⏭  Skipped deploy (no significant improvement)")

    # ── manual control ────────────────────────────────────────────────────────

    async def train_now(self) -> Dict[str, Any]:
        """Trigger a training cycle immediately (async)."""
        await self._train_iteration()
        return {"prompt": self._current_prompt, "score": self._best_score}

    def train_sync(self) -> Dict[str, Any]:
        """Trigger training synchronously (blocking)."""
        if self._loop and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._train_iteration(), self._loop)
            future.result(timeout=600)
        else:
            asyncio.run(self._train_iteration())
        return {"prompt": self._current_prompt, "score": self._best_score}

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "running":       self._running,
            "iterations":    self._iterations,
            "best_score":    self._best_score,
            "current_prompt": (self._current_prompt or "")[:80],
            "agent_id":      self.config.agent_id,
            "trigger":       type(self.config.trigger).__name__,
        }
