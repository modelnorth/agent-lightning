"""
SFTTrainer — Supervised Fine-Tuning data preparation + upload.

This trainer:
1. Loads high-reward runs
2. Converts them to chat fine-tuning format (OpenAI JSONL schema)
3. Optionally uploads to OpenAI fine-tuning API
4. Tracks the fine-tune job ID for retrieval

No local GPU needed — uses cloud fine-tuning.
For local fine-tuning, export JSONL and use with Axolotl / TRL / Unsloth.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Optional

from .base import BaseTrainer
from agent_lightning.core.models import Run, StepType
from agent_lightning.core.run_store import RunStore


def run_to_chat_example(run: Run) -> Optional[Dict[str, Any]]:
    """
    Convert a Run into OpenAI fine-tuning chat format.

    Returns None if the run has no usable prompt/response pair.
    """
    messages = []
    for step in run.steps:
        if step.step_type == StepType.PROMPT:
            role = step.metadata.get("role", "user")
            messages.append({"role": role, "content": str(step.content or "")})
        elif step.step_type == StepType.LLM_RESPONSE:
            messages.append({"role": "assistant", "content": str(step.content or "")})

    if len(messages) < 2:
        return None

    return {"messages": messages}


class SFTTrainer(BaseTrainer):
    """
    Prepare and optionally submit supervised fine-tuning jobs.

    Example:
        trainer = SFTTrainer(
            store=store,
            agent_id="my_agent",
            min_reward=0.8,          # only use high-quality runs
            model="gpt-4o-mini-2024-07-18",
            api_key="sk-...",
            upload=True,             # actually submit to OpenAI
        )
        result = trainer.run()
        print(result["job_id"])      # OpenAI fine-tune job ID
        print(result["export_path"]) # local JSONL file
    """

    def __init__(
        self,
        store: RunStore,
        agent_id: str = "default",
        min_reward: float = 0.0,
        model: str = "gpt-4o-mini-2024-07-18",
        api_key: Optional[str] = None,
        upload: bool = False,
        export_path: Optional[str] = None,
        n_epochs: int = 3,
    ):
        super().__init__(store=store, agent_id=agent_id)
        self.min_reward = min_reward
        self.model = model
        self.api_key = api_key
        self.upload = upload
        self.export_path = export_path
        self.n_epochs = n_epochs

    def build_dataset(self, runs: List[Run]) -> List[Dict[str, Any]]:
        examples = []
        for run in runs:
            ex = run_to_chat_example(run)
            if ex:
                examples.append(ex)
        return examples

    def export_jsonl(self, examples: List[Dict], path: str) -> str:
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        return path

    def train(self, runs: Optional[List[Run]] = None) -> dict:
        if runs is None:
            runs = self.load_runs(min_reward=self.min_reward)
        if not runs:
            return {"success": False, "message": "No qualifying runs found.", "result": None}

        examples = self.build_dataset(runs)
        if not examples:
            return {"success": False, "message": "Could not extract chat examples from runs.", "result": None}

        # Write JSONL
        path = self.export_path or os.path.join(tempfile.gettempdir(), f"sft_{self.agent_id}.jsonl")
        self.export_jsonl(examples, path)

        result = {
            "success": True,
            "message": f"Prepared {len(examples)} training examples from {len(runs)} runs.",
            "result": path,
            "export_path": path,
            "n_examples": len(examples),
            "runs_used": len(runs),
            "job_id": None,
        }

        if self.upload:
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key or os.environ.get("OPENAI_API_KEY"))
                with open(path, "rb") as f:
                    upload_resp = client.files.create(file=f, purpose="fine-tune")

                job = client.fine_tuning.jobs.create(
                    training_file=upload_resp.id,
                    model=self.model,
                    hyperparameters={"n_epochs": self.n_epochs},
                )
                result["job_id"] = job.id
                result["message"] += f" Fine-tune job submitted: {job.id}"
            except Exception as e:
                result["success"] = False
                result["message"] += f" Upload failed: {e}"

        return result

    def run(self, min_reward=None) -> dict:
        """Override to use self.min_reward as default."""
        effective_min = min_reward if min_reward is not None else self.min_reward
        return super().run(min_reward=effective_min)
