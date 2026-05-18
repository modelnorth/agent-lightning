"""
RAG Optimizer Recipe — optimize retrieval-augmented generation prompts.

Contributed recipe. Shows how to use APO to improve a RAG system prompt
by rewarding responses that correctly cite retrieved context.

Usage:
    from contrib.recipes.rag_optimizer import RAGOptimizer
    optimizer = RAGOptimizer(store=store, backend=OpenRouterBackend.fast(...))
    result = await optimizer.run(corpus=my_documents, questions=my_questions)
"""
from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import agent_lightning as al
from agent_lightning.algorithms.apo import APO


def rag_reward(response: str, item: dict) -> float:
    """
    Reward function for RAG tasks.
    Checks: answer present, context cited, not hallucinated.
    """
    score = 0.0
    if not response or len(response) < 20:
        return 0.0

    expected = str(item.get("answer", "")).lower()
    context  = str(item.get("context", "")).lower()
    response_lower = response.lower()

    # Answer appears in response
    if expected and expected in response_lower:
        score += 0.5

    # Response references context (not hallucinating)
    context_words = set(context.split()) - {"the", "a", "an", "is", "in", "of"}
    response_words = set(response_lower.split())
    overlap = len(context_words & response_words) / max(len(context_words), 1)
    score += min(0.3, overlap)

    # Length penalty: too short or too long
    if 50 < len(response) < 500:
        score += 0.2

    return min(1.0, score)


class RAGOptimizer:
    """
    Optimize a RAG system prompt using APO.

    Rewards prompts that produce grounded, citation-based answers.
    """

    DEFAULT_PROMPT = """You are a precise assistant that answers questions using ONLY the provided context.
Always cite the specific part of the context that supports your answer.
If the answer is not in the context, say "I cannot find this in the provided context."
Never make up information."""

    def __init__(
        self,
        store: al.LightningStore,
        agent_id: str = "rag_agent",
        backend: al.LLMBackend = None,
        initial_prompt: str = None,
        beam_width: int = 3,
        beam_rounds: int = 2,
    ):
        self.store          = store
        self.agent_id       = agent_id
        self.backend        = backend
        self.initial_prompt = initial_prompt or self.DEFAULT_PROMPT
        self.beam_width     = beam_width
        self.beam_rounds    = beam_rounds

    async def run(self, corpus: List[str], questions: List[Dict]) -> Dict:
        """
        Optimize the RAG prompt.

        Args:
            corpus:    List of document strings to use as context
            questions: List of {"question": str, "answer": str, "context": str}
        """
        if not self.backend:
            raise ValueError("backend required")

        # Build dataset: each item = question + context snippet
        dataset = []
        for q in questions:
            ctx = q.get("context") or (corpus[0][:500] if corpus else "")
            dataset.append({
                "input":   f"Context: {ctx}\nQuestion: {q['question']}",
                "answer":  q.get("answer", ""),
                "context": ctx,
            })

        def reward_fn(response: str, item_input: str) -> float:
            # Find matching item
            for item in dataset:
                if item["input"] == item_input:
                    return rag_reward(response, item)
            return 0.5

        apo = APO(
            store=self.store, agent_id=self.agent_id,
            gradient_backend=self.backend, eval_backend=self.backend,
            initial_prompt=self.initial_prompt,
            beam_width=self.beam_width, beam_rounds=self.beam_rounds,
            reward_fn=reward_fn,
        )

        return await apo.run(
            train_dataset=[d["input"] for d in dataset[:int(len(dataset)*0.8)]],
            val_dataset=[d["input"] for d in dataset[int(len(dataset)*0.8):]],
        )


async def demo():
    print("RAG Optimizer Recipe Demo")
    store = al.LightningStore(":memory:")

    corpus = [
        "Agent Lightning is an open-source AI agent optimization toolkit built in Python.",
        "It supports OpenRouter with 100+ LLM models for model-agnostic training.",
        "The APO algorithm uses textual gradients and beam search to improve prompts.",
    ]

    questions = [
        {"question": "What is Agent Lightning?",
         "answer":   "open-source AI agent optimization toolkit",
         "context":  corpus[0]},
        {"question": "How many models does OpenRouter support?",
         "answer":   "100+",
         "context":  corpus[1]},
        {"question": "What algorithm does Agent Lightning use for prompt optimization?",
         "answer":   "APO",
         "context":  corpus[2]},
    ]

    class MockBackend(al.LLMBackend):
        async def complete(self, msgs, **kw):
            ctx = next((m.content for m in msgs if "Context" in (m.content or "")), "")
            return al.LLMResponse(
                f"Based on the context: {ctx[:80]}...",
                model="mock"
            )

    optimizer = RAGOptimizer(store=store, backend=MockBackend())
    result    = await optimizer.run(corpus=corpus, questions=questions)
    print(f"Success: {result['success']}")
    print(f"Best prompt: {result['best_prompt'][:100]}...")


if __name__ == "__main__":
    asyncio.run(demo())
