"""LangChain integration helpers."""

from agent_lightning.core.wrappers import wrap_langchain_llm
from agent_lightning.core.run_store import RunStore


class LangChainIntegration:
    """Convenience class for LangChain integration."""

    @staticmethod
    def patch_llm(llm, agent_id: str = "default", store=None):
        return wrap_langchain_llm(llm, agent_id=agent_id, store=store or RunStore())
