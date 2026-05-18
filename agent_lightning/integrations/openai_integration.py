"""OpenAI integration helpers."""

from agent_lightning.core.wrappers import wrap_openai
from agent_lightning.core.run_store import RunStore


class OpenAIIntegration:
    """Convenience class for OpenAI integration."""

    @staticmethod
    def patch_client(client, agent_id: str = "default", store=None):
        return wrap_openai(client, agent_id=agent_id, store=store or RunStore())
