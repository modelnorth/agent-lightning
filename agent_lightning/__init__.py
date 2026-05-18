"""
⚡ Agent Lightning v0.2
Drop-in AI agent optimization toolkit.
Plug in → record runs → train with 100+ models → ship better prompts back.
"""

from agent_lightning.core.models    import (Run, Step, StepType, Reward, Span,
                                             Rollout, Attempt, RolloutStatus, PromptTemplate)
from agent_lightning.core.store     import LightningStore, RunStore
from agent_lightning.core.async_tracer import AsyncTracer, Tracer, Hook, trace
from agent_lightning.core.wrappers  import wrap_openai, AgentWrapper, wrap_langchain_llm
from agent_lightning.backends       import (LLMBackend, LLMMessage, LLMResponse,
                                             OpenRouterBackend, OpenAIBackend,
                                             OllamaBackend, LiteLLMBackend,
                                             MultiModelConsensus, OPENROUTER_MODELS)
from agent_lightning.algorithms     import APO
from agent_lightning.orchestrator   import (Orchestrator, OrchestratorConfig,
                                             EveryNRuns, Scheduled, Manual,
                                             OnImprovement, Evaluator, ABTest)

__version__ = "0.3.0"
__all__ = [
    # Core
    "Run","Step","StepType","Reward","Span","Rollout","Attempt","RolloutStatus","PromptTemplate",
    "LightningStore","RunStore",
    "AsyncTracer","Tracer","Hook","trace",
    "wrap_openai","wrap_langchain_llm","AgentWrapper",
    # Backends
    "LLMBackend","LLMMessage","LLMResponse","OpenRouterBackend","OpenAIBackend","OllamaBackend","LiteLLMBackend",
    "MultiModelConsensus","OPENROUTER_MODELS",
    # Algorithms
    "APO",
    # Orchestrator
    "Orchestrator","OrchestratorConfig",
    "EveryNRuns","Scheduled","Manual","OnImprovement",
    "Evaluator","ABTest",
]
