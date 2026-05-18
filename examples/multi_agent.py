"""
Multi-Agent Workflow Example
==============================

Shows how to trace individual agents inside a larger pipeline.
Each agent has its own Tracer with its own agent_id.
They share the same RunStore so everything lands in one place.

Pipeline:
  User query → Planner → Researcher → Summarizer → Final answer
"""

import agent_lightning as al
from agent_lightning.core.run_store import RunStore

# Shared store — all agents write here
shared_store = RunStore()

# One tracer per agent role
planner_tracer = al.Tracer(agent_id="planner", store=shared_store)
researcher_tracer = al.Tracer(agent_id="researcher", store=shared_store)
summarizer_tracer = al.Tracer(agent_id="summarizer", store=shared_store)


def fake_plan(query: str) -> list:
    return [f"Step 1: Research {query}", "Step 2: Synthesize findings", "Step 3: Summarize"]


def fake_research(step: str) -> str:
    return f"Research result for: {step} — found 3 relevant documents."


def fake_summarize(research: list) -> str:
    return f"Summary of {len(research)} findings: All results point to a coherent answer."


def run_pipeline(query: str) -> str:
    print(f"\n→ Processing: {query}")

    # 1. Planner
    with planner_tracer.run(session_id=query[:20]) as plan_run:
        planner_tracer.log_prompt(f"Plan how to answer: {query}")
        plan = fake_plan(query)
        planner_tracer.log_response(str(plan))
        plan_run.add_reward(1.0)

    # 2. Researcher (could be a different model/service)
    research_results = []
    for step in plan:
        with researcher_tracer.run(session_id=query[:20]) as res_run:
            researcher_tracer.log_prompt(step)
            result = fake_research(step)
            researcher_tracer.log_tool_call("web_search", {"query": step})
            researcher_tracer.log_tool_result("web_search", result)
            researcher_tracer.log_response(result)
            research_results.append(result)
            res_run.add_reward(0.9)

    # 3. Summarizer
    with summarizer_tracer.run(session_id=query[:20]) as sum_run:
        summarizer_tracer.log_prompt(f"Summarize: {research_results}")
        summary = fake_summarize(research_results)
        summarizer_tracer.log_response(summary)
        sum_run.add_reward(1.0)

    return summary


# Run the pipeline
queries = ["What is quantum computing?", "How does RLHF work?"]
for q in queries:
    result = run_pipeline(q)
    print(f"Result: {result[:60]}...")

# Inspect per-agent stats
print("\n── Per-agent stats ──────────────────────────────")
for agent_id in ["planner", "researcher", "summarizer"]:
    stats = shared_store.stats(agent_id=agent_id)
    print(f"{agent_id:>12}: {stats['total']} runs, avg_reward={stats['avg_reward']:.3f}")

print(f"\nTotal runs in store: {shared_store.count()}")
print("Start the dashboard to see all agents: python -m agent_lightning dashboard")
