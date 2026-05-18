"""
OpenAI Agent — 2-line integration
===================================

Add tracing to an existing OpenAI agent with just 2 lines:
  1. import agent_lightning as al
  2. client = al.wrap_openai(client, agent_id="my_agent")

Everything else stays the same.
"""

# Requires: pip install openai agent-lightning
# Set: OPENAI_API_KEY env var

import os
import agent_lightning as al
from openai import OpenAI

# ── Your existing code ────────────────────────────────────────────────────────
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-test"))

# ── ADD THIS ONE LINE ─────────────────────────────────────────────────────────
client = al.wrap_openai(client, agent_id="openai_demo")

# ── Everything below is unchanged ─────────────────────────────────────────────
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ],
)

print(response.choices[0].message.content)

# Add a reward signal after you evaluate the response
tracer = client._lightning_tracer
store = tracer.store
runs = store.list_runs(agent_id="openai_demo")
if runs:
    latest = runs[0]
    latest.add_reward(1.0, label="correct")
    store.save(latest)
    print(f"\nRun saved: {latest.run_id}")
    print(f"Steps: {len(latest.steps)}")
