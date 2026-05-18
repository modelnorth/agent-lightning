"""
Tool-Using Agent Example
=========================

Shows how to trace tool calls alongside LLM responses.
Uses a fake calculator tool — no real API needed.
"""

import random
import agent_lightning as al

tracer = al.Tracer(agent_id="tool_agent")


def calculator(expression: str) -> float:
    """Fake calculator tool."""
    try:
        return eval(expression, {"__builtins__": {}})
    except Exception:
        return 0.0


def fake_llm_with_tool(question: str) -> tuple:
    """Returns (text_response, tool_name, tool_args) or (text_response, None, None)"""
    if any(op in question for op in ['+', '-', '*', '/', 'calculate', 'compute']):
        expr = question.split("=")[0].strip().split()[-1] if "=" in question else "2+2"
        return None, "calculator", {"expression": expr}
    return f"The answer to '{question}' is 42.", None, None


questions = [
    "What is 15 * 7?",
    "Calculate 100 / 4",
    "What is the meaning of life?",
    "Compute 99 + 1",
    "Who wrote Hamlet?",
]

for question in questions:
    with tracer.run(tags=["tool-demo"]) as run:
        # Log user prompt
        tracer.log_prompt(question)

        # LLM decides whether to call a tool
        text, tool_name, tool_args = fake_llm_with_tool(question)

        if tool_name:
            # Log tool call
            tracer.log_tool_call(tool_name, tool_args)

            # Execute tool
            result = calculator(tool_args.get("expression", ""))

            # Log tool result
            tracer.log_tool_result(tool_name, result)

            # LLM generates final response
            final = f"The result is {result}"
            tracer.log_response(final)

            # Reward: was tool used correctly?
            reward = 1.0 if result != 0 else 0.2
        else:
            tracer.log_response(text)
            reward = random.uniform(0.3, 0.9)

        run.add_reward(reward)
        print(f"Q: {question[:40]:<40} reward={reward:.2f}")

print(f"\nTotal runs: {tracer.store.count(agent_id='tool_agent')}")
