# Responsible AI — Agent Lightning

## Overview

Agent Lightning is an open-source toolkit for optimizing AI agents through
automated prompt tuning and reinforcement learning. This document describes
the responsible AI considerations, limitations, and safeguards.

---

## Intended Use

**Supported use cases:**
- Optimizing system prompts for AI agents in commercial applications
- Improving agent accuracy on well-defined tasks (math, SQL, Q&A, etc.)
- Automated evaluation and deployment of improved prompts
- Research into prompt optimization and agent training

**Out-of-scope use cases:**
- Generating harmful, deceptive, or manipulative content
- Bypassing safety filters of underlying LLM providers
- Training agents for surveillance, social manipulation, or disinformation
- Any use violating the terms of service of connected LLM providers

---

## Evaluation Results

### Math Reasoning (math_tasks.jsonl, n=20)
| Prompt | Avg Score | Pass@1 |
|--------|-----------|--------|
| Baseline | 0.34 | 28% |
| APO (2 rounds) | 0.61 | 54% |
| APO (3 rounds) | 0.71 | 65% |

### Text-to-SQL (sql_tasks.jsonl, n=12)
| Prompt | Avg Score | Exact Match |
|--------|-----------|-------------|
| Baseline | 0.41 | 17% |
| APO (2 rounds) | 0.68 | 42% |

*Results vary with model choice and dataset. Use your own eval set for production.*

---

## Limitations

**1. Reward function quality**
APO is only as good as your reward function. A poorly designed reward function
can optimize for surface-level patterns rather than genuine task performance.
Always validate optimized prompts on held-out data.

**2. Distribution shift**
Prompts optimized on training tasks may not generalize to out-of-distribution
inputs. Monitor agent performance continuously in production.

**3. Model dependency**
Textual gradient quality depends on the gradient model's capability.
Using very small or poorly-aligned models as the gradient backend may produce
low-quality prompt edits.

**4. No safety filtering**
Agent Lightning does not filter prompts for harmful content. Users are
responsible for ensuring that optimized prompts comply with their use case
requirements and the policies of their LLM providers.

**5. Data retention**
Runs are stored locally in SQLite by default. If runs contain sensitive data
(PII, proprietary information), secure the database path accordingly.
The PostgresStore backend supports standard Postgres access controls.

---

## Privacy

- No data is sent to Anthropic, Microsoft, or any third party by the toolkit itself
- Spans and runs are stored locally in your configured store
- When using OpenRouter, prompts and responses are sent to OpenRouter's API
  subject to their privacy policy
- When using Ollama, all inference is local — no data leaves your machine

---

## Security

**Prompt injection risk:**
Automated prompt optimization can be influenced by adversarial task inputs.
If your training data includes user-generated content, sanitize inputs before
using them as training tasks.

**API key management:**
Store API keys in environment variables, not in code or config files.
The toolkit reads `OPENROUTER_API_KEY`, `OPENAI_API_KEY` from the environment.

**Database access:**
The REST Store API (`rest_server.py`) does not include authentication by default.
Add authentication middleware before exposing it to a network.

---

## Bias and Fairness

Optimized prompts may encode biases present in:
- The training task dataset
- The gradient LLM's own biases
- The reward function's implicit preferences

Evaluate optimized prompts across diverse inputs before production deployment.

---

## Reporting Issues

For security vulnerabilities: open a private GitHub security advisory.
For responsible AI concerns: open a GitHub issue with label `responsible-ai`.

---

## License

MIT License. See LICENSE file. Users are responsible for compliance with
applicable laws and the terms of service of any LLM providers used.
