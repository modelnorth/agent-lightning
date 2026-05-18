"""
agl — Agent Lightning CLI

Commands:
    agl dashboard              Start the web dashboard
    agl train                  Run a training iteration now
    agl stats [agent_id]       Show run statistics
    agl export [agent_id]      Export runs to JSONL
    agl models                 List OpenRouter models by tier
    agl rollouts [agent_id]    Show rollout queue status
    agl prompt [agent_id]      Show current optimized prompt
"""

import argparse
import asyncio
import json
import os
import sys


def cmd_dashboard(args):
    from agent_lightning.dashboard.server import run_dashboard
    run_dashboard(host=args.host, port=args.port)


def cmd_stats(args):
    from agent_lightning.core.store import LightningStore
    store = LightningStore(args.db)
    stats = store.stats(agent_id=args.agent_id or None)
    print(json.dumps(stats, indent=2))


def cmd_export(args):
    from agent_lightning.core.store import LightningStore
    store = LightningStore(args.db)
    path = args.output or f"runs_{args.agent_id or 'all'}.jsonl"
    n = store.export_jsonl(path, agent_id=args.agent_id or None)
    print(f"Exported {n} runs → {path}")


def cmd_models(args):
    from agent_lightning.backends.openrouter import OPENROUTER_MODELS
    for tier, models in OPENROUTER_MODELS.items():
        print(f"\n{tier.upper()}:")
        for m in models:
            print(f"  {m}")


def cmd_rollouts(args):
    from agent_lightning.core.store import LightningStore
    store = LightningStore(args.db)

    async def _show():
        stats = await store.rollout_stats(agent_id=args.agent_id or None)
        print(json.dumps(stats, indent=2))
        rollouts = await store.list_rollouts(agent_id=args.agent_id or None, limit=20)
        print(f"\nLatest {len(rollouts)} rollouts:")
        for r in rollouts:
            print(f"  [{r.status.value:10}] {r.rollout_id[:16]}… agent={r.agent_id} reward={r.best_reward}")

    asyncio.run(_show())


def cmd_prompt(args):
    from agent_lightning.core.store import LightningStore
    store = LightningStore(args.db)
    agent_id = args.agent_id or "default"
    prompt = store.get_optimized_prompt(agent_id)
    if prompt:
        print(f"Agent: {agent_id}")
        print(f"Prompt:\n{prompt}")
    else:
        print(f"No optimized prompt found for agent '{agent_id}'")


def cmd_train(args):
    """Quick training run from CLI."""
    from agent_lightning.core.store import LightningStore
    from agent_lightning.algorithms.apo import APO
    from agent_lightning.backends.openrouter import OpenRouterBackend

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: --api-key or OPENROUTER_API_KEY env var required")
        sys.exit(1)

    store   = LightningStore(args.db)
    backend = OpenRouterBackend(api_key=api_key, model=args.model)
    current_prompt = store.get_optimized_prompt(args.agent_id) or args.prompt or "You are a helpful assistant."

    apo = APO(
        store=store, agent_id=args.agent_id,
        gradient_backend=backend,
        initial_prompt=current_prompt,
        beam_width=int(args.beam_width),
        beam_rounds=int(args.rounds),
    )

    print(f"Training agent='{args.agent_id}' model={args.model} rounds={args.rounds}...")

    async def _run():
        result = await apo.run()
        if result["success"]:
            print(f"\n✅ Done! Best score: {result['best_score']}")
            print(f"\nOptimized prompt:\n{result['best_prompt']}")
        else:
            print(f"Training failed: {result}")

    asyncio.run(_run())


def main():
    parser = argparse.ArgumentParser(
        prog="agl",
        description="⚡ Agent Lightning CLI",
    )
    parser.add_argument("--db", default=None, help="Path to SQLite DB")

    sub = parser.add_subparsers(dest="command")

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Start web dashboard")
    p_dash.add_argument("--host", default="127.0.0.1")
    p_dash.add_argument("--port", type=int, default=7860)
    p_dash.set_defaults(func=cmd_dashboard)

    # stats
    p_stats = sub.add_parser("stats", help="Show run statistics")
    p_stats.add_argument("agent_id", nargs="?")
    p_stats.set_defaults(func=cmd_stats)

    # export
    p_export = sub.add_parser("export", help="Export runs to JSONL")
    p_export.add_argument("agent_id", nargs="?")
    p_export.add_argument("-o", "--output", default=None)
    p_export.set_defaults(func=cmd_export)

    # models
    p_models = sub.add_parser("models", help="List OpenRouter model tiers")
    p_models.set_defaults(func=cmd_models)

    # rollouts
    p_rollouts = sub.add_parser("rollouts", help="Show rollout queue")
    p_rollouts.add_argument("agent_id", nargs="?")
    p_rollouts.set_defaults(func=cmd_rollouts)

    # prompt
    p_prompt = sub.add_parser("prompt", help="Show current optimized prompt")
    p_prompt.add_argument("agent_id", nargs="?", default="default")
    p_prompt.set_defaults(func=cmd_prompt)

    # train
    p_train = sub.add_parser("train", help="Run APO training now")
    p_train.add_argument("--agent-id", default="default")
    p_train.add_argument("--api-key", default=None)
    p_train.add_argument("--model", default="qwen/qwen3-8b:free")
    p_train.add_argument("--prompt", default=None)
    p_train.add_argument("--beam-width", default=3)
    p_train.add_argument("--rounds", default=2)
    p_train.set_defaults(func=cmd_train)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()


def cmd_serve(args):
    """Start the REST store API server."""
    from agent_lightning.core.rest_server import run_rest_server
    run_rest_server(host=args.host, port=args.port, db_path=args.db)


# Patch serve into main parser
_original_main = main

def main():
    import argparse, sys
    parser = argparse.ArgumentParser(prog="agl", description="⚡ Agent Lightning CLI")
    parser.add_argument("--db", default=None)
    sub = parser.add_subparsers(dest="command")

    p_dash = sub.add_parser("dashboard", help="Start web dashboard")
    p_dash.add_argument("--host", default="127.0.0.1")
    p_dash.add_argument("--port", type=int, default=7860)
    p_dash.set_defaults(func=cmd_dashboard)

    p_serve = sub.add_parser("serve", help="Start REST store API server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    p_stats = sub.add_parser("stats", help="Show run statistics")
    p_stats.add_argument("agent_id", nargs="?")
    p_stats.set_defaults(func=cmd_stats)

    p_export = sub.add_parser("export", help="Export runs to JSONL")
    p_export.add_argument("agent_id", nargs="?")
    p_export.add_argument("-o", "--output", default=None)
    p_export.set_defaults(func=cmd_export)

    p_models = sub.add_parser("models", help="List OpenRouter model tiers")
    p_models.set_defaults(func=cmd_models)

    p_rollouts = sub.add_parser("rollouts", help="Show rollout queue")
    p_rollouts.add_argument("agent_id", nargs="?")
    p_rollouts.set_defaults(func=cmd_rollouts)

    p_prompt = sub.add_parser("prompt", help="Show current optimized prompt")
    p_prompt.add_argument("agent_id", nargs="?", default="default")
    p_prompt.set_defaults(func=cmd_prompt)

    p_train = sub.add_parser("train", help="Run APO training now")
    p_train.add_argument("--agent-id", default="default")
    p_train.add_argument("--api-key", default=None)
    p_train.add_argument("--model", default="qwen/qwen3-8b:free")
    p_train.add_argument("--prompt", default=None)
    p_train.add_argument("--beam-width", default=3)
    p_train.add_argument("--rounds", default=2)
    p_train.set_defaults(func=cmd_train)

    args = parser.parse_args()
    if not args.command:
        parser.print_help(); sys.exit(0)
    args.func(args)
