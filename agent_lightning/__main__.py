"""CLI entry point."""
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m agent_lightning [dashboard|export]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    if cmd == "dashboard":
        from agent_lightning.dashboard import run_dashboard
        run_dashboard()
    elif cmd == "export":
        agent_id = sys.argv[2] if len(sys.argv) > 2 else None
        path = sys.argv[3] if len(sys.argv) > 3 else "runs_export.jsonl"
        from agent_lightning.core.run_store import RunStore
        store = RunStore()
        n = store.export_jsonl(path, agent_id=agent_id)
        print(f"Exported {n} runs to {path}")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

if __name__ == "__main__":
    main()
