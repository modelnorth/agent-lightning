#!/usr/bin/env python3
"""
prepare_dataset.py — Convert any JSONL file into Agent Lightning training format.

Usage:
    python scripts/prepare_dataset.py --input raw_data.jsonl --output train.jsonl
    python scripts/prepare_dataset.py --input data.jsonl --split 0.8 --output-train train.jsonl --output-val val.jsonl
    python scripts/prepare_dataset.py --input runs_export.jsonl --filter-min-reward 0.7
"""
import argparse, json, random, sys
from pathlib import Path


def load_jsonl(path: str):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(data, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return len(data)


def normalize_item(item: dict) -> dict:
    """Normalize to Agent Lightning task format."""
    normalized = {}
    # Question/input field
    for k in ["question", "input", "prompt", "task", "instruction", "query"]:
        if k in item:
            normalized["question"] = item[k]; break
    # Answer field
    for k in ["answer", "output", "response", "label", "target", "completion"]:
        if k in item:
            normalized["answer"] = item[k]; break
    # Context
    if "context" in item: normalized["context"] = item["context"]
    # Reward (from runs export)
    if "total_reward" in item: normalized["reward"] = item["total_reward"]
    # Keep original fields
    normalized.update({k: v for k, v in item.items() if k not in normalized})
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Agent Lightning")
    parser.add_argument("--input",          required=True)
    parser.add_argument("--output",         default="prepared.jsonl")
    parser.add_argument("--output-train",   default=None)
    parser.add_argument("--output-val",     default=None)
    parser.add_argument("--split",          type=float, default=0.8)
    parser.add_argument("--filter-min-reward", type=float, default=None)
    parser.add_argument("--shuffle",        action="store_true")
    parser.add_argument("--limit",          type=int, default=None)
    parser.add_argument("--normalize",      action="store_true", default=True)
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    data = load_jsonl(args.input)
    print(f"  Loaded {len(data)} items")

    # Filter by reward
    if args.filter_min_reward is not None:
        before = len(data)
        data = [d for d in data if d.get("total_reward", d.get("reward", 1.0)) >= args.filter_min_reward]
        print(f"  Filtered to {len(data)} (min_reward={args.filter_min_reward}, removed {before-len(data)})")

    # Shuffle
    if args.shuffle:
        random.shuffle(data)
        print(f"  Shuffled")

    # Limit
    if args.limit:
        data = data[:args.limit]
        print(f"  Limited to {args.limit}")

    # Normalize
    if args.normalize:
        data = [normalize_item(d) for d in data]

    # Train/val split
    if args.output_train and args.output_val:
        split_idx = int(len(data) * args.split)
        train, val = data[:split_idx], data[split_idx:]
        n_train = save_jsonl(train, args.output_train)
        n_val   = save_jsonl(val,   args.output_val)
        print(f"  Saved {n_train} train → {args.output_train}")
        print(f"  Saved {n_val} val   → {args.output_val}")
    else:
        n = save_jsonl(data, args.output)
        print(f"  Saved {n} items → {args.output}")

    print("Done.")


if __name__ == "__main__":
    main()
