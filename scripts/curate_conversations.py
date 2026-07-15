"""
Review logged conversations and approve exchanges for fine-tuning.

Usage:
    python -m scripts.curate_conversations [--date 2026-07-14]

Keys:
    y / Enter  — approve and add to training data
    n          — skip
    e          — edit assistant reply before approving
    q          — quit
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from ev.config import PROJECT_ROOT, SYSTEM_PROMPT

LOG_DIR = PROJECT_ROOT / "data" / "conversations"
TRAIN_PATH = PROJECT_ROOT / "data" / "training" / "train.jsonl"


def _load_log(date_str: str) -> list[dict]:
    path = LOG_DIR / f"{date_str}.jsonl"
    if not path.exists():
        print(f"No log found for {date_str} ({path})")
        sys.exit(1)
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _load_existing_training() -> set[tuple[str, str]]:
    """Return set of (user, assistant) pairs already in train.jsonl."""
    seen = set()
    if not TRAIN_PATH.exists():
        return seen
    with open(TRAIN_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            seen.add((user, asst))
    return seen


def _append_training(user: str, assistant: str):
    entry = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }
    with open(TRAIN_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _prompt(question: str) -> str:
    try:
        return input(question).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def curate(date_str: str):
    entries = _load_log(date_str)
    existing = _load_existing_training()
    print(f"\nLoaded {len(entries)} exchanges from {date_str}.")
    print("Keys: [y/Enter] approve  [n] skip  [e] edit  [q] quit\n")
    print("=" * 60)

    approved = skipped = already_in = 0

    for i, entry in enumerate(entries, 1):
        user = entry.get("user", "").strip()
        assistant = entry.get("assistant", "").strip()
        ts = entry.get("timestamp", "")[:19]

        if not user or not assistant:
            continue

        if (user, assistant) in existing:
            already_in += 1
            continue

        print(f"\n[{i}/{len(entries)}] {ts}")
        print(f"  You : {user}")
        print(f"  EV  : {assistant}")

        answer = _prompt("\n  Approve? > ")

        if answer in ("q", "quit"):
            break
        elif answer in ("e", "edit"):
            print(f"  Current reply: {assistant}")
            edited = input("  New reply    : ").strip()
            if edited:
                assistant = edited
            _append_training(user, assistant)
            existing.add((user, assistant))
            approved += 1
            print("  Saved (edited).")
        elif answer in ("", "y", "yes"):
            _append_training(user, assistant)
            existing.add((user, assistant))
            approved += 1
            print("  Saved.")
        else:
            skipped += 1
            print("  Skipped.")

    print("\n" + "=" * 60)
    print(f"Done. Approved: {approved}  Skipped: {skipped}  Already in training: {already_in}")
    if approved:
        print(f"Added to: {TRAIN_PATH}")
        print("Run `python -m scripts.fine_tune` to retrain with new data.")


def main():
    parser = argparse.ArgumentParser(description="Curate conversation logs for fine-tuning.")
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Date to review (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available log dates and exit.",
    )
    args = parser.parse_args()

    if args.list:
        logs = sorted(LOG_DIR.glob("*.jsonl"))
        if not logs:
            print("No conversation logs found.")
        else:
            print("Available logs:")
            for p in logs:
                count = sum(1 for line in open(p) if line.strip())
                print(f"  {p.stem}  ({count} exchanges)")
        return

    curate(args.date)


if __name__ == "__main__":
    main()
