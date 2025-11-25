"""CLI to compute order-1 transitions from an EdgeStore SQLite DB.

Example:
    # Use the project's configured default DB path from config.settings
    python scripts/compute_transitions.py --db "$(python -c \"from config.settings import settings; print(settings.DEFAULT_DB)\")" --min-count 2
"""
from __future__ import annotations

import argparse
from src.persistence.transitions import compute_and_persist_transitions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True, help="Path to SQLite DB with edges table")
    p.add_argument("--min-count", type=int, default=1, help="Minimum count to persist transition")
    args = p.parse_args()

    n = compute_and_persist_transitions(args.db, min_count=args.min_count)
    print(f"Wrote {n} transitions into '{args.db}'")


if __name__ == "__main__":
    main()
