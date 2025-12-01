"""CLI to export transitions table to CSV for offline analysis.

Example:
    # Use the project's configured default DB path from config.settings
    python scripts/export_transitions.py --db "$(python -c \"from config.settings import settings; print(settings.DEFAULT_DB)\")" --out transitions.csv
"""
from __future__ import annotations

import argparse
from src.transitions import export_transitions_to_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True, help="Path to SQLite DB with transitions table")
    p.add_argument("--out", required=True, help="CSV output path")
    args = p.parse_args()

    n = export_transitions_to_csv(args.db, args.out)
    print(f"Exported {n} transitions to {args.out}")


if __name__ == "__main__":
    main()
