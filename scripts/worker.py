"""Small script to run the jobs worker loop or a single iteration.

Usage:
    # run continuously (CTRL-C to stop)
    python scripts/worker.py --jobs-db data/jobs.db --loop

    # run one iteration
    python scripts/worker.py --jobs-db data/jobs.db --once
"""
from __future__ import annotations

import argparse
from src.enrichment.jobs import worker_loop, worker_once, init_jobs_db


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs-db", required=True)
    p.add_argument("--once", action="store_true")
    p.add_argument("--loop", action="store_true")
    args = p.parse_args()

    init_jobs_db(args.jobs_db)
    if args.once:
        processed = worker_once(args.jobs_db)
        print("Processed job id:", processed)
    elif args.loop:
        print("Starting worker loop (press CTRL-C to stop)")
        worker_loop(args.jobs_db)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
