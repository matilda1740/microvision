"""Small helper to produce the DataFrames used to build `use_df` for manual inspection.

This script performs a light-weight run of the parser, cleaning and merge steps
so you can inspect the intermediate tables (`parsed_df`, `cleaned`, `aggregated`).

Usage (from project root):
  python3 scripts/inspect_use_df.py --source data/sample_raw.csv --sample 200 --out-dir data/debug_inspect

It writes CSVs to the `--out-dir` and prints small samples to stdout for quick review.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# ensure project root on path so `from src...` imports work when running the script
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src.parsing.metadata_drain_parser import MetadataDrainParser
from src.enrichment.cleaning import minimal_cleaning
from src.enrichment.merger import merge_structured_metadata


def reservoir_sample_csv(path: Path, sample_size: int, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    sample: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        try:
            start = fh.read(8192)
            fh.seek(0)
            has_header = csv.Sniffer().has_header(start)
        except Exception:
            has_header = False

        if has_header:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                if i < sample_size:
                    sample.append(row)
                else:
                    j = rng.randint(0, i)
                    if j < sample_size:
                        sample[j] = row
        else:
            for i, line in enumerate(fh):
                row = {"raw": line.rstrip("\n")}
                if i < sample_size:
                    sample.append(row)
                else:
                    j = rng.randint(0, i)
                    if j < sample_size:
                        sample[j] = row

    return sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-format", dest="log_format", default=None,
                   help="Optional LogPai-style format string (e.g. '<Date> <Time> <Pid> <Level> <Component> <Content>')")
    p.add_argument("--source", required=True, help="Path to source CSV or text file")
    p.add_argument("--sample", type=int, default=500, help="Number of rows to sample")
    p.add_argument("--out-dir", default="data/debug_inspect", help="Directory to write inspection CSVs")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sampling up to {args.sample} rows from {src} (seed={args.seed}) ...")
    sampled = reservoir_sample_csv(src, int(args.sample), seed=int(args.seed))

    # run MetadataDrainParser in fresh mode to produce structured CSV and templates
    parsed_path = out_dir / "parsed_sample.csv"
    templates_path = out_dir / "parsed_templates.csv"
    print(f"Running parser and writing parsed CSV to {parsed_path} and templates to {templates_path} ...")

    # allow supplying a log format so component/service tokens can be extracted
    parser = MetadataDrainParser(
        log_format=args.log_format,
        structured_csv=str(parsed_path),
        templates_csv=str(templates_path),
        save_every=max(1, int(args.sample) // 4),
        mode="fresh",
    )
    for i, row in enumerate(sampled):
        raw_line = row.get("raw") or row.get("content") or " ".join([str(v) for v in row.values()])
        parser.process_line(raw_line, i + 1)
    parser.finalize()

    print(f"Reading parsed results from {parsed_path} ...")
    parsed_df = pd.read_csv(parsed_path)
    parsed_out = out_dir / "parsed_df_sample.csv"
    parsed_df.to_csv(parsed_out, index=False)

    print("Running minimal cleaning/semantic grouping ...")
    cleaned = minimal_cleaning(parsed_df)
    cleaned_out = out_dir / "cleaned_templates.csv"
    cleaned.to_csv(cleaned_out, index=False)

    print("Running merge with structured metadata (produces aggregated rows) ...")
    merged_out_path = str(out_dir / "merged_templates.csv")
    aggregated = merge_structured_metadata(cleaned, parsed_df, merged_out_path)

    # save aggregated to CSV (merge already writes, but ensure we have a copy)
    aggregated_out = out_dir / "aggregated_use_df.csv"
    aggregated.to_csv(aggregated_out, index=False)

    # Print small samples to stdout for quick manual inspection
    print("\n=== parsed_df sample (head) ===")
    with pd.option_context('display.max_rows', 10, 'display.max_columns', 8):
        print(parsed_df.head(6).to_string())

    print("\n=== cleaned sample (head) ===")
    with pd.option_context('display.max_rows', 10, 'display.max_columns', 8):
        print(cleaned.head(6).to_string())

    print("\n=== aggregated (use_df) sample (head) ===")
    with pd.option_context('display.max_rows', 10, 'display.max_columns', 12):
        print(aggregated.head(6).to_string())

    # also write small JSON files with first 20 records for convenient copy-paste inspection
    try:
        parsed_df.head(20).to_json(out_dir / "parsed_head.json", orient="records", force_ascii=False)
        cleaned.head(20).to_json(out_dir / "cleaned_head.json", orient="records", force_ascii=False)
        aggregated.head(20).to_json(out_dir / "aggregated_head.json", orient="records", force_ascii=False)
    except Exception:
        # non-fatal
        pass

    print(f"Wrote inspection artifacts to {out_dir}:")
    print(f" - parsed: {parsed_out}")
    print(f" - cleaned: {cleaned_out}")
    print(f" - aggregated (use_df): {aggregated_out}")


if __name__ == "__main__":
    main()
