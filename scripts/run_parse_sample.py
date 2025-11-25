"""Lightweight runner to parse the sample logs and produce a structured CSV.

This script is intentionally minimal: it reads `data/sample_raw.csv` line-by-line,
invokes the MetadataDrainParser (which lazy-loads drain3) and writes out
`data/sample_raw_structured.csv` and `data/sample_raw_templates.csv`.

It avoids heavy dependencies (embeddings/chroma) and is meant as a fast
smoke-test to validate parsing/metadata extraction changes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import logging
import traceback

import pandas as pd

# Ensure project root is on sys.path so `from src...` imports work when this
# script is run directly from the repository root or via IDEs/terminals.
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src.parsing.metadata_drain_parser import MetadataDrainParser

# re-use the conservative cleaning/grouping from the enrichment package so the
# lightweight runner produces the same cleaned templates artifact.
try:
    from src.enrichment.cleaning import minimal_cleaning
except Exception:
    # fallback: if import fails (very unlikely), define a minimal no-op
    # cleaning that preserves template content into a cleaned dataframe
    # structure compatible with the rest of the pipeline.
    def minimal_cleaning(df: pd.DataFrame) -> pd.DataFrame:
        # Expect 'template' or 'content' column. Produce a dataframe with
        # 'semantic_text', 'template_ids' and 'occurrences'. Conservative
        # grouping: each unique normalized template becomes one semantic row.
        if "template" not in df.columns and "content" in df.columns:
            df = df.rename(columns={"content": "template"})
        out = []
        for _, row in df.iterrows():
            tpl = row.get("template") if "template" in row else ""
            out.append({"semantic_text": tpl, "template_ids": [row.get("template_id")], "occurrences": 1})
        return pd.DataFrame(out)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INPUT = DATA_DIR / "sample_raw.csv"
STRUCTURED_OUT = DATA_DIR / "sample_raw_structured.csv"
TEMPLATES_OUT = DATA_DIR / "sample_raw_templates.csv"


def main() -> int:
    if not INPUT.exists():
        print(f"Input file not found: {INPUT}")
        return 2

    # ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Use the logical mapping key 'OpenStack' so the parser picks the known format
    parser = MetadataDrainParser(log_format="OpenStack", structured_csv=str(STRUCTURED_OUT), templates_csv=str(TEMPLATES_OUT), save_every=500)

    print(f"Parsing {INPUT} -> {STRUCTURED_OUT} (templates -> {TEMPLATES_OUT})")

    with INPUT.open("r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, start=1):
            try:
                parser.process_line(line, i)
            except Exception as exc:  # guard against a single bad line
                print(f"Warning: failed to process line {i}: {exc}")

    parser.finalize()

    # Print a quick summary of the generated CSV
    if STRUCTURED_OUT.exists():
        try:
            df = pd.read_csv(STRUCTURED_OUT)
            print("\nStructured CSV columns:")
            print(",".join(df.columns.tolist()))
            print("\nFirst 5 rows:")
            print(df.head(5).to_string(index=False))
        except Exception as exc:
            print(f"Could not read generated structured CSV: {exc}")
    else:
        print("No structured CSV produced.")

    # ---- Cleaning / grouping ----
    try:
        parsed_df = pd.read_csv(STRUCTURED_OUT)
    except Exception:
        print("Could not read structured CSV for cleaning; skipping cleaning step.")
        return 0

    print("\nRunning minimal cleaning/grouping ...")
    cleaned_df = minimal_cleaning(parsed_df)

    cleaned_out = DATA_DIR / "sample_raw_cleaned_templates.csv"
    try:
        cleaned_df.to_csv(cleaned_out, index=False)
        print(f"Wrote cleaned templates to {cleaned_out}")
    except Exception as exc:
        print(f"Failed to write cleaned templates: {exc}")

    # ---- Merging (mandatory) ----
    try:
        from src.enrichment.merger import merge_structured_metadata
    except Exception as exc:
        logging.error("Merger import failed (merge is mandatory): %s", exc)
        logging.debug(traceback.format_exc())
        raise RuntimeError("Merger import failed; merge is mandatory") from exc

    merged_out = DATA_DIR / "sample_raw_merged.csv"
    try:
        logging.info("Running merger -> %s ...", merged_out)
        merged_df = merge_structured_metadata(cleaned_df, parsed_df, str(merged_out))
        logging.info("Wrote merged output to %s", merged_out)
        try:
            print("Merged preview:")
            print(merged_df.head(5).to_string(index=False))
        except Exception:
            pass
    except Exception as exc:
        logging.error("Merge failed: %s", exc)
        logging.debug(traceback.format_exc())
        raise RuntimeError("Merge failed; merge is mandatory") from exc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
