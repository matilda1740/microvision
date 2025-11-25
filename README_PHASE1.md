Phase 1 — Parsing & Enrichment (Accuracy + Validation)

This document summarizes Phase 1 of the pipeline and explains the two
stages implemented in this repository. It's intended as a short, practical
reference for contributors and reviewers validating the parsing and
enrichment behavior.

Overview
--------
Phase 1 focuses on accuracy and validation. It implements a two-stage
pipeline that is intentionally conservative and testable:

- Stage 1 — Parsing the logs
- Stage 2 — Enrichment (Cleaning, Grouping, and Merging)

The smoke runner script `scripts/run_parse_sample.py` performs both stages
end-to-end on the included OpenStack-style sample data (`data/sample_raw.csv`).

Contract (what Phase 1 guarantees)
----------------------------------
- Inputs: text log lines in `data/sample_raw.csv` (OpenStack format used for
  the sample runner).
- Stage 1 output: `data/sample_raw_structured.csv` — one row per log line
  with extracted metadata columns (timestamp, component, reqid, template_id,
  method, url, status, etc.). Diagnostic columns (e.g. ParseError/Raw)
  are removed before persisting.
- Stage 2 output: `data/sample_raw_cleaned_templates.csv` (cleaned/grouped
  templates) and `data/sample_raw_merged.csv` (aggregated, merged view).
- Merge is mandatory: the pipeline treats merging as a required step and the
  sample runner will fail with a clear RuntimeError if merging cannot be
  imported or fails.

Stage 1 — Parsing the logs
--------------------------
What it does
- Load raw logs (line-oriented) and apply parsing rules and regex-based
  metadata extraction.
- Centralized metadata extraction returns lowercase keys to keep column
  names consistent.
- Attempts to reconstruct timestamps if they were split (Date + Time) and
  coerces with pandas `to_datetime(..., errors='coerce')`.

Key code
- `src/parsing/metadata_drain_parser.py` — entry point used by the sample
  runner; lazy-loads drain3 to avoid heavy imports during test runs.
- `src/parsing/metadata_utils.py` — centralized metadata extraction helpers
  and conservative regex patterns.
- `src/parsing/log_parser.py` and `src/parsing/regex_utils.py` — testable
  primitives used by the higher-level parser.

Stage 2 — Enrichment (Cleaning, Grouping, Merging)
-------------------------------------------------
What it does
- Clean and normalize template text (the `clean_template` logic).
- Group cleaned templates into semantic rows with `semantic_text`,
  `template_ids` and `occurrences`.
- Merge the grouped templates with structured metadata (left-join on
  normalized template ids), apply conservative remapping heuristics and an
  optional fuzzy remap (requires `rapidfuzz`), then aggregate back to one
  row per `semantic_text`.

Key code
- `src/enrichment/cleaning.py` — `clean_template` and `minimal_cleaning`
  implementations. `minimal_cleaning` calls `clean_template` to normalize
  templates before grouping.
- `src/enrichment/merger.py` — `merge_structured_metadata(cleaned_df,
  structured_df, output_path, ...)` performs the merge and writes
  diagnostics to `merge_diagnostics.json` and an `unresolved_template_ids.csv`
  sample next to the output.

How the sample runner integrates both stages
--------------------------------------------
- `scripts/run_parse_sample.py` runs Stage 1 (parser) to produce
  `data/sample_raw_structured.csv` and templates CSV.
- It then runs `minimal_cleaning()` and writes
  `data/sample_raw_cleaned_templates.csv` and `_cleaned_templates.csv`.
- Finally, it calls `merge_structured_metadata(...)` and writes
  `data/sample_raw_merged.csv`. Merging is mandatory: the runner logs
  tracebacks (at debug level) and raises a RuntimeError on import or
  merge failure.

Commands — quickstart
---------------------
Create & activate the virtual environment (macOS/Linux zsh):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the minimal test deps (recommended; some features are optional):

```bash
pip install -r requirements.txt
# For fuzzy remapping (optional):
pip install rapidfuzz
```

Run unit tests:

```bash
./.venv/bin/python -m pytest tests/unit -q
```

Run the sample end-to-end (parsing + cleaning + merging):

```bash
./.venv/bin/python scripts/run_parse_sample.py
```

Expected artifacts (in `data/` after the script finishes):
- `sample_raw_structured.csv` — parsed log rows with metadata
- `sample_raw_templates.csv` — raw templates extracted during parsing
- `sample_raw_cleaned_templates.csv` — cleaned/grouped templates
- `sample_raw_merged.csv` — aggregated merge of cleaned templates with metadata
- `merge_diagnostics.json` and `unresolved_template_ids.csv` (diagnostic outputs)

Testing & validation
--------------------
- Unit tests are in `tests/unit/` and cover parsing regexes, cleaning,
  and merge logic. The CI should run these tests on each change.
- The sample runner acts as a smoke test for the end-to-end flow.

Design notes & edge cases
-------------------------
- Keys are normalized to lowercase to make CSV column names stable.
- The merge function applies conservative remapping heuristics first
  (numeric normalization, suffix/prefix matching) before attempting fuzzy
  remapping. Fuzzy remapping requires `rapidfuzz` and is optional.
- The merger writes diagnostics but does not crash the process unless the
  merge function itself raises; the sample runner enforces merge as a
  required step and will fail loudly if merging cannot be performed.

Next steps (phase roadmap)
--------------------------
- Phase 2: Efficiency — batch chroma queries, parallel encoding, reduce
  memory use.
- Phase 3: Packaging & automation — CLI, GitHub Actions CI, clear release
  artifacts and docs site.

Files touched in Phase 1 (high level)
- `src/parsing/*` — parsing primitives and helpers
- `src/enrichment/cleaning.py`, `src/enrichment/merger.py`
- `scripts/run_parse_sample.py` — smoke-runner (now performs merge)
- `tests/unit/*` — unit tests added/updated for the refactor

If you'd like, I can:
- add a short integration test that runs the sample runner and asserts the
  merged CSV exists and contains expected columns, or
- add CLI flags to `scripts/run_parse_sample.py` for more flexible
  invocation (e.g. `--no-merge`, `--outdir`).

---

Phase 1 is intended to be conservative, readable, and well-tested. The
merge step is intentionally mandatory in the sample runner because later
pipeline stages depend on its outputs.
Phase 1 Quickstart — Accuracy & Validation
==========================================

This file documents how to run the Phase 1 unit tests and what was added.

Create & activate virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install minimal deps for tests

```bash
pip install pandas pytest sentence-transformers drain3 chromadb rapidfuzz
```

Run unit tests

```bash
pytest -q
```

Files added in Phase 1

- `src/parsing/regex_utils.py` — conservative regex patterns and helpers
- `src/parsing/log_parser.py` — a testable LogParser (parsing + enrichment)
- `src/enrichment/merger.py` — refactored merge function with validation
- `tests/unit/` — unit tests for regex extraction and merge fallbacks

Next steps

- Phase 2: Efficiency (batching Chroma queries, parallel encoding)
- Phase 3: Packaging, CLI, CI, and docs site

