#!/usr/bin/env python3
"""Repair numeric tokens wrongly present in `service` lists of merged_templates.csv.

By default writes `data/merged_templates.cleaned.csv` and prints a short summary.
"""
from pathlib import Path
import ast
import re
import csv
import sys
import pandas as pd

NUM_RE = re.compile(r'^-?\d+(?:\.\d+)?$')


def parse_list(v):
    if pd.isna(v):
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str):
        s = v.strip()
        if s.startswith('[') and s.endswith(']'):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except Exception:
                pass
        # fallback: single value
        return [s]
    return [v]


def is_numeric_token(t):
    if isinstance(t, (int, float)):
        return True
    s = str(t).strip()
    return bool(NUM_RE.match(s))


def repair(input_path: Path, output_path: Path, inplace: bool = False):
    df = pd.read_csv(input_path)
    repaired = df.copy()

    repaired['service_clean'] = repaired['service'].apply(parse_list)
    kept = []
    moved = []
    for i, row in repaired.iterrows():
        lst = row['service_clean']
        non_num = [x for x in lst if not is_numeric_token(x)]
        num = [x for x in lst if is_numeric_token(x)]
        kept.append(non_num)
        moved.append(num)

    repaired['service'] = kept
    repaired['service_misaligned_values'] = moved

    out = output_path
    repaired.to_csv(out, index=False)
    total = len(repaired)
    affected = sum(1 for v in moved if v)
    print(f'Wrote cleaned merged file: {out} (rows={total}; affected_rows={affected})')


if __name__ == '__main__':
    inp = Path('data/merged_templates.csv')
    out = Path('data/merged_templates.cleaned.csv')
    if not inp.exists():
        print('Input file not found:', inp)
        sys.exit(2)
    repair(inp, out)
