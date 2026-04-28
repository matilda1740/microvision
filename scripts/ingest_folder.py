"""Helper script to ingest a folder of log files into a single CSV.

This script is useful for datasets where logs are split across
multiple files (e.g. one per service). It reads all files matching a pattern,
extracts the service name from the filename, and writes a unified CSV with
'raw' and 'service' columns.

Usage:
    python scripts/ingest_folder.py --folder data/raw_logs/openstack --out data/openstack_merged.csv
"""
import argparse
import csv
import glob
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Folder containing log files")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--pattern", default="*.log", help="File pattern to match (default: *.log)")
    args = parser.parse_args()

    folder = Path(args.folder)
    out_path = Path(args.out)
    
    if not folder.exists():
        print(f"Error: Folder {folder} does not exist.")
        return

    print(f"Scanning {folder} for {args.pattern}...")
    files = list(folder.glob(args.pattern))
    print(f"Found {len(files)} files.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["raw", "service"])

        for file_path in files:
            # Extract service name from filename
            # We take the stem.
            filename = file_path.name
            service_name = filename.split(".")[0]
            
            print(f"Processing {filename} -> service: {service_name}")
            
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f_in:
                    for line in f_in:
                        line = line.strip()
                        if line:
                            writer.writerow([line, service_name])
                            total_lines += 1
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    print(f"Done. Wrote {total_lines} lines to {out_path}")

if __name__ == "__main__":
    main()
