#!/usr/bin/env python3
"""Sample a raw log file into a CSV with header 'raw'.
Usage: python scripts/sample_log.py <src_log> <out_csv> [sample_size]
"""
import sys, random, csv, os

def reservoir_sample(src, out, n=1000, seed=42):
    rng = random.Random(seed)
    sample = []
    with open(src, 'r', encoding='utf-8', errors='ignore') as fh:
        for i, line in enumerate(fh):
            line = line.rstrip('\n')
            if i < n:
                sample.append(line)
            else:
                j = rng.randint(0, i)
                if j < n:
                    sample[j] = line
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['raw'])
        for r in sample:
            writer.writerow([r])
    print(f"Wrote sampled CSV: {out} (n={len(sample)})")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    src = sys.argv[1]
    out = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    reservoir_sample(src, out, n)
