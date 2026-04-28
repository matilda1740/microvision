"""Extract 'Dynamic Ground Truth' from OpenStack logs using Trace IDs.

This script parses a log file (CSV or raw text), extracts 'req-id' traces,
reconstructs the call chain (chronological order of services visited), and
outputs a set of 'Proven Edges'.

Rationale:
    In OpenStack, a `req-id` (e.g., `req-67a4...`) is often propagated across
    synchronous API calls. By tracking the sequence of services that log the
    same `req-id`, we can infer the actual runtime dependencies.

Limitations:
    - Async flows (RabbitMQ) might lose the req-id.
    - Clock skew between services can mess up temporal ordering (though usually minimal).

Usage:
    python scripts/extract_gold_standard.py --input data/archive_mixed/sample_raw.csv --output data/gold_standard_edges.csv
"""
import argparse
import csv
import re
import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Set, Dict

# Regex for request ID (OpenStack standard format)
# Matches: "req-67a483e0-245e-4a29-b138-bfb9e6b46f3b"
REQ_ID_PATTERN = re.compile(r"(req-[a-f0-9-]{36}|req-[a-f0-9-]+)")

# Regex for Timestamp (Standard OpenStack/LogLog format)
# Matches: "2017-05-16 22:33:39.958"
TIMESTAMP_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")


def extract_traces(df: pd.DataFrame) -> Dict[str, List[Tuple[pd.Timestamp, str]]]:
    """
    Groups logs by req-id and returns a sorted list of (time, service) tuples for each trace.
    """
    traces = defaultdict(list)
    
    # Pre-compile regex usage
    # We iterate manually for speed and better error handling on weird lines
    print(f"Scanning {len(df)} rows for traces...")
    
    count_found = 0
    
    for idx, row in df.iterrows():
        line = str(row.get("raw", ""))
        service = str(row.get("service", ""))
        if service == "None": 
            service = ""
        
        # If service not in column, try to extract from text (Fallback for raw files)
        if not service or service == "nan":
            # Heuristic: "nova-api.log.1...." -> "nova-api"
            # Or "nova-compute..."
            parts = line.split(" ")
            if parts:
                first_token = parts[0]
                if ".log" in first_token:
                    service = first_token.split(".")[0]
                    # DEBUG
                    # if idx < 5:
                    #     print(f"DEBUG: Extracted service '{service}' from '{first_token}'")
                else:
                    # try to find standard openstack service names
                    match = re.search(r"\b(nova-[a-z]+|cinder-[a-z]+|neutron-[a-z]+|glance-[a-z]+|keystone|swift-[a-z]+)\b", line)
                    if match:
                        service = match.group(1)
        
        # Extract Req ID
        req_match = REQ_ID_PATTERN.search(line)
        if not req_match:
            continue
            
        req_id = req_match.group(1)
        
        # Extract Timestamp
        ts_match = TIMESTAMP_PATTERN.search(line)
        if not ts_match:
            continue
            
        ts_str = ts_match.group(1)
        try:
            timestamp = pd.to_datetime(ts_str)
        except:
            continue
            
        # if service == 'None':
        #      print(f"DEBUG: Service is string 'None' at index {idx}. Line: {line[:50]}...")
             
        if service and service != "nan":
            traces[req_id].append((timestamp, service))
            count_found += 1

    print(f"Found {len(traces)} unique traces across {count_found} log lines.")
    return traces

def generate_edges(traces: Dict[str, List[Tuple[pd.Timestamp, str]]]) -> pd.DataFrame:
    """
    Converts traces into a DataFrame of directed edges with counts.
    """
    edge_counts = defaultdict(int) 
    # Key: (source, target), Value: count
    
    trace_examples = {} 
    # Key: (source, target), Value: example req_id
    
    print("Reconstructing call chains...")
                
    for req_id, events in traces.items():
        # Sort by time
        events.sort(key=lambda x: x[0])
        
        # Iterate through the sequence
        for i in range(len(events) - 1):
            t1, s1 = events[i]
            t2, s2 = events[i+1]
            
            # Identify a transition
            if s1 != s2:
                # We interpret this as s1 calling s2 (or passing control to s2)
                edge_counts[(s1, s2)] += 1
                if (s1, s2) not in trace_examples:
                    trace_examples[(s1, s2)] = req_id

    # Format as DataFrame
    data = []
    for (src, dst), count in edge_counts.items():
        data.append({
            "source": src,
            "target": dst,
            "weight": count,
            "type": "dynamic_ground_truth",
            "example_trace": trace_examples.get((src, dst), "")
        })
    
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description="Extract Golden Standard edges from OpenStack logs via Trace IDs")
    parser.add_argument("--input", required=True, help="Path to raw logs CSV")
    parser.add_argument("--output", required=True, help="Path to output edges CSV")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)
        
    print(f"Loading {input_path}...")
    # Try different CSV loading strategies
    try:
        # If it's a .log or .txt file, read as raw lines to avoid CSV parsing issues (quotes, commas)
        if input_path.suffix in [".log", ".txt"]:
            with open(input_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            df = pd.DataFrame(lines, columns=["raw"])
            # Strip newlines
            df["raw"] = df["raw"].str.strip()
            df["service"] = None
        else:
            # Assume proper CSV
            df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
        
    # Check columns
    if "raw" not in df.columns:
        # If it's a raw file with no header, assume first column is raw
        print("Warning: 'raw' column not found. Using first column as content.")
        df.rename(columns={df.columns[0]: "raw"}, inplace=True)
        # Create empty service column if missing
        if "service" not in df.columns:
            df["service"] = None

    traces = extract_traces(df)
    edges_df = generate_edges(traces)
    
    print(f"Identified {len(edges_df)} unique directed edges.")
    print(edges_df.sort_values("weight", ascending=False).head(10).to_string())
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edges_df.to_csv(output_path, index=False)
    print(f"Saved Dynamic Ground Truth to {output_path}")

if __name__ == "__main__":
    main()
