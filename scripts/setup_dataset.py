# scripts/setup_dataset.py
import requests
import zipfile
from pathlib import Path

def setup_loghub_dataset():
    """Download and setup LogHub dataset for batch processing"""
    dataset_dir = Path("data/datasets")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # For thesis purposes, we'll use a sample approach
    # In actual implementation, download from LogHub
    sample_logs = {
        "namenode_sample.log": [
            "2023-01-01 10:00:00 INFO NameNode: Received block report from datanode_123",
            "2023-01-01 10:00:01 INFO NameNode: Processing heartbeat from datanode_456",
            "2023-01-01 10:00:02 ERROR NameNode: Failed to connect to datanode_789",
            "2023-01-01 10:00:03 INFO NameNode: Client request for file /user/data/file1"
        ],
        "datanode_sample.log": [
            "2023-01-01 10:00:00 INFO DataNode: Sending block report to NameNode",
            "2023-01-01 10:00:01 INFO DataNode: Received heartbeat request from NameNode",
            "2023-01-01 10:00:02 INFO DataNode: Replicating block to datanode_999",
            "2023-01-01 10:00:03 ERROR DataNode: Disk write failure on block replication"
        ]
    }
    
    for filename, logs in sample_logs.items():
        with open(dataset_dir / filename, 'w') as f:
            f.write('\n'.join(logs))
    
    print("Sample dataset created for batch processing")

if __name__ == "__main__":
    setup_loghub_dataset()