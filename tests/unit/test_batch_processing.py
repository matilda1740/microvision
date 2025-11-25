# tests/unit/test_batch_processing.py
import sys
import os
import pytest


sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.settings import Settings
from src.ingestion.batch_loader import BatchDatasetLoader

@pytest.fixture
def settings():
    return Settings()

@pytest.fixture
def sample_logs_dir(settings, tmp_path):
    # Create a temporary dataset directory with small sample logs for testing
    d = tmp_path
    # update settings to point to this temporary dir
    settings.DATASET_DIR = str(d)

    # create sample files for the components the loader expects
    namenode = d / "namenode_01.log"
    namenode.write_text(
        "2025-11-11 12:00:00 INFO namenode Heartbeat from datanode1\n2025-11-11 12:00:05 ERROR namenode failed to register datanode: timeout\n"
    )
    datanode = d / "datanode_01.log"
    datanode.write_text(
        "2025-11-11 12:00:02 INFO datanode Block received from namenode\n2025-11-11 12:00:07 INFO datanode Replica copying to datanode3\n"
    )
    client = d / "client_01.log"
    client.write_text(
        "2025-11-11 12:00:10 INFO client Request to namenode: list /files\n2025-11-11 12:00:20 ERROR client failed to read file /data/test.txt: permission denied\n"
    )

    return str(d)

# tests/unit/test_batch_processing.py
import pytest

class TestBatchProcessing:
    def test_dataset_loading(self, settings, sample_logs_dir):
        loader = BatchDatasetLoader(settings)
        logs = loader.load_distributed_dataset()
        
        assert len(logs) > 0
        assert 'component' in logs[0]
        assert 'log_category' in logs[0]
        
    def test_empty_dataset_dir(self, settings):
        # Test with non-existent directory
        settings.DATASET_DIR = "non_existent_dir"
        loader = BatchDatasetLoader(settings)
        logs = loader.load_distributed_dataset()
        assert len(logs) == 0  # Should handle empty dir gracefully