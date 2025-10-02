# tests/__init__.py (empty)

# tests/conftest.py
import sys
import os
import pytest


# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.settings import Settings
from src.ingestion.batch_loader import BatchDatasetLoader

@pytest.fixture
def settings():
    return Settings()

@pytest.fixture
def sample_logs_dir(settings):
    return os.path.join(settings.DATASET_DIR)

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