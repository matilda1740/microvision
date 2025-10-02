# In config/settings.py 
import os
from dotenv import load_dotenv

load_dotenv()  # Loads from .env file, falls back to defaults

class Settings:
    # Dataset paths
    DATASET_DIR = "data/datasets"
    PROCESSED_DIR = "data/processed"
    
    # Batch processing configuration
    BATCH_SIZE = 1000
    MAX_LOG_LINES = 10000  # For thesis scalability
    
    # Drain3 configuration (unchanged)
    DRAIN_CONFIG = {
        "persistence_type": "memory",
        "parametrize_numeric_tokens": True,
        "depth": 4,
        "sim_th": 0.4
    }
    
    # Dataset-specific settings
    DATASET_CONFIG = {
        "component_metadata": {
            "namenode": ["coordinator", "metadata_operations"],
            "datanode": ["storage", "block_operations"],
            "client": ["user_operations", "requests"]
        }
    }
    
settings = Settings()