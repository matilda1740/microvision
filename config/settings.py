# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Dataset paths
    DATA_DIR = "data"
    PROCESSED_DIR = "data/processed"
    
    # Batch processing configuration
    BATCH_SIZE = 1000
    MAX_LOG_LINES = 10000
    
    DRAIN_CONFIG = {
        "persistence_type": "memory",
        "parametrize_numeric_tokens": True,
        "parametrize_suffix": ["_id", "_num", "_ip"],
        "parametrize_prefix": [],
        "depth": 4, 
        "sim_th": 0.4,  
        "max_children": 100,
        "max_clusters": 1000,
        "extra_delimiters": [":", ",", "=", ";", "|"], 
        "masking": True,           
        # "template_format": "{{}}", 
        # "cluster_merge_searcher": "none"  
    }
    
    # ChromaDB configuration
    CHROMA_DIR = "data/chroma_db"
    # Edge guard thresholds (tunable)
    MAX_TOTAL_EDGES = 200000
    PER_SOURCE_TOP_K_CAP = 100
    
    # Model configuration
    # Default embedding model used by the runner. Updated to match notebook default.
    EMBEDDING_MODEL = "all-mpnet-base-v2"

    # Runner defaults (centralized from notebook/script ad-hoc values)
    DEFAULT_SAMPLE = 1000
    DEFAULT_DB = "data/edges/edges.db"
    DEFAULT_CHROMA_DIR = "data/chroma_db/chroma_smoke"
    # Directory where embedding artifacts (embeddings.npy, embeddings_index.csv, etc.) are written
    DEFAULT_EMBEDDINGS_DIR = "data/edges"
    DEFAULT_TOP_K = 10
    DEFAULT_THRESHOLD = 0.2
    # Timestamp normalization policy: one of 'median', 'latest', 'earliest', 'first'
    DEFAULT_TIMESTAMP_POLICY = "median"
    # Default metadata columns to include when building metadatas for vector stores.
    # Can be overridden per-call by passing `meta_cols` to `df_to_metadatas`.
    DEFAULT_META_COLS = [
        "template_id",
        "template",
        "component",
        "semantic_text",
        "doc_id",
        "orig_idx",
        "timestamp",
    ]
    DEFAULT_ALPHA = 0.5
    DEFAULT_DEVICE = None

    # Service name mapping path (can be overridden via SERVICE_NAME_MAP_PATH env var)
    # Moved out of source tree to a data/ location to separate code and runtime data.
    SERVICE_NAME_MAP_PATH = "data/service_name_map.json"
    
    # Dataset-specific settings
    DATASET_CONFIG = {
        "component_metadata": {
            "namenode": ["coordinator", "metadata_operations"],
            "datanode": ["storage", "block_operations"], 
            "client": ["user_operations", "requests"]
        }
    }

    # Log format mappings used by Drain / parser to assist in parsing structured logs.
    # Keys are dataset/product names and values are drain-style format strings.
    LOG_FORMAT_MAPPINGS = {
        "OpenStack": "<Date> <Time> <Pid> <Level> <Component> <Content>",
        "Hadoop": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
        "HDFS": "<Date> <Time> <Level> <Component>: <Content>",
        "Spark": "<Date> <Time> <Level> <Component>: <Content>",
        "Zookeeper": "<Date> <Time> <Level> <Component>: <Content>",
    }

    # Optional FIELD_CONFIG for metadata extraction. If provided, it will be
    # used by MetadataDrainParser to choose which fields to extract and prefer
    # when building metadata. Keys should match the structure expected by the
    # parser (core_fields, enrich_fields, metadata_fields).
    FIELD_CONFIG = {
        "core_fields": ["Component", "Level", "Method", "URL"],
        "enrich_fields": [
            "ReqID",
            "UserID",
            "TenantID",
            "IP",
            "Status",
            "ResponseLength",
            "ResponseTime",
            "Service",
        ],
        "metadata_fields": [
            "Component",
            "Level",
            "Pid",
            "ReqID",
            "UserID",
            "TenantID",
            "IP",
            "Status",
            "Method",
            "URL",
            "ResponseLength",
            "ResponseTime",
            "Service",
        ],
    }

settings = Settings()