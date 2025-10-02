# src/ingestion/batch_loader.py
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import re

class BatchDatasetLoader:
    def __init__(self, settings):
        self.settings = settings
        self.dataset_path = Path(settings.DATASET_DIR)
        
    def load_distributed_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset with component context for dependency analysis"""
        enriched_logs = []
        
        # Process each component's log files
        for component in ["namenode", "datanode", "client"]:
            component_logs = self._load_component_logs(component)
            enriched_logs.extend(component_logs)
            
        return enriched_logs
    
    def _load_component_logs(self, component: str) -> List[Dict[str, Any]]:
        """Load and enrich logs for a specific system component"""
        component_files = self._find_component_files(component)
        logs = []
        
        for file_path in component_files:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f):
                    if line_num >= self.settings.MAX_LOG_LINES:
                        break
                        
                    if line.strip():
                        enriched_log = self._enrich_log_line(line.strip(), component)
                        logs.append(enriched_log)
                        
        return logs
    
    def _enrich_log_line(self, log_line: str, component: str) -> Dict[str, Any]:
        """Add semantic context to raw log lines"""
        return {
            'raw_log': log_line,
            'component': component,
            'component_type': self.settings.DATASET_CONFIG["component_metadata"][component][0],
            'log_category': self._categorize_log_line(log_line),
            'timestamp': self._extract_timestamp(log_line),
            'contains_service_mention': self._contains_service_mention(log_line)
        }
    
    def _categorize_log_line(self, log_line: str) -> str:
        """Categorize log line for semantic analysis"""
        categories = {
            'HEARTBEAT': r'(heartbeat|reporting)',
            'BLOCK_OPERATION': r'(block|replication|copying)',
            'METADATA_OPERATION': r'(getBlock|createFile|delete)',
            'ERROR': r'(ERROR|Exception|failed|timeout)',
            'RESOURCE_MANAGEMENT': r'(allocating|registering|removing)'
        }
        
        for category, pattern in categories.items():
            if re.search(pattern, log_line, re.IGNORECASE):
                return category
        return 'OTHER'
    
    def _extract_timestamp(self, log_line: str) -> str:
        """Extract timestamp from log line"""
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}'
        ]
        
        for pattern in timestamp_patterns:
            match = re.search(pattern, log_line)
            if match:
                return match.group()
        return "unknown"
    
    def _contains_service_mention(self, log_line: str) -> bool:
        """Check if log line mentions other services/components"""
        service_indicators = [
            r'to [a-zA-Z]+', r'from [a-zA-Z]+', r'calling [a-zA-Z]+',
            r'request to [a-zA-Z]+', r'response from [a-zA-Z]+'
        ]
        return any(re.search(indicator, log_line, re.IGNORECASE) for indicator in service_indicators)
    
    def _find_component_files(self, component: str) -> List[Path]:
        """Find log files for a specific component"""
        # Look for component-specific files
        component_patterns = [
            f"*{component}*.log",
            f"*{component}*.txt",
            f"*{component.upper()}*"
        ]
        
        files = []
        for pattern in component_patterns:
            files.extend(self.dataset_path.glob(pattern))
            
        return files[:2]  # Limit to 2 files per component for manageability