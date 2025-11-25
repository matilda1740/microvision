# src/parsing/drain3_parser.py
import json
import logging
import re
from typing import List, Dict, Any

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

class Drain3Parser:
    def __init__(self, settings):
        self.settings = settings
        self.template_miner = self._setup_drain3()
        self.log_count = 0
        
    def _setup_drain3(self) -> TemplateMiner:
        """Initialize Drain3 template miner with correct configuration"""
        config = TemplateMinerConfig()
        config.load(self.settings.DRAIN_CONFIG)
        
        # Set configuration parameters
        config.profiling_enabled = False
        config.drain_sim_th = self.settings.DRAIN_CONFIG["sim_th"]
        config.drain_depth = self.settings.DRAIN_CONFIG["depth"]
        config.drain_max_children = self.settings.DRAIN_CONFIG["max_children"]
        config.drain_extra_delimiters = [":", ",", ";", "=", "(", ")", "[", "]", "{", "}", " ", "\t"]
        config.parametrize_numeric_tokens = self.settings.DRAIN_CONFIG["parametrize_numeric_tokens"]
        config.parametrize_suffix = self.settings.DRAIN_CONFIG["parametrize_suffix"]
        config.parametrize_prefix = self.settings.DRAIN_CONFIG["parametrize_prefix"]
        # config.persistence_type = self.settings.DRAIN_CONFIG["persistence_type"]
        
        template_miner = TemplateMiner(config=config)
        # logging.info(f"Drain3 initialized with depth={config.drain_depth}, sim_th={config.drain_sim_th}")
        return template_miner


    def parse_batch(self, log_lines: List[str])-> List[Dict[str, Any]]:
        parsed_logs = []
        
        print("=== DRAIN3 DEBUG ===")
        for line in log_lines:
            try:
                # Skip empty lines
                if not line.strip():
                    continue

                # Split into parts: [timestamp] [level] [component] [message...]
                parts = line.strip().split(" ", 4)
                if len(parts) < 5:
                    continue  # skip malformed lines

                message = parts[4]  # message part goes into Drain3
                result = self.template_miner.add_log_message(message)

                # template = self._extract_template(result)
                # cluster_id = self._extract_cluster_id(result)

                self.log_count += 1 
                parsed_log = {
                    'original_log': line.strip(),
                    'template': result["template_mined"],
                    'parameters': result.get("parameter_list", []),
                    'cluster_id': result["cluster_id"],
                    'template_id': f"template_{result['cluster_id']}",
                    'log_count': self.log_count
                }
                parsed_logs.append(parsed_log)
                
            except Exception as e:
                print(f"ERROR parsing: {line} -> {e}")  # Debug info
                continue
                
        return parsed_logs
    
    def _extract_template(self, result) -> str:
        """Extract template from Drain3 result (handles different versions)"""
        try:
            # Try different possible response structures
            if hasattr(result, 'template'):
                return result.template
            elif isinstance(result, dict) and 'template' in result:
                return result['template']
            elif hasattr(result, 'get_template'):
                return result.get_template()
            else:
                # If we can't extract template, return the original approach
                return str(result)
        except:
            return "unknown_template"
    
    def _extract_cluster_id(self, result) -> int:
        """Extract cluster ID from Drain3 result"""
        try:
            if hasattr(result, 'cluster_id'):
                return result.cluster_id
            elif isinstance(result, dict) and 'cluster_id' in result:
                return result['cluster_id']
            else:
                return self.log_count  # Fallback
        except:
            return self.log_count
    
    
    def get_template_stats(self) -> Dict[str, Any]:
        """Get statistics about parsed templates"""
        try:
            clusters = []
            for cluster in self.template_miner.drain.clusters:
                clusters.append({
                    'cluster_id': cluster.cluster_id,
                    'template': cluster.get_template(),
                    'size': cluster.size
                })
            
            return {
                'total_templates': len(self.template_miner.drain.clusters),
                'total_logs': self.log_count,
                'clusters': clusters
            }
        except Exception as e:
            logging.error(f"Error getting template stats: {e}")
            return {
                'total_templates': 0,
                'total_logs': self.log_count,
                'clusters': []
            }