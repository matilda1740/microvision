# src/parsing/template_analyzer.py
from collections import Counter
import re
from typing import Any, Dict

class TemplateAnalyzer:
    def __init__(self):
        self.service_patterns = {
            'http_call': r'(call|request|invoke).*http',
            'error': r'(error|fail|exception|timeout)',
            'database': r'(database|db|query|sql)',
            'service_mention': r'([a-zA-Z-]+)-service',
        }
    
    def analyze_template(self, template: str) -> Dict[str, Any]:
        """Perform basic semantic analysis on template"""
        analysis = {
            'likely_service_calls': [],
            'contains_errors': False,
            'mentions_database': False,
            'extracted_services': []
        }
        
        # Check for service mentions
        service_matches = re.findall(self.service_patterns['service_mention'], template.lower())
        analysis['extracted_services'] = list(set(service_matches))
        
        # Check for HTTP calls
        if re.search(self.service_patterns['http_call'], template.lower()):
            analysis['likely_service_calls'] = analysis['extracted_services']
            
        # Check for errors
        analysis['contains_errors'] = bool(re.search(self.service_patterns['error'], template.lower()))
        
        # Check for database mentions
        analysis['mentions_database'] = bool(re.search(self.service_patterns['database'], template.lower()))
        
        return analysis