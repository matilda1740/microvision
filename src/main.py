# # src/main.py
# class MicroVisionPipeline:
#     def __init__(self):
#         self.settings = settings
#         self.parser = Drain3Parser(settings)
#         self.storage = LogStorage(settings)
#         self.loader = BatchDatasetLoader(settings)  # Replaces LogCollector
        
#     def process_dataset(self) -> Dict[str, Any]:
#         """Main batch processing method for thesis evaluation"""
#         logger.info("Starting batch dataset processing")
        
#         # Load and process dataset
#         enriched_logs = self.loader.load_distributed_dataset()
#         raw_logs = [log['raw_log'] for log in enriched_logs]
        
#         # Parse logs
#         parsed_logs = self.parser.parse_batch(raw_logs)
        
#         # Combine parsing results with enrichment
#         combined_logs = self._combine_parsed_enriched(parsed_logs, enriched_logs)
        
#         # Store results
#         self.storage.store_parsed_logs(combined_logs)
        
#         # Generate analysis report
#         report = self._generate_analysis_report(combined_logs)
        
#         logger.info(f"Batch processing complete: {len(combined_logs)} logs processed")
#         return report
    
#     def _combine_parsed_enriched(self, parsed_logs, enriched_logs):
#         """Combine Drain3 parsing with semantic enrichment"""
#         combined = []
#         for i, (parsed, enriched) in enumerate(zip(parsed_logs, enriched_logs)):
#             combined_log = {**parsed, **enriched}
#             combined_log['log_id'] = f"log_{i}"
#             combined.append(combined_log)
#         return combined
    
#     def _generate_analysis_report(self, processed_logs: List[Dict]) -> Dict[str, Any]:
#         """Generate comprehensive analysis report"""
#         # Component distribution
#         components = [log['component'] for log in processed_logs]
#         component_counts = pd.Series(components).value_counts().to_dict()
        
#         # Template statistics
#         template_stats = self.parser.get_template_stats()
        
#         # Semantic analysis
#         log_categories = [log['log_category'] for log in processed_logs]
#         category_counts = pd.Series(log_categories).value_counts().to_dict()
        
#         return {
#             'processing_summary': {
#                 'total_logs_processed': len(processed_logs),
#                 'unique_templates': template_stats['total_templates'],
#                 'processing_timestamp': pd.Timestamp.now().isoformat()
#             },
#             'component_breakdown': component_counts,
#             'semantic_categories': category_counts,
#             'template_statistics': template_stats,
#             'potential_dependencies': self._identify_potential_dependencies(processed_logs)
#         }
    
#     def _identify_potential_dependencies(self, processed_logs: List[Dict]) -> List[Dict]:
#         """Identify potential service dependencies from enriched logs"""
#         dependencies = []
        
#         for log in processed_logs:
#             if log['contains_service_mention']:
#                 dependency = {
#                     'source_component': log['component'],
#                     'evidence': log['raw_log'],
#                     'log_category': log['log_category'],
#                     'template': log['template'],
#                     'confidence': 'high' if log['log_category'] in ['HEARTBEAT', 'BLOCK_OPERATION'] else 'medium'
#                 }
#                 dependencies.append(dependency)
                
#         return dependencies[:10]  # Return top 10 for initial analysis