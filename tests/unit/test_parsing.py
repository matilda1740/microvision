import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ingestion.batch_loader import BatchDatasetLoader
from src.parsing.drain3_parser import Drain3Parser
from config.settings import settings

import pytest


@pytest.fixture(autouse=True)
def temp_dataset_dir(tmp_path):
    """Create a temporary dataset directory and point global settings to it.

    This fixture will be applied automatically to tests in this module. It
    writes three small sample log files (namenode, datanode, client) to the
    temporary directory and then updates ``settings.DATASET_DIR`` so the
    production loader reads them.
    """
    d = tmp_path
    (d / "namenode_01.log").write_text(
        "2025-11-11 12:00:00 INFO namenode Heartbeat from datanode1\n2025-11-11 12:00:05 ERROR namenode failed to register datanode: timeout\n"
    )
    (d / "datanode_01.log").write_text(
        "2025-11-11 12:00:02 INFO datanode Block received from namenode\n2025-11-11 12:00:07 INFO datanode Replica copying to datanode3\n"
    )
    (d / "client_01.log").write_text(
        "2025-11-11 12:00:10 INFO client Request to namenode: list /files\n2025-11-11 12:00:20 ERROR client failed to read file /data/test.txt: permission denied\n"
    )
    # point the globally-imported settings to this dataset dir
    settings.DATASET_DIR = str(d)
    yield

def test_template_extraction():
    print("🧪 Testing Template Extraction Pipeline...")
    
    # 1. Load sample data from Day 1
    print("1. Loading sample dataset...")
    loader = BatchDatasetLoader(settings)
    sample_logs = loader.load_distributed_dataset()
    
    print(f"   ✅ Loaded {len(sample_logs)} log entries")
    print(f"   Sample log: {sample_logs[0]['raw_log'][:80]}...")
    
    # 2. Initialize parser
    print("2. Initializing Drain3 parser...")
    parser = Drain3Parser(settings)
    
    # 3. Extract raw logs for parsing
    raw_logs = [log['raw_log'] for log in sample_logs[:50]]  # Test with first 50 logs
    
    # 4. Parse logs
    print("3. Parsing log samples...")
    parsed_logs = parser.parse_batch(raw_logs)
    
    print(f"   ✅ Parsed {len(parsed_logs)} logs into templates")
    
    # 5. Analyze results
    print("4. Analyzing template extraction results...")
    
    # Check template diversity
    unique_templates = set()
    for log in parsed_logs:
        if isinstance(log['template'], str):  # It's already a template string
            unique_templates.add(log['template'])
        else:  # It's the entire result object
            unique_templates.add(log['template']['template_mined'])
    print(f"   📊 Unique templates found: {len(unique_templates)}")
    
    # Show sample templates
    print("   📝 Sample templates extracted:")
    for i, template in enumerate(list(unique_templates)[:5]):
        print(f"      {i+1}. {template}")
    
    # 6. Check template statistics
    stats = parser.get_template_stats()
    print(f"   📈 Template statistics:")
    print(f"      - Total clusters: {stats['total_templates']}")
    print(f"      - Total logs processed: {stats['total_logs']}")
    
    # 7. Verify template quality
    print("5. Verifying template quality...")
    quality_issues = []
    
    for i, parsed_log in enumerate(parsed_logs[:10]):  # Check first 10
        template = parsed_log['template']
        original = parsed_log['original_log']
        
        # Basic quality checks
        if template == original:  # No templatization happened
            quality_issues.append(f"Log {i}: No templatization - '{template}'")
        elif '<*>' not in template:  # No parameters extracted
            quality_issues.append(f"Log {i}: No parameters - '{template}'")
        elif len(template) < 10:  # Suspiciously short template
            quality_issues.append(f"Log {i}: Very short template - '{template}'")
    
    if quality_issues:
        print("   ⚠️  Quality issues found:")
        for issue in quality_issues[:3]:  # Show first 3 issues
            print(f"      {issue}")
    else:
        print("   ✅ Template quality looks good!")
    
    # 8. Test with template analyzer
    print("6. Testing semantic categorization...")
    from src.parsing.template_analyzer import TemplateAnalyzer
    analyzer = TemplateAnalyzer()
    
    sample_template = parsed_logs[0]['template'] if parsed_logs else "Test template"
    analysis = analyzer.analyze_template(sample_template)
    print(f"   🔍 Sample template analysis: {analysis}")
    
    results = {
        'success': True,
        'total_parsed': len(parsed_logs),
        'unique_templates': len(unique_templates),
        'sample_templates': list(unique_templates)[:5],
        'stats': stats,
        'quality_issues': len(quality_issues)
    }

    # Assertions for pytest (tests must not return a value)
    assert results['total_parsed'] > 0, "No logs were parsed by the Drain3 parser"
    assert isinstance(stats, dict), "Parser stats should be a dict"
    assert 'total_logs' in stats and stats['total_logs'] == results['total_parsed'], (
        "Stats.total_logs should equal the number of parsed logs"
    )
    assert results['unique_templates'] >= 1, "Expected at least one unique template"

    # Only return the full results when running this module directly
    if __name__ == "__main__":
        return results
    return None

if __name__ == "__main__":
    results = test_template_extraction()
    
    print("\n" + "="*50)
    print("🎯 TEST SUMMARY")
    print("="*50)
    print(f"✅ Parsing Successful: {results['success']}")
    print(f"📊 Logs Processed: {results['total_parsed']}")
    print(f"🎭 Unique Templates: {results['unique_templates']}")
    print(f"⚠️  Quality Issues: {results['quality_issues']}")
    
    if results['quality_issues'] == 0:
        print("\n🎉 Template extraction test PASSED!")
    else:
        print(f"\n🔧 Found {results['quality_issues']} issues to investigate")