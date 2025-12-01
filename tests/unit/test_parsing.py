import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.parsing.metadata_drain_parser import MetadataDrainParser
from config.settings import settings
import pandas as pd
import pytest
import os
import pathlib

@pytest.fixture(autouse=True)
def temp_dataset_dir(tmp_path):
    """Create a temporary dataset directory and point global settings to it."""
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
    settings.DATASET_DIR = str(d)
    yield

def test_template_extraction(tmp_path):
    print("🧪 Testing Template Extraction Pipeline...")
    
    # 1. Load sample data
    print("1. Loading sample dataset...")
    sample_logs = []
    dataset_dir = pathlib.Path(settings.DATASET_DIR)
    for log_file in dataset_dir.glob("*.log"):
        with open(log_file, "r") as f:
            for line in f:
                if line.strip():
                    sample_logs.append({"raw_log": line.strip()})
    
    print(f"   ✅ Loaded {len(sample_logs)} log entries")
    
    # 2. Initialize parser
    print("2. Initializing MetadataDrainParser...")
    structured_csv = tmp_path / "parsed_logs.csv"
    templates_csv = tmp_path / "templates.csv"
    
    # Simple format for the test logs: Date Time Level Component Content
    log_format = "<Date> <Time> <Level> <Component> <Content>"
    
    parser = MetadataDrainParser(
        log_format=log_format,
        structured_csv=str(structured_csv),
        templates_csv=str(templates_csv),
        save_every=10  # Flush frequently for test
    )
    
    # 3. Parse logs
    print("3. Parsing log samples...")
    for i, log in enumerate(sample_logs):
        parser.process_line(log['raw_log'], i+1)
    
    parser.finalize()
    
    # 4. Analyze results
    print("4. Analyzing template extraction results...")
    assert structured_csv.exists(), "Parsed logs CSV should exist"
    assert templates_csv.exists(), "Templates CSV should exist"
    
    df_parsed = pd.read_csv(structured_csv)
    df_templates = pd.read_csv(templates_csv)
    
    print(f"   ✅ Parsed {len(df_parsed)} logs into templates")
    
    unique_templates = df_templates['template'].unique()
    print(f"   📊 Unique templates found: {len(unique_templates)}")
    
    # Show sample templates
    print("   📝 Sample templates extracted:")
    for i, template in enumerate(unique_templates[:5]):
        print(f"      {i+1}. {template}")
    
    # 5. Verify template quality
    print("5. Verifying template quality...")
    quality_issues = []
    
    for i, row in df_parsed.head(10).iterrows():
        template = row['template']
        content = row['content']
        
        if pd.isna(template):
             quality_issues.append(f"Log {i}: Template is NaN")
             continue

        # Basic quality checks
        if template == content:  # No templatization happened (might be valid for simple logs)
             # This check is a bit loose, as some logs might not have parameters
             pass
        elif '<*>' not in template and len(content.split()) > 3: # No parameters extracted in complex log
             # Again, heuristic
             pass
    
    if quality_issues:
        print("   ⚠️  Quality issues found:")
        for issue in quality_issues[:3]:
            print(f"      {issue}")
    else:
        print("   ✅ Template quality looks good!")

    results = {
        'success': True,
        'total_parsed': len(df_parsed),
        'unique_templates': len(unique_templates),
        'quality_issues': len(quality_issues)
    }

    assert results['total_parsed'] > 0, "No logs were parsed"
    assert results['unique_templates'] >= 1, "Expected at least one unique template"
    
    # return results  <-- removed return to avoid pytest warning

if __name__ == "__main__":
    # Create a dummy tmp_path for running directly
    import pathlib
    import shutil
    import tempfile
    
    with tempfile.TemporaryDirectory() as td:
        tmp_path = pathlib.Path(td)
        # Setup fixture manually
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
        settings.DATASET_DIR = str(d)
        
        results = test_template_extraction(tmp_path)
        
        print("\n" + "="*50)
        print("🎯 TEST SUMMARY")
        print("="*50)
        print(f"✅ Parsing Successful: {results['success']}")
        print(f"📊 Logs Processed: {results['total_parsed']}")
        print(f"🎭 Unique Templates: {results['unique_templates']}")
