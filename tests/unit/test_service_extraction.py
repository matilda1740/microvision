import pandas as pd
import pytest
from src.parsing.regex_utils import extract_service_from_path

def extract_service_from_filename(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates the logic from scripts/run_full_pipeline.py to extract service names.
    """
    if 'raw' in df.columns:
        extracted = df['raw'].apply(extract_service_from_path)
        if extracted.notna().any():
            df['service'] = extracted
    return df

def test_service_extraction_simple():
    df = pd.DataFrame({
        'raw': ['nova-api.log', 'nova-compute.log', 'neutron-server.log']
    })
    df = extract_service_from_filename(df)
    assert df['service'].tolist() == ['nova-api', 'nova-compute', 'neutron-server']

def test_service_extraction_paths():
    df = pd.DataFrame({
        'raw': ['/var/log/nova/nova-api.log', '/var/log/neutron/neutron-server.log']
    })
    df = extract_service_from_filename(df)
    assert df['service'].tolist() == ['nova-api', 'neutron-server']

def test_service_extraction_quoted():
    df = pd.DataFrame({
        'raw': ['"nova-api.log"', '"/var/log/nova/nova-compute.log"']
    })
    df = extract_service_from_filename(df)
    assert df['service'].tolist() == ['nova-api', 'nova-compute']

def test_service_extraction_mixed_content():
    # Simulating what might be in the 'raw' column if it contains more than just the filename
    df = pd.DataFrame({
        'raw': ['nova-api.log: 2023-01-01 INFO...', '/var/log/nova/nova-scheduler.log: some error']
    })
    df = extract_service_from_filename(df)
    assert df['service'].tolist() == ['nova-api', 'nova-scheduler']

def test_service_extraction_no_match():
    df = pd.DataFrame({
        'raw': ['random_file.txt', 'syslog']
    })
    df = extract_service_from_filename(df)
    # Should not create 'service' column if no matches, or leave it as NaN if column existed
    # The function creates the column if at least one match is found.
    # If no matches are found, extracted_service[0].notna().any() will be False.
    assert 'service' not in df.columns

def test_service_extraction_partial_match():
    df = pd.DataFrame({
        'raw': ['nova-api.log', 'unknown.txt']
    })
    df = extract_service_from_filename(df)
    assert df['service'].iloc[0] == 'nova-api'
    assert pd.isna(df['service'].iloc[1])
