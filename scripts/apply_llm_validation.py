import sqlite3
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
import sys

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation import LLMValidator
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_llm_columns(db_path: str):
    """Ensure the edges table has the necessary columns for LLM results."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    columns = {
        "llm_verification": "TEXT",
        "llm_confidence": "REAL",
        "validation_score": "REAL" # Used for generic numeric scores if needed
    }
    
    cur.execute("PRAGMA table_info(edges)")
    existing_cols = {row[1] for row in cur.fetchall()}
    
    for col, dtype in columns.items():
        if col not in existing_cols:
            logger.info(f"Adding column '{col}' to database...")
            cur.execute(f"ALTER TABLE edges ADD COLUMN {col} {dtype}")
            
    conn.commit()
    conn.close()

def batch_update_edges(db_path: str, updates: List[Dict[str, Any]]):
    """Update rows in the database with validation results."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    for update in updates:
        # JSON serialize the verification dict
        verification_json = json.dumps(update["llm_verification"])
        
        cur.execute("""
            UPDATE edges 
            SET llm_verification = ?, 
                llm_confidence = ?, 
                hybrid_score = ?
            WHERE id = ?
        """, (
            verification_json, 
            update["llm_confidence"], 
            update["hybrid_score"], 
            update["id"]
        ))
        
    conn.commit()
    conn.close()

def run_validation(db_path: str, limit: int = None, threshold: float = 0.5):
    """
    Read edges, run LLM validation, update DB.
    
    Args:
        db_path: Path to SQLite DB.
        limit: Max edges to process (useful for testing).
        threshold: Only validate edges with hybrid_score > threshold (don't waste LLM on trash).
    """
    if not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}")
        return

    # 1. Prepare Schema
    add_llm_columns(db_path)
    
    # 2. Load Validator
    try:
        validator = LLMValidator()
    except Exception as e:
        logger.error(f"Failed to initialize LLMValidator: {e}")
        return

    # 3. Fetch Candidates
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # We select edges that haven't been validated yet (llm_verification IS NULL)
    # And meet a basic quality threshold
    query = f"""
        SELECT id, source_semantic_text, target_semantic_text, hybrid_score
        FROM edges 
        WHERE llm_verification IS NULL 
        AND hybrid_score >= {threshold}
    """
    if limit:
        query += f" LIMIT {limit}"
        
    logger.info(f"Fetching candidates from {db_path}...")
    rows = cur.execute(query).fetchall()
    conn.close()
    
    if not rows:
        logger.info("No edges found requiring validation.")
        return

    logger.info(f"Found {len(rows)} edges to validate.")
    
    # 4. Process Batch
    candidates = [dict(row) for row in rows]
    
    # The validator modifies the list in-place
    # Note: validate_candidates expects standard keys. 
    # Our DB selection maps correctly to what validator needs (source_semantic_text, etc).
    validator.validate_candidates(candidates)
    
    # 5. Save Results
    logger.info("Writing results back to database...")
    batch_update_edges(db_path, candidates)
    logger.info("Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Retroactively apply LLM validation to an existing edge database.")
    parser.add_argument("--db", required=True, help="Path to the SQLite edges database")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of edges to process")
    parser.add_argument("--threshold", type=float, default=0.2, help="Minimum hybrid_score to validate")
    
    args = parser.parse_args()
    
    run_validation(args.db, args.limit, args.threshold)
