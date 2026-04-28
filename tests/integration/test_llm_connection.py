import sys
from pathlib import Path
import logging

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation import LLMValidator
from config.settings import settings

# Configure simple logging
logging.basicConfig(level=logging.INFO)

def test_llm_validation():
    print(f"Testing LLM Validation with model: {getattr(settings, 'OLLAMA_MODEL', 'llama3.1')}")
    
    try:
        validator = LLMValidator()
    except ImportError:
        print("Error: Ollama not installed or configured.")
        return
    except Exception as e:
        print(f"Error initializing validator: {e}")
        return

    # Test Case 1: Likely Causal
    log_a = "nova-api: Received request to create instance i-12345"
    log_b = "nova-compute: Spawning instance i-12345 on node-1"
    
    print(f"\n--- Test Case 1 (Likely Causal) ---")
    print(f"A: {log_a}")
    print(f"B: {log_b}")
    result = validator.validate_edge(log_a, log_b)
    print(f"Result: {result}")
    
    # Test Case 2: Unlikely Causal
    log_c = "neutron-server: Deleting network net-99"
    log_d = "cinder-api: Volume vol-55 created"
    
    print(f"\n--- Test Case 2 (Unlikely Causal) ---")
    print(f"A: {log_c}")
    print(f"B: {log_d}")
    result = validator.validate_edge(log_c, log_d)
    print(f"Result: {result}")

if __name__ == "__main__":
    test_llm_validation()
