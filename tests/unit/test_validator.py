"""Unit tests for the SemanticValidator class."""
import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation import SemanticValidator

class TestSemanticValidator(unittest.TestCase):
    def setUp(self):
        # Mock the sentence_transformers.CrossEncoder class
        self.patcher = patch("src.validation.CrossEncoder")
        self.MockCrossEncoder = self.patcher.start()
        
        # Setup the mock instance
        self.mock_model = MagicMock()
        self.MockCrossEncoder.return_value = self.mock_model
        
        # Initialize validator
        self.validator = SemanticValidator(model_name="test-model")

    def tearDown(self):
        self.patcher.stop()

    def test_init(self):
        """Test initialization loads the model."""
        self.MockCrossEncoder.assert_called_with("test-model", device=None)

    def test_validate_edges_empty(self):
        """Test validation with empty list."""
        result = self.validator.validate_edges([])
        self.assertEqual(result, [])

    def test_validate_edges_scoring(self):
        """Test that edges get updated with scores."""
        # Setup mock return values for predict
        # We have 2 valid edges, so we expect 2 scores
        self.mock_model.predict.return_value = [0.95, 0.15]

        candidates = [
            {
                "source_template": "Service A started",
                "target_template": "Service B responding",
                "hybrid_score": 0.5
            },
            {
                "source_template": "Service A started",
                "target_template": "Error in Service C",
                "hybrid_score": 0.4
            }
        ]

        results = self.validator.validate_edges(candidates)
        
        # Check that predict was called with correct pairs
        expected_pairs = [
            ["Service A started", "Service B responding"],
            ["Service A started", "Error in Service C"]
        ]
        self.mock_model.predict.assert_called()
        call_args = self.mock_model.predict.call_args
        self.assertEqual(call_args[0][0], expected_pairs)

        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["validation_score"], 0.95)
        self.assertEqual(results[1]["validation_score"], 0.15)

    def test_validate_edges_missing_text(self):
        """Test handling of edges missing template text."""
        candidates = [
            {
                "source_template": "Text A",
                # Missing target_template
                "hybrid_score": 0.5
            }
        ]
        
        results = self.validator.validate_edges(candidates)
        
        # Should not call predict for invalid edges
        self.mock_model.predict.assert_not_called()
        
        # Should mark with -1.0 or similar
        self.assertEqual(results[0]["validation_score"], -1.0)

    def test_filter_edges(self):
        """Test filtering based on threshold."""
        # Setup mock scores
        self.mock_model.predict.return_value = [0.8, 0.2, 0.9]

        candidates = [
            {"source_template": "A", "target_template": "B"}, # 0.8 -> Keep
            {"source_template": "C", "target_template": "D"}, # 0.2 -> Drop
            {"source_template": "E", "target_template": "F"}, # 0.9 -> Keep
        ]

        # Filter with threshold 0.5
        filtered = self.validator.filter_edges(candidates, threshold=0.5)
        
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["validation_score"], 0.8)
        self.assertEqual(filtered[1]["validation_score"], 0.9)

if __name__ == "__main__":
    unittest.main()
