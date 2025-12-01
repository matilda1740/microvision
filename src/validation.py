"""Cross-Encoder Validation Module.

This module implements the "Generation/Validation" step of the RAG pipeline.
It uses a Cross-Encoder model to re-rank or filter candidate edges found by the
Bi-Encoder retrieval step.

Cross-Encoders are computationally more expensive but significantly more accurate
than Bi-Encoders (Cosine Similarity) because they attend to the full interaction
between the two text inputs.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

logger = logging.getLogger(__name__)


class SemanticValidator:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = None):
        """
        Initialize the Cross-Encoder validator.
        
        Args:
            model_name: HuggingFace model ID. Defaults to a fast, high-quality re-ranker.
            device: 'cpu' or 'cuda'. If None, auto-detects.
        """
        if CrossEncoder is None:
            raise ImportError("sentence-transformers is required for SemanticValidator")
            
        self.model_name = model_name
        logger.info(f"Loading Cross-Encoder model: {model_name}")
        self.model = CrossEncoder(model_name, device=device)

    def validate_edges(self, candidates: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Score a list of candidate edges using the Cross-Encoder.
        
        Args:
            candidates: List of edge dicts. Must contain 'source_template' and 'target_template' 
                        (or we will construct them from metadata if available).
            batch_size: Batch size for inference.
            
        Returns:
            The input list with a new 'validation_score' field added to each edge.
        """
        if not candidates:
            return []

        # Prepare pairs for the model: (Query, Document) -> (Source, Target)
        pairs = []
        valid_indices = []
        
        for i, edge in enumerate(candidates):
            # We need the actual text content to validate
            # Assuming the edge dict has 'source_template' and 'target_template'
            # If not, we might need to fetch it or pass it in.
            # For now, let's assume the caller enriches the edge dict with text.
            src_text = edge.get("source_template", "")
            tgt_text = edge.get("target_template", "")
            
            if src_text and tgt_text:
                pairs.append([src_text, tgt_text])
                valid_indices.append(i)
            else:
                # If text is missing, we can't validate. Set score to -1 or keep original.
                edge["validation_score"] = -1.0

        if not pairs:
            return candidates

        logger.info(f"Validating {len(pairs)} edges with Cross-Encoder...")
        
        # Predict scores (returns a list of floats, usually unbounded logits or 0-1 depending on model)
        # ms-marco models usually output logits. We can apply sigmoid if we want 0-1, 
        # but raw scores are fine for ranking.
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=True)

        # Update edges with scores
        for idx, score in zip(valid_indices, scores):
            # Convert numpy float to native float
            candidates[idx]["validation_score"] = float(score)
            
            # Optional: Update the 'weight' or 'hybrid_score' to reflect validation?
            # For now, let's keep it separate so we can analyze it.
            
        return candidates

    def filter_edges(self, candidates: List[Dict[str, Any]], threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Validate and then filter edges below a certain score.
        Note: MS MARCO models are trained on logits where > 0 usually means relevant.
        """
        validated = self.validate_edges(candidates)
        filtered = [e for e in validated if e.get("validation_score", -999) > threshold]
        logger.info(f"Filtered edges: {len(candidates)} -> {len(filtered)} (threshold={threshold})")
        return filtered
