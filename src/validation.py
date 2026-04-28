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
import json
import re
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

try:
    import ollama
except ImportError:
    ollama = None

from config.settings import Settings

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
            # Extract text content for validation
            # Prefer 'semantic_text' fields which are populated by the pipeline
            src_text = edge.get("source_semantic_text")
            tgt_text = edge.get("target_semantic_text")
            
            # Fallback to template/metadata if semantic_text is missing
            if not src_text:
                src_text = edge.get("source_template", "")
            if not tgt_text:
                tgt_text = edge.get("target_template", "")
                if not tgt_text and edge.get("target_metadata"):
                    # Try to extract from metadata dict
                    tgt_text = edge["target_metadata"].get("semantic_text", "")
            
            if src_text and tgt_text:
                pairs.append([str(src_text), str(tgt_text)])
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
            val_score = float(score)
            candidates[idx]["validation_score"] = val_score
            
            # Update hybrid_score to reflect the validation result
            # We blend the original score with the validation score (sigmoid-normalized if needed)
            # For MS MARCO, scores are logits. >0 is relevant.
            # Simple heuristic: if val_score > 0, boost. If < 0, penalize.
            
            # Sigmoid to get 0-1 range
            prob = 1 / (1 + np.exp(-val_score))
            
            # New hybrid score: 50% original, 50% validation probability
            original_score = candidates[idx].get("hybrid_score", 0.5)
            candidates[idx]["hybrid_score"] = (original_score + prob) / 2.0
            
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


class LLMValidator:
    """
    Validates microservice dependency edges using a Local LLM (via Ollama).
    This acts as a "Judge" to confirm if a semantic match implies a causal or functional relationship.
    """
    def __init__(self, model_name: str = None):
        if ollama is None:
            raise ImportError("ollama library is required for LLMValidator. pip install ollama")
            
        self.model_name = model_name or getattr(Settings, "OLLAMA_MODEL", "llama3.1")
        logger.info(f"Initialized LLMValidator with model: {self.model_name}")

    def _clean_json_response(self, content: str) -> str:
        """Clean markdown code blocks from response."""
        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return content.strip()

    def validate_edge(self, source_log: str, target_log: str) -> Dict[str, Any]:
        """
        Ask the LLM if Log A -> Log B is a plausible dependency.
        """
        prompt = f"""
        You are a distributed systems expert. Analyze if a causal link is possible between these two microservice logs.
        
        Log A (Source): "{source_log}"
        Log B (Target): "{target_log}"
        
        Task:
        1. Does Log A describe an action that logically leads to Log B?
        2. Are they functionally related (e.g., API request -> Compute task)?
        
        Respond ONLY with a valid JSON object:
        {{
            "is_causal": boolean,
            "confidence_score": float (0.0 to 1.0),
            "reasoning": "short explanation"
        }}
        """
        
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': prompt},
            ])
            
            content = response['message']['content']
            json_str = self._clean_json_response(content)
            
            result = json.loads(json_str)
            # Normalize keys just in case
            return {
                "is_causal": result.get("is_causal", False),
                "confidence": float(result.get("confidence_score", result.get("confidence", 0.0))),
                "reasoning": result.get("reasoning", "No reasoning provided")
            }
            
        except Exception as e:
            logger.error(f"LLM Validation failed for pair: {str(e)}")
            return {"is_causal": False, "confidence": 0.0, "reasoning": f"Error: {str(e)}"}

    def validate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate a list of candidate edges.
        Modifies the list in-place by adding 'llm_verification' metadata.
        """
        logger.info(f"Starting LLM verification for {len(candidates)} edges using {self.model_name}...")
        
        validated_count = 0
        for i, edge in enumerate(candidates):
            # Log progress every 10 edges
            if i > 0 and i % 10 == 0:
                logger.info(f"LLM Progress: {i}/{len(candidates)} edges verified.")
                
            src_text = edge.get("source_semantic_text") or edge.get("source_template", "")
            tgt_text = edge.get("target_semantic_text") or edge.get("target_template", "")
            
            # Retrieve target text from metadata if not top-level
            if not tgt_text and edge.get("target_metadata"):
                 tgt_text = edge["target_metadata"].get("semantic_text", "")

            if not src_text or not tgt_text:
                edge["llm_verification"] = {"skipped": True, "reason": "Missing text"}
                continue

            result = self.validate_edge(str(src_text), str(tgt_text))
            
            edge["llm_verification"] = result
            edge["llm_confidence"] = result["confidence"]
            
            # Optional: Update hybrid score or simply flag based on boolean
            # If LLM says "Is Causal: True" and Confidence > 0.8, we boost significantly.
            if result["is_causal"] and result["confidence"] > 0.7:
                edge["hybrid_score"] = min(1.0, edge.get("hybrid_score", 0.5) + 0.3)
                validated_count += 1
            elif not result["is_causal"]:
                # Penalize, but don't delete (let the specific filter handle deletion if needed)
                edge["hybrid_score"] = max(0.0, edge.get("hybrid_score", 0.5) - 0.3)

        logger.info(f"LLM Verification Complete. confirmed_causal={validated_count}/{len(candidates)}")
        return candidates
