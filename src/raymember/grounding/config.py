"""
Grounding configuration and mode definitions for Raymember.

Confidence Semantics:
    >= 0.80: GROUNDED / high confidence — deterministic answer
    0.55–0.79: UNCERTAIN / moderate confidence — answer with uncertainty caveat
    0.35–0.54: UNCERTAIN / weak evidence — weak evidence warning
    < 0.35: INSUFFICIENT_EVIDENCE — abstention

Support and certainty are treated as separate concepts internally:
    - Support: whether evidence exists for a claim
    - Certainty: how strong the evidence is
    A response with an uncertainty caveat is supported but not certain.
    It should not be counted as fully grounded solely because a belief exists.
"""

from dataclasses import dataclass
from enum import Enum


class GroundingMode(Enum):
    """Configurable grounding strictness mode.

    PERMISSIVE:
        Allows reasonable inference.
        Uses LLM generation more freely.
        Accepts moderate-confidence answers as grounded.

    BALANCED:
        Allows supported inference.
        Requires evidence alignment for factual claims.
        Uses LLM for complex queries but validates output.

    STRICT:
        Only permits claims directly supported by Raymember beliefs.
        Uses deterministic abstention for missing information.
        Uses deterministic answers for direct factual queries.
        Falls back to deterministic response when validation fails.
        Never guesses between similar entities.
    """
    PERMISSIVE = "permissive"
    BALANCED = "balanced"
    STRICT = "strict"


@dataclass
class GroundingConfig:
    """Configurable confidence thresholds and grounding behavior.

    Attributes:
        mode: Grounding strictness mode (PERMISSIVE, BALANCED, STRICT).
        high_confidence_threshold: Confidence >= this value is GROUNDED.
            Default 0.80. Answers are deterministic with no LLM call needed.
        moderate_confidence_threshold: Confidence >= this value (and < high)
            is UNCERTAIN with moderate confidence caveat. Default 0.55.
        low_confidence_threshold: Confidence >= this value (and < moderate)
            is UNCERTAIN with weak evidence warning. Default 0.35.
        insufficient_threshold: Confidence < low_confidence_threshold triggers
            INSUFFICIENT_EVIDENCE abstention.
        max_regeneration_attempts: Maximum LLM retries on validation failure.
            Default 1. Prevents infinite retry loops.
        enable_false_premise_detection: Whether to check for false premises
            in questions. Default True.
        enable_temporal_gap_detection: Whether to check for temporal gaps
            in observation coverage. Default True.
        enable_entity_isolation: Whether to enforce strict entity isolation
            (prevent cross-entity fact leakage). Default True.
    """
    mode: GroundingMode = GroundingMode.STRICT
    high_confidence_threshold: float = 0.80
    moderate_confidence_threshold: float = 0.55
    low_confidence_threshold: float = 0.35
    insufficient_threshold: float = 0.15
    max_regeneration_attempts: int = 1
    enable_false_premise_detection: bool = True
    enable_temporal_gap_detection: bool = True
    enable_entity_isolation: bool = True
