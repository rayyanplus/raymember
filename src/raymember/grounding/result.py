"""
Structured grounding result types for Raymember.

Defines the typed result representing the answerable world state,
including grounding status, confidence, evidence tracking, and
validation metadata.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class GroundingStatus(Enum):
    """Status of a grounded answer.

    GROUNDED: Answer directly supported by accepted belief with high confidence.
    INSUFFICIENT_EVIDENCE: Requested attribute/entity absent from memory.
    UNCERTAIN: Evidence exists but confidence is below high threshold or conflicting.
    CONTRADICTED_PREMISE: Question contains a false assumption contradicted by memory.
    TEMPORAL_GAP: No observation covers the requested time period.
    UNSUPPORTED: LLM answer failed validation against belief state.
    """
    GROUNDED = "grounded"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNCERTAIN = "uncertain"
    CONTRADICTED_PREMISE = "contradicted_premise"
    TEMPORAL_GAP = "temporal_gap"
    UNSUPPORTED = "unsupported"


@dataclass
class GroundedResult:
    """Typed, structured result representing the answerable world state.

    Captures the factual answer, its grounding status, confidence level,
    evidence provenance, and metadata about how the answer was produced
    (deterministic vs LLM, validation status, fallback usage).
    """
    answer: str
    status: GroundingStatus
    confidence: float
    entity: Optional[str] = None
    relation: Optional[str] = None
    value: Any = None
    evidence_ids: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    uncertainty: Optional[str] = None
    deterministic: bool = False
    validation_status: Optional[str] = None  # "passed" / "failed" / "fallback" / None
    validation_failures: List[str] = field(default_factory=list)
    fallback_used: bool = False
    llm_call_made: bool = False
    grounding_mode: str = "strict"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "answer": self.answer,
            "status": self.status.value,
            "confidence": self.confidence,
            "entity": self.entity,
            "relation": self.relation,
            "value": self.value,
            "evidence_ids": self.evidence_ids,
            "sources": self.sources,
            "uncertainty": self.uncertainty,
            "deterministic": self.deterministic,
            "validation_status": self.validation_status,
            "validation_failures": self.validation_failures,
            "fallback_used": self.fallback_used,
            "llm_call_made": self.llm_call_made,
            "grounding_mode": self.grounding_mode,
        }

    def to_benchmark_json(self) -> str:
        """Format as benchmark-compatible JSON response string.

        Produces the {"answer": ..., "confidence": ..., "reason": ...} format
        expected by DeterministicScorer.parse_json_response().
        """
        answer_val = ""
        if self.status == GroundingStatus.GROUNDED and self.value is not None:
            answer_val = str(self.value)
        elif self.status == GroundingStatus.INSUFFICIENT_EVIDENCE:
            answer_val = "unknown"
        elif self.status == GroundingStatus.UNCERTAIN:
            answer_val = "uncertain" if self.value is None else str(self.value)
        elif self.status == GroundingStatus.CONTRADICTED_PREMISE:
            answer_val = "false_premise"
        elif self.status == GroundingStatus.TEMPORAL_GAP:
            answer_val = "unknown"
        elif self.status == GroundingStatus.UNSUPPORTED:
            answer_val = "unknown"
        else:
            answer_val = str(self.value) if self.value else ""

        reason = self.uncertainty or self.answer
        return json.dumps({
            "answer": answer_val,
            "confidence": self.confidence,
            "reason": reason,
        })
