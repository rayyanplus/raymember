"""
Raymember Grounding Module.

Provides structured answer-control, deterministic abstention,
uncertainty-aware responses, false-premise detection, temporal-gap
handling, entity isolation, and post-generation validation.

Public API:
    GroundedResult       — Structured grounding result type
    GroundingStatus      — Answer grounding status enum
    GroundingPolicy      — Deterministic answer policy engine
    RelationRegistry     — Extensible relation pattern registry
    GroundingValidator   — Post-generation answer validator
    ValidationResult     — Validation check result
    GroundingConfig      — Configuration and thresholds
    GroundingMode        — Grounding strictness mode enum
"""

from raymember.grounding.result import GroundedResult, GroundingStatus
from raymember.grounding.config import GroundingConfig, GroundingMode
from raymember.grounding.policy import GroundingPolicy, RelationRegistry
from raymember.grounding.validator import GroundingValidator, ValidationResult

__all__ = [
    "GroundedResult",
    "GroundingStatus",
    "GroundingPolicy",
    "RelationRegistry",
    "GroundingValidator",
    "ValidationResult",
    "GroundingConfig",
    "GroundingMode",
]
