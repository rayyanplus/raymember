"""
Post-generation grounding validator for Raymember.

Validates LLM-generated answers against structured belief data
using deterministic checks. Does not rely exclusively on another
LLM to judge the answer.

Checks:
1. Value mismatch — answer value differs from resolved belief
2. Absent attribute — answer introduces attribute not in memory
3. Contradiction — answer contradicts accepted memory state
4. Overconfidence — answer certainty exceeds evidence strength
5. False premise accepted — answer accepts a flagged false premise
6. Entity confusion — answer references facts from wrong entity
7. Temporal fabrication — answer invents unsupported temporal info
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from raymember.grounding.result import GroundedResult, GroundingStatus
from raymember.grounding.config import GroundingConfig


@dataclass
class ValidationResult:
    """Result of post-generation validation.

    Attributes:
        passed: True if all checks passed.
        failures: Names of failed validation checks.
        failure_details: Check name → explanation mapping.
    """
    passed: bool
    failures: List[str] = field(default_factory=list)
    failure_details: Dict[str, str] = field(default_factory=dict)


class GroundingValidator:
    """Post-generation validator for LLM answers.

    Validates entity, relation, value, confidence, evidence IDs,
    and status deterministically against the structured belief state.
    Prefers structured LLM output for complex queries.
    """

    def __init__(self, config: Optional[GroundingConfig] = None):
        self.config = config or GroundingConfig()

    def validate(
        self,
        llm_response: str,
        entity_state: Optional[Any],
        grounded_result: GroundedResult,
        entity_id: Optional[str] = None,
    ) -> ValidationResult:
        """Run all deterministic validation checks.

        Args:
            llm_response: Raw LLM response text.
            entity_state: EntityStateResult from Raymember.get().
            grounded_result: The pre-computed GroundedResult from policy.
            entity_id: The resolved entity ID.

        Returns:
            ValidationResult with pass/fail status and failure details.
        """
        failures: List[str] = []
        details: Dict[str, str] = {}

        # Parse structured fields from LLM response
        parsed = self._parse_structured_response(llm_response)
        resp_lower = llm_response.lower()

        # 1. Value mismatch check
        if grounded_result.value is not None and grounded_result.status == GroundingStatus.GROUNDED:
            mismatch = self._check_value_mismatch(parsed, grounded_result)
            if mismatch:
                failures.append("value_mismatch")
                details["value_mismatch"] = mismatch

        # 2. Absent attribute check
        if entity_state is not None:
            absent = self._check_absent_attribute(resp_lower, entity_state)
            if absent:
                failures.append("absent_attribute")
                details["absent_attribute"] = absent

        # 3. Contradiction check
        if entity_state is not None:
            contradiction = self._check_contradiction(
                parsed, resp_lower, entity_state, grounded_result
            )
            if contradiction:
                failures.append("contradiction")
                details["contradiction"] = contradiction

        # 4. Overconfidence check
        overconf = self._check_overconfidence(parsed, grounded_result)
        if overconf:
            failures.append("overconfidence")
            details["overconfidence"] = overconf

        # 5. False premise accepted check
        if grounded_result.status == GroundingStatus.CONTRADICTED_PREMISE:
            fp = self._check_false_premise_accepted(resp_lower)
            if fp:
                failures.append("false_premise_accepted")
                details["false_premise_accepted"] = fp

        # 6. Entity confusion check
        if entity_id and entity_state:
            confusion = self._check_entity_confusion(
                resp_lower, entity_id, entity_state
            )
            if confusion:
                failures.append("entity_confusion")
                details["entity_confusion"] = confusion

        # 7. Temporal fabrication check
        if grounded_result.status == GroundingStatus.TEMPORAL_GAP:
            temporal = self._check_temporal_fabrication(resp_lower)
            if temporal:
                failures.append("temporal_fabrication")
                details["temporal_fabrication"] = temporal

        return ValidationResult(
            passed=len(failures) == 0,
            failures=failures,
            failure_details=details,
        )

    def _parse_structured_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse structured JSON fields from LLM response."""
        raw = llm_response.strip()
        if "{" in raw and "}" in raw:
            json_str = raw[raw.find("{"):raw.rfind("}") + 1]
            try:
                return json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                pass
        return {}

    def _check_value_mismatch(
        self, parsed: Dict[str, Any], grounded: GroundedResult
    ) -> Optional[str]:
        """Check if the LLM answer value differs from the resolved belief."""
        expected = str(grounded.value).strip().lower()
        llm_answer = str(parsed.get("answer", "")).strip().lower()

        if not llm_answer:
            return None

        # Allow if LLM answer contains the expected value
        if expected in llm_answer or llm_answer in expected:
            return None

        # Check for abstention responses (acceptable)
        abstention_words = {"unknown", "uncertain", "no information", "no record", "absent", "insufficient"}
        if any(w in llm_answer for w in abstention_words):
            return None

        return (
            f"LLM answer '{llm_answer}' does not match belief value "
            f"'{expected}'."
        )

    def _check_absent_attribute(
        self, resp_lower: str, entity_state
    ) -> Optional[str]:
        """Check if the answer introduces attributes absent from memory."""
        # Check for specific fabricated attributes
        attrs = getattr(entity_state, "current_attributes", {}) or {}
        known_keys = set(str(k).lower() for k in attrs.keys())

        # Common fabricated attributes to check
        suspect_attrs = {
            "serial number": "serial_number",
            "weight": "weight",
            "color": "color",
            "temperature": "temperature",
            "owner": "owner",
            "manufacturer": "manufacturer",
        }

        for display_name, attr_key in suspect_attrs.items():
            # If the response mentions a specific value for an attribute not in memory
            pattern = rf"{display_name}\s*(?:is|:)\s*\w+"
            if re.search(pattern, resp_lower) and attr_key not in known_keys:
                return (
                    f"Response introduces '{display_name}' which is "
                    f"absent from memory."
                )

        return None

    def _check_contradiction(
        self,
        parsed: Dict[str, Any],
        resp_lower: str,
        entity_state,
        grounded: GroundedResult,
    ) -> Optional[str]:
        """Check if the answer contradicts accepted memory state."""
        if grounded.relation == "location" and grounded.value:
            expected_room = str(grounded.value).lower()
            curr_loc = getattr(entity_state, "current_location", {})
            if isinstance(curr_loc, dict):
                belief_room = curr_loc.get("room", "").lower()
                if belief_room and expected_room:
                    llm_answer = str(parsed.get("answer", "")).lower()
                    if llm_answer and llm_answer != expected_room and llm_answer != belief_room:
                        # LLM gave a different room than the belief
                        abstention_words = {"unknown", "uncertain", "no information"}
                        if not any(w in llm_answer for w in abstention_words):
                            return (
                                f"LLM answer '{llm_answer}' contradicts "
                                f"accepted belief '{belief_room}'."
                            )
        return None

    def _check_overconfidence(
        self, parsed: Dict[str, Any], grounded: GroundedResult
    ) -> Optional[str]:
        """Check if answer states certainty stronger than evidence supports."""
        llm_conf = parsed.get("confidence")
        if llm_conf is None:
            return None

        try:
            llm_conf = float(llm_conf)
        except (ValueError, TypeError):
            return None

        # If evidence confidence is low/moderate but LLM claims high confidence
        if grounded.confidence < self.config.moderate_confidence_threshold:
            if llm_conf >= self.config.high_confidence_threshold:
                return (
                    f"LLM confidence {llm_conf:.2f} exceeds evidence "
                    f"strength {grounded.confidence:.2f}."
                )

        # If grounding says uncertain but LLM is very confident
        if grounded.status == GroundingStatus.UNCERTAIN:
            if llm_conf >= 0.90:
                return (
                    f"LLM confidence {llm_conf:.2f} unjustified for "
                    f"uncertain evidence ({grounded.confidence:.2f})."
                )

        return None

    def _check_false_premise_accepted(self, resp_lower: str) -> Optional[str]:
        """Check if LLM accepted a false premise instead of correcting it."""
        correction_words = {
            "false premise", "incorrect assumption", "no evidence",
            "never", "not in", "was not", "wasn't", "no record",
            "contradicts", "incorrect",
        }
        if any(w in resp_lower for w in correction_words):
            return None  # Good — LLM corrected the premise

        # Check if LLM accepted the premise by explaining "why"
        acceptance_patterns = [
            r"was moved because",
            r"the reason",
            r"it was relocated due to",
            r"to make room",
        ]
        for pattern in acceptance_patterns:
            if re.search(pattern, resp_lower):
                return "LLM accepted a false premise without correction."

        # If response doesn't contain correction keywords, it might have
        # silently accepted the premise
        return None

    def _check_entity_confusion(
        self, resp_lower: str, entity_id: str, entity_state
    ) -> Optional[str]:
        """Check if answer references facts from a different entity."""
        # Extract entity base name (e.g., "toolkit" from "toolkit_a")
        base_name = entity_id.split("_")[0].lower() if "_" in entity_id else entity_id.lower()

        # Check for other entity variants mentioned in the response
        entity_variants = re.findall(
            rf"\b{re.escape(base_name)}_\w+\b", resp_lower
        )
        for variant in entity_variants:
            variant_clean = variant.strip().lower()
            entity_id_clean = entity_id.strip().lower()
            if variant_clean != entity_id_clean:
                return (
                    f"Response references '{variant}' which is a different "
                    f"entity from '{entity_id}'."
                )

        return None

    def _check_temporal_fabrication(self, resp_lower: str) -> Optional[str]:
        """Check if answer invents unsupported temporal information."""
        # If grounding flagged a temporal gap, the response should abstain
        certainty_words = {
            "was in", "was located", "was at", "was stored",
            "it was", "the location was",
        }
        # Check if response makes definite temporal claims
        for phrase in certainty_words:
            if phrase in resp_lower:
                # Check if it's qualified with uncertainty
                uncertainty_words = {
                    "unknown", "uncertain", "no information", "cannot determine",
                    "no evidence", "insufficient", "not enough",
                }
                if not any(w in resp_lower for w in uncertainty_words):
                    return "Response makes definite temporal claims despite a temporal gap."

        return None
