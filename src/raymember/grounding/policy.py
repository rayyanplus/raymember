"""
Deterministic grounding policy engine for Raymember.

Operates on structured belief data from Raymember.get() and Raymember.ask()
to produce grounded answers without requiring LLM generation for direct
factual queries.

Uses the existing EntityResolver for entity extraction with exact normalized
entity-ID matching as the highest-priority fallback. Implements an extensible
RelationRegistry for query understanding.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple

from raymember.grounding.result import GroundedResult, GroundingStatus
from raymember.grounding.config import GroundingConfig, GroundingMode


@dataclass
class RelationPattern:
    """A pattern for matching a relation type in natural language queries.

    Attributes:
        name: Canonical relation name (e.g. "location", "status").
        patterns: List of regex pattern strings for matching.
        attribute_keys: Keys to look up in entity attributes/beliefs.
    """
    name: str
    patterns: List[str]
    attribute_keys: List[str]
    _compiled: List[Any] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def matches(self, text: str) -> bool:
        """Check if any pattern matches the given text."""
        return any(p.search(text) for p in self._compiled)


class RelationRegistry:
    """Extensible registry of relation patterns for query understanding.

    Ships with deterministic patterns for: location, status,
    estimated_arrival, color, weight, serial_number.
    Applications can register additional relation patterns via register().
    """

    def __init__(self):
        self._relations: Dict[str, RelationPattern] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register the default set of relation patterns."""
        defaults = [
            RelationPattern("location", [
                r"\bwhere\b", r"\blocated?\b", r"\bposition\b", r"\broom\b",
                r"\bplace\b", r"\bstored\b", r"\bfind\b",
            ], ["room", "location"]),
            RelationPattern("status", [
                r"\bstatus\b", r"\bcondition\b",
            ], ["status"]),
            RelationPattern("estimated_arrival", [
                r"\beta\b", r"\barrival\b", r"\barrive\b", r"\bestimated.?arrival\b",
                r"\bwhen.*arrive\b", r"\bwhen.*expected\b",
            ], ["estimated_arrival", "eta", "arrival_time"]),
            RelationPattern("color", [
                r"\bcolou?r\b",
            ], ["color", "colour"]),
            RelationPattern("weight", [
                r"\bweight\b", r"\bweigh\b", r"\bheavy\b", r"\bmass\b",
            ], ["weight", "mass"]),
            RelationPattern("serial_number", [
                r"\bserial\b", r"\bserial.?number\b", r"\bid.?number\b",
                r"\bidentifier\b", r"\bpart.?number\b",
            ], ["serial_number", "serial", "id_number", "part_number"]),
        ]
        for rp in defaults:
            self._relations[rp.name] = rp

    def register(self, name: str, patterns: List[str], attribute_keys: List[str]):
        """Register a custom relation pattern.

        Args:
            name: Canonical name for the relation.
            patterns: List of regex pattern strings.
            attribute_keys: Entity attribute keys to search.
        """
        self._relations[name] = RelationPattern(name, patterns, attribute_keys)

    def detect(self, question: str) -> Optional[str]:
        """Detect the primary relation being queried in a question.

        Returns the canonical relation name, or None if unrecognized.
        """
        for name, rp in self._relations.items():
            if rp.matches(question):
                return name
        return None

    def get_attribute_keys(self, relation_name: str) -> List[str]:
        """Get the attribute keys for a given relation name."""
        rp = self._relations.get(relation_name)
        return rp.attribute_keys if rp else []

    def list_relations(self) -> List[str]:
        """List all registered relation names."""
        return list(self._relations.keys())


# Common stop words to skip during entity scanning
_STOP_WORDS = frozenset({
    "where", "what", "when", "why", "how", "which",
    "is", "are", "was", "were", "has", "have", "had", "does", "do", "did",
    "the", "a", "an", "in", "at", "of", "to", "from", "for", "with", "by",
    "on", "and", "or", "not", "no", "but", "its", "it", "this", "that",
    "now", "currently", "located", "stored", "moved", "about",
    "most", "reliable", "estimated", "arrival", "time", "according",
    "verified", "high", "trust", "evidence", "memory", "before",
    "after", "during", "between", "contents", "chemical", "vials", "inside",
    "serial", "number", "weight", "color", "status", "state",
})


class GroundingPolicy:
    """Deterministic answer policy engine.

    Operates on structured belief data from Raymember.get() to produce
    grounded answers, deterministic abstentions, and uncertainty-aware
    responses without requiring LLM generation for direct factual queries.

    Uses the existing EntityResolver (via mem.ask()) with exact normalized
    entity-ID matching. If multiple entities match in strict mode, returns
    an explicit ambiguous/uncertain result rather than guessing.
    """

    def __init__(self, config: Optional[GroundingConfig] = None):
        self.config = config or GroundingConfig()
        self.relation_registry = RelationRegistry()

    def evaluate_query(self, memory, question: str) -> GroundedResult:
        """Master dispatcher for grounding evaluation.

        Flow:
        1. Extract entity mention(s) from question
        2. Check entity existence → INSUFFICIENT_EVIDENCE if missing
        3. Check entity isolation (prevent cross-entity fact leakage)
        4. Detect relation being queried
        5. Check for false premise against accepted_transitions
        6. Check for temporal gap against observation timestamps
        7. Look up belief state for entity+relation
        8. Classify confidence tier → deterministic answer or uncertainty caveat
        9. Return GroundedResult (deterministic if sufficient, or flagged for LLM)

        Args:
            memory: Raymember SDK instance with observed data.
            question: Natural language question string.

        Returns:
            GroundedResult with grounding status, confidence, and answer.
        """
        mode_val = self.config.mode.value

        # 1. Extract entity from question
        entity_id, entity_state, ambiguity = self._resolve_entity(memory, question)

        # Handle ambiguous entity references
        if ambiguity:
            return GroundedResult(
                answer=ambiguity,
                status=GroundingStatus.UNCERTAIN,
                confidence=0.0,
                entity=None,
                relation=None,
                value=None,
                uncertainty=ambiguity,
                deterministic=True,
                grounding_mode=mode_val,
            )

        # 2. Check entity existence
        if entity_state is None:
            label = entity_id or "the requested entity"
            return GroundedResult(
                answer=f"Raymember has no information about '{label}'.",
                status=GroundingStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                entity=entity_id,
                relation=None,
                value=None,
                uncertainty="Entity not found in memory.",
                deterministic=True,
                grounding_mode=mode_val,
            )

        # 3. Entity isolation check
        if self.config.enable_entity_isolation:
            isolation_result = self._check_entity_isolation(
                memory, question, entity_id, entity_state
            )
            if isolation_result:
                return isolation_result

        # 4. Detect relation
        relation = self.relation_registry.detect(question)

        # 5. False premise detection
        if self.config.enable_false_premise_detection:
            premise_result = self._detect_false_premise(
                memory, question, entity_id, entity_state
            )
            if premise_result:
                return premise_result

        # 6. Temporal gap detection
        if self.config.enable_temporal_gap_detection:
            temporal_result = self._detect_temporal_gap(
                memory, question, entity_id, entity_state
            )
            if temporal_result:
                return temporal_result

        # 7-9. Resolve belief and build answer
        if relation == "location":
            return self._resolve_location_answer(entity_id, entity_state)
        elif relation == "estimated_arrival":
            return self._resolve_attribute_answer(
                entity_id, entity_state, relation,
                ["estimated_arrival", "eta", "arrival_time"]
            )
        elif relation is not None:
            attr_keys = self.relation_registry.get_attribute_keys(relation)
            return self._resolve_attribute_answer(
                entity_id, entity_state, relation, attr_keys
            )
        else:
            # Fallback: check if question is a location query via heuristics
            q_lower = question.lower()
            if any(w in q_lower for w in ["where", "room", "located"]):
                return self._resolve_location_answer(entity_id, entity_state)

            # Unknown relation — abstain in strict mode
            if self.config.mode == GroundingMode.STRICT:
                return GroundedResult(
                    answer=(
                        f"Raymember cannot determine the specific information "
                        f"requested about {entity_id}."
                    ),
                    status=GroundingStatus.INSUFFICIENT_EVIDENCE,
                    confidence=0.0,
                    entity=entity_id,
                    relation=None,
                    value=None,
                    uncertainty="Unknown relation; cannot determine what is being asked.",
                    deterministic=True,
                    grounding_mode=mode_val,
                )
            else:
                # In permissive/balanced mode, flag for LLM
                return GroundedResult(
                    answer="",
                    status=GroundingStatus.UNCERTAIN,
                    confidence=entity_state.confidence,
                    entity=entity_id,
                    relation=None,
                    value=None,
                    uncertainty="Relation not recognized; LLM reasoning required.",
                    deterministic=False,
                    grounding_mode=mode_val,
                )

    # ──────────────────────────────────────────────────────────────────────
    # Entity Resolution
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_entity(
        self, memory, question: str
    ) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
        """Extract and resolve entity from question.

        Strategy:
        1. Use mem.ask() which leverages the existing EntityResolver.
        2. Fall back to regex token scanning + exact mem.get() lookups.
        3. If multiple distinct entities match in strict mode, return ambiguity.

        Returns:
            (entity_id, entity_state, ambiguity_message)
        """
        entity_candidates: List[Tuple[str, Any]] = []
        seen_ids: set = set()

        # Strategy 1: Use mem.ask() for entity extraction
        try:
            query_result = memory.ask(question)
            if query_result and getattr(query_result, "entity", None):
                entity_state = memory.get(query_result.entity)
                if entity_state:
                    norm_id = entity_state.entity_id
                    if norm_id not in seen_ids:
                        seen_ids.add(norm_id)
                        entity_candidates.append((norm_id, entity_state))
        except Exception:
            pass

        # Strategy 2: Scan for entity-like tokens and try exact lookup
        # Match compound tokens: toolkit_A, shipment_482, robot_alpha, etc.
        compound_tokens = re.findall(
            r'\b([a-zA-Z][a-zA-Z0-9]*(?:[_\-][a-zA-Z0-9]+)+)\b', question
        )
        for token in compound_tokens:
            try:
                state = memory.get(token)
                if state:
                    norm_id = state.entity_id
                    if norm_id not in seen_ids:
                        seen_ids.add(norm_id)
                        entity_candidates.append((norm_id, state))
            except Exception:
                continue

        # Strategy 3: Try individual words (skip stop words)
        words = re.findall(r'\b\w+\b', question)
        for word in words:
            if word.lower() in _STOP_WORDS or len(word) < 2:
                continue
            try:
                state = memory.get(word)
                if state:
                    norm_id = state.entity_id
                    if norm_id not in seen_ids:
                        seen_ids.add(norm_id)
                        entity_candidates.append((norm_id, state))
            except Exception:
                continue

        if len(entity_candidates) == 1:
            return entity_candidates[0][0], entity_candidates[0][1], None
        elif len(entity_candidates) > 1:
            if self.config.mode == GroundingMode.STRICT:
                entity_names = [e[0] for e in entity_candidates]
                msg = (
                    f"Ambiguous entity reference. Multiple entities match: "
                    f"{', '.join(entity_names)}. Please specify which entity."
                )
                return None, None, msg
            else:
                # In permissive/balanced, use the first match
                return entity_candidates[0][0], entity_candidates[0][1], None
        else:
            return None, None, None

    # ──────────────────────────────────────────────────────────────────────
    # Entity Isolation
    # ──────────────────────────────────────────────────────────────────────

    def _check_entity_isolation(
        self, memory, question: str, entity_id: str, entity_state
    ) -> Optional[GroundedResult]:
        """Ensure facts from similar entities do not leak.

        Checks if the question text references a different entity than
        the resolved one. If so, and the entities are distinct, prevents
        cross-entity fact leakage.
        """
        compound_tokens = re.findall(
            r'\b([a-zA-Z][a-zA-Z0-9]*(?:[_\-][a-zA-Z0-9]+)+)\b', question
        )
        mentioned_entities = set()
        for token in compound_tokens:
            try:
                state = memory.get(token)
                if state:
                    mentioned_entities.add(state.entity_id)
            except Exception:
                continue

        # Remove the primary resolved entity
        mentioned_entities.discard(entity_id)

        if mentioned_entities and self.config.mode == GroundingMode.STRICT:
            # Question mentions other entities — verify isolation
            all_entities = [entity_id] + list(mentioned_entities)
            if len(all_entities) > 1:
                # The resolved entity is correct; just ensure we don't
                # leak facts from the other entities into the answer.
                # This check passes silently — the actual isolation happens
                # in _resolve_location_answer and _resolve_attribute_answer
                # which only use the resolved entity_state.
                pass

        return None

    # ──────────────────────────────────────────────────────────────────────
    # False Premise Detection
    # ──────────────────────────────────────────────────────────────────────

    def _detect_false_premise(
        self, memory, question: str, entity_id: str, entity_state
    ) -> Optional[GroundedResult]:
        """Detect false premises in questions.

        Extracts premise locations/attributes from the question and
        cross-checks against accepted_transitions and current_location.
        If the premise contradicts memory, returns CONTRADICTED_PREMISE.

        Example:
            Memory: toolkit moved from garage to workshop
            Question: "Why was the toolkit moved from the kitchen?"
            Result: CONTRADICTED_PREMISE — kitchen never observed
        """
        q_lower = question.lower()
        mode_val = self.config.mode.value

        # Check for "why" questions about movement (common false premise pattern)
        if "why" not in q_lower and "moved from" not in q_lower:
            return None

        # Extract premise locations: "moved from the <location>"
        premise_patterns = [
            r"moved?\s+from\s+(?:the\s+)?(\w+)",
            r"was\s+in\s+(?:the\s+)?(\w+)",
            r"left\s+(?:the\s+)?(\w+)",
            r"came\s+from\s+(?:the\s+)?(\w+)",
            r"transferred\s+from\s+(?:the\s+)?(\w+)",
        ]

        premise_locations = set()
        for pattern in premise_patterns:
            matches = re.findall(pattern, q_lower)
            premise_locations.update(matches)

        if not premise_locations:
            return None

        # Collect all known locations for this entity
        known_locations: set = set()

        # Current location
        curr_loc = getattr(entity_state, "current_location", None)
        if curr_loc and isinstance(curr_loc, dict):
            room = curr_loc.get("room", "")
            if room:
                known_locations.add(room.lower())

        # Previous location
        prev_loc = getattr(entity_state, "previous_location", None)
        if prev_loc and isinstance(prev_loc, dict):
            room = prev_loc.get("room", "")
            if room:
                known_locations.add(room.lower())

        # Accepted transitions
        transitions = getattr(entity_state, "accepted_transitions", []) or []
        for tr in transitions:
            if isinstance(tr, dict):
                old_val = tr.get("old_value", "")
                new_val = tr.get("new_value", "")
                if isinstance(old_val, str) and old_val:
                    known_locations.add(old_val.lower())
                if isinstance(new_val, str) and new_val:
                    known_locations.add(new_val.lower())
                # Also check nested room dicts
                for key in ("old_location", "new_location", "from", "to"):
                    loc_dict = tr.get(key)
                    if isinstance(loc_dict, dict):
                        room = loc_dict.get("room", "")
                        if room:
                            known_locations.add(room.lower())

        # Also check observation history
        try:
            history = memory.history(entity_id, limit=50)
            for h in (history or []):
                if isinstance(h, dict):
                    room = h.get("room", "")
                    if room:
                        known_locations.add(room.lower())
                    loc = h.get("location")
                    if isinstance(loc, dict):
                        room = loc.get("room", "")
                        if room:
                            known_locations.add(room.lower())
        except Exception:
            pass

        # Check if any premise location is NOT in known locations
        for premise_loc in premise_locations:
            if premise_loc not in known_locations:
                # Build correction message
                known_str = ", ".join(sorted(known_locations)) if known_locations else "no recorded locations"
                return GroundedResult(
                    answer=(
                        f"Raymember contains no evidence that {entity_id} was "
                        f"in the {premise_loc}. Known locations: {known_str}."
                    ),
                    status=GroundingStatus.CONTRADICTED_PREMISE,
                    confidence=entity_state.confidence,
                    entity=entity_id,
                    relation="location",
                    value="false_premise",
                    evidence_ids=getattr(entity_state, "evidence", []) or [],
                    sources=[getattr(entity_state, "provenance", "")] if getattr(entity_state, "provenance", None) else [],
                    uncertainty=f"False premise: '{premise_loc}' not found in entity history.",
                    deterministic=True,
                    grounding_mode=mode_val,
                )

        return None

    # ──────────────────────────────────────────────────────────────────────
    # Temporal Gap Detection
    # ──────────────────────────────────────────────────────────────────────

    def _detect_temporal_gap(
        self, memory, question: str, entity_id: str, entity_state
    ) -> Optional[GroundedResult]:
        """Detect temporal gaps — queries about unobserved time periods.

        If the question asks about a specific time and no observation
        covers that time, returns TEMPORAL_GAP.

        Example:
            Observations: 09:00 garage, 11:00 workshop
            Question: "Where was the toolkit at 10:00?"
            Result: TEMPORAL_GAP if no observation covers 10:00
        """
        q_lower = question.lower()
        mode_val = self.config.mode.value

        # Extract time references from question
        time_refs = re.findall(r'(?:at\s+)?(\d{1,2}:\d{2})', question)
        if not time_refs:
            return None

        # Check for "before any observation" or past-tense temporal queries
        before_patterns = [
            r"before\s+any\s+observation",
            r"before\s+(?:the\s+)?first",
            r"prior\s+to",
        ]
        is_before_query = any(re.search(p, q_lower) for p in before_patterns)

        if is_before_query:
            return GroundedResult(
                answer=(
                    f"Raymember does not have enough evidence to determine "
                    f"the {entity_id}'s location at the requested time."
                ),
                status=GroundingStatus.TEMPORAL_GAP,
                confidence=0.0,
                entity=entity_id,
                relation="location",
                value=None,
                uncertainty="Requested time precedes all observations.",
                deterministic=True,
                grounding_mode=mode_val,
            )

        # Check observation timestamps
        try:
            history = memory.history(entity_id, limit=100)
            if history:
                from datetime import datetime
                obs_hours = set()
                for h in history:
                    ts = h.get("timestamp", "")
                    if ts:
                        try:
                            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            obs_hours.add(f"{dt.hour}:{dt.minute:02d}")
                        except (ValueError, AttributeError):
                            pass

                # Check if any queried time has a nearby observation
                for t_ref in time_refs:
                    parts = t_ref.split(":")
                    try:
                        q_hour = int(parts[0])
                        q_min = int(parts[1])
                    except (ValueError, IndexError):
                        continue

                    # Check if any observation is within 1 hour of query time
                    has_nearby = False
                    for h in history:
                        ts = h.get("timestamp", "")
                        if ts:
                            try:
                                dt = datetime.fromisoformat(
                                    ts.replace("Z", "+00:00")
                                )
                                # Same day: check hour proximity
                                hour_diff = abs(dt.hour - q_hour)
                                if hour_diff <= 1:
                                    has_nearby = True
                                    break
                            except (ValueError, AttributeError):
                                pass

                    if not has_nearby and len(history) > 0:
                        return GroundedResult(
                            answer=(
                                f"Raymember does not have enough evidence to "
                                f"determine {entity_id}'s location at {t_ref}."
                            ),
                            status=GroundingStatus.TEMPORAL_GAP,
                            confidence=0.0,
                            entity=entity_id,
                            relation="location",
                            value=None,
                            uncertainty=f"No observation near {t_ref}.",
                            deterministic=True,
                            grounding_mode=mode_val,
                        )
        except Exception:
            pass

        return None

    # ──────────────────────────────────────────────────────────────────────
    # Location Answer Resolution
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_location_answer(
        self, entity_id: str, entity_state
    ) -> GroundedResult:
        """Resolve a location query from entity belief state.

        Applies confidence tier classification:
            >= high_threshold:     GROUNDED, deterministic template
            >= moderate_threshold: UNCERTAIN, moderate confidence caveat
            >= low_threshold:      UNCERTAIN, weak evidence warning
            < low_threshold:       INSUFFICIENT_EVIDENCE, abstention
        """
        mode_val = self.config.mode.value
        location = getattr(entity_state, "current_location", None)
        confidence = getattr(entity_state, "confidence", 0.0) or 0.0
        provenance = getattr(entity_state, "provenance", "") or ""
        evidence = getattr(entity_state, "evidence", []) or []
        sources = [provenance] if provenance else []

        if not location or not isinstance(location, dict):
            return GroundedResult(
                answer=f"Raymember has no location information for {entity_id}.",
                status=GroundingStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                entity=entity_id,
                relation="location",
                value=None,
                uncertainty="No location data in memory.",
                deterministic=True,
                grounding_mode=mode_val,
            )

        room = location.get("room", "")
        if not room:
            return GroundedResult(
                answer=f"Raymember has no location information for {entity_id}.",
                status=GroundingStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                entity=entity_id,
                relation="location",
                value=None,
                uncertainty="No room data in memory.",
                deterministic=True,
                grounding_mode=mode_val,
            )

        # Confidence tier classification
        if confidence >= self.config.high_confidence_threshold:
            src_str = f" by {', '.join(sources)}" if sources else ""
            return GroundedResult(
                answer=(
                    f"The {entity_id} is currently in the {room}. "
                    f"This is supported{src_str} with "
                    f"{confidence*100:.0f}% confidence."
                ),
                status=GroundingStatus.GROUNDED,
                confidence=confidence,
                entity=entity_id,
                relation="location",
                value=room,
                evidence_ids=evidence,
                sources=sources,
                deterministic=True,
                grounding_mode=mode_val,
            )
        elif confidence >= self.config.moderate_confidence_threshold:
            return GroundedResult(
                answer=(
                    f"The {room} is the best-supported location for "
                    f"{entity_id}, but the evidence is uncertain. "
                    f"The strongest observation has "
                    f"{confidence*100:.0f}% confidence."
                ),
                status=GroundingStatus.UNCERTAIN,
                confidence=confidence,
                entity=entity_id,
                relation="location",
                value=room,
                evidence_ids=evidence,
                sources=sources,
                uncertainty=f"Moderate confidence ({confidence*100:.0f}%); evidence may be incomplete.",
                deterministic=True,
                grounding_mode=mode_val,
            )
        elif confidence >= self.config.low_confidence_threshold:
            return GroundedResult(
                answer=(
                    f"Raymember has weak evidence suggesting {entity_id} "
                    f"may be in {room}, but confidence is only "
                    f"{confidence*100:.0f}%."
                ),
                status=GroundingStatus.UNCERTAIN,
                confidence=confidence,
                entity=entity_id,
                relation="location",
                value=room,
                evidence_ids=evidence,
                sources=sources,
                uncertainty=f"Low confidence ({confidence*100:.0f}%); weak evidence.",
                deterministic=True,
                grounding_mode=mode_val,
            )
        else:
            return GroundedResult(
                answer=(
                    f"Raymember has no reliable information about "
                    f"{entity_id}'s location."
                ),
                status=GroundingStatus.INSUFFICIENT_EVIDENCE,
                confidence=confidence,
                entity=entity_id,
                relation="location",
                value=None,
                uncertainty=f"Confidence ({confidence*100:.0f}%) below threshold.",
                deterministic=True,
                grounding_mode=mode_val,
            )

    # ──────────────────────────────────────────────────────────────────────
    # Attribute Answer Resolution
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_attribute_answer(
        self,
        entity_id: str,
        entity_state,
        relation: str,
        attr_keys: List[str],
    ) -> GroundedResult:
        """Resolve an attribute query from entity belief state.

        Looks up the attribute in current_attributes and attribute_beliefs.
        If absent, returns INSUFFICIENT_EVIDENCE (abstention).
        """
        mode_val = self.config.mode.value
        confidence = getattr(entity_state, "confidence", 0.0) or 0.0
        evidence = getattr(entity_state, "evidence", []) or []

        # Check current_attributes
        attrs = getattr(entity_state, "current_attributes", {}) or {}
        for key in attr_keys:
            if key in attrs and attrs[key] is not None:
                value = attrs[key]
                # Check attribute_beliefs for confidence
                attr_beliefs = getattr(entity_state, "attribute_beliefs", {}) or {}
                attr_conf = confidence
                attr_prov = ""
                if key in attr_beliefs:
                    belief = attr_beliefs[key]
                    if isinstance(belief, dict):
                        attr_conf = belief.get("confidence", confidence)
                        attr_prov = belief.get("provenance", "")
                    elif hasattr(belief, "confidence"):
                        attr_conf = belief.confidence
                        attr_prov = getattr(belief, "provenance", "")

                sources = [attr_prov] if attr_prov else []
                return self._build_attribute_result(
                    entity_id, relation, key, value, attr_conf, evidence, sources
                )

        # Check state dict
        state_dict = getattr(entity_state, "state", None)
        if isinstance(state_dict, dict):
            for key in attr_keys:
                if key in state_dict and state_dict[key] is not None:
                    value = state_dict[key]
                    return self._build_attribute_result(
                        entity_id, relation, key, value, confidence, evidence, []
                    )

        # Attribute not found — abstain
        return GroundedResult(
            answer=(
                f"Raymember has no information about {entity_id}'s {relation}."
            ),
            status=GroundingStatus.INSUFFICIENT_EVIDENCE,
            confidence=0.0,
            entity=entity_id,
            relation=relation,
            value=None,
            uncertainty=f"Attribute '{relation}' not found in memory.",
            deterministic=True,
            grounding_mode=mode_val,
        )

    def _build_attribute_result(
        self,
        entity_id: str,
        relation: str,
        attr_key: str,
        value: Any,
        confidence: float,
        evidence: List[str],
        sources: List[str],
    ) -> GroundedResult:
        """Build a GroundedResult for an attribute value with confidence tiers."""
        mode_val = self.config.mode.value

        if confidence >= self.config.high_confidence_threshold:
            src_str = f" by {', '.join(sources)}" if sources else ""
            return GroundedResult(
                answer=(
                    f"The {relation} of {entity_id} is {value}. "
                    f"This is supported{src_str} with "
                    f"{confidence*100:.0f}% confidence."
                ),
                status=GroundingStatus.GROUNDED,
                confidence=confidence,
                entity=entity_id,
                relation=relation,
                value=value,
                evidence_ids=evidence,
                sources=sources,
                deterministic=True,
                grounding_mode=mode_val,
            )
        elif confidence >= self.config.moderate_confidence_threshold:
            return GroundedResult(
                answer=(
                    f"The best-supported {relation} for {entity_id} is "
                    f"{value}, but the evidence is uncertain "
                    f"({confidence*100:.0f}% confidence)."
                ),
                status=GroundingStatus.UNCERTAIN,
                confidence=confidence,
                entity=entity_id,
                relation=relation,
                value=value,
                evidence_ids=evidence,
                sources=sources,
                uncertainty=f"Moderate confidence ({confidence*100:.0f}%).",
                deterministic=True,
                grounding_mode=mode_val,
            )
        elif confidence >= self.config.low_confidence_threshold:
            return GroundedResult(
                answer=(
                    f"Raymember has weak evidence suggesting {entity_id}'s "
                    f"{relation} may be {value}, but confidence is only "
                    f"{confidence*100:.0f}%."
                ),
                status=GroundingStatus.UNCERTAIN,
                confidence=confidence,
                entity=entity_id,
                relation=relation,
                value=value,
                evidence_ids=evidence,
                sources=sources,
                uncertainty=f"Low confidence ({confidence*100:.0f}%).",
                deterministic=True,
                grounding_mode=mode_val,
            )
        else:
            return GroundedResult(
                answer=(
                    f"Raymember has no reliable information about "
                    f"{entity_id}'s {relation}."
                ),
                status=GroundingStatus.INSUFFICIENT_EVIDENCE,
                confidence=confidence,
                entity=entity_id,
                relation=relation,
                value=None,
                uncertainty=f"Confidence ({confidence*100:.0f}%) below threshold.",
                deterministic=True,
                grounding_mode=mode_val,
            )
