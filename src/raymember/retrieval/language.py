"""
Deterministic natural-language helpers for Raymember query answer generation.

No LLMs used. All logic is rule-based and model-agnostic.

Public API
----------
location_phrase(room)            -> "in the bedroom" / "on the desk"
entity_subject(entity, tense)    -> "The car keys were" / "The backpack was"
ObservationKind                  -> classification enum
classify_observation(...)        -> ObservationKind
ConflictAnswerGenerator.build()  -> final NL answer string
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# 1. Spatial preposition formatter
# ---------------------------------------------------------------------------

# Surfaces/objects where the correct preposition is "on", not "in"
_ON_SURFACES: Set[str] = {
    "desk",
    "table",
    "shelf",
    "shelves",
    "counter",
    "countertop",
    "floor",
    "bed",
    "couch",
    "sofa",
    "chair",
    "armchair",
    "workbench",
    "tray",
    "rack",
    "mat",
    "rug",
    "staircase",
    "stairs",
    "windowsill",
    "balcony",
    "porch",
    "rooftop",
    "roof",
    "cart",
    "trolley",
    "worktop",
    "bench",
    "cabinet",      # "on the cabinet top" is informal but accepted
    "nightstand",
}


def location_phrase(room: str) -> str:
    """
    Return a grammatically correct spatial phrase for a room/surface.

    Examples
    --------
    >>> location_phrase("bedroom")
    'in the bedroom'
    >>> location_phrase("desk")
    'on the desk'
    >>> location_phrase("living_room")
    'in the living room'
    """
    clean = room.strip().replace("_", " ").lower()
    # Strip trailing "room"/"area" qualifiers for surface lookup
    base = clean.split()[0] if clean else clean
    if base in _ON_SURFACES or clean in _ON_SURFACES:
        return f"on the {clean}"
    return f"in the {clean}"


# ---------------------------------------------------------------------------
# 2. Plural-aware entity grammar helper
# ---------------------------------------------------------------------------

# Entity names (lowercased) that are grammatically plural or use "were"
_PLURAL_FORMS: Set[str] = {
    "car keys",
    "keys",
    "scissors",
    "glasses",
    "spectacles",
    "sunglasses",
    "goggles",
    "headphones",
    "earphones",
    "earbuds",
    "headset",
    "pants",
    "trousers",
    "jeans",
    "shorts",
    "tweezers",
    "pliers",
    "tongs",
    "chopsticks",
    "papers",
    "documents",
    "files",
    "tools",
    "supplies",
    "belongings",
    "belongings",
    "groceries",
    "clothes",
    "shoes",
    "boots",
    "socks",
    "gloves",
}

# Pronouns for continued sentences ("They were..." vs "It was...")
_PLURAL_PRONOUNS: Dict[str, str] = {
    "subject": "They",
    "were": "were",
    "object": "them",
}
_SINGULAR_PRONOUNS: Dict[str, str] = {
    "subject": "It",
    "were": "was",
    "object": "it",
}


def _is_plural(entity: str) -> bool:
    return entity.strip().lower() in _PLURAL_FORMS


def entity_subject(entity: str, past: bool = True) -> str:
    """
    Return "The <entity> was/were" (past) or "The <entity> is/are" (present).

    Examples
    --------
    >>> entity_subject("car keys")
    'The car keys were'
    >>> entity_subject("backpack")
    'The backpack was'
    >>> entity_subject("phone", past=False)
    'The phone is'
    """
    plural = _is_plural(entity)
    if past:
        verb = "were" if plural else "was"
    else:
        verb = "are" if plural else "is"
    return f"The {entity} {verb}"


def continuation_pronoun(entity: str) -> Dict[str, str]:
    """Return pronouns dict for follow-up sentences."""
    return _PLURAL_PRONOUNS if _is_plural(entity) else _SINGULAR_PRONOUNS


# ---------------------------------------------------------------------------
# 3. Observation classification
# ---------------------------------------------------------------------------


class ObservationKind(str, Enum):
    """Classification of a stored observation relative to current accepted state."""

    ACCEPTED_CURRENT = "ACCEPTED_CURRENT"
    """This observation is the source of the current belief (most recent accepted)."""

    ACCEPTED_TRANSITION = "ACCEPTED_TRANSITION"
    """This observation caused a confirmed accepted location transition."""

    REOBSERVATION = "REOBSERVATION"
    """Same room as current/prior accepted state — reinforcing evidence, no change."""

    CONFLICTING = "CONFLICTING"
    """Different room from current accepted state — did NOT replace the belief."""

    UNCERTAIN = "UNCERTAIN"
    """Confidence too low to classify definitively."""


def classify_observation(
    obs_room: str,
    obs_confidence: float,
    current_room: str,
    accepted_transition_rooms: Set[str],
    uncertainty_threshold: float = 0.35,
) -> ObservationKind:
    """
    Classify a single observation relative to the accepted current state.

    Parameters
    ----------
    obs_room : str
        Room the observation reported.
    obs_confidence : float
        Confidence of that observation (0.0–1.0).
    current_room : str
        The room currently held in memory as accepted state.
    accepted_transition_rooms : set of str
        Rooms that appear in confirmed state transitions for this entity.
    uncertainty_threshold : float
        Confidence below this is classified as UNCERTAIN.
    """
    if obs_confidence < uncertainty_threshold:
        return ObservationKind.UNCERTAIN

    norm_obs = obs_room.strip().lower().replace("_", " ")
    norm_curr = current_room.strip().lower().replace("_", " ")
    norm_trans = {r.strip().lower().replace("_", " ") for r in accepted_transition_rooms}

    if norm_obs == norm_curr:
        return ObservationKind.REOBSERVATION

    if norm_obs in norm_trans:
        return ObservationKind.ACCEPTED_TRANSITION

    return ObservationKind.CONFLICTING


# ---------------------------------------------------------------------------
# 4. Conflict answer generator
# ---------------------------------------------------------------------------


class ConflictAnswerGenerator:
    """
    Builds a deterministic, grammatically correct natural-language answer
    that correctly distinguishes:
    - Conflicting rejected evidence (do NOT say "previously observed in X")
    - Confirmed accepted movements (DO say "previously observed in X")
    - Reobservations (say "no confirmed location change")
    """

    @staticmethod
    def build(
        entity: str,
        current_room: str,
        current_confidence: float,
        current_provenance: str,
        state_status: str,
        confirmed_previous_room: Optional[str],
        conflicting_obs: List[Dict[str, Any]],
    ) -> str:
        """
        Build the natural-language answer.

        Parameters
        ----------
        entity : str
        current_room : str
        current_confidence : float  (0.0–1.0)
        current_provenance : str  (e.g. "user", "sensor", "agent")
        state_status : str  (e.g. "OBSERVED", "MOVED", "REOBSERVED")
        confirmed_previous_room : Optional[str]
            Room from the most recent *accepted* state transition, or None.
        conflicting_obs : list of dict
            Each dict: {room, confidence, provenance, timestamp, reason}
        """
        subj = entity_subject(entity, past=False)
        subj_past = entity_subject(entity, past=True)
        pron = continuation_pronoun(entity)
        curr_phrase = location_phrase(current_room)
        conf_pct = int(round(current_confidence * 100))
        prov_label = _provenance_label(current_provenance)

        has_conflict = bool(conflicting_obs)
        has_confirmed_prev = confirmed_previous_room is not None

        # --- Conflict case ---
        if has_conflict:
            # Main clause: current belief
            answer = (
                f"{subj} currently believed to be {curr_phrase} "
                f"with {conf_pct}% confidence, based on a {prov_label} observation."
            )
            # Append each conflicting observation
            for c_obs in conflicting_obs:
                c_room = c_obs.get("room", "unknown").replace("_", " ")
                c_prov = _short_provenance_label(c_obs.get("provenance", "unknown"))
                c_conf = int(round(float(c_obs.get("confidence", 0.0)) * 100))
                c_phrase = location_phrase(c_room)
                answer += (
                    f" A lower-confidence {c_prov} observation reported {c_phrase} "
                    f"({c_conf}% confidence), but it did not replace the current belief."
                )
            # If there was also an accepted prior location before the current one
            if has_confirmed_prev:
                prev_phrase = location_phrase(confirmed_previous_room)
                answer += (
                    f" {pron['subject']} {pron['were']} previously confirmed {prev_phrase} "
                    f"before moving to {curr_phrase}."
                )
            return answer

        # --- Confirmed movement case ---
        if has_confirmed_prev and state_status in ("MOVED", "OBSERVED"):
            prev_phrase = location_phrase(confirmed_previous_room)
            answer = f"{subj_past} last observed {curr_phrase}."
            answer += f" {pron['subject']} {pron['were']} previously observed {prev_phrase}."
            return answer

        # --- Reobservation / stable belief case ---
        answer = f"{subj_past} last observed {curr_phrase}."
        if state_status == "REOBSERVED":
            answer += f" No confirmed location change recorded."
        return answer


def _provenance_label(provenance: str) -> str:
    """Convert internal provenance tag to a readable label (used for current-state description)."""
    _MAP = {
        "user": "high-trust user",
        "sensor": "sensor",
        "tool": "tool",
        "agent": "lower-confidence agent",
        "inferred": "inferred",
        "imported": "imported",
    }
    return _MAP.get(str(provenance).strip().lower(), provenance)


def _short_provenance_label(provenance: str) -> str:
    """Compact provenance label used inside conflict clauses (no repeated qualifiers)."""
    _MAP = {
        "user": "user",
        "sensor": "sensor",
        "tool": "tool",
        "agent": "agent",
        "inferred": "inferred",
        "imported": "imported",
    }
    return _MAP.get(str(provenance).strip().lower(), provenance)
