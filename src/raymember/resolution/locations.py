"""
Canonical location resolution engine with deterministic aliases and fuzzy matching.
"""

from dataclasses import dataclass, field
import difflib
from typing import Any, Dict, List, Optional, Set, Tuple
from raymember.resolution.normalization import normalize_separators, normalize_string


DEFAULT_BUILTIN_ALIASES: Dict[str, str] = {
    "washroom": "bathroom",
    "restroom": "bathroom",
    "toilet": "bathroom",
    "livingroom": "living room",
    "lounge": "living room",
    "sitting room": "living room",
    "bed room": "bedroom",
    "cooking area": "kitchen",
}


@dataclass
class LocationResolutionResult:
    """Structured result of location resolution."""

    canonical_location: str
    raw_location: str
    normalized_location: str
    resolution_method: str  # EXACT, NORMALIZED, ALIAS, FUZZY, NEW, AMBIGUOUS
    resolution_confidence: float
    candidate_locations: List[Dict[str, Any]] = field(default_factory=list)
    requires_confirmation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_location": self.canonical_location,
            "raw_location": self.raw_location,
            "normalized_location": self.normalized_location,
            "resolution_method": self.resolution_method,
            "resolution_confidence": self.resolution_confidence,
            "candidate_locations": self.candidate_locations,
            "requires_confirmation": self.requires_confirmation,
        }


class LocationResolver:
    """
    Resolves raw location strings into canonical locations using:
    1. Exact / Normalized matching
    2. Built-in and custom Alias mapping
    3. Offline fuzzy matching (difflib)
    """

    def __init__(
        self,
        enable_builtin_aliases: bool = True,
        custom_aliases: Optional[Dict[str, str]] = None,
        fuzzy_accept_threshold: float = 0.88,
        fuzzy_confirm_threshold: float = 0.65,
    ):
        self.enable_builtin_aliases = enable_builtin_aliases
        self.custom_aliases: Dict[str, str] = {}
        if custom_aliases:
            for k, v in custom_aliases.items():
                self.custom_aliases[normalize_separators(k)] = normalize_separators(v)

        self.fuzzy_accept_threshold = fuzzy_accept_threshold
        self.fuzzy_confirm_threshold = fuzzy_confirm_threshold

        # Persistent user confirmed aliases: raw_norm -> canonical_norm
        self.user_confirmed_aliases: Dict[str, str] = {}
        # Persistent user rejected mappings: set of (raw_norm, canonical_norm)
        self.user_rejected_mappings: Set[Tuple[str, str]] = set()

    def get_active_aliases(self) -> Dict[str, str]:
        """Returns current combined alias mapping (alias -> canonical)."""
        mapping: Dict[str, str] = {}
        if self.enable_builtin_aliases:
            for k, v in DEFAULT_BUILTIN_ALIASES.items():
                mapping[normalize_separators(k)] = normalize_separators(v)
        # Custom & user confirmed overrides
        for k, v in self.custom_aliases.items():
            mapping[k] = v
        for k, v in self.user_confirmed_aliases.items():
            mapping[k] = v
        return mapping

    def add_alias(self, alias: str, canonical: str) -> None:
        """Adds a custom alias."""
        norm_alias = normalize_separators(alias)
        norm_canonical = normalize_separators(canonical)
        self.custom_aliases[norm_alias] = norm_canonical

    def remove_alias(self, alias: str) -> None:
        """Removes a custom or built-in alias override."""
        norm_alias = normalize_separators(alias)
        self.custom_aliases.pop(norm_alias, None)
        self.user_confirmed_aliases.pop(norm_alias, None)

    def confirm_alias(self, alias: str, canonical: str) -> None:
        """User explicitly confirms an alias mapping."""
        norm_alias = normalize_separators(alias)
        norm_canonical = normalize_separators(canonical)
        self.user_confirmed_aliases[norm_alias] = norm_canonical
        # Remove from rejected set if present
        self.user_rejected_mappings.discard((norm_alias, norm_canonical))

    def reject_mapping(self, raw_location: str, canonical: Optional[str] = None) -> None:
        """User explicitly rejects a fuzzy mapping."""
        norm_raw = normalize_separators(raw_location)
        if canonical:
            norm_can = normalize_separators(canonical)
            self.user_rejected_mappings.add((norm_raw, norm_can))
        else:
            # Reject all current candidate mappings for this raw location
            active = self.get_active_aliases()
            if norm_raw in active:
                self.user_rejected_mappings.add((norm_raw, active[norm_raw]))

    def resolve(
        self,
        raw_location: str,
        known_locations: Optional[List[str]] = None,
    ) -> LocationResolutionResult:
        """
        Resolves a raw location string into a structured LocationResolutionResult.
        """
        raw_str = raw_location or ""
        norm = normalize_separators(raw_str)

        if not norm:
            return LocationResolutionResult(
                canonical_location="unknown",
                raw_location=raw_str,
                normalized_location="",
                resolution_method="EXACT",
                resolution_confidence=1.0,
                requires_confirmation=False,
            )

        # Active alias lookup
        active_aliases = self.get_active_aliases()

        # Known canonical targets set
        known_set: Set[str] = set()
        if known_locations:
            for loc in known_locations:
                if loc:
                    known_set.add(normalize_separators(loc))

        # Add canonicals from aliases to known targets
        for canon in active_aliases.values():
            known_set.add(canon)

        # 1. Alias Match (including exact alias)
        if norm in active_aliases:
            canon = active_aliases[norm]
            if (norm, canon) not in self.user_rejected_mappings:
                return LocationResolutionResult(
                    canonical_location=canon,
                    raw_location=raw_str,
                    normalized_location=norm,
                    resolution_method="ALIAS",
                    resolution_confidence=1.0,
                    requires_confirmation=False,
                )

        # 2. Exact match with a known canonical room
        if norm in known_set:
            return LocationResolutionResult(
                canonical_location=norm,
                raw_location=raw_str,
                normalized_location=norm,
                resolution_method="EXACT" if raw_str.strip() == norm else "NORMALIZED",
                resolution_confidence=1.0,
                requires_confirmation=False,
            )

        # 3. Fuzzy match against known locations and alias keys
        candidates: List[Dict[str, Any]] = []
        candidates_seen: Set[str] = set()

        all_targets = list(known_set)

        for target in all_targets:
            if (norm, target) in self.user_rejected_mappings:
                continue

            ratio = difflib.SequenceMatcher(None, norm, target).ratio()

            if ratio >= self.fuzzy_confirm_threshold:
                if target not in candidates_seen:
                    candidates_seen.add(target)
                    candidates.append({
                        "location": target,
                        "confidence": round(float(ratio), 3),
                    })

        # Sort candidates descending by confidence
        candidates.sort(key=lambda x: x["confidence"], reverse=True)

        if candidates:
            best = candidates[0]
            best_conf = best["confidence"]
            best_target = best["location"]

            if best_conf >= self.fuzzy_accept_threshold:
                return LocationResolutionResult(
                    canonical_location=best_target,
                    raw_location=raw_str,
                    normalized_location=norm,
                    resolution_method="FUZZY",
                    resolution_confidence=best_conf,
                    candidate_locations=candidates,
                    requires_confirmation=False,
                )
            elif best_conf >= self.fuzzy_confirm_threshold:
                # Medium-confidence fuzzy match -> requires confirmation, do NOT silently merge!
                return LocationResolutionResult(
                    canonical_location=norm,
                    raw_location=raw_str,
                    normalized_location=norm,
                    resolution_method="AMBIGUOUS",
                    resolution_confidence=best_conf,
                    candidate_locations=candidates,
                    requires_confirmation=True,
                )

        # 4. Low confidence or nonsense input -> treat as NEW location
        return LocationResolutionResult(
            canonical_location=norm,
            raw_location=raw_str,
            normalized_location=norm,
            resolution_method="NEW",
            resolution_confidence=1.0,
            candidate_locations=candidates,
            requires_confirmation=False,
        )
