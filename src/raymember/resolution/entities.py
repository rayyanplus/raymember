"""
Entity resolution and normalization module.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from raymember.resolution.normalization import normalize_entity_name


@dataclass
class EntityResolutionResult:
    """Structured result of entity resolution."""

    canonical_name: str
    raw_name: str
    normalized_name: str
    resolution_method: str  # EXACT, NORMALIZED, ALIAS


class EntityResolver:
    """
    Resolves raw entity names into canonical normalized entity names.
    Preserves distinctions like 'black backpack' vs 'blue backpack'.
    """

    def __init__(self, custom_aliases: Optional[Dict[str, str]] = None):
        self.custom_aliases: Dict[str, str] = {}
        if custom_aliases:
            for k, v in custom_aliases.items():
                self.custom_aliases[normalize_entity_name(k)] = normalize_entity_name(v)

    def resolve(self, raw_name: str) -> EntityResolutionResult:
        """Resolves an entity name string."""
        raw_str = raw_name or ""
        norm = normalize_entity_name(raw_str)

        if norm in self.custom_aliases:
            return EntityResolutionResult(
                canonical_name=self.custom_aliases[norm],
                raw_name=raw_str,
                normalized_name=norm,
                resolution_method="ALIAS",
            )

        if raw_str.strip() == norm:
            method = "EXACT"
        else:
            method = "NORMALIZED"

        return EntityResolutionResult(
            canonical_name=norm,
            raw_name=raw_str,
            normalized_name=norm,
            resolution_method=method,
        )
