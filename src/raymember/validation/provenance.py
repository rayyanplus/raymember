"""Provenance definitions and default source reliability mappings."""

from enum import Enum
from typing import Dict


class ProvenanceType(str, Enum):
    SENSOR = "sensor"
    USER = "user"
    TOOL = "tool"
    AGENT = "agent"
    INFERRED = "inferred"
    IMPORTED = "imported"


class ProvenanceValidator:
    """Validates observation provenance tags and default reliability trust multipliers."""

    DEFAULT_TRUST_MAP: Dict[str, float] = {
        "user": 0.95,
        "sensor": 0.85,
        "tool": 0.80,
        "agent": 0.55,
        "inferred": 0.40,
        "imported": 0.75,
    }

    @classmethod
    def get_trust_multiplier(cls, provenance: str, custom_trust_map: Dict[str, float] = None) -> float:
        prov_lower = str(provenance).lower()
        mapping = custom_trust_map if custom_trust_map is not None else cls.DEFAULT_TRUST_MAP
        return mapping.get(prov_lower, 0.70)

    @classmethod
    def normalize_provenance(cls, provenance: str) -> str:
        p_str = str(provenance).lower()
        valid = [pt.value for pt in ProvenanceType]
        return p_str if p_str in valid else ProvenanceType.SENSOR.value
