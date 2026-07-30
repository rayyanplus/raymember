"""Memory write safety validator enforcing provenance trust rules."""

from typing import Any, Dict, Optional, Tuple
from raymember.schemas import ObservationInput
from raymember.validation.provenance import ProvenanceValidator


class MemoryWriteValidator:
    """
    Validates whether an incoming observation should be allowed to modify existing memory state
    based on relative provenance trust, confidence thresholds, and write safety policies.
    """

    def __init__(self, source_reliability: Optional[Dict[str, float]] = None):
        self.trust_map = source_reliability or ProvenanceValidator.DEFAULT_TRUST_MAP

    def validate_write(
        self,
        current_state: Optional[Dict[str, Any]],
        new_obs: ObservationInput,
        provenance: str = "sensor",
    ) -> Tuple[bool, str, float]:
        norm_prov = ProvenanceValidator.normalize_provenance(provenance)
        trust_mult = ProvenanceValidator.get_trust_multiplier(norm_prov, self.trust_map)
        effective_conf = float(new_obs.confidence) * trust_mult

        if current_state is None:
            return True, f"Valid initial write with provenance '{norm_prov}' (effective conf: {effective_conf:.2f}).", effective_conf

        curr_prov = str(current_state.get("provenance", "sensor")).lower()
        curr_conf = float(current_state.get("confidence", 0.5)) * ProvenanceValidator.get_trust_multiplier(curr_prov, self.trust_map)

        # Rule 1: High-trust user observation cannot be overwritten by low-trust agent/inferred observation unless confidence is high
        if curr_prov == "user" and norm_prov in ("agent", "inferred") and effective_conf < curr_conf:
            return (
                False,
                f"Write blocked: Low-trust '{norm_prov}' observation (effective conf {effective_conf:.2f}) cannot overwrite high-trust 'user' memory (effective conf {curr_conf:.2f}).",
                effective_conf,
            )

        # Rule 2: Minimum confidence threshold for inferred observations
        if norm_prov == "inferred" and effective_conf < 0.25:
            return False, f"Write blocked: Inferred observation effective confidence {effective_conf:.2f} is below minimum threshold (0.25).", effective_conf

        return True, f"Write validated for provenance '{norm_prov}' (effective conf {effective_conf:.2f}).", effective_conf
