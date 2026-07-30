"""Confidence evaluation and validation logic."""

from datetime import datetime, timezone
import math
from typing import Optional


class ConfidenceEngine:
    """Deterministic validation and confidence scaling."""

    def __init__(self, min_confidence_threshold: float = 0.1, decay_rate: float = 0.05):
        self.min_confidence_threshold = min_confidence_threshold
        self.decay_rate = decay_rate

    def validate(self, confidence: float) -> float:
        """Enforces 0.0 <= confidence <= 1.0."""
        if math.isnan(confidence) or math.isinf(confidence):
            return 0.0
        return float(max(0.0, min(1.0, confidence)))

    def is_sufficient(self, confidence: float) -> bool:
        """Checks if confidence meets minimum required threshold."""
        return self.validate(confidence) >= self.min_confidence_threshold

    def calculate_decayed_confidence(
        self,
        base_confidence: float,
        last_seen_iso: str,
        current_time_iso: Optional[str] = None,
    ) -> float:
        """Applies exponential decay over time to past confidence."""
        base_conf = self.validate(base_confidence)
        if not last_seen_iso:
            return base_conf
        try:
            t_last = datetime.fromisoformat(last_seen_iso.replace("Z", "+00:00"))
            if current_time_iso:
                t_curr = datetime.fromisoformat(current_time_iso.replace("Z", "+00:00"))
            else:
                t_curr = datetime.now(timezone.utc)
            delta_hours = max(0.0, (t_curr - t_last).total_seconds() / 3600.0)
            decay_factor = math.exp(-self.decay_rate * delta_hours)
            return self.validate(base_conf * decay_factor)
        except Exception:
            return base_conf
