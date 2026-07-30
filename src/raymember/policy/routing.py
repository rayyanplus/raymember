"""Policy router evaluating observation characteristics to select memory update strategies."""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from raymember.schemas import Location, ObservationInput


@dataclass
class RoutingDecision:
    selected_mode: str  # "direct", "probabilistic", "hybrid", "preserve"
    reason: str
    is_conflicting: bool
    is_low_confidence: bool
    is_out_of_order: bool


class PolicyRouter:
    """
    Evaluates incoming observation against current memory state
    to determine whether to apply direct update, probabilistic fusion, or hybrid trust weighting.
    """

    @staticmethod
    def route_observation(
        current_state: Optional[Dict[str, Any]],
        new_obs: ObservationInput,
    ) -> RoutingDecision:
        conf = float(new_obs.confidence)
        src = str(new_obs.source).lower()

        if current_state is None:
            return RoutingDecision(
                selected_mode="direct",
                reason="Initial observation for new entity; initialized memory state directly.",
                is_conflicting=False,
                is_low_confidence=conf < 0.4,
                is_out_of_order=False,
            )

        curr_loc = current_state.get("location", {})
        curr_room = curr_loc.get("room", "").lower()

        obs_loc = new_obs.location.to_dict() if isinstance(new_obs.location, Location) else new_obs.location
        obs_room = obs_loc.get("room", "").lower()

        is_room_conflict = (curr_room != "" and obs_room != "" and curr_room != obs_room)
        is_low_conf = conf < 0.35

        if is_low_conf:
            return RoutingDecision(
                selected_mode="preserve",
                reason=f"Low observation confidence ({conf:.2f} < 0.35); preserved existing memory belief in {curr_room}.",
                is_conflicting=is_room_conflict,
                is_low_confidence=True,
                is_out_of_order=False,
            )

        if not is_room_conflict and conf >= 0.85:
            return RoutingDecision(
                selected_mode="direct",
                reason=f"High-confidence consistent observation ({conf:.2f}) in {obs_room}; confirmed current belief.",
                is_conflicting=False,
                is_low_confidence=False,
                is_out_of_order=False,
            )

        if is_room_conflict and conf < 0.65:
            return RoutingDecision(
                selected_mode="hybrid",
                reason=f"Conflicting observation in {obs_room} with moderate confidence ({conf:.2f}); applied Hybrid Policy trust weighting.",
                is_conflicting=True,
                is_low_confidence=False,
                is_out_of_order=False,
            )

        return RoutingDecision(
            selected_mode="probabilistic",
            reason=f"Observation in {obs_room} processed via Probabilistic Bayesian Belief Fusion.",
            is_conflicting=is_room_conflict,
            is_low_confidence=False,
            is_out_of_order=False,
        )
