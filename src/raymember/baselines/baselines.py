"""Baseline implementations for memory update decision making."""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from raymember.belief.engine import BeliefEngine, BeliefState
from raymember.schemas import Location, ObservationInput


class BaseMemoryPolicy:
    """Abstract interface for memory update policies."""

    def predict_action(
        self,
        current_state: Optional[Dict[str, Any]],
        new_obs: ObservationInput,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Predicts memory update action:
        Returns (action_str, updated_location_dict).
        Actions: INITIALIZE, UPDATE, REOBSERVE, PRESERVE, UNCERTAIN, NEW_ENTITY
        """
        raise NotImplementedError


class LatestObservationBaseline(BaseMemoryPolicy):
    """
    Baseline 1: Latest Observation.
    Blindly updates current state to the newest observation timestamp.
    """

    def predict_action(
        self,
        current_state: Optional[Dict[str, Any]],
        new_obs: ObservationInput,
    ) -> Tuple[str, Dict[str, Any]]:
        new_loc = new_obs.location.to_dict() if isinstance(new_obs.location, Location) else new_obs.location
        if current_state is None:
            return "INITIALIZE", new_loc

        # Check timestamps
        current_ts_str = current_state.get("last_seen", "")
        new_ts_str = new_obs.get_iso_timestamp()

        try:
            t_curr = datetime.fromisoformat(current_ts_str.replace("Z", "+00:00"))
            t_new = datetime.fromisoformat(new_ts_str.replace("Z", "+00:00"))
            if t_new < t_curr:
                # Out-of-order timestamp: reject update
                return "PRESERVE", current_state.get("location", new_loc)
        except Exception:
            pass

        old_room = current_state.get("location", {}).get("room")
        new_room = new_loc.get("room")

        if old_room != new_room:
            return "UPDATE", new_loc
        return "REOBSERVE", new_loc


class DeterministicRulesBaseline(BaseMemoryPolicy):
    """
    Baseline 2: Deterministic Confidence & Distance Rules.
    Uses thresholds (confidence >= 0.5, distance / room delta) to filter noisy observations.
    """

    def __init__(self, min_confidence: float = 0.5, movement_threshold: float = 0.5):
        self.min_confidence = min_confidence
        self.movement_threshold = movement_threshold

    def predict_action(
        self,
        current_state: Optional[Dict[str, Any]],
        new_obs: ObservationInput,
    ) -> Tuple[str, Dict[str, Any]]:
        new_loc = new_obs.location.to_dict() if isinstance(new_obs.location, Location) else new_obs.location
        if current_state is None:
            return "INITIALIZE", new_loc

        current_loc = current_state.get("location", new_loc)

        # Rule 1: Out-of-order timestamp check
        try:
            t_curr = datetime.fromisoformat(current_state.get("last_seen", "").replace("Z", "+00:00"))
            t_new = datetime.fromisoformat(new_obs.get_iso_timestamp().replace("Z", "+00:00"))
            if t_new < t_curr:
                return "PRESERVE", current_loc
        except Exception:
            pass

        # Rule 2: Confidence threshold
        if new_obs.confidence < self.min_confidence:
            return "PRESERVE", current_loc

        # Rule 3: Spatial movement detection
        old_room = current_loc.get("room", "")
        new_room = new_loc.get("room", "")

        if old_room.lower() != new_room.lower():
            return "UPDATE", new_loc

        # 3D coordinate distance check if available
        if (
            "x" in current_loc and "y" in current_loc and "z" in current_loc
            and "x" in new_loc and "y" in new_loc and "z" in new_loc
        ):
            dist = (
                (new_loc["x"] - current_loc["x"]) ** 2
                + (new_loc["y"] - current_loc["y"]) ** 2
                + (new_loc["z"] - current_loc["z"]) ** 2
            ) ** 0.5
            if dist >= self.movement_threshold:
                return "UPDATE", new_loc
            return "REOBSERVE", new_loc

        return "REOBSERVE", new_loc


class ProbabilisticEngineBaseline(BaseMemoryPolicy):
    """
    Baseline 3: Probabilistic Belief Engine without learned policy.
    Uses Bayesian belief fusion and entropy calculations to determine state updates.
    """

    def __init__(self, decay_rate: float = 0.05):
        self.engine = BeliefEngine(decay_rate=decay_rate)

    def predict_action(
        self,
        current_state: Optional[Dict[str, Any]],
        new_obs: ObservationInput,
    ) -> Tuple[str, Dict[str, Any]]:
        new_loc = new_obs.location.to_dict() if isinstance(new_obs.location, Location) else new_obs.location
        if current_state is None:
            return "INITIALIZE", new_loc

        # Convert current state to BeliefState object if present
        belief_data = current_state.get("belief_data")
        curr_b = None
        if belief_data and "location_beliefs" in belief_data:
            from raymember.belief.engine import LocationBeliefItem
            items = [LocationBeliefItem(**lb) for lb in belief_data["location_beliefs"]]
            curr_b = BeliefState(
                entity_id=current_state.get("entity_id", "entity_1"),
                location_beliefs=items,
                most_likely_location=belief_data.get("most_likely_location", new_loc),
                belief_confidence=belief_data.get("belief_confidence", 0.8),
            )

        updated_b = self.engine.fuse_observation(
            current_belief=curr_b,
            obs=new_obs,
            entity_id=current_state.get("entity_id", "entity_1"),
        )

        most_likely_loc = updated_b.most_likely_location
        if isinstance(most_likely_loc, str):
            most_likely_dict = {"room": most_likely_loc}
        else:
            most_likely_dict = dict(most_likely_loc)

        current_loc = current_state.get("location", new_loc)

        if updated_b.entropy > 0.7:
            return "UNCERTAIN", current_loc

        if most_likely_dict.get("room") != current_loc.get("room"):
            return "UPDATE", most_likely_dict

        return "REOBSERVE", most_likely_dict
