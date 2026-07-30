"""Feature engineering module extracting tabular features for learned policy."""

from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

from raymember.schemas import Location, ObservationInput


class FeatureExtractor:
    """Extracts numerical features from (current_state, new_observation) pair."""

    FEATURE_NAMES: List[str] = [
        "obs_confidence",
        "source_reliability",
        "time_delta_hours",
        "is_out_of_order",
        "room_match",
        "spatial_distance",
        "attribute_similarity",
        "current_confidence",
        "current_entropy",
    ]

    SOURCE_WEIGHTS: Dict[str, float] = {
        "simulator": 1.0,
        "camera": 0.95,
        "lidar": 0.95,
        "user": 0.90,
        "sensor": 0.85,
        "unknown": 0.70,
    }

    def extract(
        self,
        current_state: Optional[Dict[str, Any]],
        new_obs: ObservationInput,
    ) -> np.ndarray:
        """
        Extracts 1D feature array of shape (9,).
        """
        obs_conf = float(new_obs.confidence)
        src_weight = self.SOURCE_WEIGHTS.get(new_obs.source.lower(), 0.7)

        if current_state is None:
            # Initial observation default feature values
            return np.array(
                [obs_conf, src_weight, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                dtype=np.float32,
            )

        # 1. Time delta and out-of-order check
        curr_ts_str = current_state.get("last_seen", "")
        new_ts_str = new_obs.get_iso_timestamp()
        time_delta_hours = 0.0
        is_out_of_order = 0.0

        try:
            t_curr = datetime.fromisoformat(curr_ts_str.replace("Z", "+00:00"))
            t_new = datetime.fromisoformat(new_ts_str.replace("Z", "+00:00"))
            sec_delta = (t_new - t_curr).total_seconds()
            if sec_delta < 0:
                is_out_of_order = 1.0
                time_delta_hours = abs(sec_delta) / 3600.0
            else:
                time_delta_hours = sec_delta / 3600.0
        except Exception:
            pass

        # 2. Spatial room match and 3D coordinate distance
        curr_loc = current_state.get("location", {})
        new_loc = new_obs.location.to_dict() if isinstance(new_obs.location, Location) else new_obs.location

        curr_room = curr_loc.get("room", "").lower()
        new_room = new_loc.get("room", "").lower()
        room_match = 1.0 if (curr_room and curr_room == new_room) else 0.0

        spatial_dist = 1.0 if room_match == 0.0 else 0.0
        if (
            "x" in curr_loc and "y" in curr_loc and "z" in curr_loc
            and "x" in new_loc and "y" in new_loc and "z" in new_loc
            and curr_loc["x"] is not None and new_loc["x"] is not None
        ):
            spatial_dist = float(
                (
                    (new_loc["x"] - curr_loc["x"]) ** 2
                    + (new_loc["y"] - curr_loc["y"]) ** 2
                    + (new_loc["z"] - curr_loc["z"]) ** 2
                )
                ** 0.5
            )

        # 3. Attribute overlap similarity
        attr_sim = self._compute_attribute_similarity(
            current_state.get("attributes", {}), new_obs.attributes or {}
        )

        # 4. Current state belief confidence & entropy
        curr_conf = float(current_state.get("confidence", 0.8))
        curr_entropy = float(current_state.get("belief_data", {}).get("entropy", 0.0))

        return np.array(
            [
                obs_conf,
                src_weight,
                time_delta_hours,
                is_out_of_order,
                room_match,
                spatial_dist,
                attr_sim,
                curr_conf,
                curr_entropy,
            ],
            dtype=np.float32,
        )

    def _compute_attribute_similarity(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> float:
        if not dict1 or not dict2:
            return 1.0
        shared = set(dict1.keys()).intersection(set(dict2.keys()))
        if not shared:
            return 0.5
        matches = sum(1 for k in shared if str(dict1[k]).lower() == str(dict2[k]).lower())
        return float(matches / len(shared))
