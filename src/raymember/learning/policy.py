"""Learned Update Policies: Model A (Action Classifier) & Model B (Experimental Hybrid Policy)."""

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from raymember.baselines.baselines import BaseMemoryPolicy
from raymember.belief.engine import BeliefEngine, BeliefState, LocationBeliefItem
from raymember.learning.dataset import DatasetGenerator
from raymember.learning.features import FeatureExtractor
from raymember.schemas import Location, ObservationInput


# ----------------------------------------------------------------------
# 1. Model A: Learned Action Classifier Policy
# ----------------------------------------------------------------------

class LearnedUpdatePolicy(BaseMemoryPolicy):
    """
    Model A Action Classifier: Predicts optimal memory update action
    (INITIALIZE, UPDATE, REOBSERVE, PRESERVE, UNCERTAIN, NEW_ENTITY)
    using trained Random Forest or Logistic Regression classifier.
    """

    def __init__(self, model_type: str = "random_forest", random_seed: int = 42):
        self.model_type = model_type
        self.random_seed = random_seed
        self.feature_extractor = FeatureExtractor()
        self.dataset_gen = DatasetGenerator(random_seed=random_seed)

        if model_type == "logistic_regression":
            self.clf = LogisticRegression(max_iter=1000, random_state=random_seed)
        else:
            self.clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_seed)

        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains policy classifier on extracted feature matrix X and target labels y."""
        if len(X_train) == 0:
            return
        self.clf.fit(X_train, y_train)
        self.is_trained = True

    def predict_action(
        self,
        current_state: Optional[Dict[str, Any]],
        new_obs: ObservationInput,
    ) -> Tuple[str, Dict[str, Any]]:
        new_loc = new_obs.location.to_dict() if isinstance(new_obs.location, Location) else new_obs.location

        if current_state is None:
            return "INITIALIZE", new_loc

        feat = self.feature_extractor.extract(current_state, new_obs).reshape(1, -1)

        if not self.is_trained:
            room_match = feat[0, 4]
            conf = feat[0, 0]
            if conf < 0.3:
                return "PRESERVE", current_state.get("location", new_loc)
            if room_match == 0.0:
                return "UPDATE", new_loc
            return "REOBSERVE", new_loc

        pred_idx = int(self.clf.predict(feat)[0])
        action = self.dataset_gen.INT_TO_ACTION.get(pred_idx, "UPDATE")
        current_loc = current_state.get("location", new_loc)

        if action in ("PRESERVE", "UNCERTAIN"):
            return action, current_loc
        if action in ("UPDATE", "INITIALIZE", "REOBSERVE"):
            return action, new_loc

        return action, new_loc

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump({"clf": self.clf, "model_type": self.model_type, "is_trained": self.is_trained}, filepath)

    def load(self, filepath: str) -> None:
        data = joblib.load(filepath)
        self.clf = data["clf"]
        self.model_type = data["model_type"]
        self.is_trained = data["is_trained"]


# ----------------------------------------------------------------------
# 2. Model B: Experimental Hybrid Policy (Continuous Trust Weight)
# ----------------------------------------------------------------------

@dataclass
class HybridDecisionOutput:
    """Interpretable output details of Hybrid Policy decision."""

    action: str
    location: Dict[str, Any]
    obs_confidence: float
    source_reliability: float
    learned_trust_weight: float
    effective_evidence_weight: float


class HybridPolicy(BaseMemoryPolicy):
    """
    Experimental Hybrid Policy: Preserves the Probabilistic Belief Engine
    and uses machine learning to estimate a continuous observation trust weight [0.0, 1.0].
    
    effective_weight = obs_confidence * source_reliability * learned_trust_weight
    """

    def __init__(self, random_seed: int = 42, decay_rate: float = 0.05):
        self.random_seed = random_seed
        self.feature_extractor = FeatureExtractor()
        self.belief_engine = BeliefEngine(decay_rate=decay_rate)
        self.regressor = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=random_seed)
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_trust_train: np.ndarray) -> None:
        """Trains regressor to predict continuous trust weight y_trust in range [0.0, 1.0]."""
        if len(X_train) == 0:
            return
        self.regressor.fit(X_train, y_trust_train)
        self.is_trained = True

    def predict_trust_weight(self, current_state: Optional[Dict[str, Any]], new_obs: ObservationInput) -> float:
        """Predicts continuous trust weight w_trust in range [0.0, 1.0]."""
        if not self.is_trained:
            return 1.0 if new_obs.confidence >= 0.4 else 0.2

        feat = self.feature_extractor.extract(current_state, new_obs).reshape(1, -1)
        raw_pred = float(self.regressor.predict(feat)[0])
        return float(max(0.0, min(1.0, raw_pred)))

    def evaluate_decision(
        self,
        current_state: Optional[Dict[str, Any]],
        new_obs: ObservationInput,
    ) -> HybridDecisionOutput:
        """Returns action, location, and full interpretable intermediate weight values."""
        new_loc = new_obs.location.to_dict() if isinstance(new_obs.location, Location) else new_obs.location
        src_rel = float(self.belief_engine.SOURCE_RELIABILITY.get(new_obs.source.lower(), 0.7))
        obs_conf = float(new_obs.confidence)

        if current_state is None:
            return HybridDecisionOutput(
                action="INITIALIZE",
                location=new_loc,
                obs_confidence=obs_conf,
                source_reliability=src_rel,
                learned_trust_weight=1.0,
                effective_evidence_weight=obs_conf * src_rel,
            )

        w_trust = self.predict_trust_weight(current_state, new_obs)
        effective_weight = obs_conf * src_rel * w_trust
        current_loc = current_state.get("location", new_loc)

        belief_data = current_state.get("belief_data")
        curr_b = None
        if belief_data and "location_beliefs" in belief_data:
            items = [LocationBeliefItem(**lb) for lb in belief_data["location_beliefs"]]
            curr_b = BeliefState(
                entity_id=current_state.get("entity_id", "entity_1"),
                location_beliefs=items,
                most_likely_location=belief_data.get("most_likely_location", current_loc),
                belief_confidence=belief_data.get("belief_confidence", 0.8),
            )

        scaled_obs = ObservationInput(
            entity=new_obs.entity,
            attributes=new_obs.attributes,
            location=new_obs.location,
            confidence=effective_weight,
            source=new_obs.source,
            timestamp=new_obs.timestamp,
        )

        updated_b = self.belief_engine.fuse_observation(
            current_belief=curr_b,
            obs=scaled_obs,
            entity_id=current_state.get("entity_id", "entity_1"),
        )

        most_likely_loc = updated_b.most_likely_location
        most_likely_dict = most_likely_loc if isinstance(most_likely_loc, dict) else {"room": str(most_likely_loc)}

        if effective_weight < 0.15:
            action = "PRESERVE"
            target_loc = current_loc
        elif most_likely_dict.get("room") != current_loc.get("room"):
            action = "UPDATE"
            target_loc = most_likely_dict
        else:
            action = "REOBSERVE"
            target_loc = most_likely_dict

        return HybridDecisionOutput(
            action=action,
            location=target_loc,
            obs_confidence=obs_conf,
            source_reliability=src_rel,
            learned_trust_weight=w_trust,
            effective_evidence_weight=effective_weight,
        )

    def predict_action(
        self,
        current_state: Optional[Dict[str, Any]],
        new_obs: ObservationInput,
    ) -> Tuple[str, Dict[str, Any]]:
        out = self.evaluate_decision(current_state, new_obs)
        return out.action, out.location

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump({"regressor": self.regressor, "is_trained": self.is_trained}, filepath)

    def load(self, filepath: str) -> None:
        data = joblib.load(filepath)
        self.regressor = data["regressor"]
        self.is_trained = data["is_trained"]
