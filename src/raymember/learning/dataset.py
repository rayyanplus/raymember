"""Dataset generator producing scenario-level train/val/test split datasets using counterfactual label assignment."""

import random
from typing import Any, Dict, List, Tuple
import numpy as np

from raymember.learning.features import FeatureExtractor
from raymember.simulation.world import SimulationScenario, SimulationWorld


class DatasetGenerator:
    """Generates feature matrices, counterfactual target labels, and continuous trust targets."""

    # 5 Target Action Classes (NEW_ENTITY removed; handled deterministically in SDK)
    ACTION_MAP: Dict[str, int] = {
        "INITIALIZE": 0,
        "UPDATE": 1,
        "REOBSERVE": 2,
        "PRESERVE": 3,
        "UNCERTAIN": 4,
    }

    INT_TO_ACTION: Dict[int, str] = {v: k for k, v in ACTION_MAP.items()}

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.feature_extractor = FeatureExtractor()

    def generate_split_dataset(
        self,
        num_scenarios: int = 50,
        steps_per_scenario: int = 20,
        noise_condition: str = "mixed",
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray],  # Train (X, y_action, y_trust)
        Tuple[np.ndarray, np.ndarray, np.ndarray],  # Val (X, y_action, y_trust)
        Tuple[np.ndarray, np.ndarray, np.ndarray],  # Test (X, y_action, y_trust)
        List[SimulationScenario],                    # All generated scenarios
    ]:
        """
        Generates scenarios and splits strictly by COMPLETE SCENARIO (70% train, 15% val, 15% test).
        Ensures zero scenario trajectory overlap between splits!
        """
        world = SimulationWorld(random_seed=self.random_seed)
        scenarios: List[SimulationScenario] = []

        for idx in range(num_scenarios):
            sc_id = f"sc_{idx+1:03d}"
            sc = world.generate_scenario(
                scenario_id=sc_id,
                num_steps=steps_per_scenario,
                noise_condition=noise_condition,
            )
            scenarios.append(sc)

        rng = random.Random(self.random_seed)
        shuffled_scenarios = list(scenarios)
        rng.shuffle(shuffled_scenarios)

        n_sc = len(shuffled_scenarios)
        n_train = int(n_sc * 0.70)
        n_val = int(n_sc * 0.15)

        train_scenarios = shuffled_scenarios[:n_train]
        val_scenarios = shuffled_scenarios[n_train : n_train + n_val]
        test_scenarios = shuffled_scenarios[n_train + n_val :]

        X_train, y_act_tr, y_tr_tr = self._build_matrix(train_scenarios)
        X_val, y_act_val, y_tr_val = self._build_matrix(val_scenarios)
        X_test, y_act_te, y_tr_te = self._build_matrix(test_scenarios)

        return (
            (X_train, y_act_tr, y_tr_tr),
            (X_val, y_act_val, y_tr_val),
            (X_test, y_act_te, y_tr_te),
            scenarios,
        )

    def _build_matrix(
        self, scenarios: List[SimulationScenario]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_list: List[np.ndarray] = []
        y_action_list: List[int] = []
        y_trust_list: List[float] = []

        for sc in scenarios:
            state_history: Dict[str, Dict] = {}
            for step in sc.steps:
                target_eid = step.ground_truth_entity_id
                curr_state = state_history.get(target_eid)

                feat = self.feature_extractor.extract(curr_state, step.observation)

                best_action, trust_weight = self._evaluate_counterfactual_best_action(
                    curr_state, step.observation, step.ground_truth_location
                )

                target_action_idx = self.ACTION_MAP[best_action]

                X_list.append(feat)
                y_action_list.append(target_action_idx)
                y_trust_list.append(trust_weight)

                loc_dict = step.observation.location.to_dict()
                if curr_state is None or best_action in ("INITIALIZE", "UPDATE", "REOBSERVE"):
                    state_history[target_eid] = {
                        "entity_id": target_eid,
                        "location": loc_dict,
                        "confidence": step.observation_confidence,
                        "last_seen": step.observation.get_iso_timestamp(),
                        "attributes": step.observation.attributes or {},
                    }

        if not X_list:
            return (
                np.empty((0, 9), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
            )

        return (
            np.vstack(X_list),
            np.array(y_action_list, dtype=np.int64),
            np.array(y_trust_list, dtype=np.float32),
        )

    def _evaluate_counterfactual_best_action(
        self,
        curr_state: Dict[str, Any] | None,
        obs: Any,
        gt_location: Dict[str, Any],
    ) -> Tuple[str, float]:
        """
        Evaluates candidate actions against hidden physical ground truth.
        Selects the action minimizing resulting world-state location error.
        Returns: (best_action_string, continuous_trust_target_0_to_1).
        """
        gt_room = gt_location.get("room", "").lower()
        obs_loc = obs.location.to_dict()
        obs_room = obs_loc.get("room", "").lower()

        trust_weight = 1.0 if (obs_room == gt_room and obs.confidence >= 0.4) else 0.0

        if curr_state is None:
            return "INITIALIZE", trust_weight

        curr_loc = curr_state.get("location", {})
        curr_room = curr_loc.get("room", "").lower()

        err_obs = 0.0 if (obs_room == gt_room) else 1.0
        err_curr = 0.0 if (curr_room == gt_room) else 1.0

        if err_obs < err_curr:
            best_action = "UPDATE" if (obs_room != curr_room) else "REOBSERVE"
        elif err_curr < err_obs:
            best_action = "PRESERVE" if obs.confidence >= 0.3 else "UNCERTAIN"
        else:
            if obs_room == curr_room and obs_room == gt_room:
                best_action = "REOBSERVE"
            elif obs.confidence < 0.3:
                best_action = "UNCERTAIN"
            elif obs_room != curr_room and obs_room == gt_room:
                best_action = "UPDATE"
            else:
                best_action = "PRESERVE"

        return best_action, trust_weight
