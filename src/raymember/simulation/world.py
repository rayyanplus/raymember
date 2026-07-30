"""Hidden Ground Truth Synthetic World Simulator supporting Scenario Families A-H and OOD scenarios."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import random
from typing import Any, Dict, List, Optional
from raymember.schemas import Location, ObservationInput


@dataclass
class GroundTruthStep:
    step_id: str
    ground_truth_entity_id: str
    ground_truth_location: Dict[str, Any]
    ground_truth_movement_event: bool
    observation: ObservationInput
    observation_confidence: float
    ground_truth_target_action: str = "UPDATE"
    is_false_detection: bool = False
    is_delayed: bool = False
    is_out_of_order: bool = False


@dataclass
class SimulationScenario:
    scenario_id: str
    noise_condition: str
    steps: List[GroundTruthStep]


class SimulationWorld:
    """
    Maintains physical hidden ground truth world state independently of Raymember policies.
    Generates synthetic observation streams with configurable noise conditions and scenario families.
    """

    ROOMS = ["bedroom", "living_room", "kitchen", "office", "hallway", "garage"]
    ENTITIES = ["backpack", "laptop", "keys", "wallet", "mug"]

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def generate_scenario(
        self,
        scenario_id: str,
        num_steps: int = 20,
        noise_condition: str = "mixed",
        scenario_family: Optional[str] = None,
    ) -> SimulationScenario:
        rng = random.Random(self.random_seed + hash(scenario_id))

        if scenario_family == "repeated_movement":
            return self._generate_family_a_repeated_movement(scenario_id, num_steps, rng)
        if scenario_family == "long_gaps":
            return self._generate_family_b_long_gaps(scenario_id, num_steps, rng)
        if scenario_family == "multiple_similar_entities":
            return self._generate_family_c_similar_entities(scenario_id, num_steps, rng)
        if scenario_family == "dynamic_source_reliability":
            return self._generate_family_d_dynamic_sources(scenario_id, num_steps, rng)
        if scenario_family == "stale_observation_bursts":
            return self._generate_family_e_stale_bursts(scenario_id, num_steps, rng)
        if scenario_family == "partial_observations":
            return self._generate_family_f_partial_obs(scenario_id, num_steps, rng)
        if scenario_family == "ambiguous_identity":
            return self._generate_family_g_ambiguous_identity(scenario_id, num_steps, rng)
        if scenario_family == "adversarial_mixed_noise":
            return self._generate_family_h_adversarial(scenario_id, num_steps, rng)

        return self._generate_standard_noise_scenario(scenario_id, num_steps, noise_condition, rng)

    def generate_ood_scenario(
        self,
        scenario_id: str,
        num_steps: int = 25,
    ) -> SimulationScenario:
        """Generates Out-of-Distribution (OOD) test scenario with extreme noise parameters."""
        rng = random.Random(self.random_seed + 9999 + hash(scenario_id))
        target_entity = "laptop"
        curr_room = "office"
        curr_time = datetime(2026, 7, 29, 12, 0, 0, tzinfo=timezone.utc)
        steps: List[GroundTruthStep] = []

        for i in range(num_steps):
            step_time = curr_time + timedelta(minutes=i * 5)
            is_false = rng.random() < 0.40
            moved = (rng.random() < 0.35 and not is_false)

            if moved:
                avail = [r for r in self.ROOMS if r != curr_room]
                curr_room = rng.choice(avail)

            gt_loc = {"room": curr_room, "x": round(rng.uniform(0, 10), 1), "y": 0.0, "z": round(rng.uniform(0, 10), 1)}

            if is_false:
                obs_room = rng.choice([r for r in self.ROOMS if r != curr_room])
                conf = round(rng.uniform(0.70, 0.95), 2)
            else:
                obs_room = curr_room
                conf = round(rng.uniform(0.80, 0.98), 2)

            obs_loc = Location(room=obs_room, x=gt_loc["x"], y=gt_loc["y"], z=gt_loc["z"])
            obs = ObservationInput(
                entity=target_entity,
                attributes={"color": "silver"},
                location=obs_loc,
                confidence=conf,
                source="camera",
                timestamp=step_time.isoformat(),
            )

            steps.append(
                GroundTruthStep(
                    step_id=f"{scenario_id}_s{i+1}",
                    ground_truth_entity_id=target_entity,
                    ground_truth_location=gt_loc,
                    ground_truth_movement_event=moved,
                    ground_truth_target_action="UPDATE" if moved else "REOBSERVE",
                    observation=obs,
                    observation_confidence=conf,
                    is_false_detection=is_false,
                )
            )

        return SimulationScenario(scenario_id=scenario_id, noise_condition="ood_extreme", steps=steps)

    # ------------------------------------------------------------------
    # Scenario Families A - H
    # ------------------------------------------------------------------

    def _generate_family_a_repeated_movement(self, scenario_id: str, num_steps: int, rng: random.Random) -> SimulationScenario:
        path = ["bedroom", "kitchen", "living_room", "bedroom"]
        steps: List[GroundTruthStep] = []
        curr_time = datetime(2026, 7, 29, 10, 0, 0, tzinfo=timezone.utc)
        target_entity = "keys"

        for i in range(num_steps):
            room = path[i % len(path)]
            moved = (i > 0 and path[i % len(path)] != path[(i - 1) % len(path)])
            gt_loc = {"room": room, "x": 1.0, "y": 0.0, "z": 1.0}

            step_time = curr_time + timedelta(minutes=i * 10)
            obs = ObservationInput(
                entity=target_entity,
                location=Location(room=room, x=1.0, y=0.0, z=1.0),
                confidence=0.95,
                source="beacon",
                timestamp=step_time.isoformat(),
            )
            steps.append(
                GroundTruthStep(
                    step_id=f"{scenario_id}_s{i+1}",
                    ground_truth_entity_id=target_entity,
                    ground_truth_location=gt_loc,
                    ground_truth_movement_event=moved,
                    ground_truth_target_action="UPDATE" if moved else "REOBSERVE",
                    observation=obs,
                    observation_confidence=0.95,
                )
            )
        return SimulationScenario(scenario_id=scenario_id, noise_condition="family_a_repeated", steps=steps)

    def _generate_family_b_long_gaps(self, scenario_id: str, num_steps: int, rng: random.Random) -> SimulationScenario:
        gaps_minutes = [5, 30, 240, 1440]
        steps: List[GroundTruthStep] = []
        curr_time = datetime(2026, 7, 29, 10, 0, 0, tzinfo=timezone.utc)
        target_entity = "wallet"
        curr_room = "bedroom"

        for i in range(num_steps):
            gap = gaps_minutes[i % len(gaps_minutes)]
            curr_time += timedelta(minutes=gap)
            moved = (gap > 60)
            if moved:
                curr_room = rng.choice(["living_room", "office"])

            gt_loc = {"room": curr_room, "x": 2.0, "y": 0.0, "z": 2.0}
            obs = ObservationInput(
                entity=target_entity,
                location=Location(room=curr_room, x=2.0, y=0.0, z=2.0),
                confidence=0.88,
                source="rfid",
                timestamp=curr_time.isoformat(),
            )
            steps.append(
                GroundTruthStep(
                    step_id=f"{scenario_id}_s{i+1}",
                    ground_truth_entity_id=target_entity,
                    ground_truth_location=gt_loc,
                    ground_truth_movement_event=moved,
                    ground_truth_target_action="UPDATE" if moved else "REOBSERVE",
                    observation=obs,
                    observation_confidence=0.88,
                )
            )
        return SimulationScenario(scenario_id=scenario_id, noise_condition="family_b_gaps", steps=steps)

    def _generate_family_c_similar_entities(self, scenario_id: str, num_steps: int, rng: random.Random) -> SimulationScenario:
        entities = ["black_backpack_1", "black_backpack_2", "blue_backpack_1"]
        steps: List[GroundTruthStep] = []
        curr_time = datetime(2026, 7, 29, 10, 0, 0, tzinfo=timezone.utc)

        for i in range(num_steps):
            e_id = entities[i % len(entities)]
            room = rng.choice(self.ROOMS)
            gt_loc = {"room": room, "x": 3.0, "y": 0.0, "z": 3.0}
            curr_time += timedelta(minutes=5)

            obs = ObservationInput(
                entity=e_id,
                attributes={"color": "black" if "black" in e_id else "blue"},
                location=Location(room=room, x=3.0, y=0.0, z=3.0),
                confidence=0.90,
                source="camera",
                timestamp=curr_time.isoformat(),
            )
            steps.append(
                GroundTruthStep(
                    step_id=f"{scenario_id}_s{i+1}",
                    ground_truth_entity_id=e_id,
                    ground_truth_location=gt_loc,
                    ground_truth_movement_event=False,
                    ground_truth_target_action="REOBSERVE",
                    observation=obs,
                    observation_confidence=0.90,
                )
            )
        return SimulationScenario(scenario_id=scenario_id, noise_condition="family_c_similar", steps=steps)

    def _generate_family_d_dynamic_sources(self, scenario_id: str, num_steps: int, rng: random.Random) -> SimulationScenario:
        steps: List[GroundTruthStep] = []
        curr_time = datetime(2026, 7, 29, 10, 0, 0, tzinfo=timezone.utc)

        for i in range(num_steps):
            src = "camera_a" if i < num_steps // 2 else "camera_b"
            conf = max(0.2, 0.95 - (i * 0.05)) if src == "camera_a" else min(0.95, 0.3 + (i * 0.05))

            gt_loc = {"room": "living_room", "x": 4.0, "y": 0.0, "z": 4.0}
            curr_time += timedelta(minutes=5)

            obs = ObservationInput(
                entity="laptop",
                location=Location(room="living_room", x=4.0, y=0.0, z=4.0),
                confidence=round(conf, 2),
                source=src,
                timestamp=curr_time.isoformat(),
            )
            steps.append(
                GroundTruthStep(
                    step_id=f"{scenario_id}_s{i+1}",
                    ground_truth_entity_id="laptop",
                    ground_truth_location=gt_loc,
                    ground_truth_movement_event=False,
                    ground_truth_target_action="REOBSERVE",
                    observation=obs,
                    observation_confidence=round(conf, 2),
                )
            )
        return SimulationScenario(scenario_id=scenario_id, noise_condition="family_d_dynamic_src", steps=steps)

    def _generate_family_e_stale_bursts(self, scenario_id: str, num_steps: int, rng: random.Random) -> SimulationScenario:
        steps: List[GroundTruthStep] = []
        curr_time = datetime(2026, 7, 29, 10, 0, 0, tzinfo=timezone.utc)

        for i in range(num_steps):
            moved = (i == 3)
            true_room = "kitchen" if i >= 3 else "bedroom"
            gt_loc = {"room": true_room, "x": 1.0, "y": 0.0, "z": 1.0}

            if i in (4, 5, 6):
                obs_room = "bedroom"
                conf = 0.85
            else:
                obs_room = true_room
                conf = 0.92

            curr_time += timedelta(minutes=5)
            obs = ObservationInput(
                entity="backpack",
                location=Location(room=obs_room, x=1.0, y=0.0, z=1.0),
                confidence=conf,
                source="sensor",
                timestamp=curr_time.isoformat(),
            )
            steps.append(
                GroundTruthStep(
                    step_id=f"{scenario_id}_s{i+1}",
                    ground_truth_entity_id="backpack",
                    ground_truth_location=gt_loc,
                    ground_truth_movement_event=moved,
                    ground_truth_target_action="UPDATE" if moved else "PRESERVE" if i in (4, 5, 6) else "REOBSERVE",
                    observation=obs,
                    observation_confidence=conf,
                )
            )
        return SimulationScenario(scenario_id=scenario_id, noise_condition="family_e_stale_burst", steps=steps)

    def _generate_family_f_partial_obs(self, scenario_id: str, num_steps: int, rng: random.Random) -> SimulationScenario:
        steps: List[GroundTruthStep] = []
        curr_time = datetime(2026, 7, 29, 10, 0, 0, tzinfo=timezone.utc)

        for i in range(num_steps):
            curr_time += timedelta(minutes=5)
            gt_loc = {"room": "office", "x": 2.5, "y": 0.0, "z": 3.5}

            obs_loc = Location(room="office", x=None, y=None, z=None) if (i % 2 == 0) else Location(room="office", x=2.5, y=0.0, z=3.5)
            obs = ObservationInput(
                entity="mug",
                attributes={} if (i % 2 == 1) else {"color": "white"},
                location=obs_loc,
                confidence=0.88,
                source="sensor",
                timestamp=curr_time.isoformat(),
            )
            steps.append(
                GroundTruthStep(
                    step_id=f"{scenario_id}_s{i+1}",
                    ground_truth_entity_id="mug",
                    ground_truth_location=gt_loc,
                    ground_truth_movement_event=False,
                    ground_truth_target_action="REOBSERVE",
                    observation=obs,
                    observation_confidence=0.88,
                )
            )
        return SimulationScenario(scenario_id=scenario_id, noise_condition="family_f_partial", steps=steps)

    def _generate_family_g_ambiguous_identity(self, scenario_id: str, num_steps: int, rng: random.Random) -> SimulationScenario:
        steps: List[GroundTruthStep] = []
        curr_time = datetime(2026, 7, 29, 10, 0, 0, tzinfo=timezone.utc)

        for i in range(num_steps):
            curr_time += timedelta(minutes=5)
            gt_loc = {"room": "hallway", "x": 5.0, "y": 0.0, "z": 5.0}
            obs = ObservationInput(
                entity="backpack",
                attributes={"color": "dark"},
                location=Location(room="hallway", x=5.0, y=0.0, z=5.0),
                confidence=0.75,
                source="user",
                timestamp=curr_time.isoformat(),
            )
            steps.append(
                GroundTruthStep(
                    step_id=f"{scenario_id}_s{i+1}",
                    ground_truth_entity_id="backpack",
                    ground_truth_location=gt_loc,
                    ground_truth_movement_event=False,
                    ground_truth_target_action="REOBSERVE",
                    observation=obs,
                    observation_confidence=0.75,
                )
            )
        return SimulationScenario(scenario_id=scenario_id, noise_condition="family_g_ambiguous", steps=steps)

    def _generate_family_h_adversarial(self, scenario_id: str, num_steps: int, rng: random.Random) -> SimulationScenario:
        steps: List[GroundTruthStep] = []
        curr_time = datetime(2026, 7, 29, 10, 0, 0, tzinfo=timezone.utc)
        curr_room = "bedroom"

        for i in range(num_steps):
            curr_time += timedelta(minutes=i * 3)
            is_false = rng.random() < 0.25
            moved = (rng.random() < 0.20 and not is_false)

            if moved:
                curr_room = rng.choice([r for r in self.ROOMS if r != curr_room])

            gt_loc = {"room": curr_room, "x": 1.0, "y": 0.0, "z": 1.0}
            obs_room = rng.choice(self.ROOMS) if is_false else curr_room
            conf = round(rng.uniform(0.5, 0.95), 2)

            obs = ObservationInput(
                entity="keys",
                location=Location(room=obs_room, x=1.0, y=0.0, z=1.0),
                confidence=conf,
                source="sensor",
                timestamp=curr_time.isoformat(),
            )
            steps.append(
                GroundTruthStep(
                    step_id=f"{scenario_id}_s{i+1}",
                    ground_truth_entity_id="keys",
                    ground_truth_location=gt_loc,
                    ground_truth_movement_event=moved,
                    ground_truth_target_action="UPDATE" if moved else "PRESERVE" if is_false else "REOBSERVE",
                    observation=obs,
                    observation_confidence=conf,
                    is_false_detection=is_false,
                )
            )
        return SimulationScenario(scenario_id=scenario_id, noise_condition="family_h_adversarial", steps=steps)

    def _generate_standard_noise_scenario(self, scenario_id: str, num_steps: int, noise_condition: str, rng: random.Random) -> SimulationScenario:
        target_entity = rng.choice(self.ENTITIES)
        curr_room = rng.choice(self.ROOMS)
        curr_time = datetime(2026, 7, 29, 12, 0, 0, tzinfo=timezone.utc)
        steps: List[GroundTruthStep] = []

        for i in range(num_steps):
            moved = (rng.random() < 0.20 and i > 0)
            if moved:
                curr_room = rng.choice([r for r in self.ROOMS if r != curr_room])

            gt_loc = {"room": curr_room, "x": round(rng.uniform(0, 10), 1), "y": 0.0, "z": round(rng.uniform(0, 10), 1)}
            is_false = (noise_condition in ("false_detection", "mixed", "conflicting") and rng.random() < 0.25)

            obs_room = rng.choice([r for r in self.ROOMS if r != curr_room]) if is_false else curr_room
            conf = round(rng.uniform(0.8, 0.98) if not is_false else rng.uniform(0.6, 0.9), 2)

            curr_time += timedelta(minutes=5)
            obs = ObservationInput(
                entity=target_entity,
                location=Location(room=obs_room, x=gt_loc["x"], y=gt_loc["y"], z=gt_loc["z"]),
                confidence=conf,
                source="camera",
                timestamp=curr_time.isoformat(),
            )
            steps.append(
                GroundTruthStep(
                    step_id=f"{scenario_id}_s{i+1}",
                    ground_truth_entity_id=target_entity,
                    ground_truth_location=gt_loc,
                    ground_truth_movement_event=moved,
                    ground_truth_target_action="UPDATE" if moved else "PRESERVE" if is_false else "REOBSERVE",
                    observation=obs,
                    observation_confidence=conf,
                    is_false_detection=is_false,
                )
            )
        return SimulationScenario(scenario_id=scenario_id, noise_condition=noise_condition, steps=steps)
