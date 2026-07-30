"""Unit tests for synthetic world simulator and hidden ground truth generation."""

import pytest
from raymember.simulation.world import SimulationWorld


def test_simulation_world_generation():
    sim = SimulationWorld(random_seed=42)
    scenario = sim.generate_scenario("test_sc_01", num_steps=10, noise_condition="clean")

    assert scenario.scenario_id == "test_sc_01"
    assert scenario.noise_condition == "clean"
    assert len(scenario.steps) == 10

    for step in scenario.steps:
        assert step.ground_truth_entity_id is not None
        assert "room" in step.ground_truth_location
        assert step.ground_truth_target_action in ("INITIALIZE", "UPDATE", "REOBSERVE", "PRESERVE", "UNCERTAIN")


def test_simulation_noise_conditions():
    sim = SimulationWorld(random_seed=123)
    conditions = ["clean", "missing", "false_detection", "delayed", "out_of_order", "conflicting", "mixed"]

    for cond in conditions:
        sc = sim.generate_scenario(f"sc_{cond}", num_steps=15, noise_condition=cond)
        assert len(sc.steps) > 0
