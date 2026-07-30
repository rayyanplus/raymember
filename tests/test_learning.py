"""Unit tests for Layer 3 Feature Engineering, dataset splitting, and Scikit-Learn policy."""

import pytest
import numpy as np

from raymember.learning.dataset import DatasetGenerator
from raymember.learning.features import FeatureExtractor
from raymember.learning.policy import LearnedUpdatePolicy
from raymember.schemas import Location, ObservationInput


def test_feature_extractor():
    extractor = FeatureExtractor()
    obs = ObservationInput(entity="backpack", location=Location(room="bedroom", x=1.0, y=2.0, z=0.0), confidence=0.9, source="camera")
    feat = extractor.extract(current_state=None, new_obs=obs)

    assert isinstance(feat, np.ndarray)
    assert feat.shape == (9,)
    assert feat[0] == 0.9
    assert feat[1] == 0.95


def test_dataset_generator_no_data_leakage():
    ds = DatasetGenerator(random_seed=42)
    (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, _), scenarios = ds.generate_split_dataset(
        num_scenarios=20, steps_per_scenario=10, noise_condition="mixed"
    )

    assert len(X_tr) > 0
    assert len(X_va) > 0
    assert len(X_te) > 0

    total_steps = sum(len(sc.steps) for sc in scenarios)
    assert len(X_tr) + len(X_va) + len(X_te) == total_steps


def test_learned_policy_training_and_prediction():
    ds = DatasetGenerator(random_seed=42)
    (X_tr, y_tr, _), _, _, _ = ds.generate_split_dataset(num_scenarios=10, steps_per_scenario=5)

    policy = LearnedUpdatePolicy(model_type="random_forest", random_seed=42)
    policy.train(X_tr, y_tr)
    assert policy.is_trained is True

    obs = ObservationInput(entity="backpack", location=Location(room="living_room"), confidence=0.92)
    action, loc = policy.predict_action(current_state=None, new_obs=obs)
    assert action in ("INITIALIZE", "UPDATE", "REOBSERVE", "PRESERVE", "UNCERTAIN", "NEW_ENTITY")
