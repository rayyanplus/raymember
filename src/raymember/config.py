"""Configuration settings for Raymember."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class RaymemberConfig:
    """Global and per-instance configuration for WorldMemory."""

    database_path: str = "raymember.db"
    update_policy: Literal["learned", "rule_based", "latest_observation", "belief_engine"] = "learned"
    movement_distance_threshold: float = 0.5
    min_confidence_threshold: float = 0.1
    belief_decay_rate: float = 0.05  # time decay factor lambda for prior beliefs
    random_seed: int = 42
    model_path: str | None = None

    # Resolution settings
    enable_builtin_aliases: bool = True
    custom_location_aliases: dict | None = None
    fuzzy_accept_threshold: float = 0.88
    fuzzy_confirm_threshold: float = 0.65

