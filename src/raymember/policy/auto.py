"""Unified Automatic Memory Policy combining direct updates, Bayesian fusion, and Hybrid trust weighting."""

from typing import Any, Dict, Optional, Tuple
from raymember.baselines.baselines import BaseMemoryPolicy, ProbabilisticEngineBaseline
from raymember.learning.policy import HybridPolicy
from raymember.policy.routing import PolicyRouter, RoutingDecision
from raymember.schemas import Location, ObservationInput


class AutoMemoryPolicy(BaseMemoryPolicy):
    """
    Automatic memory update policy. Automatically routes observations
    to direct update, probabilistic belief fusion, or hybrid trust weighting based on evidence quality.
    Exposes human-readable update explanations.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.probabilistic_engine = ProbabilisticEngineBaseline()
        self.hybrid_policy = HybridPolicy(random_seed=random_seed)
        self.last_routing_decision: Optional[RoutingDecision] = None
        self.last_explanation: str = "No observations processed yet."

    def predict_action_with_explanation(
        self,
        current_state: Optional[Dict[str, Any]],
        new_obs: ObservationInput,
    ) -> Tuple[str, Dict[str, Any], str]:
        decision = PolicyRouter.route_observation(current_state, new_obs)
        self.last_routing_decision = decision
        self.last_explanation = decision.reason

        new_loc = new_obs.location.to_dict() if isinstance(new_obs.location, Location) else new_obs.location
        if current_state is None:
            return "INITIALIZE", new_loc, decision.reason

        curr_loc = current_state.get("location", new_loc)

        if decision.selected_mode == "preserve":
            return "PRESERVE", curr_loc, decision.reason

        if decision.selected_mode == "direct":
            action = "UPDATE" if new_loc.get("room") != curr_loc.get("room") else "REOBSERVE"
            return action, new_loc, decision.reason

        if decision.selected_mode == "hybrid":
            act, loc = self.hybrid_policy.predict_action(current_state, new_obs)
            exp = f"{decision.reason} Resulting belief estimated in {loc.get('room')}."
            return act, loc, exp

        act, loc = self.probabilistic_engine.predict_action(current_state, new_obs)
        return act, loc, decision.reason

    def predict_action(
        self,
        current_state: Optional[Dict[str, Any]],
        new_obs: ObservationInput,
    ) -> Tuple[str, Dict[str, Any]]:
        act, loc, _ = self.predict_action_with_explanation(current_state, new_obs)
        return act, loc
