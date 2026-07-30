"""Uncertainty quantification and entropy calculations."""

import math
from typing import List
from raymember.belief.state import LocationBeliefItem


def calculate_entropy(location_beliefs: List[LocationBeliefItem]) -> float:
    """
    Calculates normalized Shannon entropy of belief distribution:
    H = - sum(p * log2(p)) / log2(N)
    Returns float in range [0.0, 1.0].
    """
    if not location_beliefs or len(location_beliefs) <= 1:
        return 0.0

    probs = [item.probability for item in location_beliefs if item.probability > 0]
    if not probs:
        return 0.0

    total = sum(probs)
    norm_probs = [p / total for p in probs]

    h = -sum(p * math.log2(p) for p in norm_probs)
    max_h = math.log2(len(norm_probs)) if len(norm_probs) > 1 else 1.0

    return float(max(0.0, min(1.0, h / max_h)))


def calculate_belief_confidence(top_prob: float, entropy: float) -> float:
    """
    Calculates overall belief confidence combining top candidate probability
    and distribution entropy.
    """
    return float(max(0.0, min(1.0, top_prob * (1.0 - 0.5 * entropy))))
