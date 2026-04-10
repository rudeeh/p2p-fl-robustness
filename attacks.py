"""
attacks.py
----------
Adversarial node subclasses.  Each overrides `get_update()` so that the rest
of the simulation (P2P / FL aggregation) needs no special-casing.

Attack taxonomy
---------------
PoisoningAttacker : Negates (flips) its legitimate weights → pulls the
                    aggregate in the opposite direction.
SybilAttacker     : Like PoisoningAttacker but sets weight_multiplier > 1,
                    causing the aggregator to count its vote multiple times,
                    simulating fake peer identities.
NoiseAttacker     : Replaces weights entirely with scaled Gaussian noise,
                    simulating a Byzantine-random adversary.

All attackers use a deterministic internal RNG seeded at construction time
so that experiments are fully reproducible.
"""

import numpy as np
from node import Node


class PoisoningAttacker(Node):
    """
    Gradient-poisoning adversary.

    Sends the *negation* of its legitimate update, pushing the global model
    away from a good solution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_malicious = True

    def get_update(self) -> np.ndarray:
        legitimate = self.model.get_weights()
        return -legitimate.copy()          # flip sign → poisoned gradient


class SybilAttacker(Node):
    """
    Sybil adversary.

    Behaves like PoisoningAttacker but claims to represent `sybil_count`
    independent peers.  The aggregator honours this via `weight_multiplier`,
    amplifying the attacker's influence proportionally.

    Parameters
    ----------
    sybil_count : int
        Number of fake identities (≥ 1).  A value of 3 means this node's
        poisoned update is counted three times during aggregation.
    """

    def __init__(self, *args, sybil_count: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_malicious      = True
        self.weight_multiplier = max(1, sybil_count)

    def get_update(self) -> np.ndarray:
        legitimate = self.model.get_weights()
        return -legitimate.copy()          # same poisoning strategy


class NoiseAttacker(Node):
    """
    Random-noise (Byzantine) adversary.

    Replaces its model update with zero-mean Gaussian noise scaled by
    `noise_scale`.  The noise seed is fixed so results are reproducible,
    but the pattern is unrelated to any real gradient.

    Parameters
    ----------
    noise_scale : float
        Standard deviation of the injected noise.  Default 5.0 is chosen to
        be substantially larger than typical legitimate weight magnitudes
        (which are clipped to ±10) to guarantee measurable degradation.
    """

    def __init__(self, *args, noise_scale: float = 5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_malicious = True
        self.noise_scale  = noise_scale
        # Deterministic RNG – seeded once, advanced each call for variety
        self._rng = np.random.RandomState(seed=42 + self.node_id)

    def get_update(self) -> np.ndarray:
        weight_dim = self.model.weight_dim()
        return self._rng.normal(0.0, self.noise_scale, size=weight_dim)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_attacker(
    attack_type : str,
    node_id     : int,
    X_train     : np.ndarray,
    y_train     : np.ndarray,
    X_val       : np.ndarray,
    y_val       : np.ndarray,
    random_state: int = 42,
    **kwargs,
) -> Node:
    """
    Instantiate the correct attacker subclass by name.

    Parameters
    ----------
    attack_type : 'poisoning' | 'sybil' | 'noise'
    **kwargs    : forwarded to the attacker constructor (e.g. sybil_count)
    """
    registry = {
        "poisoning": PoisoningAttacker,
        "sybil"    : SybilAttacker,
        "noise"    : NoiseAttacker,
    }
    cls = registry.get(attack_type.lower())
    if cls is None:
        raise ValueError(
            f"Unknown attack type '{attack_type}'. "
            f"Choose from: {list(registry)}"
        )
    return cls(
        node_id      = node_id,
        X_train      = X_train,
        y_train      = y_train,
        X_val        = X_val,
        y_val        = y_val,
        random_state = random_state,
        **kwargs,
    )
