"""
node.py
-------
Defines the base Node class used in both P2P and Federated Learning systems.

Each node owns:
  • a local dataset slice (X_train, y_train) and validation set
  • a local LogisticModel
  • metadata for identifying honest vs malicious nodes

Subclasses in attacks.py override `get_update()` to inject adversarial behaviour.
"""

import numpy as np
from model import LogisticModel, N_FEATURES


class Node:
    """
    A single participant node in the network simulation.

    Parameters
    ----------
    node_id      : unique integer identifier
    X_train      : local training features  (n_local_samples, n_features)
    y_train      : local training labels
    X_val        : shared validation features (used for evaluation only)
    y_val        : shared validation labels
    random_state : seed for reproducibility; offset by node_id internally
    """

    def __init__(
        self,
        node_id     : int,
        X_train     : np.ndarray,
        y_train     : np.ndarray,
        X_val       : np.ndarray,
        y_val       : np.ndarray,
        random_state: int = 42,
    ):
        self.node_id      = node_id
        self.X_train      = X_train
        self.y_train      = y_train
        self.X_val        = X_val
        self.y_val        = y_val
        self.is_malicious = False

        # Each node gets its own seeded model to avoid correlated initialisations
        self.model = LogisticModel(
            n_features   = X_train.shape[1],
            random_state = random_state + node_id,
        )

        # weight_multiplier > 1 simulates Sybil amplification:
        # the aggregator counts this node's update as if it came from
        # `weight_multiplier` separate peers.
        self.weight_multiplier: int = 1

    # ------------------------------------------------------------------
    # Core interface (overridden by attack subclasses)
    # ------------------------------------------------------------------

    def train(self, epochs: int = 3) -> None:
        """Run one round of local SGD training."""
        self.model.train(self.X_train, self.y_train, epochs=epochs)

    def get_update(self) -> np.ndarray:
        """
        Return the node's current model weights to share with peers / server.
        Honest nodes return genuine weights; attackers override this method.
        """
        return self.model.get_weights().copy()

    def apply_update(self, weights: np.ndarray) -> None:
        """Overwrite local model with aggregated weights from the network."""
        self.model.set_weights(weights)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> tuple[float, float]:
        """Return (accuracy, log_loss) on the shared validation set."""
        acc  = self.model.score(self.X_val, self.y_val)
        loss = self.model.loss(self.X_val,  self.y_val)
        return acc, loss

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def n_samples(self) -> int:
        return len(self.X_train)

    def __repr__(self) -> str:
        kind = "Malicious" if self.is_malicious else "Honest"
        return f"Node(id={self.node_id}, {kind}, n={self.n_samples()})"
