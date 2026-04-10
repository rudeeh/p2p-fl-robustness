"""
federated.py
------------
Federated Learning (FL) simulation with a central aggregator.

Protocol (one round)
--------------------
1. Each node trains locally on its private data.
2. Every node sends its current model weights to the central server.
3. The server aggregates the received updates:
     default  – weighted average (weight ∝ local dataset size), per FedAvg.
     optional – coordinate-wise median for robust aggregation (bonus feature).
4. The server broadcasts the global model back to all nodes.

Failure-mode mitigations
------------------------
• Shape mismatch  – weights that don't match the expected dimension are
                    silently dropped before aggregation.
• NaN / Inf       – invalid weight vectors are discarded; if ALL updates are
                    invalid the global model is kept unchanged.
• Division by zero – guarded by checking total sample count before dividing.
• Gradient explosion – handled upstream in LogisticModel.set_weights (clipping).
• Sybil amplification – honoured via node.weight_multiplier; the server counts
                        a Sybil node's update as if it came from multiple peers.

Reference
---------
McMahan et al. (2017) – "Communication-Efficient Learning of Deep Networks
from Decentralized Data" (FedAvg algorithm).
"""

import numpy as np
from typing import List, Literal, Optional

from node import Node

AggStrategy = Literal["mean", "median"]


class FederatedSystem:
    """
    Parameters
    ----------
    nodes        : list of Node (or attacker subclass) instances
    aggregation  : 'mean' | 'median'
    random_state : kept for API consistency
    """

    def __init__(
        self,
        nodes       : List[Node],
        aggregation : AggStrategy = "mean",
        random_state: int = 42,
    ):
        self.nodes          = nodes
        self.aggregation    = aggregation
        self.n_nodes        = len(nodes)
        self.global_weights : Optional[np.ndarray] = None

        # Infer expected weight dimension from the first node's model
        self._weight_dim = nodes[0].model.weight_dim()

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def run_round(self, epochs: int = 3) -> None:
        """
        Execute one FL communication round.

        Steps
        -----
        1. Local training on every node.
        2. Collect and validate weight updates.
        3. Aggregate (server-side).
        4. Broadcast global model.
        """
        # --- Step 1: local training ---
        for node in self.nodes:
            node.train(epochs=epochs)

        # --- Step 2: collect updates ---
        valid_updates : List[np.ndarray] = []
        valid_counts  : List[int]        = []

        for node in self.nodes:
            w = node.get_update()

            # Shape validation
            if w.shape[0] != self._weight_dim:
                continue  # discard mismatched update

            # NaN / Inf validation
            if not np.all(np.isfinite(w)):
                continue  # discard corrupted update

            # Sybil amplification: count the update weight_multiplier times
            for _ in range(node.weight_multiplier):
                valid_updates.append(w)
                valid_counts.append(node.n_samples())

        if not valid_updates:
            # No valid updates this round – retain the previous global model
            return

        # --- Step 3: server-side aggregation ---
        self.global_weights = self._aggregate(valid_updates, valid_counts)

        # --- Step 4: broadcast ---
        for node in self.nodes:
            node.apply_update(self.global_weights.copy())

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        updates: List[np.ndarray],
        counts : List[int],
    ) -> np.ndarray:
        """
        Aggregate weight vectors.

        'mean'   – FedAvg weighted average (weight ∝ dataset size).
        'median' – coordinate-wise median (robust aggregation bonus).
        """
        stacked = np.stack(updates, axis=0)   # shape (n_valid, weight_dim)

        if self.aggregation == "median":
            return np.median(stacked, axis=0)

        # Weighted mean (FedAvg)
        total = sum(counts)
        if total == 0:
            return np.mean(stacked, axis=0)

        weights = np.array(counts, dtype=float) / total
        return np.einsum("i,ij->j", weights, stacked)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> tuple[float, float]:
        """
        Return mean (accuracy, loss) over *honest* nodes only.
        """
        accs, losses = [], []
        for node in self.nodes:
            if not node.is_malicious:
                acc, loss = node.evaluate()
                accs.append(acc)
                losses.append(loss)

        if not accs:
            return 0.0, 1.0

        return float(np.mean(accs)), float(np.mean(losses))
