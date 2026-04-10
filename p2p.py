"""
p2p.py
------
Peer-to-Peer (P2P) simulation system.

Architecture
------------
• Every node is both a client and a server.
• Each round:
    1. All nodes train locally on their private data.
    2. All nodes share their current model weights with their neighbours
       (determined by the chosen topology).
    3. Each node aggregates the received weights (including its own) via a
       configurable aggregation function and updates its local model.

Topologies supported
--------------------
'fully_connected' – every node communicates with every other node.
'ring'            – each node communicates only with its two immediate
                    neighbours (good for testing convergence under limited
                    connectivity).

Aggregation strategies
----------------------
'mean'   – simple arithmetic mean (default, vulnerable to poisoning).
'median' – coordinate-wise median; robust aggregation bonus feature.

Failure-mode mitigations
------------------------
• NaN / Inf weights from malicious nodes are detected and discarded before
  aggregation.  If ALL updates are invalid the node keeps its current weights.
• Sybil amplification is handled by repeating a node's update according to
  its weight_multiplier attribute.
"""

import numpy as np
from typing import List, Literal

from node import Node


AggStrategy = Literal["mean", "median"]


class P2PSystem:
    """
    Parameters
    ----------
    nodes        : list of Node (or attacker subclass) instances
    topology     : 'fully_connected' | 'ring'
    aggregation  : 'mean' | 'median'
    random_state : unused directly but kept for API consistency
    """

    def __init__(
        self,
        nodes       : List[Node],
        topology    : str = "fully_connected",
        aggregation : AggStrategy = "mean",
        random_state: int = 42,
    ):
        self.nodes       = nodes
        self.topology    = topology
        self.aggregation = aggregation
        self.n_nodes     = len(nodes)
        self.neighbors   = self._build_topology()

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def _build_topology(self) -> dict[int, List[int]]:
        """Return adjacency list keyed by node index."""
        n = self.n_nodes
        if self.topology == "fully_connected":
            return {i: [j for j in range(n) if j != i] for i in range(n)}
        elif self.topology == "ring":
            return {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}
        else:
            raise ValueError(
                f"Unknown topology '{self.topology}'. "
                "Choose 'fully_connected' or 'ring'."
            )

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def run_round(self, epochs: int = 3) -> None:
        """
        Execute one full P2P gossip round.

        Steps
        -----
        1. Each node trains locally.
        2. Collect updates (respecting Sybil weight_multiplier).
        3. Each node aggregates its own update + neighbours' updates.
        4. Each node applies the aggregated weights.
        """
        # Step 1 – local training
        for node in self.nodes:
            node.train(epochs=epochs)

        # Step 2 – snapshot all updates BEFORE any node applies changes
        #          (avoids order-of-update inconsistencies)
        raw_updates: dict[int, np.ndarray] = {}
        for i, node in enumerate(self.nodes):
            raw_updates[i] = node.get_update()

        # Expand updates for Sybil nodes (weight_multiplier > 1)
        expanded: dict[int, List[np.ndarray]] = {
            i: [raw_updates[i]] * node.weight_multiplier
            for i, node in enumerate(self.nodes)
        }

        # Step 3 & 4 – aggregate and apply
        for i, node in enumerate(self.nodes):
            pool: List[np.ndarray] = []
            # Own update(s)
            pool.extend(expanded[i])
            # Neighbours' updates
            for j in self.neighbors[i]:
                pool.extend(expanded[j])

            agg = self._aggregate(pool)
            node.apply_update(agg)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, weight_list: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate a list of weight vectors.

        • Discards any vector containing NaN or Inf.
        • Falls back to the first element if everything is invalid.
        """
        valid = [w for w in weight_list if np.all(np.isfinite(w))]
        if not valid:
            # Last resort: return whatever the first entry was (may be clipped
            # downstream by set_weights)
            return weight_list[0] if weight_list else np.zeros(self.nodes[0].model.weight_dim())

        stacked = np.stack(valid, axis=0)

        if self.aggregation == "mean":
            return np.mean(stacked, axis=0)
        elif self.aggregation == "median":
            return np.median(stacked, axis=0)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> tuple[float, float]:
        """
        Return mean (accuracy, loss) computed only over *honest* nodes.
        Malicious nodes are excluded so we measure the real impact on legitimate
        participants.
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
