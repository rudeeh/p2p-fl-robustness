"""
influence.py
-------------
InfluenceTracker: measures which nodes actually shaped the final model
at each round, and quantifies how the protocol structure distributes
influence.

Why this matters for AI alignment
----------------------------------
The protocol's aggregation rule is a hidden governance mechanism.
FedAvg's weighted averaging doesn't just aggregate — it defines a specific
Pareto optimum among nodes' local objectives.  P2P gossip creates a
propagation dynamics that determines which nodes' data gets represented
in the final model and which doesn't.

If attackers gain disproportionate influence, the protocol has failed as
a governance system — not just a robustness system.  The model's behaviour
on minority data slices is then a direct consequence of protocol design.

Influence is measured as the cosine similarity between each node's update
direction and the actual model change direction, normalised to sum to 1.0.
This captures that the protocol rewards *alignment with consensus*, not just
magnitude.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InfluenceRecord:
    """Immutable snapshot of per-node influence for one round."""
    round_num            : int
    per_node_influence   : np.ndarray   # shape (n_nodes,), non-negative, sums ~1.0
    influence_gini       : float        # 0 = equal, 1 = one node controls everything
    honest_influence     : float        # fraction held by honest nodes
    max_single_influence : float        # largest single-node share


@dataclass
class InfluenceReport:
    """Complete influence analysis for one experiment."""
    system_name : str
    attack_type : str
    attack_pct  : float
    records     : List[InfluenceRecord] = field(default_factory=list)

    def final_honest_influence(self) -> float:
        if not self.records:
            return 1.0
        return self.records[-1].honest_influence

    def influence_gini_history(self) -> List[float]:
        return [r.influence_gini for r in self.records]

    def cumulative_influence(self) -> np.ndarray:
        """Per-node total influence across all rounds. Shape (n_nodes,)."""
        if not self.records:
            return np.array([])
        return np.sum([r.per_node_influence for r in self.records], axis=0)

    def to_dict(self) -> dict:
        return {
            "system_name": self.system_name,
            "attack_type": self.attack_type,
            "attack_pct": self.attack_pct,
            "final_honest_influence": self.final_honest_influence(),
            "mean_gini": float(np.mean(self.influence_gini_history())),
            "records": [
                {
                    "round": r.round_num,
                    "influence_gini": r.influence_gini,
                    "honest_influence": r.honest_influence,
                    "max_single_influence": r.max_single_influence,
                }
                for r in self.records
            ],
        }


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class InfluenceTracker:
    """
    Observes weight updates and their aggregation to compute per-node
    influence at each round.

    Parameters
    ----------
    n_nodes : int
    """

    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.records: List[InfluenceRecord] = []
        self._prev_weights: Optional[np.ndarray] = None

    def observe_round(
        self,
        round_num: int,
        raw_updates: Dict[int, np.ndarray],
        aggregated_weights: np.ndarray,
        node_is_malicious: List[bool],
    ) -> InfluenceRecord:
        """
        Compute per-node influence for one round.

        Parameters
        ----------
        raw_updates       : what each node sent (pre-aggregation)
        aggregated_weights: what the system produced (post-aggregation)
        node_is_malicious : which nodes are attackers
        """
        # First round: no previous weights to compare against
        if self._prev_weights is None:
            self._prev_weights = aggregated_weights.copy()
            record = InfluenceRecord(
                round_num=round_num,
                per_node_influence=np.ones(self.n_nodes) / self.n_nodes,
                influence_gini=0.0,
                honest_influence=1.0,
                max_single_influence=1.0 / self.n_nodes,
            )
            self.records.append(record)
            return record

        # Actual model change this round
        delta = aggregated_weights - self._prev_weights
        delta_norm = np.linalg.norm(delta)

        if delta_norm < 1e-10:
            self._prev_weights = aggregated_weights.copy()
            record = InfluenceRecord(
                round_num=round_num,
                per_node_influence=np.ones(self.n_nodes) / self.n_nodes,
                influence_gini=0.0,
                honest_influence=1.0,
                max_single_influence=1.0 / self.n_nodes,
            )
            self.records.append(record)
            return record

        # Per-node influence = cosine similarity with actual delta
        influences = np.zeros(self.n_nodes)
        for i, w in raw_updates.items():
            if i >= self.n_nodes:
                continue
            node_delta = w - self._prev_weights
            node_norm = np.linalg.norm(node_delta)
            if node_norm > 1e-10:
                influences[i] = np.dot(node_delta, delta) / (node_norm * delta_norm)

        # Clip negatives to 0 and normalise
        influences = np.maximum(influences, 0.0)
        total = influences.sum()
        if total > 1e-10:
            influences /= total
        else:
            influences = np.ones(self.n_nodes) / self.n_nodes

        gini = self._gini(influences)

        honest_idx = [i for i in range(self.n_nodes) if i < len(node_is_malicious) and not node_is_malicious[i]]
        honest_inf = influences[honest_idx].sum() if honest_idx else 0.0

        self._prev_weights = aggregated_weights.copy()

        record = InfluenceRecord(
            round_num=round_num,
            per_node_influence=influences,
            influence_gini=gini,
            honest_influence=float(honest_inf),
            max_single_influence=float(influences.max()),
        )
        self.records.append(record)
        return record

    def get_report(
        self,
        system_name: str,
        attack_type: str,
        attack_pct: float,
    ) -> InfluenceReport:
        """Return the full influence report and reset for next experiment."""
        report = InfluenceReport(
            system_name=system_name,
            attack_type=attack_type,
            attack_pct=attack_pct,
            records=list(self.records),
        )
        self.records = []
        self._prev_weights = None
        return report

    def reset(self) -> None:
        self.records = []
        self._prev_weights = None

    @staticmethod
    def _gini(values: np.ndarray) -> float:
        """Gini coefficient: 0 = perfectly equal, 1 = one entity has everything."""
        sorted_v = np.sort(values)
        n = len(sorted_v)
        if n == 0 or sorted_v.sum() < 1e-10:
            return 0.0
        index = np.arange(1, n + 1)
        return float(
            (2 * np.sum(index * sorted_v) / (n * sorted_v.sum())) - (n + 1) / n
        )
