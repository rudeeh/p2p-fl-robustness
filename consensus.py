"""
consensus.py
-------------
ConsensusDivergenceProbe: measures the degree to which honest nodes
learn *different* models under different protocols, and quantifies the
systemic alignment risk this creates.

Why this matters for AI alignment
----------------------------------
FL guarantees all nodes converge to the same model.  P2P — especially ring
— does not.  This is usually treated as a defect ("P2P converges slower").
But from an alignment perspective, it's a structural property with both
risks and benefits:

  - High disagreement means the 'system' doesn't have a single behaviour
    to evaluate.  Safety analysis assumes one model to probe.  If honest
    nodes have different models, you need to evaluate each — and the worst
    one is what matters for alignment.
  - Under attack, disagreement can indicate *fragmentation*: the attack
    split the honest population into factions, each internally consistent
    but mutually incompatible.  This is a protocol-level alignment failure.
  - Paradoxically, moderate disagreement can be alignment-*positive*: it
    means the system hasn't converged to a single point of failure.

Metrics
-------
  - weight_dispersion: mean pairwise L2 distance among honest nodes
  - prediction_disagreement: fraction of val set where honest nodes disagree
  - model_clusters: number of distinct model 'factions' (k=2 clustering)
  - worst_pair_distance: max pairwise distance (fragmentation risk)
  - convergence_velocity: rate of dispersion change (positive = converging)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConsensusRecord:
    """Immutable snapshot of honest-node consensus for one round."""
    round_num               : int
    weight_dispersion       : float
    prediction_disagreement : float
    model_clusters          : int
    worst_pair_distance     : float
    convergence_velocity    : float


@dataclass
class ConsensusReport:
    """Complete consensus analysis for one experiment."""
    system_name : str
    attack_type : str
    attack_pct  : float
    records     : List[ConsensusRecord] = field(default_factory=list)

    def peak_disagreement(self) -> float:
        return max((r.prediction_disagreement for r in self.records), default=0.0)

    def peak_dispersion(self) -> float:
        return max((r.weight_dispersion for r in self.records), default=0.0)

    def fragmentation_round(self) -> Optional[int]:
        """First round where model_clusters > 1."""
        for r in self.records:
            if r.model_clusters > 1:
                return r.round_num
        return None

    def final_dispersion(self) -> float:
        return self.records[-1].weight_dispersion if self.records else 0.0

    def to_dict(self) -> dict:
        return {
            "system_name": self.system_name,
            "attack_type": self.attack_type,
            "attack_pct": self.attack_pct,
            "peak_disagreement": self.peak_disagreement(),
            "peak_dispersion": self.peak_dispersion(),
            "fragmentation_round": self.fragmentation_round(),
            "final_dispersion": self.final_dispersion(),
            "records": [
                {
                    "round": r.round_num,
                    "weight_dispersion": r.weight_dispersion,
                    "prediction_disagreement": r.prediction_disagreement,
                    "model_clusters": r.model_clusters,
                    "worst_pair_distance": r.worst_pair_distance,
                    "convergence_velocity": r.convergence_velocity,
                }
                for r in self.records
            ],
        }


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

class ConsensusDivergenceProbe:
    """
    Measures how much honest nodes disagree with each other after each round.

    Parameters
    ----------
    n_features : int
        Dimensionality of the weight vector (used for logistic reconstruction).
    """

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.records: List[ConsensusRecord] = []
        self._prev_dispersion: Optional[float] = None

    def observe_round(
        self,
        round_num: int,
        honest_weights: List[np.ndarray],
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> ConsensusRecord:
        """
        Compute consensus metrics from all honest nodes' current weights.

        Parameters
        ----------
        honest_weights : list of weight vectors, one per honest node
        X_val, y_val  : shared validation set
        """
        n = len(honest_weights)
        if n < 2:
            record = ConsensusRecord(
                round_num=round_num, weight_dispersion=0.0,
                prediction_disagreement=0.0, model_clusters=1,
                worst_pair_distance=0.0, convergence_velocity=0.0,
            )
            self.records.append(record)
            return record

        stacked = np.stack(honest_weights)  # (n_nodes, weight_dim)

        # --- Mean pairwise L2 distance ---
        # Efficient: use broadcasting
        diff = stacked[:, None, :] - stacked[None, :, :]  # (n, n, d)
        pairwise = np.linalg.norm(diff, axis=2)            # (n, n)
        triu_idx = np.triu_indices(n, k=1)
        pairwise_vals = pairwise[triu_idx]

        weight_dispersion = float(np.mean(pairwise_vals))
        worst_pair_distance = float(np.max(pairwise_vals))

        # --- Convergence velocity ---
        convergence_velocity = 0.0
        if self._prev_dispersion is not None:
            convergence_velocity = self._prev_dispersion - weight_dispersion
        self._prev_dispersion = weight_dispersion

        # --- Prediction disagreement ---
        # Reconstruct predictions from weights directly (cheap for logistic)
        coefs = stacked[:, :self.n_features]      # (n, n_features)
        intercepts = stacked[:, self.n_features]  # (n,)

        logits = X_val @ coefs.T + intercepts[None, :]  # (n_val, n_nodes)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        predictions = (probs >= 0.5).astype(int)            # (n_val, n_nodes)

        # Fraction of samples where NOT all honest nodes agree
        all_same = np.all(predictions == predictions[:, :1], axis=1)
        prediction_disagreement = 1.0 - float(np.mean(all_same))

        # --- Model clustering (k=2) ---
        model_clusters = 1
        if n >= 4:
            import warnings
            from sklearn.cluster import KMeans
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km = KMeans(n_clusters=2, n_init=1, random_state=42, max_iter=10)
                labels = km.fit_predict(stacked)
                model_clusters = len(set(labels))

        record = ConsensusRecord(
            round_num=round_num,
            weight_dispersion=weight_dispersion,
            prediction_disagreement=prediction_disagreement,
            model_clusters=model_clusters,
            worst_pair_distance=worst_pair_distance,
            convergence_velocity=convergence_velocity,
        )
        self.records.append(record)
        return record

    def get_report(
        self,
        system_name: str,
        attack_type: str,
        attack_pct: float,
    ) -> ConsensusReport:
        """Return the full consensus report and reset."""
        report = ConsensusReport(
            system_name=system_name,
            attack_type=attack_type,
            attack_pct=attack_pct,
            records=list(self.records),
        )
        self.records = []
        self._prev_dispersion = None
        return report

    def reset(self) -> None:
        self.records = []
        self._prev_dispersion = None
