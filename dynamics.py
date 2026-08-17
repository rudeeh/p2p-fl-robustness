"""
dynamics.py
-----------
TrainingDynamicsMonitor: tracks how the training process evolves over
communication rounds, detecting when adversarial nodes cause the optimization
trajectory to diverge from a clean training path.

Why this matters for AI alignment
--------------------------------
In alignment research, a core concern is whether training produces a model that
optimises the intended objective versus a proxy.  In distributed training,
adversarial nodes can inject gradient noise that causes the optimizer to find
a *different* local minimum — one with similar loss but different behavioural
properties.  This is a literal instance of inner misalignment: training appears
to converge but converges to a qualitatively different model.

The monitor detects this by comparing the attacked training trajectory against
a clean (no-attack) trajectory and measuring:
  - Weight divergence at each round
  - Gradient alignment (cosine similarity of update directions)
  - Per-node update variance (spikes indicate coordinated attacks)
  - Cumulative drift rate (linear = benign variance, superlinear = active attack)
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
class DynamicsSnapshot:
    """Immutable snapshot of one round's training dynamics."""
    round_num                : int
    weight_l2_from_clean      : float      # ||w_attacked - w_clean||
    weight_cosine_from_clean  : float      # cos(w_attacked, w_clean)
    gradient_alignment        : float      # cos(delta_attacked, delta_clean)
    honest_update_variance    : float      # var(||update_i||) over honest nodes
    malicious_update_variance : float      # var(||update_i||) over malicious nodes
    loss_delta                : float      # attacked_loss - clean_loss


@dataclass
class DynamicsReport:
    """Complete dynamics analysis for one experiment."""
    system_name      : str
    attack_type      : str
    attack_pct       : float
    snapshots        : List[DynamicsSnapshot] = field(default_factory=list)

    # --- Derived summary metrics ---
    def final_divergence(self) -> float:
        """L2 distance between final attacked and clean weights."""
        return self.snapshots[-1].weight_l2_from_clean if self.snapshots else 0.0

    def mean_gradient_alignment(self) -> float:
        """Mean cosine similarity of update directions across rounds."""
        if not self.snapshots:
            return 0.0
        return float(np.mean([s.gradient_alignment for s in self.snapshots]))

    def divergence_onset_round(self, threshold: float = 2.0) -> Optional[int]:
        """First round where weight divergence exceeds `threshold` * median."""
        if len(self.snapshots) < 3:
            return None
        divergences = [s.weight_l2_from_clean for s in self.snapshots]
        median_div = float(np.median(divergences))
        for s in self.snapshots:
            if s.weight_l2_from_clean > threshold * median_div and median_div > 1e-10:
                return s.round_num
        return None

    def cumulative_drift_rate(self) -> float:
        """
        Ratio of final divergence to total accumulated step-by-step drift.
        Values >> 1 indicate superlinear drift (active attack).
        Values ~ 1 indicate linear, benign accumulation.
        """
        if len(self.snapshots) < 2:
            return 0.0
        divergences = [s.weight_l2_from_clean for s in self.snapshots]
        final = divergences[-1]
        # Sum of per-round increments
        increments = np.diff(divergences)
        total_accumulated = float(np.sum(np.abs(increments)))
        if total_accumulated < 1e-10:
            return 0.0
        return final / total_accumulated

    def to_dict(self) -> dict:
        return {
            "system_name": self.system_name,
            "attack_type": self.attack_type,
            "attack_pct": self.attack_pct,
            "final_divergence": self.final_divergence(),
            "mean_gradient_alignment": self.mean_gradient_alignment(),
            "divergence_onset": self.divergence_onset_round(),
            "cumulative_drift_rate": self.cumulative_drift_rate(),
            "snapshots": [
                {
                    "round": s.round_num,
                    "weight_l2": s.weight_l2_from_clean,
                    "weight_cosine": s.weight_cosine_from_clean,
                    "gradient_alignment": s.gradient_alignment,
                    "honest_var": s.honest_update_variance,
                    "malicious_var": s.malicious_update_variance,
                    "loss_delta": s.loss_delta,
                }
                for s in self.snapshots
            ],
        }


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class TrainingDynamicsMonitor:
    """
    Records per-round weight snapshots for both clean and attacked training
    and computes divergence / alignment metrics.

    Usage
    -----
    monitor = TrainingDynamicsMonitor(n_nodes)
    for rnd in range(1, n_rounds + 1):
        clean_system.run_round(epochs)
        attacked_system.run_round(epochs)
        monitor.record_round(
            round_num=rnd,
            clean_weights=clean_system.get_global_weights(),
            clean_loss=clean_system.evaluate()[1],
            attacked_weights=attacked_system.get_global_weights(),
            attacked_loss=attacked_system.evaluate()[1],
            clean_updates=clean_system.get_last_updates(),
            attacked_updates=attacked_system.get_last_updates(),
            nodes=attacked_system.nodes,
        )
    report = monitor.get_report(system_name, attack_type, attack_pct)
    """

    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self._clean_history: List[np.ndarray] = []
        self._attacked_history: List[np.ndarray] = []
        self._snapshots: List[DynamicsSnapshot] = []
        self._prev_clean: Optional[np.ndarray] = None
        self._prev_attacked: Optional[np.ndarray] = None

    def record_round(
        self,
        round_num: int,
        clean_weights: np.ndarray,
        clean_loss: float,
        attacked_weights: np.ndarray,
        attacked_loss: float,
        clean_updates: Optional[Dict[int, np.ndarray]] = None,
        attacked_updates: Optional[Dict[int, np.ndarray]] = None,
        nodes: Optional[list] = None,
    ) -> DynamicsSnapshot:
        """Record a single round and return the computed snapshot."""
        # --- Weight divergence from clean ---
        diff = attacked_weights - clean_weights
        weight_l2 = float(np.linalg.norm(diff))
        norm_product = np.linalg.norm(clean_weights) * np.linalg.norm(attacked_weights)
        weight_cosine = (
            float(np.dot(clean_weights, attacked_weights) / norm_product)
            if norm_product > 1e-10
            else 1.0
        )

        # --- Gradient alignment (direction of change) ---
        gradient_alignment = 1.0
        if self._prev_clean is not None and self._prev_attacked is not None:
            delta_clean = clean_weights - self._prev_clean
            delta_attacked = attacked_weights - self._prev_attacked
            norm_c = np.linalg.norm(delta_clean)
            norm_a = np.linalg.norm(delta_attacked)
            if norm_c > 1e-10 and norm_a > 1e-10:
                gradient_alignment = float(
                    np.dot(delta_clean, delta_attacked) / (norm_c * norm_a)
                )

        # --- Per-node update variance ---
        honest_var = 0.0
        malicious_var = 0.0
        if attacked_updates is not None and nodes is not None:
            honest_norms = []
            malicious_norms = []
            for i, w in attacked_updates.items():
                norm_val = float(np.linalg.norm(w))
                if nodes[i].is_malicious:
                    malicious_norms.append(norm_val)
                else:
                    honest_norms.append(norm_val)
            if len(honest_norms) > 1:
                honest_var = float(np.var(honest_norms))
            if len(malicious_norms) > 1:
                malicious_var = float(np.var(malicious_norms))

        snapshot = DynamicsSnapshot(
            round_num=round_num,
            weight_l2_from_clean=weight_l2,
            weight_cosine_from_clean=weight_cosine,
            gradient_alignment=gradient_alignment,
            honest_update_variance=honest_var,
            malicious_update_variance=malicious_var,
            loss_delta=attacked_loss - clean_loss,
        )
        self._snapshots.append(snapshot)
        self._prev_clean = clean_weights.copy()
        self._prev_attacked = attacked_weights.copy()

        return snapshot

    def get_report(
        self,
        system_name: str,
        attack_type: str,
        attack_pct: float,
    ) -> DynamicsReport:
        """Return the full dynamics report for this experiment."""
        report = DynamicsReport(
            system_name=system_name,
            attack_type=attack_type,
            attack_pct=attack_pct,
            snapshots=list(self._snapshots),
        )
        # Reset for next experiment
        self._snapshots = []
        self._prev_clean = None
        self._prev_attacked = None
        return report

    def reset(self) -> None:
        """Clear all state for a fresh experiment."""
        self._clean_history = []
        self._attacked_history = []
        self._snapshots = []
        self._prev_clean = None
        self._prev_attacked = None
