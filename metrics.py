"""
metrics.py
----------
Lightweight metrics bookkeeping for the simulation.

Classes
-------
RoundRecord      – immutable snapshot of one round's performance.
ExperimentResult – collection of RoundRecords for one experiment run,
                   plus derived statistics (convergence rate, robustness score).

The robustness_score is the headline metric:

    robustness_score = mean_accuracy_with_attack
                       ─────────────────────────
                       mean_accuracy_without_attack

A score of 1.0 → attack had no effect.
A score of 0.0 → system completely collapsed under attack.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoundRecord:
    """Immutable record for a single round."""
    round_num : int
    accuracy  : float
    loss      : float


@dataclass
class ExperimentResult:
    """
    Accumulates per-round metrics for one (system, attack_type, attack_pct, trial).

    Parameters
    ----------
    system_name : 'P2P-FC' | 'P2P-Ring' | 'FL'
    attack_type : 'none' | 'poisoning' | 'sybil' | 'noise'
    attack_pct  : fraction of malicious nodes (0.0 – 1.0)
    topology    : topology string, stored for labelling only
    trial       : trial index (0-based)
    """
    system_name : str
    attack_type : str
    attack_pct  : float
    topology    : str = ""
    trial       : int = 0
    rounds      : List[RoundRecord] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def add_round(self, round_num: int, accuracy: float, loss: float) -> None:
        self.rounds.append(RoundRecord(round_num, accuracy, loss))

    # ------------------------------------------------------------------
    # Derived statistics
    # ------------------------------------------------------------------

    def accuracies(self) -> List[float]:
        return [r.accuracy for r in self.rounds]

    def losses(self) -> List[float]:
        return [r.loss for r in self.rounds]

    def final_accuracy(self, n_tail: int = 5) -> float:
        """Mean accuracy over the last `n_tail` rounds."""
        accs = self.accuracies()
        return float(np.mean(accs[-n_tail:])) if accs else 0.0

    def convergence_round(self, threshold: float = 0.70) -> int:
        """
        First round at which accuracy reaches `threshold` × peak accuracy.
        Returns total rounds if the threshold is never reached.
        """
        accs = self.accuracies()
        if not accs:
            return 0
        target = threshold * max(accs)
        for i, a in enumerate(accs):
            if a >= target:
                return i + 1
        return len(accs)

    def robustness_score(self, baseline: "ExperimentResult") -> float:
        """
        Compute robustness relative to a no-attack baseline.

        Returns a value in [0, 1].  Values > 1 are clipped to 1.
        """
        base_acc = baseline.final_accuracy()
        if base_acc == 0.0:
            return 0.0
        return min(self.final_accuracy() / base_acc, 1.0)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "system_name": self.system_name,
            "attack_type": self.attack_type,
            "attack_pct" : self.attack_pct,
            "topology"   : self.topology,
            "trial"      : self.trial,
            "rounds"     : [
                {"round": r.round_num, "accuracy": r.accuracy, "loss": r.loss}
                for r in self.rounds
            ],
        }

    def log_summary(self) -> None:
        logger.info(
            "[%s | %s | %.0f%% attack | trial %d] "
            "final_acc=%.4f  convergence_round=%d",
            self.system_name,
            self.attack_type,
            self.attack_pct * 100,
            self.trial,
            self.final_accuracy(),
            self.convergence_round(),
        )


# ---------------------------------------------------------------------------
# Utility: save a list of results to disk
# ---------------------------------------------------------------------------

def save_results_json(results: List[ExperimentResult], path: str | Path) -> None:
    """Persist all results as JSON for offline analysis."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    logger.info("Results saved → %s", path)


def save_results_csv(results: List[ExperimentResult], path: str | Path) -> None:
    """Persist per-round metrics as a flat CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["system", "attack_type", "attack_pct", "topology", "trial",
             "round", "accuracy", "loss"]
        )
        for r in results:
            for rec in r.rounds:
                writer.writerow([
                    r.system_name, r.attack_type, f"{r.attack_pct:.2f}",
                    r.topology, r.trial,
                    rec.round_num, f"{rec.accuracy:.6f}", f"{rec.loss:.6f}",
                ])
    logger.info("CSV saved → %s", path)


def compute_robustness_table(
    results   : List[ExperimentResult],
    baseline_attack: str = "none",
) -> dict:
    """
    For every (system, attack_type, attack_pct) group, compute mean robustness
    score across trials relative to the no-attack baseline.

    Returns a nested dict:
        {system_name: {attack_type: {attack_pct: mean_robustness_score}}}
    """
    # Build baseline lookup: system → mean final accuracy with no attack
    baselines: dict[str, float] = {}
    for r in results:
        if r.attack_type == baseline_attack:
            baselines.setdefault(r.system_name, []).append(r.final_accuracy())
    baselines = {k: float(np.mean(v)) for k, v in baselines.items()}

    table: dict = {}
    for r in results:
        if r.attack_type == baseline_attack:
            continue
        base_acc = baselines.get(r.system_name, 1.0)
        score = min(r.final_accuracy() / base_acc, 1.0) if base_acc > 0 else 0.0
        (
            table
            .setdefault(r.system_name, {})
            .setdefault(r.attack_type, {})
            .setdefault(r.attack_pct, [])
            .append(score)
        )

    # Average across trials
    for sys in table:
        for atk in table[sys]:
            for pct in table[sys][atk]:
                table[sys][atk][pct] = float(np.mean(table[sys][atk][pct]))

    return table
