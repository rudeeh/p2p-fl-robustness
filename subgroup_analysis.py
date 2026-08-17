"""
subgroup_analysis.py
---------------------
SubgroupHarmAnalyzer: detects whether adversarial attacks create
disproportionate harm to specific data subgroups — even when aggregate
accuracy appears stable.

Why this matters for AI alignment
----------------------------------
A core alignment concern is hidden failure modes: a model that appears to
perform well on aggregate metrics but fails systematically in ways we don't
monitor.  Subgroup-targeted harm is a concrete instance of this pattern.

In a healthcare FL system, a sybil attacker could degrade the model's
performance for patients with rare conditions while maintaining overall
diagnostic accuracy.  No accuracy metric would catch this; only per-subgroup
analysis would.

This connects to specification gaming: if the training objective optimises
aggregate loss, an adversary can exploit the gap between "minimise average
loss" and "minimise worst-case subgroup loss."  The model does exactly what
the objective says — but the objective is mis-specified relative to actual
safety requirements.

Subgroup labels
---------------
For the synthetic binary classification dataset, subgroups are defined by
a latent cluster variable: samples whose first feature is above vs. below
the median.  This is a natural, non-arbitrary split because the logistic
decision boundary depends on feature values.
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
class SubgroupMetrics:
    """Metrics for a single subgroup."""
    subgroup_label    : str
    n_samples         : int
    accuracy          : float
    recall            : float      # true positive rate
    precision         : float
    f1                : float


@dataclass
class SubgroupHarmReport:
    """
    Full subgroup harm analysis comparing a (potentially attacked) model
    against a clean baseline.
    """
    system_name    : str
    attack_type    : str
    attack_pct     : float
    current        : Dict[str, SubgroupMetrics] = field(default_factory=dict)
    baseline       : Optional[Dict[str, SubgroupMetrics]] = None

    # --- Derived alignment-critical metrics ---
    def accuracy_parity_gap(self) -> float:
        """max - min accuracy across subgroups.  Higher = more harm concentration."""
        if not self.current:
            return 0.0
        accs = [m.accuracy for m in self.current.values()]
        return float(max(accs) - min(accs))

    def worst_subgroup_accuracy(self) -> float:
        """Accuracy of the worst-performing subgroup."""
        if not self.current:
            return 0.0
        return min(m.accuracy for m in self.current.values())

    def attack_induced_harm(self) -> Dict[str, float]:
        """
        Per-subgroup accuracy degradation caused by the attack.
        Requires a baseline report.
        """
        if not self.baseline:
            return {}
        harm = {}
        for label, cur in self.current.items():
            if label in self.baseline:
                harm[label] = self.baseline[label].accuracy - cur.accuracy
            else:
                harm[label] = 0.0
        return harm

    def harm_concentration_index(self) -> float:
        """
        Gini-like measure: is failure concentrated in specific subgroups?
        0 = harm evenly distributed, 1 = harm entirely in one subgroup.
        """
        harms = list(self.attack_induced_harm().values())
        total = sum(harms)
        if total < 1e-10:
            return 0.0
        harms_sorted = sorted(harms)
        n = len(harms_sorted)
        index = np.arange(1, n + 1)
        return float(
            (2 * np.sum(index * harms_sorted) / (n * total)) - (n + 1) / n
        )

    def equal_opportunity_gap(self) -> float:
        """max TPR disparity across subgroups."""
        if not self.current:
            return 0.0
        tprs = [m.recall for m in self.current.values()]
        return float(max(tprs) - min(tprs))

    def to_dict(self) -> dict:
        result = {
            "system_name": self.system_name,
            "attack_type": self.attack_type,
            "attack_pct": self.attack_pct,
            "accuracy_parity_gap": self.accuracy_parity_gap(),
            "worst_subgroup_accuracy": self.worst_subgroup_accuracy(),
            "harm_concentration_index": self.harm_concentration_index(),
            "equal_opportunity_gap": self.equal_opportunity_gap(),
            "attack_induced_harm": self.attack_induced_harm(),
            "subgroups": {
                label: {
                    "n_samples": m.n_samples,
                    "accuracy": m.accuracy,
                    "recall": m.recall,
                    "precision": m.precision,
                    "f1": m.f1,
                }
                for label, m in self.current.items()
            },
        }
        if self.baseline:
            result["baseline_subgroups"] = {
                label: {
                    "accuracy": m.accuracy,
                    "recall": m.recall,
                }
                for label, m in self.baseline.items()
            }
        return result


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class SubgroupHarmAnalyzer:
    """
    Analyzes per-subgroup model performance to detect attack-induced
    disparate impact.

    Parameters
    ----------
    subgroup_labels : array of shape (n_samples,)
        subgroup_labels[i] is the subgroup identifier for validation sample i.
    """

    def __init__(self, subgroup_labels: np.ndarray):
        self.subgroup_labels = np.asarray(subgroup_labels)
        self.unique_subgroups = np.unique(self.subgroup_labels)
        logger.info(
            "SubgroupHarmAnalyzer initialised with %d subgroups: %s",
            len(self.unique_subgroups),
            list(self.unique_subgroups),
        )

    def _compute_subgroup_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, SubgroupMetrics]:
        """Compute per-subgroup metrics for samples where mask is True."""
        results = {}
        for sg in self.unique_subgroups:
            sg_mask = mask & (self.subgroup_labels == sg)
            n = int(sg_mask.sum())
            if n == 0:
                continue

            yt = y_true[sg_mask]
            yp = y_pred[sg_mask]
            yp_prob = y_prob[sg_mask]

            acc = float(np.mean(yp == yt))

            # Recall (TPR): of the actual positives, how many did we catch?
            tp = float(np.sum((yp == 1) & (yt == 1)))
            fn = float(np.sum((yp == 0) & (yt == 1)))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # Precision: of predicted positives, how many were correct?
            fp = float(np.sum((yp == 1) & (yt == 0)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            # F1
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            results[str(sg)] = SubgroupMetrics(
                subgroup_label=str(sg),
                n_samples=n,
                accuracy=acc,
                recall=recall,
                precision=precision,
                f1=f1,
            )
        return results

    def analyze(
        self,
        model,
        X_val: np.ndarray,
        y_val: np.ndarray,
        system_name: str = "",
        attack_type: str = "",
        attack_pct: float = 0.0,
        baseline_model=None,
    ) -> SubgroupHarmReport:
        """
        Run subgroup analysis on a trained model, optionally against a baseline.

        Parameters
        ----------
        model         : object with predict() and predict_proba() methods
        X_val, y_val  : shared validation set
        baseline_model: optional clean model for delta computation
        """
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]  # P(class=1)

        current = self._compute_subgroup_metrics(y_val, y_pred, y_prob, np.ones(len(y_val), dtype=bool))

        baseline_metrics = None
        if baseline_model is not None:
            b_pred = baseline_model.predict(X_val)
            b_prob = baseline_model.predict_proba(X_val)[:, 1]
            baseline_metrics = self._compute_subgroup_metrics(y_val, b_pred, b_prob, np.ones(len(y_val), dtype=bool))

        report = SubgroupHarmReport(
            system_name=system_name,
            attack_type=attack_type,
            attack_pct=attack_pct,
            current=current,
            baseline=baseline_metrics,
        )

        if baseline_metrics:
            logger.info(
                "[%s | %s | %.0f%%] worst_subgroup_acc=%.3f  "
                "parity_gap=%.3f  harm_concentration=%.3f",
                system_name,
                attack_type,
                attack_pct * 100,
                report.worst_subgroup_accuracy(),
                report.accuracy_parity_gap(),
                report.harm_concentration_index(),
            )

        return report


# ---------------------------------------------------------------------------
# Subgroup label generation
# ---------------------------------------------------------------------------

def make_subgroup_labels(X: np.ndarray, n_subgroups: int = 2) -> np.ndarray:
    """
    Assign subgroup labels based on feature-space clustering.

    Uses the first feature's median as a split point for binary subgroups,
    or quantile boundaries for more subgroups.  This is non-arbitrary because
    the logistic decision boundary depends on feature values, so different
    regions of feature space may experience different robustness.

    Parameters
    ----------
    X            : (n_samples, n_features)
    n_subgroups  : number of subgroups (default 2)

    Returns
    -------
    labels : (n_samples,) int array
    """
    if n_subgroups <= 1:
        return np.zeros(len(X), dtype=int)

    # Use first feature quantiles as split points
    f0 = X[:, 0]
    boundaries = np.quantile(f0, np.linspace(0, 1, n_subgroups + 1))
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf

    labels = np.zeros(len(X), dtype=int)
    for i in range(n_subgroups):
        mask = (f0 >= boundaries[i]) & (f0 < boundaries[i + 1])
        labels[mask] = i
    # Handle rightmost edge
    labels[f0 >= boundaries[-2]] = n_subgroups - 1

    return labels
